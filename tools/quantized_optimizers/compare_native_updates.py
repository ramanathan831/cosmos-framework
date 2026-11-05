"""Development-only CUDA comparison of cumulative native parameter updates.

The FP32-state bitsandbytes control uses the same supported backend with every
tensor below its minimum quantization size. It must allocate zero uint8 moments.
This is an arithmetic diagnostic, not a new public optimizer or a speed trial.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from safetensors import safe_open

from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    if not torch.cuda.is_available():
        raise RuntimeError("native update comparison requires real CUDA reductions")
    torch.set_num_threads(8)
    arms = ("fused_fp32", "bnb_fp32", "bnb_8bit")
    manager = CheckpointManager()
    states, provenance, reports, checkpoints = {}, {}, {}, {}
    for arm in arms:
        root = args.root / arm
        matches = list((root / "native").rglob("model.pt"))
        if len(matches) != 1:
            raise ValueError("diagnostic requires one actual committed checkpoint per arm")
        checkpoints[arm] = matches[0].parent
        provenance[arm] = manager.checkpoint_provenance(checkpoints[arm])
        states[arm] = torch.load(matches[0], map_location="cpu", weights_only=True, mmap=True)["net"]
        reports[arm] = json.loads((root / "diagnostics/complete.json").read_text())["state_report"]
    if reports["bnb_fp32"]["quantized_elements"] or reports["bnb_fp32"]["uninitialized_elements"]:
        raise ValueError("FP32-state diagnostic unexpectedly quantized or left moments uninitialized")
    if not reports["bnb_8bit"]["quantized_elements"]:
        raise ValueError("eight-bit diagnostic allocated no quantized states")
    names = [name for group in reports["bnb_8bit"]["backend"]["groups"] for name in group["names"]]
    if names != [name for group in reports["bnb_fp32"]["backend"]["groups"] for name in group["names"]]:
        raise ValueError("diagnostic changed the exact trainable membership or ordering")
    model = Path(os.environ["SFT_MODEL_PATH"])
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    pairs = (("fused_fp32", "bnb_fp32"), ("bnb_fp32", "bnb_8bit"), ("fused_fp32", "bnb_8bit"))
    rows = []
    for name in names:
        if name not in index or any(name not in state for state in states.values()):
            raise ValueError(f"exact pretrained/committed parameter name missing: {name}")
        with safe_open(model / index[name], framework="pt", device="cpu") as source:
            initial = source.get_tensor(name).to(device="cuda").float()
        updates = {}
        for arm, state in states.items():
            value = state[name]
            if value.dtype != torch.float32 or value.shape != initial.shape:
                raise ValueError(f"parameter precision/shape changed: {arm} {name}")
            updates[arm] = value.to(device="cuda") - initial
            if not torch.isfinite(updates[arm]).all():
                raise ValueError("nonfinite native parameter update")
        for reference, candidate in pairs:
            a, b = updates[reference], updates[candidate]
            delta = b - a
            a2 = torch.sum(a.square(), dtype=torch.float64).item()
            b2 = torch.sum(b.square(), dtype=torch.float64).item()
            d2 = torch.sum(delta.square(), dtype=torch.float64).item()
            dot = torch.sum(a * b, dtype=torch.float64).item()
            rows.append(
                dict(
                    name=name,
                    reference=reference,
                    candidate=candidate,
                    elements=a.numel(),
                    reference_squared_norm=a2,
                    candidate_squared_norm=b2,
                    difference_squared_norm=d2,
                    dot=dot,
                    relative_update_l2=math.sqrt(d2 / a2) if a2 else None,
                    cosine=dot / math.sqrt(a2 * b2) if a2 and b2 else None,
                    max_abs_difference=delta.abs().max().item(),
                )
            )
        del initial, updates, a, b, delta
    torch.cuda.synchronize()
    for arm, checkpoint in checkpoints.items():
        if manager.checkpoint_provenance(checkpoint) != provenance[arm]:
            raise ValueError("checkpoint identity changed during update inspection")
    args.output.write_text(
        json.dumps(
            dict(
                complete=True,
                checkpoint_provenance=provenance,
                state_reports=reports,
                cumulative_updates=15,
                trainable_parameters=len(names),
                trainable_elements=sum(states["bnb_8bit"][name].numel() for name in names),
                rows=rows,
                elapsed_seconds=time.monotonic() - started,
                scope="One development seed; cumulative updates include gradient feedback and are not one-step arithmetic error or task-quality inference.",
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

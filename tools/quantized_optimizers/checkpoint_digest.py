"""Value identities and actual state bytes from committed dense checkpoints (CPU)."""

import argparse
import json
from pathlib import Path

import torch

from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
from tools.quantized_optimizers.state_digest import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    manager = CheckpointManager()
    report = {"checkpoint_provenance": manager.checkpoint_provenance(args.checkpoint)}
    for name in ("model", "optim", "trainer", "data"):
        value = manager.load_component(args.checkpoint, name)
        if value is None:
            raise ValueError(f"missing checkpoint component {name}")
        # Objective/corpus and validation histories belong to exact-resume
        # identity, but timestamps/publication roots differ across restart runs.
        if name == "trainer":
            report["rng"] = digest(value["rng"])
            report["scheduler"] = digest(value["lr_scheduler"])
            report["optimizer_step"] = value["optimizer_step"]
        else:
            report[name] = digest(value)
        if name == "optim":
            counts = {}
            for state in value["state"].values():
                for tensor in state.values():
                    if isinstance(tensor, torch.Tensor):
                        key = str(tensor.dtype)
                        counts[key] = counts.get(key, 0) + tensor.numel() * tensor.element_size()
            report["state_bytes_by_dtype"] = counts
            report["state_bytes"] = sum(counts.values())
        del value
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

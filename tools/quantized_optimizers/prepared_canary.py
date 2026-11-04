"""Native entrypoint with post-run value digests; excluded from timing claims."""

import json
import os
from pathlib import Path

import torch

import cosmos_framework.posttrain.entrypoints.sft as entry
from cosmos_framework.posttrain.checkpoint.rng import collect_rng_state
from tools.quantized_optimizers.state_digest import digest, tensor_digest

original = entry.build_trainer
captured = {}
original_step = entry.train_step


def capture_completed_step(data, policy, rl):
    metrics = original_step(data, policy, rl)
    step = int(metrics["optim/optimizer_step"])
    if 41 <= step <= 44:
        trainer = captured["trainer"]
        # Correctness-only instrumentation after scheduler/update/cursor advancement.
        # Read live state without making an eager, full CPU optimizer snapshot.
        record = {
            "step": step,
            "trainables": digest({n: p for n, p in trainer._net().named_parameters() if p.requires_grad}),
            "optimizer": digest(torch.optim.Optimizer.state_dict(trainer.opt)),
            "scheduler": digest(trainer.lr_scheduler.state_dict()),
            "update": digest(trainer._update_state_dict()),
            "cursor": digest(data.state_dict()),
            "rng": digest(collect_rng_state()),
        }
        captured.setdefault("completed_update_states", []).append(record)
    return metrics


def capture(config, rl, data, validation, checkpoint):
    trainer = original(config, rl, data, validation, checkpoint)
    captured.update(trainer=trainer, data=data)
    if os.environ.get("RECOVERY_ARM"):
        name, parameter = next((name, p) for name, p in trainer._net().named_parameters() if p.requires_grad)
        captured["update_probes"] = []

        def probe(optimizer, _args, _kwargs):
            record = {"step": int(optimizer.state[parameter]["step"]), "name": name, "sha256": tensor_digest(parameter)}
            previous = captured["update_probes"]
            # Torch can wrap both the upstream step and Cosmos's overriding step.
            # Require their views of the same clock to agree, recording it once.
            if previous and previous[-1]["step"] == record["step"]:
                if previous[-1] != record:
                    raise RuntimeError("optimizer wrapper changed a parameter after its inner update")
            else:
                previous.append(record)

        captured["probe_handle"] = trainer.opt.register_step_post_hook(probe)
    return trainer


if __name__ == "__main__":
    entry.build_trainer = capture
    if os.environ.get("RECOVERY_ARM"):
        entry.train_step = capture_completed_step
    entry.main()
    trainer = captured["trainer"]
    # State after the same native loop/finalization. Fingerprint each component
    # by values, excluding deliberate objective identity differences (cache use).
    torch.cuda.synchronize()
    cursor = captured["data"].state_dict()
    result = {
        "model": digest(trainer._net().state_dict()),
        "optimizer": digest(trainer.opt.state_dict()),
        "scheduler": digest(trainer.lr_scheduler.state_dict()),
        "update": digest(trainer._update_state_dict()),
        "cursor": digest(cursor),
        "cursor_state": cursor,
        "rng": digest(collect_rng_state()),
        "update_probes": captured.get("update_probes"),
        "completed_update_states": captured.get("completed_update_states"),
        "state_report": trainer.opt.state_report()
        if hasattr(trainer.opt, "state_report")
        else {
            "state_bytes": sum(
                v.numel() * v.element_size()
                for s in trainer.opt.state.values()
                for v in s.values()
                if isinstance(v, torch.Tensor)
            )
        },
        "peak_allocated": torch.cuda.max_memory_allocated(),
        "peak_reserved": torch.cuda.max_memory_reserved(),
    }
    Path(os.environ["STATE_DIGEST_PATH"]).write_text(json.dumps(result, indent=2) + "\n")

"""Development-only task curves and phase costs around the native SFT entrypoint.

Training mathematics and checkpoint transactions remain native. Extra task forwards
are declared development diagnostics; their wall cost is retained, not used as an
uninstrumented performance claim. No final selection is loaded by this module.
"""

import json
import os
import time
from pathlib import Path

import cosmos_framework.posttrain.algorithm.sft as algorithm
import cosmos_framework.posttrain.entrypoints.sft as entry
from tools.quantized_optimizers.score_dense import aggregate, evaluate_row

root = Path(os.environ["TASK_DIAGNOSTIC_OUTPUT"])
root.mkdir(parents=True, exist_ok=False)
started = time.monotonic()
captured = {}
original_objective = algorithm.make_objective
original_build = entry.build_trainer
original_evaluate = entry._evaluate
original_step = entry.train_step


def record(name, value):
    with (root / name).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def timed(owner, name, phase):
    original = getattr(owner, name)

    def measured(*args, **kwargs):
        before = time.monotonic()
        try:
            return original(*args, **kwargs)
        finally:
            record(
                "phases.jsonl",
                {"phase": phase, "seconds": time.monotonic() - before, "elapsed_seconds": time.monotonic() - started},
            )

    setattr(owner, name, measured)


def objective(packer, **kwargs):
    from cosmos_framework.posttrain.data.datasets.sft import JSONLSFTDataset
    from cosmos_framework.posttrain.data.packer.prepared_sft import PreparedSFTCorpus

    captured["task_dataset"] = JSONLSFTDataset(os.environ["TASK_DEVELOPMENT_DATA"])
    # The native builder already bound and verified train/validation sources.
    # Bind only the added task source, retaining all verified media stamps.
    previous = packer.sft_process_sample.__self__
    additional = PreparedSFTCorpus(packer, os.environ["SFT_PREPARED_CORPUS"], [captured["task_dataset"]])
    if previous.identity != additional.identity:
        raise ValueError("development sources must share the same immutable corpus")
    additional.media_stamps.update(previous.media_stamps)
    packer.sft_process_sample = additional.process
    captured["packer"] = packer
    return original_objective(packer, **kwargs)


def build(*args, **kwargs):
    before = time.monotonic()
    trainer = original_build(*args, **kwargs)
    captured["trainer"] = trainer
    record(
        "phases.jsonl",
        {"phase": "build_trainer", "seconds": time.monotonic() - before, "elapsed_seconds": time.monotonic() - started},
    )
    original_save = trainer.save

    def save(*save_args, **save_kwargs):
        before_save = time.monotonic()
        result = original_save(*save_args, **save_kwargs)
        record(
            "phases.jsonl",
            {
                "phase": "checkpoint",
                "seconds": time.monotonic() - before_save,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        return result

    trainer.save = save
    timed(trainer.opt, "state_dict", "optimizer_snapshot")
    return trainer


def evaluate(units, rl, model_step, reasons):
    report = original_evaluate(units, rl, model_step, reasons)
    before = time.monotonic()
    trainer, packer = captured["trainer"], captured["packer"]
    dataset = captured["task_dataset"]
    rows = [dataset[index] for index in range(len(dataset))]
    scored = []
    for row in rows:
        item = {"id": row["id"], "step": model_step, **evaluate_row(trainer.model, packer, trainer.objective, row)}
        item["strict_first_token_text"] = packer.tokenizer.decode(
            [item["strict_first_token_id"]], skip_special_tokens=False
        )
        record("task_rows.jsonl", item)
        scored.append(item)
    for item, row in zip(scored, rows, strict=True):
        item["media_sha256"] = dataset.media_hashes[row["messages"][-2]["content"][0]["video"]]
    record(
        "task_curve.jsonl",
        {
            "step": model_step,
            "elapsed_seconds": time.monotonic() - started,
            "task_seconds": time.monotonic() - before,
            "substantive_validation": report.metrics,
            "metrics": aggregate(scored),
        },
    )
    from dataclasses import replace

    task = aggregate(scored)
    duration = report.duration_s + time.monotonic() - before
    return replace(
        report,
        duration_s=duration,
        metrics={
            **report.metrics,
            "timer/validation_s": duration,
            "val/task_answer_nll": task["answer_nll"],
            "val/task_accuracy": task["restricted_choice_accuracy"],
        },
    )


def step(*args, **kwargs):
    before = time.monotonic()
    result = original_step(*args, **kwargs)
    record(
        "phases.jsonl",
        {
            "phase": "train_step",
            "step": result["optim/optimizer_step"],
            "seconds": time.monotonic() - before,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    return result


def install():
    import cosmos_framework.posttrain.config.build as builders
    import cosmos_framework.posttrain.data.datasets.sft as datasets
    import cosmos_framework.posttrain.data.packer.prepared_sft as prepared
    import cosmos_framework.posttrain.model.hf as hf

    timed(builders, "build_backbone", "build_backbone")
    timed(hf, "load_weights", "load_weights")
    timed(prepared, "attach_prepared_corpus", "prepared_verification")
    timed(datasets.JSONLSFTDataset, "__init__", "dataset_identity")
    algorithm.make_objective = objective
    entry.build_trainer = build
    entry._evaluate = evaluate
    entry.train_step = step


def finish():
    import torch

    trainer = captured["trainer"]
    optimizer = trainer.opt
    tensors = [v for state in optimizer.state.values() for v in state.values() if isinstance(v, torch.Tensor)]
    state = (
        optimizer.state_report()
        if hasattr(optimizer, "state_report")
        else {
            "state_bytes": sum(v.numel() * v.element_size() for v in tensors),
            "backend": type(optimizer).__qualname__,
        }
    )
    (root / "complete.json").write_text(
        json.dumps(
            {
                "complete": True,
                "elapsed_seconds": time.monotonic() - started,
                "state_report": state,
                "trainable_elements": sum(p.numel() for p in trainer._net().parameters() if p.requires_grad),
                "peak_allocated": torch.cuda.max_memory_allocated(),
                "peak_reserved": torch.cuda.max_memory_reserved(),
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


if __name__ == "__main__":
    install()
    entry.main()
    finish()

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


def objective(packer, **kwargs):
    from cosmos_framework.posttrain.data.datasets.sft import JSONLSFTDataset
    from cosmos_framework.posttrain.data.packer.prepared_sft import attach_prepared_corpus

    data = Path(os.environ["SFT_DATA_ROOT"])
    captured["task_dataset"] = JSONLSFTDataset(os.environ["TASK_DEVELOPMENT_DATA"])
    attach_prepared_corpus(
        packer,
        os.environ["SFT_PREPARED_CORPUS"],
        [JSONLSFTDataset(str(data / name)) for name in ("train.jsonl", "train-mixed.jsonl", "validation.jsonl")]
        + [captured["task_dataset"]],
    )
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
    return report


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


if __name__ == "__main__":
    algorithm.make_objective = objective
    entry.build_trainer = build
    entry._evaluate = evaluate
    entry.train_step = step
    entry.main()
    (root / "complete.json").write_text(
        json.dumps({"complete": True, "elapsed_seconds": time.monotonic() - started}) + "\n"
    )

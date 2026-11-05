"""Reproduce this campaign's fixed development screen from its immutable raw receipts."""

import collections
import gzip
import hashlib
import json
import pathlib
import statistics

p = pathlib.Path(__file__).resolve().parent
raw = json.load(gzip.open(p / "kv_precision_raw.json.gz", "rt"))
original = json.load(gzip.open(p / "outcomes_raw.json.gz", "rt"))
plan = json.loads((p / "kv_precision_probe.json").read_text())
allocation = next(
    row for row in json.loads((p / "allocation_ledger.json").read_text())["jobs"] if row["job"] == "37988222"
)
assert allocation["state"] == "COMPLETED" and allocation["elapsed_seconds"] == 733
receipt = json.loads(raw["_collection.json"])
for name, meta in receipt["files"].items():
    value = raw[name].encode()
    assert len(value) == meta["bytes"] and hashlib.sha256(value).hexdigest() == meta["sha256"], name
assert json.loads(raw["executed_spec.json"]) == plan
assert json.loads(raw["complete.json"])["cases"] == [r["name"] for r in plan["cases"]]
runtime = json.loads(raw["runtime-37988222.json"])
assert runtime["source_archive_sha256"] == plan["source_archive_sha256"]
assert runtime["versions"] == {"torch": "2.11.0+cu128", "transformers": "5.8.1", "bitsandbytes": "0.49.2"}
assert runtime["prepared_corpus_sha256"] == "ea2e64893ab3d7da0f4c8be33001ae99c96dfd2c54fe293449c29255cd1f2d70"


def get(prefix, suffix, source=raw):
    keys = [k for k in source if k.startswith(prefix) and k.endswith(suffix)]
    assert len(keys) == 1, keys
    return source[keys[0]]


def rows(value):
    return [json.loads(v) for v in value.splitlines()]


def memory(prefix, source=raw):
    values = rows(source[prefix + "/memory.jsonl"])
    names = ["device_used", "matched_process_gpu_bytes", "process_tree_pss", "process_tree_rss"]
    return {n: max(r[n] for r in values if r.get(n) is not None) for n in names}


result = []
for seed in (17, 29, 43):
    name = f"kv_fp32_seed{seed}"
    old = f"final-seed{seed}/matched_adamw_bnb_8bit"
    attempt = json.loads(raw[name + "/attempt.json"])
    assert attempt["returncode"] == 0
    curves = rows(raw[name + "/diagnostics/task_curve.jsonl"])
    assert [c["step"] for c in curves] == [0, 5, 10, 15]
    complete = json.loads(raw[name + "/diagnostics/complete.json"])
    assert complete["complete"]
    state = complete["state_report"]
    expect = plan["expected_state"]
    for field in ("trainable_elements", "quantized_elements", "fp32_elements"):
        assert state[field] == expect[field]
    assert state["uninitialized_elements"] == 0 and state["paged_moment_bytes"] == 0
    state_bytes = state["moment_bytes"] + state["metadata_bytes"]
    assert state_bytes == expect["total_tensor_state_bytes"]
    assert sorted(n for n, v in state["exceptions"].items() if v == "explicit") == sorted(plan["fp32_state_names"])
    counts = dict(collections.Counter(state["exceptions"].values()))
    assert counts == {"explicit": 32, "small_tensor": 64}
    assert state["backend"]["fp32_state_names"] == sorted(plan["fp32_state_names"])
    metrics = rows(get(name + "/native/", "metrics.jsonl"))
    train = [r for r in metrics if r["step"] > 0]
    assert [r["step"] for r in train] == list(range(1, 16))
    assert all(
        r["optim/update_outcome"] == "applied"
        and r["optim/optimizer_step"] == r["step"]
        and r["optim/nonfinite_skips_total"] == 0
        for r in train
    )
    old_train = [r for r in rows(get(old + "/native/", "metrics.jsonl", original)) if r["step"] > 0]
    for field in ("train/valid_token_count", "train/valid_sequence_count", "train/learning_rate"):
        assert [r[field] for r in train] == [r[field] for r in old_train], field
    task = rows(raw[name + "/diagnostics/task_rows.jsonl"])
    oldtask = rows(original[old + "/diagnostics/task_rows.jsonl"])
    assert len(task) == len(oldtask) == 128
    for a, b in zip(task, oldtask):
        if a["step"] == 0:
            assert a == b
        for key in (
            "id",
            "step",
            "input_sha256",
            "gold",
            "candidate_token_ids",
            "inputs",
            "sequence_length",
            "native_supervised_tokens",
        ):
            assert a[key] == b[key], key
    model = json.loads(raw[name + "/model-file-reads.json"])
    old_model = json.loads(original[old + "/model-file-reads.json"])
    assert model["files"] == old_model["files"]
    control = plan["control_development"][str(seed)]
    last = curves[-1]
    assert rows(original[old + "/diagnostics/task_curve.jsonl"])[-1]["metrics"] == control["metrics"]
    phases = rows(raw[name + "/diagnostics/phases.jsonl"])
    result.append(
        {
            "seed": seed,
            "curve": curves,
            "control": control,
            "task_nll_ratio": last["metrics"]["answer_nll"] / control["metrics"]["answer_nll"],
            "accuracy_change": last["metrics"]["restricted_choice_accuracy"]
            - control["metrics"]["restricted_choice_accuracy"],
            "substantive_nll_ratio": last["substantive_validation"]["val/loss_avg"]
            / control["substantive_validation"]["val/loss_avg"],
            "state": {k: v for k, v in state.items() if k not in ("backend", "exceptions")},
            "total_tensor_state_bytes": state_bytes,
            "state_saving_fraction": 1 - state_bytes / expect["baseline_tensor_state_bytes"],
            "exception_counts": counts,
            "verified_explicit_names": state["backend"]["fp32_state_names"],
            "complete": complete["complete"],
            "actual_updates": 15,
            "skipped_updates": 0,
            "attempt_seconds": attempt["elapsed_seconds"],
            "native_training_seconds": sum(r["timer/e2e_step_s"] for r in train),
            "original_control_native_training_seconds": sum(r["timer/e2e_step_s"] for r in old_train),
            "peak_allocated": complete["peak_allocated"],
            "peak_reserved": complete["peak_reserved"],
            "external_memory_peaks": memory(name),
            "original_control_external_memory_peaks": memory(old, original),
            "phase_seconds": {
                phase: sum(r["seconds"] for r in phases if r["phase"] == phase)
                for phase in sorted({r["phase"] for r in phases})
            },
            "matched_control_checks": [
                "exact initialization predictions",
                "same pretrained file hashes and prepared corpus",
                "same input/gold/positions, data tokens and learning rate per update",
                "same 15 applied updates; no skips; exact 32 FP32 selectors",
            ],
        }
    )
ratio = statistics.mean(r["curve"][-1]["metrics"]["answer_nll"] for r in result) / statistics.mean(
    r["control"]["metrics"]["answer_nll"] for r in result
)
gates = {
    "mean_task_nll_improvement_at_least_2_percent": ratio <= 0.98,
    "no_seed_accuracy_loss": all(r["accuracy_change"] >= 0 for r in result),
    "each_substantive_nll_ratio_at_most_1_01": all(r["substantive_nll_ratio"] <= 1.01 for r in result),
    "state_saving_at_least_60_percent": all(r["state_saving_fraction"] >= 0.60 for r in result),
    "exact_selectors_and_no_skips": True,
}
x = {
    "complete": True,
    "job": "37988222",
    "charged_gpu_seconds": allocation["charged_or_reserved_gpu_seconds"],
    "raw_sha256": hashlib.sha256((p / "kv_precision_raw.json.gz").read_bytes()).hexdigest(),
    "protocol_sha256": hashlib.sha256((p / "kv_precision_probe.json").read_bytes()).hexdigest(),
    "source_commit": plan["source_commit"],
    "runtime": runtime,
    "scope": "Three-seed development-only FP32 K/V state screen; no final scoring. Timing/memory are descriptive, historical controls are not a matched repeated timing experiment.",
    "candidate_mean_task_nll_over_control": ratio,
    "gates": gates,
    "accepted": all(gates.values()),
    "rows": result,
}
x["decision"] = (
    "Development screen passed; requires separately reserved recovery and locked final comparisons, no default change."
    if x["accepted"]
    else "Rejected by the predeclared screen; retain default state precision and every prior final result. No final scoring or automatic follow-up."
)
(p / "kv_precision_summary.json").write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")
print(json.dumps({k: v for k, v in x.items() if k not in ("rows", "runtime")}, indent=2))
for r in result:
    print(
        r["seed"],
        r["curve"][-1]["metrics"]["answer_nll"],
        r["curve"][-1]["metrics"]["restricted_choice_accuracy"],
        r["substantive_nll_ratio"],
        r["native_training_seconds"],
        r["external_memory_peaks"],
    )

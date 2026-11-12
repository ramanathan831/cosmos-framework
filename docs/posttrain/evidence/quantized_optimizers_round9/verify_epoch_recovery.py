"""Whole-epoch cursor coverage and fresh-process continuation, no excluded fields."""

import argparse
import json
from pathlib import Path

from docs.posttrain.evidence.quantized_optimizers_round8.verify_interleave_recovery import compare_components, rows

p = argparse.ArgumentParser()
p.add_argument("root", type=Path)
p.add_argument("arm")
a = p.parse_args()
control, resumed = a.root / (a.arm + "_seed17"), a.root / (a.arm + "_resume17")
x, y = rows(control / "diagnostics/training_inputs.jsonl"), rows(resumed / "diagnostics/training_inputs.jsonl")
assert len(x) == 174 and len(y) == 87 and x[87:] == y
assert [len(v["ids"]) for v in x] == ([4] * 86 + [2]) * 2
for start in (0, 87):
    ids = [i for batch in x[start : start + 87] for i in batch["ids"]]
    assert len(ids) == len(set(ids)) == 346
states = rows(control / "diagnostics/recovery_states.jsonl")
assert [v["step"] for v in states] == [88, 89, 174]
assert states == rows(resumed / "diagnostics/recovery_states.jsonl")
paths = []
for run in (control, resumed):
    found = list((run / "native").rglob("iter_000174/model.pt"))
    assert len(found) == 1
    paths.append(found[0].parent)
components, provenance = compare_components(paths)
(a.root / (a.arm + "_recovery_verified.json")).write_text(
    json.dumps(
        dict(
            complete=True,
            exact_next_input_batches=87,
            total_rows=692,
            real_two_row_tails=[87, 174],
            exact_live_updates=states,
            exact_final_components=components,
            provenance=provenance,
            excluded=[],
        ),
        indent=2,
    )
    + "\n"
)

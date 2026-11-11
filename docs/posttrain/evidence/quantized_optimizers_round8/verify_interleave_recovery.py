"""Compare every resumed update and exact next inputs for the new order policy."""

import argparse
import json
from pathlib import Path

from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
from tools.quantized_optimizers.state_digest import digest


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("prefix")
    args = parser.parse_args()
    reference = args.root / (args.prefix + "_seed17")
    resumed = args.root / "resume17"
    a = rows(reference / "diagnostics/recovery_states.jsonl")
    b = rows(resumed / "diagnostics/recovery_states.jsonl")
    assert [v["step"] for v in a] == [11, 12, 13, 14, 15] and a == b
    inputs = rows(reference / "diagnostics/training_inputs.jsonl")
    next_inputs = rows(resumed / "diagnostics/training_inputs.jsonl")
    assert len(inputs) == 15 and len(next_inputs) == 5 and inputs[10:] == next_inputs
    paths = []
    for run in (reference, resumed):
        matches = list((run / "native").rglob("iter_000015/model.pt"))
        assert len(matches) == 1
        paths.append(matches[0].parent)
    manager = CheckpointManager()
    provenance = {str(p): manager.checkpoint_provenance(p) for p in paths}
    matches = {}
    for component in ("model", "optim", "data", "trainer"):
        hashes = []
        for path in paths:
            value = manager.load_component(path, component)
            assert value is not None
            if component == "trainer":
                value = {k: v for k, v in value.items() if k != "validation"}
            hashes.append(digest(value))
            del value
        assert hashes[0] == hashes[1], component
        matches[component] = hashes[0]
    assert all(manager.checkpoint_provenance(p) == v for p, v in provenance.items())
    (args.root / "recovery_verified.json").write_text(
        json.dumps(
            {
                "complete": True,
                "resumed_updates": [11, 12, 13, 14, 15],
                "exact_live_update_states": a,
                "exact_next_input_batches": 5,
                "exact_final_components": matches,
                "provenance": provenance,
                "excluded": "validation publication/timing bookkeeping; all training clocks, RNG, objective identity and scheduler retained",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

"""Required replay parity and fresh-process attribution; development data only."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager
from tools.quantized_optimizers.score_dense import read_checkpoint_file
from tools.quantized_optimizers.state_digest import digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("spec", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    manager = CheckpointManager()
    results = {}
    for arm, definition in spec["references"].items():
        case = args.root / arm
        old = Path(definition["checkpoint"])
        checkpoints = {p.parent.name: p.parent for p in (case / "native").rglob("model.pt")}
        assert set(checkpoints) == {"iter_000010", "iter_000015"}, checkpoints
        new = checkpoints["iter_000015"]
        provenance = {str(p): manager.checkpoint_provenance(p) for p in (old, new)}
        matches = {}
        for component in ("model", "optim", "data", "trainer"):
            values = []
            for checkpoint in (old, new):
                value = manager.load_component(checkpoint, component)
                if value is None:
                    raise ValueError(f"missing {component}")
                values.append(digest(value))
                del value
            matches[component] = values[0] == values[1]
            (case / "state_parity.json").write_text(
                json.dumps({"matches": matches, "digests": values, "last_component": component}, indent=2)
            )
            if not matches[component]:
                raise ValueError(f"replay state mismatch: {arm}/{component}")
        for path, expected in provenance.items():
            assert manager.checkpoint_provenance(path) == expected
        prior = Path(definition["diagnostics"])
        assert (case / "diagnostics/task_rows.jsonl").read_text() == (prior / "task_rows.jsonl").read_text()
        curves = [json.loads(x) for x in (case / "diagnostics/task_curve.jsonl").read_text().splitlines()]
        old_curves = [json.loads(x) for x in (prior / "task_curve.jsonl").read_text().splitlines()]
        assert [v["step"] for v in curves] == [0, 5, 10, 15]
        for current, previous in zip(curves, old_curves, strict=True):
            assert current["metrics"] == previous["metrics"]
            for key in ("val/loss_avg", "val/samples"):
                if key in current["substantive_validation"]:
                    assert current["substantive_validation"][key] == previous["substantive_validation"][key]
        live = {
            v["step"]: v
            for v in map(json.loads, (case / "diagnostics/late_substantive.jsonl").read_text().splitlines())
        }
        assert set(live) == set(range(10, 16))
        for step in (10, 15):
            checkpoint = checkpoints[f"iter_{step:06d}"]
            pinned = read_checkpoint_file(checkpoint / "model.pt")
            (case / f"checkpoint{step}_pin.json").write_text(json.dumps(pinned, indent=2))
            output = case / f"reload{step}"
            argv = [
                "torchrun",
                "--standalone",
                "--nproc-per-node=1",
                "-m",
                "tools.quantized_optimizers.score_dense",
                "--checkpoint",
                str(checkpoint),
                "--checkpoint-sha256",
                pinned["sha256"],
                "--data",
                os.environ["SFT_DATA_ROOT"] + "/dev-task.jsonl",
                "--selection",
                os.environ["SFT_DATA_ROOT"] + "/dev-task-manifest.json",
                "--substantive-data",
                os.environ["SFT_DATA_ROOT"] + "/validation.jsonl",
                "--substantive-batch-size",
                "4",
                "--expected-substantive-nll",
                str(live[step]["native_loss"]),
                "--output",
                str(output),
                "--seed",
                "43",
            ]
            with (case / f"reload{step}.log").open("x") as log:
                subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=True)
            restored = json.loads((output / "substantive-batches.json").read_text())
            assert all(restored[k] == live[step][k] for k in restored.keys() - {"expected_native_loss"})
        results[arm] = {
            "state_match": matches,
            "task_observations": 128,
            "reload_steps": [10, 15],
            "excluded_trainer_key": None,
            "provenance": provenance,
        }
        (args.root / "replay_parity.json").write_text(json.dumps(results, indent=2))
    (args.root / "replay_verified.json").write_text(json.dumps({"complete": True, "arms": results}, indent=2))


if __name__ == "__main__":
    main()

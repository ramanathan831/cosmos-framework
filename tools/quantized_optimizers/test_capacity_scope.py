"""Completing a deferred pressure arm must not rerun its completed control."""

import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_deferred_paged_arm_preserves_scope_and_token_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("RESULTS", str(tmp_path))
    monkeypatch.setenv("SOURCE", str(tmp_path))
    monkeypatch.setenv("CAPACITY_PRESSURE_ONLY", "1")
    monkeypatch.setenv("CAPACITY_PRESSURE_ARM", "paged_adamw_8bit")
    calls = []

    def invoke(command, *, env):
        calls.append(env["ARM_ONLY"])
        folder = Path(env["RESULTS"])
        folder.mkdir()
        (folder / (env["ARM_ONLY"] + ".log")).write_text("native training completed")
        metrics = folder / env["ARM_ONLY"] / "metrics.jsonl"
        metrics.parent.mkdir()
        metrics.write_text(
            json.dumps(
                {
                    "optim/optimizer_step": 2,
                    "optim/nonfinite_skips_total": 0,
                    "train/valid_token_count": 41,
                    "train/valid_sequence_count": 1,
                }
            )
            + "\n"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", invoke)
    with pytest.raises(RuntimeError, match="insufficient actual supervised tokens"):
        runpy.run_path(str(Path(__file__).with_name("capacity_followup.py")), run_name="__main__")
    assert calls == ["paged_adamw_8bit"]
    assert json.loads((tmp_path / "results.json").read_text())[0]["status"] == "qualification_failure"
    assert not (tmp_path / "complete.json").exists()


@pytest.mark.parametrize("pressure_only,arm", [("0", "paged_adamw_8bit"), ("1", "adamw")])
def test_invalid_pressure_scope_refused_before_launch(tmp_path, monkeypatch, pressure_only, arm):
    monkeypatch.setenv("RESULTS", str(tmp_path))
    monkeypatch.setenv("CAPACITY_PRESSURE_ONLY", pressure_only)
    monkeypatch.setenv("CAPACITY_PRESSURE_ARM", arm)
    with pytest.raises(ValueError, match="CAPACITY_PRESSURE_ARM"):
        runpy.run_path(str(Path(__file__).with_name("capacity_followup.py")), run_name="__main__")

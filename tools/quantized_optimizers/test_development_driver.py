"""A whole-seed allocation must not claim that other seed groups completed."""

import json
import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "seed,expected",
    [(29, ["adamw_bnb_8bit", "paged_adamw_8bit", "adamw"]), (43, ["paged_adamw_8bit", "adamw", "adamw_bnb_8bit"])],
)
def test_seed_scope_keeps_rotated_order_and_requires_all_seeds(tmp_path, monkeypatch, seed, expected):
    (tmp_path / "runtime_optimizer_tests.passed").write_text("passed\n")
    monkeypatch.setenv("RESULTS", str(tmp_path))
    monkeypatch.setenv("SOURCE", str(tmp_path))
    monkeypatch.setenv("TRIAL_SEED", str(seed))
    monkeypatch.setenv("WORK_SECONDS", "3500")
    calls = []

    def invoke(command, *, env, **kwargs):
        from types import SimpleNamespace

        calls.append((int(env["SEED"]), env["ARM_ONLY"]))
        path = Path(env["RESULTS"]) / env["ARM_ONLY"] / "metrics.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text('{"optim/optimizer_step":40,"optim/nonfinite_skips_total":0}\n')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", invoke)
    runpy.run_path(str(Path(__file__).with_name("run_development.py")), run_name="__main__")
    assert calls == [(seed, arm) for arm in expected]
    marker = json.loads((tmp_path / f"pending-seed{seed}.json").read_text())
    assert marker["scope_complete"] and not marker["complete"]
    assert sorted(p.name for p in tmp_path.glob("seed*")) == [f"seed{seed}"]

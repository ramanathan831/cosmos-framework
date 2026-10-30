# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from protocol import Experiment, assert_budget


def test_paired_arms_change_only_base_recipe_and_label():
    left = Experiment("nf4_control", "nf4", seed=17, microbatch=2)
    right = dataclasses.replace(left, name="bf16_control", arm="bf16")
    assert [x for x in left.overrides() if not x.startswith("job.name=")] == [
        x for x in right.overrides() if not x.startswith("job.name=")
    ]
    assert "rl.grad_accum=4" in left.overrides()
    assert "model.model_config.adapter.initialization_seed=17" in left.overrides()
    assert "rl.validation.seed=17" in left.overrides()


@pytest.mark.parametrize(
    "changes",
    [
        {"arm": "fp4"},
        {"rank": True},
        {"microbatch": 3},
        {"learning_rate": float("nan")},
        {"warmup_steps": 300},
        {"frames": 7},
        {"prefetch": 1},
        {"video_min_pixels": 999999},
    ],
)
def test_invalid_experiments_fail_before_launch(changes):
    with pytest.raises(ValueError):
        Experiment("test", "nf4", **changes) if "arm" not in changes else Experiment("test", **changes)


def test_budget_counts_failed_jobs_and_outstanding_reservations():
    records = [
        {"job_id": "failed", "terminal": True, "elapsed_gpu_hours": 2},
        {"job_id": "pending", "terminal": False, "reserved_gpu_hours": 4},
    ]
    assert assert_budget(records, next_gpu_hours=1) == 6
    with pytest.raises(ValueError, match="exceeded"):
        assert_budget(records, next_gpu_hours=43)
    with pytest.raises(ValueError, match="duplicate"):
        assert_budget(records + records, next_gpu_hours=0)

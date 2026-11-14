# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real child imports must respect the native simulator's separate package policy."""

import os
import subprocess
import sys

import pytest

from cosmos_framework.posttrain.embodied.sims.libero.adapter import LiberoEnvironment
from cosmos_framework.posttrain.embodied.sims.libero.config import LiberoConfig
from cosmos_framework.posttrain.roles.env.ppo_window import build_window_gateway
from cosmos_framework.posttrain.roles.env.sim_gateway import SimGatewayBody


@pytest.mark.parametrize("isolated", [False, True])
def test_native_gateway_child_imports_do_not_override_parent_policy(tmp_path, monkeypatch, isolated):
    model, simulator = tmp_path / "model", tmp_path / "simulator"
    for directory, version in ((model, "model-pin"), (simulator, "simulator-pin")):
        directory.mkdir()
        (directory / "separate_dependency.py").write_text(f"VERSION = {version!r}\n")
    monkeypatch.setenv("PYTHONPATH", str(model))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("B1K_SIM_IPC_DIR", str(tmp_path))
    monkeypatch.setattr(SimGatewayBody, "_start", lambda self: None)
    environment = LiberoEnvironment(
        LiberoConfig(sim_python=sys.executable, sim_pythonpath=str(simulator) if isolated else None)
    )
    gateway = build_window_gateway(environment, replica=0, n_lanes=1, seed=17)
    child_env = gateway._child_env()
    actual = subprocess.check_output(
        [sys.executable, "-c", "from separate_dependency import VERSION; print(VERSION)"],
        env=child_env,
        cwd=tmp_path,
        text=True,
    ).strip()
    assert actual == ("simulator-pin" if isolated else "model-pin")
    assert os.environ["PYTHONPATH"] == str(model)
    assert child_env["CUDA_VISIBLE_DEVICES"] == "0"

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Orchestration contracts; these do not qualify accelerator kernels or performance."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from protocol import file_sha256
from run import Stages, verify_kernel_gate


def test_failed_stage_retains_command_log_and_exit_code(tmp_path):
    stages = Stages(tmp_path, Path("/usr/bin/time"))
    with pytest.raises(RuntimeError, match="exited 7"):
        stages.run("intentional_failure", [sys.executable, "-c", "print('retained'); raise SystemExit(7)"])
    stage = tmp_path / "stages/intentional_failure"
    assert "retained" in (stage / "output.log").read_text()
    assert json.loads((stage / "outcome.json").read_text())["exit_code"] == 7
    assert json.loads((stage / "command.json").read_text())[0] == sys.executable
    with pytest.raises(FileExistsError):
        stages.run("intentional_failure", [sys.executable, "-c", "pass"])


@pytest.mark.parametrize("invalid", ["skip", "missing", "other_build", "changed_xml", None])
def test_gpu_gate_requires_complete_exact_build_without_skips(tmp_path, invalid):
    receipt = {"code_revision": "a" * 40, "wheel_sha256": "b" * 64}
    xml = tmp_path / "gpu.xml"
    count = 13 if invalid == "missing" else 14
    body = "".join(
        '<testcase name="kernel_'
        + str(i)
        + '">'
        + ("<skipped/>" if invalid == "skip" and i == 0 else "")
        + "</testcase>"
        for i in range(count)
    )
    xml.write_text("<testsuite>" + body + "</testsuite>")
    gate = tmp_path / "gate.json"
    gate.write_text(
        json.dumps(
            {
                "completed": True,
                "build": receipt if invalid != "other_build" else {},
                "junit_xml": str(xml),
                "junit_sha256": file_sha256(xml),
            }
        )
    )
    if invalid == "changed_xml":
        xml.write_text(xml.read_text() + "\n")
    if invalid is None:
        assert verify_kernel_gate(gate, receipt)["completed"]
    else:
        with pytest.raises(ValueError):
            verify_kernel_gate(gate, receipt)

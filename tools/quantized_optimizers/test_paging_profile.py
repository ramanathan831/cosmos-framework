"""A profiler can export successfully after its training child was terminated."""

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("paging_profile", Path(__file__).with_name("paging_profile.py"))
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


@pytest.mark.parametrize("receipt", [None, "invalid json", "{}", '{"returncode":false}', '{"returncode":-15}'])
def test_trace_export_does_not_hide_missing_or_failed_native_completion(tmp_path, receipt):
    if receipt is not None:
        (tmp_path / "attempt.json").write_text(receipt)
    assert profile.completion(tmp_path, 0) == 1
    assert json.loads((tmp_path / "profile-completion.json").read_text())["passed"] is False


@pytest.mark.parametrize("profiler_code,expected", [(0, 0), (1, 1)])
def test_both_native_and_profiler_must_succeed(tmp_path, profiler_code, expected):
    (tmp_path / "attempt.json").write_text('{"returncode":0}')
    assert profile.completion(tmp_path, profiler_code) == expected


def test_default_variant_reproduces_round_ten_um_fault_capture():
    assert profile.profile_options({}) == (
        "um_faults",
        ["--cuda-um-cpu-page-faults=true", "--cuda-um-gpu-page-faults=true"],
    )


def test_cuda_only_variant_disables_um_fault_tracing():
    assert profile.profile_options({"PROFILE_VARIANT": "cuda_only"})[1] == [
        "--cuda-um-cpu-page-faults=false",
        "--cuda-um-gpu-page-faults=false",
    ]


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="unknown PROFILE_VARIANT"):
        profile.profile_options({"PROFILE_VARIANT": "everything"})

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Trace accounting never adds annotations to their GPU kernel durations."""

import pytest

from tools.native_qlora_qualification.profile_trace import summarize, union, worker_spans


def event(name, start, duration, category="user_annotation", **args):
    return {"ph": "X", "name": name, "cat": category, "pid": 1, "tid": 1, "ts": start, "dur": duration, "args": args}


def test_nested_annotations_and_overlapping_device_streams_not_double_counted():
    events = [
        event("trainer.backward", 0, 100),
        event("sft.module.adapter_a.backward_recompute", 10, 40),
        event("aten::mm", 15, 10, "cpu_op", **{"External id": 7}),
        event("gemm", 1000, 30, "kernel", **{"External id": 7, "device": 0, "stream": 1}),
        event("kDequantizeBlockwise", 1010, 40, "kernel", device=0, stream=2),
        event("Memcpy HtoD", 1040, 20, "gpu_memcpy", device=0, stream=3),
    ]
    result = summarize({"traceEvents": events})
    assert result["cpu_phases"]["trainer.backward"]["inclusive_wall_us"] == 100
    assert result["cpu_phases"]["trainer.backward"]["exclusive_named_phase_wall_us"] == 60
    assert result["device_observation_union_us"] == {"0": 60}
    assert sum(r["summed_duration_us"] for r in result["device_operations"].values()) == 90
    assert result["device_operations"]["sft.module.adapter_a.backward_recompute"]["count"] == 1
    assert result["device_operations"]["dequantization"]["summed_duration_us"] == 40
    assert not result["end_to_end_speed_evidence"]


def test_ambiguous_external_id_retains_unattributed_kernel():
    result = summarize(
        {
            "traceEvents": [
                event("trainer.optimizer", 0, 10),
                event("a", 1, 2, "cpu_op", **{"External id": 1}),
                event("b", 4, 2, "cpu_op", **{"External id": 1}),
                event("gemm", 12, 2, "kernel", **{"External id": 1}),
            ]
        }
    )
    assert result["unattributed_kernel_count"] == 1
    assert "trainer.optimizer" not in result["device_operations"]


def test_peer_child_union_and_different_cpu_threads():
    a = event("sft.parent", 0, 100)
    b = event("sft.child", 10, 30)
    c = event("sft.peer", 20, 30)
    other = event("sft.other_thread", 0, 90)
    other["tid"] = 2
    report = summarize({"traceEvents": [a, b, c, other]})
    assert report["cpu_phases"]["sft.parent"]["exclusive_named_phase_wall_us"] == 60
    assert union([(2, 3), (0, 5), (4, 8)]) == 8


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), -1, True, None])
def test_invalid_complete_event_duration_rejected(duration):
    with pytest.raises(ValueError):
        summarize({"traceEvents": [event("sft.invalid", 0, duration)]})


def test_worker_spans_preserve_nested_decode_and_process_identity():
    def row(name, pid, start, stop):
        return {
            "protocol": "cosmos.sft-cpu-span.v1",
            "name": name,
            "pid": pid,
            "thread": 1,
            "begin_monotonic_ns": start,
            "end_monotonic_ns": stop,
            "inclusive_wall_ns": stop - start,
        }

    result = worker_spans(
        [
            row("sft.processor", 1, 0, 100000),
            row("sft.video_decode_and_sample", 1, 10000, 50000),
            row("sft.prefetch_wait", 2, 0, 90000),
        ]
    )
    assert result["sft.processor"]["exclusive_named_phase_wall_us"] == 60
    assert result["sft.prefetch_wait"]["exclusive_named_phase_wall_us"] == 90
    bad = row("sft.processor", 1, 0, 10)
    bad["inclusive_wall_ns"] = 9
    with pytest.raises(ValueError, match="duration"):
        worker_spans([bad])

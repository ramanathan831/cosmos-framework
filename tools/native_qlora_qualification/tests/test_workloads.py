# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_DIRECTORY = Path(__file__).parents[1]
sys.path.insert(0, str(_DIRECTORY))
_SPEC = importlib.util.spec_from_file_location("workloads", _DIRECTORY / "workloads.py")
assert _SPEC and _SPEC.loader
workloads = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(workloads)


def original():
    return {
        "id": "fixture",
        "messages": [
            {"role": "system", "content": "Answer with a letter."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "/fixture/video.mp4"},
                    {"type": "text", "text": "Original question A: first B: second"},
                ],
            },
            {"role": "assistant", "content": "B"},
        ],
    }


def test_long_target_preserves_media_question_and_original_gold():
    row = original()
    snapshot = copy.deepcopy(row)
    changed = workloads.long_row(row, 3, 2)
    assert row == snapshot
    assert changed["messages"][1]["content"][:2] == row["messages"][1]["content"]
    answer = changed["messages"][-1]["content"]
    assert answer.startswith("B\nRecord 2001:") and "Record 2003:" in answer
    assert "synthetic text, not a claim" in changed["messages"][1]["content"][-1]["text"]
    assert changed == workloads.long_row(row, 3, 2)
    assert changed != workloads.long_row(row, 3, 3)


def test_video_total_pixel_budgets_and_microbatch_padding():
    small = workloads.processor_options(8, 65_536)["videos_kwargs"]
    large = workloads.processor_options(16, 262_144)["videos_kwargs"]
    assert small["size"] == {"shortest_edge": 32768, "longest_edge": 524288}
    assert large["size"] == {"shortest_edge": 65536, "longest_edge": 4194304}
    assert large["fps"] is None
    assert workloads.padding_summary([100, 105, 200, 150], 2) == {
        "microbatch": 2,
        "unpadded_tokens": 555,
        "padded_tokens": 610,
        "padding_tokens": 55,
    }
    for frames in (1, 3, True):
        with pytest.raises(ValueError):
            workloads.processor_options(frames, 65536)
    with pytest.raises(ValueError):
        workloads.padding_summary([1, 2, 3], 2)


def test_fits_actual_complete_targets_and_refuses_decoder_failure():
    def process(row):
        records = row["messages"][-1]["content"].count("Record ")
        length = 200 + records * 41
        if length > 4096:
            raise ValueError(f"SFT sequence length {length} exceeds max_length=4096; truncation is forbidden")
        return SimpleNamespace(input_ids=[1] * length, labels=[-100] * 200 + [2] * (length - 200))

    row, sample, count = workloads.fit_long_row(original(), 0, process)
    assert 3840 <= len(sample.input_ids) <= 4096 and count == 95
    assert row["messages"][-1]["content"].count("Record ") == count

    def fail(row):
        raise ValueError("decoder failed")

    with pytest.raises(ValueError, match="decoder failed"):
        workloads.fit_long_row(original(), 0, fail)
    with pytest.raises(ValueError, match="lower bound"):
        workloads.fit_long_row(original(), 0, lambda row: SimpleNamespace(input_ids=[1] * 100, labels=[2] * 100))


def test_selected_rows_are_fixed_and_duplicates_fail(tmp_path):
    path = tmp_path / "source.jsonl"
    rows = [{**original(), "id": str(i)} for i in range(8)]
    media = tmp_path / "fixture.mp4"
    media.write_bytes(b"read-only metadata fixture; no decoder called")
    for row in rows:
        row["messages"][1]["content"][0]["video"] = str(media)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    first = workloads.select_rows(path, 4, 42)
    assert first == workloads.select_rows(path, 4, 42)
    assert [int(row["id"]) for row in first] == sorted(int(row["id"]) for row in first)
    path.write_text(path.read_text() + json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="distinct"):
        workloads.select_rows(path, 4, 42)

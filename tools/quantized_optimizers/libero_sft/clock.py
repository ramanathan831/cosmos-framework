# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Locked time-to-target clock and evaluation index for the LIBERO-10 SFT protocol.

Clock: cumulative training-process wall seconds across allocations, from each link's
first log line (process start: setup and checkpoint load included) to the completed
save of that link's final checkpoint (iteration 250k). Queue time and evaluation are
excluded. Standard library only (runs on the login node).
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

STAMP = re.compile(r"^\[(\d\d-\d\d \d\d:\d\d:\d\d)\|")
SAVED = re.compile(r"save_state_dict_worker\] Saved checkpoint to .*/checkpoints/?")
COMPLETED = re.compile(r"Checkpoint save completed")
ARMS = {"fused": "FusedAdam", "8bit": "adamw_bnb_8bit"}


def _time(line: str, year: int) -> datetime | None:
    match = STAMP.match(line)
    return datetime.strptime(f"{year}-{match.group(1)}", "%Y-%m-%d %H:%M:%S") if match else None


def link_seconds(log: Path, year: int) -> float:
    """Seconds from the first timestamped line to the last completed checkpoint save."""
    start = end = None
    for line in log.read_text(errors="replace").splitlines():
        stamp = _time(line, year)
        if stamp is None:
            continue
        start = start or stamp
        if COMPLETED.search(line):
            end = stamp
    if start is None or end is None:
        raise ValueError(f"{log}: no completed checkpoint save")
    seconds = (end - start).total_seconds()
    if seconds < 0:  # midnight/new-year rollover is not expected within one 4 h link
        raise ValueError(f"{log}: negative link duration")
    return seconds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True, help="campaign work directory W")
    parser.add_argument("--year", type=int, default=2026)
    parser.add_argument("--clock-out", type=Path, required=True)
    parser.add_argument("--index-out", type=Path, required=True)
    args = parser.parse_args()
    clock: dict = {}
    index: dict = {}
    for tag, arm in ARMS.items():
        for seed in ("1", "2", "3"):
            run = f"cmp-{tag}-s{seed}"
            total = 0.0
            for k in range(1, 9):
                link = args.root / "train" / run / f"link-{k:02d}"
                status = link / "exit-code.txt"
                if not (link / "train.log").exists() or not status.exists() or status.read_text().strip() != "0":
                    break  # incomplete or failed link: the clock stops at the last completed link
                total += link_seconds(link / "train.log", args.year)
                iteration = str(250 * k)
                clock.setdefault(arm, {}).setdefault(seed, {})[iteration] = round(total, 1)
                evaluation = args.root / "eval" / run / f"iter_{250 * k:09d}"
                if (evaluation / "exit-code.txt").exists() and (evaluation / "exit-code.txt").read_text().strip() == "0":
                    index.setdefault(arm, {}).setdefault(seed, {})[iteration] = str(evaluation)
    args.clock_out.write_text(json.dumps(clock, indent=2) + "\n")
    args.index_out.write_text(json.dumps(index, indent=2) + "\n")
    print(json.dumps({arm: {s: len(v) for s, v in seeds.items()} for arm, seeds in index.items()}))


if __name__ == "__main__":
    main()

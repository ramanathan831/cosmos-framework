# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""CPU-only diagnostic trace accounting; never a throughput/quality selector.

Chrome CPU annotation durations overlap their operators and GPU kernels. Keep
inclusive annotation spans separate from non-overlapping interval unions and
from summed GPU operation durations (which can overlap on different streams).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path


def union(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, -math.inf
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def interval(event: dict) -> tuple[float, float]:
    values = [event.get("ts"), event.get("dur")]
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
        raise ValueError("trace duration/timestamp must be finite")
    if values[1] < 0:
        raise ValueError("negative trace duration")
    return values[0], values[0] + values[1]


def phase_totals(events: list[dict]) -> dict:
    """Exclusive means subtracting nested *named phase* wall spans on one thread.

    It is not CPU execution time: scheduling and waits remain within these spans.
    Overlapping peer spans are unioned and never subtracted twice.
    """
    phases = [
        e for e in events if e.get("cat") == "user_annotation" and e.get("name", "").startswith(("sft.", "trainer."))
    ]
    groups = defaultdict(list)
    for event in phases:
        groups[event["name"]].append(event)
    threads = defaultdict(list)
    for event in phases:
        threads[(event.get("pid"), event.get("tid"))].append(event)
    starts = {}
    for key, entries in threads.items():
        entries.sort(key=lambda e: e["ts"])
        starts[key] = [e["ts"] for e in entries]
    result = {}
    for name, group in sorted(groups.items()):
        inclusive = exclusive = 0.0
        for event in group:
            start, stop = interval(event)
            inclusive += stop - start
            nested = []
            key = (event.get("pid"), event.get("tid"))
            lo, hi = bisect_left(starts[key], start), bisect_right(starts[key], stop)
            for child in threads[key][lo:hi]:
                a, b = interval(child)
                if (
                    child is not event
                    and child.get("pid") == event.get("pid")
                    and child.get("tid") == event.get("tid")
                    and start <= a <= b <= stop
                    and (a, b) != (start, stop)
                ):
                    nested.append((a, b))
            exclusive += stop - start - union(nested)
        result[name] = {"calls": len(group), "inclusive_wall_us": inclusive, "exclusive_named_phase_wall_us": exclusive}
    return result


def summarize(trace: dict) -> dict:
    raw = trace.get("traceEvents")
    if not isinstance(raw, list):
        raise ValueError("Chrome traceEvents list required")
    events = [e for e in raw if e.get("ph") == "X"]
    for event in events:
        interval(event)
    annotations = [e for e in events if e.get("cat") == "user_annotation"]
    # External IDs may be reused across processes. Never silently select one.
    external = defaultdict(list)
    for event in events:
        if event.get("cat") == "cpu_op" and "External id" in event.get("args", {}):
            external[str(event["args"]["External id"])].append(event)
    # Resolve enclosing named module scopes in a per-thread sweep. This avoids
    # an O(kernel_count * annotation_count) scan on full native traces.
    lanes = defaultdict(list)
    for event in annotations + [e for e in events if e.get("cat") == "cpu_op"]:
        lanes[(event.get("pid"), event.get("tid"))].append(event)
    enclosing_by_event = {}
    for lane in lanes.values():
        active = []
        for event in sorted(lane, key=lambda e: (e["ts"], -e["dur"])):
            start, stop = interval(event)
            active = [e for e in active if interval(e)[1] >= start]
            if event.get("cat") == "user_annotation":
                if event.get("name", "").startswith(("sft.module.", "trainer.optimizer", "trainer.backward")):
                    active.append(event)
            else:
                parents = [e for e in active if interval(e)[1] >= stop]
                if parents:
                    enclosing_by_event[id(event)] = min(parents, key=lambda e: e["dur"])["name"]
    device, classes = defaultdict(list), defaultdict(list)
    unresolved = 0
    for event in events:
        category = event.get("cat", "")
        if category not in {"kernel", "gpu_memcpy", "gpu_memset"}:
            continue
        name, args = event.get("name", ""), event.get("args", {})
        key = str(args.get("device", event.get("pid", "unknown")))
        device[key].append(interval(event))
        if category == "gpu_memcpy":
            label = "transfers"
        elif category == "gpu_memset":
            label = "device_memset"
        elif "dequantize" in name.lower():
            label = "dequantization"
        else:
            candidates = external.get(str(args.get("External id")), [])
            enclosing = enclosing_by_event.get(id(candidates[0])) if len(candidates) == 1 else None
            if enclosing:
                label = enclosing
            else:
                label = "unattributed_kernel"
                unresolved += 1
        classes[label].append(interval(event))
    return {
        "protocol": "cosmos.native-qlora-trace-accounting.v1",
        "diagnostic_only": True,
        "end_to_end_speed_evidence": False,
        "cpu_phases": phase_totals(events),
        "device_observation_union_us": {k: union(v) for k, v in sorted(device.items())},
        "device_operations": {
            k: {"count": len(v), "summed_duration_us": sum(b - a for a, b in v), "wall_union_us": union(v)}
            for k, v in sorted(classes.items())
        },
        "unattributed_kernel_count": unresolved,
        "limitations": [
            "CPU ranges are wall time, including waits; inclusive parent/child values must not be added.",
            "Device class unions can overlap each other; summed operation durations can exceed elapsed time across streams.",
            "GPU attribution requires one unique CPU external ID and its enclosing module range; unmatched kernels remain unattributed.",
            "Traced measurements cannot establish end-to-end speed or quality; retain cold and untraced complete runs separately.",
        ],
    }


def worker_spans(rows: list[dict]) -> dict:
    events = []
    for row in rows:
        if row.get("protocol") != "cosmos.sft-cpu-span.v1" or row.get("inclusive_wall_ns") != row.get(
            "end_monotonic_ns", 0
        ) - row.get("begin_monotonic_ns", 0):
            raise ValueError("CPU span protocol/duration differs")
        events.append(
            {
                "ph": "X",
                "cat": "user_annotation",
                "pid": row["pid"],
                "tid": row["thread"],
                "name": row["name"],
                "ts": row["begin_monotonic_ns"] / 1000,
                "dur": row["inclusive_wall_ns"] / 1000,
            }
        )
    for event in events:
        interval(event)
    return phase_totals(events)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--cpu-spans", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.trace.read_bytes()
    result = summarize(json.loads(gzip.decompress(raw) if args.trace.suffix == ".gz" else raw))
    files = {str(args.trace): hashlib.sha256(raw).hexdigest()}
    spans = []
    for path in args.cpu_spans:
        content = path.read_bytes()
        files[str(path)] = hashlib.sha256(content).hexdigest()
        spans.extend(json.loads(line) for line in content.splitlines())
    result["worker_and_parent_cpu_spans"] = worker_spans(spans)
    result["source_files_sha256"] = files
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()

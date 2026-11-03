"""Extract UVM traffic directly; nsys summary scripts require absent CPU faults."""

import argparse
import json
import sqlite3
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("database", type=Path)
p.add_argument("output", type=Path)
args = p.parse_args()
conn = sqlite3.connect(args.database)
result = {
    "scope": "instrumented synthetic pressure diagnostic; not speed qualification",
    "time_unit": "ns",
    "duration_sum_note": "event durations may overlap; sums are not wall time",
    "cpu_faults": "no CPU page-fault table emitted",
    "updates": [],
}
for start, end, label in conn.execute("SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE 'optimizer_update_%'"):
    copies = []
    for direction, cause, count, size, duration in conn.execute(
        """SELECT o.label,c.label,COUNT(*),SUM(m.bytes),SUM(m.end-m.start)
           FROM CUPTI_ACTIVITY_KIND_MEMCPY m
           JOIN ENUM_CUDA_MEMCPY_OPER o ON o.id=m.copyKind
           LEFT JOIN ENUM_CUDA_UNIF_MEM_MIGRATION c ON c.id=m.migrationCause
           WHERE m.start>=? AND m.end<=? AND m.migrationCause IS NOT NULL
           GROUP BY m.copyKind,m.migrationCause""",
        (start, end),
    ):
        copies.append(dict(direction=direction, cause=cause, events=count, bytes=size, duration_sum_ns=duration))
    faults = conn.execute(
        "SELECT COUNT(*),SUM(numberOfPageFaults),SUM(end-start) FROM CUDA_UM_GPU_PAGE_FAULT_EVENTS WHERE start>=? AND end<=?",
        (start, end),
    ).fetchone()
    result["updates"].append(
        dict(
            label=label,
            elapsed_ns=end - start,
            copies=copies,
            gpu_fault_events=faults[0],
            gpu_fault_count=faults[1],
            gpu_fault_duration_sum_ns=faults[2],
        )
    )
args.output.write_text(json.dumps(result, indent=2) + "\n")

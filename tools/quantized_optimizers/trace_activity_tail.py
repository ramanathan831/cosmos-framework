"""Locate where GPU work stops in an exported Nsight Systems SQLite trace.

Usage: python trace_activity_tail.py <managed-memory.sqlite> > tail.json
Per (process, device): kernel/memcpy counts, first/last activity relative to session
start, the last kernels issued, and bitsandbytes optimizer-kernel durations (if any).
Read-only; never modifies the trace.
"""

import json
import sqlite3
import sys


def main(path: str) -> dict:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    start = db.execute("SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0]
    out = {"tables_present": sorted(t for t in tables if t.startswith(("CUPTI", "CUDA", "TARGET")))}
    names = dict(db.execute("SELECT id, value FROM StringIds")) if "StringIds" in tables else {}
    per = {}
    for pid, device, count, first, last in db.execute(
        "SELECT globalPid, deviceId, COUNT(*), MIN(start), MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL GROUP BY 1, 2"
    ):
        key = f"pid{pid >> 24 & 0xFFFFFF}-dev{device}"
        tail = db.execute(
            "SELECT shortName, start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE globalPid=? AND deviceId=? "
            "ORDER BY end DESC LIMIT 5",
            (pid, device),
        ).fetchall()
        optimizer = db.execute(
            "SELECT COUNT(*), AVG(end-start), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "JOIN StringIds s ON k.shortName = s.id WHERE k.globalPid=? AND k.deviceId=? "
            "AND (s.value LIKE '%Optimizer%' OR s.value LIKE '%optimizer%')",
            (pid, device),
        ).fetchone()
        per[key] = {
            "kernels": count,
            "first_s": (first - start) / 1e9,
            "last_s": (last - start) / 1e9,
            "last_kernels": [[names.get(n, n), (s - start) / 1e9, (e - s) / 1e6] for n, s, e in tail],
            "optimizer_kernels": {"count": optimizer[0], "mean_ms": (optimizer[1] or 0) / 1e6, "total_s": (optimizer[2] or 0) / 1e9},
        }
    out["per_process_device"] = per
    if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
        out["memcpy_by_kind"] = [
            {"copyKind": kind, "count": count, "bytes": total, "seconds": seconds / 1e9}
            for kind, count, total, seconds in db.execute(
                "SELECT copyKind, COUNT(*), SUM(bytes), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY 1"
            )
        ]
    out["session_span_s"] = (db.execute("SELECT MAX(end) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0] - start) / 1e9
    return out


if __name__ == "__main__":
    print(json.dumps(main(sys.argv[1]), indent=1))

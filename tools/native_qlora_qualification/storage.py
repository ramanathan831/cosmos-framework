# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Small disposable writes on the actual output filesystem before model loading.

Successful probes establish only point-in-time availability, not reserved capacity
or a guarantee that a later checkpoint will fit. Callers retain their existing
subprocess timeout and allocation limits; no quota or scheduler settings change.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import stat
import time
import uuid
from pathlib import Path


class StorageProbeError(RuntimeError):
    def __init__(self, report: dict):
        self.report = report
        super().__init__("output storage probe failed; no capacity was reserved")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _identity(path: Path) -> tuple[int, int]:
    value = path.lstat()
    return value.st_dev, value.st_ino


def _error(error: Exception, operation: str) -> dict:
    return {
        "operation": operation,
        "type": type(error).__name__,
        "errno": getattr(error, "errno", None),
        "message": str(error),
    }


def probe_storage(paths: list[Path], *, context: str, files_per_path: int = 4, bytes_per_file: int = 65536) -> dict:
    """Probe existing parents using only fresh private children, then remove them.

    Refuse invalid/ambiguous targets before creating anything. Cleanup uses only
    the exact paths and inode identities created here, never a recursive deletion.
    Operational errors are returned, including cleanup errors; callers must check
    ``passed`` or use ``require_storage``. A compute probe runs under the job UID.
    """
    if context not in {"login", "compute", "process"}:
        raise ValueError("storage context must be login, compute or process")
    if not 1 <= len(paths) <= 8:
        raise ValueError("storage probe requires one to eight explicit paths")
    if type(files_per_path) is not int or not 1 <= files_per_path <= 16 or len(paths) * files_per_path > 32:
        raise ValueError("storage probe permits at most 32 temporary files")
    if type(bytes_per_file) is not int or not 1 <= bytes_per_file <= 65536:
        raise ValueError("storage probe file size must be between 1 and 65536 bytes")
    parents = [Path(path) for path in paths]
    if any(not p.is_absolute() or ".." in p.parts for p in parents):
        raise ValueError("storage probe paths must be absolute without parent traversal")
    resolved = [p.resolve(strict=True) for p in parents]
    if len(set(resolved)) != len(resolved) or any(p.is_symlink() or not p.is_dir() for p in parents):
        raise ValueError("storage probe requires distinct existing non-symlink directories")
    job = os.environ.get("SLURM_JOB_ID")
    if os.getuid() != os.geteuid():
        raise ValueError("storage probe refuses differing real and effective UID")
    if context == "compute":
        if not job or not job.isdecimal():
            raise ValueError("compute storage probe requires the actual SLURM_JOB_ID")
        allocated_uid = os.environ.get("SLURM_JOB_UID")
        if allocated_uid is not None and (not allocated_uid.isdecimal() or int(allocated_uid) != os.geteuid()):
            raise ValueError("compute storage probe UID differs from SLURM_JOB_UID")
    report = {
        "protocol": "cosmos.native-output-storage-probe.v1",
        "context": context,
        "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "uid": os.getuid(),
        "effective_uid": os.geteuid(),
        "gid": os.getgid(),
        "effective_gid": os.getegid(),
        "supplementary_groups": os.getgroups(),
        "allocation_id": job if context == "compute" else None,
        "host_sha256": hashlib.sha256(os.uname().nodename.encode()).hexdigest(),
        "reservation_created": False,
        "scope": "Point-in-time small-file and nested-directory availability only; not sustained capacity or checkpoint-size assurance.",
        "files_per_path": files_per_path,
        "bytes_per_file": bytes_per_file,
        "paths": [],
    }
    payload = (bytes(range(256)) * ((bytes_per_file + 255) // 256))[:bytes_per_file]
    for parent, canonical in zip(parents, resolved):
        started = time.monotonic()
        row = {
            "path": str(parent),
            "resolved_path": str(canonical),
            "passed": False,
            "operations": [],
            "cleanup_errors": [],
        }
        report["paths"].append(row)
        created: list[tuple[Path, tuple[int, int] | None]] = []
        operation = "parent_identity"
        try:
            parent_identity = _identity(parent)
            probe = parent / (".native-storage-probe-" + uuid.uuid4().hex)
            row["probe"] = str(probe)
            for directory in (probe, probe / "new-parent", probe / "new-parent" / "work"):
                operation = "mkdir"
                directory.mkdir(mode=0o700)
                created.append((directory, None))
                created[-1] = (directory, _identity(directory))
                row["operations"].append({"operation": operation, "path": str(directory)})
            leaf = created[-1][0]
            for index in range(files_per_path):
                source = leaf / f"part-{index:02d}.tmp"
                destination = leaf / f"part-{index:02d}.bin"
                operation = "create_write_fsync"
                with source.open("xb") as stream:
                    created.append((source, None))
                    created[-1] = (source, _identity(source))
                    if stream.write(payload) != len(payload):
                        raise OSError("short storage probe write")
                    stream.flush()
                    os.fsync(stream.fileno())
                operation = "atomic_rename"
                source.rename(destination)
                created[-1] = (destination, created[-1][1])
                operation = "directory_fsync"
                _fsync_directory(leaf)
                operation = "readback"
                if destination.read_bytes() != payload:
                    raise OSError("storage probe readback differs")
                row["operations"].append({"operation": "write_fsync_rename_readback", "bytes": len(payload)})
            operation = "directory_fsync"
            for directory in (leaf.parent, probe, parent):
                _fsync_directory(directory)
            operation = "parent_identity"
            if _identity(parent) != parent_identity or parent.resolve(strict=True) != canonical:
                raise OSError("storage probe parent changed during operations")
            row["passed"] = True
        except Exception as error:
            row["error"] = _error(error, operation)
        finally:
            for path, identity in reversed(created):
                try:
                    if identity is None or _identity(path) != identity:
                        raise OSError("storage probe cleanup refuses changed inode")
                    if stat.S_ISDIR(path.lstat().st_mode):
                        path.rmdir()
                    else:
                        path.unlink()
                except Exception as error:
                    row["cleanup_errors"].append({"path": str(path), **_error(error, "cleanup")})
            try:
                _fsync_directory(parent)
            except Exception as error:
                row["cleanup_errors"].append(_error(error, "cleanup_directory_fsync"))
            row["cleanup_complete"] = not row["cleanup_errors"]
            row["passed"] = row["passed"] and row["cleanup_complete"]
            row["elapsed_seconds"] = time.monotonic() - started
    report["passed"] = all(row["passed"] for row in report["paths"])
    return report


def require_storage(paths: list[Path], *, context: str) -> dict:
    report = probe_storage(paths, context=context)
    if not report["passed"]:
        raise StorageProbeError(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, action="append", required=True)
    parser.add_argument("--context", choices=("login", "compute", "process"), required=True)
    parser.add_argument("--files-per-path", type=int, default=4)
    parser.add_argument("--bytes-per-file", type=int, default=65536)
    parser.add_argument("--output", type=Path, help="Optional new receipt; its existing parent must be writable.")
    args = parser.parse_args()
    try:
        report = probe_storage(
            args.path, context=args.context, files_per_path=args.files_per_path, bytes_per_file=args.bytes_per_file
        )
    except Exception as error:
        report = {
            "protocol": "cosmos.native-output-storage-probe.v1",
            "passed": False,
            "reservation_created": False,
            "error": _error(error, "validate"),
        }
    raw = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(raw, end="", flush=True)
    if args.output:
        with args.output.open("x") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(args.output.parent)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

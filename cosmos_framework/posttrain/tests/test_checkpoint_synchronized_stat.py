# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Checkpoint identity must not depend on stale client-side network attributes."""

import ctypes
import errno
import os
import stat
import struct
import sys
from types import SimpleNamespace

import pytest

from cosmos_framework.posttrain.checkpoint.provenance import checkpoint_inventory_signature
from cosmos_framework.posttrain.checkpoint.synchronized_stat import synchronized_stat

pytestmark = [
    pytest.mark.L0,
    pytest.mark.CPU,
    pytest.mark.skipif(sys.platform != "linux", reason="Linux server-synchronized statx qualification"),
]


def test_stale_client_timestamp_does_not_change_checkpoint_generation(tmp_path, monkeypatch):
    payload = tmp_path / "model.distcp"
    payload.write_bytes(b"a" * (5 * 1024 * 1024))
    expected = checkpoint_inventory_signature(str(tmp_path))
    original = os.stat

    def stale(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if os.fspath(path) == str(payload):
            return SimpleNamespace(
                st_size=result.st_size,
                st_dev=result.st_dev,
                st_ino=result.st_ino,
                st_mode=result.st_mode,
                st_mtime_ns=result.st_mtime_ns - 2_000_000_000,
                st_ctime_ns=result.st_ctime_ns - 2_000_000_000,
            )
        return result

    # Observed native failure: same inode and size, one rank's normal stat
    # reports mtime/ctime two seconds behind its peers. Never drop these fields
    # or retry until signatures happen to agree.
    with monkeypatch.context() as patch:
        patch.setattr(os, "stat", stale)
        assert checkpoint_inventory_signature(str(tmp_path)) == expected
    os.utime(payload, ns=(original(payload).st_atime_ns, original(payload).st_mtime_ns + 2_000_000_000))
    assert checkpoint_inventory_signature(str(tmp_path)) != expected


def test_synchronized_fields_match_lstat_and_preserve_symlink_identity(tmp_path):
    payload = tmp_path / "unicode-\u00e9 space"
    payload.write_bytes(b"checkpoint")
    link = tmp_path / "link"
    link.symlink_to(payload.name)
    for path in (payload, link):
        expected, actual = os.lstat(path), synchronized_stat(path)
        for field in ("st_mode", "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"):
            assert getattr(actual, field) == getattr(expected, field)
    assert stat.S_ISLNK(synchronized_stat(link).st_mode)
    with pytest.raises(FileNotFoundError):
        synchronized_stat(tmp_path / "missing")
    with pytest.raises(ValueError, match="null byte"):
        synchronized_stat("a\0b")


@pytest.mark.parametrize("failure", ["error", "missing_fields"])
def test_statx_failure_does_not_fall_back_to_cached_metadata(tmp_path, monkeypatch, failure):
    import importlib

    module = importlib.import_module("cosmos_framework.posttrain.checkpoint.synchronized_stat")
    payload = tmp_path / "payload"
    payload.write_bytes(b"value")

    def statx(dirfd, path, flags, mask, buffer):
        assert flags == 0x2100 and mask == 0x7FF and dirfd == -100
        if failure == "error":
            ctypes.set_errno(errno.EIO)
            return -1
        struct.pack_into("=I", buffer, 0, 0)
        return 0

    monkeypatch.setattr(module, "_statx", lambda: statx)
    with pytest.raises(OSError if failure == "error" else RuntimeError):
        synchronized_stat(payload)

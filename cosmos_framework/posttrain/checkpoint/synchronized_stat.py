# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Read server-synchronized Linux attributes for immutable checkpoint identity.

Opt-in: set ``COSMOS_CHECKPOINT_SHARED_FS_IDENTITY=1`` for multi-rank runs on a network
filesystem (observed on Lustre) whose ranks see different client-cached timestamps for one
finished checkpoint. It makes checkpoint provenance use forced-sync ``statx`` and validation
publication bind full payload content. Without it, identity uses plain ``os.stat`` and the
metadata + small-file signature, exactly as before, at no extra cost. Keep the setting
constant for a run: it changes the publication identity recorded in checkpoints.

Normal stat may return stale network-client mtime/ctime after a distributed
writer finishes. statx(AT_STATX_FORCE_SYNC) asks the filesystem to synchronize
attributes with its server. It does not weaken identity, retry disagreement or
make a concurrently modified checkpoint valid.

Linux interface: https://man7.org/linux/man-pages/man2/statx.2.html
Layout: the original basic-stat prefix of Linux UAPI include/uapi/linux/stat.h.
"""

from __future__ import annotations

import ctypes
import errno
import os
import struct
import sys
from functools import lru_cache
from types import SimpleNamespace

_AT_FDCWD = -100
_AT_SYMLINK_NOFOLLOW = 0x100
_AT_STATX_FORCE_SYNC = 0x2000
_AT_EMPTY_PATH = 0x1000
_STATX_BASIC_STATS = 0x7FF
_REQUIRED = 0x001 | 0x040 | 0x080 | 0x100 | 0x200  # type, ctime, mtime, inode, size


SHARED_FS_IDENTITY_ENV = "COSMOS_CHECKPOINT_SHARED_FS_IDENTITY"


def shared_fs_identity() -> bool:
    """Whether the opt-in shared-filesystem checkpoint identity is enabled for this process."""
    return os.environ.get(SHARED_FS_IDENTITY_ENV, "").strip().lower() in ("1", "true", "yes")


def checkpoint_stat(path):
    """``synchronized_stat`` under the shared-filesystem opt-in; otherwise the plain no-follow stat."""
    if shared_fs_identity():
        return synchronized_stat(path)
    return os.fstat(path) if isinstance(path, int) else os.stat(path, follow_symlinks=False)


@lru_cache(maxsize=1)
def _statx():
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        function = libc.statx
    except AttributeError as error:
        raise RuntimeError(
            "Linux checkpoint provenance requires libc statx with server-synchronized attributes"
        ) from error
    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint, ctypes.c_void_p]
    function.restype = ctypes.c_int
    return function


def synchronized_stat(path):
    """Read a path without following symlinks, or the object pinned by an open fd.

    Linux statx's fixed-width UAPI is 256 bytes. Only its original basic-stat
    fields are read; later kernel extensions occupy the reserved tail. Keep
    integer nanoseconds and device/inode identity exactly as os.stat does.
    Non-Linux local callers retain their previous stat behavior; the network
    filesystem qualification uses the Linux server-synchronized path.
    """
    if sys.platform != "linux":
        return os.fstat(path) if isinstance(path, int) else os.stat(path, follow_symlinks=False)
    descriptor = isinstance(path, int)
    encoded = b"" if descriptor else os.fsencode(path)
    if b"\0" in encoded:
        raise ValueError("embedded null byte in checkpoint path")
    buffer = ctypes.create_string_buffer(256)
    dirfd = path if descriptor else _AT_FDCWD
    flags = _AT_STATX_FORCE_SYNC | (_AT_EMPTY_PATH if descriptor else _AT_SYMLINK_NOFOLLOW)
    try:
        statx = _statx()
    except RuntimeError:
        statx = None  # libc without statx: no synchronized attributes to ask for
    if statx is None:
        return os.fstat(path) if descriptor else os.stat(encoded, follow_symlinks=False)
    if statx(dirfd, encoded, flags, _STATX_BASIC_STATS, buffer):
        error = ctypes.get_errno()
        if error == errno.ENOSYS:  # kernel without statx: plain attributes are all there is
            return os.fstat(path) if descriptor else os.stat(encoded, follow_symlinks=False)
        raise OSError(error, os.strerror(error), path if descriptor else os.fsdecode(path))
    mask = struct.unpack_from("=I", buffer, 0)[0]
    if mask & _REQUIRED != _REQUIRED:
        raise RuntimeError(f"checkpoint filesystem omitted required synchronized stat fields: mask={mask:#x}")
    inode, size = struct.unpack_from("=QQ", buffer, 32)
    ctime_seconds, ctime_ns = struct.unpack_from("=qI", buffer, 96)
    mtime_seconds, mtime_ns = struct.unpack_from("=qI", buffer, 112)
    major, minor = struct.unpack_from("=II", buffer, 136)
    return SimpleNamespace(
        st_mode=struct.unpack_from("=H", buffer, 28)[0],
        st_dev=os.makedev(major, minor),
        st_ino=inode,
        st_size=size,
        st_mtime_ns=mtime_seconds * 1_000_000_000 + mtime_ns,
        st_ctime_ns=ctime_seconds * 1_000_000_000 + ctime_ns,
    )

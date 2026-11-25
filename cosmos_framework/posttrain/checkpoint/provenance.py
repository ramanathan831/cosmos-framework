# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Checkpoint provenance and atomic control-journal updates."""

from __future__ import annotations

import errno
import os
import stat
from collections.abc import Callable
from hashlib import sha256
from typing import Any

from cosmos_framework.posttrain.checkpoint.io import (
    atomic_write_json,
    component_kind,
    fsync_directory,
    is_complete_iter_dir,
)
from cosmos_framework.posttrain.checkpoint.synchronized_stat import synchronized_stat

EXTERNAL_COPY_PROVENANCE_FILE = ".external_copy_provenance.json"
CheckpointProvenance = dict[str, int | str]
_IDENTITY_FIELDS = ("st_mode", "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")


def _publication_file(path: str):
    """Hash the open regular file once and verify its descriptor and namespace identity.

    Opening comes before the initial attributes: a network filesystem may refresh
    pathname attributes when the file is opened. Both mutation checks describe
    the actual object read, rather than comparing a pre-open cache entry to it.
    A second no-follow open (attributes only, no second read) catches replacement
    of the pathname during the read. Every identity field is checked; there is no
    retry or relaxed timestamp. A same-size in-place rewrite that also preserves
    every server-synchronized attribute is outside what one read can observe.
    """

    def nofollow(name, flags):
        # A replaced FIFO must fail the regular-file check without blocking open.
        return os.open(name, flags | os.O_NOFOLLOW | os.O_NONBLOCK)

    def attributes(handle):
        result = synchronized_stat(handle.fileno())
        if not stat.S_ISREG(result.st_mode):
            raise RuntimeError(f"checkpoint publication requires regular files: {path!r}")
        return result

    def unchanged(before, after, stage):
        changed = {
            field: (getattr(before, field), getattr(after, field))
            for field in _IDENTITY_FIELDS
            if getattr(before, field) != getattr(after, field)
        }
        if changed:
            raise RuntimeError(f"checkpoint file changed while hashing: {path!r}; {stage}={changed}")

    try:
        with open(path, "rb", opener=nofollow) as handle:
            before = attributes(handle)
            content, size = sha256(), 0
            while chunk := handle.read(1024 * 1024):
                content.update(chunk)
                size += len(chunk)
            unchanged(before, attributes(handle), "descriptor")
            if size != before.st_size:
                raise RuntimeError(
                    f"checkpoint file changed while hashing: {path!r}; bytes_read={size}, expected={before.st_size}"
                )
            with open(path, "rb", opener=nofollow) as current:
                unchanged(before, attributes(current), "namespace")
        return before, content
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise RuntimeError(f"checkpoint publication requires regular files: {path!r}") from error
        raise


def _rank() -> int:
    import torch.distributed as dist

    return dist.get_rank() if dist.is_initialized() else 0


_EXCLUDED_FILES = (".uploaded", ".uploaded.tmp", EXTERNAL_COPY_PROVENANCE_FILE, EXTERNAL_COPY_PROVENANCE_FILE + ".tmp")


def _inventory_walk(checkpoint_dir: str, overrides: dict[str, str] | None, full_content: bool):
    """Yield ``(relative_directory, None)`` then ``(relative_file, path)`` in a deterministic order."""
    normalized_overrides = {relative.replace(os.sep, "/"): path for relative, path in (overrides or {}).items()}
    for directory, dirnames, filenames in os.walk(checkpoint_dir):
        dirnames.sort()
        filenames.sort()
        if full_content:
            # os.walk does not descend directory symlinks. Reject them rather
            # than declaring a checkpoint content-bound with an unread subtree.
            for child in dirnames:
                child_path = os.path.join(directory, child)
                if not stat.S_ISDIR(synchronized_stat(child_path).st_mode):
                    raise RuntimeError(f"checkpoint publication requires real directories: {child_path!r}")
        relative_directory = os.path.relpath(directory, checkpoint_dir)
        yield relative_directory, None
        for filename in filenames:
            relative = filename if relative_directory == "." else os.path.join(relative_directory, filename)
            if relative in _EXCLUDED_FILES or relative.startswith(".") and relative.endswith(".pt.tmp"):
                continue
            yield relative, normalized_overrides.get(relative.replace(os.sep, "/"), os.path.join(directory, filename))


def publication_content_shard(checkpoint_dir: str, rank: int, world_size: int) -> dict[str, tuple[int, str]]:
    """Hash this rank's share of a checkpoint's payload files: ``{relative: (size, sha256 hex)}``.

    Files are dealt round-robin in the deterministic inventory order, so every byte
    is read once by exactly one rank. Combine the gathered shards with
    ``checkpoint_inventory_signature(..., full_content=True, content_digests=...)``.
    """
    if type(world_size) is not int or world_size < 1 or type(rank) is not int or not 0 <= rank < world_size:
        raise ValueError(f"invalid publication hashing rank {rank!r} of {world_size!r}")
    files = [(relative, path) for relative, path in _inventory_walk(checkpoint_dir, None, True) if path is not None]
    digests = {}
    for relative, path in files[rank::world_size]:
        attributes, content = _publication_file(path)
        digests[relative] = (attributes.st_size, content.hexdigest())
    return digests


def checkpoint_inventory_signature(
    checkpoint_dir: str,
    overrides: dict[str, str] | None = None,
    *,
    inventory: list[dict[str, Any]] | None = None,
    full_content: bool = False,
    content_digests: dict[str, tuple[int, str]] | None = None,
) -> str:
    """Identify a checkpoint, optionally reading every payload for publication.

    The historical metadata identity remains the load/external-copy contract.
    Publication compares full content across ranks: even FORCE_SYNC attributes
    differed between readers of a finished native Lustre checkpoint. Content mode
    checks each reader's metadata before/after every read and rejects any observed
    mutation; it does not retry or ignore a changed payload. ``content_digests``
    (from ``publication_content_shard`` on every rank) supplies already-verified
    hashes, so a distributed publication reads each byte once in total; this rank
    then checks only that every file is a regular file of the hashed size.
    """
    if content_digests is not None and not full_content:
        raise ValueError("content_digests require full_content=True")
    digest = sha256()
    if full_content:
        digest.update(b"cosmos-checkpoint-content-v2\0")
    seen: set[str] = set()
    for relative, path in _inventory_walk(checkpoint_dir, overrides, full_content):
        if path is None:
            digest.update(f"dir\0{relative}\0".encode("utf-8", errors="surrogateescape"))
            if inventory is not None:
                inventory.append({"directory": relative})
            continue
        if full_content:
            if content_digests is None:
                attributes, content = _publication_file(path)
                size, hexdigest = attributes.st_size, content.hexdigest()
            else:
                if relative not in content_digests:
                    raise RuntimeError(f"checkpoint publication inventory differs across ranks: {relative!r}")
                size, hexdigest = content_digests[relative]
                attributes = synchronized_stat(path)
                if not stat.S_ISREG(attributes.st_mode):
                    raise RuntimeError(f"checkpoint publication requires regular files: {path!r}")
                if attributes.st_size != size:
                    raise RuntimeError(
                        f"checkpoint file differs from its hashed size: {path!r}; size={attributes.st_size}, "
                        f"hashed={size}"
                    )
                seen.add(relative)
            record = f"file\0{relative}\0{size}\0"
            digest.update(record.encode("utf-8", errors="surrogateescape"))
            digest.update(bytes.fromhex(hexdigest))
            if inventory is not None:
                inventory.append({"record": record.replace("\0", "|"), "sha256": hexdigest})
            continue
        attributes = synchronized_stat(path)
        kind = "symlink" if stat.S_ISLNK(attributes.st_mode) else "file"
        if kind == "file" and attributes.st_size <= 4 * 1024 * 1024:
            # Small files are content-bound below. Ignoring inode/timestamps makes an atomically
            # renamed temp describe the same logical successor after promotion.
            record = f"{kind}\0{relative}\0{attributes.st_size}\0"
        else:
            record = (
                f"{kind}\0{relative}\0{attributes.st_dev}\0{attributes.st_ino}\0{attributes.st_size}\0"
                f"{attributes.st_mtime_ns}\0{attributes.st_ctime_ns}\0"
            )
        digest.update(record.encode("utf-8", errors="surrogateescape"))
        detail = {"record": record.replace("\0", "|")}
        if kind == "symlink":
            target = os.readlink(path)
            digest.update(target.encode("utf-8", errors="surrogateescape"))
            detail["symlink"] = target
        elif attributes.st_size <= 4 * 1024 * 1024:
            content = sha256()
            with open(path, "rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
                    content.update(chunk)
            detail["sha256"] = content.hexdigest()
        if inventory is not None:
            inventory.append(detail)
    if content_digests is not None and seen != set(content_digests):
        raise RuntimeError(
            f"checkpoint publication inventory differs across ranks: {sorted(set(content_digests) - seen)!r}"
        )
    return digest.hexdigest()


def checkpoint_provenance(
    source_dir: str,
    *,
    inventory: list[dict[str, Any]] | None = None,
    full_content: bool = False,
    content_digests: dict[str, tuple[int, str]] | None = None,
) -> CheckpointProvenance:
    """Build an identity for one immutable training checkpoint."""
    source = os.path.realpath(os.path.expanduser(source_dir))
    if not os.path.isdir(source):
        raise FileNotFoundError(f"external pending checkpoint source is not a directory: {source!r}")
    if not is_complete_iter_dir(source) or component_kind(source, "optim") is None:
        raise RuntimeError(
            f"external pending checkpoint source is incomplete: {source!r} "
            "(model, optimizer, and trainer components are required)"
        )
    signature = checkpoint_inventory_signature(
        source, inventory=inventory, full_content=full_content, content_digests=content_digests
    )
    return {"version": 2 if full_content else 1, "source": source, "signature": signature}


def load_with_stable_provenance(
    source_dir: str,
    load: Callable[[], Any],
    *,
    provenance: Callable[[str], CheckpointProvenance] = checkpoint_provenance,
) -> tuple[Any, CheckpointProvenance]:
    """Load only when the checkpoint identity stays stable across the entire read."""
    from cosmos_framework.posttrain.utils.dist import rank_identical, run_rank_converged

    before = run_rank_converged(
        lambda: provenance(source_dir),
        "checkpoint pre-load provenance capture",
    )
    before = rank_identical(before, "checkpoint pre-load provenance")
    value = load()
    after = run_rank_converged(
        lambda: provenance(source_dir),
        "checkpoint post-load provenance capture",
    )
    after = rank_identical(after, "checkpoint post-load provenance")
    if after != before:
        raise RuntimeError(f"checkpoint changed while it was being loaded: {source_dir!r}")
    return value, before


def read_external_copy_provenance(target_dir: str) -> dict[str, Any] | None:
    """Read a valid external-copy marker, failing closed on malformed state."""
    import json

    marker = os.path.join(target_dir, EXTERNAL_COPY_PROVENANCE_FILE)
    if not os.path.isfile(marker):
        return None
    try:
        with open(marker, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    if set(payload) not in (
        {"version", "source", "signature", "copy_signature"},
        {"version", "source", "signature", "copy_signature", "successor_signatures"},
    ):
        return None
    if payload.get("version") != 1 or not isinstance(payload.get("source"), str):
        return None
    if not isinstance(payload.get("signature"), str) or not isinstance(payload.get("copy_signature"), str):
        return None
    successors = payload.get("successor_signatures", [])
    if not isinstance(successors, list) or any(not isinstance(signature, str) for signature in successors):
        return None
    return payload


def target_matches_external_copy(target_dir: str, provenance: CheckpointProvenance) -> bool:
    """Return whether target is the exact copied source or an authorized journal successor."""
    if not is_complete_iter_dir(target_dir) or component_kind(target_dir, "optim") is None:
        return False
    marker = read_external_copy_provenance(target_dir)
    if marker is None:
        return False
    marker_source = {key: marker[key] for key in ("version", "source", "signature")}
    accepted_signatures = {str(marker["copy_signature"]), *marker.get("successor_signatures", [])}
    return marker_source == provenance and checkpoint_inventory_signature(target_dir) in accepted_signatures


def write_external_copy_provenance(staging_dir: str, provenance: CheckpointProvenance) -> None:
    """Durably record the source and initial copy signature inside staging."""
    marker_payload = {
        **provenance,
        "copy_signature": checkpoint_inventory_signature(staging_dir),
        "successor_signatures": [],
    }
    atomic_write_json(os.path.join(staging_dir, EXTERNAL_COPY_PROVENANCE_FILE), marker_payload)


def replace_component_atomic(iter_dir: str, name: str, state: dict[str, Any]) -> None:
    """Atomically replace one small component, extending external-copy provenance first."""
    import torch

    if not name or os.path.basename(name) != name:
        raise ValueError(f"checkpoint component name must be one path segment, got {name!r}")
    replacement_error: BaseException | None = None
    if _rank() == 0:
        temporary = os.path.join(iter_dir, f".{name}.pt.tmp")
        marker_temporary = os.path.join(iter_dir, EXTERNAL_COPY_PROVENANCE_FILE + ".tmp")
        try:
            from cosmos_framework.posttrain.checkpoint.worker import join_pending_rank0

            join_pending_rank0()
            if not os.path.isdir(iter_dir):
                raise FileNotFoundError(f"cannot replace component in missing checkpoint {iter_dir!r}")
            destination = os.path.join(iter_dir, f"{name}.pt")
            torch.save(state, temporary)
            with open(temporary, "rb") as handle:
                os.fsync(handle.fileno())
            marker_payload = read_external_copy_provenance(iter_dir)
            if marker_payload is not None:
                current_signature = checkpoint_inventory_signature(iter_dir)
                accepted_signatures = {
                    str(marker_payload["copy_signature"]),
                    *marker_payload.get("successor_signatures", []),
                }
                if current_signature not in accepted_signatures:
                    raise RuntimeError("external-copy provenance does not authorize the checkpoint's current inventory")
                successor_signature = checkpoint_inventory_signature(
                    iter_dir,
                    overrides={f"{name}.pt": temporary},
                )
                if successor_signature not in accepted_signatures:
                    # Marker first is deliberate: a crash before component rename leaves the old
                    # inventory authorized; a crash after it also authorizes the new inventory.
                    successors = [*marker_payload.get("successor_signatures", []), successor_signature]
                    atomic_write_json(
                        os.path.join(iter_dir, EXTERNAL_COPY_PROVENANCE_FILE),
                        {**marker_payload, "successor_signatures": successors},
                    )
            os.replace(temporary, destination)
            fsync_directory(iter_dir)
        except BaseException as error:  # noqa: BLE001 -- every distributed rank must exit together
            replacement_error = error
        finally:
            try:
                if os.path.exists(temporary):
                    os.remove(temporary)
                if os.path.exists(marker_temporary):
                    os.remove(marker_temporary)
            except BaseException as cleanup_error:  # noqa: BLE001 -- preserve cleanup failure too
                if replacement_error is None:
                    replacement_error = cleanup_error
    from cosmos_framework.posttrain.utils.dist import raise_if_any_rank_failed

    raise_if_any_rank_failed(replacement_error, f"atomic checkpoint component replacement ({name})")

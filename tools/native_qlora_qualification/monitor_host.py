# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Auditable, non-atomic Linux process-tree memory samples, without Torch/NVML.

Only flags=0 process pidfd POLLIN proves that the entire thread group exited.
A zombie leader alone does not prove the shared address space was released:
https://github.com/torvalds/linux/blob/v5.15/kernel/fork.c#L1748-L1765
All reads and failed attempts remain in the receipt. A retry is permitted only
for a proven exit or newly observed descendant. Unreadable live processes,
identity changes and stage races are incomplete, never interpreted as zero.
V3 binds PID/start to the process pidfd; proc inode ownership is retained
only as mutable metadata, never used to infer identity or released memory:
https://github.com/torvalds/linux/blob/v5.15/fs/proc/base.c#L1709-L1763
Ancestry admits new descendants. Already pinned descendants remain owned if
their parent exits and they are reparented; PPID is not process-birth identity.
"""

from __future__ import annotations

import errno
import os
import select
import time
from pathlib import Path
from typing import Any

try:
    from .processes import _PidfdBackend
except ImportError:  # Direct monitor.py invocation from the sealed tool directory.
    from processes import _PidfdBackend

LEGACY_METHOD = "proc-identity-state-read-attempts-v1"
EXIT_OWNER_METHOD = "proc-identity-state-read-attempts-v2"
METHOD = "proc-identity-state-read-attempts-v3"
IDENTITY_SEMANTICS = "pid_start_pidfd_owned_ancestry"
MAX_ATTEMPTS = 3
MAX_COLLECTION_SECONDS = 0.75
MAX_EXIT_WAIT_MILLISECONDS = 350
EXIT_WAIT_RETRY_RESERVE_SECONDS = 0.25


def _error(error: Exception) -> dict:
    return {"type": type(error).__name__, "errno": getattr(error, "errno", None)}


def read_proc(pid: int, name: str) -> dict:
    result: dict[str, Any] = {"begin_unix_seconds": time.time(), "begin_monotonic_seconds": time.monotonic()}
    try:
        path = Path("/proc") / str(pid) / name
        text = path.read_text()
        if name == "stat":
            tail = text.rsplit(")", 1)[1].split()
            value = {
                "pid": pid,
                "state": tail[0],
                "ppid": int(tail[1]),
                "start_ticks": int(tail[19]),
                # Historical wire name: this is proc inode ownership, not a
                # credential UID or immutable process identity. Linux
                # task_dump_owner() derives it from credentials/dumpability/mm;
                # it can change while the same process remains alive. Also,
                # this stat() is later than the text read, not an atomic snapshot.
                "uid": path.stat().st_uid,
            }
        else:
            fields = {"VmRSS", "VmHWM"} if name == "status" else {"Pss", "Rss"}
            value = {}
            for line in text.splitlines():
                key, _, raw = line.partition(":")
                if key in fields:
                    parts = raw.split()
                    if len(parts) != 2 or parts[1] != "kB" or int(parts[0]) < 0:
                        raise ValueError("invalid proc memory field")
                    value[key] = int(parts[0]) * 1024
        result["value"] = value
    except (OSError, ValueError, IndexError) as error:
        result["error"] = _error(error)
    result.update(end_unix_seconds=time.time(), end_monotonic_seconds=time.monotonic())
    return result


def _identity(value: dict, *, owner_is_identity: bool = True) -> tuple:
    # Keep the historical tuple exclusively for decoding v1/v2 evidence.
    fields = ("pid", "start_ticks", "uid") if owner_is_identity else ("pid", "start_ticks")
    return tuple(value.get(k) for k in fields)


def _valid_identity(value: Any, *, owner_is_identity: bool = True) -> bool:
    fields = ("pid", "start_ticks", "uid") if owner_is_identity else ("pid", "start_ticks", "ppid", "uid")
    return isinstance(value, dict) and all(type(value.get(k)) is int and value[k] >= 0 for k in fields)


def _released_owner_transition(value: Any, expected: Any, released: bool) -> bool:
    """An owner-only change is admissible solely for this proven exited task.

    A zombie leader is insufficient: `released` comes from the identity-bound
    process pidfd after final discovery, which waits for the entire group.
    Live owner changes, reparenting and changed birth identities remain errors.
    https://github.com/torvalds/linux/blob/v5.15/fs/proc/base.c#L1846-L1870
    """
    return (
        released
        and _valid_identity(value)
        and _valid_identity(expected)
        and value.get("state") in ("Z", "X")
        and all(value.get(key) == expected.get(key) for key in ("pid", "start_ticks", "ppid"))
        and type(value.get("ppid")) is int
        and value["uid"] != expected["uid"]
    )


def _read_stage(path: Path) -> dict:
    result = {"begin_unix_seconds": time.time(), "begin_monotonic_seconds": time.monotonic()}
    try:
        result["value"] = path.read_text().strip()
    except FileNotFoundError:
        result["value"] = "startup"
    except OSError as error:
        result["error"] = _error(error)
    result.update(end_unix_seconds=time.time(), end_monotonic_seconds=time.monotonic())
    return result


def _poll(fd: int | None, *, timeout_milliseconds: int = 0) -> dict:
    result = {"begin_unix_seconds": time.time(), "begin_monotonic_seconds": time.monotonic()}
    if timeout_milliseconds:
        result["timeout_milliseconds"] = timeout_milliseconds
    try:
        if fd is None:
            raise OSError(errno.EBADF, "no identity-bound pidfd")
        poller = select.poll()
        poller.register(fd, select.POLLIN)
        result["events"] = [mask for _, mask in poller.poll(timeout_milliseconds)]
    except OSError as error:
        result["error"] = _error(error)
    result.update(end_unix_seconds=time.time(), end_monotonic_seconds=time.monotonic())
    return result


def _released(poll: dict) -> bool | None:
    masks = poll.get("events")
    if "error" in poll or not isinstance(masks, list) or any(type(mask) is not int for mask in masks):
        return None
    if any(mask & ~(select.POLLIN | select.POLLHUP) or not mask & select.POLLIN for mask in masks):
        return None
    return any(mask & select.POLLIN for mask in masks)


def _receipt_valid(read: Any) -> bool:
    if not isinstance(read, dict):
        return False
    stamps = [
        read.get(key)
        for key in ("begin_monotonic_seconds", "end_monotonic_seconds", "begin_unix_seconds", "end_unix_seconds")
    ]
    return (
        all(type(value) in (int, float) and 0 <= value < float("inf") for value in stamps)
        and stamps[0] <= stamps[1]
        and (("error" in read) != ("value" in read or "events" in read))
    )


def classify_process(
    record: dict, *, released_after_discovery: bool = False, owner_is_identity: bool = True
) -> tuple[dict, list[str], bool]:
    """Recompute memory, hard errors and proven exit transition from raw reads."""
    errors: list[str] = []
    pin = record.get("pin", {})
    expected = pin.get("identity")
    if not _valid_identity(expected, owner_is_identity=owner_is_identity) or pin.get("flags") != 0 or pin.get("error"):
        errors.append("unbound_process_identity")
    for key in ("discovery", "before", "after"):
        read = pin.get(key, {})
        if (
            not _receipt_valid(read)
            or not _valid_identity(read.get("value"), owner_is_identity=owner_is_identity)
            or _identity(read["value"], owner_is_identity=owner_is_identity)
            != _identity(expected or {}, owner_is_identity=owner_is_identity)
        ):
            errors.append("invalid_pidfd_binding")
    if all(_receipt_valid(pin.get(key)) for key in ("discovery", "before", "after")):
        if any(
            pin[left]["end_monotonic_seconds"] > pin[right]["begin_monotonic_seconds"]
            for left, right in (("discovery", "before"), ("before", "after"))
        ):
            errors.append("invalid_pidfd_binding_order")
    reads = {
        name: record.get(name, {})
        for name in ("stat_before", "status", "rollup", "stat_after", "poll_before", "poll_after")
    }
    if not all(_receipt_valid(read) for read in reads.values()):
        errors.append("invalid_read_receipt")
    else:
        order = ("stat_before", "poll_before", "status", "rollup", "stat_after", "poll_after")
        if any(
            reads[left]["end_monotonic_seconds"] > reads[right]["begin_monotonic_seconds"]
            for left, right in zip(order, order[1:])
        ):
            errors.append("invalid_read_order")
    for name in ("stat_before", "stat_after"):
        value = reads[name].get("value")
        if (
            value is not None
            and (
                not _valid_identity(value, owner_is_identity=owner_is_identity)
                or _identity(value, owner_is_identity=owner_is_identity)
                != _identity(expected or {}, owner_is_identity=owner_is_identity)
            )
            and not (owner_is_identity and _released_owner_transition(value, expected, released_after_discovery))
        ):
            errors.append("process_identity_changed")
    before, after = _released(reads["poll_before"]), _released(reads["poll_after"])
    if before is None or after is None or (before is True and after is not True):
        errors.append("invalid_pidfd_exit_proof")
    transition = before is False and after is True
    if transition:
        for name in ("stat_before", "status", "rollup", "stat_after"):
            error = reads[name].get("error")
            if error is not None and error.get("errno") not in (errno.ENOENT, errno.ESRCH):
                errors.append("unexplained_read_error_before_exit")
    status, rollup = reads["status"].get("value", {}), reads["rollup"].get("value", {})
    memory = {key: status[key] for key in ("VmRSS", "VmHWM") if type(status.get(key)) is int and status[key] >= 0}
    memory.update(
        {key + "_rollup": rollup[key] for key in ("Pss", "Rss") if type(rollup.get(key)) is int and rollup[key] >= 0}
    )
    if before is True and after is True and not errors:
        # pidfd POLLIN requires the last thread to exit; never infer this from Z.
        memory.update(VmRSS=0, Pss_rollup=0, Rss_rollup=0, address_space_released=True)
    elif not transition:
        for name in ("stat_before", "stat_after"):
            if "error" in reads[name]:
                errors.append("live_or_unknown_identity_unreadable")
        if "VmRSS" not in memory or "Pss_rollup" not in memory or "Rss_rollup" not in memory:
            errors.append("live_or_unknown_memory_unreadable")
    if errors or transition:
        memory["rollup_unavailable"] = "Pss_rollup" not in memory
    return memory, sorted(set(errors)), transition


def _exit_wait_eligible(record: dict) -> bool:
    """Wait only for exit-compatible missing memory on a valid, still-owned pidfd."""
    _, errors, transition = classify_process(record, owner_is_identity=False)
    if (
        transition
        or "live_or_unknown_memory_unreadable" not in errors
        or set(errors) - {"live_or_unknown_memory_unreadable", "live_or_unknown_identity_unreadable"}
        or _released(record["poll_before"]) is not False
        or _released(record["poll_after"]) is not False
    ):
        return False
    # Permission failures and malformed/unknown reads must remain hard failures,
    # even if the same process happens to exit shortly afterward.
    return all(
        read.get("error", {}).get("errno") in (errno.ENOENT, errno.ESRCH) if "error" in read else True
        for read in (record[name] for name in ("stat_before", "status", "rollup", "stat_after"))
    )


def _validate_exit_wait(record: dict, *, method: str, collection_start: float) -> None:
    """Optional v3 acquisition detail; historical reads retain identical semantics."""
    wait = record.get("exit_wait")
    if wait is None:
        if "timeout_milliseconds" in record.get("poll_after", {}):
            raise ValueError("timed pidfd poll lacks its initial read and budget")
        return
    if (
        method != METHOD
        or not isinstance(wait, dict)
        or set(wait) != {"initial_poll", "deadline_monotonic_seconds", "retry_reserve_seconds"}
        or wait["retry_reserve_seconds"] != EXIT_WAIT_RETRY_RESERVE_SECONDS
        or wait["deadline_monotonic_seconds"] != collection_start + MAX_COLLECTION_SECONDS
    ):
        raise ValueError("invalid exit-wait bounds or method")
    initial, final = wait["initial_poll"], record.get("poll_after", {})
    timeout = final.get("timeout_milliseconds")
    if (
        not _receipt_valid(initial)
        or not _receipt_valid(final)
        or type(timeout) is not int
        or not 0 < timeout <= MAX_EXIT_WAIT_MILLISECONDS
        or record["stat_after"]["end_monotonic_seconds"] > initial["begin_monotonic_seconds"]
        or initial["end_monotonic_seconds"] > final["begin_monotonic_seconds"]
        or final["begin_monotonic_seconds"] + timeout / 1000
        > wait["deadline_monotonic_seconds"] - EXIT_WAIT_RETRY_RESERVE_SECONDS
        or not _exit_wait_eligible({**record, "poll_after": initial})
    ):
        raise ValueError("exit wait was not justified by owned exit-compatible reads")


def summarize_collection(receipt: dict) -> dict:
    """Fail closed and independently derive the final attempt's legacy fields."""
    method = receipt.get("protocol")
    if method not in (LEGACY_METHOD, EXIT_OWNER_METHOD, METHOD):
        raise ValueError("unsupported host collection protocol")
    owner_is_identity = method != METHOD
    if method == METHOD:
        if receipt.get("identity_semantics") != IDENTITY_SEMANTICS:
            raise ValueError("missing pidfd-bound process identity semantics")
    elif "identity_semantics" in receipt:
        raise ValueError("v3 identity semantics cannot be relabeled as v1/v2")
    final_polls_required = method in (EXIT_OWNER_METHOD, METHOD)
    if final_polls_required and receipt.get("uid_semantics") != "proc_inode_owner":
        raise ValueError("missing proc inode ownership semantics")
    attempts = receipt.get("attempts")
    if (
        not isinstance(attempts, list)
        or not 1 <= len(attempts) <= MAX_ATTEMPTS
        or receipt.get("final_attempt") != len(attempts) - 1
    ):
        raise ValueError("invalid host collection attempts")
    if receipt.get("max_attempts") != MAX_ATTEMPTS or receipt.get("max_collection_seconds") != MAX_COLLECTION_SECONDS:
        raise ValueError("unsupported collection bounds")
    hard_errors: list[str] = []
    first_stage = None
    first_start = None
    prior_after: set[str] = set()
    prior_discovery: dict = {}
    observed_rss, observed_pss = [], []
    final_memory: dict = {}
    final_transition = False
    for index, attempt in enumerate(attempts):
        started = attempt.get("begin_monotonic_seconds")
        if type(started) not in (int, float) or not 0 <= started < float("inf"):
            raise ValueError("invalid attempt start")
        if first_start is None:
            first_start = started
        elif not first_start <= started <= first_start + MAX_COLLECTION_SECONDS:
            raise ValueError("retry outside bounded collection interval")
        records = attempt.get("processes")
        before, after = attempt.get("selected_before"), attempt.get("selected_after")
        if not isinstance(records, dict) or not isinstance(before, list) or not isinstance(after, list) or not before:
            raise ValueError("missing selected process receipts")
        if (
            any(type(pid) is not int or pid <= 0 for pid in before + after)
            or len(set(before)) != len(before)
            or len(set(after)) != len(after)
        ):
            raise ValueError("invalid selected process identities")
        if (
            set(records) != {str(pid) for pid in before}
            or not prior_after <= set(records)
            or not set(before) <= set(after)
        ):
            raise ValueError("selected process silently omitted")
        if receipt.get("root_pid") not in before or receipt.get("excluded_monitor_pid") in before:
            raise ValueError("invalid observed tree scope")
        for suffix, selected in (("before", before), ("after", after)):
            discovery = attempt.get("discovery_" + suffix)
            if (
                not isinstance(discovery, dict)
                or set(discovery) != {str(pid) for pid in selected}
                or not all(_receipt_valid(read) for read in discovery.values())
            ):
                raise ValueError("missing discovery receipts")
        final_polls = attempt.get("post_discovery_polls", {})
        if final_polls_required:
            if not isinstance(final_polls, dict) or set(final_polls) != set(records):
                raise ValueError("missing post-discovery pidfd receipts")
            for pid, poll in final_polls.items():
                if not _receipt_valid(poll):
                    raise ValueError("invalid post-discovery pidfd receipt")
                preceding = [attempt["discovery_after"][pid], records[pid].get("poll_after", {})]
                if any(
                    not _receipt_valid(read) or read["end_monotonic_seconds"] > poll["begin_monotonic_seconds"]
                    for read in preceding
                ):
                    raise ValueError("post-discovery pidfd read precedes discovery")
                if (
                    not _receipt_valid(attempt.get("stage_after"))
                    or poll["end_monotonic_seconds"] > attempt["stage_after"]["begin_monotonic_seconds"]
                ):
                    raise ValueError("post-discovery pidfd read follows stage boundary")
        elif final_polls or "uid_semantics" in receipt:
            raise ValueError("v2 exit proof cannot be relabeled as v1")
        for pid in set(after) - set(before):
            ancestor, visited = pid, set()
            while ancestor not in before:
                value = attempt["discovery_after"].get(str(ancestor), {}).get("value", {})
                if (
                    ancestor in visited
                    or not _valid_identity(value, owner_is_identity=owner_is_identity)
                    or value.get("ppid") not in after
                ):
                    hard_errors.append("unproven_new_descendant")
                    break
                visited.add(ancestor)
                ancestor = value["ppid"]
        stages = [attempt.get(key, {}) for key in ("stage_before", "stage_after")]
        if not all(_receipt_valid(stage) and isinstance(stage.get("value"), str) for stage in stages) or stages[0].get(
            "value"
        ) != stages[1].get("value"):
            hard_errors.append("stage_changed_or_unreadable")
        if first_stage is None:
            first_stage = stages[0].get("value")
        elif stages[0].get("value") != first_stage:
            hard_errors.append("stage_changed_or_unreadable")
        memory, transitioned, process_errors = {}, False, []
        for pid, record in records.items():
            _validate_exit_wait(record, method=method, collection_start=first_start)
            if record.get("pin", {}).get("identity", {}).get("pid") != int(pid):
                hard_errors.append("process_key_identity_mismatch")
            expected = record.get("pin", {}).get("identity", {})
            final_released = _released(final_polls[pid]) if final_polls_required else False
            previous_released = _released(record.get("poll_after", {}))
            if final_polls_required and (
                final_released is None or (previous_released is True and final_released is not True)
            ):
                hard_errors.append("invalid_post_discovery_exit_proof")
            for discovery in (attempt["discovery_before"], attempt["discovery_after"], prior_discovery):
                observed = discovery.get(pid, {}).get("value")
                if (
                    observed is not None
                    and (
                        (not owner_is_identity and not _valid_identity(observed, owner_is_identity=False))
                        or _identity(observed, owner_is_identity=owner_is_identity)
                        != _identity(expected, owner_is_identity=owner_is_identity)
                    )
                    and not (
                        owner_is_identity and _released_owner_transition(observed, expected, final_released is True)
                    )
                ):
                    hard_errors.append("discovery_identity_changed")
            values, errors, changed = classify_process(
                record, released_after_discovery=final_released is True, owner_is_identity=owner_is_identity
            )
            memory[pid] = values
            hard_errors.extend(errors)
            process_errors.extend(errors)
            transitioned |= changed
            transitioned |= final_polls_required and previous_released is False and final_released is True
        if not process_errors:
            if all("VmRSS" in value for value in memory.values()):
                observed_rss.append(sum(value["VmRSS"] for value in memory.values()))
            if all("Pss_rollup" in value for value in memory.values()):
                observed_pss.append(sum(value["Pss_rollup"] for value in memory.values()))
        transitioned |= set(before) != set(after)
        if index < len(attempts) - 1 and (hard_errors or not transitioned):
            raise ValueError("retry lacks a proven membership transition")
        prior_after = {str(pid) for pid in after}
        prior_discovery = attempt["discovery_after"]
        final_memory, final_transition = memory, transitioned
    published = receipt.get("stage_publish")
    if published is not None and (not _receipt_valid(published) or published.get("value") != first_stage):
        hard_errors.append("stage_changed_or_unreadable")
    complete = (
        not hard_errors
        and not final_transition
        and all("Pss_rollup" in value and "VmRSS" in value for value in final_memory.values())
    )
    return {
        "process_host_memory": final_memory,
        "selected_host_rss_sum_bytes": sum(value.get("VmRSS", 0) for value in final_memory.values()),
        "selected_host_pss_sum_bytes": sum(value.get("Pss_rollup", 0) for value in final_memory.values()),
        "host_pss_complete": complete,
        "host_observed_attempt_rss_peak_bytes": max(observed_rss, default=None),
        "host_observed_complete_attempt_pss_peak_bytes": max(observed_pss, default=None),
        "host_collection_errors": sorted(
            set(hard_errors + (["membership_transition_retry_exhausted"] if final_transition else []))
        ),
    }


def _validate_sample(sample: dict) -> dict:
    """Validate each version without trusting producer totals or upgrading old evidence."""
    if sample.get("host_process_memory_method") not in (LEGACY_METHOD, EXIT_OWNER_METHOD, METHOD):
        raise ValueError("unsupported host memory method")
    receipt = sample.get("host_collection", {})
    if receipt.get("protocol") != sample["host_process_memory_method"]:
        raise ValueError("host method differs from collection protocol")
    if not _receipt_valid(receipt.get("stage_publish")):
        raise ValueError("missing final stage read receipt")
    derived = summarize_collection(receipt)
    if any(sample.get(key) != value for key, value in derived.items()):
        raise ValueError("host sample disagrees with retained read attempts")
    stages = [attempt["stage_before"].get("value") for attempt in sample["host_collection"]["attempts"]]
    if sample.get("stage") != stages[0]:
        raise ValueError("host sample stage mismatch")
    return derived


def validate_sample(sample: dict) -> dict:
    """Validate externally read JSON, converting malformed schema to ValueError."""
    try:
        return _validate_sample(sample)
    except (KeyError, TypeError, AttributeError, IndexError, OverflowError) as error:
        raise ValueError("malformed host collection receipt") from error


def finish_sample(sample: dict, stage_file: Path) -> None:
    """Recheck stage after device telemetry without concealing a stage race."""
    sample["host_collection"]["stage_publish"] = _read_stage(stage_file)
    sample.update(summarize_collection(sample["host_collection"]))
    validate_sample(sample)


class HostSampler:
    """Observe descendants; retain identity-bound orphaned members until exit.

    Discovery is a /proc snapshot, not a proof that no short-lived process fell
    between samples. RSS/status and PSS/rollup reads are not simultaneous.
    """

    def __init__(self, root_pid: int, stage_file: Path, *, exclude_pid: int | None = None):
        self.root_pid, self.stage_file = root_pid, stage_file
        self.exclude_pid = os.getpid() if exclude_pid is None else exclude_pid
        self.backend = _PidfdBackend()
        self.tracked: dict[int, dict] = {}

    def close(self):
        for item in self.tracked.values():
            if item["fd"] is not None:
                os.close(item["fd"])
        self.tracked.clear()

    def _discover(self) -> set[int]:
        parents, reads = {}, {}
        for path in Path("/proc").iterdir():
            if path.name.isdigit():
                read = read_proc(int(path.name), "stat")
                reads[str(int(path.name))] = read
                if "value" in read:
                    parents[int(path.name)] = read["value"]["ppid"]
        selected = set(self.tracked) | {self.root_pid}
        while True:
            more = {pid for pid, parent in parents.items() if parent in selected and pid != self.exclude_pid}
            if more <= selected:
                selected -= {self.exclude_pid}
                self.discovery = {str(pid): reads.get(str(pid)) or read_proc(pid, "stat") for pid in selected}
                return selected
            selected |= more

    def _pin(self, pid: int, discovery: dict):
        before = read_proc(pid, "stat")
        pin = {"discovery": discovery, "before": before, "flags": 0, "backend": self.backend.backend}
        fd = None
        try:
            if (
                "value" not in before
                or "value" not in discovery
                or not _valid_identity(before["value"], owner_is_identity=False)
                or not _valid_identity(discovery["value"], owner_is_identity=False)
                or _identity(before["value"], owner_is_identity=False)
                != _identity(discovery["value"], owner_is_identity=False)
            ):
                raise ProcessLookupError(errno.ESRCH, "identity unavailable before pin")
            fd = self.backend.open(pid)
            after = read_proc(pid, "stat")
            pin["after"] = after
            if (
                "value" not in after
                or not _valid_identity(after["value"], owner_is_identity=False)
                or _identity(before["value"], owner_is_identity=False)
                != _identity(after["value"], owner_is_identity=False)
            ):
                raise ProcessLookupError(errno.ESRCH, "identity changed while pinning")
            pin["identity"] = before["value"]
        except OSError as error:
            pin["error"] = _error(error)
            if fd is not None:
                os.close(fd)
                fd = None
        self.tracked[pid] = {"pin": pin, "fd": fd}

    def _process(self, pid: int) -> dict:
        item = self.tracked[pid]
        record = {"pin": item["pin"], "stat_before": read_proc(pid, "stat"), "poll_before": _poll(item["fd"])}
        record["status"] = read_proc(pid, "status")
        record["rollup"] = read_proc(pid, "smaps_rollup")
        record["stat_after"] = read_proc(pid, "stat")
        record["poll_after"] = _poll(item["fd"])
        return record

    def _wait_for_exit(self, pid: int, record: dict, deadline: float) -> None:
        if not _exit_wait_eligible(record):
            return
        remaining = deadline - time.monotonic() - EXIT_WAIT_RETRY_RESERVE_SECONDS
        # Reserve time for a new complete collection. Scheduler delay may still
        # exceed a requested poll timeout; only an in-budget retry is accepted.
        timeout = min(MAX_EXIT_WAIT_MILLISECONDS, int(remaining * 1000) - 1)
        if timeout <= 0:
            return
        initial = record["poll_after"]
        record["exit_wait"] = {
            "initial_poll": initial,
            "deadline_monotonic_seconds": deadline,
            "retry_reserve_seconds": EXIT_WAIT_RETRY_RESERVE_SECONDS,
        }
        record["poll_after"] = _poll(self.tracked[pid]["fd"], timeout_milliseconds=timeout)

    def collect(self) -> dict:
        started = time.monotonic()
        receipt = {
            "protocol": METHOD,
            "uid_semantics": "proc_inode_owner",
            "identity_semantics": IDENTITY_SEMANTICS,
            "root_pid": self.root_pid,
            "excluded_monitor_pid": self.exclude_pid,
            "attempts": [],
            "max_attempts": MAX_ATTEMPTS,
            "max_collection_seconds": MAX_COLLECTION_SECONDS,
        }
        for _ in range(MAX_ATTEMPTS):
            attempt_started = time.monotonic()
            if not receipt["attempts"]:
                started = attempt_started
            stage = _read_stage(self.stage_file)
            selected = self._discover()
            discovery_before = self.discovery
            for pid in selected - self.tracked.keys():
                self._pin(pid, discovery_before[str(pid)])
            records = {}
            for pid in sorted(selected):
                record = self._process(pid)
                if len(receipt["attempts"]) < MAX_ATTEMPTS - 1:
                    self._wait_for_exit(pid, record, started + MAX_COLLECTION_SECONDS)
                records[str(pid)] = record
            after = self._discover()
            discovery_after = self.discovery
            for pid in after - self.tracked.keys():
                self._pin(pid, discovery_after[str(pid)])
            final_polls = {str(pid): _poll(self.tracked[pid]["fd"]) for pid in sorted(selected)}
            attempt = {
                "begin_monotonic_seconds": attempt_started,
                "discovery_before": discovery_before,
                "discovery_after": discovery_after,
                "stage_before": stage,
                "stage_after": _read_stage(self.stage_file),
                "selected_before": sorted(selected),
                "selected_after": sorted(after),
                "processes": records,
                "post_discovery_polls": final_polls,
            }
            receipt["attempts"].append(attempt)
            receipt["final_attempt"] = len(receipt["attempts"]) - 1
            derived = summarize_collection(receipt)
            if (
                derived["host_pss_complete"]
                or derived["host_collection_errors"] != ["membership_transition_retry_exhausted"]
                or time.monotonic() - started >= MAX_COLLECTION_SECONDS
            ):
                break
        receipt["stage_publish"] = _read_stage(self.stage_file)
        derived = summarize_collection(receipt)
        record = {
            "host_process_memory_method": METHOD,
            "host_collection": receipt,
            "stage": receipt["attempts"][0]["stage_before"].get("value", "unreadable"),
        }
        record.update(derived)
        # Keep a released root as scope evidence; retire other members only after
        # an explicit released record, never merely on a failed /proc read.
        for pid in list(self.tracked):
            if pid != self.root_pid and derived["process_host_memory"].get(str(pid), {}).get("address_space_released"):
                fd = self.tracked.pop(pid)["fd"]
                if fd is not None:
                    os.close(fd)
        return record

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real Linux child lifetimes plus hostile receipt tests; no models or CUDA."""

import copy
import ctypes
import json
import select
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

from tools.native_qlora_qualification import monitor_host as host


class _CapabilityHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]


class _CapabilityWord(ctypes.Structure):
    _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32), ("inheritable", ctypes.c_uint32)]


def _capabilities(values=None):
    """Linux v3 capget/capset ABI, pid=0 means this sampling thread only."""
    libc = ctypes.CDLL(None, use_errno=True)
    header = _CapabilityHeader(0x20080522, 0)
    words = (_CapabilityWord * 2)()
    call = libc.capget if values is None else libc.capset
    call.argtypes = [ctypes.POINTER(_CapabilityHeader), ctypes.POINTER(_CapabilityWord)]
    call.restype = ctypes.c_int
    if values is not None:
        for index, value in enumerate(values):
            for key in ("effective", "permitted", "inheritable"):
                setattr(words[index], key, value[key])
    if call(ctypes.byref(header), words) != 0:
        code = ctypes.get_errno()
        raise OSError(code, "current-thread capability operation failed")
    return [{key: getattr(word, key) for key in ("effective", "permitted", "inheritable")} for word in words]


def _capability_status():
    return {
        key: value.strip()
        for line in Path("/proc/thread-self/status").read_text().splitlines()
        for key, _, value in (line.partition(":"),)
        if key.startswith("Cap")
    }


@contextmanager
def without_ptrace_capability(request):
    """Permission fixture only; restore the complete original structure even on failure.

    Containers may grant CAP_SYS_PTRACE, which overrides nondumpable protection.
    Clear only that effective bit, retaining permitted/inheritable/bounding/ambient
    capabilities. No training or monitor production policy uses this context.
    """
    initial = _capabilities()
    initial_status = _capability_status()
    expected = copy.deepcopy(initial)
    expected[0]["effective"] &= ~(1 << 19)  # Linux CAP_SYS_PTRACE.
    evidence = {"initial": initial, "initial_proc_status": initial_status, "drop_expected": expected}
    try:
        _capabilities(expected)
        evidence["dropped"] = _capabilities()
        evidence["dropped_proc_status"] = _capability_status()
        assert evidence["dropped"] == expected
        assert all(
            evidence["dropped_proc_status"][key] == initial_status[key] for key in initial_status if key != "CapEff"
        )
        yield evidence
    finally:
        try:
            _capabilities(initial)
            evidence["restored"] = _capabilities()
            evidence["restored_proc_status"] = _capability_status()
            assert evidence["restored"] == initial
            assert evidence["restored_proc_status"] == initial_status
        finally:
            request.node.user_properties.append(("ptrace_capability_fixture", json.dumps(evidence, sort_keys=True)))


@pytest.fixture
def child():
    children = []

    def start(*, nondumpable=False):
        code = "import ctypes,sys; data=bytearray(4*1024*1024); "
        if nondumpable:
            code += "libc=ctypes.CDLL(None); assert libc.prctl(4,0,0,0,0)==0; assert libc.prctl(3,0,0,0,0)==0; "
        code += "print('ready',flush=True); sys.stdin.read(1)"
        process = subprocess.Popen(
            [sys.executable, "-I", "-S", "-c", code], stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        children.append(process)
        assert process.stdout.readline() == b"ready\n"
        return process

    yield start
    for process in children:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5)
        process.stdin.close()
        process.stdout.close()


@pytest.fixture
def sampler(tmp_path, child):
    process = child()
    marker = tmp_path / "stage"
    marker.write_text("training")
    value = host.HostSampler(process.pid, marker)
    yield value, process
    value.close()


def test_real_live_child_receipt_and_descriptor_cleanup(sampler):
    value, process = sampler
    before = len(list(Path("/proc/self/fd").iterdir()))
    sample = value.collect()
    assert host.validate_sample(sample)["host_pss_complete"]
    assert sample["selected_host_pss_sum_bytes"] > 4 * 1024 * 1024
    assert set(sample["process_host_memory"]) == {str(process.pid)}
    item = sample["host_collection"]["attempts"][0]["processes"][str(process.pid)]
    assert item["pin"]["identity"]["start_ticks"] > 0
    assert item["stat_before"]["value"]["state"] in "RS"
    assert item["poll_before"]["events"] == []
    assert len(list(Path("/proc/self/fd").iterdir())) == before + 1
    value.close()
    assert len(list(Path("/proc/self/fd").iterdir())) == before


def test_real_exit_between_status_and_rollup_retries_and_preserves_failure(sampler, monkeypatch):
    value, process = sampler
    read = host.read_proc
    fired = False

    def exit_between(pid, name):
        nonlocal fired
        result = read(pid, name)
        if pid == process.pid and name == "status" and not fired:
            fired = True
            process.terminate()
            process.wait(timeout=5)
        return result

    monkeypatch.setattr(host, "read_proc", exit_between)
    sample = value.collect()
    attempts = sample["host_collection"]["attempts"]
    assert len(attempts) == 2
    first = attempts[0]["processes"][str(process.pid)]
    assert first["rollup"]["error"]["errno"] in (2, 3)
    assert first["poll_before"]["events"] == []
    assert first["poll_after"]["events"][0] & select.POLLIN
    assert host.validate_sample(sample)["host_pss_complete"]
    assert sample["selected_host_pss_sum_bytes"] == 0
    assert sample["process_host_memory"][str(process.pid)]["address_space_released"] is True


def test_actual_live_nondumpable_child_is_incomplete(child, tmp_path, request):
    process = child(nondumpable=True)
    value = host.HostSampler(process.pid, tmp_path / "stage")
    before = host.read_proc(process.pid, "smaps_rollup")
    try:
        with without_ptrace_capability(request) as evidence:
            evidence.update(child_pr_get_dumpable=0, before_read=before)
            sample = value.collect()
            item = sample["host_collection"]["attempts"][0]["processes"][str(process.pid)]
            evidence["dropped_read"] = item["rollup"]
            assert item["rollup"]["error"]["errno"] == 13
            assert item["poll_after"]["events"] == []
            assert not host.validate_sample(sample)["host_pss_complete"]
            assert "live_or_unknown_memory_unreadable" in sample["host_collection_errors"]
            assert len(sample["host_collection"]["attempts"]) == 1
        restored = host.read_proc(process.pid, "smaps_rollup")
        request.node.user_properties.append(("restored_read", json.dumps(restored, sort_keys=True)))
        assert ("error" in restored) == ("error" in before)
        if "error" in before:
            assert restored["error"]["errno"] == before["error"]["errno"]
    finally:
        value.close()


def test_pid_identity_mismatch_not_released_even_with_readable_pidfd(sampler):
    value, process = sampler
    sample = value.collect()
    receipt = sample["host_collection"]
    item = receipt["attempts"][0]["processes"][str(process.pid)]
    item["stat_after"]["value"]["start_ticks"] += 1
    item["poll_before"]["events"] = [select.POLLIN]
    item["poll_after"]["events"] = [select.POLLIN]
    result = host.summarize_collection(receipt)
    assert not result["host_pss_complete"]
    assert "process_identity_changed" in result["host_collection_errors"]
    assert "address_space_released" not in result["process_host_memory"][str(process.pid)]


def test_stage_race_during_collection_remains_incomplete(sampler, monkeypatch):
    value, process = sampler
    read = host.read_proc

    def change_stage(pid, name):
        result = read(pid, name)
        if pid == process.pid and name == "status":
            value.stage_file.write_text("portable_publication")
        return result

    monkeypatch.setattr(host, "read_proc", change_stage)
    sample = value.collect()
    assert sample["stage"] == "training"
    assert not host.validate_sample(sample)["host_pss_complete"]
    assert "stage_changed_or_unreadable" in sample["host_collection_errors"]
    assert len(sample["host_collection"]["attempts"]) == 1


def test_stage_race_during_device_sampling_remains_incomplete(sampler):
    value, _ = sampler
    sample = value.collect()
    assert sample["host_pss_complete"]
    value.stage_file.write_text("next_stage")
    host.finish_sample(sample, value.stage_file)
    assert sample["stage"] == "training"
    assert not host.validate_sample(sample)["host_pss_complete"]


def test_repeated_proven_transitions_exhaust_retry_without_hiding_attempts(sampler, monkeypatch):
    value, _ = sampler
    original = value._process

    def transitioning(pid):
        record = original(pid)
        record["poll_before"]["events"] = []
        record["poll_after"]["events"] = [select.POLLIN]
        return record

    monkeypatch.setattr(value, "_process", transitioning)
    poll = host._poll

    def final_released(fd):
        result = poll(fd)
        result["events"] = [select.POLLIN]
        return result

    monkeypatch.setattr(host, "_poll", final_released)
    sample = value.collect()
    assert len(sample["host_collection"]["attempts"]) == host.MAX_ATTEMPTS
    assert not host.validate_sample(sample)["host_pss_complete"]
    assert sample["host_collection_errors"] == ["membership_transition_retry_exhausted"]


def test_actual_exit_after_memory_before_final_discovery_retains_owner_change_and_peak(sampler, monkeypatch, request):
    value, process = sampler
    discover = value._discover
    calls = 0

    def exit_before_final_discovery():
        nonlocal calls
        calls += 1
        if calls == 2:
            process.terminate()
            poller = select.poll()
            poller.register(value.tracked[process.pid]["fd"], select.POLLIN)
            assert any(mask & select.POLLIN for _, mask in poller.poll(5000)), "child did not fully exit"
            # Do not reap: the real zombie's proc inode owner must be observed.
        return discover()

    monkeypatch.setattr(value, "_discover", exit_before_final_discovery)
    sample = value.collect()
    attempts = sample["host_collection"]["attempts"]
    assert len(attempts) == 2
    first, last = attempts
    key = str(process.pid)
    item = first["processes"][key]
    before = first["discovery_before"][key]["value"]
    after = first["discovery_after"][key]["value"]
    assert all(before[name] == after[name] for name in ("pid", "start_ticks", "ppid"))
    assert before["uid"] != after["uid"], "qualification requires an actual proc inode owner transition"
    assert after["state"] == "Z"
    assert item["poll_after"]["events"] == []
    assert first["post_discovery_polls"][key]["events"][0] & select.POLLIN
    assert item["rollup"]["value"]["Pss"] > 4 * 1024 * 1024
    assert host.validate_sample(sample)["host_pss_complete"]
    assert sample["selected_host_pss_sum_bytes"] == 0
    assert sample["host_observed_complete_attempt_pss_peak_bytes"] >= item["rollup"]["value"]["Pss"]
    assert last["processes"][key]["stat_before"]["value"]["uid"] == after["uid"]
    request.node.user_properties.append(
        (
            "actual_exit_inode_owner_transition",
            json.dumps(
                {
                    "before": before,
                    "after": after,
                    "pre_exit_pss_bytes": item["rollup"]["value"]["Pss"],
                    "post_discovery_events": first["post_discovery_polls"][key]["events"],
                    "retained_attempts": len(attempts),
                    "final_pss_bytes": sample["selected_host_pss_sum_bytes"],
                    "observed_attempt_peak_bytes": sample["host_observed_complete_attempt_pss_peak_bytes"],
                },
                sort_keys=True,
            ),
        )
    )


def test_actual_live_group_owner_change_preserves_valid_memory_without_exit_proof(tmp_path, monkeypatch, request):
    """The leader exits after a real read; a live thread still owns the shared mm.

    This makes proc ownership change on the qualified user-namespace target,
    without requiring an unmapped UID or granting memory-read capabilities.
    The existing unreadable-leader test separately requires incompleteness when
    the actual memory read occurs after the leader's exit instead.
    """
    compiler = shutil.which("cc")
    assert compiler, "real owner-transition qualification requires a C compiler"
    source = tmp_path / "owner_transition.c"
    source.write_text("""
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdatomic.h>
static atomic_int leader_leaving;
static void *worker(void *unused) {
  char *p = malloc(8 * 1024 * 1024);
  if (!p) _exit(11);
  for (int i = 0; i < 8 * 1024 * 1024; i += 4096) p[i] = 1;
  puts("ready"); fflush(stdout);
  while (!atomic_load(&leader_leaving)) usleep(1000);
  char c; if (read(0, &c, 1) != 1) _exit(12);
  free(p); return NULL;
}
int main(void) {
  pthread_t thread;
  if (pthread_create(&thread, NULL, worker, NULL)) return 13;
  char c; if (read(0, &c, 1) != 1) return 14;
  atomic_store(&leader_leaving, 1);
  pthread_exit(NULL);
}
""")
    binary = tmp_path / "owner_transition"
    subprocess.run([compiler, "-pthread", "-O0", str(source), "-o", str(binary)], check=True, timeout=15)
    process = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stage = tmp_path / "stage"
    stage.write_text("training")
    sampler = host.HostSampler(process.pid, stage)
    read = host.read_proc
    fired = False
    try:
        assert process.stdout.readline() == b"ready\n"

        def leave_leader_after_actual_memory(pid, name):
            nonlocal fired
            result = read(pid, name)
            if pid == process.pid and name == "smaps_rollup" and not fired:
                assert result["value"]["Pss"] > 4 * 1024 * 1024
                fired = True
                process.stdin.write(b"x")
                process.stdin.flush()
                deadline = time.monotonic() + 5
                while read(pid, "stat").get("value", {}).get("state") != "Z":
                    assert time.monotonic() < deadline, "leader did not exit while worker remained alive"
                    time.sleep(0.001)
            return result

        monkeypatch.setattr(host, "read_proc", leave_leader_after_actual_memory)
        sample = sampler.collect()
        key = str(process.pid)
        attempts = sample["host_collection"]["attempts"]
        assert len(attempts) == 1
        item = attempts[0]["processes"][key]
        before, after = item["stat_before"]["value"], item["stat_after"]["value"]
        assert before["uid"] != after["uid"], "qualification requires actual mutable proc inode ownership"
        assert all(before[name] == after[name] for name in ("pid", "start_ticks", "ppid"))
        assert after["state"] == "Z"
        assert item["poll_before"]["events"] == item["poll_after"]["events"] == []
        assert attempts[0]["post_discovery_polls"][key]["events"] == []
        assert host.validate_sample(sample)["host_pss_complete"]
        assert sample["selected_host_pss_sum_bytes"] == item["rollup"]["value"]["Pss"] > 4 * 1024 * 1024
        assert "address_space_released" not in sample["process_host_memory"][key]
        request.node.user_properties.append(
            (
                "actual_live_group_inode_owner_transition",
                json.dumps(
                    {
                        "before": before,
                        "after": after,
                        "memory_read": item["rollup"],
                        "poll_before": item["poll_before"],
                        "poll_after": item["poll_after"],
                        "post_discovery_poll": attempts[0]["post_discovery_polls"][key],
                        "final_pss_bytes": sample["selected_host_pss_sum_bytes"],
                        "address_space_released": False,
                        "complete": sample["host_pss_complete"],
                    },
                    sort_keys=True,
                ),
            )
        )
        process.stdin.write(b"y")
        process.stdin.flush()
        process.wait(timeout=5)
        assert process.returncode == 0
    finally:
        sampler.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stdin.close()
        process.stdout.close()


@pytest.mark.parametrize("state", ["R", "S", "Z"])
def test_v3_owner_only_is_metadata_and_never_zero_memory(sampler, state):
    value, process = sampler
    sample = value.collect()
    receipt = sample["host_collection"]
    attempt = receipt["attempts"][0]
    key = str(process.pid)
    for read in (attempt["processes"][key]["stat_after"], attempt["discovery_after"][key]):
        read["value"].update(uid=read["value"]["uid"] + 1, state=state)
    sample.update(host.summarize_collection(receipt))
    assert host.validate_sample(sample)["host_pss_complete"]
    assert sample["selected_host_pss_sum_bytes"] > 4 * 1024 * 1024
    assert "address_space_released" not in sample["process_host_memory"][key]


@pytest.mark.parametrize("field", ["pid", "start_ticks"])
def test_v3_owner_change_cannot_conceal_bound_identity_change(sampler, field):
    value, process = sampler
    sample = value.collect()
    attempt = sample["host_collection"]["attempts"][0]
    key = str(process.pid)
    for read in (attempt["processes"][key]["stat_after"], attempt["discovery_after"][key]):
        read["value"]["uid"] += 1
        read["value"][field] += 1
    result = host.summarize_collection(sample["host_collection"])
    assert not result["host_pss_complete"]
    assert "process_identity_changed" in result["host_collection_errors"]
    assert "discovery_identity_changed" in result["host_collection_errors"]


def test_actual_pinned_descendant_remains_owned_after_reparenting(tmp_path, request):
    compiler = shutil.which("cc")
    assert compiler, "real owned-reparent qualification requires a C compiler"
    source = tmp_path / "reparent.c"
    source.write_text("""
#include <sys/prctl.h>
#include <sys/wait.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
int main(void) {
  int parent_go[2], child_go[2];
  if (prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) || pipe(parent_go) || pipe(child_go)) return 11;
  pid_t parent = fork(); if (parent < 0) return 12;
  if (!parent) {
    pid_t child = fork(); if (child < 0) _exit(13);
    if (!child) {
      char *p = malloc(4 * 1024 * 1024); if (!p) _exit(14);
      for (int i = 0; i < 4 * 1024 * 1024; i += 4096) p[i] = 1;
      printf("child %d\\n", getpid()); fflush(stdout);
      char c; if (read(child_go[0], &c, 1) != 1) _exit(15);
      free(p); _exit(0);
    }
    char c; if (read(parent_go[0], &c, 1) != 1) _exit(16);
    _exit(0);
  }
  printf("parent %d\\n", parent); fflush(stdout);
  char c; int status;
  if (read(0, &c, 1) != 1 || write(parent_go[1], "x", 1) != 1) return 17;
  if (waitpid(parent, &status, 0) != parent || status != 0) return 18;
  puts("reparented"); fflush(stdout);
  if (read(0, &c, 1) != 1 || write(child_go[1], "x", 1) != 1) return 19;
  if (waitpid(-1, &status, 0) <= 0 || status != 0) return 20;
  return 0;
}
""")
    binary = tmp_path / "reparent"
    subprocess.run([compiler, "-O0", str(source), "-o", str(binary)], check=True, timeout=15)
    process = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stage = tmp_path / "stage"
    stage.write_text("training")
    sampler = host.HostSampler(process.pid, stage)
    try:
        members = dict(process.stdout.readline().decode().split() for _ in range(2))
        parent, child = int(members["parent"]), int(members["child"])
        before = sampler.collect()
        assert host.validate_sample(before)["host_pss_complete"]
        original = before["host_collection"]["attempts"][0]["processes"][str(child)]["pin"]["identity"]
        assert original["ppid"] == parent
        process.stdin.write(b"p")
        process.stdin.flush()
        assert process.stdout.readline() == b"reparented\n"
        after = sampler.collect()
        item = after["host_collection"]["attempts"][0]["processes"][str(child)]
        assert item["pin"]["identity"] == original
        assert item["stat_before"]["value"]["ppid"] == process.pid != parent
        assert item["poll_before"]["events"] == item["poll_after"]["events"] == []
        assert host.validate_sample(after)["host_pss_complete"]
        assert after["process_host_memory"][str(child)]["Pss_rollup"] > 3 * 1024 * 1024
        request.node.user_properties.append(
            (
                "actual_owned_descendant_reparenting",
                json.dumps(
                    {"pin": original, "observed": item["stat_before"], "complete": after["host_pss_complete"]},
                    sort_keys=True,
                ),
            )
        )
        process.stdin.write(b"c")
        process.stdin.flush()
        process.wait(timeout=5)
        assert process.returncode == 0
    finally:
        # Release both private fixture gates even if a receipt assertion fails;
        # the isolated subreaper waits for its own child, leaving no orphan.
        if process.poll() is None:
            try:
                process.stdin.write(b"pc")
                process.stdin.flush()
                process.wait(timeout=5)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                for item in sampler.tracked.values():
                    if item["fd"] is not None:
                        try:
                            sampler.backend.send_signal(item["fd"], 9)
                        except ProcessLookupError:
                            pass
                process.wait(timeout=5)
        sampler.close()
        process.stdin.close()
        process.stdout.close()


def test_unrelated_new_member_cannot_use_owner_metadata_to_gain_admission(sampler):
    value, process = sampler
    sample = value.collect()
    receipt = sample["host_collection"]
    attempt = receipt["attempts"][0]
    stranger = process.pid + 10000000
    attempt["selected_after"].append(stranger)
    read = copy.deepcopy(attempt["discovery_after"][str(process.pid)])
    read["value"].update(pid=stranger, ppid=stranger, uid=read["value"]["uid"] + 1)
    attempt["discovery_after"][str(stranger)] = read
    result = host.summarize_collection(receipt)
    assert not result["host_pss_complete"]
    assert "unproven_new_descendant" in result["host_collection_errors"]


@pytest.mark.parametrize("method", [host.LEGACY_METHOD, host.EXIT_OWNER_METHOD])
def test_v3_receipt_cannot_be_relabelled_as_prior_method(sampler, method):
    value, _ = sampler
    sample = value.collect()
    sample["host_process_memory_method"] = sample["host_collection"]["protocol"] = method
    with pytest.raises(ValueError, match="v3 identity semantics"):
        host.validate_sample(sample)


def test_v2_owner_only_change_retains_original_incomplete_classification(sampler):
    value, process = sampler
    sample = value.collect()
    receipt = sample["host_collection"]
    sample["host_process_memory_method"] = receipt["protocol"] = host.EXIT_OWNER_METHOD
    del receipt["identity_semantics"]
    receipt["attempts"][0]["discovery_after"][str(process.pid)]["value"]["uid"] += 1
    sample.update(host.summarize_collection(receipt))
    assert not host.validate_sample(sample)["host_pss_complete"]
    assert sample["host_collection_errors"] == ["discovery_identity_changed"]


@pytest.mark.parametrize("damage", ["live_owner", "birth", "parent", "no_exit", "poll_error"])
def test_final_discovery_owner_exception_is_strictly_exit_bound(sampler, damage):
    value, process = sampler
    sample = value.collect()
    # This is the immutable v2 classifier, including its owner-as-identity rule.
    sample["host_process_memory_method"] = sample["host_collection"]["protocol"] = host.EXIT_OWNER_METHOD
    del sample["host_collection"]["identity_semantics"]
    attempt = sample["host_collection"]["attempts"][0]
    key = str(process.pid)
    changed = attempt["discovery_after"][key]["value"]
    changed["uid"] += 1
    changed["state"] = "R" if damage == "live_owner" else "Z"
    if damage == "birth":
        changed["start_ticks"] += 1
    if damage == "parent":
        changed["ppid"] += 1
    attempt["post_discovery_polls"][key]["events"] = (
        [] if damage == "no_exit" else [select.POLLERR] if damage == "poll_error" else [select.POLLIN]
    )
    result = host.summarize_collection(sample["host_collection"])
    assert not result["host_pss_complete"]
    assert "discovery_identity_changed" in result["host_collection_errors"]


@pytest.mark.parametrize("damage", ["missing", "order", "downgrade", "semantics", "scope"])
def test_final_discovery_proof_receipt_cannot_be_omitted_or_relabeled(sampler, damage):
    value, process = sampler
    sample = value.collect()
    receipt = sample["host_collection"]
    attempt = receipt["attempts"][0]
    key = str(process.pid)
    if damage == "missing":
        del attempt["post_discovery_polls"]
    elif damage == "order":
        attempt["post_discovery_polls"][key]["begin_monotonic_seconds"] = 0
    elif damage == "downgrade":
        sample["host_process_memory_method"] = receipt["protocol"] = host.LEGACY_METHOD
    elif damage == "semantics":
        receipt["uid_semantics"] = "credential_uid"
    else:
        attempt["post_discovery_polls"][str(process.pid + 10000000)] = copy.deepcopy(
            attempt["post_discovery_polls"][key]
        )
    with pytest.raises(ValueError):
        host.validate_sample(sample)


def test_legacy_v1_owner_mismatch_remains_incomplete_without_new_proof(sampler):
    from tools.native_qlora_qualification.performance import sampled_memory

    value, process = sampler
    sample = value.collect()
    receipt = sample["host_collection"]
    sample["host_process_memory_method"] = receipt["protocol"] = host.LEGACY_METHOD
    del receipt["uid_semantics"]
    del receipt["identity_semantics"]
    for attempt in receipt["attempts"]:
        del attempt["post_discovery_polls"]
    changed = receipt["attempts"][0]["discovery_after"][str(process.pid)]["value"]
    changed["uid"] += 1
    changed["state"] = "Z"
    sample.update(host.summarize_collection(receipt))
    assert host.validate_sample(sample)["host_pss_complete"] is False
    assert sample["host_collection_errors"] == ["discovery_identity_changed"]
    sample.update(
        unix_seconds=time.time(),
        device_uuid="gpu",
        device_memory_total_bytes=100,
        device_memory_used_bytes=10,
        device_compute_processes=[],
    )
    assert sampled_memory([sample], expected_uuid="gpu", device_bytes=100)["host_pss_complete"] is False


@pytest.mark.parametrize("mutation", ["complete", "total", "omit", "pid", "poll_error", "retry", "stage", "timestamp"])
def test_receipt_validator_refuses_tampered_totals_or_process_proof(sampler, mutation):
    value, process = sampler
    sample = value.collect()
    attempt = sample["host_collection"]["attempts"][0]
    item = attempt["processes"][str(process.pid)]
    if mutation == "complete":
        sample["host_pss_complete"] = False
    elif mutation == "total":
        sample["selected_host_pss_sum_bytes"] += 1
    elif mutation == "omit":
        attempt["processes"].clear()
    elif mutation == "pid":
        item["pin"]["identity"]["pid"] += 1
    elif mutation == "poll_error":
        item["poll_after"]["events"] = [select.POLLERR]
    elif mutation == "retry":
        sample["host_collection"]["attempts"].append(copy.deepcopy(attempt))
        sample["host_collection"]["final_attempt"] = 1
    elif mutation == "stage":
        sample["stage"] = "other"
    else:
        item["status"]["end_monotonic_seconds"] = 0
    with pytest.raises(ValueError):
        host.validate_sample(sample)


def test_unknown_missing_process_stays_selected_and_incomplete(tmp_path):
    # Reap a real PID first: no pin may be fabricated for the absent process.
    process = subprocess.Popen([sys.executable, "-I", "-S", "-c", "pass"])
    process.wait(timeout=5)
    value = host.HostSampler(process.pid, tmp_path / "stage")
    try:
        sample = value.collect()
        assert set(sample["process_host_memory"]) == {str(process.pid)}
        assert not host.validate_sample(sample)["host_pss_complete"]
        assert "unbound_process_identity" in sample["host_collection_errors"]
    finally:
        value.close()


def test_thread_group_leader_zombie_with_live_thread_is_not_released(tmp_path):
    compiler = shutil.which("cc")
    assert compiler, "allocated Linux validation requires a C compiler for the actual pthread-exit case"
    source = tmp_path / "leader_exit.c"
    source.write_text("""#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
static void *worker(void *unused) {
  char *p = malloc(8 * 1024 * 1024);
  for (int i = 0; i < 8 * 1024 * 1024; i += 4096) p[i] = 1;
  puts("ready"); fflush(stdout);
  char c; read(0, &c, 1); free(p); return NULL;
}
int main(void) { pthread_t thread; pthread_create(&thread, NULL, worker, NULL); pthread_exit(NULL); }
""")
    binary = tmp_path / "leader_exit"
    subprocess.run([compiler, "-pthread", "-O0", str(source), "-o", str(binary)], check=True, timeout=15)
    process = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    value = host.HostSampler(process.pid, tmp_path / "stage")
    try:
        assert process.stdout.readline() == b"ready\n"
        deadline = time.monotonic() + 5
        while host.read_proc(process.pid, "stat").get("value", {}).get("state") != "Z":
            assert time.monotonic() < deadline
            time.sleep(0.01)
        sample = value.collect()
        item = sample["host_collection"]["attempts"][0]["processes"][str(process.pid)]
        assert item["stat_before"]["value"]["state"] == "Z"
        assert item["poll_after"]["events"] == []
        assert not host.validate_sample(sample)["host_pss_complete"]
        assert "address_space_released" not in sample["process_host_memory"][str(process.pid)]
        process.stdin.write(b"x")
        process.stdin.flush()
        process.wait(timeout=5)
        ended = value.collect()
        assert host.validate_sample(ended)["host_pss_complete"]
        assert ended["process_host_memory"][str(process.pid)]["address_space_released"]
    finally:
        value.close()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=5)
        process.stdin.close()
        process.stdout.close()


def test_new_receipt_performance_validation_preserves_legacy_semantics(sampler):
    from tools.native_qlora_qualification.performance import sampled_memory

    value, _ = sampler
    sample = value.collect()
    sample.update(
        unix_seconds=time.time(),
        device_uuid="gpu",
        device_memory_total_bytes=100,
        device_memory_used_bytes=10,
        device_compute_processes=[],
    )
    assert sampled_memory([sample], expected_uuid="gpu", device_bytes=100)["host_pss_complete"]
    corrupt = copy.deepcopy(sample)
    corrupt["selected_host_pss_sum_bytes"] += 1
    result = sampled_memory([corrupt], expected_uuid="gpu", device_bytes=100)
    assert not result["host_pss_complete"]
    assert "invalid_host_collection_receipt" in result["errors"]
    legacy = {
        key: value for key, value in sample.items() if key not in ("host_collection", "host_process_memory_method")
    }
    legacy["host_pss_complete"] = False
    assert not sampled_memory([legacy], expected_uuid="gpu", device_bytes=100)["host_pss_complete"]


def test_discovery_to_pin_identity_reuse_fails_closed(sampler, monkeypatch):
    value, process = sampler
    original = value._discover

    def changed_discovery():
        selected = original()
        value.discovery[str(process.pid)]["value"]["start_ticks"] += 1
        return selected

    monkeypatch.setattr(value, "_discover", changed_discovery)
    sample = value.collect()
    assert not host.validate_sample(sample)["host_pss_complete"]
    assert "unbound_process_identity" in sample["host_collection_errors"]
    assert len(sample["host_collection"]["attempts"]) == 1


def test_actual_live_eacces_followed_by_exit_cannot_be_hidden_by_retry(child, tmp_path, monkeypatch, request):
    process = child(nondumpable=True)
    value = host.HostSampler(process.pid, tmp_path / "stage")
    read = host.read_proc
    before = read(process.pid, "smaps_rollup")
    try:
        with without_ptrace_capability(request) as evidence:
            evidence.update(child_pr_get_dumpable=0, before_read=before)

            def error_then_exit(pid, name):
                result = read(pid, name)
                if pid == process.pid and name == "smaps_rollup":
                    evidence["dropped_read"] = result
                    assert result["error"]["errno"] == 13
                    process.terminate()
                    process.wait(timeout=5)
                return result

            monkeypatch.setattr(host, "read_proc", error_then_exit)
            sample = value.collect()
            evidence["after_exit_read"] = sample["host_collection"]["attempts"][0]["processes"][str(process.pid)][
                "stat_after"
            ]
            assert not host.validate_sample(sample)["host_pss_complete"]
            assert len(sample["host_collection"]["attempts"]) == 1
            assert "unexplained_read_error_before_exit" in sample["host_collection_errors"]
    finally:
        value.close()


def test_capability_fixture_restores_exact_state_after_assertion_failure(request):
    original = _capabilities()
    original_status = _capability_status()
    with pytest.raises(AssertionError, match="deliberate fixture body failure"):
        with without_ptrace_capability(request):
            assert False, "deliberate fixture body failure"
    assert _capabilities() == original
    assert _capability_status() == original_status


def test_valid_pre_exit_attempt_peak_is_not_lost_when_final_retry_is_zero(sampler, monkeypatch):
    from tools.native_qlora_qualification.performance import sampled_memory

    value, process = sampler
    read = host.read_proc
    fired = False

    def read_then_exit(pid, name):
        nonlocal fired
        result = read(pid, name)
        if pid == process.pid and name == "smaps_rollup" and not fired:
            assert result["value"]["Pss"] > 4 * 1024 * 1024
            fired = True
            process.terminate()
            process.wait(timeout=5)
        return result

    monkeypatch.setattr(host, "read_proc", read_then_exit)
    sample = value.collect()
    assert host.validate_sample(sample)["host_pss_complete"]
    assert sample["selected_host_pss_sum_bytes"] == 0
    assert sample["host_observed_complete_attempt_pss_peak_bytes"] > 4 * 1024 * 1024
    sample.update(
        unix_seconds=time.time(),
        device_uuid="gpu",
        device_memory_total_bytes=100,
        device_memory_used_bytes=10,
        device_compute_processes=[],
    )
    result = sampled_memory([sample], expected_uuid="gpu", device_bytes=100)
    assert result["host_pss_sum_peak_bytes"] == sample["host_observed_complete_attempt_pss_peak_bytes"]


@pytest.mark.parametrize("field", ["stage_publish", "discovery_after", "rollup"])
def test_missing_receipt_parts_fail_closed(sampler, field):
    value, process = sampler
    sample = value.collect()
    receipt = sample["host_collection"]
    if field == "stage_publish":
        del receipt[field]
    elif field == "discovery_after":
        receipt["attempts"][0][field][str(process.pid)]["value"]["start_ticks"] += 1
    else:
        receipt["attempts"][0]["processes"][str(process.pid)][field] = None
    with pytest.raises(ValueError):
        host.validate_sample(sample)


def test_new_membership_cycle_is_not_a_proven_descendant(sampler):
    value, process = sampler
    sample = value.collect()
    receipt = sample["host_collection"]
    attempt = receipt["attempts"][0]
    unexpected = max(attempt["selected_after"]) + 10000000
    attempt["selected_after"].append(unexpected)
    read = copy.deepcopy(attempt["discovery_after"][str(process.pid)])
    read["value"].update(pid=unexpected, ppid=unexpected)
    attempt["discovery_after"][str(unexpected)] = read
    result = host.summarize_collection(receipt)
    assert not result["host_pss_complete"]
    assert "unproven_new_descendant" in result["host_collection_errors"]


def test_retry_start_after_bound_is_rejected(sampler):
    value, _ = sampler
    sample = value.collect()
    receipt = sample["host_collection"]
    attempt = copy.deepcopy(receipt["attempts"][0])
    attempt["begin_monotonic_seconds"] += host.MAX_COLLECTION_SECONDS + 1
    receipt["attempts"].append(attempt)
    receipt["final_attempt"] = 1
    # Check the bound independently; prior transition proof is also required.
    for record in receipt["attempts"][0]["processes"].values():
        record["poll_after"]["events"] = [select.POLLIN]
    for poll in receipt["attempts"][0]["post_discovery_polls"].values():
        poll["events"] = [select.POLLIN]
    with pytest.raises(ValueError, match="bounded collection interval"):
        host.summarize_collection(receipt)

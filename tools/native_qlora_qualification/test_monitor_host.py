# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: OpenMDW-1.1
"""Real Linux child lifetimes plus hostile receipt tests; no models or CUDA."""

import copy
import select
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tools.native_qlora_qualification import monitor_host as host


@pytest.fixture
def child():
    children = []

    def start(*, nondumpable=False):
        code = "import ctypes,sys; data=bytearray(4*1024*1024); "
        if nondumpable:
            code += "assert ctypes.CDLL(None).prctl(4,0,0,0,0)==0; "
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


def test_actual_live_nondumpable_child_is_incomplete(child, tmp_path):
    process = child(nondumpable=True)
    value = host.HostSampler(process.pid, tmp_path / "stage")
    try:
        sample = value.collect()
        item = sample["host_collection"]["attempts"][0]["processes"][str(process.pid)]
        assert item["rollup"]["error"]["errno"] == 13
        assert item["poll_after"]["events"] == []
        assert not host.validate_sample(sample)["host_pss_complete"]
        assert "live_or_unknown_memory_unreadable" in sample["host_collection_errors"]
        assert len(sample["host_collection"]["attempts"]) == 1
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
    sample = value.collect()
    assert len(sample["host_collection"]["attempts"]) == host.MAX_ATTEMPTS
    assert not host.validate_sample(sample)["host_pss_complete"]
    assert sample["host_collection_errors"] == ["membership_transition_retry_exhausted"]


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


def test_actual_live_eacces_followed_by_exit_cannot_be_hidden_by_retry(child, tmp_path, monkeypatch):
    process = child(nondumpable=True)
    value = host.HostSampler(process.pid, tmp_path / "stage")
    read = host.read_proc

    def error_then_exit(pid, name):
        result = read(pid, name)
        if pid == process.pid and name == "smaps_rollup":
            assert result["error"]["errno"] == 13
            process.terminate()
            process.wait(timeout=5)
        return result

    monkeypatch.setattr(host, "read_proc", error_then_exit)
    try:
        sample = value.collect()
        assert not host.validate_sample(sample)["host_pss_complete"]
        assert len(sample["host_collection"]["attempts"]) == 1
        assert "unexplained_read_error_before_exit" in sample["host_collection_errors"]
    finally:
        value.close()


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
    with pytest.raises(ValueError, match="bounded collection interval"):
        host.summarize_collection(receipt)

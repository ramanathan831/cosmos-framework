"""Bounded full media reads preserve source identity and fail closed."""

import json
import random
import threading

import pytest
import torch

import cosmos_framework.posttrain.data.datasets.sft as source
from cosmos_framework.posttrain.config.sft import SFTDatasetConfig
from cosmos_framework.posttrain.roles.env.sft import SFTDatasetEnv


def make_data(tmp_path):
    files = []
    for i in range(8):
        path = tmp_path / f"{i}.bin"
        path.write_bytes(bytes([i]) * 100)
        files.append(path)
    rows = [
        dict(
            id=str(i),
            messages=[
                dict(role="user", content=[dict(type="video", video=str(files[i % 8]))]),
                dict(role="assistant", content="answer"),
            ],
        )
        for i in range(16)
    ]
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path, files


def test_bounded_concurrent_full_reads_exact_identity_and_next_inputs(tmp_path, monkeypatch):
    path, files = make_data(tmp_path)
    control = source.JSONLSFTDataset(str(path))
    original = source.sha256_file
    gate = threading.Barrier(4)
    lock = threading.Lock()
    active = peak = 0
    reads = []

    def read(name):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
            reads.append(name)
        try:
            gate.wait(timeout=10)
            return original(name)
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(source, "sha256_file", read)
    rng = random.getstate(), torch.get_rng_state().clone()
    candidate = source.build_sft_dataset(SFTDatasetConfig(path=str(path), media_hash_workers=4))
    assert active == 0 and peak == 4 and sorted(reads) == sorted(map(str, files))
    assert control.rows == candidate.rows
    assert list(control.media_hashes.items()) == list(candidate.media_hashes.items())
    assert control.cursor_fingerprint() == candidate.cursor_fingerprint()
    assert random.getstate() == rng[0] and torch.equal(torch.get_rng_state(), rng[1])
    a, b = SFTDatasetEnv(control, seed=43), SFTDatasetEnv(candidate, seed=43)
    a.get_samples(7)
    b.load_state_dict(a.state_dict())
    assert a.get_samples(19) == b.get_samples(19)


def test_same_size_mutation_during_hash_is_rejected(tmp_path, monkeypatch):
    path, files = make_data(tmp_path)
    original = source.sha256_file

    def mutate(name):
        result = original(name)
        if name == str(files[0]):
            files[0].write_bytes(b"changed!!!" * 10)
        return result

    monkeypatch.setattr(source, "sha256_file", mutate)
    obj = source.JSONLSFTDataset.__new__(source.JSONLSFTDataset)
    with pytest.raises(ValueError, match="media changed"):
        obj.__init__(str(path), media_hash_workers=4)
    assert not hasattr(obj, "_fingerprint")


def test_worker_failure_cannot_publish_partial_fingerprint(tmp_path, monkeypatch):
    path, files = make_data(tmp_path)
    original = source.sha256_file

    def fail(name):
        if name == str(files[2]):
            raise OSError("injected read failure")
        return original(name)

    monkeypatch.setattr(source, "sha256_file", fail)
    obj = source.JSONLSFTDataset.__new__(source.JSONLSFTDataset)
    with pytest.raises(OSError, match="injected"):
        obj.__init__(str(path), media_hash_workers=4)
    assert not hasattr(obj, "_fingerprint")


@pytest.mark.parametrize("workers", [0, 5, True, 1.5, "4"])
def test_invalid_workers_fail_before_file_access(workers):
    with pytest.raises(ValueError, match="media_hash_workers"):
        source.JSONLSFTDataset("/missing", media_hash_workers=workers)
    with pytest.raises(ValueError, match="media_hash_workers"):
        SFTDatasetConfig(media_hash_workers=workers)


def test_factory_option_is_not_silently_ignored():
    with pytest.raises(ValueError, match="factories"):
        SFTDatasetConfig(factory="some.factory", media_hash_workers=4)

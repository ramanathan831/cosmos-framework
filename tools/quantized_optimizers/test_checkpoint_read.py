"""An overlapped verification must join, retain exact bytes and propagate failures."""

import hashlib
import threading

import pytest

from tools.quantized_optimizers import score_dense


@pytest.mark.parametrize("overlap", [False, True])
def test_restore_returns_only_after_complete_hash(tmp_path, overlap):
    path = tmp_path / "model.pt"
    payload = b"immutable model bytes" * 1024
    path.write_bytes(payload)
    restored = object()
    value, receipt, phases = score_dense.restore_with_checkpoint_read(path, lambda: restored, overlap=overlap)
    assert value is restored
    assert receipt["sha256"] == hashlib.sha256(payload).hexdigest()
    assert phases["checkpoint_read_and_restore"] >= phases["checkpoint_read"]


def test_overlap_joins_read_even_when_restore_fails(tmp_path, monkeypatch):
    started, release, finished = threading.Event(), threading.Event(), threading.Event()

    def read(path):
        started.set()
        assert release.wait(2)
        finished.set()
        return {"seconds": 0}

    def restore():
        assert started.wait(2)
        release.set()
        raise RuntimeError("restore failed")

    monkeypatch.setattr(score_dense, "read_checkpoint_file", read)
    with pytest.raises(RuntimeError, match="restore failed"):
        score_dense.restore_with_checkpoint_read(tmp_path, restore, overlap=True)
    assert finished.is_set()


@pytest.mark.parametrize("overlap", [False, True])
def test_read_failure_cannot_publish_success(tmp_path, monkeypatch, overlap):
    def fail(path):
        raise OSError("read failed")

    monkeypatch.setattr(score_dense, "read_checkpoint_file", fail)
    with pytest.raises(OSError, match="read failed"):
        score_dense.restore_with_checkpoint_read(tmp_path, lambda: object(), overlap=overlap)


def test_checkpoint_mutation_during_hash_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "model.pt"
    path.write_bytes(b"model bytes")
    digest = hashlib.sha256()

    class MutatingDigest:
        def update(self, value):
            digest.update(value)
            path.write_bytes(b"replacement model bytes")

        def hexdigest(self):
            return digest.hexdigest()

    monkeypatch.setattr(score_dense.hashlib, "sha256", MutatingDigest)
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        score_dense.read_checkpoint_file(path)

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


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("failure", [None, "checksum", "provenance"])
def test_scoring_receipt_requires_hash_and_provenance(tmp_path, overlap, failure):
    import json

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    payload = b"complete immutable model"
    (checkpoint / "model.pt").write_bytes(payload)
    output = tmp_path / "scores"
    output.mkdir()
    digest = hashlib.sha256(payload).hexdigest()

    class Manager:
        def load_checkpoint(self, source, load):
            value = load()
            assert not (output / "checkpoint-read.json").exists()
            if failure == "provenance":
                raise RuntimeError("checkpoint provenance changed")
            return value, {"signature": "stable"}

    def restore():
        assert not list(output.iterdir())
        return "restored-model"

    if failure:
        with pytest.raises(RuntimeError, match="SHA-256|provenance"):
            score_dense.restore_scoring_checkpoint(
                Manager(),
                checkpoint,
                restore,
                output,
                overlap=overlap,
                expected_sha256="0" * 64 if failure == "checksum" else digest,
            )
        assert not list(output.iterdir())
    else:
        value, provenance, _ = score_dense.restore_scoring_checkpoint(
            Manager(), checkpoint, restore, output, overlap=overlap, expected_sha256=digest
        )
        assert value == "restored-model" and provenance == {"signature": "stable"}
        assert json.loads((output / "checkpoint-read.json").read_text())["sha256"] == digest


@pytest.mark.parametrize("digest", ["", "g" * 64, "A" * 64])
def test_invalid_pinned_digest_fails_before_restoration(tmp_path, digest):
    def restore():
        pytest.fail("invalid expected identity reached model restoration")

    with pytest.raises(ValueError, match="SHA-256"):
        score_dense.restore_with_checkpoint_read(tmp_path / "absent", restore, overlap=True, expected_sha256=digest)

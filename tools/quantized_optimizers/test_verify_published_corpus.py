import hashlib
import json

import pytest

from tools.quantized_optimizers.verify_published_corpus import verify


@pytest.mark.parametrize("corrupt", [None, "candidate", "reference", "media", "manifest", "parse_read"])
def test_complete_hashes_are_required(tmp_path, monkeypatch, corrupt):
    candidate, reference = tmp_path / "candidate", tmp_path / "reference"
    media = tmp_path / "video.mp4"
    media.write_bytes(b"source video")
    key, payload = "0" * 64, b"serialized inputs and masks"
    manifest = {
        "entries": {key: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}},
        "media": {str(media): hashlib.sha256(media.read_bytes()).hexdigest()},
    }
    value = json.dumps(manifest).encode()
    for root in (candidate, reference):
        root.mkdir()
        (root / "manifest.json").write_bytes(value)
        (root / f"{key}.pt").write_bytes(payload)
    if corrupt in {"candidate", "reference"}:
        (tmp_path / corrupt / f"{key}.pt").write_bytes(b"x" * len(payload))
    elif corrupt == "media":
        media.write_bytes(b"changed video")
    elif corrupt == "manifest":
        (candidate / "manifest.json").write_bytes(value + b" ")
    elif corrupt == "parse_read":
        # Emulate different contents on the parse read without a stat change.
        # A second read must not replace the externally bound payload mapping.
        from pathlib import Path

        changed = json.dumps({**manifest, "entries": {}}).encode()
        monkeypatch.setattr(Path, "read_bytes", lambda self: changed)
        monkeypatch.setattr(Path, "read_text", lambda self: changed.decode())
    if corrupt:
        with pytest.raises(ValueError, match="mismatch|identity"):
            verify(candidate, reference, hashlib.sha256(value).hexdigest())
    else:
        result = verify(candidate, reference, hashlib.sha256(value).hexdigest())
        assert result["complete"]
        assert result["verified_files_including_manifests"] == 5


@pytest.mark.parametrize("damage", [None, "selection", "processor", "entry", "media", "mutation"])
def test_subset_byte_proof_is_exact_and_fail_closed(tmp_path, monkeypatch, damage):
    from tools.quantized_optimizers import verify_published_corpus as module

    candidate, reference = tmp_path / "candidate", tmp_path / "reference"
    candidate.mkdir()
    reference.mkdir()
    media = tmp_path / "source.mp4"
    media.write_bytes(b"unchanged source")
    key, other = "a" * 64, "b" * 64
    payload = b"exact serialized tensor bytes"
    entry = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "supervised_tokens": 3}
    value = {
        "format": "native",
        "packer": {"identity": 1},
        "sources": ["subset"],
        "entries": {key: entry},
        "media": {str(media): hashlib.sha256(media.read_bytes()).hexdigest()},
    }
    oracle = {**value, "sources": ["whole"], "entries": {key: entry, other: entry}}
    for root, manifest in ((candidate, value), (reference, oracle)):
        (root / (key + ".pt")).write_bytes(payload)
        (root / "manifest.json").write_text(json.dumps(manifest))
    expected = hashlib.sha256((reference / "manifest.json").read_bytes()).hexdigest()
    selected = [key]
    if damage == "selection":
        selected.append(other)
    if damage == "processor":
        value["packer"] = {"identity": 2}
    if damage == "entry":
        value["entries"] = {key: {**entry, "supervised_tokens": 2}}
    if damage == "media":
        value["media"] = {str(media): "0" * 64}
    (candidate / "manifest.json").write_text(json.dumps(value))
    pinned = hashlib.sha256((candidate / "manifest.json").read_bytes()).hexdigest()
    if damage == "mutation":
        original = module.checked_hash

        def mutate(path):
            result = original(path)
            if path == candidate / (key + ".pt"):
                path.write_bytes(b"x" * len(payload))
            return result

        monkeypatch.setattr(module, "checked_hash", mutate)
    if damage:
        with pytest.raises(ValueError, match="mismatch|changed"):
            module.verify(candidate, reference, expected, candidate_manifest_sha256=pinned, selected_keys=selected)
    else:
        result = module.verify(candidate, reference, expected, candidate_manifest_sha256=pinned, selected_keys=selected)
        assert result["exact_payloads_per_corpus"] == 1
        assert result["verified_files_including_manifests"] == 5

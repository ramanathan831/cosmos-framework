import hashlib
import json

import pytest

from tools.quantized_optimizers.verify_published_corpus import verify


@pytest.mark.parametrize("corrupt", [None, "candidate", "reference", "media", "manifest"])
def test_complete_hashes_are_required(tmp_path, corrupt):
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
    if corrupt:
        with pytest.raises(ValueError, match="mismatch|identity"):
            verify(candidate, reference, hashlib.sha256(value).hexdigest())
    else:
        result = verify(candidate, reference, hashlib.sha256(value).hexdigest())
        assert result["complete"]
        assert result["verified_files_including_manifests"] == 5

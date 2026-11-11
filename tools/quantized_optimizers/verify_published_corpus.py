"""Complete byte-identity verification of an already published native corpus.

This does not create a preparation receipt or assert that a timed-out producer
completed. The native producer publishes its manifest only after all workers'
independent raw replays succeeded. Require that manifest's externally recorded
identity, then hash every payload in both corpora and all bound source media.
No model, tokenizer, tensor library, or CUDA context is needed.
"""

import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def checked_hash(path):
    def identity(stat):
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    before = identity(path.stat())
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if identity(os.fstat(stream.fileno())) != before:
            raise ValueError(f"file replaced before verification: {path}")
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
        if identity(os.fstat(stream.fileno())) != before or identity(path.stat()) != before:
            raise ValueError(f"file changed during verification: {path}")
    return digest.hexdigest(), before


def verify(candidate, reference, expected_manifest_sha256, *, candidate_manifest_sha256=None, selected_keys=None):
    """Hash every byte, with an explicit pinned-subset mode for development.

    Native source/processor binding remains the caller's responsibility in subset
    mode. Equal serialized bytes establish exact tensor/token/mask equality without
    deserializing and rehashing a source video once for every target using it.
    """
    started = time.monotonic()
    manifests = [root / "manifest.json" for root in (candidate, reference)]
    stamps = {}
    if (selected_keys is None) != (candidate_manifest_sha256 is None):
        raise ValueError("subset verification requires both selection and candidate identity")
    expected_hashes = [candidate_manifest_sha256 or expected_manifest_sha256, expected_manifest_sha256]
    parsed = []
    for path, expected in zip(manifests, expected_hashes, strict=True):
        digest, stamps[path] = checked_hash(path)
        if digest != expected:
            raise ValueError("manifest differs from the externally qualified identity")
        value = path.read_bytes()
        if hashlib.sha256(value).hexdigest() != expected:
            raise ValueError("manifest changed before parsing its qualified identity")
        parsed.append(json.loads(value))
    manifest, oracle = parsed
    if selected_keys is not None:
        if set(selected_keys) != set(manifest["entries"]):
            raise ValueError("subset target selection mismatch")
        if any(manifest.get(k) != oracle.get(k) for k in ("format", "packer")):
            raise ValueError("subset processor/format mismatch")
        if any(oracle["entries"].get(k) != v for k, v in manifest["entries"].items()):
            raise ValueError("subset serialized entry identity mismatch")
        if any(oracle["media"].get(k) != v for k, v in manifest["media"].items()):
            raise ValueError("subset media identity mismatch")
    checks = {}
    for key, entry in manifest["entries"].items():
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("invalid prepared entry key")
        for root in (candidate, reference):
            path = root / f"{key}.pt"
            if path.resolve().parent != root.resolve() or path.stat().st_size != entry["bytes"]:
                raise ValueError("prepared entry path/size mismatch")
            checks[path] = entry["sha256"]
    for name, digest in manifest["media"].items():
        path = Path(name)
        if path in checks and checks[path] != digest:
            raise ValueError("conflicting content identities")
        checks[path] = digest

    def check(item):
        path, expected = item
        digest, stamp = checked_hash(path)
        if digest != expected:
            raise ValueError(f"content mismatch: {path}")
        return path, stamp

    with ThreadPoolExecutor(max_workers=4) as pool:
        stamps.update(pool.map(check, checks.items()))
    for path, before in stamps.items():
        stat = path.stat()
        if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != before:
            raise ValueError(f"file changed before verification completed: {path}")
    return {
        "complete": True,
        "scope": "Published corpus byte-identity verification; not an uninterrupted preparation receipt",
        "candidate": str(candidate),
        "reference": str(reference),
        "manifest_sha256": expected_hashes[0],
        "reference_manifest_sha256": expected_manifest_sha256,
        "exact_payloads_per_corpus": len(manifest["entries"]),
        "payload_bytes_per_corpus": sum(x["bytes"] for x in manifest["entries"].values()),
        "source_media_files": len(manifest["media"]),
        "verified_files_including_manifests": len(stamps),
        "wall_seconds": time.monotonic() - started,
        "python": os.sys.version,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.candidate, args.reference, args.manifest_sha256)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

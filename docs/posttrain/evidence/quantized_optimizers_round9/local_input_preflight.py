"""Audit transferred immutable native payloads on CPU, without running a processor.

The cluster preflight independently binds the live processor and source media.
This narrower check proves the planned row permutation's native collation before
GPU admission; all60 payload hashes and canonical normalized target keys match.
"""

import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from cosmos_framework.posttrain.data.packer.prepared_sft import target_from_row, target_key
from cosmos_framework.posttrain.data.packer.sft import QwenSFTDataPacker, SFTPolicyInput
from tools.quantized_optimizers.training_order_observer import check_collation, sample_identity

started = time.monotonic()
torch.set_num_threads(8)
root = Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text())
metadata = json.loads((root / "payload-metadata.json").read_text())
plan = json.loads(Path(__file__).with_name("input_order_plan.json").read_text())
packer = object.__new__(QwenSFTDataPacker)
packer.max_length = 4096
packer.tokenizer = SimpleNamespace(pad_token_id=151643)
proof = []
for item in plan["batches"]:
    identities, counts = {}, []
    for arm in ("original", "permuted"):
        samples = []
        for i in item[arm]:
            meta = metadata[str(i)]
            key = target_key(target_from_row(meta["target"]))
            assert key == meta["key"] and manifest["entries"][key] == meta["entry"]
            path = root / (key + ".pt")
            assert hashlib.sha256(path.read_bytes()).hexdigest() == meta["entry"]["sha256"]
            sample = SFTPolicyInput(**torch.load(path, map_location="cpu", weights_only=True, mmap=True))
            identity = sample_identity(sample)
            if i in identities:
                assert identities[i] == identity
            identities[i] = identity
            samples.append(sample)
        batch = packer.sft_collate_fn(samples, packer.sft_compute_max_len(samples))
        check_collation(samples, batch)
        valid = (
            (batch["labels"][:, 1:] != -100)
            & batch["attention_mask"][:, 1:].bool()
            & batch["attention_mask"][:, :-1].bool()
        )
        counts.append(valid.sum(dim=1).tolist())
    assert sorted(counts[0]) == sorted(counts[1]) and all(c > 0 for c in counts[0])
    proof.append(dict(step=item["step"], changed=item["changed"], aligned_rows=identities, counts=counts))
assert not torch.cuda.is_initialized()
Path(__file__).with_name("local_input_preflight.json").write_text(
    json.dumps(
        dict(
            complete=True,
            scope=__doc__,
            seconds=time.monotonic() - started,
            torch_version=torch.__version__,
            cuda_initialized=False,
            corpus_identity=hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
            batches=proof,
        ),
        indent=2,
    )
    + "\n"
)
print("Exact native collation, all60 payloads/15batches; CUDA uninitialized")

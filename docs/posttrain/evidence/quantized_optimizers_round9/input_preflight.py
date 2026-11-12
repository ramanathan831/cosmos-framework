"""CPU-only real prepared-row alignment proof, before GPU admission."""

import json
import os
from pathlib import Path

import torch

from cosmos_framework.posttrain.data.datasets.sft import JSONLSFTDataset
from cosmos_framework.posttrain.data.packer.prepared_sft import PreparedSFTCorpus, target_from_row
from cosmos_framework.posttrain.data.packer.sft import build_sft_packer
from cosmos_framework.posttrain.entrypoints.sft import sft_policy_source, validate_config_structure
from cosmos_framework.utils.config import load_config
from tools.quantized_optimizers.training_order_observer import check_collation, sample_identity

config = load_config("cosmos_framework/posttrain/configs/sft_optimizer_task.py", [])
rl = validate_config_structure(config)
model, trust = sft_policy_source(config)
packer = build_sft_packer(
    model,
    max_length=rl.max_length,
    processor_kwargs=rl.processor_kwargs,
    trust_remote_code=trust,
    mask_think_opener=rl.mask_think_opener,
)
data = JSONLSFTDataset(rl.data.path)
corpus = PreparedSFTCorpus(packer, os.environ["REFERENCE_CORPUS"], [data])
plan = json.loads(Path("docs/posttrain/evidence/quantized_optimizers_round9/input_order_plan.json").read_text())
assert len(data) == plan["training_rows"] == 346
proof = []
for item in plan["batches"]:
    rows = {}
    counts = []
    for arm in ("original", "permuted"):
        samples = [corpus.process(target_from_row(data[i])) for i in item[arm]]
        batch = packer.sft_collate_fn(samples, packer.sft_compute_max_len(samples))
        check_collation(samples, batch)
        for i, sample in zip(item[arm], samples, strict=True):
            identity = sample_identity(sample)
            if i in rows:
                assert rows[i] == identity
            rows[i] = identity
        valid = (
            (batch["labels"][:, 1:] != -100)
            & batch["attention_mask"][:, 1:].bool()
            & batch["attention_mask"][:, :-1].bool()
        )
        counts.append(valid.sum(dim=1).tolist())
    assert sorted(counts[0]) == sorted(counts[1]) and all(x > 0 for x in counts[0])
    proof.append(
        dict(
            step=item["step"],
            changed=item["changed"],
            aligned_rows=rows,
            counts=counts,
            objective="sum(row_token_CE / row_token_count) / 4",
            microbatch=4,
            accumulation=1,
        )
    )
assert not torch.cuda.is_initialized()
Path(os.environ["RESULTS"], "input_preflight.json").write_text(
    json.dumps(
        dict(
            complete=True,
            cuda_initialized=False,
            corpus_identity=corpus.identity,
            source_identity=data.cursor_fingerprint(),
            batches=proof,
        ),
        indent=2,
    )
    + "\n"
)

"""Read-only native training observations; diagnostic timings are never product timings."""

import contextlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cosmos_framework.posttrain.checkpoint.rng import collect_rng_state
from tools.quantized_optimizers.state_digest import digest, tensor_digest


def sample_identity(sample):
    return {
        "tokens": digest(sample.input_ids),
        "labels": digest(sample.labels),
        "length": len(sample.input_ids),
        "media": {k: tensor_digest(v) for k, v in sorted((sample.mm_kwargs or {}).items())},
        "media_shapes": {k: list(v.shape) for k, v in sorted((sample.mm_kwargs or {}).items())},
    }


def check_collation(samples, batch):
    """Prove each padded row and each concatenated media segment is associated correctly."""
    for i, sample in enumerate(samples):
        n = len(sample.input_ids)
        assert batch["input_ids"][i, :n].tolist() == sample.input_ids
        assert batch["labels"][i, :n].tolist() == sample.labels
        assert batch["attention_mask"][i, :n].eq(1).all()
        assert batch["attention_mask"][i, n:].eq(0).all()
        assert batch["labels"][i, n:].eq(-100).all()
    for key in set().union(*(set((s.mm_kwargs or {}).keys()) for s in samples)):
        if key == "mm_token_type_ids":
            for i, sample in enumerate(samples):
                value = (sample.mm_kwargs or {}).get(key)
                if value is not None:
                    assert torch.equal(batch[key][i, : len(sample.input_ids)], value[0])
            continue
        offset = 0
        for sample in samples:
            value = (sample.mm_kwargs or {}).get(key)
            if value is not None:
                assert torch.equal(batch[key][offset : offset + value.shape[0]], value)
                offset += value.shape[0]
        assert offset == batch[key].shape[0]


@contextlib.contextmanager
def forward_observation(trainer, packer, trajectories, output):
    original = packer.sft_collate_fn
    captured = {}

    def collate(samples, *args, **kwargs):
        batch = original(samples, *args, **kwargs)
        check_collation(samples, batch)
        captured["rows"] = [dict(id=t.id, **sample_identity(s)) for t, s in zip(trajectories, samples, strict=True)]
        captured["labels"] = batch["labels"][:, 1:].clone()
        captured["valid"] = (
            (captured["labels"] != -100)
            & batch["attention_mask"][:, 1:].bool()
            & batch["attention_mask"][:, :-1].bool()
        )
        captured["inputs"] = {k: tensor_digest(v) for k, v in batch.items() if k != "labels"}
        return batch

    def forward(_module, _args, kwargs, result):
        # Top-level model hook only; checkpointed decoder recomputations do not call it.
        assert "losses" not in captured
        for k, expected in captured["inputs"].items():
            assert tensor_digest(kwargs[k]) == expected, k
        logits = result if isinstance(result, torch.Tensor) else result.logits
        valid = captured["valid"].to(logits.device)
        labels = captured["labels"].to(logits.device)
        with torch.no_grad(), torch.autocast(logits.device.type, enabled=False):
            selected = logits[:, :-1][valid].float()
            losses = F.cross_entropy(selected, labels[valid], reduction="none").cpu()
            top = selected.topk(4, dim=-1).values.cpu()
        captured["losses"] = losses
        offset = 0
        for i, row in enumerate(captured["rows"]):
            mask = captured["valid"][i]
            n = int(mask.sum())
            row.update(
                supervised_tokens=n,
                target_ids=captured["labels"][i][mask].tolist(),
                token_losses=losses[offset : offset + n].tolist(),
                top4_logits=top[offset : offset + n].tolist(),
            )
            offset += n

    packer.sft_collate_fn = collate
    handle = trainer._net().register_forward_hook(forward, with_kwargs=True)
    try:
        yield
        output.update(rows=captured["rows"], packed_inputs=captured["inputs"])
    finally:
        handle.remove()
        packer.sft_collate_fn = original


def sample_indices(elements, device):
    """Exact endpoints even beyond float32's integer precision (2**24)."""
    if elements < 1:
        raise ValueError("diagnostic sampling requires a nonempty tensor")
    count = min(256, elements)
    return torch.arange(count, dtype=torch.int64, device=device) * (elements - 1) // max(count - 1, 1)


def tensor_stats(value):
    flat = value.detach().reshape(-1)
    indexes = sample_indices(flat.numel(), flat.device)
    return {
        "l2": float(torch.linalg.vector_norm(flat)),
        "max_abs": float(flat.abs().max()),
        "samples": flat[indexes].cpu().tolist(),
        "elements": flat.numel(),
    }


def install(trainer, packer, root, *, reference=None, write_full=False):
    """Observe actual forwards and clipped gradients/updates at1,13,14,15.

    Full first-update gradients are compared elementwise against the same-source
    reference; later sampled differences describe diverged trajectories, not
    same-state counterfactual gradients. No optimizer or RNG mutation is added.
    """
    root = Path(root)
    original_train, original_step = trainer.train, trainer.opt.step
    active = {}
    net = trainer._net()
    stochastic = []
    for name, module in net.named_modules():
        for key in ("p", "dropout", "attention_dropout", "drop_path_rate"):
            value = getattr(module, key, None)
            if isinstance(value, (int, float)) and value:
                stochastic.append({"module": name, "field": key, "value": value})
    (root / "stochastic_audit.json").write_text(
        json.dumps(
            {
                "nonzero_probability_fields": stochastic,
                "grad_accum": trainer.grad_accum,
                "strict_determinism": torch.are_deterministic_algorithms_enabled(),
            },
            indent=2,
        )
    )
    if trainer.grad_accum != 1:
        raise ValueError("this diagnostic requires one native four-row microbatch per update")

    def step(*args, **kwargs):
        if not active:
            return original_step(*args, **kwargs)
        before = {}
        records = {}
        for name, p in net.named_parameters():
            if not p.requires_grad:
                continue
            flat = p.detach().reshape(-1)
            indexes = sample_indices(flat.numel(), p.device)
            before[name] = (indexes, flat[indexes].clone())
            if p.grad is None:
                records[name] = None
                continue
            records[name] = tensor_stats(p.grad)
            if active["step"] == 1:
                path = root / "first_gradients" / (name + ".npy")
                if write_full:
                    path.parent.mkdir(exist_ok=True)
                    np.save(path, p.grad.detach().cpu().numpy(), allow_pickle=False)
                if reference:
                    prior = np.load(Path(reference) / "first_gradients" / (name + ".npy"), mmap_mode="r")
                    current = p.grad.detach().cpu().numpy()
                    assert prior.shape == current.shape
                    delta = current - prior
                    records[name]["reference_difference"] = {
                        "squared_l2": float(np.sum(delta.astype(np.float64) ** 2)),
                        "max_abs": float(np.max(np.abs(delta))),
                        "exact": bool(np.array_equal(prior, current)),
                    }
        result = original_step(*args, **kwargs)
        for name, p in net.named_parameters():
            if name in before and records[name] is not None:
                indexes, old = before[name]
                records[name]["update_samples"] = (p.detach().reshape(-1)[indexes] - old).cpu().tolist()
        active["parameters"] = records
        return result

    def train(batch):
        update = trainer.iteration() + 1
        if update not in (1, 13, 14, 15):
            return original_train(batch)
        active.update(step=update)
        rng = digest(collect_rng_state())
        try:
            with forward_observation(trainer, packer, batch, active):
                result = original_train(batch)
            active.update(
                metrics={k: v for k, v in result.items() if isinstance(v, (int, float, str))},
                rng_before=rng,
                rng_after=digest(collect_rng_state()),
            )
            with (root / "training_observations.jsonl").open("a") as stream:
                stream.write(json.dumps(active, allow_nan=False) + "\n")
            return result
        finally:
            active.clear()

    trainer.train, trainer.opt.step = train, step

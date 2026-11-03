# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
import copy
import json
import subprocess
import sys
from dataclasses import asdict

import pytest
import torch

from cosmos_framework.posttrain.config.build import build_lr_scheduler, build_optimizer
from cosmos_framework.posttrain.config.rl import OptimizerConfig


def test_config():
    assert asdict(OptimizerConfig(name="adamw_bnb_8bit"))["name"] == "adamw_bnb_8bit"
    with pytest.raises(ValueError):
        OptimizerConfig(name="adamw_8bit")
    for kwargs in ({"impl": "fused"}, {"bnb_min_8bit_size": 0}, {"bnb_fp32_state_names": ["x", "x"]}):
        with pytest.raises(ValueError):
            OptimizerConfig(name="adamw_bnb_8bit", **kwargs)


def test_adam_does_not_import_bnb():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, torch; from cosmos_framework.posttrain.config.build import build_optimizer; from cosmos_framework.posttrain.config.rl import OptimizerConfig; build_optimizer(torch.nn.Linear(2,2), OptimizerConfig()); assert 'bitsandbytes' not in sys.modules",
        ],
        check=True,
    )


@pytest.mark.parametrize("name", ["adamw_bnb_8bit", "paged_adamw_8bit"])
def test_gpu_resume_and_membership(name, tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA kernel gate")
    torch.manual_seed(17)
    net = torch.nn.Linear(512, 256, device="cuda")
    net.register_parameter("unused", torch.nn.Parameter(torch.ones(8192, device="cuda")))
    net.register_parameter("frozen", torch.nn.Parameter(torch.ones(8192, device="cuda"), requires_grad=False))
    net.register_parameter("tied", net.weight)
    cfg = OptimizerConfig(
        name=name, lr=0.001, warmup_steps=3, lr_multipliers={"bias": 2.0}, bnb_fp32_state_names=["bias"]
    )
    opt = build_optimizer(net, cfg)
    scheduler = build_lr_scheduler(opt, cfg)
    assert len([p for g in opt.param_groups for p in g["params"]]) == 3
    assert not opt.state_dict()["state"]
    for _ in range(3):
        net.weight.grad = torch.ones_like(net.weight)
        net.bias.grad = torch.ones_like(net.bias)
        opt.step()
        scheduler.step()
    report = opt.state_report()
    assert report["quantized_elements"] == net.weight.numel()
    assert report["uninitialized_elements"] == net.unused.numel()
    assert bool(report["paged_moment_bytes"]) == (name == "paged_adamw_8bit")
    saved = opt.state_dict()
    before = net.weight.detach().clone()
    net.bias.grad.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        opt.step()
    assert torch.equal(before, net.weight)
    assert saved["state"][0]["step"] == opt.state[net.weight]["step"]
    net.bias.grad.fill_(1)
    ckpt = tmp_path / "resume.pt"
    torch.save({"net": net.state_dict(), "opt": saved, "scheduler": scheduler.state_dict(), "cfg": asdict(cfg)}, ckpt)
    # A separate interpreter exercises CPU deserialization -> managed allocation.
    code = """
import torch, sys
from cosmos_framework.posttrain.config.build import build_optimizer, build_lr_scheduler
from cosmos_framework.posttrain.config.rl import OptimizerConfig
s=torch.load(sys.argv[1], weights_only=False)
n=torch.nn.Linear(512,256,device='cuda')
n.register_parameter('unused',torch.nn.Parameter(torch.ones(8192,device='cuda')))
n.register_parameter('frozen',torch.nn.Parameter(torch.ones(8192,device='cuda'),requires_grad=False))
n.register_parameter('tied',n.weight)
n.load_state_dict(s['net'])
c=OptimizerConfig(**s['cfg']);o=build_optimizer(n,c);lr=build_lr_scheduler(o,c)
o.load_state_dict(s['opt']);lr.load_state_dict(s['scheduler'])
assert n.unused not in o.state
assert bool(o.state_report()['paged_moment_bytes']) == (c.name == 'paged_adamw_8bit')
for _ in range(4):
 n.weight.grad=torch.ones_like(n.weight);n.bias.grad=torch.ones_like(n.bias);o.step();lr.step()
torch.save({'net':n.state_dict(),'opt':o.state_dict(),'scheduler':lr.state_dict()},sys.argv[2])
"""
    output = tmp_path / "continued.pt"
    subprocess.run([sys.executable, "-c", code, str(ckpt), str(output)], check=True)
    for _ in range(4):
        opt.step()
        scheduler.step()
    resumed = torch.load(output, weights_only=False)
    assert torch.equal(net.weight.cpu(), resumed["net"]["weight"].cpu())
    assert torch.equal(net.bias.cpu(), resumed["net"]["bias"].cpu())
    assert scheduler.state_dict() == resumed["scheduler"]
    assert saved["state"][0]["step"] == 3  # snapshot remains immutable
    broken = copy.deepcopy(saved)
    broken["cosmos_quantized_optimizer"]["name"] = "adamw"
    with pytest.raises(ValueError, match="identity"):
        opt.load_state_dict(broken)
    changed = copy.deepcopy(cfg)
    changed.lr *= 2
    with pytest.raises(ValueError, match="identity"):
        build_optimizer(net, changed).load_state_dict(saved)
    (tmp_path / "report.json").write_text(json.dumps(report))


def test_reject_cpu_and_missing_selector():
    cfg = OptimizerConfig(name="adamw_bnb_8bit")
    with pytest.raises(ValueError, match="CUDA FP32"):
        build_optimizer(torch.nn.Linear(4, 4), cfg)


@pytest.mark.parametrize("name", ["adamw_bnb_8bit", "paged_adamw_8bit"])
def test_native_sft_transaction(name, tmp_path):
    from types import SimpleNamespace

    from cosmos_framework.posttrain.algorithm.sft import make_objective
    from cosmos_framework.posttrain.roles.trainer.llm import LLMTrainer
    from cosmos_framework.posttrain.tests.helpers.sft_objective import MemoryPacker, Policy

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(512, 256, device="cuda")
            self.head = torch.nn.Linear(256, 512, device="cuda")

        def forward(self, input_ids, attention_mask):
            return SimpleNamespace(logits=self.head(self.embedding(input_ids)))

    packer = MemoryPacker()
    batch = [packer.add(list(range(64)), [-100] * 8 + list(range(8, 64)))]
    cfg = OptimizerConfig(name=name, lr=0.001, warmup_steps=3)

    def build():
        policy = Policy(Net())
        opt = build_optimizer(policy, cfg)
        return LLMTrainer(
            policy,
            make_objective(packer),
            opt,
            grad_accum=2,
            lr_scheduler=build_lr_scheduler(opt, cfg),
            max_grad_norm=1.0,
        )

    trainer = build()
    for _ in range(4):
        trainer.train(batch)
    before_state = trainer.opt.state_dict()
    before_scheduler = copy.deepcopy(trainer.lr_scheduler.state_dict())
    before_weights = [p.detach().clone() for p in trainer.model.parameters()]
    before_step = trainer._trainer_state()["optimizer_step"]
    assert trainer.train(batch)["optim/update_outcome"] == "accumulating"
    hook = next(trainer.model.parameters()).register_hook(lambda grad: grad * float("nan"))
    try:
        refused = trainer.train(batch)
    finally:
        hook.remove()
    assert refused["optim/skip_reason"] == "nonfinite_gradient"
    assert trainer._trainer_state()["optimizer_step"] == before_step
    assert trainer.lr_scheduler.state_dict() == before_scheduler
    assert all(torch.equal(a, b) for a, b in zip(before_weights, trainer.model.parameters()))
    after_state = trainer.opt.state_dict()
    for index, state in before_state["state"].items():
        for key, value in state.items():
            actual = after_state["state"][index][key]
            assert torch.equal(value, actual) if isinstance(value, torch.Tensor) else value == actual
    path = str(tmp_path / "iter_000002")
    trainer.save(path, data_state={"cursor": 4})
    expected_rng = torch.cuda.get_rng_state()
    restored = build()
    restored.load(path)
    assert torch.equal(torch.cuda.get_rng_state(), expected_rng)
    assert restored.checkpoint_manager.load_component(path, "data") == {"cursor": 4}
    assert restored.iteration() == trainer.iteration()
    for _ in range(4):
        trainer.train(batch)
        restored.train(batch)
    for a, b in zip(trainer.model.parameters(), restored.model.parameters()):
        assert torch.equal(a, b)
    assert trainer.lr_scheduler.state_dict() == restored.lr_scheduler.state_dict()


def test_missing_dependency(monkeypatch):
    import importlib.metadata

    from cosmos_framework.utils.quantized_optimizer import require_backend

    def missing(_):
        raise importlib.metadata.PackageNotFoundError("bitsandbytes")

    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(ImportError, match="quantized-optimizers"):
        require_backend()


def test_cuda_exceptions_are_local_and_restore_is_transactional():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    a = torch.nn.Linear(128, 128, device="cuda")
    b = torch.nn.Linear(128, 128, device="cuda")
    first = build_optimizer(a, OptimizerConfig(name="adamw_bnb_8bit", bnb_fp32_state_names=["weight"]))
    second = build_optimizer(b, OptimizerConfig(name="adamw_bnb_8bit"))
    for net, opt in [(a, first), (b, second)]:
        net.weight.grad = torch.ones_like(net.weight)
        net.bias.grad = torch.ones_like(net.bias)
        opt.step()
    assert first.state[a.weight]["state1"].dtype == torch.float32
    assert second.state[b.weight]["state1"].dtype == torch.uint8
    assert second.state[b.bias]["state1"].dtype == torch.float32
    assert second.state_report()["exceptions"]["bias"] == "small_tensor"
    saved = second.state_dict()
    broken = copy.deepcopy(saved)
    broken["state"][0]["absmax1"] = torch.zeros(1)
    previous = second.state
    with pytest.raises(ValueError, match="shape/dtype"):
        second.load_state_dict(broken)
    assert second.state is previous
    b.weight.grad = torch.ones_like(b.weight).to_sparse()
    with pytest.raises(ValueError, match="sparse"):
        second.step()
    assert second.state[b.weight]["step"] == 1
    for cfg in [
        OptimizerConfig(name="adamw_bnb_8bit", bnb_fp32_state_names=["missing"]),
        OptimizerConfig(name="adamw_bnb_8bit", bnb_fp32_state_names=["bias", "missing"]),
    ]:
        with pytest.raises(ValueError, match="unmatched"):
            build_optimizer(b, cfg)
    with pytest.raises(ValueError, match="FP32"):
        build_optimizer(b.to(torch.bfloat16), OptimizerConfig(name="adamw_bnb_8bit"))


def test_frozen_selection_stays_out():
    net = torch.nn.Linear(2, 2)
    net.weight.requires_grad_(False)
    opt = build_optimizer(net, OptimizerConfig(train_keys=["weight", "bias"]))
    assert opt.param_groups[0]["params"] == [net.bias]


def test_paged_buffer_lifetime_follows_storage_aliases(monkeypatch):
    import gc

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    from bitsandbytes.cextension import lib

    actual_free = lib._lib.cudaFree
    results = []

    def recorded_free(pointer):
        result = actual_free(pointer)
        results.append(result)
        return result

    monkeypatch.setattr(lib._lib, "cudaFree", recorded_free)
    net = torch.nn.Linear(512, 256, device="cuda", bias=False)
    opt = build_optimizer(net, OptimizerConfig(name="paged_adamw_8bit"))
    net.weight.grad = torch.ones_like(net.weight)
    opt.step()
    buffer = opt.state[net.weight]["state1"]
    owner = buffer._cosmos_managed_owner
    alias = buffer.detach()
    saved = opt.state_dict()
    assert not hasattr(saved["state"][0]["state1"], "is_paged")
    del buffer, opt
    gc.collect()
    assert owner() is not None
    # The detached storage alias is still valid after optimizer destruction.
    torch.testing.assert_close(alias, saved["state"][0]["state1"])
    del alias
    gc.collect()
    assert owner() is None
    assert len(results) >= 4  # scratch preflight + two live moments
    assert results == [0] * len(results)  # real cudaFree succeeded for every allocation


def test_only_native_sft_advertises_quantized_optimizers():
    from cosmos_framework.posttrain.config.rl import RunConfig
    from cosmos_framework.posttrain.config.sft import SFTConfig

    config = OptimizerConfig(name="adamw_bnb_8bit")
    SFTConfig(optimizer=config)
    with pytest.raises(ValueError, match="only by native SFT"):
        RunConfig(optimizer=config)
    with pytest.raises(ValueError, match="force_finite_grads"):
        OptimizerConfig(name="adamw_bnb_8bit", force_finite_grads=False)


def test_strategy_and_checkpoint_refusal_precedes_backend_allocation(monkeypatch):
    from types import SimpleNamespace

    from cosmos_framework.posttrain.config.build import refuse_rounding_or_unavailable_optimizer
    from cosmos_framework.posttrain.config.sft import SFTConfig
    from cosmos_framework.utils import quantized_optimizer

    def unexpected_backend(**kwargs):
        raise AssertionError("unsupported runtime must fail before backend allocation")

    monkeypatch.setattr(quantized_optimizer, "require_backend", unexpected_backend)
    config = SimpleNamespace(
        rl=SFTConfig(optimizer=OptimizerConfig(name="adamw_bnb_8bit"), checkpoint_format="consolidated")
    )
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="single CUDA"):
        refuse_rounding_or_unavailable_optimizer(config, config.rl.optimizer)
    monkeypatch.setenv("WORLD_SIZE", "1")
    config.rl.checkpoint_format = "dcp"
    with pytest.raises(ValueError, match="consolidated"):
        refuse_rounding_or_unavailable_optimizer(config, config.rl.optimizer)


def test_empty_and_noncontiguous_trainables_fail():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    for tensor in (torch.empty(0, device="cuda"), torch.ones(8, 16, device="cuda").T):
        net = torch.nn.Module()
        net.register_parameter("weight", torch.nn.Parameter(tensor))
        with pytest.raises(ValueError, match="nonempty contiguous"):
            build_optimizer(net, OptimizerConfig(name="adamw_bnb_8bit"))


def test_native_timing_hook_handles_nested_backend_steps(monkeypatch, tmp_path):
    import runpy
    from pathlib import Path

    from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    monkeypatch.setenv("RESULTS", str(tmp_path))
    probe = runpy.run_path(str(Path(__file__).resolve().parents[3] / "tools/quantized_optimizers/native_probe.py"))
    before = register_optimizer_step_pre_hook(probe["before"])
    after = register_optimizer_step_post_hook(probe["after"])
    try:
        net = torch.nn.Linear(128, 128, device="cuda")
        opt = build_optimizer(net, OptimizerConfig(name="adamw_bnb_8bit"))
        net.weight.grad = torch.ones_like(net.weight)
        opt.step()
    finally:
        before.remove()
        after.remove()
    rows = [json.loads(line) for line in (tmp_path / "native.telemetry.jsonl").read_text().splitlines()]
    assert len([row for row in rows if row["backend"] == "CosmosAdamW8bit"]) == 1
    assert not probe["timers"]

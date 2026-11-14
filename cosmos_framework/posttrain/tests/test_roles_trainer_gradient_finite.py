# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Gradient finite-check semantics; CUDA cases exercise the actual foreach kernels."""

from __future__ import annotations

import copy
import random
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from cosmos_framework.posttrain.roles.trainer import finite

DTYPES = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
QUALIFIED_TORCH_REVISIONS = (
    "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0",
    "70d99e998b4955e0049d13a98d77ae1b14db1f45",
)
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual CUDA foreach kernels")
QUALIFIED_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available() or getattr(torch.version, "git_version", None) not in QUALIFIED_TORCH_REVISIONS,
    reason="requires a qualified PyTorch source revision and actual CUDA foreach kernels",
)


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().resolve_conj().resolve_neg().contiguous().reshape(-1).view(torch.uint8).cpu().clone()


def _assert_exact(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_exact(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            _assert_exact(a, b)
    else:
        assert left == right


def _edge_gradients(dtype, device, fault):
    info = torch.finfo(dtype)
    gradients = [
        torch.tensor([0.0, -0.0, info.max, -info.max, info.tiny, info.tiny * info.eps], dtype=dtype, device=device),
        torch.zeros(65537, dtype=dtype, device=device),
        torch.tensor(0.0, dtype=dtype, device=device),
    ]
    if fault != "finite":
        gradients[1][-1] = {"nan": float("nan"), "positive_inf": float("inf"), "negative_inf": -float("inf")}[fault]
    return gradients


def _check_read_only_flags(gradients):
    before = [(_bytes(tensor), tensor._version, tensor.stride(), tensor.data_ptr()) for tensor in gradients]
    rng = torch.get_rng_state().clone()
    cuda_rng = {tensor.device: torch.cuda.get_rng_state(tensor.device) for tensor in gradients if tensor.is_cuda}
    flags = finite._gradient_finite_flags_by_device(gradients)
    expected = finite._finite_flags_by_device(gradients)
    assert set(flags) == set(expected)
    for device in flags:
        assert bool(flags[device]) == bool(expected[device])
        assert not flags[device].requires_grad
    for tensor, (values, version, stride, pointer) in zip(gradients, before):
        assert torch.equal(_bytes(tensor), values)
        assert tensor._version == version
        assert tensor.stride() == stride and tensor.data_ptr() == pointer
    assert torch.equal(torch.get_rng_state(), rng)
    for device, state in cuda_rng.items():
        assert torch.equal(torch.cuda.get_rng_state(device), state)
    return flags


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fault", ("finite", "nan", "positive_inf", "negative_inf"))
def test_cpu_gradient_path_keeps_elementwise_semantics(dtype, fault, monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("CPU gradients must retain the existing elementwise path")

    monkeypatch.setattr(torch, "_foreach_norm", unexpected)
    flags = _check_read_only_flags(_edge_gradients(dtype, "cpu", fault))
    assert bool(flags[torch.device("cpu")]) is (fault == "finite")


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fault", ("finite", "nan", "positive_inf", "negative_inf"))
def test_grouped_reduction_contract_with_cpu_norms(dtype, fault, monkeypatch):
    # Exercise grouping/immutability on CPU without presenting this as CUDA
    # qualification; the separately marked tests below use actual CUDA kernels.
    monkeypatch.setattr(torch.version, "git_version", QUALIFIED_TORCH_REVISIONS[0])
    monkeypatch.setattr(finite, "_foreach_finite_eligible", lambda _tensor: True)
    flags = _check_read_only_flags(_edge_gradients(dtype, "cpu", fault))
    assert bool(flags[torch.device("cpu")]) is (fault == "finite")


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize("revision", QUALIFIED_TORCH_REVISIONS)
@pytest.mark.parametrize("fault", ("finite", "nan"))
def test_qualified_source_revisions_retain_grouped_dispatch(revision, fault, monkeypatch):
    # This exercises dispatch and CPU arithmetic, not new accelerator qualification.
    monkeypatch.setattr(torch.version, "git_version", revision)
    monkeypatch.setattr(finite, "_foreach_finite_eligible", lambda _tensor: True)
    original = torch._foreach_norm
    calls = []

    def observed(tensors, order):
        calls.append((len(tensors), order))
        return original(tensors, order)

    monkeypatch.setattr(torch, "_foreach_norm", observed)
    flags = _check_read_only_flags(_edge_gradients(torch.float32, "cpu", fault))
    assert bool(flags[torch.device("cpu")]) is (fault == "finite")
    assert calls == [(3, float("inf"))]


@pytest.mark.L0
@pytest.mark.CPU
@pytest.mark.parametrize(
    "revision", (None, "", "unqualified-torch-2.13-source", QUALIFIED_TORCH_REVISIONS[0] + "-dirty")
)
@pytest.mark.parametrize("fault", ("finite", "nan", "negative_inf"))
def test_unqualified_source_uses_original_predicate_before_grouping(revision, fault, monkeypatch):
    if revision is None:
        monkeypatch.delattr(torch.version, "git_version")
    else:
        monkeypatch.setattr(torch.version, "git_version", revision)

    def unexpected(*_args, **_kwargs):
        pytest.fail("unqualified builds must not select or execute grouped CUDA reductions")

    monkeypatch.setattr(finite, "_foreach_finite_eligible", unexpected)
    monkeypatch.setattr(torch, "_foreach_norm", unexpected)
    flags = _check_read_only_flags(_edge_gradients(torch.float64, "cpu", fault))
    assert bool(flags[torch.device("cpu")]) is (fault == "finite")


@pytest.mark.L0
@pytest.mark.CPU
def test_empty_missing_nonfloating_and_tensor_wrapper_semantics(monkeypatch):
    assert finite.all_ranks_gradients_finite([])
    assert not finite.all_ranks_gradients_finite([], require_gradients=True)
    empty = torch.nn.Parameter(torch.empty(0))
    empty.grad = torch.empty_like(empty)
    assert finite.all_ranks_gradients_finite([empty], require_gradients=True)
    assert not finite._foreach_finite_eligible(empty.grad)
    assert finite._gradient_finite_flags_by_device([torch.ones(2, dtype=torch.int64)]) == {}
    assert finite._gradient_finite_flags_by_device([torch.ones(2, dtype=torch.bool)]) == {}

    class LocalGradient:
        def to_local(self):
            return torch.tensor([float("nan")])

    assert not finite.all_ranks_gradients_finite([SimpleNamespace(grad=LocalGradient())])
    assert not finite.all_ranks_gradients_finite([SimpleNamespace(grad=None)], require_gradients=True)


@pytest.mark.L0
@pytest.mark.CPU
def test_unsupported_layouts_delegate_without_conversion(monkeypatch):
    sparse = torch.sparse_coo_tensor(torch.tensor([[0, 1]]), torch.tensor([1.0, float("nan")]), (3,))
    observed = []

    def existing(tensors):
        observed.extend(tensors)
        return {}

    monkeypatch.setattr(finite, "_finite_flags_by_device", existing)
    assert finite._gradient_finite_flags_by_device([sparse]) == {}
    assert len(observed) == 1 and observed[0] is sparse


@pytest.mark.L0
@pytest.mark.CPU
def test_only_gradient_checks_use_the_new_reduction(monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("loss/model/optimizer checks must retain the original reducer")

    monkeypatch.setattr(finite, "_gradient_finite_flags_by_device", unexpected)
    assert finite.all_ranks_tensor_finite(torch.tensor(1.0))
    assert not finite.all_ranks_tensors_finite({"nested": [torch.tensor(float("nan"))]})
    net = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(net.parameters())
    assert finite.all_ranks_model_optimizer_finite(net, optimizer)


def _consensus_worker(rank, rendezvous, output):
    import torch.distributed as dist

    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        parameter = torch.nn.Parameter(torch.ones(3))
        parameter.grad = torch.full_like(parameter, float("nan") if rank else 1.0)
        assert not finite.all_ranks_gradients_finite([parameter], require_gradients=True)
        parameter.grad = torch.ones_like(parameter)
        assert not finite.all_ranks_gradients_finite([parameter] if rank == 0 else [], require_gradients=True)
        assert finite.all_ranks_gradients_finite([parameter], require_gradients=True)
        parameters = [parameter]
        if rank:
            misplaced = torch.nn.Parameter(torch.ones(1, device="meta"))
            misplaced.grad = torch.ones_like(misplaced)
            parameters.append(misplaced)
        with pytest.raises(RuntimeError, match="placement error, not a non-finite update"):
            finite.all_ranks_gradients_finite(parameters)
        Path(output, f"rank-{rank}").write_text("nonfinite/missing/finite/placement-consensus-passed")
    finally:
        dist.destroy_process_group()


@pytest.mark.L0
@pytest.mark.CPU
def test_two_rank_rejection_missing_and_placement_consensus(tmp_path):
    torch.multiprocessing.spawn(
        _consensus_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=2, join=True
    )
    assert (tmp_path / "rank-0").read_text() == (tmp_path / "rank-1").read_text()


@pytest.mark.L0
@pytest.mark.GPU
@QUALIFIED_CUDA
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fault", ("finite", "nan", "positive_inf", "negative_inf"))
def test_cuda_foreach_edges_and_input_immutability(dtype, fault, monkeypatch):
    original = torch._foreach_norm
    calls = []

    def observed(tensors, order):
        calls.append((len(tensors), order, tensors[0].dtype))
        return original(tensors, order)

    monkeypatch.setattr(torch, "_foreach_norm", observed)
    flags = _check_read_only_flags(_edge_gradients(dtype, "cuda", fault))
    assert bool(next(iter(flags.values()))) is (fault == "finite")
    assert calls == [(3, float("inf"), dtype)]


@pytest.mark.L0
@pytest.mark.GPU
@QUALIFIED_CUDA
@pytest.mark.parametrize("dtype", DTYPES)
def test_cuda_nonfinite_survives_warp_block_and_chunk_boundaries(dtype):
    # Exercise both sides of warp, 512-thread block and 65536-element foreach
    # chunk boundaries, including a short tail after two complete chunks.
    gradient = torch.ones(131073, device="cuda", dtype=dtype)
    for index in (0, 1, 31, 32, 511, 512, 65535, 65536, 65537, 131071, 131072):
        for value in (float("nan"), float("inf"), -float("inf")):
            gradient[index] = value
            flags = _check_read_only_flags([gradient])
            assert not bool(next(iter(flags.values()))), (dtype, index, value)
        gradient[index] = 1.0
    assert bool(next(iter(_check_read_only_flags([gradient]).values())))


@pytest.mark.L0
@pytest.mark.GPU
@QUALIFIED_CUDA
def test_cuda_mixed_groups_fallback_and_multiple_foreach_launches(monkeypatch):
    class TensorSubclass(torch.Tensor):
        pass

    gradients = [torch.full((3,), 1.0, dtype=dtype, device="cuda") for dtype in DTYPES]
    gradients[0].requires_grad_(True)
    gradients += [torch.ones(7, device="cuda") for _ in range(405)]
    gradients += [torch.empty(0, device="cuda"), torch.ones(3, 5, device="cuda").t()]
    complex_gradient = torch.tensor([1.0 + 1.0j, 2.0 + 0.0j], device="cuda")
    gradients += [complex_gradient]
    gradients += [torch._neg_view(torch.ones(4, device="cuda"))]
    gradients += [torch.ones(3, device="cuda").as_subclass(TensorSubclass)]
    assert all(not finite._foreach_finite_eligible(tensor) for tensor in gradients[-5:])
    original = torch._foreach_norm
    calls = []

    def observed(tensors, order):
        calls.append((len(tensors), tensors[0].dtype))
        return original(tensors, order)

    monkeypatch.setattr(torch, "_foreach_norm", observed)
    flags = _check_read_only_flags(gradients)
    assert bool(next(iter(flags.values())))
    assert sorted(count for count, _dtype in calls) == [1, 1, 1, 406]
    gradients[402][0] = float("nan")
    assert not bool(next(iter(_check_read_only_flags(gradients).values())))
    gradients[402][0] = 1.0
    complex_gradient[0] = complex(1.0, float("inf"))
    assert not bool(next(iter(_check_read_only_flags(gradients).values())))


@pytest.mark.L0
@pytest.mark.GPU
@CUDA
def test_cuda_unqualified_source_uses_unchanged_elementwise_predicate(monkeypatch):
    monkeypatch.setattr(torch.version, "git_version", "unqualified-source")

    def unexpected(*_args, **_kwargs):
        pytest.fail("unqualified CUDA builds must retain the elementwise predicate")

    monkeypatch.setattr(torch, "_foreach_norm", unexpected)
    for fault in ("finite", "nan", "positive_inf", "negative_inf"):
        flags = _check_read_only_flags(_edge_gradients(torch.float32, "cuda", fault))
        assert bool(next(iter(flags.values()))) is (fault == "finite")


@pytest.mark.L0
@pytest.mark.parametrize(
    "device",
    [pytest.param("cpu", marks=pytest.mark.CPU), pytest.param("cuda", marks=[pytest.mark.GPU, QUALIFIED_CUDA])],
)
def test_native_multiupdate_and_recovery_equal_elementwise(device, tmp_path, monkeypatch):
    from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
    from cosmos_framework.posttrain.roles.trainer import LLMTrainer

    new_check = finite._gradient_finite_flags_by_device
    if device == "cpu":
        # CPU validates native lifecycle with real CPU foreach reductions; only
        # the separately marked CUDA case qualifies accelerator execution.
        monkeypatch.setattr(torch.version, "git_version", QUALIFIED_TORCH_REVISIONS[0])
        monkeypatch.setattr(finite, "_foreach_finite_eligible", lambda _tensor: True)

    def run(grouped, resume):
        monkeypatch.setattr(
            finite, "_gradient_finite_flags_by_device", new_check if grouped else finite._finite_flags_by_device
        )
        random.seed(62)
        np.random.seed(62)
        torch.manual_seed(62)
        if device == "cuda":
            torch.cuda.manual_seed_all(62)

        def build():
            model = SimpleNamespace(
                net=torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.Dropout(0.1), torch.nn.Linear(16, 2)).to(
                    device
                )
            )
            optimizer = torch.optim.AdamW(model.net.parameters(), lr=1e-3, weight_decay=0.02, fused=device == "cuda")
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

            def objective(policy, _batch):
                x = torch.randn(3, 8, device=device)
                return policy.net(x).square().mean(), {}

            trainer = LLMTrainer(
                model,
                Objective(objective, Reduction.MEAN),
                optimizer,
                lr_scheduler=scheduler,
                grad_accum=2,
                max_grad_norm=1.0,
                require_gradients=True,
                nonfinite_update="fail_stop",
            )
            return model, optimizer, scheduler, trainer

        model, optimizer, scheduler, trainer = build()
        for microstep in range(10):
            trainer.train([])
            if resume and microstep == 5:
                path = tmp_path / f"resume-{grouped}"
                trainer.save(str(path))
                model, optimizer, scheduler, trainer = build()
                assert trainer.load(str(path)) == 3
        return {
            "model": copy.deepcopy(model.net.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scheduler": copy.deepcopy(scheduler.state_dict()),
            "trainer": copy.deepcopy(trainer._trainer_state()),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if device == "cuda" else None,
        }

    reference = run(False, False)
    for candidate in (run(True, False), run(True, True)):
        _assert_exact(candidate, reference)

"""Two-rank CUDA kernel/recovery gate for the native local-master path.

This tests shared infrastructure, not robot/driving task quality. Invoke once as
reference, then in a fresh torchrun as resume against the same output directory.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import weakref
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from cosmos_framework.posttrain.algorithm.primitives.objective import Objective, Reduction
from cosmos_framework.posttrain.checkpoint import CheckpointManager
from cosmos_framework.posttrain.config.build import build_lr_scheduler, build_optimizer
from cosmos_framework.posttrain.config.rl import OptimizerConfig
from cosmos_framework.posttrain.roles.trainer import LLMTrainer
from cosmos_framework.utils.optimizer_master import local_tensor


def equal(a, b, path="root"):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a.cpu(), b.cpu()), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            equal(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            equal(x, y, f"{path}[{i}]")
    else:
        # RNG numpy arrays have an elementwise comparison.
        import numpy as np

        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b), path
        else:
            assert a == b, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("reference", "resume"), required=True)
    parser.add_argument("--optimizer", choices=("adamw", "adamw_bnb_8bit", "paged_adamw_8bit"), required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    # Exercise multiple opaque chunks on the small accelerator fixture. The
    # production threshold is 64 MiB; model-sized GR00T validates >4-GiB payloads.
    from cosmos_framework.posttrain.checkpoint import master_optimizer as checkpoint

    checkpoint._CHUNK_BYTES = 256 * 1024
    torch.manual_seed(17 if args.phase == "reference" else 91)
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    net = torch.nn.Sequential(torch.nn.Linear(512, 512), torch.nn.Tanh(), torch.nn.Linear(512, 512))
    net.to(device="cuda", dtype=torch.bfloat16)
    # Mixed native storage in separate FSDP units: FP32 storage already owns its
    # master; BF16 storage needs a retained optimizer master. Compute stays BF16.
    net[0].float()
    net.register_parameter(
        "frozen", torch.nn.Parameter(torch.ones(4, device="cuda", dtype=torch.bfloat16), requires_grad=False)
    )
    net.register_parameter("unused", torch.nn.Parameter(torch.ones(1, device="cuda", dtype=torch.bfloat16)))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
    fully_shard(net[0], mesh=mesh, mp_policy=policy)
    fully_shard(net[2], mesh=mesh, mp_policy=policy)
    fully_shard(net, mesh=mesh, mp_policy=policy)
    cfg = OptimizerConfig(
        name=args.optimizer,
        update_policy="fp32_master",
        lr=1e-4,
        warmup_steps=6,
        impl="fused" if args.optimizer == "adamw" else "",
        bnb_fp32_state_names=[],
    )
    from cosmos_framework.utils.optimizer_master import MasterWeightOptimizer

    constructed = []
    initialize = MasterWeightOptimizer.__init__

    def track(instance, *args, **kwargs):
        initialize(instance, *args, **kwargs)
        constructed.append(weakref.ref(instance))

    MasterWeightOptimizer.__init__ = track
    try:
        try:
            build_optimizer(net, cfg, parameter_groups=() if rank == 1 else None)
        except RuntimeError as error:
            construction_error = error
            assert "native master optimizer construction" in str(error)
        else:
            raise AssertionError("one-rank construction failure did not converge")
    finally:
        MasterWeightOptimizer.__init__ = initialize
    gc.collect()
    assert construction_error.__traceback__ is not None and all(ref() is None for ref in constructed)
    opt = build_optimizer(net, cfg)
    objective = Objective(
        lambda m, b: (m.net(b[0]).float().square().mean(), {}),
        reduction=Reduction.MEAN,
        identity={"gate": "fsdp2-bf16-local-master-v1"},
    )
    trainer = LLMTrainer(
        SimpleNamespace(net=net),
        objective,
        opt,
        grad_accum=2,
        max_grad_norm=1.0,
        lr_scheduler=build_lr_scheduler(opt, cfg),
        checkpoint_manager=CheckpointManager(
            checkpoint_format="dcp", save_async=False, keep_optim_last=0, keep_model_last=0
        ),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = local_tensor(net.frozen).detach().clone()
    start = 0
    if args.phase == "resume":
        start = trainer.load(str(args.output / "iter_000004"))
        assert start == 4
        assert trainer.checkpoint_manager.load_component(str(args.output / "iter_000004"), "data") == {"cursor": 8}
    records = []
    for update in range(start + 1, 9):
        inputs = []
        for _ in range(2):
            x = torch.randn(4, 512, device="cuda", dtype=torch.bfloat16)
            inputs.append(x.cpu())
            result = trainer.train([x])
        assert result["optim/update_outcome"] == "applied"
        assert torch.equal(frozen, local_tensor(net.frozen))
        actual = {
            "inputs": inputs,
            "optimizer": opt.state_dict(),
            "scheduler": trainer.lr_scheduler.state_dict(),
            "model": {n: local_tensor(p).detach().cpu().clone() for n, p in net.named_parameters()},
            "control": trainer._trainer_state(),
        }
        record = args.output / f"rank{rank}-update{update}.pt"
        if args.phase == "reference":
            torch.save(actual, record)
        else:
            equal(actual, torch.load(record, map_location="cpu", weights_only=False))
        records.append(update)
        if update == 4 and args.phase == "reference":
            trainer.save(str(args.output / "iter_000004"), data_state={"cursor": 8})
    # A nonfinite gradient on ONE rank must skip both, including moments/masters.
    before = opt.state_dict()
    before_clock = trainer._trainer_state()["optimizer_step"]
    trainer.train([torch.ones(4, 512, device="cuda", dtype=torch.bfloat16)])
    if rank == 1:
        for p in net.parameters():
            if p.grad is not None and local_tensor(p.grad).numel():
                local_tensor(p.grad).fill_(float("nan"))
                break
    result = trainer.train([torch.ones(4, 512, device="cuda", dtype=torch.bfloat16)])
    assert result["optim/skip_reason"] == "nonfinite_gradient"
    assert before_clock == trainer._trainer_state()["optimizer_step"]
    equal(before, opt.state_dict())
    # Corrupt one rank's optimizer envelope after durable read. Both ranks must
    # reject it before changing any live checkpoint component.
    from cosmos_framework.posttrain.checkpoint.master_optimizer import _deserialize_rank_state, _serialize_rank_state

    reader = trainer.checkpoint_manager.load_dcp_component
    before_parameters = {n: local_tensor(p).detach().cpu().clone() for n, p in net.named_parameters()}
    before_control = trainer._trainer_state()

    def corrupt(path, name, target, **kwargs):
        found = reader(path, name, target, **kwargs)
        if name == "optim" and rank == 1:
            payload = _deserialize_rank_state(target["ranks"][str(rank)], target["format"])
            del payload["masters"]
            target["ranks"][str(rank)] = _serialize_rank_state(payload)
        return found

    trainer.checkpoint_manager.load_dcp_component = corrupt

    staged_backends, managed_owners = [], []
    create_backend = opt._backend

    def tracked_backend(masters):
        backend = create_backend(masters)
        staged_backends.append(weakref.ref(backend))
        return backend

    opt._backend = tracked_backend
    retained_errors = []
    try:
        trainer.load(str(args.output / "iter_000004"))
    except Exception as exc:
        assert "identity" in str(exc)
        retained_errors.append(exc)
    else:
        raise AssertionError("corrupt rank-local state was accepted")
    finally:
        trainer.checkpoint_manager.load_dcp_component = reader
    equal(before, opt.state_dict())
    equal(before_parameters, {n: local_tensor(p).detach().cpu() for n, p in net.named_parameters()})
    equal(before_control, trainer._trainer_state())
    control_file = args.output / "iter_000004" / "trainer.pt"
    control_bytes = control_file.read_bytes() if rank == 0 else None

    def mutate_generation(path, name, target, **kwargs):
        found = reader(path, name, target, **kwargs)
        if name == "optim":
            if rank == 0:
                control_file.write_bytes(control_bytes + b"changed generation")
            dist.barrier()
        return found

    trainer.checkpoint_manager.load_dcp_component = mutate_generation
    try:
        trainer.load(str(args.output / "iter_000004"))
    except Exception as exc:
        assert "changed while it was being loaded" in str(exc)
        retained_errors.append(exc)
    else:
        raise AssertionError("mutation during checkpoint reads was accepted")
    finally:
        trainer.checkpoint_manager.load_dcp_component = reader
        if rank == 0:
            control_file.write_bytes(control_bytes)
        dist.barrier()
    equal(before, opt.state_dict())
    equal(before_parameters, {n: local_tensor(p).detach().cpu() for n, p in net.named_parameters()})
    equal(before_control, trainer._trainer_state())
    commit = opt.commit_state_dict
    commit_calls = 0

    def fail_after_commit(staged):
        nonlocal commit_calls
        commit(staged)
        commit_calls += 1
        if commit_calls == 1:
            for state in opt.backend.state.values():
                for value in state.values():
                    if getattr(value, "is_paged", False):
                        managed_owners.append(value._cosmos_managed_owner)
            if rank == 1:
                raise RuntimeError("injected post-commit failure")

    opt.commit_state_dict = fail_after_commit
    try:
        trainer.load(str(args.output / "iter_000004"))
    except Exception as exc:
        assert "injected" in str(exc)
        retained_errors.append(exc)
    else:
        raise AssertionError("rank-local commit failure was not converged")
    finally:
        opt.commit_state_dict, opt._backend = commit, create_backend
    equal(before, opt.state_dict())
    equal(before_parameters, {n: local_tensor(p).detach().cpu() for n, p in net.named_parameters()})
    equal(before_control, trainer._trainer_state())
    gc.collect()
    assert len(retained_errors) == 3 and all(reference() is None for reference in staged_backends)
    assert all(reference() is None for reference in managed_owners)
    report = dict(
        phase=args.phase,
        rank=rank,
        optimizer=args.optimizer,
        checked_updates=records,
        state=opt.state_report(),
        cuda_peak=torch.cuda.max_memory_allocated(),
        torch=torch.__version__,
        device=torch.cuda.get_device_name(),
        passed=True,
        construction_failure_converged_and_released=True,
        rejected_state_released_with_retained_errors=True,
        changed_generation_rejected_before_mutation=True,
    )
    (args.output / f"{args.phase}-rank{rank}.json").write_text(json.dumps(report, indent=2))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

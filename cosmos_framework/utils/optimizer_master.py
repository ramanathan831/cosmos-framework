# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Explicit FP32 update ownership for native BF16 parameters and FSDP2 shards.

The model owns its Parameters throughout. The optimizer owns one ordinary FP32
master per *local* shard; gradients arrive after native reduction and clipping.
No forward casts, collectives over parameters, or extra clipping happen here.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from cosmos_framework.utils.quantized_optimizer import NAMES, _snapshot, build_quantized_optimizer


def local_tensor(parameter):
    """The currently owned FSDP shard, never a cached gathered forward parameter."""
    return parameter.to_local() if isinstance(parameter, DTensor) else parameter


def parameter_layout(parameter):
    if not isinstance(parameter, torch.nn.Parameter) or not parameter.requires_grad:
        raise ValueError("master updates require final trainable Parameters")
    if parameter.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("master updates support BF16/FP32 storage; FP16 unscaling is not implemented")
    local = local_tensor(parameter)
    if not local.is_contiguous():
        raise ValueError("master updates require contiguous local shards")
    layout = {"shape": list(parameter.shape), "local_shape": list(local.shape), "dtype": str(parameter.dtype)}
    if isinstance(parameter, DTensor):
        mesh = parameter.device_mesh
        if mesh.ndim != 1 or parameter.placements != (Shard(0),):
            raise ValueError("master updates require a one-dimensional FSDP2 Shard(0) mesh; no TP/HSDP/Partial")
        if not dist.is_initialized() or mesh.mesh.flatten().tolist() != list(range(dist.get_world_size())):
            raise ValueError("master updates require the complete learner process group in rank order")
        layout.update(mesh=mesh.mesh.tolist(), placements=["Shard(0)"])
    elif dist.is_initialized() and dist.get_world_size() > 1:
        raise ValueError("distributed master updates require native FSDP2 parameters, not replicated tensors")
    return layout


class MasterWeightOptimizer(torch.optim.Optimizer):
    """Model-facing optimizer with tracked local masters and a single AdamW backend.

    Native schedulers and trainers see live model parameters. The backend sees
    ordinary local FP32 Parameters only. Checkpoints use the rank-local schema,
    never torch's generic optimizer/DTensor conversion or dtype casting.
    """

    cosmos_local_master_optimizer = True

    def __init__(self, net, params, config):
        self.config = deepcopy(config)
        groups = params if params and isinstance(params[0], dict) else [{"params": params}]
        names = {id(p): n for n, p in net.named_parameters() if p.requires_grad}
        selected = [p for g in groups for p in g["params"]]
        if not selected or len({id(p) for p in selected}) != len(selected):
            raise ValueError("master optimizer needs nonempty, unique trainables")
        if any(id(p) not in names for p in selected):
            raise ValueError("master optimizer parameter is not owned by the final model")
        layouts = [parameter_layout(p) for p in selected]
        if len({local_tensor(p).device for p in selected}) != 1:
            raise ValueError("master optimizer requires one local device")
        if any(isinstance(p, DTensor) != isinstance(selected[0], DTensor) for p in selected):
            raise ValueError("mixed sharded/unsharded trainable ownership is unsupported")
        # CPU is useful for control-flow/numerical unit tests of the native AdamW
        # bridge. Quantized construction still performs the real CUDA preflight.
        self.live = tuple(selected)
        self.names = tuple(names[id(p)] for p in selected)
        defaults = dict(lr=config.lr, betas=tuple(config.betas), eps=config.eps, weight_decay=config.weight_decay)
        groups = [dict(g, param_names=[names[id(p)] for p in g["params"]]) for g in groups]
        super().__init__(groups, defaults)
        self.identity = {
            "format": "cosmos_local_master_v1",
            "optimizer": config.name,
            "torch_version": str(torch.__version__),
            "implementation": config.impl,
            "update_policy": "fp32_master",
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "rank": dist.get_rank() if dist.is_initialized() else 0,
            "parameters": [dict(name=n, **layout) for n, layout in zip(self.names, layouts)],
            "groups": [
                dict(
                    names=g["param_names"],
                    name=g.get("name"),
                    initial_lr=g["lr"],
                    betas=g["betas"],
                    eps=g["eps"],
                    weight_decay=g["weight_decay"],
                )
                for g in self.param_groups
            ],
        }
        # Already-FP32 storage is itself the master. Wrap its owned local storage
        # without another dense copy; BF16 parameters need one retained FP32 copy.
        self.masters = tuple(
            torch.nn.Parameter(
                local_tensor(p).detach() if p.dtype == torch.float32 else local_tensor(p).detach().float().clone()
            )
            for p in self.live
        )
        self.backend = self._backend(self.masters)
        self.cosmos_quantized_optimizer = config.name in NAMES
        self._reported_states = -1
        self._refresh_state()

    def _backend(self, masters):
        mapping = {id(p): m for p, m in zip(self.live, masters)}
        groups = [
            dict(g, params=[mapping[id(p)] for p in g["params"]], lr=identity["initial_lr"])
            for g, identity in zip(self.param_groups, self.identity["groups"])
        ]
        if self.config.name in NAMES:
            net = SimpleNamespace(named_parameters=lambda: iter(zip(self.names, masters)))
            return build_quantized_optimizer(net, groups, self.config, _local_owner=self.identity)
        kernel = {"fused": True} if self.config.impl == "fused" else {}
        if self.config.impl in ("foreach", "for-loop"):
            kernel["foreach"] = self.config.impl == "foreach"
        return torch.optim.AdamW(
            groups,
            lr=self.config.lr,
            betas=tuple(self.config.betas),
            eps=self.config.eps,
            weight_decay=self.config.weight_decay,
            **kernel,
        )

    def _refresh_state(self):
        self.state = defaultdict(
            dict, {p: self.backend.state[m] for p, m in zip(self.live, self.masters) if m in self.backend.state}
        )

    def sync_groups(self):
        """Capture the scheduler-owned groups, including changes after a step."""
        for live, backend in zip(self.param_groups, self.backend.param_groups):
            backend.update({k: v for k, v in live.items() if k != "params"})

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("master optimizer closures are unsupported")
        # Stage every gradient and converge failure before any rank mutates state.
        from cosmos_framework.posttrain.utils.dist import run_rank_converged

        def stage():
            staged = []
            for p, master, expected in zip(self.live, self.masters, self.identity["parameters"]):
                if parameter_layout(p) != {k: v for k, v in expected.items() if k != "name"}:
                    raise ValueError("live master parameter layout changed after construction")
                if p.dtype == torch.float32 and local_tensor(p).data_ptr() != master.data_ptr():
                    raise ValueError("FP32 model shard storage changed after optimizer binding")
                grad = p.grad
                if grad is not None:
                    if isinstance(grad, DTensor) and (
                        grad.placements != p.placements or grad.device_mesh != p.device_mesh
                    ):
                        raise ValueError("gradient ownership does not match model shard")
                    grad = local_tensor(grad)
                    if grad.is_sparse or not torch.isfinite(grad).all():
                        raise ValueError("sparse/nonfinite master gradient rejected before update")
                    grad = grad.float().contiguous() if grad.numel() else None
                staged.append(grad)
            return staged

        gradients = run_rank_converged(stage, "master optimizer gradient staging")
        for master, grad in zip(self.masters, gradients):
            master.grad = grad
        self.sync_groups()
        timing = None
        if self.masters and self.masters[0].is_cuda:
            # Device-time of the backend call only (moment/master arithmetic plus any
            # managed-memory prefetch or fault service it triggers). No synchronization
            # here; the events are read at the next update receipt.
            timing = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            timing[0].record()
        result = self.backend.step()
        if timing is not None:
            timing[1].record()
        self.last_backend_step_events = timing
        for p, m in zip(self.live, self.masters):
            local_tensor(p).copy_(m)
            if p.dtype != torch.float32:
                # FP32 staging is consumed by this step and is not recovery state.
                # Holding it until the next window's zero_grad keeps a full FP32
                # gradient copy (2x the BF16 trainables) alive through weight sync,
                # rollout and target preparation. FP32 parameters alias their own
                # gradient, which the trainer still owns, so only true copies go.
                m.grad = None
        self._refresh_state()
        if len(self.state) != self._reported_states:
            import logging

            logging.getLogger("posttrain").info("Native master optimizer state: %s", self.state_report())
            self._reported_states = len(self.state)
        return result

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        self.backend.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        self.sync_groups()
        return {
            "identity": deepcopy(self.identity),
            "masters": _snapshot(self.masters),
            "backend": _snapshot(self.backend.state_dict()),
        }

    def stage_state_dict(self, saved):
        """Allocate and validate a replacement without touching model or live state."""
        if set(saved) != {"identity", "masters", "backend"} or saved["identity"] != self.identity:
            raise ValueError("master optimizer identity/topology mismatch; exact resume requires the same layout")
        values = saved["masters"]
        if len(values) != len(self.masters):
            raise ValueError("missing master weights")
        for value, master in zip(values, self.masters):
            if value.shape != master.shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                raise ValueError("invalid master weight shape/dtype/value")
        saved_groups = saved["backend"].get("param_groups", [])
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("master optimizer group count mismatch")
        for old, live in zip(saved_groups, self.param_groups):
            for key in ("param_names", "name", "betas", "eps", "weight_decay"):
                if old.get(key) != live.get(key):
                    raise ValueError(f"master optimizer group {key} mismatch")
            if (
                not isinstance(old.get("lr"), (int, float))
                or not torch.isfinite(torch.tensor(old["lr"]))
                or old["lr"] < 0
            ):
                raise ValueError("invalid master optimizer learning rate")
        masters = tuple(
            torch.nn.Parameter(local_tensor(p).detach() if p.dtype == torch.float32 else v.to(m.device, copy=True))
            for v, m, p in zip(values, self.masters, self.live)
        )
        backend = self._backend(masters)
        if self.config.name not in NAMES:
            # AdamW.load_state_dict casts moments. Reject schema/dtype mismatches
            # first rather than silently repairing a damaged full-precision control.
            ids = [i for g in saved["backend"]["param_groups"] for i in g["params"]]
            if len(ids) != len(masters) or len(set(ids)) != len(ids) or set(saved["backend"]["state"]) - set(ids):
                raise ValueError("invalid master AdamW parameter mapping")
            for i, m in zip(ids, masters):
                state = saved["backend"]["state"].get(i, {})
                if state and set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                    raise ValueError("invalid master AdamW state schema")
                for key in ("exp_avg", "exp_avg_sq"):
                    if state and (state[key].shape != m.shape or state[key].dtype != torch.float32):
                        raise ValueError("invalid master AdamW moment shape/dtype")
        backend.load_state_dict(saved["backend"])
        return masters, backend, values

    def commit_state_dict(self, staged):
        masters, backend = staged[:2]
        if len(staged) == 3:
            # Only after validation/consensus: FP32 masters alias live model
            # storage and must not be copied during staging.
            with torch.no_grad():
                for p, m, value in zip(self.live, masters, staged[2]):
                    if p.dtype == torch.float32:
                        m.copy_(value)
        self.masters, self.backend = masters, backend
        for live, restored in zip(self.param_groups, self.backend.param_groups):
            live.update({k: v for k, v in restored.items() if k != "params"})
        self._refresh_state()

    def load_state_dict(self, saved):
        # The native checkpoint owner coordinates model and optimizer restore.
        # BF16 model values are restored there, never reconstructed by an update.
        self.commit_state_dict(self.stage_state_dict(saved))

    def state_report(self):
        report = (
            self.backend.state_report()
            if self.config.name in NAMES
            else {
                "moment_bytes": sum(
                    v.numel() * v.element_size()
                    for s in self.backend.state.values()
                    for k, v in s.items()
                    if k in ("exp_avg", "exp_avg_sq")
                ),
                "metadata_bytes": sum(
                    value.numel() * value.element_size()
                    for state in self.backend.state.values()
                    for key, value in state.items()
                    if key not in ("exp_avg", "exp_avg_sq") and isinstance(value, torch.Tensor)
                ),
            }
        )
        return dict(
            report,
            master_identity=self.identity,
            master_bytes=sum(
                m.numel() * m.element_size() for p, m in zip(self.live, self.masters) if p.dtype != torch.float32
            ),
            fp32_model_alias_bytes=sum(
                m.numel() * m.element_size() for p, m in zip(self.live, self.masters) if p.dtype == torch.float32
            ),
            gradient_staging_bytes=sum(
                m.grad.numel() * m.grad.element_size()
                for p, m in zip(self.live, self.masters)
                if m.grad is not None and p.dtype != torch.float32
            ),
            fp32_gradient_alias_bytes=sum(
                m.grad.numel() * m.grad.element_size()
                for p, m in zip(self.live, self.masters)
                if m.grad is not None and p.dtype == torch.float32
            ),
        )

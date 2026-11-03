# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Ordinary CUDA FP32 parameters with pinned, optionally paged eight-bit moments.

Imported only by the selected factory branch. Quantization mathematics is upstream.
No DTensor views, implicit parameter conversion, or global override registration.
"""

from __future__ import annotations

import importlib.metadata
import logging
from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace

import torch

VERSION = "0.49.2"
NAMES = ("adamw_bnb_8bit", "paged_adamw_8bit")


def require_backend():
    try:
        version = importlib.metadata.version("bitsandbytes")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError("Install cosmos-framework[quantized-optimizers] for bitsandbytes optimizers") from exc
    if version != VERSION:
        raise RuntimeError(f"bitsandbytes=={VERSION} required; found {version}")
    if not torch.cuda.is_available():
        raise ValueError("bitsandbytes optimizers require CUDA")
    import bitsandbytes as bnb
    from bitsandbytes.cextension import lib

    if not getattr(lib, "compiled_with_cuda", False):
        raise RuntimeError("bitsandbytes CUDA kernels are unavailable")
    return bnb


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        # Do not deepcopy paged tensors: Python attributes include managed pointers.
        out = torch.empty(value.shape, dtype=value.dtype, device="cpu")
        out.copy_(value.detach())
        return out
    if isinstance(value, dict):
        return {k: _snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_snapshot(v) for v in value)
    return deepcopy(value)


def build_quantized_optimizer(net, params, config):
    """Build from the final trainable objects and existing stable LR/decay partition."""
    groups = params if params and isinstance(params[0], dict) else [{"params": params}]
    names = {id(p): n for n, p in net.named_parameters() if p.requires_grad}
    selected = [p for g in groups for p in g["params"]]
    if not selected or len({id(p) for p in selected}) != len(selected):
        raise ValueError("quantized optimizer requires nonempty, identity-unique trainables")
    for p in selected:
        if type(p) is not torch.nn.Parameter or not p.requires_grad or id(p) not in names:
            raise ValueError(
                "quantized optimizer requires ordinary trainable Parameters, not DTensor/quantized weights"
            )
        if p.device.type != "cuda" or p.dtype != torch.float32 or not p.is_contiguous() or not p.numel():
            raise ValueError(
                "quantized optimizer requires nonempty contiguous CUDA FP32 trainables; no implicit masters"
            )
    if len({p.device for p in selected}) != 1:
        raise ValueError("quantized optimizer requires one CUDA device")
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError("quantized optimizer distributed strategies are not supported")
    exceptions = set(config.bnb_fp32_state_names)
    missing = exceptions - {names[id(p)] for p in selected}
    if missing:
        raise ValueError(f"unmatched FP32-state parameter names (use canonical tied name): {sorted(missing)}")
    groups = [dict(g, param_names=[names[id(p)] for p in g["params"]]) for g in groups]
    for g in groups:
        invalid = set(g) & {"fused", "foreach", "capturable", "amsgrad", "master_weights", "percentile_clipping"}
        if invalid:
            raise ValueError(f"unsupported bitsandbytes group options: {sorted(invalid)}")
    bnb = require_backend()

    class CosmosAdamW8bit(bnb.optim.AdamW8bit):
        cosmos_quantized_optimizer = True

        def __init__(self):
            super().__init__(
                groups,
                lr=config.lr,
                betas=tuple(config.betas),
                eps=config.eps,
                weight_decay=config.weight_decay,
                min_8bit_size=config.bnb_min_8bit_size,
                percentile_clipping=100,
                block_wise=True,
                is_paged=config.name == NAMES[1],
            )
            # Upstream stores managers globally. Own both registries per optimizer so
            # another trial's index overrides cannot change this model's numerics.
            self.mng = SimpleNamespace(index2config={}, module_weight_config_triple=[])
            self.page_mng = SimpleNamespace(paged_tensors=[])
            for gi, group in enumerate(self.param_groups):
                for pi, name in enumerate(group["param_names"]):
                    if name in exceptions:
                        self.mng.index2config[(gi, pi)] = {"optim_bits": 32}
            self.identity = {
                "format": "cosmos_bnb_v1",
                "backend": "bitsandbytes",
                "version": VERSION,
                "name": config.name,
                "storage": "float32",
                "topology": "single_cuda",
                "min_8bit_size": config.bnb_min_8bit_size,
                "fp32_state_names": sorted(exceptions),
                "groups": [
                    {
                        "names": g["param_names"],
                        "shapes": [list(p.shape) for p in g["params"]],
                        "name": g.get("name"),
                        "betas": tuple(g["betas"]),
                        "eps": g["eps"],
                        "weight_decay": g["weight_decay"],
                    }
                    for g in self.param_groups
                ],
            }
            self._reported_states = -1

        @torch.no_grad()
        def step(self, closure=None):
            if closure is not None:
                raise ValueError("quantized optimizer closures are unsupported")
            # Precheck the WHOLE update, before moments or clocks can change.
            grads = [p.grad for g in self.param_groups for p in g["params"] if p.grad is not None]
            if any(g.is_sparse for g in grads):
                raise ValueError("quantized optimizer does not support sparse gradients")
            # A single host decision and no full-size boolean intermediates.
            if grads and not torch.isfinite(torch.stack(torch._foreach_norm(grads, float("inf")))).all():
                raise FloatingPointError("nonfinite gradients: quantized update refused without mutation")
            result = super().step()
            count = sum(bool(s) for s in self.state.values())
            if count != self._reported_states:
                logging.getLogger("posttrain").info("Quantized optimizer state: %s", self.state_report())
                self._reported_states = count
            return result

        def state_report(self):
            report = {
                "backend": self.identity,
                "trainable_elements": sum(p.numel() for p in selected),
                "quantized_elements": 0,
                "fp32_elements": 0,
                "moment_bytes": 0,
                "metadata_bytes": 0,
                "paged_moment_bytes": 0,
                "uninitialized_elements": 0,
                "exceptions": {},
            }
            seen = set()
            for p in selected:
                state = self.state.get(p, {})
                if not state:
                    report["uninitialized_elements"] += p.numel()
                    continue
                quantized = state["state1"].dtype == torch.uint8
                report["quantized_elements" if quantized else "fp32_elements"] += p.numel()
                if not quantized:
                    report["exceptions"][names[id(p)]] = "explicit" if names[id(p)] in exceptions else "small_tensor"
                for key, value in state.items():
                    if not isinstance(value, torch.Tensor) or value.data_ptr() in seen:
                        continue
                    seen.add(value.data_ptr())
                    size = value.numel() * value.element_size()
                    report["moment_bytes" if key in ("state1", "state2") else "metadata_bytes"] += size
                    if key in ("state1", "state2") and getattr(value, "is_paged", False):
                        report["paged_moment_bytes"] += size
            return report

        def state_dict(self):
            torch.cuda.synchronize(selected[0].device)
            # Bypass upstream FSDP wrapping: this format explicitly refuses sharding.
            result = _snapshot(torch.optim.Optimizer.state_dict(self))
            result["cosmos_quantized_optimizer"] = deepcopy(self.identity)
            return result

        def load_state_dict(self, saved):
            if saved.get("cosmos_quantized_optimizer") != self.identity:
                raise ValueError(
                    "quantized optimizer exact-resume identity mismatch; use an explicit weights-only restart"
                )
            live_groups = self.param_groups
            if len(saved["param_groups"]) != len(live_groups):
                raise ValueError("quantized optimizer group count mismatch")
            # Validate mapping and schema before changing live state. Allocation is
            # transactional: failures restore the previous state and page ownership.
            ids = [i for g in saved["param_groups"] for i in g["params"]]
            if len(ids) != len(selected) or len(set(ids)) != len(ids) or set(saved["state"]) - set(ids):
                raise ValueError("invalid quantized optimizer parameter mapping")
            for old, new in zip(saved["param_groups"], live_groups):
                if old["param_names"] != new["param_names"] or len(old["params"]) != len(new["params"]):
                    raise ValueError("quantized optimizer parameter order mismatch")
                for key in ("betas", "eps", "weight_decay"):
                    if old[key] != new[key]:
                        raise ValueError(f"quantized optimizer {key} mismatch")
            previous, pages = self.state, self.page_mng.paged_tensors
            self.state, self.page_mng.paged_tensors = defaultdict(dict), []
            try:
                for gi, (old, group) in enumerate(zip(saved["param_groups"], live_groups)):
                    for pi, (idx, p) in enumerate(zip(old["params"], group["params"])):
                        source = saved["state"].get(idx, {})
                        if not source:
                            continue
                        self.init_state(group, p, gi, pi)  # allocation only, never step()
                        target = self.state[p]
                        if set(source) != set(target) or type(source["step"]) is not int or source["step"] < 0:
                            raise ValueError("invalid quantized optimizer state schema/counter")
                        for key, value in source.items():
                            if isinstance(target[key], torch.Tensor):
                                if (
                                    not isinstance(value, torch.Tensor)
                                    or value.shape != target[key].shape
                                    or value.dtype != target[key].dtype
                                ):
                                    raise ValueError(f"invalid quantized optimizer state {key} shape/dtype")
                                # qmaps can be shared; check equality instead of overwriting
                                # live/shared lookup tables from an unvalidated checkpoint.
                                if key.startswith("qmap") and not torch.equal(value.cpu(), target[key].cpu()):
                                    raise ValueError("quantized optimizer quantization map mismatch")
                                target[key].copy_(value)
                            else:
                                target[key] = value
                torch.cuda.synchronize(selected[0].device)
            except Exception:
                self.state, self.page_mng.paged_tensors = previous, pages
                raise
            self.param_groups = [
                dict(_snapshot(old), params=new["params"]) for old, new in zip(saved["param_groups"], live_groups)
            ]
            self.initialized = True
            self._reported_states = -1

    return CosmosAdamW8bit()

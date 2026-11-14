# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Controlled native action outputs for artifact parity, separate from task-quality claims."""

from __future__ import annotations

import hashlib

import numpy as np


def capture_outputs(policy, batch, *, seed):
    import torch

    from cosmos_framework.posttrain.entrypoints.action_sft import preserved_evaluation
    from cosmos_framework.posttrain.model.gr00t.rollout import predict_processed

    with preserved_evaluation(policy, seed):
        inputs = policy.packer(batch)
        identity = {}
        for name, value in inputs.items():
            if not torch.is_tensor(value):
                raise TypeError(f"native GR00T processed input {name} is not a tensor")
            value = value.detach().cpu().contiguous()
            identity[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
            }
        flow = policy.flow_matching_loss(inputs)
        normalized, native = predict_processed(policy, inputs)
        if not torch.isfinite(flow.loss):
            raise ValueError("nonfinite native GR00T qualification loss")
        return {
            "format": "gr00t-native-output-probe-v1",
            "seed": seed,
            "ids": [item.id for item in batch],
            "inputs": identity,
            "masked_elements": float(flow.action_mask.sum()),
            "flow_loss": float(flow.loss),
            "normalized_actions": normalized.cpu().tolist(),
            "native_decoded_actions": native.tolist(),
        }


def compare_outputs(reference, candidate, *, normalized_atol, decoded_atol, flow_loss_atol):
    """Call with prospectively declared tolerances; no default numerical/task acceptance margin."""
    for key in ("format", "seed", "ids", "inputs", "masked_elements"):
        if reference[key] != candidate[key]:
            raise ValueError(f"GR00T artifact comparison changed {key}")
    maxima = {}
    for key, tolerance in (
        ("normalized_actions", normalized_atol),
        ("native_decoded_actions", decoded_atol),
        ("flow_loss", flow_loss_atol),
    ):
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("artifact tolerances must be explicit finite nonnegative values")
        left, right = np.asarray(reference[key]), np.asarray(candidate[key])
        if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
            raise ValueError(f"GR00T artifact output shape/finiteness differs for {key}")
        maximum = float(np.max(np.abs(left - right)))
        if maximum > tolerance:
            raise ValueError(f"GR00T {key} maximum error {maximum} exceeds declared tolerance {tolerance}")
        maxima[key] = maximum
    return maxima

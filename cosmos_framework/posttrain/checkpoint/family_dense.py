# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Bounded dense deployment artifacts for native driving/action policy families.

Merge against the effective frozen base, never the untouched source weights.
Only merged language matrices use the requested output dtype; all other native
parameters and buffers retain their dtypes, including trained FP32 action heads.
The artifact requires its family-owned constructor/processor, not a generic HF
language-model loader. It is deliberately not a training-resume checkpoint.
"""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterator, Mapping
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from torch import nn

from cosmos_framework.posttrain.checkpoint.adapter import _require_subset, load_adapter_weights, read_adapter_metadata
from cosmos_framework.posttrain.checkpoint.dense_adapter import (
    DEFAULT_MAX_SHARD_SIZE_BYTES,
    _digest_json,
    _effective_weight,
    _file_record,
    _write_dense_shards,
    merge_effective_weight,
)
from cosmos_framework.posttrain.checkpoint.io import atomic_write_json, fsync_directory
from cosmos_framework.posttrain.checkpoint.provenance import checkpoint_inventory_signature
from cosmos_framework.posttrain.model.quantization.loader import tensor_digest
from cosmos_framework.utils.generator.lora import NativeLoRAMixin

FAMILY_DENSE_MANIFEST = "cosmos_family_dense_export.json"
FAMILY_DENSE_PROTOCOL = "cosmos.native-family-dense-export.v1"
_DTYPES = {
    name: getattr(torch, name)
    for name in ("bfloat16", "float32", "float64", "float16", "int64", "int32", "int16", "int8", "uint8", "bool")
}
_SAFE_DTYPES = {
    "bfloat16": "BF16",
    "float32": "F32",
    "float64": "F64",
    "float16": "F16",
    "int64": "I64",
    "int32": "I32",
    "int16": "I16",
    "int8": "I8",
    "uint8": "U8",
    "bool": "BOOL",
}


def _tree_signatures(root: Path) -> dict[str, tuple[int, ...]]:
    result = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("family dense artifacts must not contain symlinks")
        info = path.stat()
        result[str(path.relative_to(root))] = (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
    return result


def _sidecar_sources(sidecars: Mapping[str, str] | None) -> dict[str, Path]:
    result = {}
    for name, source in (sidecars or {}).items():
        relative = Path(name)
        if (
            not isinstance(name, str)
            or not name
            or relative.is_absolute()
            or any(part in {".", ".."} for part in relative.parts)
            or str(relative) != name
            or name in {FAMILY_DENSE_MANIFEST, "model.safetensors.index.json"}
            or name.endswith(".safetensors")
        ):
            raise ValueError(f"invalid family dense sidecar path: {name!r}")
        path = Path(source).resolve()
        if not path.is_file():
            raise ValueError(f"family dense sidecar is not a regular source file: {name}")
        result[name] = path
    if any(any(str(parent) in result for parent in Path(name).parents if str(parent) != ".") for name in result):
        raise ValueError("family dense sidecar paths collide")
    return result


def _validate_source_inventory(manifest: Mapping[str, Any]) -> None:
    from cosmos_framework.posttrain.model.checkpoint import ADAPTER_CHECKPOINT_PROTOCOL

    metadata, tensors = manifest["source_metadata"], manifest["tensors"]
    identity = metadata["identity"]
    if (
        set(metadata) != {"protocol", "identity", "topology", "parameters", "optimizer"}
        or metadata["protocol"] != ADAPTER_CHECKPOINT_PROTOCOL
        or metadata["topology"] != {"strategy": "single_device", "world_size": 1}
    ):
        raise ValueError("family dense source checkpoint schema/topology differs")
    targets = {f"{name}.weight" for name in identity["targets"]}
    adapters = (
        {f"{name}.lora_{factor}.weight" for name in identity["targets"] for factor in ("A", "B")}
        if identity["adapter"]["mode"].endswith("_lora")
        else set()
    )
    auxiliaries = set(identity["auxiliary_parameters"])
    trained = manifest["trained_tensors"]
    if set(metadata["parameters"]) != adapters | auxiliaries or set(trained) != adapters | auxiliaries:
        raise ValueError("family dense trained inventory omits adapters or auxiliary task parameters")
    for name, record in trained.items():
        if (
            set(record) != {"shape", "dtype", "sha256"}
            or {key: record[key] for key in ("shape", "dtype")} != metadata["parameters"][name]
        ):
            raise ValueError(f"family dense trained tensor metadata differs: {name}")
        if record["dtype"] != "float32" or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None:
            raise ValueError(f"family dense trained tensor precision/digest differs: {name}")
    expected_parameters = set(identity["frozen_tensors"]) | targets | auxiliaries
    if {name for name, record in tensors.items() if record["kind"] == "parameter"} != expected_parameters:
        raise ValueError("family dense parameter inventory differs from the source family")
    if {name for name, record in tensors.items() if record["kind"] == "buffer"} != set(identity["base_buffers"]):
        raise ValueError("family dense buffer inventory differs from the source family")
    for name, record in tensors.items():
        digest = {key: record[key] for key in ("shape", "dtype", "sha256")}
        expected = (
            trained[name]
            if name in auxiliaries
            else identity["base_buffers"][name]
            if record["kind"] == "buffer"
            else identity["frozen_tensors"].get(name)
            if name not in targets
            else None
        )
        if expected is not None and digest != {key: expected[key] for key in ("shape", "dtype", "sha256")}:
            raise ValueError(f"family dense changed a preserved head/base/buffer tensor: {name}")
        if name in targets:
            if (
                record["shape"] != identity["targets"][name.removesuffix(".weight")]
                or record["dtype"] != manifest["numerical_contract"]["merged_backbone_dtype"]
            ):
                raise ValueError(f"family dense target shape/output dtype differs: {name}")


def _buffers(net: nn.Module) -> dict[str, tuple[nn.Module, str, torch.Tensor, bool]]:
    return {
        f"{name}.{leaf}" if name else leaf: (owner, leaf, value, leaf not in owner._non_persistent_buffers_set)
        for name, owner in net.named_modules(remove_duplicate=False)
        for leaf, value in owner._buffers.items()
        if value is not None
    }


def _mixed_shard_plan(inventory: Mapping[str, Mapping[str, Any]], max_bytes: int) -> dict[str, str]:
    if type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("max_shard_size_bytes must be a positive integer")
    groups: list[list[str]] = []
    pending: list[str] = []
    size = 0
    for name, record in sorted(inventory.items()):
        shape, dtype = record["shape"], record["dtype"]
        if dtype not in _DTYPES or not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
            raise ValueError(f"unsupported dense family tensor shape/dtype: {name}")
        nbytes = math.prod(shape) * torch.empty((), dtype=_DTYPES[dtype]).element_size()
        if pending and size + nbytes > max_bytes:
            groups.append(pending)
            pending, size = [], 0
        pending.append(name)
        size += nbytes
        if size >= max_bytes:
            groups.append(pending)
            pending, size = [], 0
    if pending:
        groups.append(pending)
    return {
        name: f"model-{number:05d}-of-{len(groups):05d}.safetensors"
        for number, group in enumerate(groups, start=1)
        for name in group
    }


def _inventory(net: nn.Module, identity: Mapping[str, Any], dtype: str):
    parameters = dict(net.named_parameters(remove_duplicate=False))
    buffers = _buffers(net)
    if len({id(value) for value in parameters.values()}) != len(parameters):
        raise ValueError("family dense export does not support shared/tied parameter aliases")
    if set(parameters) & set(buffers):
        raise ValueError("family dense parameter and buffer names overlap")
    modules = {name: net.get_submodule(name) for name in identity["targets"]}
    adapted = identity["adapter"]["mode"].endswith("_lora")
    adapter_names = {f"{name}.lora_{factor}.weight" for name in modules for factor in ("A", "B")} if adapted else set()
    actual_adapters = {name for name, module in net.named_modules() if isinstance(module, NativeLoRAMixin)}
    if actual_adapters != (set(modules) if adapted else set()):
        raise ValueError("family dense adapter coverage differs from the source identity")
    if set(net._cosmos_adapter_parameters) != adapter_names | set(identity["auxiliary_parameters"]):
        raise ValueError("family dense trainables differ from the complete adapter/auxiliary inventory")
    if set(buffers) != set(identity["base_buffers"]):
        raise ValueError("family dense native buffers differ from the source identity")
    inventory = {}
    for name, parameter in parameters.items():
        if name in adapter_names:
            continue
        target = name.removesuffix(".weight") if name.endswith(".weight") else None
        inventory[name] = {
            "shape": list(identity["targets"][target]) if target in modules else list(parameter.shape),
            "dtype": dtype if target in modules else str(parameter.dtype).removeprefix("torch."),
            "kind": "parameter",
        }
    for name, (_, _, value, persistent) in buffers.items():
        inventory[name] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "kind": "buffer",
            "persistent": persistent,
        }
    return inventory, parameters, buffers, modules


def _dense_values(net: nn.Module, identity: Mapping[str, Any], dtype: str) -> Iterator[tuple[str, torch.Tensor]]:
    inventory, parameters, buffers, modules = _inventory(net, identity, dtype)
    for name in sorted(inventory):
        target = name.removesuffix(".weight") if name.endswith(".weight") else None
        if target in modules:
            module = modules[target]
            if isinstance(module, NativeLoRAMixin):
                effective = _effective_weight(module)
                try:
                    value = merge_effective_weight(
                        effective,
                        module.lora_A.weight,
                        module.lora_B.weight,
                        scale=module._lora_scale,
                        dtype=_DTYPES[dtype],
                    )
                finally:
                    del effective
            elif type(module) is nn.Linear:
                value = module.weight.detach().to(device="cpu", dtype=_DTYPES[dtype]).clone()
            else:
                from cosmos_framework.posttrain.model.quantization.bnb import NativeFrozenLinear4bit

                if not isinstance(module, NativeFrozenLinear4bit):
                    raise ValueError(f"unsupported frozen family target implementation: {target}")
                effective = module.effective_weight_for_initialization()
                try:
                    value = effective.to(device="cpu", dtype=_DTYPES[dtype]).clone()
                finally:
                    del effective
        else:
            tensor = parameters[name] if name in parameters else buffers[name][2]
            value = tensor.detach().cpu().clone()
        if list(value.shape) != inventory[name]["shape"] or not torch.isfinite(value).all().item():
            raise ValueError(f"family dense tensor is nonfinite or has the wrong shape: {name}")
        yield name, value.contiguous()
        del value


@torch.no_grad()
def export_family_dense(
    net: nn.Module,
    *,
    checkpoint: str,
    output: str,
    merged_backbone_dtype: str,
    exporter_build: Mapping[str, str],
    max_shard_size_bytes: int = DEFAULT_MAX_SHARD_SIZE_BYTES,
    sidecars: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Load a committed trained policy, stream the effective-base merge, commit last.

    CPU working memory is one output shard plus one current matrix and row-merge
    temporary. The resident policy and checkpoint's auxiliary payload remain
    additional costs. GPU dequantization holds one logical matrix at a time.
    """
    from cosmos_framework.posttrain.checkpoint.manager import CheckpointManager

    if merged_backbone_dtype not in {"bfloat16", "float32"}:
        raise ValueError("merged_backbone_dtype must be bfloat16 or float32")
    if set(exporter_build) != {"code_revision", "wheel_sha256"} or any(
        not isinstance(exporter_build[key], str) or re.fullmatch(r"[0-9a-f]{%d}" % length, exporter_build[key]) is None
        for key, length in (("code_revision", 40), ("wheel_sha256", 64))
    ):
        raise ValueError("family dense export requires the exact exporter revision and wheel SHA256")
    root = Path(output).absolute()
    if root.exists():
        raise FileExistsError(root)
    if root.resolve().is_relative_to(Path(checkpoint).resolve()):
        raise ValueError("family dense output must be separate from its checkpoint")
    copied = _sidecar_sources(sidecars)
    if any(root.resolve().is_relative_to(source.parent) for source in copied.values()):
        raise ValueError("family dense output must be separate from its source sidecars")
    before = checkpoint_inventory_signature(checkpoint)
    metadata = read_adapter_metadata(checkpoint)
    identity = metadata["identity"]
    if identity.get("format") != "cosmos.family-backbone.v1" or not identity.get("task_contract"):
        raise ValueError("family dense export requires a complete native family identity")
    codec = getattr(net, "_cosmos_adapter_checkpoint_codec", None)
    if codec is None:
        raise ValueError("family dense export requires the constructed native adapter policy")
    load_adapter_weights(checkpoint, codec, manager=CheckpointManager(checkpoint_format="dcp"))
    trained = {name: tensor_digest(value) for name, value in codec.parameters.items()}
    inventory, _, _, _ = _inventory(net, identity, merged_backbone_dtype)
    weight_map = _mixed_shard_plan(inventory, max_shard_size_bytes)
    root.mkdir(parents=False, exist_ok=False)
    fsync_directory(str(root.parent))
    records, total = _write_dense_shards(root, _dense_values(net, identity, merged_backbone_dtype), weight_map)
    tensors = {name: {**record, **inventory[name]} for name, record in records.items()}
    atomic_write_json(
        str(root / "model.safetensors.index.json"), {"metadata": {"total_size": total}, "weight_map": weight_map}
    )
    for name, source in copied.items():
        expected = _file_record(source)
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as reader, destination.open("xb") as writer:
            while block := reader.read(4 * 1024 * 1024):
                writer.write(block)
            writer.flush()
            os.fsync(writer.fileno())
        if _file_record(destination) != expected or _file_record(source) != expected:
            raise RuntimeError(f"family source sidecar changed while copying: {name}")
        fsync_directory(str(destination.parent))
    codec.validate_frozen_base()
    if trained != {name: tensor_digest(value) for name, value in codec.parameters.items()}:
        raise RuntimeError("trained family parameters changed during dense export")
    if checkpoint_inventory_signature(checkpoint) != before:
        raise RuntimeError("family source checkpoint changed during dense export")
    manifest = {
        "protocol": FAMILY_DENSE_PROTOCOL,
        "complete": True,
        "format": "cosmos_native_family_safetensors",
        "source_metadata": metadata,
        "source_metadata_sha256": _digest_json(metadata),
        "trained_tensors": trained,
        "numerical_contract": {
            "base": "dequantize(Q(W))" if identity["quantization"] else "W_bf16",
            "merge": "effective_base + (alpha/rank) * B @ A"
            if identity["adapter"]["mode"].endswith("_lora")
            else "effective_base",
            "accumulation_dtype": "float32",
            "merge_device": "cpu",
            "merged_backbone_dtype": merged_backbone_dtype,
            "other_parameters_and_buffers": "preserve_native_dtype",
            "requantized": False,
        },
        "resumable_training_checkpoint": False,
        "native_loader_required": True,
        "exporter_build": dict(exporter_build),
        "sidecars": sorted(copied),
        "sharding": {
            "policy": "sorted_greedy_payload_bytes_v1",
            "max_shard_size_bytes": max_shard_size_bytes,
            "oversized_tensor": "alone",
        },
        "tensors": tensors,
        "files": {
            str(path.relative_to(root)): _file_record(path) for path in sorted(root.rglob("*")) if path.is_file()
        },
    }
    atomic_write_json(str(root / FAMILY_DENSE_MANIFEST), manifest)
    fsync_directory(str(root))
    return read_family_dense(root)


def read_family_dense(
    path: str | Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify all bytes, inventory and numerical contracts before native construction.

    ``expected_identity`` is an explicit subset of the source family codec identity.
    The family loader must validate its architecture, processor and task contract;
    a self-consistent artifact alone does not select the intended policy.
    """
    from safetensors import safe_open

    root = Path(path)
    marker = root / FAMILY_DENSE_MANIFEST
    if not marker.is_file() or marker.is_symlink():
        raise ValueError("family dense export is incomplete: missing regular commit manifest")
    raw = marker.read_bytes()
    if expected_manifest_sha256 is not None and sha256(raw).hexdigest() != expected_manifest_sha256:
        raise ValueError("family dense manifest differs from the requested SHA256")
    manifest = json.loads(raw)
    expected_keys = {
        "protocol",
        "complete",
        "format",
        "source_metadata",
        "source_metadata_sha256",
        "trained_tensors",
        "numerical_contract",
        "resumable_training_checkpoint",
        "native_loader_required",
        "exporter_build",
        "sharding",
        "tensors",
        "files",
        "sidecars",
    }
    if (
        set(manifest) != expected_keys
        or manifest["protocol"] != FAMILY_DENSE_PROTOCOL
        or manifest["complete"] is not True
    ):
        raise ValueError("unsupported or incomplete family dense manifest schema")
    metadata = manifest["source_metadata"]
    identity = metadata.get("identity", {})
    numerical = manifest["numerical_contract"]
    if (
        manifest["format"] != "cosmos_native_family_safetensors"
        or manifest["resumable_training_checkpoint"] is not False
        or manifest["native_loader_required"] is not True
        or identity.get("format") != "cosmos.family-backbone.v1"
        or not identity.get("task_contract")
        or _digest_json(metadata) != manifest["source_metadata_sha256"]
        or numerical
        != {
            "base": "dequantize(Q(W))" if identity.get("quantization") else "W_bf16",
            "merge": "effective_base + (alpha/rank) * B @ A"
            if identity.get("adapter", {}).get("mode", "").endswith("_lora")
            else "effective_base",
            "accumulation_dtype": "float32",
            "merge_device": "cpu",
            "merged_backbone_dtype": numerical.get("merged_backbone_dtype"),
            "other_parameters_and_buffers": "preserve_native_dtype",
            "requantized": False,
        }
        or numerical["merged_backbone_dtype"] not in {"bfloat16", "float32"}
    ):
        raise ValueError("family dense source or numerical contract differs")
    if expected_identity is not None:
        _require_subset(identity, json.loads(json.dumps(dict(expected_identity), allow_nan=False)))
    files, tensors = manifest["files"], manifest["tensors"]
    if not isinstance(files, dict) or not isinstance(tensors, dict) or not files or not tensors:
        raise ValueError("family dense export requires complete file/tensor inventories")
    if {str(member.relative_to(root)) for member in root.rglob("*") if member.is_file()} != {
        *files,
        FAMILY_DENSE_MANIFEST,
    }:
        raise ValueError("family dense directory membership differs from its committed inventory")
    signatures = _tree_signatures(root)
    for name, record in files.items():
        member = root / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or str(Path(name)) != name
            or member.is_symlink()
            or not member.is_file()
            or _file_record(member) != record
        ):
            raise ValueError(f"family dense file differs from its committed digest: {name}")
    for name, record in tensors.items():
        keys = {"shape", "dtype", "sha256", "file", "kind"}
        if record.get("kind") == "buffer":
            keys.add("persistent")
            if type(record.get("persistent")) is not bool:
                raise ValueError(f"family dense buffer persistence is invalid: {name}")
        elif record.get("kind") != "parameter":
            raise ValueError(f"family dense tensor kind is invalid: {name}")
        if set(record) != keys or not isinstance(name, str) or not name or any(not part for part in name.split(".")):
            raise ValueError(f"family dense tensor schema differs: {name}")
    _validate_source_inventory(manifest)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weight_map = {name: record["file"] for name, record in tensors.items()}
    sharding = manifest["sharding"]
    if sharding != {
        "policy": "sorted_greedy_payload_bytes_v1",
        "max_shard_size_bytes": sharding.get("max_shard_size_bytes"),
        "oversized_tensor": "alone",
    }:
        raise ValueError("family dense sharding policy differs")
    if (
        _mixed_shard_plan(tensors, sharding["max_shard_size_bytes"]) != weight_map
        or index.get("weight_map") != weight_map
    ):
        raise ValueError("family dense index differs from its deterministic tensor inventory")
    sidecars = manifest["sidecars"]
    if (
        not isinstance(sidecars, list)
        or sidecars != sorted(set(sidecars))
        or set(sidecars) & {"model.safetensors.index.json", *weight_map.values()}
    ):
        raise ValueError("family dense sidecar inventory differs")
    if set(files) != {"model.safetensors.index.json", *weight_map.values(), *sidecars}:
        raise ValueError("family dense has missing or unindexed files")
    total = sum(
        math.prod(record["shape"]) * torch.empty((), dtype=_DTYPES[record["dtype"]]).element_size()
        for record in tensors.values()
    )
    if index.get("metadata") != {"total_size": total}:
        raise ValueError("family dense index total size differs")
    groups: dict[str, list[str]] = {}
    for name, filename in weight_map.items():
        groups.setdefault(filename, []).append(name)
    for filename, names in groups.items():
        with safe_open(str(root / filename), framework="pt", device="cpu") as reader:
            if set(reader.keys()) != set(names):
                raise ValueError("family dense shard membership differs")
            for name in names:
                record, header = tensors[name], reader.get_slice(name)
                if header.get_shape() != record["shape"] or header.get_dtype() != _SAFE_DTYPES[record["dtype"]]:
                    raise ValueError(f"family dense tensor header differs: {name}")
                if tensor_digest(reader.get_tensor(name)) != {key: record[key] for key in ("shape", "dtype", "sha256")}:
                    raise ValueError(f"family dense tensor payload differs: {name}")
    if _tree_signatures(root) != signatures or marker.read_bytes() != raw:
        raise RuntimeError("family dense export changed during verification")
    return manifest


@torch.no_grad()
def load_family_dense(
    net: nn.Module,
    path: str | Path,
    *,
    device: torch.device | str,
    expected_identity: Mapping[str, Any],
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Install a verified dense artifact into its native, unadapted CPU/meta policy.

    No base weights are loaded, no adapter is initialized, and no quantizer runs.
    The family constructor owns non-tensor state and must repair any unregistered
    buffer aliases and preferred layouts after this call. Serialized tensors use
    canonical contiguous values; arbitrary original strides are not an export
    parity promise. This does not create a resumable trainer.
    """
    from safetensors import safe_open

    if not expected_identity:
        raise ValueError("native family dense loading requires an explicit expected identity")
    verified = read_family_dense(path, expected_identity=expected_identity)
    if manifest is not None and dict(manifest) != verified:
        raise ValueError("family dense artifact changed after native construction preflight")
    if hasattr(net, "_cosmos_adapter_checkpoint_codec") or any(
        isinstance(module, NativeLoRAMixin) for module in net.modules()
    ):
        raise ValueError("family dense loading requires a fresh unadapted native architecture")
    parameters = dict(net.named_parameters(remove_duplicate=False))
    buffers = _buffers(net)
    if len({id(value) for value in parameters.values()}) != len(parameters):
        raise ValueError("family dense loading does not support shared/tied parameter aliases")
    inventory = verified["tensors"]
    expected_parameters = {name for name, record in inventory.items() if record["kind"] == "parameter"}
    expected_buffers = set(inventory) - expected_parameters
    if set(parameters) != expected_parameters or set(buffers) != expected_buffers:
        raise ValueError("native family architecture parameter/buffer inventory differs from dense artifact")
    for name, value in parameters.items():
        if value.device.type not in {"cpu", "meta"} or list(value.shape) != inventory[name]["shape"]:
            raise ValueError(f"native dense parameter must have matching CPU/meta shape: {name}")
    for name, (_, _, value, persistent) in buffers.items():
        if (
            value.device.type not in {"cpu", "meta"}
            or list(value.shape) != inventory[name]["shape"]
            or persistent != inventory[name]["persistent"]
        ):
            raise ValueError(f"native dense buffer shape/persistence differs: {name}")
    root = Path(path)
    signatures = _tree_signatures(root)
    for name, record in sorted(inventory.items()):
        with safe_open(str(root / record["file"]), framework="pt", device="cpu") as reader:
            value = reader.get_tensor(name)
        if tensor_digest(value) != {key: record[key] for key in ("shape", "dtype", "sha256")}:
            raise ValueError(f"family dense tensor changed during load: {name}")
        value = value.to(device=device, copy=True)
        if name in parameters:
            parent, _, leaf = name.rpartition(".")
            setattr(net.get_submodule(parent), leaf, nn.Parameter(value, requires_grad=False))
        else:
            owner, leaf, _, _ = buffers[name]
            setattr(owner, leaf, value)
    if _tree_signatures(root) != signatures:
        raise RuntimeError("family dense artifact changed during native load")
    net.eval()
    net._cosmos_dense_family_identity = verified["source_metadata"]["identity"]
    return verified

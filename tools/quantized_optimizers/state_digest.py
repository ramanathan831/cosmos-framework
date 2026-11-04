"""Streaming value identities for qualification, independent of serialization pointers."""

import hashlib
import json

import numpy as np
import torch


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def value_digest(value):
    if isinstance(value, torch.Tensor):
        return {"tensor": tensor_digest(value), "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.ndarray):
        return {
            "numpy": hashlib.sha256(value.tobytes()).hexdigest(),
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {str(key): value_digest(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [value_digest(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported qualification state: {type(value)}")


def digest(value):
    return hashlib.sha256(json.dumps(value_digest(value), sort_keys=True, allow_nan=False).encode()).hexdigest()

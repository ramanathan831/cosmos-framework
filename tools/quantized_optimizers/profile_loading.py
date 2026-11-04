"""Qualification-only host-observed weight-loading phases; never a timing trial."""

import json
import time

import safetensors
import torch


def install(owner, output):
    original_load = owner.load_weights

    def load(*args, **kwargs):
        original_open, original_to, original_copy = safetensors.safe_open, torch.Tensor.to, torch.Tensor.copy_
        current = {"key": None}
        with output.open("x") as stream:

            def record(phase, before, tensor=None):
                row = {"phase": phase, "key": current["key"], "seconds": time.monotonic() - before}
                if tensor is not None:
                    row.update(
                        bytes=tensor.numel() * tensor.element_size(), dtype=str(tensor.dtype), device=str(tensor.device)
                    )
                stream.write(json.dumps(row) + "\n")

            class Open:
                def __init__(self, *open_args, **open_kwargs):
                    self.context = original_open(*open_args, **open_kwargs)

                def __enter__(self):
                    self.inner = self.context.__enter__()
                    return self

                def __exit__(self, *exit_args):
                    return self.context.__exit__(*exit_args)

                def keys(self):
                    return self.inner.keys()

                def get_tensor(self, key):
                    current["key"] = key
                    before = time.monotonic()
                    tensor = self.inner.get_tensor(key)
                    record("get_tensor", before, tensor)
                    return tensor

            def to(tensor, *to_args, **to_kwargs):
                before = time.monotonic()
                result = original_to(tensor, *to_args, **to_kwargs)
                if result.device.type == "cuda":
                    torch.cuda.synchronize(result.device)
                record("to_" + result.device.type, before, result)
                return result

            def copy(target, value, *copy_args, **copy_kwargs):
                before = time.monotonic()
                result = original_copy(target, value, *copy_args, **copy_kwargs)
                if target.device.type == "cuda":
                    torch.cuda.synchronize(target.device)
                record("copy_" + target.device.type, before, result)
                return result

            safetensors.safe_open, torch.Tensor.to, torch.Tensor.copy_ = Open, to, copy
            try:
                return original_load(*args, **kwargs)
            finally:
                safetensors.safe_open, torch.Tensor.to, torch.Tensor.copy_ = original_open, original_to, original_copy

    owner.load_weights = load

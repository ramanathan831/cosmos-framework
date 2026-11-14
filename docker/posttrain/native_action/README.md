# Native action source runtime

This isolated **Linux x86_64, Python 3.12.13, CUDA 12.8** profile is shared by native
Alpamayo-R1 trajectory training and GR00T-N1.7 action training. It preserves the
version closure recorded by `runtime-flow-857f597-attempt02`, removes its four
private Cosmos/profile distributions, and adds bitsandbytes 0.49.2. Important pins
are Torch 2.10.0+cu128, torchvision 0.25.0+cu128, Transformers 4.57.6, Diffusers
0.35.1, NumPy 1.26.4, safetensors 0.8.0 and PyAV 18.0.0.

From the selected Cosmos checkout, with a bootstrap `uv==0.11.29` executable:

```bash
python docker/posttrain/native_action/install.py \
  --uv /path/to/uv-0.11.29 --venv /durable/venvs/cosmos-native-action
source /durable/venvs/cosmos-native-action/native-action-env.sh
python docker/posttrain/native_action/install.py \
  --venv /durable/venvs/cosmos-native-action --verify-only
```

The installer refuses an existing environment, creates a new managed Python
environment, installs exact package versions from PyPI and the official PyTorch
CUDA index, and runs `uv pip check`. It clones public Isaac-GR00T at
`51d4c89f72fda44cbf77285c6a8114b52676b8a1` and NVlabs/Alpamayo at
`11a0e01c13a5622377c45ee37d653351453ec43b`, then checks clean revisions and the same
source hashes required by the native loaders. Activation sets `PYTHONPATH` to
those sources and this Cosmos checkout. It does not modify Slurm routing, GPU
placement, `LD_LIBRARY_PATH`, the main project dependencies or Qwen's runtime.
The environment records versions, source hashes and the checkout revision in
`native-action-runtime.json`; rerun verification after moving or changing it.

This is intentionally a **source-checkout installation**. Do not subsequently
`pip install cosmos-framework`, Isaac-GR00T or Alpamayo into it and let their
independent dependency ranges resolve: current Cosmos metadata requires
Diffusers >=0.39, upstream GR00T pins Torch 2.9/Transformers 4.57.3, and Alpamayo
pins Torch 2.8/Transformers 4.57.1. Those metadata contracts do not describe this
measured compatibility profile. Public model Python code is unchanged and its
bytes are verified; this does not claim every upstream feature works with the
selected dependency overrides. The version-pinned requirements are not a
wheel-byte lock. A fresh installation still needs import/kernel/task checks
before its artifacts can inherit qualification from a retained container.

After activation use the normal [Alpamayo](../../../docs/post_training/native_qlora_alpamayo.md)
or [GR00T](../../../docs/post_training/native_qlora_gr00t.md) `torchrun` recipe.
Checkpoints and datasets are separate inputs. The profile contains no LIBERO or
AlpaSim simulator and makes no rollout, serving-latency or distributed claim.
Dense export additionally records `COSMOS_TRAINING_CODE_REVISION` and
`COSMOS_TRAINING_WHEEL_SHA256`; supply the exact reviewed build provenance rather
than inventing a wheel hash for a source checkout.

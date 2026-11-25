# 8-bit AdamW optimizers

Post-training can keep AdamW moments in 8 bits with
[bitsandbytes 0.49.2](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/0.49.2/bitsandbytes/optim/adamw.py).
This cuts optimizer-state memory by about 75% for eligible tensors. Model weights are not
quantized, and existing recipes are unchanged unless a config selects one of the names below.

| Name               | Moments                            | Memory                                                                 |
| ------------------ | ---------------------------------- | ---------------------------------------------------------------------- |
| `adamw_bnb_8bit`   | Blockwise uint8, FP32 block scales | Ordinary CUDA memory                                                   |
| `paged_adamw_8bit` | Same                               | CUDA managed (paged) memory for tensors with at least 100,000 elements |

`adamw_8bit` is rejected on purpose, so a checkpoint's optimizer name always identifies the
backend that wrote it.

## Install

The backend is an optional extra. Ordinary optimizers work without it, because imports are
lazy.

```sh
uv sync --extra quantized-optimizers
```

## Where it is supported

| Workflow                                | How to select                                                             | Update policy                          |
| --------------------------------------- | ------------------------------------------------------------------------- | -------------------------------------- |
| Native posttrain SFT                    | `rl.optimizer.name=adamw_bnb_8bit`                                        | `parameter` (default) or `fp32_master` |
| GR00T Joint PPO                         | `rl.optimizer.name=adamw_bnb_8bit rl.optimizer.update_policy=fp32_master` | `fp32_master` required                 |
| Generator (VFM) SFT, e.g. action policy | `optimizer.optimizer_type=adamw_bnb_8bit`                                 | `fp32_master` behaviour, always        |
| AlpaGym CLRL                            | `--optimizer adamw_bnb_8bit --optimizer-update-policy fp32_master`        | `fp32_master`                          |

Other workflows reject both names, and `update_policy=fp32_master`, at config validation, before
a model is built: only the SFT and Joint PPO trainers save and restore the master checkpoint
schema.

- **Native SFT, `parameter`:** one CUDA device, FP32 trainable storage and
  `rl.checkpoint_format=consolidated`.
- **Native SFT, `fp32_master`:** BF16 or FP32 storage, one process or one-dimensional FSDP2,
  and either checkpoint format.
- **Generator (VFM) SFT:** set `checkpoint.dcp_async_mode_enabled=false`. The rank-local
  optimizer payload isn't staged through the pinned-memory async save, so the run refuses it
  when it starts.

Config validation (`--validate-manifest`) checks only that the pinned bitsandbytes package is
installed; it doesn't use a GPU. The CUDA kernels are probed once, on every rank, before the
model is built.

## Update policies

`rl.optimizer.update_policy` controls where the FP32 update happens.

- **`parameter`** (default): the optimizer updates the model's own parameters.
  - Trainables must be contiguous, nonempty CUDA FP32 parameters on one device.
  - BF16 trainables are rejected. Without FP32 masters, updates smaller than the BF16 spacing
    round away.
  - DDP, FSDP and DTensor are rejected.
- **`fp32_master`**: the optimizer owns one FP32 master per local shard.
  - BF16 storage gets a tracked FP32 copy; FP32 shards alias the model storage without a copy.
  - Gradients are staged after the framework's own reduction and global-norm clipping.
  - It runs one backend step and copies the rounded result back into the model shard.
  - It supports a single process, or one-dimensional FSDP2 `Shard(0)` meshes over the full world.
  - FSDP1, HSDP, TP/EP, FP16 and optimizer offload are rejected.
  - With `name=adamw` it is the matched full-precision control.
  - `bnb_min_8bit_size` is compared with the logical parameter size, not the local shard, so a
    parameter gets the same moment precision on every rank and at every world size.
  - The 8-bit quantization blocks (256 elements, each with its own scale) are laid over each
    local shard. Exact 8-bit numerics therefore depend on the world size, which is why exact
    resume requires the same topology.

## Options

| Option                 | Default     | Meaning                                                                                                                                                      |
| ---------------------- | ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `bnb_min_8bit_size`    | `4096`      | Tensors with fewer elements keep FP32 moments                                                                                                                |
| `bnb_fp32_state_names` | `[]`        | Exact canonical parameter names that keep FP32 moments. Use the final name, including wrapper prefixes. Unknown, frozen, duplicate or tied-alias names fail. |
| `update_policy`        | `parameter` | See above                                                                                                                                                    |

Both `bnb_*` options are only valid with the 8-bit names. Other restrictions:

- `impl`, `fused`, `foreach`, `capturable` and `amsgrad` are rejected.
- `force_finite_grads` must stay true.
- Backend percentile clipping is disabled; clipping and gradient accumulation stay with Cosmos.

A window with any nonfinite gradient changes no moments and does not advance the scheduler.

## Checkpointing and resume

Exact resume requires the same backend and version, optimizer name, update policy,
parameter names and shapes, group order, hyperparameters and topology.

- **What is saved:** moments, block scales and maps, counters and FP32 masters. The
  `fp32_master` path writes rank-owned DCP state.
- **What is checked on load:** every rank checks that the restored masters equal its model
  shard before anything live is replaced.
- **Failures:** a partial, corrupted or changed checkpoint is rejected and leaves the live
  optimizer untouched.
- **Paged state:** paged moments are restored into managed memory.
- **Changing optimizer:** switching optimizer families needs an explicit weights-only restart.
- **Exported weights:** exported model and adapter weights don't need bitsandbytes at
  inference time.

The consolidated SFT checkpoint keeps only the newest optimizer state by default. Set
`rl.keep_optim_last=0` to keep several exact-resume points.

## Example: native SFT

```sh
torchrun --standalone --nproc-per-node=1 \
  -m cosmos_framework.posttrain.entrypoints.sft \
  --config cosmos_framework/posttrain/configs/sft_quantized_optimizer.py -- \
  rl.optimizer.name=adamw_bnb_8bit rl.optimizer.bnb_min_8bit_size=16384
```

- **Paged state:** use `rl.optimizer.name=paged_adamw_8bit`.
- **Full-precision baseline for comparison:**
  `rl.optimizer.name=adamw rl.optimizer.impl=fused rl.optimizer.bnb_min_8bit_size=4096`.
  The last override clears the 8-bit-only option.

The GR00T recipes are `configs/gr00t_n1d7_libero_quantized_optimizer.py` and
`configs/gr00t_n1d7_libero10_quantized_optimizer.py`.

On a network filesystem where ranks can see different cached timestamps for one finished
checkpoint (seen on Lustre in multi-GPU GR00T runs), set
`COSMOS_CHECKPOINT_SHARED_FS_IDENTITY=1`. Checkpoint provenance then uses server-synchronized
attributes, and validation publication identifies a checkpoint by its full content. Each byte is
read once across the ranks. Keep the setting the same for the whole run, including resumes.

## GR00T PPO options

These are off by default; at their defaults, native behaviour and checkpoint identity are
unchanged.

| Option                         | Default | Meaning                                                                                                   |
| ------------------------------ | ------- | --------------------------------------------------------------------------------------------------------- |
| `model.ppo_reference_kl_beta`  | `0.0`   | Adds `beta · KL(π_θ ‖ π_init)` to the actor loss, against a frozen copy of the initial action head        |
| `model.value_feature_gradient` | `true`  | `false` detaches the shared features feeding the value head, so the value loss trains only the value head |

## Measured results

The tests used 8×A100-80GB, with protocols locked before each comparison.

|                                              | GR00T-N1.7 LIBERO-10 PPO | Cosmos3-Nano LIBERO-10 SFT                                   |
| -------------------------------------------- | ------------------------ | ------------------------------------------------------------ |
| Optimizer moments per rank                   | 13.00 → 3.30 GB (−74.6%) | 6.98 → 1.77 GB (−74.6%)                                      |
| Exact reload and resume, non-paged and paged | Pass                     | Pass                                                         |
| Quality vs full precision (3 seeds)          | Within a 2-point margin  | 1.6 points lower at the final checkpoint; 95% CI [−3.3, 0.0] |
| Step time                                    | Unchanged                | Unchanged (46.2 s/iter)                                      |

Two caveats about what the savings buy:

- **Memory saving includes masters only partly.** With `fp32_master`, the FP32 masters remain;
  counting them, GR00T optimizer-owned memory falls 49.7%.
- **The freed memory does not make training faster.** On this hardware, a larger per-GPU batch
  doesn't change step time, and turning off activation recomputation doesn't fit for either
  optimizer.

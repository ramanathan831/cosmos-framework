# Native GR00T N1.7 backbone adaptation

This integration connects native LIBERO action flow matching to ordinary LoRA on a frozen
language backbone. The implementation is available; task quality, performance and device support
must be read from the accompanying qualification evidence. CPU tests and configuration acceptance
alone do not establish accelerator support or robotic task success.

The initial architecture is `nvidia/GR00T-N1.7-LIBERO` at
`2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21`, LIBERO-spatial policy, with the checkpoint's
16 retained Cosmos-Reason2-2B language layers, 2048 hidden width and 40 × 132 action head.
The native processor supplies the `libero_sim` embodiment, state/action statistics, two RGB cameras
and 16 valid embodiment action slots. The selected prepared demonstration has ten supervised
7-dimensional environment-action steps. State is xyz, axis-angle and two finger positions;
training converts environment gripper `[-1,1]` to native `[0,1]`, then uses checkpoint normalization.
Action evaluation reverses normalization and gripper conversion. Native four-step flow sampling
is unchanged. Offline action error is not rollout success or a real-robot safety result.

The language allowlist contains exactly 112 bias-free matrices: q/k/v/o and gate/up/down in each
retained layer. Vision, embeddings, norms, output head and all action components are excluded
from NF4. The source architecture is built on meta and official safetensors are read one tensor at
a time into their final owned CUDA representation. This avoids loading a dense base on the GPU.
Both precision arms store the remaining backbone in BF16; selected trainable action-head/projector
parameters and ordinary rank-32/alpha-64 LoRA are FP32, with FP32 fused AdamW moments. Task state/action inputs and training noise/time math retain native FP32 precision; the
backbone embedding dtype does not change the native action noise schedule. Native BF16
autocast also applies to adapter GEMMs; FP32 parameter storage does not imply FP32 GEMM execution.
TF32 controls are explicitly enabled as in the upstream head-training path and recorded in identity.

`model.backbone_adaptation.mode` selects four distinct controls:

| Mode | Language weights | Trainable language adapters | Native action projector/DiT |
|---|---|---|---|
| `nf4_lora` | frozen NF4, nested quantization, uint8 storage, BF16 compute | ordinary FP32 LoRA | trained FP32 by default |
| `bf16_lora` | frozen BF16 | identical LoRA | identical policy |
| `nf4_frozen` | frozen NF4 | none | trained FP32 |
| `bf16_frozen` | frozen BF16 | none | trained FP32 |

`train_action_head=false` is permitted only with adapters. It is a separate learning recipe,
not the selected jointly adapted recipe. The old cached-feature PPO path is rejected for backbone
adaptation because its no-grad features bypass language adapters. The flow-BC forward retains the
original noise distribution, time sampling, velocity objective and masked denominator. No language
SFT surrogate is used. The matched head-only controls above use fused AdamW and a single decay group;
the older native official recipe uses unfused AdamW and HF decay groups, and remains a distinct
comparison rather than an interchangeable control.

Use pinned Isaac-GR00T source `51d4c89f72fda44cbf77285c6a8114b52676b8a1`, Transformers 4.57.6,
and bitsandbytes 0.49.2 for NF4. Construction verifies the upstream source digest and content-bound
checkpoint/processor identity. No missing-kernel, dense-weight or alternate-backend fallback occurs.
The shared [native action source profile](../../docker/posttrain/native_action/README.md)
reproduces the retained CUDA 12.8 package versions for both GR00T and Alpamayo without
changing Qwen's runtime:

```bash
python docker/posttrain/native_action/install.py \
  --uv /path/to/uv-0.11.29 --venv /durable/venvs/cosmos-native-action
source /durable/venvs/cosmos-native-action/native-action-env.sh
```

The initial native strategy is one CUDA process, BF16-capable hardware, no FSDP/DTensor/TP/PP/CP,
no gradient accumulation and no compilation. Only configurations with completed accelerator and
task evidence can be described as qualified. Other embodiments, model revisions and online PPO
are outside this initial integration.

```bash
export IMAGINAIRE_OUTPUT_ROOT=/durable/path/gr00t-runs
export PYTHONHASHSEED=17 CUBLAS_WORKSPACE_CONFIG=:4096:8

torchrun --standalone --nproc-per-node=1 \
  -m cosmos_framework.posttrain.entrypoints.action_sft \
  --config cosmos_framework/posttrain/configs/gr00t_n1d7_libero_qlora.py -- \
  model.checkpoint_path=/staged/GR00T-N1.7-LIBERO/libero_spatial \
  model.backbone_adaptation.backbone_path=/staged/Cosmos-Reason2-2B-metadata \
  model.backbone_adaptation.mode=nf4_lora \
  data.train_path=/prepared/gr00t/train.jsonl \
  data.validation_path=/prepared/gr00t/validation.jsonl \
  rl.max_steps=100 rl.evaluation_every=25 rl.save_every=25 rl.seed=17
```

Prepared JSONL rows require `id`, independent `episode_id`, `observation`, `actions` and `media`.
The loader binds every row and image/packed-media payload to the data cursor identity, rejects
train/validation episode, row or media overlap, and detects source mutation. It does not establish
independence between semantically related episodes; provenance screening remains necessary.

The entrypoint uses the existing `LLMTrainer`, flow-BC objective, optimizer/scheduler and DCP
publication. Resident evaluation preserves all RNG streams and mixed module modes, uses a separate
validation traversal and leaves the training cursor unchanged, including when scoring raises.
Each claimed target is checked after checkpoint publication/readback. Set target metric/threshold
before measuring outcomes; no target is inferred from candidate performance. Resumed runs explicitly
report process-only timing and do not reconstruct missing uninterrupted clocks.

For a deliberate recovery check, keep `rl.max_steps` fixed and first set `rl.pause_after_step=50`.
Rerun against the same native run identity with `rl.pause_after_step=0` and `rl.resume=auto`.
Only committed optimizer boundaries are resumable; a mid-update interruption restarts from the last
committed checkpoint. Optimizer, scheduler, RNG, data cursor and evaluation history are restored.
No simulator state is captured, so this is not a claim of exact online rollout continuation.

Every checkpoint/portable policy includes all trained action-head/projector tensors alongside LoRA,
with source, model, quantization, processor/statistics, embodiment, action settings, resolved targets,
implementation hashes and dtype identities. Portable reload reconstructs the same quantized base.
Use a new output directory for a fresh-process native action evaluation:

```bash
torchrun --standalone --nproc-per-node=1 \
  -m cosmos_framework.posttrain.entrypoints.action_eval \
  --config cosmos_framework/posttrain/configs/gr00t_n1d7_libero_qlora.py \
  --checkpoint /run/checkpoints/iter_000100 \
  --output /durable/path/fresh-action-evaluation -- \
  model.checkpoint_path=/staged/GR00T-N1.7-LIBERO/libero_spatial \
  model.backbone_adaptation.backbone_path=/staged/Cosmos-Reason2-2B-metadata \
  data.validation_path=/prepared/gr00t/validation.jsonl
```

The same command accepts the portable path printed in `completion.json`. It verifies the commit,
parameter inventory and complete numerical identity before loading trained components. It reports
masked normalized-action MSE, per-component decoded environment-action MAE and native teacher-forced
flow loss separately. Prediction never receives demonstration actions as RTC conditioning. Generic
HF language-model loading and serving acceleration are not established by this portable loader.

The native dense export path separately reconstructs `dequantize(Q(W)) + (alpha/rank) B A` with
FP32 CPU accumulation and declared BF16 or FP32 merged-language storage. Non-target tensors retain
their native dtype, including FP32 trained action components. Standard indexed shards have a
configurable payload limit (512 MiB by default; a larger individual tensor gets its own shard).
The commit-last manifest includes per-tensor/file digests and copied native processor, embodiment,
statistics and backbone-tokenizer metadata. Fresh dense reconstruction needs no original base weights.

```bash
# Set these to the exact installed build identities, not guessed values.
export COSMOS_TRAINING_CODE_REVISION=<40-character-code-revision>
export COSMOS_TRAINING_WHEEL_SHA256=<64-character-wheel-sha256>
torchrun --standalone --nproc-per-node=1 \
  -m cosmos_framework.posttrain.checkpoint.tools.export_family_dense \
  --config cosmos_framework/posttrain/configs/gr00t_n1d7_libero_qlora.py \
  --checkpoint /run/checkpoints/iter_000100 --output /durable/gr00t-dense \
  --merged-backbone-dtype bfloat16 --max-shard-size-bytes 536870912 \
  --override model.checkpoint_path=/staged/GR00T-N1.7-LIBERO/libero_spatial \
  --override model.backbone_adaptation.backbone_path=/staged/Cosmos-Reason2-2B-metadata

torchrun --standalone --nproc-per-node=1 \
  -m cosmos_framework.posttrain.entrypoints.action_eval \
  --config cosmos_framework/posttrain/configs/gr00t_n1d7_libero_qlora.py --dense \
  --checkpoint /durable/gr00t-dense --output /durable/dense-evaluation -- \
  data.validation_path=/prepared/gr00t/validation.jsonl
```

This effective-base dense artifact is a different numerical execution from separate BF16-autocast
LoRA arithmetic. A further BF16 merge cast also changes the numerical model. Dense tensor checks,
fresh native task-output tolerances and task-quality evaluation are separate qualification gates;
implementation or file conversion alone does not establish their pass. Merging onto the untouched
original BF16 base, requantization, generic serving and deployment-latency improvements remain outside
this export contract. Dense artifacts cannot resume training.

An observation-only native rollout adapter and `entrypoints.gr00t_rollout` also connect a loaded
portable or dense policy to the existing isolated LIBERO simulator gateway. The separate
`gr00t_qlora_spatial_8_v1` profile executes eight supervised actions, allows 256 control steps and
keeps the native 15-step reset; legacy spatial's 16-action protocol is unchanged. Canonical simulator
images are already oriented correctly, and the simulator alone binarizes/inverts native gripper
commands. The policy adapter performs neither conversion a second time.

```bash
# cases.json is a prospectively fixed list of {suite, task_id, init_state_id} objects.
# The simulator interpreter must satisfy the existing pinned LIBERO runtime and asset preflight.
torchrun --standalone --nproc-per-node=1 \
  -m cosmos_framework.posttrain.entrypoints.gr00t_rollout \
  --config cosmos_framework/posttrain/configs/gr00t_n1d7_libero_qlora.py \
  --checkpoint /run/checkpoints/iter_000100 \
  --sim-python /verified/libero-venv/bin/python --cases /protocol/cases.json \
  --seed 17 --output /durable/rollout-seed17 -- \
  model.checkpoint_path=/staged/GR00T-N1.7-LIBERO/libero_spatial \
  model.backbone_adaptation.backbone_path=/staged/Cosmos-Reason2-2B-metadata
```

For dense policy inference, add `--dense` and point `--checkpoint` to the effective-base dense
artifact. The training runtime does not itself contain the pinned simulator stack. Until an actual
simulator campaign passes, this connection and its CPU convention tests do not establish rollout
success, simulator compatibility or deployment latency. The artifact-output probe in
`model/gr00t/qualification.py` separately captures actual normalized and decoded actions, native
flow loss and processed-input digests; its comparison requires explicit predeclared tolerances.

The prospective `gr00t-development-v1` study compares `nf4_lora`, `bf16_lora` and `bf16_frozen`
with the same FP32 action-head policy, fused AdamW, learning rate `1e-4`, constant schedule,
effective batch one and training seeds 17/42/123. It fixes 300 committed updates, full resident
validation at updates 0/100/200/300, and publication at updates 100/200/300 (also update zero if it
already meets the target); there is no quality-based early stop.
Arm order rotates across seeds. The outcome-blind cohort contains all ten spatial tasks:
eight training episodes per task with sixteen time-spaced windows (1,280 rows), and four validation
episodes per task with four windows (160 rows). Its manifest SHA256 is
`c726e37725c2d4be88efb17e8f96b956d018a07db9f72bde1f6825084392dd00`.
All 2,880 source images are RGB 256 × 256 with verified PNG payloads/content hashes; processed
model shapes must still be measured. Episode, row and media-content splits are disjoint.
These previously exposed episodes provide development evidence, not independent confirmation.

Before either LoRA candidate runs, use native `action_eval --base` with `mode=bf16_frozen` and
`rl.evaluation_seed=123` on the locked validation cohort. Freeze the offline action-quality target
as `0.90 * M0`, where `M0` is that original source model's `normalized_action_mse`; record the
reference evaluation, identities and threshold before admission. Training uses
`rl.target_metric=normalized_action_mse`, the frozen `rl.target_threshold`, and
`rl.stop_on_target=false`. Count target time through verified checkpoint publication, retain
nonattainment at update 300, and charge the reference scoring cost explicitly. This is time to an
offline action-error target, not time to closed-loop success. Equal update counts are a controlled
work comparison, not proof of equal-compute quality or faster learning.

Prospective preservation guards are a one-sided 95% upper bound of 5% relative normalized action-MSE
increase and two percentage points of closed-loop success loss against the matched BF16 LoRA arm.
The first permits at most 2.47% relative RMSE amplitude increase; the second permits at most one
additional failure per fifty episodes. Neither is inherited from WTS answer accuracy. Weight tasks
equally, then episodes/windows equally. Use a paired crossed bootstrap of training seeds and tasks,
with episode resamples shared across all seeds and precision arms; repeated frames or scoring seeds
are not independent tasks. Three training seeds and ten task clusters may be insufficient to establish
these bounds. The separate planned rollout comparison uses ten fixed initial states per task under
the eight-action profile and must pass actual simulator preflight first. Improvement, preservation,
and useful adaptation over the source are separate claims; retain all failed trials.

Before inspecting dense-conversion outputs, its numerical gates are fixed at maximum absolute
error 0.02 on the valid 16 × 7 normalized actions and decoded controller commands, plus both
0.002 absolute and 2% relative masked flow-loss error (relative denominator floor `1e-6`).
The full 40 × 132 latent output is shape-checked and its unused-slot error reported separately.
Every decoded gripper command must preserve `sign(2*g-1)` exactly, and trained auxiliary tensors
and processed-input identities must match exactly. At the pinned OSC_POSE scales, a 0.02 command
difference corresponds to at most 1 mm position-command or 0.01 rad orientation-command difference
per step; this is a numerical budget, not a safety or rollout-quality claim. Same-quantized-base
portable reload retains zero numerical tolerances. A failed conversion gate is retained as a failure.

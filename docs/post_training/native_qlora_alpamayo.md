# Alpamayo-R1 native trajectory QLoRA

This integration connects Alpamayo's supervised trajectory flow loss to the native
Cosmos `action_sft` trainer. It does not replace the separate AlpaGym replay-GRPO
entrypoint, and it does not establish closed-loop driving or real-world safety.
Accelerator and real-data qualification results must accompany a supported recipe;
the CPU tests below establish the integration contract only.

The first configuration binds `nvidia/Alpamayo-R1-10B` revision
`dd4a24cacefc9a6477a6dfc7354de2443401409d`, its public model source
`NVlabs/alpamayo@11a0e01c13a5622377c45ee37d653351453ec43b`, Transformers 4.57.6,
and the Qwen3-VL-8B processor at `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
Use the family runtime, not the Qwen qualification runtime using Transformers 5.8.
The shared [native action source profile](../../docker/posttrain/native_action/README.md)
pins the retained CUDA 12.8 dependency closure for both Alpamayo and GR00T:

```bash
python docker/posttrain/native_action/install.py \
  --uv /path/to/uv-0.11.29 --venv /durable/venvs/cosmos-native-action
source /durable/venvs/cosmos-native-action/native-action-env.sh
```

NF4 additionally requires bitsandbytes 0.49.2 and CUDA BF16 kernels. Checkpoint and
processor bytes, source implementation, task configuration, and resolved trainables
are recorded in native artifact metadata.

The explicit allowlist contains 247 ordinary language Linear weights: seven
attention/MLP projections in layers 0–34, and only K/V projections in layer 35.
The action expert consumes prefix K/V; the final language layer's Q/O/MLP output
and final normalization do not reach this loss. Those weights, vision/projectors,
embeddings, output head, and language normalization stay frozen BF16. NF4 uses
nested quantization, uint8 packed storage and BF16 computation. LoRA uses rank 32,
alpha 64, FP32 parameters; native BF16 autocast applies to adapter GEMMs. The native expert and
input/output action projections are explicitly trained in FP32 by default. Their
parameters and optimizer state are material memory costs, and all are included
in checkpoint and portable artifacts.

`model.backbone_adaptation.mode=bf16_lora` is the matched precision control.
`nf4_frozen`/`bf16_frozen` train only the explicitly selected native task head;
they are distinct controls and are not backbone QLoRA. `train_action_head=false`
is permitted only with trainable language adapters. No mode changes the task loss.

The model loads its complete approximately 22 GB BF16 source on the host, then
moves/quantizes one matrix at a time; no complete dense base is placed on a GPU
before quantization. Host loading peaks must be measured. No `device_map=auto`,
blanket post-quantization cast, distributed sharding, or CPU offload is supported.
Use one process under `torchrun`. Full activation checkpointing replays the whole
prefix/expert flow graph with a fresh mutable KV cache. Generic per-language-block
checkpointing is deliberately absent because its mutable-cache gradient path is
not equivalent. The configured alternatives are `none` and vision-only checkpointing;
only measured geometries/checkpointing policies may be advertised as qualified.

PhysicalAI input rows preserve four cameras in native order `[0,1,2,6]`, four
frames per camera, 16 history poses, and 64 future poses at 0.1-second intervals.
Positions are meters in the vehicle frame at the current time. Camera calibration,
frame times, media contents, and episode identities are checked. The official
processor renders the driving prefix with empty reasoning; no text-answer loss is
substituted. The native action space transforms future poses into normalized
acceleration/curvature and owns the flow target and noise/time schedule.

Evaluation reports native flow loss and open-loop planar ADE/FDE in meters at
1, 3, and 6.4 seconds. Trajectories use the checkpoint's action denormalization and
ten Euler flow steps. These are open-loop metrics, not simulator success. The
resident evaluator restores training RNG and mixed module modes, does not consume
the training cursor, and associates target crossings with committed checkpoints.

Stage immutable checkpoints and content-bound JSONL rows first, then run:

```bash
torchrun --standalone --nproc-per-node=1 \
  -m cosmos_framework.posttrain.entrypoints.action_sft \
  --config cosmos_framework/posttrain/configs/sft_alpamayo_qlora.py -- \
  model.checkpoint_path=/staged/Alpamayo-R1-10B-dd4a24c \
  model.processor_path=/staged/Qwen3-VL-8B-processor-0c351dd \
  data.train_path=/data/train.jsonl data.validation_path=/data/validation.jsonl
```

Use the existing native output/resume settings; resumable DCP includes the full
trainable expert/projections/adapters, AdamW state, scheduler/update clock, RNG,
data cursor and evaluation history. Portable export contains every trained
parameter and requires the same recorded quantized base. It is not a complete
standalone BF16 policy export. Applying adapters to the original BF16 backbone is
a different numerical model. A fresh native portable/checkpoint evaluation uses
`posttrain.entrypoints.action_eval` with `--config`, `--checkpoint`, `--output`
and the same model/data overrides.

The separate `posttrain.checkpoint.tools.export_family_dense` command reconstructs
`dequantize(Q(W)) + (alpha/rank) BA` with FP32 accumulation. Specify
`--merged-backbone-dtype=bfloat16` or `float32`; all other parameters and buffers
retain their native dtypes, including the complete trained FP32 expert/projections.
It copies pinned config/processor sidecars, writes deterministic size-bounded
safetensors shards, and commits the verified manifest last. The native Alpamayo
meta constructor streams those values through `action_eval --dense`; original
base-weight files are unnecessary for dense reload. Supply the exact export code
revision/wheel digest through `COSMOS_TRAINING_CODE_REVISION` and
`COSMOS_TRAINING_WHEEL_SHA256`. This dense route is inference-only and does not
replace resumable checkpoints. BF16 casting changes the model's numerical
arithmetic; implementation and tiny-graph checks do not establish full-model task
parity or deployment latency.

For the first full-model conversion check, the prospectively fixed budgets are:
controlled normalized flow velocity RMSE ≤0.01 and maximum absolute error ≤0.05;
flow-loss difference ≤max(1e-5, 1% of reference); paired-noise ten-step trajectory
deviation ≤0.10 m ADE and ≤0.25 m final displacement at 6.4 seconds; and each
actual open-loop ADE/FDE change ≤max(0.02 m, 1% of reference). Processed inputs
must match exactly, and all values must be finite. The optional
`model.alpamayo.qualification.capture_outputs`/`compare_outputs` helpers record
these outputs and every gate. Two development rows can validate conversion
behavior only; they cannot establish recipe quality, closed-loop driving, or
inference speed. A failed budget remains a failed conversion decision.

The implementation reuses reviewed task forward/data contracts from Cosmos
`48f6aa082b38a598a1fe59735e36a543f8834a02` (wheel SHA256
`fea15c5db109d63b1184eb90327b746260c5192e31ce1d4f7b79fc85c5b8389b`),
without importing its separate adapter framework. CPU contract tests cover exact
allowlist exclusions, units/media validation and trajectory metrics. The actual
tiny public Alpamayo tests run three native flow updates with full checkpointing
both enabled and disabled, require real adapter/head gradients and updates,
verify optimizer membership and frozen-base integrity, and reconstruct trained
state. The same test file includes NF4 cases that require real CUDA kernels under
a one-rank `torchrun`; skipped NF4 tests provide no quantization evidence.

## Prospective development comparison

The `alpamayo-development-v1` study is fixed before candidate outcomes. Its
selection digest is
`78c6643ea8bfb3ff36e30e7e635a1c167052abf973123a90fa27411b3b2f3080`.
It admits only the official TRAIN partition of PhysicalAI revision
`2ac2e9c056e0c8426face5bfad64b0961a295b28`: 1,024 training clips from 128
coarse blocks and 256 development clips from 32 different blocks, eight clips
per block. A block is `(platform_class, country, month)`. All seven blocks
containing the ten engineering-v2 clips are excluded. A fixed SHA256 rule
assigns whole blocks to splits and orders blocks/clips without model outcomes;
the selection and cohort manifests retain the exact rule, metadata hashes,
source receipts, and verified row/media hashes. TEST and ALPASIM are excluded.
Admission manifest
`2272c9fae7cd50eb13adaa8fd3bdaaa8467b91621c4983441b201b1faa97c810`
binds 21,760 verified files (32.51 GB), including 20,480 source images at
1920×1080. All 1,280 rows pass the native camera/timestamp/calibration contract
and finite 16-history/64-future pose checks; cross-split media-hash overlap is zero.
This is data admission, not measured processed shapes or task-quality evidence.

These are development blocks, not verified independent recording families.
The pinned metadata has no year, session ID, or parent-event mapping. Combining
years makes blocks coarser, but a session can cross a month or country boundary.
Thus block-disjoint data cannot establish recording independence. The selected
training/development sets cover 22/17 countries and both platform classes, with
eight examples per block; they do not estimate the full dataset's natural clip
distribution. Official TEST remains reserved until the candidate and confirmation
protocol are frozen and recording/event provenance is available.

Compare `nf4_lora`, `bf16_lora`, and `bf16_frozen` with the same native trajectory
objective, FP32 expert/action projections, fused AdamW state, four cameras by
four frames, rank 32/alpha 64 where applicable, batch one, and full activation
checkpointing. The first pair isolates backbone quantization; the third measures
adaptation relative to the native expert-only policy. It is not numerically
identical to public Stage-2 training: that recipe loads BF16 expert parameters,
whereas these matched controls retain FP32 trainables. No optimizer-precision
change or hyperparameter search is included in this comparison.

Use paired training seeds **17, 42, 123**, evaluation seed **123**, **512 updates**,
and full development evaluation/checkpoint publication every **128 updates**.
All arms use LR **1e-4**, **100** warmup steps and a cosine floor of **1e-6**;
these values follow the pinned public
[Stage-2 base configuration](https://github.com/NVlabs/alpamayo-recipes/blob/1d99fc50370c96637157455da386323a44663c9d/recipes/alpamayo1_sft/configs/sft_base.yaml).
Set `rl.optimizer.grad_clip_norm=1.0`, `weight_decay=0.0`,
`betas=[0.9,0.999]`, and `eps=1e-8` explicitly. These match the inherited pinned
Transformers 4.57.6 `TrainingArguments` defaults; its Torch ≥2.8 optimizer default
is fused AdamW. The earlier ten-update engineering runs disabled clipping and
remain distinct evidence. No production default or historical run is rewritten.
The native schedule uses `rl.lr_schedule.kind=cosine`, `f_start=0.01`,
`f_min=0.01`, `cycle_steps=512`; its nonzero initial LR is an explicit difference
from Hugging Face's warmup. The bounded 512-update horizon is not the public
recipe's three epochs. Counterbalance arm order across seeds, record placement,
and give every arm identical resources and shared execution optimizations.

The primary task endpoint is single-sample, ten-Euler-step **planar ADE at 6.4s
in meters**; FDE at 6.4s is a required guard. Retain flow loss and ADE/FDE at
1s/3s as diagnostics. Before candidate training, measure the frozen BF16 policy's
development ADE, `A0`, and freeze the common time-to-quality target at
`0.95 * A0`. Count a crossing only after the associated checkpoint is published
and verified. A target crossing alone does not pass the quality guards. Continue
all runs to update 512 for the matched-work comparison; nonattainment remains
unreached, never omitted. For matched elapsed compute, use the earlier final
checkpoint-publication time of each NF4/BF16 seed pair, then compare each arm's
last published evaluation at or before that budget. This rule does not inspect
quality or interpolate unmeasured checkpoints; report the checkpoint lag.

Prospective noninferiority requires the one-sided 95% upper bounds for ADE
degradation to be at most **both 5% and 0.10m**, and for FDE degradation to be at
most **both 5% and 0.25m**, versus matched BF16 LoRA. These proposed development
budgets limit relative error while capping absolute displacement; they are not
official project safety requirements, WTS margins, or the artifact-conversion
tolerances above. Quality improvement additionally requires the ADE-difference
upper bound below zero. Apply these checks separately to the final update-512
comparison and the matched elapsed-compute comparison, never to a retrospectively
chosen best checkpoint. All guards must pass; do not choose a favorable endpoint.
Carry these margins unchanged into independent confirmation.

Average clips within each block, then weight blocks and paired seeds equally.
For development uncertainty, use 10,000 paired crossed-bootstrap replicates
(bootstrap seed 70241): resample paired seeds and whole blocks, using the same
block draw across seed pairs. Report per-seed results and the 95th percentile
upper bounds for relative and absolute differences. Three training seeds and
unverified coarse blocks do not establish a population recipe claim; these
intervals remain exploratory. Final confirmation needs genuine recording/event
families and a variance pilot to determine adequate seed/group counts.

Record startup, preparation, training, evaluation, checkpoint, export/reload,
GPU allocated/reserved/process peaks, and host memory separately. Time to target
includes startup through verified checkpoint publication. Target calibration and
search costs are additional reported costs, not hidden speedups. The nine-run
study contains 4,608 updates and 11,520 evaluation rows before separate baseline
and artifact checks. Admit runs only after the engineering measurements establish
their duration and durable capacity; if correctness, finite loss, or storage
checks fail, stop the affected attempt and preserve it. Do not change the target,
margin, seed list, or one arm's horizon in response to quality outcomes.

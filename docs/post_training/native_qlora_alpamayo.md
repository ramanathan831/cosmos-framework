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
NF4 additionally requires bitsandbytes 0.49.2 and CUDA BF16 kernels. Checkpoint and
processor bytes, source implementation, task configuration, and resolved trainables
are recorded in native artifact metadata.

The explicit allowlist contains 247 ordinary language Linear weights: seven
attention/MLP projections in layers 0–34, and only K/V projections in layer 35.
The action expert consumes prefix K/V; the final language layer's Q/O/MLP output
and final normalization do not reach this loss. Those weights, vision/projectors,
embeddings, output head, and language normalization stay frozen BF16. NF4 uses
nested quantization, uint8 packed storage and BF16 computation. LoRA uses rank 32,
alpha 64, FP32 parameters and FP32 adapter computation. The native expert and
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
a different numerical model; family dense export needs its own verified conversion.
A fresh native artifact evaluation uses `posttrain.entrypoints.action_eval` with
`--config`, `--checkpoint`, `--output` and the same model/data overrides.

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

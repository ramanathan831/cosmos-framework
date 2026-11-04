Title: Add native SFT eight-bit AdamW with managed-state checkpoint recovery

Native posttrain SFT can select `adamw_bnb_8bit` or `paged_adamw_8bit` with ordinary
CUDA FP32 trainables, independently of model-weight quantization. The factory
preserves trainability and LR/decay groups, resolves exact-name FP32-state
exceptions, and reports actual allocated state. Both dependency environments pin
optional bitsandbytes 0.49.2; ordinary optimizers keep lazy dependency behavior.

Consolidated checkpoints preserve uint8 moments, maps, block scales, counters and
group identity without generic dtype casting. Exact load validates backend,
version, precision, topology and mapping, then reconstructs owned managed buffers
without an optimizer update. The pinned upstream buffer lacks a free owner;
Cosmos retains a cudaFree owner until its last storage alias dies. Quantization
and update mathematics remain upstream. FSDP/DTensor, DCP, low-precision trainables
without masters, sparse gradients and optimizer offload fail explicitly.

A separate HF-builder fix enables the existing Transformers-5 SFT environment's
native unsharded Qwen3-VL forward. Native SFT also releases completed gradients
before validation: a pressure canary exposed an OOM after two successful paged
updates while dead gradients remained resident. Cleanup rejects open accumulation
and preserves the direct evaluator's state contract.

Validation includes real CUDA kernels, fresh-process bitwise continuation,
accumulation/clipping/skips, LR groups, native checkpoint transactions, managed
allocation lifetime, CLI serialization and unchanged high-precision regressions.
The focused suite passes 42 tests; the validation cleanup passes 55 regressions
with one Transformers-5-only test skipped locally. Existing HF FSDP regressions
pass under two-rank torchrun; quantized distributed support is not advertised.
The disposable BF16/NF4 integration remains separate from the feature branch.

The requested product release is not qualified. The mainline 1.54B-trainable
video workload allocates 12,348,572,000 baseline state bytes versus 3,135,784,960
eight-bit state/metadata bytes (74.6% less). All nine 40-update mainline runs
completed without skipped updates and passed the 1% development-NLL bound.
Nonpaged/paged throughput ratios are 0.966 [0.827, 1.105] and 0.973 [0.842, 1.105]
(95% paired-seed intervals), so neither speed gate passes. Equal-budget quality
gain is also unproven. Sampled process GPU peaks average 58.26 GiB fused, 50.18 GiB
nonpaged and 47.30 GiB paged; these samples are lower bounds on full-run peaks and do not establish total
paged residency. Device-wide telemetry is incomplete across the three seeds. Adapter-only savings are about 248.45 MiB, excluding
frozen base weights. Kernel probes remain slower than fused AdamW. Native capacity
canaries identify a useful no-recomputation configuration, but two updates do not
establish a speed or quality gain. Three-seed terminal results, confidence bounds,
recovery receipts and cost accounting are reported in the linked evidence report.
All twelve BF16/NF4 × fused/eight-bit adapter cells completed, and both native
8→16-update recovery checks pass bitwise. Final task-NLL ratio upper confidence
bounds are 1.02163 (BF16) and 1.03853 (NF4), exceeding the fixed 1.01 limit despite
identical paired task accuracy. Recording-group adaptation bounds, speed and
equal-budget quality-gain gates also fail. No inference-latency or broad
distributed claim follows from these memory results.

The full-model paged canary completes two substantive updates and validation
with 8.77B FP32 trainables where the nonpaged control OOMs allocating moments.
This is a capacity result only. All owned jobs are terminal; charged usage is
7h20m38s including failures and the conservative local allowance, below eight GPU-hours.

The user approved reuse of the QLoRA data. Recording-connected train/development
and final-score groups are separated; historical inspection is disclosed. The
original overlap failure, pruned-midpoint recovery failure and OOM attempts are
retained. Correctness-only reference replays never replace quality/timing trials.

See [implementation, qualification and limits](../../quantized_optimizers.md),
[reproduction commands](../../../../tools/quantized_optimizers/README.md), and raw
machine-readable evidence beside this file. PR #334 remains closed. This is a
prepared description; no PR has been published or merged.

# Add native single-device eight-bit AdamW and paged optimizer-state recovery

Ordinary FP32 trainables can now select `adamw_bnb_8bit` or
`paged_adamw_8bit` through the native posttrain configuration/factory. The pinned
bitsandbytes backend reduces moment storage while keeping FP32 parameter updates.
The change is independent of model-weight quantization and QLoRA loading.

**Release qualification remains failed.** This contribution includes the measured
negative results and does not claim the requested product objective is achieved.
No PR has been published or merged; closed PR #334 remains untouched.

## Behavior

- Lazy optional `bitsandbytes==0.49.2` extra; existing optimizer paths need no import.
- Final trainable membership, canonical names, stable LR/decay groups, exact FP32
  state selectors, and allocated state/exception accounting.
- Explicit rejection of BF16/FP16 trainables without masters, DTensor/distributed
  strategies, DCP, optimizer offload and incompatible flags. No fallback or implicit
  local-tensor conversion. Supported numerical target: ordinary CUDA FP32 trainables,
  one GPU, consolidated checkpoints.
- Instance-owned overrides and managed-allocation ownership. Typed immutable CPU
  snapshots preserve quantization bytes/maps/scales; paged restoration reconstructs
  managed allocations without a hidden update. Exact resume validates backend,
  format, precision, topology and parameter-group identity.
- Native SFT recipes, early validation, strict determinism as an opt-in, and a
  verified optional prepared-input corpus. Normal raw-input/high-precision defaults
  retain their behavior. Completed gradients are released before native validation.

## Evidence

The original optimizer-isolation and BF16/NF4 adapter factorial evidence remains
sealed and separate from the independent-mainline deployment follow-up. The latter
uses ordinary Qwen3-VL-8B, 3.087B FP32 trainables, real eight-frame videos, matched
micro/effective batch four, a common cosine schedule, three seeds and three paired
process restarts per seed. Fused AdamW needs activation recomputation; eight-bit
AdamW uses its memory saving to remove it.

All 18 runs completed 40 updates with zero skips. All 72 repeated checkpoint-value
comparisons passed, and all seven fixed final score files completed. Each seed's
paired runs used the same allocated GPU and CPU placement. Source equivalence is
verified across the later scheduling repair. Failed allocations and original
training timings are retained.

| Measurement | Result |
|---|---|
| Actual optimizer state | 24,697,144,000 → 6,271,567,872 bytes, 74.61% saved |
| Native training throughput / fused | 1.329, 95% CI [1.304, 1.353] |
| Full launch/completion ratio | 1.136 [0.702, 1.570]; no proven end-to-end win |
| Launch time-to-quality ratio | 1.023 [0.243, 1.803]; fails fixed >1.05 lower bound |
| Cold-pipeline time-to-quality ratio | 0.993 [0.898, 1.088]; fails |
| Final task NLL / fused | 1.00618 [0.96976, 1.04260]; fails fixed ≤1.01 upper bound |
| Final task bootstrap NLL ratio | [0.98870, 1.02695]; also fails |
| Equal-budget quality gain | Not established; one launch/cold comparison is censored |

The deployment configuration spends the saved state memory on activations:
whole-device peaks are 73.06 versus 72.78 GiB, without a significant reduction.
Host PSS peaks increase from 6.17 to 10.66 GiB. Both optimizers score 50%, 50%,
56.25% task accuracy across seeds, versus 68.75% frozen; positive adaptation is
not established. Final groups are disjoint within this campaign but belong to the
approved historically inspected pool. Final task scores are at matched updates,
not equal-time stopping points. No quality-gain or inference-speed claim is made.

## Validation and remaining work

Existing real-CUDA tests cover state allocation, precision exceptions, finite and
skipped updates, LR groups, lazy states, managed ownership and fresh-process native
checkpoint continuation. The unchanged high-precision distributed regression and
the separate large adapter integration recovery tests retain their documented scopes.
The follow-up native CPU suite passed 73 tests with one optional skip; seven local
qualification-control tests pass. Changed Python files pass Ruff and format checks.

An extra large ordinary-dense 40→44 continuation attempt was canceled before its
first training update after finding a missing midpoint-retention override. The launcher is
corrected, but this additional full-workload check remains unqualified and must be
rerun. It is not replaced by checkpoint-repeat equality.

All 16 follow-up allocations are terminal: 8,632 GPU-seconds of a 9,000-second cap,
including failures, and 1,678 CPU-wall seconds of 1,800. Slurm rounding caused an
earlier seven-second excess in planned reservations; the audit is retained and
subsequent admission uses effective wall limits. Actual spending stayed below the
cap. The remaining 368 seconds cannot reserve a complete corrected recovery run.

Before release: complete that recovery check, meet the unchanged task-quality and
end-to-end timing gates, retain equal-time checkpoints for final quality scoring,
and obtain sealed uninspected recording/trajectory groups with annotations/media
and a content manifest. Further tuning must use development data.

See [implementation/support matrix and results](../../quantized_optimizers.md),
[commands](../../../../tools/quantized_optimizers/README.md),
[qualification status](qualification_summary.json), [analysis](summary.json),
[raw evidence](deployment_raw.json.gz), and [cost receipt](slurm_final_receipt.txt).

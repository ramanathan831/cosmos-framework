# Add native quantized AdamW with explicit precision and checkpoint contracts

Ordinary native SFT can select `adamw_bnb_8bit` or `paged_adamw_8bit` without model-weight quantization. The implementation uses pinned bitsandbytes 0.49.2, keeps eligible trainable parameters and updates in FP32, and preserves quantized or managed optimizer state through consolidated checkpoint recovery.

**The requested product qualification remains open.** Memory savings, native-loop throughput and positive task adaptation are demonstrated. The unchanged matched-quality and end-to-end speed confidence gates still fail. This contribution preserves those results rather than weakening its thresholds. This is a local PR draft; nothing has been pushed or published, and PR #334 stays closed.

## Behavior and scope

- Typed canonical names, CLI/resolved configuration, lazy optional dependency, early device/strategy/precision validation and no fallback. Existing optimizer paths do not need bitsandbytes.
- Final trainable membership, tied-parameter deduplication, stable LR/decay groups, exact FP32-state selectors, and actual allocated state/exception accounting. Trainables are ordinary contiguous CUDA FP32 parameters; forward compute is independently configurable.
- One CUDA GPU and consolidated same-identity recovery. BF16/FP16 trainables without masters, DTensor/distributed strategies, DCP, offload and incompatible flags fail explicitly. No untracked local tensor conversion or automatic embedding replacement.
- Instance-owned overrides and managed allocation ownership. Independent CPU snapshots retain uint8 moments, maps, scales, counters and lazy states. Restoration validates format/backend/version/policy/group identity and reconstructs paging without an optimizer update. Failure injection verifies rollback after partial copying or metadata failure. Full-model checkpoint publication remains synchronous.
- Runnable ordinary-model recipes and optional verified native prepared inputs. Shared HF loading transfers stored 16-bit tensors before CUDA widening; exact kernel/value and prediction parity checks protect the high-precision path. Selective activation checkpointing is explicit and part of exact-resume identity.

## Completed evidence

The first optimizer-isolation campaign and BF16/NF4 adapter factorial remain separate from mainline evidence. The new ordinary Qwen3-VL workload trains 3,087,142,912 FP32 elements with real video conditioning and effective batch four. The fused baseline uses full recomputation; eight-bit uses state savings to remove it. Smaller microbatches combined with selective recomputation were tested and failed capacity; all attempts remain recorded.

Dense uninterrupted 44-update versus fresh-process 40→44 continuation now passes for fused and nonpaged eight-bit: all live parameters, moments/scales/maps, scheduler, RNG, cursor and clocks match after every resumed update, including the first. This closes the prior large-workload recovery gap; checkpoint-repeat equality was not used as a substitute.

The audit corrected evaluation-only annotation footer contamination and made the training/evaluation relationship explicit. A shared development-selected mixed objective retains every substantive multi-question target and adds an original single-choice target per video. A shorter decay-horizon candidate worsened development quality and was rejected. Historical letter scoring is unchanged, including exact frozen parity on the earlier 16-question selection.

All 36 original outcome training runs and 37 fresh-process score files completed. Three paired seeds and three timing restarts per seed use fixed controls. The 128-question/32-family manifest is disjoint from repaired training/development but historically exposed; this is the authorized reused-pool comparison, not independent replication.

| Measurement | Result |
|---|---|
| Actual optimizer state | 24,697,144,000 → 6,271,567,872 bytes; 74.61% saved |
| Matched-15 task accuracy | Fused 76.82%, eight-bit 76.56%, frozen 59.375%; both adaptation bootstrap lower bounds positive |
| Matched task NLL ratio | 1.00458, 95% interval [0.96779, 1.04138]; fails upper ≤1.01 |
| Matched accuracy-drop upper bound | 1.381 percentage points; fails ≤1 point |
| Equal 60-second training-budget quality | Actual states 10/10/10 versus 10/12/11; no demonstrated NLL gain |
| Prepared product time-to-target ratio | 1.05885 [1.01560, 1.10210]; excludes parity but fails lower >1.05 |
| Composed cold ratio | 1.00868 [1.00307, 1.01428]; fails; only one full preparation measurement |
| Repeated native-loop throughput | 1.35568 [1.26611, 1.44525]; five-update target restarts, narrower-clock gain |

The original 390-second product endpoint remains censored because one run committed update 15 with required validation pending. Its complete native journal is retained. A qualification-wrapper correction completes scheduled validation before an advance budget notice while retaining immediate external-signal handling. Both arms/all three seeds are being repeated under the same budget, manifest and gates; original outcomes are unchanged. The separate native arithmetic diagnostic remains pending. These pending results must be appended before this draft is final.

Matched mean whole-device peaks are 72.58/71.80 GiB and process GPU 71.89/71.12 GiB. Training/checkpoint host PSS rises 5.76→10.71 GiB; full delivery including scoring peaks 33.38/33.68 GiB. State savings partly fund activations. Paging remains a separately measured capacity mode, with no sustained paging speed or task-quality claim. Optimizer training results imply no inference-latency gain.

## Validation and remaining requirements

Real CUDA coverage includes optional dependency behavior, trainable membership, state precision, LR groups, nonfinite/skipped updates, snapshot ownership, paged restoration and fresh-process native recovery. The pinned regression suite passes 49 tests, plus seven weight-loading checks; nine qualification-boundary controls cover scheduled validation and actual-signal preemption. Unchanged high-precision distributed and separate BF16/NF4 adapter recovery evidence retain their original scope.

Before product qualification, meet the unchanged matched-quality and end-to-end speed bounds and demonstrate equal-budget held-out quality improvement. Independent replication also needs suitable independent evidence; the user-authorized reused data remains valid for engineering progress. The finite ledger includes every failed allocation and reserves dependent scoring before exploratory work. A remaining allowance is not described as exhaustion or a passed release.

See the [support matrix and runnable native recipe](../../quantized_optimizers.md), [requirement-to-evidence table](requirements.md), [locked analysis](summary.json), [phase/memory accounting](phase_memory_summary.json), [raw evidence](outcomes_raw.json.gz), [resource ledger](allocation_ledger.json), and [Slurm commands](../../../../tools/quantized_optimizers/README.md).

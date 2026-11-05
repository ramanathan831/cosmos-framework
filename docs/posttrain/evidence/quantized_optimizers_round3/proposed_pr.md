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

The original 390-second product endpoint remains censored because one run committed update 15 with required validation pending. Its complete native journal is retained. A qualification-wrapper correction completes scheduled validation before an advance budget notice while retaining immediate external-signal handling. Both arms/all three seeds completed under the same budget, manifest and gates; original outcomes are unchanged. Every corrected state is eligible: fused updates 5/10/5, eight-bit 15/15/15. Product-budget accuracy improves 67.71%→76.56%, with paired gain interval [6.61,11.10] percentage points and recording bootstrap [2.08,15.63]. Mean answer NLL is worse, 0.8160→1.1599, and the declared NLL-gain gate fails. The accuracy result does not substitute for that requirement.

The native arithmetic diagnostic is complete: identical full-recomputation runs of fused, bitsandbytes FP32-state and eight-bit AdamW all reach 87.5% development accuracy at 15 updates. NLL is 0.522715/0.524023/0.519571. Cumulative update relative L2 differences are 2.24% for fused versus bitsandbytes FP32 state and 3.48% for bitsandbytes FP32 versus eight-bit, including gradient feedback. Large local K/V differences motivated a separately locked development precision screen; these diagnostics cannot pass the failed final quality gates.

The predeclared three-seed K/V precision screen completed and was rejected: all accuracies were unchanged, but mean development task NLL worsened 1.805%. It verified exactly 32 FP32-state exceptions and 7,072,679,936 tensor-state bytes (71.36% saving). Default precision and final results are unchanged. The development confidence audit identifies a shared wrong answer becoming almost certain in both arms as accuracy rises; this supports shared-recipe investigation, without proving a label or quantization defect.

Matched mean whole-device peaks are 72.58/71.80 GiB and process GPU 71.89/71.12 GiB. Training/checkpoint host PSS rises 5.76→10.71 GiB; full delivery including scoring peaks 33.38/33.68 GiB. State savings partly fund activations. Paging remains a separately measured capacity mode, with no sustained paging speed or task-quality claim. Optimizer training results imply no inference-latency gain.

## Validation and remaining requirements

Real CUDA coverage includes optional dependency behavior, trainable membership, state precision, LR groups, nonfinite/skipped updates, snapshot ownership, paged restoration and fresh-process native recovery. The pinned regression suite passes 49 tests, plus seven weight-loading checks; nine qualification-boundary controls cover scheduled validation and actual-signal preemption. Unchanged high-precision distributed and separate BF16/NF4 adapter recovery evidence retain their original scope.

Before product qualification, meet the unchanged matched-quality and end-to-end speed bounds and demonstrate equal-budget held-out quality improvement. Independent replication also needs suitable independent evidence; the user-authorized reused data remains valid for engineering progress. The finite ledger includes every failed allocation and reserves dependent scoring before exploratory work. All 37 owned allocations are terminal; the campaign charged 23,384 GPU-seconds and 7,015 CPU-wall seconds including failures/local checks. Remaining 5,416/185 seconds are insufficient to reserve the next complete comparison. The executable next intervention tests shared LR 7.5e-6 under the same 15-update/40-horizon recipe, with strict development screening and equal optimizer access. Conditional development plus complete three-seed final evaluation reserves 15,660 GPU-seconds (4.35 hours); three fresh full cold-preparation replications reserve 7,200 CPU-wall seconds. This candidate is not submitted and its outcome is unknown. Remaining allowance is not described as exhaustion or a passed release.

See the [support matrix and runnable native recipe](../../quantized_optimizers.md), [requirement-to-evidence table](requirements.md), [locked analysis](summary.json), [phase/memory accounting](phase_memory_summary.json), [raw evidence](outcomes_raw.json.gz), [resource ledger](allocation_ledger.json), and [Slurm commands](../../../../tools/quantized_optimizers/README.md).

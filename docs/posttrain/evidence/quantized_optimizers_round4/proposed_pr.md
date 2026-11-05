# Add native quantized AdamW with explicit precision and recovery contracts

Native SFT can select `adamw_bnb_8bit` or `paged_adamw_8bit` independently of
model-weight quantization. Pinned bitsandbytes 0.49.2 reduces eligible optimizer
state while FP32 trainable storage preserves the existing update contract.
Consolidated recovery retains quantized bytes, scales/maps, lazy state, counters,
parameter identity and paging ownership. Optional imports remain lazy; unsupported
precision, distribution, checkpoint and optimizer flags fail explicitly.

**Product performance qualification remains open.** State savings and positive
task adaptation are demonstrated. The unchanged matched-quality, equal-budget
NLL-gain and end-to-end speed confidence gates have not all passed. This is a local
PR draft; nothing is pushed, published or merged. PR #334 remains closed.

## Scope and validation

- Supported execution: one CUDA GPU, ordinary contiguous FP32 trainables, independently configured forward compute, consolidated same-identity recovery. BF16/FP16 trainables without master weights, DTensor/distributed and DCP/offload combinations are rejected. Paging is a capacity mode with no sustained speed/quality claim.
- Canonical typed configuration and CLI overrides, stable trainable groups/LR multipliers, tied-parameter deduplication, explicit FP32-state exceptions, actual allocated-state accounting and no silent optimizer fallback. Frozen base parameters never receive optimizer state.
- Independent CPU snapshots and transactional restoration retain quantized state and reconstruct managed allocations. Failure injection covers partial-copy/metadata failures. Dense fresh-process 40→44 continuation matches uninterrupted fused/eight-bit training after every resumed update, including the first, with live parameters, moments/scales/maps, scheduler, RNG, cursor and clocks checked.
- Existing pinned regressions 49, loading checks 7 and budget/validation-boundary checks 9 remain valid. New prepared-input integrity tests 16 and checkpoint-read controls 6 pass both locally and in the pinned environment; published-corpus integrity tests 6 pass locally. Real CUDA scoring checks exact parity on 128development observations per loading canary.
- Shared loading work retains complete source hashes, immutable checkpoint provenance and independent fresh-process reloads. Four-worker source verification passes its locked screen; overlap of checkpoint hashing and model restoration passes exact score/identity parity and reduces the combined phase 29.50% in a small counterbalanced canary. Concurrent timers are not summed. These are shared improvements, not new final optimizer speed evidence.

## Measured results and remaining gates

The ordinary Qwen3-VL workload trains 3,087,142,912 FP32 elements with real video
conditioning. The strongest demonstrated feasible fused baseline uses full
recomputation; eight-bit spends state savings to remove it. Failed selective
checkpointing/smaller-microbatch controls remain recorded. This deployment
comparison is distinct from the earlier isolation campaign, whose eight-bit
optimizer calls were slower. Separate BF16/NF4 adapter compatibility evidence
remains confined to the disposable integration checkout.

| Endpoint | Result |
|---|---|
| Optimizer state | 24,697,144,000→6,271,567,872 bytes; 74.61% saving |
| Matched 15-update task accuracy | Frozen 59.375%, fused 76.82%, eight-bit 76.56%; positive adaptation passes |
| Matched NLL guard | Ratio 1.004584[0.96779,1.04138]; upper≤1.01 fails |
| Matched accuracy guard | Drop upper 1.381 percentage points;≤1 point fails |
| Native-loop throughput | 1.35568[1.26611,1.44525], using saved memory to remove recomputation |
| Prepared end-to-end target speed | 1.05885[1.01560,1.10210]; lower>1.05 fails |
| Equal 60-second training budget | Actual states scored; no demonstrated NLL gain |
| Corrected 390-second product budget | Accuracy 67.71%→76.56%, NLL 0.8160→1.1599; NLL-gain gate fails. Updates fused 5/10/5 versus eight-bit 15/15/15 |

The shared mixed objective retains substantive multi-question supervision and
adds an original single-choice target per video with equal row weighting. The
latest three-seed 7.5e-6 LR screen improves mean task NLL 9.04%, but loses accuracy
in two seeds and worsens seed 43 substantive NLL 36.81%. It is rejected before any
eight-bit candidate or new final scoring. The prior shorter schedule and K/V
exceptions remain rejected. Calibration, restricted-choice accuracy and substantive
answer loss stay separate; no scoring/tie-breaking changes mask the failures.

Whole-device peaks remain 72.58/71.80 GiB, training/checkpoint host PSS 5.76/10.71 GiB,
and full-delivery host PSS 33.38/33.68 GiB. These are fused/eight-bit values; 74.61%
state saving is not total-memory saving. Loading canaries do not change these
historical measurements or establish inference-latency gains.

Two new full preparations pass exact replay in 2193/2166 allocated seconds. A third
times out after corpus publication; a 29-second full byte-verification continuation
recovers every payload/source identity. Its initial 2390-second combined cost includes the
failed allocation. A manifest hash/parse binding fix passes six integrity tests
and a 25-second full recheck; total third-pipeline cost is 2,415 seconds. All three corpora are exact, but only two uninterrupted timing
replications completed. The prespecified three-replication cold interval is withheld;
the historical single-preparation composed result is unchanged.

The 128-question/32-recording-family final pool is disjoint from repaired training
and development but historically exposed. This remains the user-authorized reused
benchmark, not independent replication. No final examples were rescored in this
additional campaign.

Additional charges are 1575 GPU-seconds and 6954 CPU-wall seconds, including timeout,
recovery and conservative local-check allowance. All eight owned allocations are
terminal. Remaining 14085/246 seconds are not an exhausted allowance; they cannot
reserve the next complete conditional comparison (15660 GPU-seconds) or a same-workflow
cold replacement (3000 CPU-seconds). The next executable LR hypothesis is 8.75e-6 with
unchanged guards; it is unsubmitted and its outcome is unknown.

See the [support matrix and native recipes](../../quantized_optimizers.md),
[requirement-to-evidence table](requirements.md), [commands](commands.md),
[raw artifact manifest](evidence_manifest.json) and [resource accounting](final_accounting.json).

# Add native quantized AdamW with explicit precision and recovery contracts

Native SFT can select `adamw_bnb_8bit` or `paged_adamw_8bit` independently of model
weight quantization. Pinned bitsandbytes 0.49.2 reduces eligible optimizer state
while ordinary FP32 trainables preserve parameter-update precision. Configuration,
CLI, stable groups/LR multipliers, optional imports, precision exceptions and
actual allocated-state reporting are integrated. Unsupported precision, strategy,
checkpoint and optimizer flags fail explicitly; no silent fallback occurs.

Consolidated recovery preserves quantized bytes, scales/maps, lazy state, counters,
parameter identity and paging ownership. Independent snapshots and staged restore
metadata retain transaction semantics. Supported engineering scope is one CUDA
GPU with ordinary contiguous FP32 trainables and independently configured forward
compute. BF16/FP16 trainables without masters, DTensor/distributed and unsupported
checkpoint/offload combinations are rejected. Paging is a capacity mode, not a
qualified speed improvement.

**Product performance qualification remains open.** The contribution implements
and tests the stated engineering scope; unchanged quality and end-to-end outcome
requirements have not all passed. This is a local draft: no push, publication or
merge, original QLoRA checkout untouched, PR #334 kept closed.

## Validation and measured behavior

- Dense fresh-process 40→44 recovery matches every resumed update's parameters,
  optimizer state, scheduler, RNG, data cursor and clocks for fused/nonpaged AdamW.
  Core paths are unchanged in the latest continuation; 49 pinned regressions,
  seven loading and nine budget/validation controls retain their scope.
- Shared loading preserves all source hashes and independent native reloads.
  Scoring now optionally binds `model.pt` to an independently pinned SHA-256 and
  publishes its read-success receipt only after final provenance succeeds.
  Fifteen local/pinned controls and a real CUDA checksum-failure test pass.
- Three counterbalanced loading repetitions preserve 384 development observations
  exactly. Verification/read/restore phases improve for both checkpoint arms.
  Whole-process speed ratio 1.2453 [0.9365,1.5541] fails its prospective confidence
  gate; these timings do not establish optimizer end-to-end or inference gains.

The ordinary Qwen3-VL workload trains 3,087,142,912 FP32 elements with real video.
The strongest demonstrated feasible fused baseline uses full recomputation;
eight-bit spends state savings to remove it. This deployment comparison differs
from optimizer isolation, where eight-bit optimizer calls were slower.

| Product endpoint | Evidence / remaining requirement |
|---|---|
| Optimizer state | 24,697,144,000→6,271,567,872 bytes ; 74.61% saving |
| Useful task adaptation | Prior frozen 59.375%, fused 76.82%, eight-bit 76.56% accuracy |
| Matched quality | NLL ratio 1.004584 [0.96779,1.04138] fails upper≤1.01; accuracy-drop upper 1.381 points fails ≤1 point |
| Native-loop throughput | Prior 1.35568× using saved state memory to remove recomputation |
| Prepared target delivery | Prior 1.05885 [1.01560,1.10210] fails lower>1.05 |
| Equal elapsed budgets | Actual 60-second training and 390-second product states scored; positive NLL-gain requirement fails |
| Cold delivery | Three exact full preparation timings 2193/2166/2121s; historical composed target ratio 1.00809 [1.00573,1.00965] still fails lower>1.05 |

The latest continuation executes nine development training runs. LR 8.75e-6 fails
accuracy/substantive guards. A single shared objective-weighting intervention,
original LR with `ce_exponent=0.95`, improves mean task NLL 4.83% fused  / 5.38%
eight-bit and preserves every seed's accuracy. Fused passes its screen, but
eight-bit substantive losses rise 1.51% / 1.33% in seeds 29/43, above the unchanged 1%
guard. The recipe is rejected before new final scoring. Default recipes remain
unchanged. A completed four-checkpoint reload diagnosis reproduces 128 native task observations
and localizes shared worsening text examples;
no failed guard is replaced by a different denominator or scoring convention.

Prior whole-device peaks 72.58/71.80GiB, training/checkpoint host PSS 5.76/10.71GiB,
and full-delivery host PSS 33.38/33.68GiB remain separate from state savings. New
loading trials peak at 34.99GiB process GPU and about 34 GiB host PSS. Dense results
do not establish equal benefits for LoRA/QLoRA. Adapter compatibility remains
confined to the separately tested mainline/disposable integration scopes.

The 128-question / 32-family final pool is disjoint from repaired training/development
but historically exposed. This is the user-authorized reused benchmark, not
independent replication. Earlier negative results and the preparation timeout,
recovery and all costs remain recorded. Current cold timing is a composition with
replicated preparation variability, not three directly measured cold GPU runs.

All eight new allocations are terminal. Combined round-four/continuation charges
are 1.4242 GPU allocation-hours and 2.5708 CPU-wall hours against 6.00/3.25-hour
ceilings. Headroom remains; the rejected candidate receives no final-comparison
spending. The next shared-weighting bracket is executable but unsubmitted.

See [support and native recipes](../../quantized_optimizers.md),
[requirements](requirements.md), [commands](commands.md),
[raw identities](evidence_manifest.json) and [accounting](final_accounting.json).

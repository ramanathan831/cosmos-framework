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

The preceding continuation rejected LR 8.75e-6 and exponent 0.95 under the unchanged
accuracy/substantive guards. Round six tests only the authorized exponent 0.975
at original LR 1e-5, retaining 15 updates/40-update horizon, data and precision.
Its three fused runs improve mean task NLL 3.4503%, with no per-seed accuracy loss,
but seed 43 substantive loss rises 1.2970669%, exceeding the 1% guard. The candidate
is rejected. No eight-bit candidate or new final comparison is admitted, and
removal of the aggregate resource ceilings does not authorize another recipe sweep. Original recipe defaults
remain unchanged. This fused-only result does not implicate state quantization.

The next continuation implements bounded, worker-local video-intermediate reuse
in the existing native preparation tool. Every target still receives an uncached
native replay, complete source/payload hashes and exact qualified-reference
comparison. The default remains disabled; the inspected scope is the pinned
Qwen3-VL CPU video processor with one local video per target. Three paired
15-target slice trials preserve 90 exact comparisons and improve complete Python
process time 1.24037× [1.14525, 1.37167]. Shared container startup is separate; this
is not a full cold or optimizer speed claim. The initial inner-timer screen and
failed peak-equality test assertion are retained. Corrected payload accounting
passes 19 pinned tests, including real wrong-hash/cached-pixel-corruption rejection,
and all 15 exact targets. Full 751-row/562-target repetitions are in progress.

Three fresh-process fused seed 43 checkpoint diagnostics reproduce 96 task
observations and all three native batch-four substantive losses exactly. The
observer retains inputs, masks, counts and native reductions. Both weighting
candidates worsen 10 of 16 substantive rows, with prominent brightness/traffic/weather
terms represented in training; this is a shared recipe trade-off, without an
established sole cause. Original/.975 saved data cursors are byte-identical and
reconstruct the same 60 training rows. No new final optimizer comparison is admitted.

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

Aggregate GPU/CPU/time ceilings were removed by the user's latest instruction;
all historical charges, failed allocations and new allocation costs remain
recorded. Scheduler routing, numerical gates, finite prospective experiment
scope and local-only publication remain unchanged. Product qualification is
open; the failed quality and speed requirements are not relabeled as completed.

See [support and native recipes](../../quantized_optimizers.md),
[requirements](requirements.md), [commands](commands.md),
[source identities](source_identity.json) and the retained raw evidence.

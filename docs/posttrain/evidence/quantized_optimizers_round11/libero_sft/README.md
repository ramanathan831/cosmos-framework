# Cosmos3-Nano LIBERO-10 action-policy SFT: FusedAdam vs adamw_bnb_8bit (round 11, workstream 2)

Native `action_policy_libero_nano` recipe (preset A), unchanged except for a justified
8x A100-80GB one-dimensional FSDP topology (64 x grad_accum 4, global batch 2048).
Protocol locked before comparison training (`protocol.json`, commit 17797bfa); the
single deferred value, the fixed target T = 0.92, was set from the development run
by the locked rule before any comparison result was read (commit 681a2c64).

## Integration (engineering) — passed

- The generator/VFM `OptimizersContainer` now selects `adamw_bnb_8bit` / `paged_adamw_8bit`
  through the shared FP32-update owner (`MasterWeightOptimizer` + bitsandbytes 0.49.2) with
  the identical lr/weight-decay grouping, frozen-key selection and LR multipliers; FP32
  FSDP2 shards are aliased as masters (no copy). Legacy optimizers are unchanged.
- `DistributedCheckpointer` saves the rank-local opaque payload under per-rank DCP keys and
  restores it with a strict non-dedup planner; staging validates topology, schema and
  master == restored model shard on every rank before commit. HSDP / async saves are refused.
- Tests: CPU legacy/rejection, CUDA grouping/scheduler/transactional DCP resume, and a
  two-rank FSDP2 fresh-process exact-resume gate through the production checkpointer
  (nonpaged and paged).

## Native evidence

| Check | Result |
|---|---|
| Optimizer tensor state per rank | FusedAdam 6.982 GB moments; 8-bit 1.746 GB moments + 0.027 GB scales = **74.6% less** (>= 60% required). No masters (aliased FP32 shards). Paged: 1.745 GB managed. |
| Peak device memory, 64x4 | FusedAdam ~61 GiB, 8-bit ~56 GiB |
| Native 128x2 microbatch | 8-bit trains and saves (80.1 GiB peak); FusedAdam fails its first checkpoint save (81.2 GiB) |
| Step time | ~46.2 s/iter for both arms and both microbatches (compute-bound) |
| Fresh-process resume 3 -> 6 | FusedAdam, 8-bit and paged pass (strict restore validation, finite updates, published checkpoint). The native recipe does not checkpoint the dataloader cursor, so post-resume losses are not replayed exactly in either arm. |

## Locked comparison (3 paired seeds, 500 episodes per checkpoint; `locked_analysis.json`)

| Criterion | Result | Pass |
|---|---|---|
| Quality preservation at 2000: lower 95% bound of (8-bit - fused) > -0.03 | 0.9573 vs 0.9733, diff -0.016, 95% CI [-0.0327, 0.000] | **No** (narrowly) |
| Equal-budget improvement: lower bound > 0 | same | **No** |
| Protected secondary: control steps <= +5% | 269.8 vs 267.0 | Yes |
| Time to T = 0.92: lower 95% bound of fused/8-bit > 1.05 | geometric mean 0.929, CI [0.80, 1.00] (fused seed 1 reached T at 1000, all others at 1250) | **No** |
| Optimizer-state reduction >= 60% | 74.6% | Yes |

Mean success by checkpoint (250..2000):
FusedAdam 0.185, 0.612, 0.812, 0.921, 0.958, 0.970, 0.962, 0.973;
8-bit 0.188, 0.631, 0.789, 0.900, 0.953, 0.969, 0.962, 0.957.
The arms track each other through 1750; the gap appears at the locked 2000 endpoint,
where 8-bit is lower in all three seeds. No endpoint or margin was changed after scoring.

## Speed route (closed; `deployment_variant_decision.json`)

The memory saving makes the native 128x2 microbatch feasible for 8-bit only, but step time
is unchanged (ratio 1.00), and lighter activation checkpointing OOMs for both arms. No
eight-bit-exclusive faster configuration exists on this node.

## Evaluation harness note

Policy servers run with `--no-guardrails` (new eval-only opt-in, default on): guardrails never
alter actions, LIBERO prompts are fixed benchmark strings, and the offline evaluation image
lacks the guardrail dependency/weights. Applied identically to every evaluation.

Cost: 1,943.8 GPU-hours including all failures (`final_accounting.json`).

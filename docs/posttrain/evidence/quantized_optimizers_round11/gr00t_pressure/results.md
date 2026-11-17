# Round 11 — GR00T fixed-pressure fixes (workstream 3)

Protocol (locked before results, with amendment 1): `protocol.json`.
Recipe, seed 17, 4×A100-80GB routing, 55 GiB co-resident reservation per GPU and the
round-10 checkpoint-16 sources are unchanged. No allocator environment policy was added.
Source: `b630c0c2` (`source-r11-pressure-b630c0c2.tar.gz`, sha256 `fad16b53…22eeca`).

## A. Nonpaged 8-bit OOM at update 20

**Diagnosis (round-10 records).** Not a leak: the learner's per-minute GPU peak was flat
(15.38–15.42 GiB) over updates 17–19. It was a capacity edge. At the OOM, the GPU held
55.48 GiB reservation + 7.08 GiB rollout + 1.15 GiB simulator + 15.06 GiB learner, out of
79.33 GiB, and a 32 MiB request failed with 0.77 GiB fragmented in the cache. Paged
survived because its 0.81 GB/rank of moments are evictable.

**Root causes fixed (shared, optimizer-independent, `55e680a9`):**
1. `JointPPOTrainer.train_batch` kept a device clone of every local model shard
   (~0.8 GB BF16 per rank) through all forward/backward passes, only to check that both
   parameter groups changed. It is now host-resident; update norms are bit-identical
   (frozen-reference test, atol 0).
2. The FP32-master optimizer kept consumed FP32 gradient staging (~1.6 GB/rank) until
   the next window's `zero_grad`. It is now released after the backend step, which
   bitsandbytes synchronizes.

Fail-before/pass-after: `cosmos_framework/posttrain/tests/test_joint_ppo_update_memory.py`
(4 of the 6 original tests fail on `84ff3143`, all 8 pass after; CUDA tests ran on a local A100 and in
the pinned gate).

| Run (55 GiB) | Job | Result | Peak alloc / reserved (GiB, max rank) | Fragmentation peak | Update s (mean excl. first) | Backend step ms |
|---|---|---|---|---|---|---|
| Round 10 nonpaged 16→32 | 38536844 | OOM at update 20 | 11.99 allocated at OOM | 0.77 at OOM | — | — |
| **Nonpaged 16→32** | 38655842 | **PASS**: 16 updates, ckpt 32 published, val 92/100 | 11.31 / 13.55 | 2.12 | 4.77 | 47.4 |
| Paged 16→32 (try 1) | 38656801 | FAIL: rank-0 illegal memory access, update 18 | 10.50 / 12.97 | 1.81 | — | — |
| **Paged 16→32 (rerun)** | 38657817 | **PASS**: 16 updates, ckpt 32 published, val 96/100 | 10.50 / 12.97 | 1.38 | 4.86 | 162.2 |
| Fused, fresh 0→32 (conditional control) | 38661542 | OOM at update 5 (round 10: update 2) | 13.55 / 13.98 | 2.10 | 8.90 | 9.6 |

Both passing runs restored a learner/collector record byte-identical to round 10's
verified restore of the same checkpoint.

Under this fixed pressure, nonpaged and paged 8-bit now both complete recovery, and fused
FP32-state AdamW does not. The fused–8-bit peak gap (2.24 GiB) matches the FP32-moment
excess (2.44 GB/rank). This is one seed and one node per run.

**Open:** paged try 1 failed with an unexplained `cudaErrorIllegalAddress`, surfacing in
update-18 staging after bitsandbytes' final synchronize. The fault came from a non-optimizer
kernel (forward/backward/clip/staging). The single predeclared identical rerun passed on
another node. One failure in two paged pressure runs (3 of 4 including round 10's passes)
is retained. Next step if it recurs: rerun with round-10 source v42 per amendment 1.

## B. Instrumented paged stall

Hypothesis B was falsified: disabling CUPTI UM fault tracing (`PROFILE_VARIANT=cuda_only`,
job 38660245) did not produce a completed run. It ran 6 updates (round 10: 3), including
the checkpoint-4 save and validation, then stopped making progress.

Evidence (`trace_activity_tail.py`): the exported trace spans only 431 s of the configured
600 s. All 4 learners and all 4 rollout workers issued kernels until the trace's last
0.5–6 s, and the native log stops at the same moment (15:15:35 local). No rank stopped
early, and no OOM or Python error was logged. The process tree stayed resident at a stable
host PSS until the profiler exited and srun tore the step down.

The stall therefore coincides with Nsight ending collection early. It does not coincide
with training, paging or rank divergence. Round 10 also reported "an injected process timed
out stopping recording". The exact cause inside the profiler is unresolved.

A separate launcher flaw: the helper exits when nsys returns, so a run longer than delay +
duration (~20 min) can never produce a completion receipt under `paging_profile.sbatch`.
The 8-update diagnostic shape exceeds that. No further profiler GPU trials were run (finite
plan).

**Paging overhead (uninstrumented CUDA-event receipts, matched 55 GiB):** backend step
162.2 ms paged vs 47.4 ms nonpaged, so +115 ms per optimizer call. In the profiled run,
bitsandbytes optimizer kernels take ~18 ms of device time per step per rank (0.109 s over ~6 captured steps). The update
phase is 4.86 s vs 4.77 s (+0.09 s), in ~28 s windows. Within the paged profile run,
backend-step time rose from ~75 ms (updates 2–4) to 172–218 ms after the checkpoint/
validation boundary, consistent with re-migrating evicted moments. These are single-run,
different-node observations. They are not a sustained-speed or time-to-quality claim.

## Accounting

| Job | Purpose | State | Elapsed s | GPUs | GPU-s |
|---|---|---|---|---|---|
| 38653802 | 2-GPU gate (151 + 36 tests, 12 receipts) | COMPLETED | 545 | 2 | 1,090 |
| 38655842 | nonpaged 55 GiB 16→32 | COMPLETED | 1,124 | 4 | 4,496 |
| 38656801 | paged 55 GiB 16→32 try 1 | FAILED | 1,255 | 4 | 5,020 |
| 38657817 | paged 55 GiB 16→32 rerun | COMPLETED | 1,159 | 4 | 4,636 |
| 38660245 | paged cuda_only Nsight | FAILED | 1,516 | 4 | 6,064 |
| 38661542 | fused 55 GiB fresh control | FAILED | 1,932 | 4 | 7,728 |
| 38661439 | trace analysis (CPU) | COMPLETED | 39 | 0 | 0 |

Total: **29,034 GPU allocation-seconds (8.07 GPU-hours)** plus 39 CPU-only seconds.
Failures are included. Trace files (not committed, 0.68 GB + 1.7 GB) stay on the cluster
under `r11-pressure/paged55-profile-cudaonly-r11v1/` with sha256 `89540efc…ae293d`
(nsys-rep) and `d28e031b…6d74c` (sqlite).

# Requirements and unchanged gates

The independent branch starts this continuation at `582ee70`. The cumulative
allowance starts at `40885d9`; older campaign headroom is not borrowed. Nine new
training runs and the shared-runtime checks use training/development data. No new
final comparison is admitted after the failed two-optimizer screen. The authorized
128-question/32-family final pool remains historically exposed and separated from
training/development by recording family.

| Requirement | Status and evidence |
|---|---|
| Native capability | Implemented: `adamw_bnb_8bit` and `paged_adamw_8bit`, one CUDA GPU, ordinary contiguous FP32 trainables, independent compute precision and consolidated recovery. [Exact support matrix](../../quantized_optimizers.md). No new distributed/dtype/QLoRA support. |
| Dense recovery | Pass retained: fused/nonpaged updates 41–44 match uninterrupted parameters, moments/scales/maps, scheduler, RNG, cursor and clocks. Core optimizer/model/transaction paths are unchanged. Prior 49 pinned regressions, seven loading and nine budget controls retain their scope. |
| Loading correctness | **Pass.** Independently pinned SHA and native provenance must succeed before score/read-success publication. Fifteen local/pinned controls pass; real CUDA wrong-hash failure publishes no scores/receipt. All 384 paired fresh-score observations match exactly. [Loading evidence](loading_summary.json). |
| Shared loading speed | **Phase improvement measured; whole-process statistical gate fails.** Verification plus read/restore ratios 1.4911 [1.2348,1.7474] fused and 1.4054 [1.1917,1.6191] eight-bit. Overall process 1.2453 [0.9365,1.5541] fails lower>1.05; fused arm also fails lower>1.0. No added restarts after seeing intervals. |
| LR 8.75e-6 development | **Numerical rejection.** Mean task NLL improves 6.61%, seed 29 loses one correct answer, seed 43 substantive NLL rises 5.59%. No conditional eight-bit/final runs. [Screen](fused_lr875_summary.json). |
| Shared weighting development | **Fused passes; eight-bit fails numerically.** Original LR, exponent 0.95: mean task NLL improves 4.83% fused / 5.38% eight-bit with unchanged accuracy in every seed. Eight-bit substantive ratios 1.01512/1.01327 in seeds 29/43 exceed 1.01. [Fused](fused_weight95_summary.json), [eight-bit](eightbit_weight95_summary.json). No promotion or new final scoring. |
| Useful intended-task adaptation | Prior final positive adaptation retained: 59.375% frozen, 76.82% fused, 76.56% eight-bit. Substantive supervision is retained. Restricted-choice results and teacher-forced answer-text loss do not establish free-generation answer quality. |
| Matched NLL upper ≤1.01 | **Prior statistical gate remains failed.** Ratio 1.004584 [0.96779,1.04138]. No new final result replaces it. [Unchanged final comparison](../quantized_optimizers_round3/corrected_product/summary.json). |
| Matched accuracy drop upper ≤1 point | **Prior statistical gate remains failed:** upper bound 1.381 percentage points. Paired-seed and recording-family procedures unchanged. |
| Equal 60-second training-budget quality | **Prior gate remains failed.** NLL gain −0.14824 [−0.64459,+0.34811]. Actual checkpoints and validation costs retained. |
| Equal 390-second product-budget quality | **Prior numerical/statistical failure remains.** Accuracy 67.71%→76.56%, but NLL 0.8160→1.1599. Actual updates fused 5/10/5 versus eight-bit 15/15/15; reserve boundaries are not intrinsic numerical superiority. |
| Prepared time-to-target lower>1.05 | **Prior statistical gate remains failed:** 1.05885 [1.01560,1.10210]. New development loading timings are not substituted into final optimizer timings. |
| Three full cold timing replications | **Operational gap closed.** Same producer/worker policy, exact inputs: 2193/2166/2121 allocation-seconds. Historical timeout/recovery stays charged. [Cold evidence](cold_preparation_summary.json). |
| Cold time-to-target lower>1.05 | **Gate fails with preparation variability now included:** composed ratio 1.00809 [1.00573,1.00965]. Not directly measured cold GPU pipelines or empty filesystem caches. [Planning bound](critical_path_plan.json): zero candidate training still gives about 1.011; roughly 90.5% preparation reduction needed just for a 1.05 mean. |
| State and total memory | Actual state remains 24,697,144,000→6,271,567,872 bytes (74.61%). Prior whole-device peaks 72.58/71.80GiB, training/checkpoint PSS 5.76/10.71GiB, full scoring PSS 33.38/33.68GiB retained. New scoring process GPU 34.99GiB; host PSS 33.98GiB serial / 34.02GiB shared. State saving is not total-memory saving. |
| Publication / limits | Local commits and PR draft only. Original QLoRA checkout untouched; PR334 closed. Paging remains a separately tested capacity mode, with no sustained task-quality/speed claim. Dense improvements do not transfer automatically to adapters or inference. |

The [completed per-example reload diagnosis](substantive_diagnosis_summary.json) checks the
existing seed-29 checkpoints without retraining or altering the substantive guard.
[Costs](final_accounting.json) include failed historical allocations, rounded wall
reservations while active, termination grace reserves and local-check allowance.
[Runnable commands](commands.md) and [raw identities](evidence_manifest.json)
retain every result. Remaining resource headroom does not waive the user's limit
of one additional shared intervention after the LR probe. The product objective
remains open; failed research requirements are distinct from implemented capability.

All eight new allocations are terminal. Combined charges are 5,127 GPU / 9,255 CPU-wall seconds; remaining headroom is 16,473 / 2,445 seconds. The [next executable proposal](next_intervention.json) remains unsubmitted.

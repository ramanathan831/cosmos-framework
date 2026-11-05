# Requirements and unchanged gates

This additional campaign starts from `40885d9` on the independent optimizer branch.
The failed LR screen admits no new eight-bit LR run or final comparison. The
historically exposed 128-question/32-family evaluation pool remains unchanged;
no new final predictions were inspected. Old matched-update, actual 60-second
training and actual 390-second product endpoints remain separate.

| Requirement | Status and evidence |
|---|---|
| Supported implementation | Implemented: ordinary FP32 trainables on one CUDA GPU, independent compute precision, quantized/paged state and consolidated exact recovery. [Support matrix](../../quantized_optimizers.md). No new dtype/distribution/QLoRA support claims. |
| Dense interruption and recovery | Pass retained: fused/eight-bit updates 41–44 match uninterrupted parameters, optimizer tensors/metadata, scheduler, RNG, cursor and clocks. [Prior evidence map](../quantized_optimizers_round3/requirements.md). Optimizer/transaction/restore paths did not change. |
| Positive adaptation versus frozen | Prior final comparison passes: 59.375% frozen, 76.82% fused, 76.56% eight-bit. Mixed multi-question plus original-choice objective retained. This does not establish free-generation answer quality. |
| Shared LR 7.5e-6 development screen | **Fails numerically.** Mean task NLL improves 9.04%, but two seeds lose 3.125 percentage points accuracy each; seed 43 substantive NLL ratio is 1.36815, above 1.01. [All curves, predictions and checks](fused_development_summary.json). No conditional final spending. |
| Matched optimizer NLL ratio upper ≤1.01 | **Statistical gate remains failed.** Prior mean 1.004584, interval [0.96779,1.04138]. Favorable mean compatibility is insufficient. [Unchanged final analysis](../quantized_optimizers_round3/corrected_product/summary.json). |
| Accuracy-drop upper ≤1 percentage point | **Statistical gate remains failed.** Prior upper bound 1.381 points. Same recording-family and paired-seed analyses retained. |
| Equal 60-second training-budget NLL gain | **Fails.** Prior fused-minus-eight-bit mean −0.14824, interval [−0.64459,+0.34811]. Actual checkpoint updates remain fused 10/10/10 versus eight-bit 10/12/11. |
| Equal 390-second prepared-product NLL gain, with accuracy protection | **Fails numerically and statistically.** Prior accuracy improves 67.71%→76.56%, but NLL worsens 0.8160→1.1599. Gain interval [−1.03976,+0.35205] includes no gain. Actual updates fused 5/10/5 versus eight-bit 15/15/15; reserve effects are not an intrinsic numerical advantage. |
| End-to-end time-to-target lower bound >1.05 | **Statistical gate remains failed.** Prior prepared ratio 1.05885[1.01560,1.10210]. New development loader canaries cannot substitute for complete prospective timing/quality comparisons. |
| Three full cold timing replications | **Operational gate incomplete.** Two uninterrupted allocations 2193/2166 seconds pass exact replay; third times out 2361 seconds after corpus publication. Byte verification recovers it in 29 seconds. The corrected manifest-binding verifier also passes a 25-second full recheck. [All corpora exact](cold_preparation_summary.json), but no three-homogeneous-replication CI or direct cold-GPU claim. |
| Shared measured runtime improvement | Development screens pass: source verification ratio 0.47462; combined checkpoint hash/restore ratio 0.70503. Exact inputs, scores, masks and checkpoint provenance retained. [Source screen](runtime_canary_summary.json), [overlap screen](checkpoint_overlap_summary.json). One counterbalanced pair per optimizer checkpoint; no final confidence claim. |
| Optimizer state versus total memory | Prior state 24,697,144,000→6,271,567,872 bytes, 74.61% saving. Whole-device 72.58/71.80 GiB; training/checkpoint host PSS 5.76/10.71 GiB; full scoring 33.38/33.68 GiB. [Unchanged phase report](../quantized_optimizers_round3/phase_memory_summary.json). Recomputation differs in memory-feasible deployment; pure optimizer-isolation kernels were slower. |
| New regression coverage | 16 prepared-input controls locally/pinned; six hash/restore controls locally/pinned; six corpus-integrity controls locally. Real CUDA fresh-process parity covers 128 observations per canary. No model, optimizer, save-state or transaction changes require repeating adequate dense recovery/factorial evidence. |
| Budget and publication | [Actual charges](final_accounting.json): 1575 GPU/6954 CPU-wall seconds; all eight allocations terminal, including one timeout and its separate recovery. Headroom 14085/246 seconds, not exhausted. Original QLoRA checkout untouched; PR #334 closed; local commits and PR draft only. |

The next [8.75e-6 recipe hypothesis](next_shared_lr_probe.json) retains every guard
and reserves 15,660 GPU-seconds for both conditional stages and complete dependent
final work. The [cold replacement](next_cold_replication.json) reserves 3,000 CPU
seconds. Neither is submitted. The product objective remains open; the failures
are not relabeled experimental or converted into passes.

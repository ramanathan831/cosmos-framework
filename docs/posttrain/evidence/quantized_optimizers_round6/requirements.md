# Requirement-to-evidence map

The product objective remains open. The frozen numerical margins and endpoint
clocks are unchanged. Round six did not admit a new final optimizer comparison.

| Requirement | Current evidence and outcome |
|---|---|
| Native optimizer implementation | Retained: `adamw_bnb_8bit`, `paged_adamw_8bit`, lazy optional bitsandbytes 0.49.2, FP32 trainables, one CUDA GPU, consolidated same-identity recovery. No optimizer/transaction changes in this continuation. |
| Recovery and skipped updates | Retained round-three dense fused/nonpaged40→44 fresh-process comparison and scoped failure-injection tests. No new broad distributed/dtype/paging claims. |
| Shared 0.975 development selection | **Failed numerically:** task NLL improves 3.4503%, identical accuracy in all seeds; seed 43 native substantive ratio 1.012970669 exceeds 1.01. See `fused_weight975_summary.json`. No rounding or batch-one substitution. |
| Eight-bit 0.975 stage | Not admitted because fused failed. No optimizer-specific conclusion from the fused-only rejection. |
| Useful task adaptation | Historical final accuracy 59.375% frozen /76.82% fused /76.56% eight-bit. New development results do not replace those final estimates. |
| Matched-quality preservation | Historical NLL ratio 1.004584[.96779,1.04138] and accuracy-drop upper 1.381 pp still fail the≤1.01/≤1pp guards. No new final scores. |
| Equal-budget quality | Historical positive NLL-gain bound still fails, including the corrected390-second endpoint. Accuracy gain alone cannot pass it. |
| Prepared time to fixed target | Historical 1.05885[1.01560,1.10210] still fails lower >1.05. Fixed target1.7987394576095224 unchanged. |
| Cold speed | Historical composed 1.00809[1.00573,1.00965] still fails lower >1.05. Three full producer timings 2193/2166/2121 remain historical controls. Slice profiles cannot replace them. |
| Memory | Retained initialized optimizer tensor-state saving 74.61%; whole-device 72.58/71.80GiB, training/checkpoint PSS 5.76/10.71GiB, full-delivery PSS 33.38/33.68GiB. State savings are not total-memory savings. |
| Shared loading | Retained 384 exact development observations and fail-closed checksum/provenance behavior. Whole-process 1.2453[.9365,1.5541] does not establish reliable end-to-end or inference speed. |
| New preparation checks | Nine focused tests pass, including oracle failure rejection and corrected timer boundaries/ownership. The real slice reproduces 15 targets exactly; its detailed cProfile percentages are unreliable. No acceleration candidate was admitted. See `preparation_summary.json`. |
| Dataset honesty | Authorized reused pool, repaired family separation, 128 final questions / 32 historically exposed families. No untouched-data or independent replication claim. |
| Publication | Local commits/draft only. No push, publication, merge, PR334 reopening, or original QLoRA checkout changes. |

The midpoint weighting is not monotonic. Seed43 substantive loss improves against
control at updates 5/10, then exceeds it at 15. Only 6/32 task NLLs improve at 15,
although the mean falls 2.06%; this is not broad calibration or generated-answer
quality evidence. See `development_diagnosis.json`. No additional candidate is
automatically authorized by unused GPU headroom.

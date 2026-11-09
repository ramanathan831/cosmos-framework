# Requirement-to-evidence map

The product objective remains open. Round seven changes preparation tooling and
read-only development attribution. It does not change optimizer mathematics,
training defaults, quality margins, or final evaluation definitions.

| Requirement | Evidence and current outcome |
|---|---|
| Native capability | Retained: `adamw_bnb_8bit`, `paged_adamw_8bit`, optional pinned bitsandbytes0.49.2; one CUDA GPU, ordinary contiguous FP32 trainables, independent forward precision, consolidated same-identity recovery. No broader dtype/distribution/paging-speed claim. |
| Recovery | Retained dense fused/nonpaged40→44 continuation and scoped transaction failure tests; unchanged core source hashes are checked in `source_identity.json`. |
| Exponent0.975 selection | **Failed numerically in round six:** mean task NLL−3.4503%, accuracy unchanged in all seeds; native substantive ratios0.992275053/1.006648854/**1.012970669** against≤1.01. No eight-bit or final stage admitted. |
| Verified development diagnosis | Three seed43 fresh-process checkpoints reproduce96 task observations and all three native batch-four substantive aggregates **exactly**, with identical batch inputs and1346supervised tokens each. See `substantive_summary.json`. No scoring/count/mask/checkpoint defect found. |
| Useful task adaptation | Historical final accuracy59.375%frozen/76.82%fused/76.56%eight-bit retained; the new diagnostics do not replace final estimates or establish free-form generation quality. |
| Matched quality | Historical NLL ratio1.004584[0.96779,1.04138] fails upper≤1.01; accuracy-drop upper1.381points fails≤1point. No new final comparison. |
| Equal-budget quality | Historical positive NLL-gain requirement remains unmet; accuracy gain alone does not pass it. Actual60-second and390-second endpoints remain unchanged. |
| Prepared time to target | Historical1.05885[1.01560,1.10210] fails lower>1.05. Fixed quality target1.7987394576095224 unchanged. |
| Cold speed | Historical composed1.00809[1.00573,1.00965] fails lower>1.05. New slice timing is not full cold evidence. Conditional full preparation protocol is separate. |
| State and total memory | Historical optimizer tensor state24,697,144,000→6,271,567,872bytes (74.61%); whole-device72.58/71.80GiB; training/checkpointPSS5.76/10.71GiB; full-deliveryPSS33.38/33.68GiB. These are distinct scopes. |
| Shared preparation slice | **Passed its development runtime screen:** full Python process ratio1.24037[1.14525,1.37167], three paired trials;90exact target comparisons, unchanged corpus manifest. Four workers/two threads, native decoder, independent replay and complete hashes retained. See `reuse_summary.json`. |
| Preparation correctness/current code | Initial44local affected checks pass. Corrected payload-container accounting passes18local checks. A pinned real-slice check exposed an invalid current-size-equals-peak test assertion (40bytes lower, exact tensors already passed). The corrected retry passes19pinned tests, real hash/pixel-corruption rejection and all15exact targets. Failure remains in `accounting_failed_raw.json.gz`; full-source replications are in progress. |
| Data honesty | Authorized reused QLoRA train/eval pool; repaired recording-family separation; final128questions/32families historically exposed. No untouched-data or independent replication claim. |
| Publication/resource policy | Local commits/draft only; no push/publish/merge, PR334 remains closed, original QLoRA checkout untouched. User removed aggregate GPU/CPU/time ceilings; historical charges and all new failures remain recorded. |

The substantive trade-off affects10of16rows in the same direction for both
weighting candidates; these rows come from five families. Prominent worsening
words include `Dim`, `Light`, and `Clear`, which are represented in training.
This does not support a rare-label oversampling fix or a quantization-specific
cause. See `development_diagnosis.json` and `development_label_counts.json`.

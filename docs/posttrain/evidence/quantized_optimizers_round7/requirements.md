# Requirement-to-evidence map

The product objective remains open. Round seven changes preparation tooling and
read-only development attribution. It does not change optimizer mathematics,
training defaults, quality margins, or final evaluation definitions.

| Requirement | Evidence and current outcome |
|---|---|
| Native capability | Retained: `adamw_bnb_8bit` and `paged_adamw_8bit`, optional bitsandbytes 0.49.2; one CUDA GPU, ordinary contiguous FP32 trainables, independent compute precision, consolidated same-identity recovery. No broader dtype, distributed or paging-speed claim. |
| Recovery | Retained dense fused/nonpaged 40→44 continuation and transaction failure tests. Unchanged core hashes are checked in `source_identity.json`. |
| Exponent 0.975 selection | **Failed numerically in round six.** Mean task NLL improves 3.4503%, accuracy is unchanged in all seeds. Native substantive ratios are 0.992275053 / 1.006648854 / **1.012970669**, against ≤1.01. Eight-bit and final stages were not admitted. |
| Development diagnosis | Three seed43 fresh-process checkpoints reproduce 96 task observations and all three native batch-four substantive aggregates **exactly**, with identical inputs and 1,346 supervised tokens each. See `substantive_summary.json`. No scoring, count, mask or checkpoint defect was found. |
| Useful task adaptation | Historical final accuracy is 59.375% frozen / 76.82% fused / 76.56% eight-bit. The new diagnostics do not replace these estimates or establish free-form generation quality. |
| Matched quality | Historical NLL ratio 1.004584 [0.96779, 1.04138] fails upper ≤1.01. Accuracy-drop upper bound 1.381 percentage points fails ≤1 point. No new final comparison. |
| Equal-budget quality | The historical positive NLL-gain requirement remains unmet; accuracy gain alone does not pass it. Actual 60-second and 390-second endpoints remain unchanged. |
| Prepared time to target | Historical ratio 1.05885 [1.01560, 1.10210] fails lower >1.05. Fixed quality target 1.7987394576095224 is unchanged. |
| Cold speed | Historical composed ratio 1.00809 [1.00573, 1.00965] fails lower >1.05. New full preparation replications remain in progress; slice timing is not substituted for this endpoint. |
| State and total memory | Historical optimizer tensor state falls 24,697,144,000→6,271,567,872 bytes (74.61%). Whole-device peaks are 72.58/71.80 GiB; training/checkpoint PSS 5.76/10.71 GiB; full-delivery PSS 33.38/33.68 GiB. These are distinct scopes. |
| Shared preparation slice | **Passed its development runtime screen:** complete Python-process ratio 1.24037 [1.14525, 1.37167], three paired trials, 90 exact comparisons and unchanged corpus manifest. Four workers, two threads each, native decoder, independent replay and full hashes remain. See `reuse_summary.json`. |
| Preparation correctness | 44 affected local checks pass. Corrected payload accounting passes 19 pinned checks, including real wrong-hash and cached-pixel-corruption rejection. The invalid current-size-equals-peak assertion and its failed allocation remain recorded. The first two full runs pass all 751 rows / 562 targets; the third replication is pending. |
| Data honesty | Authorized reused QLoRA train/eval pool, repaired recording-family separation, final 128 questions / 32 families historically exposed. No untouched-data or independent replication claim. |
| Publication and resources | Local commits/draft only; no push, publication or merge. PR334 remains closed and the original QLoRA checkout untouched. The user removed aggregate GPU/CPU/time ceilings; all historical and new costs, including failures, remain recorded. |

Both weighting candidates worsen 10 of 16 substantive rows from five recording
families. Prominent worsening words include `Dim`, `Light` and `Clear`, all
represented in training. This does not support rare-label oversampling or a
quantization-specific explanation. The original and 0.975 seed43 data cursors
are byte-identical and reconstruct the same 60 training rows.

At update 10, the 0.975 seed43 substantive loss is lower than control; at update
15 it is higher. Both absolute losses improve over that interval. Missing
update-10 checkpoints prevent per-question attribution of that relative reversal.
`development_diagnosis.json` records the next exact diagnostic commands and cost;
it does not admit another training recipe or final comparison.

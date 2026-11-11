# Requirement-to-evidence map

The product objective remains open. Round eight retains all original numerical
gates and uses development examples only for diagnosis and recipe selection.

| Requirement | Evidence and current outcome |
|---|---|
| Native optimizer capability | Retained: optional bitsandbytes 0.49.2, `adamw_bnb_8bit` / `paged_adamw_8bit`, one CUDA GPU, ordinary contiguous FP32 trainables, separate compute precision, consolidated same-identity recovery. |
| Quantized recovery | Retained fused/nonpaged dense 40→44 recovery, transaction failure coverage and pinned regressions. Optimizer and transaction code are unchanged. |
| Controlled diagnosis | **Passed:** original and .975 seed43 curves and full final training states reproduce; four fresh-process native batch-four endpoint reloads match. `replay_summary.json` and `replay_raw.json.gz`. |
| Mechanism | Relative reversal occurs mainly at updates 13–14, chiefly traffic-volume answer tokens, before all-choice 15. Transient competition is a testable hypothesis, not proof of a sole cause. `development_diagnosis.json`. |
| Shared ordering correction | Opt-in proportional interleaving preserves exact 20-row-window/epoch exposure and within-kind order. Legacy defaults remain. Local tests pass; fused three-seed development and native 10→15 recovery are admitted, pending. `interleave_protocol.json`. Eight-bit and final stages remain conditional. |
| Shared reference verification | **Passed its measured phase screen:** three counterbalanced full-corpus pairs, exact751 rows/562 targets. Verification ratio 4.0882 [3.9940,4.2654], whole verifier-process 1.7174 [1.5804,1.8340]. Existing corpora, uncontrolled storage caches; not cold preparation. `verification_summary.json`. |
| Verification correctness | Full hashes, source/processor binding, mutation detection, all independent native target replays and final success sequencing retained. Fresh 15-target slice and affected local/pinned checks pass. Counts overlap. |
| Useful adaptation | Historical final accuracy 59.375% frozen / 76.82% fused / 76.56% eight-bit. New teacher-forced development attribution does not establish generated-answer quality. |
| Matched quality | Historical NLL ratio 1.004584 [0.96779,1.04138] fails upper≤1.01; accuracy-drop upper 1.381 pp fails≤1 pp. No new final comparison yet. |
| Equal-budget quality | Historical positive NLL-gain guard remains unmet. Actual 60-second training/validation and 390-second prepared-product states remain distinct. |
| Prepared time to target | Historical 1.05885 [1.01560,1.10210] fails lower>1.05. Fixed target 1.7987394576095224 unchanged. |
| Cold time to target | Historical composition 1.008735 [1.006317,1.010647] fails lower>1.05. New verifier timings do not replace full preparation or GPU deliveries. Even free historical verification cannot close the gap under retained costs. `critical_path_plan.json`. |
| Memory | Retained optimizer tensor-state reduction 74.61%, exceeding 60% requirement. Whole-device 72.58/71.80 GiB; training/checkpoint PSS 5.76/10.71 GiB; full-delivery PSS 33.38/33.68 GiB. These scopes differ. New verifier PSS 0.80–0.87/0.71–0.81 GiB excludes pre-sampler imports. |
| Data / attribution | Authorized reused pool, repaired family separation; final 128 questions / 32 families historically exposed. No untouched replication, inference-latency, distributed, additional-dtype, sustained-paging or automatic LoRA/QLoRA claim. |
| Resources / publication | No aggregate caps. Actual allocations and failed attempts retained, configured routing/concurrency unchanged. Local commits/draft only; no push, publication, merge, QLoRA checkout change or PR #334 reopening. |

The first recipe intervention changes only order, at original exponent 1.0 and
LR 1e-5, 15 updates / 40-update horizon. Its screen requires ≥2% lower mean task NLL, no
per-seed accuracy loss and every native substantive ratio ≤1.01 against original
matching controls. A second intervention is not selected or admitted. It requires
evidence supporting a concrete revised mechanism after the first result.

# Add native quantized AdamW with explicit precision and recovery contracts

Native SFT can select `adamw_bnb_8bit` or `paged_adamw_8bit` independently of model
weight quantization. Pinned bitsandbytes 0.49.2 provides state quantization; ordinary
FP32 trainables preserve update precision. Typed configuration, CLI, stable
parameter groups/LR multipliers, optional imports, FP32 state exceptions and
actual allocated-state reports are integrated. Unsupported precision, strategy,
checkpoint and optimizer options fail explicitly.

Consolidated recovery preserves integer states, scales/maps, lazy initialization,
counters, group/parameter identity and paging ownership. Snapshots are independent
and restore metadata is staged before mutation. Engineering support is one CUDA
GPU with ordinary contiguous FP32 trainables and separately configured forward
compute. Distributed/DTensor, unsupported offload/checkpoint combinations and
BF16/FP16 trainables without masters are rejected. Paging remains a capacity mode.

**Product outcome qualification remains open.** Dense fused/nonpaged fresh-process
40→44 recovery passes every resumed update. Historical optimizer tensor state
falls 24,697,144,000→6,271,567,872 bytes (74.61%). The 1.356× native-loop deployment
gain spends saved memory on less activation recomputation; optimizer-isolation
tests did not establish faster eight-bit calls. Whole-device and phase-specific
host peaks remain separately reported in the support document.

Historical task adaptation improves accuracy from 59.375% frozen to 76.82% fused
and 76.56% eight-bit. Matched NLL and accuracy uncertainty guards, positive
equal-budget NLL gain and prepared/cold time-to-target margins remain unmet.
The final 128-question/32-family pool is historically exposed, with repaired
training/development separation. It is an authorized comparative benchmark,
not independent replication. No inference-latency or automatic LoRA/QLoRA benefit
is claimed.

Round-eight exact original/.975 seed 43 replay and four fresh-process endpoint
reloads localize the relative substantive-loss reversal primarily to updates 13–14
and traffic-volume answer tokens. Inputs, masks, 1,346 supervised tokens, final
parameters, optimizer, RNG, scheduler and cursor all match their respective
stored runs. No scoring or optimizer defect was found. An opt-in resumable
within-window ordering policy tests the resulting competition hypothesis.
Its local coverage includes window/epoch membership, tails and legacy
compatibility. Native seed 17 fresh-process 10→15 continuation matches every next
input batch and update, plus final model/optimizer/data/trainer components.
The three-seed fused screen improves mean task NLL 19.07% without accuracy loss,
but substantive ratios 0.694146/1.542966/1.208833 fail the unchanged ≤1.01 guard.
The recipe is rejected; defaults remain unchanged, with no eight-bit/final stage.
Three further native batch-four reloads reproduce96 task observations and all
substantive losses, identifying shared semantic-confidence shifts across families.
The second/final trial adds optional training-only uniform-vocabulary label
smoothing 0.01 while restoring original order. Hard-target evaluation and reported
CE remain unchanged; nonzero regularization enters the exact-resume identity.
Its fused screen fails: mean task NLL increases 2.4075%, seed 17 loses one correct
answer, and seed 43 substantive loss rises 13.9053%. No eight-bit/final stage follows.
Seventeen pinned objective checks (including CUDA) pass. The disabled default
reproduces the stored seed 43 run exactly, and new-objective seed 17 continuation
matches all five inputs/updates and final state, including the validation journal.
Both recipe trials remain rejected; no product quality or end-to-end speed gate
is promoted from these development results.
A native batch-four seed 43 reload reproduces 32 task observations and all 1,346
supervised-token losses in aggregate. Its worsening combines semantic-answer
(62.8%), formatting (35.9%) and end-token (1.3%) contributions. No scoring/masking
or checkpoint-identity defect explains the failed guards. All admitted work is
terminal, with failed/cancelled attempts retained in actual accounting.

Shared preparation reference verification can now compare exact serialized
identities using bounded complete hashing, retaining source/processor checks,
mutation rejection and every uncached native target replay. Three counterbalanced
pairs on complete existing corpora preserve 751 rows/562 targets and improve the
verification phase 4.0882× [3.9940,4.2654] and whole verifier process 1.7174×
[1.5804,1.8340]. A fresh 15-target slice also passes. These measured scopes exclude
new full preparation and cold optimizer delivery; storage caches are uncontrolled.
The change cannot close the historical cold gap under retained critical-path
costs, so no equivalent full cold campaign is repeated.

See [support and recipes](../../quantized_optimizers.md),
[requirement map](requirements.md), [commands](commands.md),
[source identities](source_identity.json) and raw evidence. All historical
negative results remain. This is a local draft; no push, publication or merge,
original QLoRA checkout untouched, and PR #334 remains closed.

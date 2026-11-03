# Quantized optimizer design and qualification

Base: upstream main e644a833bf909af9c23a02a5c1069967dc501111 (fetched 2026-11-03).
Independent branch: feat/quantized-optimizers. QLoRA repository is read-only;
observed component a7fd524242f4f06ac488a3d96d331b1f46e2b385, PR 334 closed.
No equivalent optimizer backend found in main or the optimizer PR search.

## Design

Use bitsandbytes 0.49.2 AdamW8bit, with paging a separate public name. Do not
change model architecture or quantize weights. Require ordinary contiguous CUDA
FP32 trainables; BF16 forward uses explicit autocast. Frozen weights may have
other dtypes. No implicit master weights. Reject DTensor/FSDP/HSDP, low precision
trainables, sparse gradients, offload, and DCP until their ownership contracts are
implemented and qualified. Mainline release target: native posttrain SFT, HF model
builder, one CUDA device, consolidated transactional checkpoints.

Keep existing trainability, decay/LR groups and scheduler. Exact-name FP32 state
exceptions resolve against final trainables; reject missing/duplicate selectors.
Use instance-owned backend override and page managers, not process-global
registration. Backend percentile clipping remains 100 (disabled).

Snapshots synchronize and copy values into independent ordinary CPU tensors.
Restore validates backend/version, topology, precision, parameter names/shapes,
state policy and group identities before rebuilding backend allocations. Initialize
only recorded states, without an update. Paged moments are reallocated as managed
memory; lazy parameters remain lazy. Exported model state has no bnb requirement.
Report actual moment bytes and metadata after allocation, not selected names.

## Preregistered qualification (before measurements)

Initial budget: 8 GPU-hours including failed attempts; canaries first, at most
one new Slurm allocation at once, maximum 60 minutes per allocation. Do not touch
other campaigns. Stop large jobs on correctness failures or unavailable independent
evaluation data. Development kernels use paired seeds 17, 29, 43 and fused torch
FP32 AdamW; bnb32 is diagnostic only. Final comparisons require three paired seeds,
95% paired t confidence intervals across seeds, lower bound of speed ratio >1.05,
upper bound of held-out loss ratio <=1.01, and task score degradation <=0.01
absolute (or stricter existing workload criterion). Equal-budget quality gain
requires a positive lower confidence bound. State memory must decrease >=60% for
eligible large tensors; report total peak separately. No thresholds change after
results. Use five paired timing trials after warmup. Synthetic training canaries
are correctness/performance diagnostics, never multimodal quality qualification.

Final multimodal evaluation needs recording/trajectory-disjoint untouched data.
Existing WTS evaluation groups have been inspected; they cannot be reused as an
untouched test set. No release or speed/quality claim until all gates pass.

Title: Add single-GPU bitsandbytes AdamW with managed-state recovery

Posttrain SFT can select `adamw_bnb_8bit` or `paged_adamw_8bit` with ordinary CUDA
FP32 trainables, independently of model-weight quantization. The factory preserves
trainability and LR/decay groups and supports exact-name FP32-state exceptions.
The optional extra pins bitsandbytes 0.49.2; ordinary optimizer imports remain lazy.

Consolidated checkpoints store immutable CPU values and validate optimizer/version,
precision, group mapping and state policy on resume. Paged moments are reconstructed
as managed allocations without advancing the optimizer. FSDP/DTensor, DCP, low
precision trainables without masters, sparse gradients and optimizer offload are
explicitly unsupported. Existing high-precision paths remain available.

Validation: real A100 CUDA state/update tests; fresh-process bitwise continuation;
native SFT transaction tests covering accumulation, clipping, scheduler, RNG and
data cursor; standard optimizer regression tests. Disposable QLoRA integration
uses component a7fd524 and optimizer 2131a42, with a separable adapter allowlist
change. It does not alter the main feature's prerequisites; PR 334 stays closed.

Release qualification is NOT complete. In the 16.8M-parameter all-resident
synthetic diagnostic, moments plus metadata decrease from 134.2 MB to 34.1 MB,
but nonpaged eight-bit throughput is 0.544x fused torch AdamW (95% seed CI
0.540–0.548). A finite-check optimization improved the first attempt but did not
pass the speed gate. These are development diagnostics, not task-quality results.
No quality gain, end-to-end speedup, paging capacity, distributed support, or
inference benefit is claimed. Untouched trajectory-disjoint final evaluation data
and real-workload time-to-quality qualification remain required before release.

See [design and support matrix](../../quantized_optimizers.md)
and the raw evidence beside this file for commands, current support matrix,
negative results, and Slurm outcomes.

# Add native quantized AdamW with explicit precision and recovery contracts

Native SFT can select `adamw_bnb_8bit` or `paged_adamw_8bit` independently of model
weight quantization. Pinned bitsandbytes 0.49.2 provides quantized states for
ordinary FP32 trainables, preserving update precision. Typed configuration/CLI,
stable groups and LR multipliers, lazy optional imports, FP32 state exceptions
and actual allocation reports are integrated. Unsupported precision, strategy,
checkpoint and optimizer options fail explicitly.

Consolidated recovery preserves quantized bytes, scales/maps, lazy state,
counters, logical parameter identity and paging ownership. Snapshots are
independent; restore metadata is validated before mutation. Engineering support
is one CUDA GPU with contiguous FP32 trainables and independently specified
forward compute. Paging retains capacity evidence, without a sustained speed claim.

Native opt-in partial epoch batches now preserve complete data exposure, including
real tails and exact resumable ordering. An optional bounded source-hashing path
retains full hashes, deterministic identities and fail-closed mutation handling.
Defaults remain unchanged. On three counterbalanced CPU pairs, source construction
improves 4.5345× and whole Python-process time 1.2481× [1.1092,1.4520]; this is a
shared loading result, not cold training or optimizer acceleration.

Validation includes real CUDA observer checks, 72 pinned native regressions,
parallel-hashing parity/failure checks and fused/nonpaged fresh-process recovery.
The new 87→174 continuation matches all 87 next inputs and semantic update
records, full state at 88/89/174 and complete final checkpoints. Prior dense
40→44 and quantized/paged transaction recovery remain applicable.

**Product outcome qualification remains open.** Six fixed two-pass development
runs process 692 rows each. Both optimizers reach 27/32 correct in all three seeds
versus 15/32 frozen. Native substantive ratios and accuracy guards pass, but
matched NLL ratio 0.9770 [0.6302,1.3238] fails the ≤1.01 upper guard; the crossed
seed/family upper bound is 1.1554. No new final comparison is admitted. A controlled
within-update permutation reproduces forward losses but exposes small backward
numerical differences and trajectory amplification, without demonstrating an
optimizer or data-alignment defect. No new recipe is selected from that probe.

Actual optimizer tensor state falls 24,697,144,000→6,271,567,872 bytes (74.6061%).
The two-pass study's device peaks average 78.61/72.15 GiB and host PSS 5.85/10.69
GiB, fused/eight-bit, including training/validation/checkpoint diagnostics.
Historical 1.356× native-loop deployment gains use saved memory to remove
recomputation; pure optimizer-isolation tests did not show faster eight-bit calls.
Original matched-quality uncertainty guards, equal-budget NLL gain and
prepared/cold time-to-target margins remain unmet. Instrumented matched-update
training is not an equal-time result. All negative trials remain preserved.

The authorized final 128-question/32-family pool is historically exposed, with
repaired training/development separation; this is not independent replication.
No inference-latency, distributed/dtype, sustained-paging or automatic LoRA/QLoRA
benefit is claimed. See [support and recipes](../../quantized_optimizers.md),
[requirements](requirements.md), [commands](commands.md),
[source identities](source_identity.json) and [actual costs](final_accounting.json).
All admitted jobs are terminal. This draft and all commits remain local;
no push, publication, merge, QLoRA checkout change or PR #334 reopening.

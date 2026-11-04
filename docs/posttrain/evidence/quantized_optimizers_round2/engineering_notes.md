# Bounded deployment follow-up

This directory extends, and does not replace, the sealed first-campaign evidence.
The release gates in `plan.json` remain unchanged. The follow-up was authorized by
“continue” after the first campaign's negative speed/quality report.

The first campaign spent roughly 494–514 seconds/run preprocessing and validating
videos, while optimizer calls were slower with eight-bit states. The 16-block
capacity canary established a concrete deployment comparison: the fused baseline
needs activation checkpointing at microbatch four; nonpaged eight-bit AdamW can
run without it. Both arms now consume exactly the same prepared native CPU inputs.
Preparation is a shared optimization, not an optimizer speedup. Analysis must show
launch-to-quality from the prepared corpus and a cold pipeline clock including the
full CPU preparation/verification cost, alongside the direct native training clock.

The optional `rl.prepared_corpus` path is checked before model construction. The
manifest binds the native processor, source fingerprints, media content, and each
serialized CPU input. Collation, labels, autocast and model arithmetic are unchanged.
Preparation rejects RNG-consuming preprocessing. Readers have no fallback, retain
private mmap ownership, and reject changed contents/settings. The manifest digest
joins exact-resume objective identity. Raw-input recipes retain their previous
identity when the option is empty.

The first pinned CPU check exposed a real filesystem assumption: a same-size media
rewrite retained the same stat timestamps inside the container. Per-read content
hashes now cover both media and prepared payloads; coarse-stamp regression tests
exercise this explicitly. Failed job 37949431 consumed 181 CPU-wall seconds, zero
GPU seconds. The initial submission to a GPU partition was rejected before allocation.
The corrected retry is on cpu_short, capped at 26 minutes, keeping total admitted
CPU preparation below 30 minutes. Job 37949524's native regression passed 55 tests
with one skip before full corpus preparation.

The dense scorer reuses the checked answer alignment and task metric definitions
from integration commit 188379861a9d43cb99c57b2b083ada8684aca272, with no QLoRA imports.
It loads only the committed dense model through CheckpointManager, checks real video
conditioning and exact input hashes, and restores no optimizer. `log_softmax`
avoids cancellation in very small gold-answer NLL values. This is teacher-forced
restricted-choice/first-token scoring, not free-generation task accuracy.

All GPU runs use the exact A100-SXM4-80GB model check and a pre-resolved allocated GPU
UUID for external NVML samples. Training timing uses the native entrypoint without
synchronizing Python model/optimizer hooks. Post-run checkpoint-value verification
and scoring are separate from training timing but count against the allocation
budget. Full checkpoint writes remain inside launch-to-completion measurements.

Final selection uses the remaining 16 recording components from the user-approved
historically reused QLoRA pool, excludes this campaign's train/development groups
and the previous final groups, and was fixed before model scoring. No untouched-pool
claim is made. Score fixed restart zero only. All three restarts must agree in
model, moments, RNG, scheduler and data-cursor values before final scoring.

Before any round-two training outcomes, the launcher was corrected to resolve the
single CUDA-visible GPU inside an exiting preflight child and bind NVML to that
UUID. The batch-shell GPU variable did not provide a usable UUID selection.
The first guard failure used five GPU seconds. A second attempt reached native
setup and correctly refused an unset CUBLAS_WORKSPACE_CONFIG after CUDA init;
it used 102 GPU seconds. The launcher now exports :4096:8 and PYTHONHASHSEED=42.
These failed attempts remain charged. The unused local-check reservation covers
107 seconds of launcher failures without changing the 9000-second campaign cap.

Inspection of the first campaign's already-sealed startup logs showed native
weight loading taking 263,125,8 seconds in seed17's three successive arms. To
avoid an accidentally cold first-arm control, every new arm performs identical
sequential reads/hashes of all pinned model safetensors before model construction.
This common step stays inside its launch clock, and its separate timing/content
receipt is retained. It changes no model bytes or loader arithmetic. Full timing
trials have not started, and no new quality outcomes informed this control.

The v6 canary completed four raw fused updates but emitted PyTorch's explicit
nondeterministic-attention warning. Mainline enable_determinism had only a
warning-only mode; the strict API existed on the disposable integration branch,
not mainline. The canary was stopped after331GPU-seconds. Mainline now exposes
an optional strict helper and typed SFT strict_determinism flag, retaining the
old warning-only default. The deployment recipe enables strict mode, and its
checkpoint objective identity records it. This is a small independent mainline
API change, not a QLoRA dependency or a merge of that branch. Local control tests
pass58with3environment skips and1missing-torchvision case deselected (the latter
passed in the earlier pinned CPU job). Strict canary and full trials remain gated.

The GPU cap remains9000seconds. Reservation reallocation now covers438seconds
of retained failures,720seconds for the strict canary,2600seconds for each of the
three unchanged six-run seed groups, and42unused local-check seconds. Trial
steps, seeds, paired restarts, final scoring and all acceptance thresholds are
unchanged. Returned job time remains available for dependent verification.

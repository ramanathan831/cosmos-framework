# Native optimizer integration reproduction

These commands use the configured scheduler routing and existing approved assets.
No original QLoRA checkout or inherited runtime is writable. Round-nine WTS
commands and results remain unchanged. These are integration/development commands,
not a completed final quality protocol.

## Source and shared accelerator checks

```bash
git archive --format=tar.gz <tested-commit> cosmos_framework tools pyproject.toml .pytest.toml -o source.tar.gz
# On the configured cluster; R is the existing authorized per-user project root.
export SOURCE_ARCHIVE="$R/cosmos-optimizer-native-20261113/source-v39.tar.gz"
export CONTAINER_IMAGE="$R/cosmos-automl-focused-20261012/image-conversion/cosmos-framework-focused.sqsh"
export ASSETS="$R/cosmos-qlora-wts-20261028T234324Z"
export RESULTS="$R/cosmos-optimizer-native-20261113/new-master-attempt"
sbatch --partition=polar,polar3,polar4,grizzly tools/quantized_optimizers/master_distributed.sbatch
```

This script runs fused/nonpaged/paged two-rank FSDP2 reference and fresh-process
resume pairs, including one-rank failure tests. It does not run an action model.
Do not overlap owned GPU allocations on the same node; use explicit dependencies
when placement cannot be guaranteed without changing configured routing.

## Retained native SFT training, evaluation and continuation

The executed SFT sequence is preserved in
`../quantized_optimizers_round9/full_seed17_spec.json`: fused and nonpaged
reference training through174updates, retained checkpoint87, then fresh-process
87→174 continuations with native validation and next-input/update observers.
It uses the SFT environment and prepared-input identities in the corresponding
`full_seed17_launch.json`, not GR00T's separate Transformers runtime. The launcher
calls `task_development`, which runs the native SFT entrypoint and read-only
development observers. No SFT surrogate is used for either action-model path.

From the configured Slurm host, reproduce the retained sequence in a new directory:

```bash
python - <<'PYCODE'
import json, os, subprocess
from pathlib import Path
receipt = json.loads(Path(
    'docs/posttrain/evidence/quantized_optimizers_round9/full_seed17_launch.json'
).read_text())
env = os.environ.copy()
env.update(receipt['environment'])
env['RESULTS'] += '-reproduction'
assert not Path(env['RESULTS']).exists()
subprocess.run([
    'sbatch', '--parsable', '--export=ALL', '--time=60', '--partition=polar',
    '--job-name=optimizer-sft-reproduction',
    '--output=' + env['RESULTS'] + '-%j.log',
    'tools/quantized_optimizers/deployment.sbatch',
], env=env, check=True)
PYCODE
```

This reproduces the pinned historical source and full chain; it does not claim a
new round-ten SFT benchmark. Per-allocation work/grace settings remain scheduler
termination guards, not aggregate compute ceilings. For a current-source run,
explicitly replace `SOURCE_ARCHIVE` with the newly recorded immutable archive and
retain every other model/data/recipe field; record that changed source identity.
The direct SFT CLI and independent model-only `score_dense` path are documented in
`tools/quantized_optimizers/README.md`. Existing SFT recovery evidence is retained
because the class-lifetime correction changes ownership, not update arithmetic.

## Native GR00T Joint PPO

The checked model is the existing GR00T-N1.7-LIBERO Spatial snapshot
`2ea293aa20ba7cf5bbf3ba17a5fbcb1a01cbfe21`. The complete checkpoint supplies backbone
weights; its exact approved metadata supplies the graph/processor. The isolated
model and simulator environments are recorded in source/runtime evidence.

```bash
export RESOURCE_ROOT="$R"
export MODEL_PYTHON="$R/cosmos-optimizer-native-20261113/action-runtime-v2/model/bin/python"
export SIM_PYTHON="$R/cosmos-optimizer-native-20261113/sim-runtime-v1/environment/bin/python"
export RESULTS="$R/cosmos-optimizer-native-20261113/new-gr00t-reference"
export OPTIMIZER_NAME=adamw UPDATES=8 RESUME_CHECKPOINT= EVALUATE_ONLY=0 NATIVE_OUTPUT_ROOT=
sbatch tools/quantized_optimizers/gr00t_native.sbatch
# Same sequence for adamw_bnb_8bit, then paged_adamw_8bit after nonpaged recovery.
# Fresh-process evaluation of the actual retained midpoint:
export RESULTS="$R/cosmos-optimizer-native-20261113/new-gr00t-reload"
export RESUME_CHECKPOINT=/exact/reference/output/checkpoints/iter_000004
export EVALUATE_ONLY=1
sbatch tools/quantized_optimizers/gr00t_native.sbatch
# Fresh-process completed-window continuation through update eight:
export RESULTS="$R/cosmos-optimizer-native-20261113/new-gr00t-resume"
export EVALUATE_ONLY=0 UPDATES=8
# A completely published source resumes into a fresh output root.
export NATIVE_OUTPUT_ROOT=
# Only if the checkpoint has pending validation on this Lustre mount, use the
# original output root after preserving receipts: atomic directory promotion
# to a different root is unavailable.
sbatch tools/quantized_optimizers/gr00t_native.sbatch
```

The script performs the native GPU/model preflight, real simulator render/control
preflight, manifest validation, then the native entrypoint. Direct native CLI:

```bash
python -m cosmos_framework.posttrain.entrypoints.joint_ppo \
  --config cosmos_framework/posttrain/configs/gr00t_n1d7_libero_quantized_optimizer.py \
  -- rl.optimizer.name=adamw_bnb_8bit model.checkpoint_path=/approved/GR00T/libero_spatial \
  rl.max_steps=8 rl.save_every=4 rl.keep_optim_last=0 rl.keep_model_last=0
python -m cosmos_framework.posttrain.entrypoints.joint_ppo \
  --config cosmos_framework/posttrain/configs/gr00t_n1d7_libero_quantized_optimizer.py \
  --evaluate-only -- rl.resume=/exact/checkpoint/iter_000004
```

Use matching model/recipe/topology overrides for reload and resume. Evaluation-only
requires enabled validation and a complete checkpoint. It publishes evaluation
results after all native state/provenance checks, with no training or checkpoint
rewrite. It checks the restored generation before publishing scores. On the tested
Lustre mount, pending-validation training recovery must use the original run
root: atomic no-replace directory promotion is unsupported and fails before
model setup. The native collector declares environment reset on resume; exact physical
simulator continuation is not promised.

## Native AlpaGym

```bash
python -m cosmos_framework.posttrain.entrypoints.alpagym_clrl \
  --resolved-config /approved/AlpaGym/resolved_config.yaml --gpus 4 --steps 8 \
  --group-size <resolved-group-size> --optimizer adamw_bnb_8bit \
  --optimizer-update-policy fp32_master
```

The external AlpaGym runtime, resolved driving scenarios and AlpaSim assets have
not been located. This construction command is not an executed driving run or a
complete recovery recipe. Reference-policy and simulator/control recovery require
that external lifecycle; do not use an SFT surrogate or call current integration
complete.


The resident paged chain38515804→38515815→38515818 uses source-v32 (`59eccff3`).
The superseded pending v30 jobs and mistaken pending checkpoint-path submissions
were canceled before allocation and remain in the ledger. The native result
path is `runs/cosmos3_action/joint_ppo/gr00t-master-<optimizer>/integration`.

The following exposure settings were locked before the development outcomes.
Use source-v39 (`ef9f6958`) for the corrected publication implementation:

```bash
export SOURCE_ARCHIVE="$R/cosmos-optimizer-native-20261113/source-v39.tar.gz"
export TRAINING_SEED=17 UPDATES=80 CHECKPOINT_EVERY=40 VALIDATION_EVERY=40
export CO_RESIDENT_GIB=0 EVALUATE_ONLY=0 RESUME_CHECKPOINT= NATIVE_OUTPUT_ROOT=
# Set a fresh RESULTS for each optimizer. Keep the existing resource requests.
sbatch --time=120 tools/quantized_optimizers/gr00t_native.sbatch
# Fresh-process final state evaluation uses the same recipe and overrides,
# EVALUATE_ONLY=1, a new RESULTS, and RESUME_CHECKPOINT=.../iter_000080.
```

Historical admission IDs38520212/213/214/215 were superseded after final
publication failed in38520212; the dependent jobs never allocated. They followed
pending-checkpoint recovery38520018. Source-v35 job38516570
failed update40 publication; its99/100 initial and93/100 midpoint scores remain
recorded. Its unallocated dependent jobs and earlier source-v33 jobs were canceled
and remain in the ledger. No numerical or quality threshold changed.
This is a single paired development seed, with no final statistical speed/quality
claim. See `gr00t_exposure_development_protocol.json`; neither the 80-update horizon
nor a later checkpoint substitutes for a historical WTS endpoint.


The affected v3 native continuation uses source-v39, including the read-only
pre-update learner/cursor receipt introduced at`dcf57752`. Numerical behavior
is unchanged from v35:

```bash
export SOURCE_ARCHIVE="$R/cosmos-optimizer-native-20261113/source-v39.tar.gz"
export TRAINING_SEED=17 UPDATES=80 CHECKPOINT_EVERY=40 VALIDATION_EVERY=40
export CO_RESIDENT_GIB=0 EVALUATE_ONLY=0 NATIVE_OUTPUT_ROOT=
export RESUME_CHECKPOINT=/exact/reference/output/checkpoints/iter_000040
# Set OPTIMIZER_NAME to match that reference and a fresh RESULTS directory.
sbatch tools/quantized_optimizers/gr00t_native.sbatch
```

Nonpaged admission38526679 follows the fresh references/reloads;38520216 never
allocated. Fused admission
38520018 recovers the original pending checkpoint40 in its original output root
after preserving all control/evaluation receipts; it is not an uninterrupted
quality/timing control. The superseded38518227/229 and38517262 never allocated.
Before the next rollout, `resume-state-step-40.json` must match
the retained learner state and collector boundary. New physical trajectories
are expected after the declared environment reset; final parameter equality is
not the native simulator contract.

The fixed pressure protocol is `gr00t_pressure_protocol.json`: 32 updates,
checkpoints/evaluation at 16/32, then matched `CO_RESIDENT_GIB=0` or `55` runs.
The reservation is synthetic co-resident CUDA memory, separately reported from
model/optimizer allocations. Pressure dependencies and profiling are admitted
only under the protocol's stated prerequisites; a successful resident run does
not establish migration.

AlpaGym additionally accepts `--optimizer-bnb-min-8bit-size` (default 4096) and
repeatable `--optimizer-bnb-fp32-state-name` exact logical names. These use the
same typed config and selector validation as the other native paths. Invalid,
duplicate or backend-incompatible options fail; runtime unmatched names are
rejected by shared construction after final model setup. No exception is enabled
by default. Sixteen local construction/CLI checks pass; they are control-flow
coverage, not real AlpaGym training evidence.

For the compatible AlpaGym high-precision control, select `--optimizer adamw --optimizer-impl fused --optimizer-update-policy fp32_master`. Omitting these flags retains the legacy direct-parameter/default torch route. Quantized modes reject a torch-only `--optimizer-impl`; sixteen local routing/validation checks pass. This command still needs the approved external native runtime and scenarios.

AlpaGym CLI selections are atomically recorded in `optimizer_config_resolved.json` beside its host resolved config before workers start. Worker logs separately report the constructed wrapper/backend classes and complete typed configuration. No external host config or environment secrets are copied into this artifact.


The earlier source-v39 correction (`ef9f6958`) introduced full-content local checkpoint
publication, retaining existing v1 load/copy provenance. Source-v37 fused38520212
finished all80 updates but failed final publication. Its original97/91/95 scores
remain recorded; no additional fused80 timing rerun is admitted after that
adaptation failure. Pending80 recovery38526258 completed with identical restored
learner/data, zero new updates and successful publication (fresh score97/100).

The completed source-v39 dependency order was38526258 → resident-paged32 `38526671` → nonpaged80
`38526673` → fused fresh80 `38526676` → nonpaged fresh80 `38526677` → nonpaged40-to-80
`38526679`. These source-v39 jobs supersede the canceled unallocated v37 chain.
Use the same commands above with `SOURCE_ARCHIVE=.../source-v39.tar.gz`, a fresh
RESULTS and the exact checkpoint source. For resident32, set `UPDATES=32`,
`CHECKPOINT_EVERY=16`, `VALIDATION_EVERY=16`, `OPTIMIZER_NAME=paged_adamw_8bit`,
`CO_RESIDENT_GIB=0`; keep the same4-GPU/32-CPU/256GiB request and routing.
The full-content change adds publication I/O; no mixed-source successful
end-to-end comparison is inferred. Pressure arms remain separately conditional
on the resident fresh-writer gate and unchanged55GiB protocol.

Resident32 job38526671 completed with new checkpoint16/32 publication. The first
pressure arm38528405 follows38526679. Run each pressure arm in a fresh RESULTS:

```bash
export SOURCE_ARCHIVE="$R/cosmos-optimizer-native-20261113/source-v39.tar.gz"
export TRAINING_SEED=17 UPDATES=32 CHECKPOINT_EVERY=16 VALIDATION_EVERY=16
export CO_RESIDENT_GIB=55 EVALUATE_ONLY=0 RESUME_CHECKPOINT= NATIVE_OUTPUT_ROOT=
# OPTIMIZER_NAME=adamw, then adamw_bnb_8bit, then paged_adamw_8bit.
# Admit the next independent arm only after classifying the preceding result.
sbatch tools/quantized_optimizers/gr00t_native.sbatch
```

The launcher holds the explicit reservation on every GPU throughout native
training. A completed pressure checkpoint16 is the source for read-only reload
and16-to-32 continuation, with the same reservation. If pressure fails, resident
checkpoint16 can establish resident recovery only; preserve the pressure failure.

The single-seed native quality/state/total-memory figure is reproduced from the
collected summary without new jobs:

```bash
python docs/posttrain/evidence/quantized_optimizers_round10/native_development_plot.py
```

It has no confidence intervals: one training seed cannot establish the requested
recipe-level quality or speed bounds. The failed fused publication remains labeled.

The completed-gradient checkpoint correction is source-v40 (`36610372`). The
corrected nonpaged pressure reference38533459 is admitted with the unchanged
32-update/55GiB settings. Its diagnostic launcher, `checkpoint_memory_diagnostic.sbatch`,
adds only the pinned33-test preflight and NCCL INFO files to the existing launcher.
Do not use this instrumented run for a speed claim:

```bash
export SOURCE_ARCHIVE="$R/cosmos-optimizer-native-20261113/source-v40.tar.gz"
export OPTIMIZER_NAME=adamw_bnb_8bit TRAINING_SEED=17 UPDATES=32
export CHECKPOINT_EVERY=16 VALIDATION_EVERY=16 CO_RESIDENT_GIB=55
export EVALUATE_ONLY=0 RESUME_CHECKPOINT= NATIVE_OUTPUT_ROOT=
# A fresh RESULTS is required; retain the same configured model/simulator paths.
sbatch docs/posttrain/evidence/quantized_optimizers_round10/checkpoint_memory_diagnostic.sbatch
```

The original nonpaged pressure attempt38532517 completed16updates but failed the
model DCP write with a rank-consistent NCCL CUDA error. It produced no complete
midpoint checkpoint. The correction releases consumed model/master gradients and
unused CUDA cache only after the completed-boundary guard. Partial accumulation
must still fail before any gradients are discarded. The earlier fused55 failure
occurs at update2, before this checkpoint-only change can execute.

The descriptor-bound publication implementation is source-v41 (`677e7e03`).
The existing native model/optimizer/control format is unchanged; each publication
payload is read twice and checked for identity/content stability. The exact
pressure launcher adds NCCL INFO files only:

```bash
export SOURCE_ARCHIVE="$R/cosmos-optimizer-native-20261113/source-v41.tar.gz"
export TRAINING_SEED=17 UPDATES=32 CHECKPOINT_EVERY=16 VALIDATION_EVERY=16
export CO_RESIDENT_GIB=55 EVALUATE_ONLY=0 RESUME_CHECKPOINT= NATIVE_OUTPUT_ROOT=
export OPTIMIZER_NAME=paged_adamw_8bit
export RESULTS="$R/cosmos-optimizer-native-20261113/new-paged-pressure-reference"
sbatch docs/posttrain/evidence/quantized_optimizers_round10/pressure_diagnostic.sbatch
# Only from a completed, published midpoint, after the reference is terminal:
export RESUME_CHECKPOINT="$RESULTS/runs/cosmos3_action/joint_ppo/gr00t-master-paged_adamw_8bit/integration/checkpoints/iter_000016"
export RESULTS="$R/cosmos-optimizer-native-20261113/new-paged-pressure-reload"
export EVALUATE_ONLY=1
sbatch docs/posttrain/evidence/quantized_optimizers_round10/pressure_diagnostic.sbatch
# After reload succeeds; retain the same source checkpoint and recipe:
export RESULTS="$R/cosmos-optimizer-native-20261113/new-paged-pressure-resume"
export EVALUATE_ONLY=0
sbatch docs/posttrain/evidence/quantized_optimizers_round10/pressure_diagnostic.sbatch
```

These commands are serial, not simultaneous submissions. All standard resource,
container, model and simulator variables above remain required. Read-only reload
must reproduce every saved learner field and collector state; continuation must
finish new updates and checkpoint32. Physical simulation resets are explicit.
If pressure fails, the same sequence with `CO_RESIDENT_GIB=0` and the retained
resident checkpoint16 qualifies resident recovery only.

Source-v41 pending16 recovery38534388 reproduced every learner/data field and
published the original midpoint. It then failed the first resumed backward with
a rank0 `OutOfMemoryError`, leaving three peers blocked; read-only stack evidence
was captured and the allocation terminated. No checkpoint32 or pressure recovery
pass is inferred. The later reporting-only correction `d3f7bf16` logs a failed
cell immediately while preserving the drain barrier;17 local checks pass.
It is not present in source-v41 accelerator runs and does not alter their math.


## Optimizer ownership correction

Source-v42 is `ff2572f266d9463c6cb820f883e514387aecd869`, archive SHA256
`748841d879ba76692462738e5c24130d218a9d4d1d5d0004c6a68162c87dded9`.
It removes per-instance tensor closures from the backend class without changing
numerical updates or checkpoint identities. The affected real-kernel gate uses
the same pinned model interpreter as native GR00T:

```bash
export SOURCE_ARCHIVE="$R/cosmos-optimizer-native-20261113/source-v42.tar.gz"
export MODEL_PYTHON="$R/cosmos-optimizer-native-20261113/action-runtime-v2/model/bin/python"
export RESULTS="$R/cosmos-optimizer-native-20261113/new-ownership-gate"
sbatch --partition=polar,polar3,polar4,grizzly \
  docs/posttrain/evidence/quantized_optimizers_round10/ownership_gate.sbatch
```

After that gate, corrected pressure continuation retains `UPDATES=32`,
`CHECKPOINT_EVERY=16`, `VALIDATION_EVERY=16`, `TRAINING_SEED=17`,
`CO_RESIDENT_GIB=55`, `EVALUATE_ONLY=0`, and empty `NATIVE_OUTPUT_ROOT`.
Use the matching fully published `iter_000016` from the original paged55 or
nonpaged55 reference and a fresh `RESULTS`; submit the existing native launcher.
This is recovery validation, not a new uninterrupted quality/timing comparison.
The previous first-resumed-backward OOMs remain recorded.


The conditional managed-memory trace uses the same source-v42 and native recipe,
with eight diagnostic updates and checkpoint/evaluation at4/8. Copy the committed
`paging_profile.py` to the owned cluster root as `profile-native-v1.py`; its hash
is frozen in `gr00t_pressure_protocol.json`. Submit `paging_profile.sbatch` only
after the pressure recovery prerequisite passes, with `CO_RESIDENT_GIB=55`,
`OPTIMIZER_NAME=paged_adamw_8bit`, `UPDATES=8`, `CHECKPOINT_EVERY=4`,
`VALIDATION_EVERY=4`, empty resume/root overrides and a fresh RESULTS. The profiler
records CUDA/NVTX and unified-memory faults during the fixed600–1200second
window, exports SQLite, and continues required evaluation/publication afterward.
The instrumented run is not a timing comparison; native memory sampling excludes
profiler ancestors. No inherited credential variables are forwarded.

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


The latest correction is source-v39 (`ef9f6958`): full-content local checkpoint
publication, retaining existing v1 load/copy provenance. Source-v37 fused38520212
finished all80 updates but failed final publication. Its original97/91/95 scores
remain recorded; no additional fused80 timing rerun is admitted after that
adaptation failure. Pending80 recovery38526258 completed with identical restored
learner/data, zero new updates and successful publication (fresh score97/100).

Current dependency order is38526258 → resident-paged32 `38526671` → nonpaged80
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

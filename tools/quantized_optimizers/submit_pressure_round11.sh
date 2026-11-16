#!/usr/bin/env bash
# Round-11 fixed-pressure GR00T submissions. Run on the cluster login node.
# Usage: ARM=<adamw|adamw_bnb_8bit|paged_adamw_8bit> MODE=<resume16|fresh|profile> RUN_NAME=<new dir> \
#        SOURCE_ARCHIVE=<archive> submit_pressure_round11.sh
# Recipe, pressure (55 GiB), seed, topology and checkpoints are unchanged from round 10;
# only the source archive differs. No allocator environment is set.
set -euo pipefail
R=/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam
ROOT=$R/cosmos-optimizer-native-20261113
: "${ARM:?}" "${MODE:?}" "${RUN_NAME:?}" "${SOURCE_ARCHIVE:?}"
case "$ARM" in adamw|adamw_bnb_8bit|paged_adamw_8bit) ;; *) exit 2;; esac
export SOURCE_ARCHIVE
export RESOURCE_ROOT="$R" MODEL_PYTHON="$ROOT/action-runtime-v2/model/bin/python" SIM_PYTHON="$ROOT/sim-runtime-v1/environment/bin/python"
export CONTAINER_IMAGE="$R/cosmos-automl-focused-20261012/image-conversion/cosmos-framework-focused.sqsh"
export RESULTS="$ROOT/r11-pressure/$RUN_NAME" OPTIMIZER_NAME="$ARM" TRAINING_SEED=17
export CO_RESIDENT_GIB=55 EVALUATE_ONLY=0 NATIVE_OUTPUT_ROOT=
script="$ROOT/gr00t-native-diagnostics-v2.sbatch"
case "$MODE" in
 resume16)
  export UPDATES=32 CHECKPOINT_EVERY=16 VALIDATION_EVERY=16
  case "$ARM" in
   adamw_bnb_8bit) source_run=gr00t-pressure-eightbit55-v2 ;;
   paged_adamw_8bit) source_run=gr00t-pressure-paged55-v1 ;;
   *) echo "no published round-10 pressure checkpoint for $ARM" >&2; exit 2 ;;
  esac
  export RESUME_CHECKPOINT="$ROOT/$source_run/runs/cosmos3_action/joint_ppo/gr00t-master-$ARM/integration/checkpoints/iter_000016"
  test -d "$RESUME_CHECKPOINT"
  ;;
 fresh)
  export UPDATES=32 CHECKPOINT_EVERY=16 VALIDATION_EVERY=16 RESUME_CHECKPOINT=
  ;;
 profile)
  # Round-10 diagnostic shape: 8 updates, capture 600 s after a 600 s delay.
  export UPDATES=8 CHECKPOINT_EVERY=4 VALIDATION_EVERY=4 RESUME_CHECKPOINT=
  export PROFILE_VARIANT="${PROFILE_VARIANT:?}"
  script="$ROOT/r11-pressure/paging_profile.sbatch"
  ;;
 *) exit 2 ;;
esac
test ! -e "$RESULTS"
mkdir -p "$RESULTS"
sbatch --parsable --output="$RESULTS/slurm-%j.out" "$script"

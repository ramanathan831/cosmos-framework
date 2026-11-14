#!/bin/bash
#SBATCH --job-name=optimizer-action-tests
#SBATCH --partition=cpu_short
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=15
#SBATCH --signal=B:TERM@60
#SBATCH --no-requeue
set -euo pipefail
R=/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam
ROOT="$R/cosmos-optimizer-native-20261113"
scontrol show job "$SLURM_JOB_ID" > "$ROOT/tests-v15-job.txt"
srun --kill-on-bad-exit=1 --container-image="$R/cosmos-automl-focused-20261012/image-conversion/cosmos-framework-focused.sqsh" --container-mounts=/lustre:/lustre --container-writable bash -c '
set -euo pipefail
R=/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam
ROOT="$R/cosmos-optimizer-native-20261113"
ENV_DIR="$ROOT/action-runtime-v2/model"
unset UV_OVERRIDE UV_CONSTRAINT
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/action-runtime-v1/source" UV_CACHE_DIR="$ROOT/action-runtime-v1/cache"
RESULTS="$ROOT/action-tests-v15"
mkdir -p "$RESULTS/source"
tar -xzf "$ROOT/source-v38.tar.gz" -C "$RESULTS/source"
export RESULTS RESOURCE_ROOT="$R" HF_HOME="$RESULTS/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$RESULTS/source"
cd "$RESULTS/source"
export CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
"$ENV_DIR/bin/python" -m pytest cosmos_framework/posttrain/tests/test_entrypoints_gr00t_lifecycle.py cosmos_framework/posttrain/tests/test_checkpoint_synchronized_stat.py cosmos_framework/posttrain/tests/test_distributed_validation_publication.py cosmos_framework/posttrain/tests/test_checkpoint_archive_marker.py cosmos_framework/posttrain/tests/test_checkpoint_manager.py cosmos_framework/posttrain/tests/test_checkpoint_store.py cosmos_framework/posttrain/tests/test_optimizer_master.py --confcutdir=cosmos_framework/posttrain/tests --override-ini=addopts= -p no:env -q --tb=short
"$ENV_DIR/bin/python" "$ROOT/content-probe.py" "$ROOT/gr00t-exposure-fused-seed17-v3/runs/cosmos3_action/joint_ppo/gr00t-master-adamw/integration/checkpoints/iter_000080" "$RESULTS"
'

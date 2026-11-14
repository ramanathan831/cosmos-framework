#!/bin/bash
#SBATCH --job-name=optimizer-worker-imports
#SBATCH --partition=cpu_short
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=20
#SBATCH --signal=B:TERM@60
#SBATCH --no-requeue
set -euo pipefail
R=/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam
ROOT="$R/cosmos-optimizer-native-20261113"
scontrol show job "$SLURM_JOB_ID" > "$ROOT/worker-imports-v2-job.txt"
srun --kill-on-bad-exit=1 --container-image="$R/cosmos-automl-focused-20261012/image-conversion/cosmos-framework-focused.sqsh" --container-mounts=/lustre:/lustre --container-writable bash -c '
set -euo pipefail
R=/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam
ROOT="$R/cosmos-optimizer-native-20261113"
ENV_DIR="$ROOT/action-runtime-v1/model"
unset UV_OVERRIDE UV_CONSTRAINT
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/action-runtime-v1/source" UV_CACHE_DIR="$ROOT/action-runtime-v1/cache"
RESULTS="$ROOT/worker-imports-v2"
mkdir -p "$RESULTS/source"
tar -xzf "$ROOT/source-v13.tar.gz" -C "$RESULTS/source"
export RESULTS RESOURCE_ROOT="$R" HF_HOME="$RESULTS/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$RESULTS/source"
export PYTHONPATH="$RESULTS/source:$ENV_DIR/lib/python3.12/site-packages"
cd "$RESULTS/source"
export CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
"$ENV_DIR/bin/python" -m tools.quantized_optimizers.action_runtime --ray-worker --output "$RESULTS/imports.json"
'

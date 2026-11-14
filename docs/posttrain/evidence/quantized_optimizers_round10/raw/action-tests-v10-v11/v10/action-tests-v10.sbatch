#!/bin/bash
#SBATCH --job-name=optimizer-action-tests
#SBATCH --partition=cpu_short
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10
#SBATCH --signal=B:TERM@60
#SBATCH --no-requeue
set -euo pipefail
R=/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam
ROOT="$R/cosmos-optimizer-native-20261113"
scontrol show job "$SLURM_JOB_ID" > "$ROOT/tests-v10-job.txt"
srun --kill-on-bad-exit=1 --container-image="$R/cosmos-automl-focused-20261012/image-conversion/cosmos-framework-focused.sqsh" --container-mounts=/lustre:/lustre --container-writable bash -c '
set -euo pipefail
R=/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam
ROOT="$R/cosmos-optimizer-native-20261113"
ENV_DIR="$ROOT/action-runtime-v2/model"
unset UV_OVERRIDE UV_CONSTRAINT
export NO_ALBUMENTATIONS_UPDATE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$ROOT/action-runtime-v1/source" UV_CACHE_DIR="$ROOT/action-runtime-v1/cache"
RESULTS="$ROOT/action-tests-v10"
mkdir -p "$RESULTS/source"
tar -xzf "$ROOT/source-v31.tar.gz" -C "$RESULTS/source"
export RESULTS RESOURCE_ROOT="$R" HF_HOME="$RESULTS/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$RESULTS/source"
cd "$RESULTS/source"
export CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
set +e
/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys --version > "$RESULTS/nsys-version.txt" 2>&1
printf "%s\n" "$?" > "$RESULTS/nsys-version-exit.txt"
/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys profile --help > "$RESULTS/nsys-profile-help.txt" 2>&1
printf "%s\n" "$?" > "$RESULTS/nsys-help-exit.txt"
set -e
"$ENV_DIR/bin/python" -m pytest cosmos_framework/posttrain/tests/test_entrypoints_gr00t_lifecycle.py cosmos_framework/posttrain/tests/test_optimizer_master.py tools/quantized_optimizers/test_process_memory.py tools/quantized_optimizers/test_device_reservation.py --confcutdir=cosmos_framework/posttrain/tests --override-ini=addopts= -p no:env -q --tb=short
'

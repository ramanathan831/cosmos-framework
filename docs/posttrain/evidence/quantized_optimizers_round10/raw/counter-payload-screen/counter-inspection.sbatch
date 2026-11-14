#!/bin/bash
#SBATCH --job-name=optimizer-state-accounting
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
ROOT=$R/cosmos-optimizer-native-20261113
export ROOT
RESULTS=$ROOT/fused-state-accounting-v1
export RESULTS
mkdir -p "$RESULTS"
scontrol show job "$SLURM_JOB_ID" > "$RESULTS/job.txt"
srun --kill-on-bad-exit=1 --container-image="$R/cosmos-automl-focused-20261012/image-conversion/cosmos-framework-focused.sqsh" --container-mounts=/lustre:/lustre --container-env=ROOT,RESULTS --container-writable bash -c '
set -euo pipefail
mkdir "$RESULTS/source"
tar -xzf "$ROOT/source-v34.tar.gz" -C "$RESULTS/source"
export PYTHONPATH="$RESULTS/source" PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$RESULTS/source"
"$ROOT/action-runtime-v2/model/bin/python" -m pytest cosmos_framework/posttrain/tests/test_optimizer_master.py --confcutdir=cosmos_framework/posttrain/tests --override-ini=addopts= -p no:env -q --tb=short
/usr/bin/time -v "$ROOT/action-runtime-v2/model/bin/python" "$ROOT/inspect-fused-state.py"
'

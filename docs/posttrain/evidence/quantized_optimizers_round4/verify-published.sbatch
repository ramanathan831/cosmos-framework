#!/bin/bash
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --partition=cpu_short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=3
#SBATCH --no-requeue
set -euo pipefail
ROOT=/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam/cosmos-optimizer-lr-20261105T1830
scontrol show job "$SLURM_JOB_ID" > "$ROOT/preparation-recovery/job-$SLURM_JOB_ID.txt"
trap 'status=$?; echo "$status" > "$ROOT/preparation-recovery/exit-code-$SLURM_JOB_ID.txt"' EXIT
python "$ROOT/verify_published_corpus.py" \
 --candidate "$ROOT/cold-preparation-1/trial0" \
 --reference /lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam/cosmos-optimizer-outcomes-20261104T1850/prepare-evaluation-v1/prepared-corpus \
 --manifest-sha256 ea2e64893ab3d7da0f4c8be33001ae99c96dfd2c54fe293449c29255cd1f2d70 \
 --output "$ROOT/preparation-recovery/verification.json"

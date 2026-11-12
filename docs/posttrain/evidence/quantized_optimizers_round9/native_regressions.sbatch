#!/usr/bin/env bash
#SBATCH --job-name=optimizer-r9-regressions
#SBATCH --partition=cpu_short
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=15
#SBATCH --no-requeue
set -euo pipefail
mkdir -p "$RESULTS"
scontrol show job "$SLURM_JOB_ID" > "$RESULTS/job-$SLURM_JOB_ID.txt"
trap 'status=$?; echo "$status" > "$RESULTS/exit-code-$SLURM_JOB_ID.txt"' EXIT
srun --kill-on-bad-exit=1 --container-image="$CONTAINER_IMAGE" \
 --container-mounts=/lustre:/lustre --container-workdir="$SOURCE" --container-writable \
 --container-env=SOURCE,SOURCE_ARCHIVE,ASSETS,RESULTS,REFERENCE_CORPUS \
 timeout --signal=TERM --kill-after=60 780 bash -c '
 set -euo pipefail
 CAMPAIGN="$SOURCE"
 SOURCE=$(mktemp -d /tmp/cosmos-optimizer-hash.XXXXXX); export SOURCE
 tar -xzf "$SOURCE_ARCHIVE" -C "$SOURCE"
 export PATH="$ASSETS/environment/bin:$PATH" PYTHONPATH="$SOURCE" PYTHONDONTWRITEBYTECODE=1
 export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
 export SFT_MODEL_PATH="$ASSETS/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
 export SFT_DATA_ROOT="$CAMPAIGN/data"
 cd "$SOURCE"
 python -m pytest cosmos_framework/posttrain/tests/test_config_sft.py cosmos_framework/posttrain/tests/test_roles_env_sft.py cosmos_framework/posttrain/tests/test_entrypoints_sft.py cosmos_framework/posttrain/tests/test_sft_epoch_tail.py --confcutdir=cosmos_framework/posttrain/tests --override-ini=addopts= -p no:env -q > "$RESULTS/tests.txt" 2>&1
 echo complete > "$RESULTS/complete.txt"
 '

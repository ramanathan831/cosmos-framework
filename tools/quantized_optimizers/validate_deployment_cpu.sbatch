#!/usr/bin/env bash
#SBATCH --job-name=optimizer-regression-cpu
#SBATCH --partition=cpu_short
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=5
#SBATCH --no-requeue
set -euo pipefail
: "${SOURCE:?}" "${SOURCE_ARCHIVE:?}" "${ASSETS:?}" "${RESULTS:?}" "${CONTAINER_IMAGE:?}"
scontrol show job "$SLURM_JOB_ID" > "$RESULTS/job-$SLURM_JOB_ID.txt"
srun --kill-on-bad-exit=1 --container-image="$CONTAINER_IMAGE" \
 --container-mounts=/lustre:/lustre --container-workdir="$SOURCE" --container-writable \
 --container-env=SOURCE,SOURCE_ARCHIVE,ASSETS,RESULTS \
 bash -c '
 set -euo pipefail
 SOURCE=$(mktemp -d /tmp/cosmos-optimizer-regression.XXXXXX); export SOURCE
 tar -xzf "$SOURCE_ARCHIVE" -C "$SOURCE"
 export PATH="$ASSETS/environment/bin:$PATH" PYTHONPATH="$SOURCE" PYTHONDONTWRITEBYTECODE=1
 export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
 cd "$SOURCE"
 python -m pytest tools/quantized_optimizers/test_deployment.py \
  cosmos_framework/posttrain/tests/test_prepared_sft.py \
  cosmos_framework/posttrain/tests/test_config_sft.py \
  cosmos_framework/posttrain/tests/test_entrypoints_sft.py \
  cosmos_framework/posttrain/tests/test_config_optimizer_impl.py \
  cosmos_framework/posttrain/tests/test_config_optimizer_decay_groups.py \
  --confcutdir=cosmos_framework/posttrain/tests --override-ini="addopts=" -q \
  > "$RESULTS/tests.log" 2>&1
 '

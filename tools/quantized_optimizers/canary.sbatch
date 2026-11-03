#!/usr/bin/env bash
#SBATCH --job-name=quantized-optimizer-canary
#SBATCH --partition=polar,polar3,polar4,grizzly
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=60
#SBATCH --no-requeue
set -euo pipefail
: "${SOURCE:?}" "${ASSETS:?}" "${RESULTS:?}" "${CONTAINER_IMAGE:?}"
mkdir -p "$RESULTS"
scontrol show job "$SLURM_JOB_ID" > "$RESULTS/job.txt"
trap 'status=$?; echo "$status" > "$RESULTS/exit-code.txt"' EXIT
srun --kill-on-bad-exit=1 --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/lustre:/lustre --container-workdir="$SOURCE" --container-writable \
  --container-env=SOURCE,ASSETS,RESULTS,RUNTIME_OVERLAY,RUNTIME_ARCHIVE,STEPS,SEED,TELEMETRY,VAL_SAMPLES,SOURCE_ARCHIVE \
  bash -c 'if [[ -n "${SOURCE_ARCHIVE:-}" ]]; then SOURCE=$(mktemp -d /tmp/cosmos-optimizer-source.XXXXXX); export SOURCE; tar -xzf "$SOURCE_ARCHIVE" -C "$SOURCE"; fi; exec bash "$SOURCE/tools/quantized_optimizers/native_canary.sh"' 

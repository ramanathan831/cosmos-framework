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
  --container-env=SOURCE,ASSETS,RESULTS,RUNTIME_OVERLAY,RUNTIME_ARCHIVE,STEPS,SEED,TELEMETRY,VAL_SAMPLES,SOURCE_ARCHIVE,SFT_DATA_ROOT,DEPLOYMENT,TRAIN_KEYS,FRAMES,VAL_EVERY,CAPACITY_FILE,SEEDS,VAL_BATCH,SLURM_JOB_GPUS,ARM_ONLY,DEVELOPMENT_DRIVER,VIDEO_PIXELS,VIDEO_MIN_PIXELS,WORK_SECONDS,CAPACITY_ONLY,CAPACITY_ARMS,GRAD_CKPT,CAPACITY_FOLLOWUP \
  bash -c 'if [[ -n "${SOURCE_ARCHIVE:-}" ]]; then SOURCE=$(mktemp -d /tmp/cosmos-optimizer-source.XXXXXX); export SOURCE; tar -xzf "$SOURCE_ARCHIVE" -C "$SOURCE"; fi; if [[ "${CAPACITY_FOLLOWUP:-0}" == 1 ]]; then export PATH="$ASSETS/environment/bin:$PATH"; exec python "$SOURCE/tools/quantized_optimizers/capacity_followup.py"; fi; if [[ "${DEVELOPMENT_DRIVER:-0}" == 1 ]]; then export PATH="$ASSETS/environment/bin:$PATH"; exec python "$SOURCE/tools/quantized_optimizers/run_development.py"; fi; exec bash "$SOURCE/tools/quantized_optimizers/native_canary.sh"'

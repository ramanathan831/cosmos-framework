#!/usr/bin/env bash
#SBATCH --job-name=optimizer-deployment
#SBATCH --partition=polar,grizzly
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=45
#SBATCH --no-requeue
set -euo pipefail
: "${SOURCE:?}" "${SOURCE_ARCHIVE:?}" "${ASSETS:?}" "${RESULTS:?}" "${CONTAINER_IMAGE:?}" "${PRIOR:?}" "${PREPARATION:?}"
mkdir -p "$RESULTS"
scontrol show job "$SLURM_JOB_ID" > "$RESULTS/job-$SLURM_JOB_ID.txt"
trap 'status=$?; echo "$status" > "$RESULTS/exit-code-$SLURM_JOB_ID.txt"' EXIT
srun --kill-on-bad-exit=1 --container-image="$CONTAINER_IMAGE" \
 --container-mounts=/lustre:/lustre --container-workdir="$SOURCE" --container-writable \
 --container-env=SOURCE,SOURCE_ARCHIVE,ASSETS,RESULTS,PRIOR,PREPARATION,MODE,TRIAL_SEED,SLURM_JOB_ID,SLURM_JOB_NODELIST,WORK_SECONDS \
 bash -c '
 set -euo pipefail
 SOURCE=$(mktemp -d /tmp/cosmos-optimizer-deployment.XXXXXX); export SOURCE
 tar -xzf "$SOURCE_ARCHIVE" -C "$SOURCE"
 export PATH="$ASSETS/environment/bin:$PATH" PYTHONPATH="$SOURCE" PYTHONDONTWRITEBYTECODE=1
 export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled OMP_NUM_THREADS=8
 export SFT_MODEL_PATH="$ASSETS/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
 export SFT_DATA_ROOT="$PRIOR/recording-clean-v4"
 cd "$SOURCE"
 exec python -m tools.quantized_optimizers.run_deployment
 '

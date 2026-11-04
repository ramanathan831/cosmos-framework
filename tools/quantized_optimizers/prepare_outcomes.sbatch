#!/usr/bin/env bash
#SBATCH --job-name=optimizer-outcomes-prepare
#SBATCH --partition=cpu_short
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=45
#SBATCH --no-requeue
set -euo pipefail
: "${SOURCE:?}" "${SOURCE_ARCHIVE:?}" "${ASSETS:?}" "${RESULTS:?}" "${CONTAINER_IMAGE:?}"
scontrol show job "$SLURM_JOB_ID" > "$RESULTS/job-$SLURM_JOB_ID.txt"
trap 'status=$?; echo "$status" > "$RESULTS/exit-code-$SLURM_JOB_ID.txt"' EXIT
srun --kill-on-bad-exit=1 --container-image="$CONTAINER_IMAGE" \
 --container-mounts=/lustre:/lustre --container-workdir="$SOURCE" --container-writable \
 --container-env=SOURCE,SOURCE_ARCHIVE,ASSETS,RESULTS,PREVIOUS_CORPUS,EXTRA_FINAL_DATA bash -c '
 set -euo pipefail
 CAMPAIGN="$SOURCE"
 SOURCE=$(mktemp -d /tmp/cosmos-optimizer-outcomes.XXXXXX); export SOURCE
 tar -xzf "$SOURCE_ARCHIVE" -C "$SOURCE"
 export PATH="$ASSETS/environment/bin:$PATH" PYTHONPATH="$SOURCE" PYTHONDONTWRITEBYTECODE=1
 export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
 export SFT_MODEL_PATH="$ASSETS/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
 export SFT_DATA_ROOT="$CAMPAIGN/data"
 cd "$SOURCE"
 extra=()
 if [ -n "${PREVIOUS_CORPUS:-}" ]; then
   extra+=(--reuse-corpus "$PREVIOUS_CORPUS" --extra-data "${EXTRA_FINAL_DATA:?}")
 fi
 python -m tools.quantized_optimizers.prepare_corpus \
 --config cosmos_framework/posttrain/configs/sft_optimizer_deployment.py \
 --extra-data "$SFT_DATA_ROOT/train-mixed.jsonl" \
 --extra-data "$SFT_DATA_ROOT/dev-task.jsonl" \
 --extra-data "$SFT_DATA_ROOT/fixed-final.jsonl" \
 --output "$RESULTS/prepared-corpus" "${extra[@]}" > "$RESULTS/preparation.log" 2>&1
 '

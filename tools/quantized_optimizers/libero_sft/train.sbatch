#!/usr/bin/env bash
# Native Cosmos3-Nano LIBERO-10 action-policy SFT, one optimizer arm and seed.
# Re-submitting with the same RUN_NAME/OUTPUT_ROOT resumes from the latest
# published checkpoint (native DistributedCheckpointer auto-resume).
#SBATCH --job-name=optimizer-l10sft-train
#SBATCH --partition=polar,polar3,polar4,grizzly
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --mem=1000G
#SBATCH --time=235
#SBATCH --no-requeue
set -euo pipefail
: "${SOURCE_ARCHIVE:?}" "${RESULTS:?}" "${CONTAINER_IMAGE:?}" "${PYDEPS:?}" "${OPTIMIZER:?}" "${SEED:?}"
: "${RUN_NAME:?}" "${OUTPUT_ROOT:?}" "${LIBERO_ROOT:?}" "${BASE_CHECKPOINT_PATH:?}" "${WAN_VAE_PATH:?}"
export MAX_ITER="${MAX_ITER:-2000}" SAVE_ITER="${SAVE_ITER:-250}" EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
mkdir -p "$RESULTS"
date -u +%FT%TZ > "$RESULTS/allocation-start.txt"
scontrol show job "$SLURM_JOB_ID" > "$RESULTS/job.txt"
trap 'status=$?; echo "$status" > "$RESULTS/exit-code.txt"; date -u +%FT%TZ > "$RESULTS/allocation-end.txt"' EXIT
srun --kill-on-bad-exit=1 --container-image="$CONTAINER_IMAGE" --container-mounts=/lustre:/lustre \
 --container-env=SOURCE_ARCHIVE,RESULTS,PYDEPS,OPTIMIZER,SEED,RUN_NAME,OUTPUT_ROOT,LIBERO_ROOT,BASE_CHECKPOINT_PATH,WAN_VAE_PATH,MAX_ITER,SAVE_ITER,EXTRA_OVERRIDES \
 --container-writable bash -c '
set -euo pipefail
src=$(mktemp -d /tmp/l10sft-src.XXXXXX); tar -xzf "$SOURCE_ARCHIVE" -C "$src"; cd "$src"
export PYTHONPATH="$src:$PYDEPS" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$RESULTS/hf" WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 IMAGINAIRE_OUTPUT_ROOT="$OUTPUT_ROOT" COSMOS_OPTIMIZER_STATE_REPORT=1
export LD_LIBRARY_PATH="" TORCH_NCCL_ASYNC_ERROR_HANDLING=1
nvidia-smi --query-gpu=name,memory.total,uuid --format=csv > "$RESULTS/devices.csv"
python -c "import importlib.metadata as m; print({n: m.version(n) for n in (\"torch\", \"transformer-engine\", \"bitsandbytes\", \"flash-attn\")})" > "$RESULTS/versions.txt"
( while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader >> "$RESULTS/gpu-memory.csv"; date -u +%s >> "$RESULTS/gpu-memory.csv"; sleep 30; done ) &
monitor=$!
set +e
torchrun --standalone --nproc_per_node=8 -m cosmos_framework.scripts.train \
  --sft-toml=tools/quantized_optimizers/libero_sft/action_policy_libero_10_nano_8gpu.toml -- \
  job.name="$RUN_NAME" optimizer.optimizer_type="$OPTIMIZER" trainer.seed="$SEED" \
  dataloader_train.dataloader.datasets.libero.dataset.episode_shuffle_seed="$SEED" \
  trainer.max_iter="$MAX_ITER" checkpoint.save_iter="$SAVE_ITER" $EXTRA_OVERRIDES \
  > "$RESULTS/train.log" 2>&1
status=$?
kill $monitor
exit $status
'

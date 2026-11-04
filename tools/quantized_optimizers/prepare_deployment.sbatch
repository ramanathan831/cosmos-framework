#!/usr/bin/env bash
#SBATCH --job-name=optimizer-prepare-cpu
#SBATCH --partition=polar,grizzly
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=30
#SBATCH --no-requeue
set -euo pipefail
: "${SOURCE:?}" "${SOURCE_ARCHIVE:?}" "${ASSETS:?}" "${RESULTS:?}" "${CONTAINER_IMAGE:?}" "${PRIOR:?}"
scontrol show job "$SLURM_JOB_ID" > "$RESULTS/job-$SLURM_JOB_ID.txt"
srun --kill-on-bad-exit=1 --container-image="$CONTAINER_IMAGE" \
  --container-mounts=/lustre:/lustre --container-workdir="$SOURCE" --container-writable \
  --container-env=SOURCE,SOURCE_ARCHIVE,ASSETS,RESULTS,PRIOR \
  bash -c '
    set -euo pipefail
    SOURCE=$(mktemp -d /tmp/cosmos-optimizer-prepare.XXXXXX); export SOURCE
    tar -xzf "$SOURCE_ARCHIVE" -C "$SOURCE"
    export PATH="$ASSETS/environment/bin:$PATH" PYTHONPATH="$SOURCE" PYTHONDONTWRITEBYTECODE=1
    export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
    export SFT_MODEL_PATH="$ASSETS/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    export SFT_DATA_ROOT="$PRIOR/recording-clean-v4"
    cd "$SOURCE"
    python -m tools.quantized_optimizers.select_final \
      --inventory "$ASSETS/data/full/media_inventory.jsonl" \
      --validation "$ASSETS/data/full/validation.jsonl" \
      --campaign-manifest "$PRIOR/recording-clean-v4/manifest.json" \
      --exclude-selection "$PRIOR/final-v5-manifest.json" \
      --output "$RESULTS/final-selection" > "$RESULTS/selection.log" 2>&1
    python -m pytest cosmos_framework/posttrain/tests/test_prepared_sft.py \
      cosmos_framework/posttrain/tests/test_config_sft.py \
      cosmos_framework/posttrain/tests/test_entrypoints_sft.py \
      --confcutdir=cosmos_framework/posttrain/tests --override-ini="addopts=" -q \
      > "$RESULTS/cpu-tests.log" 2>&1
    python -m tools.quantized_optimizers.prepare_corpus \
      --config cosmos_framework/posttrain/configs/sft_optimizer_deployment.py \
      --extra-data "$RESULTS/final-selection/validation.jsonl" \
      --output "$RESULTS/prepared-corpus" > "$RESULTS/preparation.log" 2>&1
  '

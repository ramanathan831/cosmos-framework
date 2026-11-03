#!/usr/bin/env bash
set -euo pipefail
: "${SOURCE:?}" "${ASSETS:?}" "${RESULTS:?}"
export PATH="$ASSETS/environment/bin:/usr/local/bin:/usr/bin:/bin"
export PYTHONPATH="$SOURCE" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled OMP_NUM_THREADS=8
export SFT_MODEL_PATH="$ASSETS/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
export SFT_DATA_ROOT="$ASSETS/data/canary"
cd "$SOURCE"
mkdir -p "$RESULTS"
python -c 'import importlib.metadata; print("\n".join(sorted(f"{d.metadata["Name"]}=={d.version}" for d in importlib.metadata.distributions())))' > "$RESULTS/dependencies.txt"
nvidia-smi -q > "$RESULTS/device.txt"
for arm in adamw adamw_bnb_8bit paged_adamw_8bit; do
  export IMAGINAIRE_OUTPUT_ROOT="$RESULTS/$arm"
  impl=''
  if [[ "$arm" == adamw ]]; then impl=fused; fi
  date -u +%FT%TZ > "$RESULTS/$arm.start"
  torchrun --standalone --nproc-per-node=1 -m cosmos_framework.posttrain.entrypoints.sft \
    --config cosmos_framework/posttrain/configs/sft_quantized_optimizer.py -- \
    "job.name=$arm" rl.seed=17 rl.max_steps=3 rl.grad_accum=1 rl.save_every=0 \
    rl.max_length=4096 'rl.optimizer.train_keys=[layers.31.]' \
    "rl.optimizer.name=$arm" "rl.optimizer.impl=$impl" \
    rl.validation.max_samples=2 rl.validation.every_steps=3 \
    'rl.processor_kwargs.videos_kwargs.num_frames=4' \
    > "$RESULTS/$arm.log" 2>&1
  date -u +%FT%TZ > "$RESULTS/$arm.end"
done

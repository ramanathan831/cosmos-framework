#!/usr/bin/env bash
set -euo pipefail
: "${SOURCE:?}" "${ASSETS:?}" "${RESULTS:?}"
export PATH="$ASSETS/environment/bin:/usr/local/bin:/usr/bin:/bin"
if [[ -n "${RUNTIME_ARCHIVE:-}" ]]; then
  RUNTIME_OVERLAY=$(mktemp -d /tmp/cosmos-optimizer-runtime.XXXXXX)
  tar -xzf "$RUNTIME_ARCHIVE" -C "$RUNTIME_OVERLAY"
fi
export PYTHONPATH="$SOURCE${RUNTIME_OVERLAY:+:$RUNTIME_OVERLAY}" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled OMP_NUM_THREADS=8
export SFT_MODEL_PATH="$ASSETS/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
export SFT_DATA_ROOT="${SFT_DATA_ROOT:-$ASSETS/data/canary}"
cd "$SOURCE"
mkdir -p "$RESULTS"
python -c 'import importlib.metadata; print("\n".join(sorted(f"{d.metadata["Name"]}=={d.version}" for d in importlib.metadata.distributions())))' > "$RESULTS/dependencies.txt"
nvidia-smi -q > "$RESULTS/device.txt"
if [[ -n "${SEEDS:-}" ]]; then
  root_results=$RESULTS
  for seed in $SEEDS; do
    SEEDS='' SEED="$seed" RESULTS="$root_results/seed$seed" bash "$SOURCE/tools/quantized_optimizers/native_canary.sh"
  done
  exit 0
fi
monitor_pid=''
trap 'if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true; fi' EXIT
arms=(adamw adamw_bnb_8bit paged_adamw_8bit)
if [[ "${SEED:-17}" == 29 ]]; then arms=(adamw_bnb_8bit paged_adamw_8bit adamw); fi
if [[ "${SEED:-17}" == 43 ]]; then arms=(paged_adamw_8bit adamw adamw_bnb_8bit); fi
if [[ -n "${ARM_ONLY:-}" ]]; then arms=("$ARM_ONLY"); fi
if [[ "${DEPLOYMENT:-0}" == 1 && -n "${CAPACITY_FILE:-}" && ! -f "$CAPACITY_FILE" ]]; then
  python "$SOURCE/tools/quantized_optimizers/calibrate_capacity.py"
fi
for arm in "${arms[@]}"; do
  export PROBE_ARM="$arm"
  export IMAGINAIRE_OUTPUT_ROOT="$RESULTS/$arm"
  impl=''
  micro=1; accum=1; extra=()
  if [[ "${DEPLOYMENT:-0}" == 1 ]]; then
    micro=$(python -c 'import json,os; print(json.load(open(os.environ["CAPACITY_FILE"]))[os.environ["PROBE_ARM"]]["micro_batch"])')
    accum=$((4 / micro))
    if [[ "$arm" != adamw ]]; then extra=(rl.optimizer.bnb_min_8bit_size=16384); fi
  fi
  micro=${MICRO_OVERRIDE:-$micro}; accum=${ACCUM_OVERRIDE:-$accum}
  if [[ "${CALIBRATING:-0}" == 1 && "$arm" != adamw ]]; then extra=(rl.optimizer.bnb_min_8bit_size=16384); fi
  if [[ "$arm" == adamw ]]; then impl=fused; fi
  date -u +%FT%TZ > "$RESULTS/$arm.start"
  if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
    nvidia-smi --id="$SLURM_JOB_GPUS" --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,power.draw --format=csv --loop-ms=250 > "$RESULTS/$arm.device_samples.csv" 2> "$RESULTS/$arm.device_samples.err" &
    monitor_pid=$!
  fi
  entry=(-m cosmos_framework.posttrain.entrypoints.sft)
  if [[ "${TELEMETRY:-0}" == 1 ]]; then entry=("$SOURCE/tools/quantized_optimizers/native_probe.py"); fi
  torchrun --standalone --nproc-per-node=1 "${entry[@]}" \
    --config cosmos_framework/posttrain/configs/sft_quantized_optimizer.py -- \
    "job.name=$arm" "rl.seed=${SEED:-17}" "rl.max_steps=${STEPS:-3}" "rl.grad_accum=$accum" "rl.micro_batch_size=$micro" rl.save_every=0 \
    rl.max_length=4096 "rl.optimizer.train_keys=${TRAIN_KEYS:-[layers.31.]}" "${extra[@]}" \
    "rl.optimizer.name=$arm" "rl.optimizer.impl=$impl" \
    "rl.validation.batch_size=${VAL_BATCH:-1}" "rl.validation.max_samples=${VAL_SAMPLES:-2}" "rl.validation.every_steps=${VAL_EVERY:-${STEPS:-3}}" \
    "rl.processor_kwargs.videos_kwargs.num_frames=${FRAMES:-4}" \
    "rl.processor_kwargs.videos_kwargs.size.longest_edge=${VIDEO_PIXELS:-65536}" \
    "rl.processor_kwargs.videos_kwargs.size.shortest_edge=${VIDEO_MIN_PIXELS:-4096}" \
    > "$RESULTS/$arm.log" 2>&1
  date -u +%FT%TZ > "$RESULTS/$arm.end"
  if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; wait "$monitor_pid" 2>/dev/null || true; monitor_pid=''; fi
done

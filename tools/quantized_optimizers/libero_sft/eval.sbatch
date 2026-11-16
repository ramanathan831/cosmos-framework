#!/usr/bin/env bash
# Native closed-loop LIBERO-10 evaluation of one trained checkpoint (EMA weights, the
# server default). Five GPUs each serve one policy server; each client runs two
# tasks x TRIALS fixed LIBERO init states with 8 vectorized simulator envs.
#SBATCH --job-name=optimizer-l10sft-eval
#SBATCH --partition=polar,polar3,polar4,grizzly
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=5
#SBATCH --cpus-per-task=120
#SBATCH --mem=1000G
#SBATCH --time=235
#SBATCH --no-requeue
set -euo pipefail
: "${HF_HOME_SHARED:?}" "${SOURCE_ARCHIVE:?}" "${RESULTS:?}" "${CONTAINER_IMAGE:?}" "${RESOURCE_ROOT:?}" "${SIM_PYTHON:?}" "${SIM_OVERLAY:?}"
: "${CHECKPOINT:?}" "${WAN_VAE_PATH:?}"
export TRIALS="${TRIALS:-50}" EVAL_SEED="${EVAL_SEED:-0}" TASKS_PER_GPU="${TASKS_PER_GPU:-2}"
mkdir -p "$RESULTS"
date -u +%FT%TZ > "$RESULTS/allocation-start.txt"
scontrol show job "$SLURM_JOB_ID" > "$RESULTS/job.txt"
trap 'status=$?; echo "$status" > "$RESULTS/exit-code.txt"; date -u +%FT%TZ > "$RESULTS/allocation-end.txt"' EXIT
# Renderer closure identical to the qualified GR00T LIBERO runtime (round ten).
closure="$RESOURCE_ROOT/cosmos-adapter-all-family-2node-20261111/runtime-pi05-sim-closure-v1-attempt01/payload/rootfs"
family=/localhome/local-rarunachalam/cosmos-adapter-all-family-2node-20261111
python_base=/localhome/local-rarunachalam/cosmos-family-adapters-20261030/local-gpu-v1/flow-runtime/python/cpython-3.12.13-linux-x86_64-gnu
glvnd="$RESOURCE_ROOT/cosmos-adapter-all-family-2node-20261111/runtime-pi05-glvnd-frontend-v1-attempt01"
mounts="/lustre:/lustre,$closure$family:$family:ro,$closure$python_base:$python_base:ro"
for lib in libEGL.so.1 libGLdispatch.so.0 libOpenGL.so.0; do
 mounts="$mounts,$glvnd/$lib:/usr/lib/x86_64-linux-gnu/$lib:ro"
done
export NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display
srun --kill-on-bad-exit=1 --container-image="$CONTAINER_IMAGE" --container-mounts="$mounts" \
 --container-env=HF_HOME_SHARED,SOURCE_ARCHIVE,RESULTS,SIM_PYTHON,SIM_OVERLAY,CHECKPOINT,WAN_VAE_PATH,TRIALS,EVAL_SEED,TASKS_PER_GPU,NVIDIA_DRIVER_CAPABILITIES \
 --container-workdir=/tmp --container-writable bash -c '
set -euo pipefail
src=$(mktemp -d /tmp/l10sft-src.XXXXXX); tar -xzf "$SOURCE_ARCHIVE" -C "$src"; cd "$src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$HF_HOME_SHARED" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 LD_LIBRARY_PATH=""
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl __EGL_VENDOR_LIBRARY_FILENAMES="$RESULTS/egl.json"
export OMP_NUM_THREADS=2 NUMBA_CACHE_DIR=/tmp/numba XDG_CACHE_HOME=/tmp/xdg LIBERO_CONFIG_PATH=/tmp/libero-config
echo "{\"file_format_version\": \"1.0.0\", \"ICD\": {\"library_path\": \"libEGL_nvidia.so.0\"}}" > "$RESULTS/egl.json"
mkdir -p /tmp/libero-config && cp /localhome/local-rarunachalam/cosmos-adapter-all-family-2node-20261111/libero-renderer-v1/installed-attempt02/config/config.yaml /tmp/libero-config/config.yaml && touch /tmp/libero-config/.config_lock
nvidia-smi --query-gpu=name,memory.total,uuid --format=csv > "$RESULTS/devices.csv"
stats=cosmos_framework/data/generator/action/normalizer_stats/libero_native_frame_wise_relative_rot6d.json
pids=()
for g in 0 1 2 3 4; do
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH="$src" python -m cosmos_framework.scripts.action_policy_server_libero \
    --experiment action_policy_libero_nano --experiment-overrides "model.config.tokenizer.vae_path=$WAN_VAE_PATH" \
    --checkpoint-path "$CHECKPOINT" --action-normalization quantile_rot --action-stats-path "$stats" \
    --raw-action-dim 10 --fps 20 --seed "$EVAL_SEED" --port $((8100 + g)) --output-dir "/tmp/server-$g" \
    > "$RESULTS/server-$g.log" 2>&1 &
  pids+=($!)
done
start=$(date +%s)
clients=()
for g in 0 1 2 3 4; do
  tasks=$(python -c "print(\",\".join(str(t) for t in range($g * $TASKS_PER_GPU, min(10, ($g + 1) * $TASKS_PER_GPU))))")
  [ -n "$tasks" ] || continue
  PYTHONPATH="$src:$SIM_OVERLAY" "$SIM_PYTHON" cosmos_framework/simulation/libero/closed_loop_eval.py \
    --server_url http://localhost:$((8100 + g)) --wait_timeout 1800 --timeout 600 \
    --task_suite libero_10 --num_trials_per_task "$TRIALS" --task_ids "$tasks" --num_envs 8 \
    --camera agentview,wrist --image_size 256 --action_space frame_wise_relative --rotation_space 6d \
    --action_dim 10 --seed "$EVAL_SEED" --output_dir "$RESULTS/client-$g" > "$RESULTS/client-$g.log" 2>&1 &
  clients+=($!)
done
status=0
for p in "${clients[@]}"; do wait $p || status=1; done
echo $(( $(date +%s) - start )) > "$RESULTS/eval-seconds.txt"
kill "${pids[@]}" 2>/dev/null || true
exit $status
'

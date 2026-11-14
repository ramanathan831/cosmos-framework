#!/usr/bin/env bash
#SBATCH --job-name=optimizer-gr00t-native
#SBATCH --partition=polar,polar3,polar4,grizzly
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=90
#SBATCH --signal=B:USR1@600
#SBATCH --no-requeue
set -euo pipefail
: "${SOURCE_ARCHIVE:?}" "${RESULTS:?}" "${CONTAINER_IMAGE:?}" "${RESOURCE_ROOT:?}" "${MODEL_PYTHON:?}" "${SIM_PYTHON:?}"
mkdir -p "$RESULTS"
date -u +%FT%TZ > "$RESULTS/allocation-start.txt"
scontrol show job "$SLURM_JOB_ID" > "$RESULTS/job.txt"
trap 'status=$?; echo "$status" > "$RESULTS/exit-code.txt"; date -u +%FT%TZ > "$RESULTS/allocation-end.txt"' EXIT
export SLURM_LOG_DIR="$RESULTS"
export OPTIMIZER_NAME="${OPTIMIZER_NAME:-adamw}" RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}" UPDATES="${UPDATES:-8}"
export EVALUATE_ONLY="${EVALUATE_ONLY:-0}"
trap 'touch "$SLURM_LOG_DIR/SIGUSR1_RECEIVED"' USR1
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
 --container-env=SOURCE_ARCHIVE,RESULTS,RESOURCE_ROOT,MODEL_PYTHON,SIM_PYTHON,NVIDIA_DRIVER_CAPABILITIES,SLURM_LOG_DIR,OPTIMIZER_NAME,RESUME_CHECKPOINT,UPDATES,EVALUATE_ONLY \
 --container-workdir=/tmp --container-writable bash -c '
set -euo pipefail
source_dir=$(mktemp -d /tmp/optimizer-gr00t.XXXXXX)
tar -xzf "$SOURCE_ARCHIVE" -C "$source_dir"
export PATH="${MODEL_PYTHON%/*}:$PATH" PYTHONPATH="$source_dir" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export HF_HOME="$RESULTS/hf" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NO_ALBUMENTATIONS_UPDATE=1 TOKENIZERS_PARALLELISM=false
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl GR00T_DIT_SDPA_MODE=math CUBLAS_WORKSPACE_CONFIG=:4096:8
export GPUS_PER_NODE=4
export LIBERO_MANIFEST_ROOT="$RESULTS/manifests" LIBERO_CONFIG_PATH="$RESULTS/libero-config"
export POSTTRAIN_SIM_PYTHON="$SIM_PYTHON"
export RAY_TMPDIR=/tmp/optim-ray B1K_SIM_IPC_DIR=/tmp/optim-sim B1K_SHM_DIR=/tmp/optim-shm
export NUMBA_CACHE_DIR=/tmp/optim-numba XDG_CACHE_HOME=/tmp/optim-cache CUDA_CACHE_PATH=/tmp/optim-cuda-cache
export IMAGINAIRE_OUTPUT_ROOT="$RESULTS/runs" __EGL_VENDOR_LIBRARY_FILENAMES="$RESULTS/egl.json"
mkdir -p "$RAY_TMPDIR" "$B1K_SIM_IPC_DIR" "$B1K_SHM_DIR" "$NUMBA_CACHE_DIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH"
cd "$source_dir"
nvidia-smi --query-gpu=name,memory.total,uuid --format=csv > "$RESULTS/devices.csv"
extra=()
if [ "$EVALUATE_ONLY" = 1 ]; then extra+=(--evaluate-only); fi
python -m tools.quantized_optimizers.gr00t_native --optimizer "$OPTIMIZER_NAME" --resume "$RESUME_CHECKPOINT" --updates "$UPDATES" "${extra[@]}"
' &
step_pid=$!
set +e
while true; do
 wait "$step_pid"
 status=$?
 kill -0 "$step_pid" 2>/dev/null || break
done
exit "$status"

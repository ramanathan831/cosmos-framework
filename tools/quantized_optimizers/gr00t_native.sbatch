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
#SBATCH --signal=B:TERM@120
#SBATCH --no-requeue
set -euo pipefail
: "${SOURCE_ARCHIVE:?}" "${RESULTS:?}" "${CONTAINER_IMAGE:?}" "${RESOURCE_ROOT:?}" "${MODEL_PYTHON:?}"
mkdir -p "$RESULTS"
date -u +%FT%TZ > "$RESULTS/allocation-start.txt"
scontrol show job "$SLURM_JOB_ID" > "$RESULTS/job.txt"
trap 'status=$?; echo "$status" > "$RESULTS/exit-code.txt"; date -u +%FT%TZ > "$RESULTS/allocation-end.txt"' EXIT
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
 --container-env=SOURCE_ARCHIVE,RESULTS,RESOURCE_ROOT,MODEL_PYTHON,NVIDIA_DRIVER_CAPABILITIES \
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
export POSTTRAIN_SIM_PYTHON=/localhome/local-rarunachalam/cosmos-adapter-all-family-2node-20261111/runtime-gr00t-trace-5fbdf7d/bin/python
export RAY_TMPDIR=/tmp/optim-ray B1K_SIM_IPC_DIR=/tmp/optim-sim B1K_SHM_DIR=/tmp/optim-shm
export IMAGINAIRE_OUTPUT_ROOT="$RESULTS/runs" __EGL_VENDOR_LIBRARY_FILENAMES="$RESULTS/egl.json"
mkdir -p "$RAY_TMPDIR" "$B1K_SIM_IPC_DIR" "$B1K_SHM_DIR"
cd "$source_dir"
nvidia-smi --query-gpu=name,memory.total,uuid --format=csv > "$RESULTS/devices.csv"
python -m tools.quantized_optimizers.gr00t_native
'

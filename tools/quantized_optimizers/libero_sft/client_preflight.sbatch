#!/usr/bin/env bash
# CPU-only preflight: the eval client and LIBERO-10 suite import in the simulator runtime.
#SBATCH --job-name=optimizer-l10sft-clientcheck
#SBATCH --partition=cpu_short
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20
#SBATCH --no-requeue
set -euo pipefail
: "${SOURCE_ARCHIVE:?}" "${RESULTS:?}" "${CONTAINER_IMAGE:?}" "${RESOURCE_ROOT:?}" "${SIM_PYTHON:?}" "${SIM_OVERLAY:?}"
mkdir -p "$RESULTS"
closure="$RESOURCE_ROOT/cosmos-adapter-all-family-2node-20261111/runtime-pi05-sim-closure-v1-attempt01/payload/rootfs"
family=/localhome/local-rarunachalam/cosmos-adapter-all-family-2node-20261111
python_base=/localhome/local-rarunachalam/cosmos-family-adapters-20261030/local-gpu-v1/flow-runtime/python/cpython-3.12.13-linux-x86_64-gnu
mounts="/lustre:/lustre,$closure$family:$family:ro,$closure$python_base:$python_base:ro"
srun --container-image="$CONTAINER_IMAGE" --container-mounts="$mounts" \
 --container-env=SOURCE_ARCHIVE,RESULTS,SIM_PYTHON,SIM_OVERLAY --container-workdir=/tmp --container-writable bash -c '
set -euo pipefail
src=$(mktemp -d /tmp/l10sft-src.XXXXXX); tar -xzf "$SOURCE_ARCHIVE" -C "$src"; cd "$src"
mkdir -p /tmp/libero-config && cp /localhome/local-rarunachalam/cosmos-adapter-all-family-2node-20261111/libero-renderer-v1/installed-attempt02/config/config.yaml /tmp/libero-config/config.yaml
export LIBERO_CONFIG_PATH=/tmp/libero-config PYTHONPATH="$src:$SIM_OVERLAY"
"$SIM_PYTHON" - <<EOF 2>&1 | tee "$RESULTS/client-preflight.txt"
import sys
sys.argv = ["x"]
import cosmos_framework.simulation.libero.closed_loop_eval as m
from libero.libero import benchmark
suite = benchmark.get_benchmark_dict()["libero_10"]()
print("tasks", suite.n_tasks, [suite.get_task(i).language for i in range(suite.n_tasks)])
print("init_states", [len(suite.get_task_init_states(i)) for i in range(suite.n_tasks)])
print("max_steps", m.TASK_MAX_STEPS["libero_10"])
EOF
'

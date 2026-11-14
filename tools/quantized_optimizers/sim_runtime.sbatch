#!/usr/bin/env bash
#SBATCH --job-name=optimizer-sim-runtime
#SBATCH --partition=cpu_short
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=20
#SBATCH --signal=B:TERM@60
#SBATCH --no-requeue
set -euo pipefail
: "${SOURCE_ARCHIVE:?}" "${RESULTS:?}" "${CONTAINER_IMAGE:?}" "${RESOURCE_ROOT:?}" "${UV:?}"
mkdir -p "$RESULTS"
scontrol show job "$SLURM_JOB_ID" > "$RESULTS/job.txt"
date -u +%FT%TZ > "$RESULTS/allocation-start.txt"
trap 'status=$?; echo "$status" > "$RESULTS/exit-code.txt"; date -u +%FT%TZ > "$RESULTS/allocation-end.txt"' EXIT
closure="$RESOURCE_ROOT/cosmos-adapter-all-family-2node-20261111/runtime-pi05-sim-closure-v1-attempt01/payload/rootfs"
family=/localhome/local-rarunachalam/cosmos-adapter-all-family-2node-20261111
python_base=/localhome/local-rarunachalam/cosmos-family-adapters-20261030/local-gpu-v1/flow-runtime/python/cpython-3.12.13-linux-x86_64-gnu
srun --kill-on-bad-exit=1 --container-image="$CONTAINER_IMAGE" \
 --container-mounts="/lustre:/lustre,$closure$family:$family:ro,$closure$python_base:$python_base:ro" \
 --container-env=SOURCE_ARCHIVE,RESULTS,RESOURCE_ROOT,UV --container-writable bash -c '
set -euo pipefail
base=/localhome/local-rarunachalam/cosmos-adapter-all-family-2node-20261111
sim="$base/runtime-libero-renderer-31377dd-v2"
unset UV_OVERRIDE UV_CONSTRAINT
export UV_CACHE_DIR="$RESULTS/cache" PYTHONDONTWRITEBYTECODE=1
"$UV" --no-config venv --python "$sim/bin/python" "$RESULTS/environment"
printf "%s\n" "$sim/lib/python3.12/site-packages" > "$RESULTS/environment/lib/python3.12/site-packages/qualified-simulator.pth"
# Preserve the frozen closure, including its documented NumPy metadata conflict.
# Only this owned overlay is writable; no inherited package is changed.
"$UV" --no-config pip install --python "$RESULTS/environment/bin/python" --no-deps contourpy==1.4.0
mkdir -p "$RESULTS/source" "$RESULTS/config"
tar -xzf "$SOURCE_ARCHIVE" -C "$RESULTS/source"
cp "$base/libero-renderer-v1/installed-attempt02/config/config.yaml" "$RESULTS/config/config.yaml"
touch "$RESULTS/config/.config_lock"
export PYTHONPATH="$RESULTS/source" LIBERO_CONFIG_PATH="$RESULTS/config" RESULTS
"$RESULTS/environment/bin/python" - <<"PY"
import importlib.metadata, json, os
from pathlib import Path
from cosmos_framework.posttrain.embodied.sims.libero.protocol import runtime_packages
from cosmos_framework.posttrain.embodied.sims.libero.runtime import LiberoSimulator
LiberoSimulator.probe(object.__new__(LiberoSimulator))
Path(os.environ["RESULTS"], "identity.json").write_text(json.dumps({
 "packages": {name: importlib.metadata.version(name) for name in runtime_packages()},
 "source_and_packages_passed": True, "rendering_tested": False,
}, indent=2))
PY
'

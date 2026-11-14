#!/usr/bin/env bash
#SBATCH --job-name=optimizer-action-runtime
#SBATCH --partition=cpu_short
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=30
#SBATCH --signal=B:TERM@60
#SBATCH --no-requeue
set -euo pipefail
: "${SOURCE_ARCHIVE:?}" "${RESULTS:?}" "${CONTAINER_IMAGE:?}" "${ASSETS:?}" "${UV:?}"
mkdir -p "$RESULTS"
date -u +%FT%TZ > "$RESULTS/allocation-start.txt"
scontrol show job "$SLURM_JOB_ID" > "$RESULTS/job.txt"
trap 'status=$?; echo "$status" > "$RESULTS/exit-code.txt"; date -u +%FT%TZ > "$RESULTS/allocation-end.txt"' EXIT
srun --kill-on-bad-exit=1 --container-image="$CONTAINER_IMAGE" --container-mounts=/lustre:/lustre \
 --container-env=SOURCE_ARCHIVE,RESULTS,ASSETS,UV --container-writable bash -c '
set -euo pipefail
mkdir -p "$RESULTS/source" "$RESULTS/cache"
tar -xzf "$SOURCE_ARCHIVE" -C "$RESULTS/source"
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_CACHE_DIR="$RESULTS/cache"
"$UV" venv --python "$ASSETS/environment/bin/python" "$RESULTS/model"
# Read-only inherited packages; installs below go exclusively to this new venv.
printf "%s\n" "$ASSETS/environment/lib/python3.12/site-packages" > "$RESULTS/model/lib/python3.12/site-packages/qualified-base.pth"
export PATH="$RESULTS/model/bin:$PATH" PYTHONPATH="$RESULTS/source"
"$UV" pip install --python "$RESULTS/model/bin/python" --no-deps \
 transformers==4.57.6 huggingface-hub==0.36.2 tokenizers==0.22.2 diffusers==0.35.1 accelerate==1.12.0 \
 albumentations==1.4.18 albucore==0.0.17 cache-dit==1.3.0 aenum==3.1.16 janus==2.0.0 \
 torchsde==0.2.6 x-transformers==2.12.2 soundfile==0.13.1 \
 dm-tree==0.1.10 lmdb==1.7.5 msgpack-numpy==0.4.8 tyro==0.9.17 jsonlines==4.0.0 \
 gymnasium==1.2.2 shtab==1.7.2 peft==0.17.1 \
 "git+https://github.com/NVIDIA/Isaac-GR00T@51d4c89f72fda44cbf77285c6a8114b52676b8a1" \
 "git+https://github.com/vllm-project/vllm-omni@2d0b1d8a29ff686698dbf0a123ca15be20fb0a8f"
python "${SOURCE_ARCHIVE%/*}/action_runtime.py" --output "$RESULTS/imports.json"
'

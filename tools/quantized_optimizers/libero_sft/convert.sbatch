#!/usr/bin/env bash
# Convert the approved local nvidia/Cosmos3-Nano snapshot to the native training DCP.
#SBATCH --job-name=optimizer-l10sft-convert
#SBATCH --partition=polar,polar3,polar4,grizzly
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=60
#SBATCH --no-requeue
set -euo pipefail
: "${HF_HOME_SHARED:?}" "${SOURCE_ARCHIVE:?}" "${RESULTS:?}" "${CONTAINER_IMAGE:?}" "${SNAPSHOT:?}" "${OUTPUT:?}"
mkdir -p "$RESULTS"
date -u +%FT%TZ > "$RESULTS/allocation-start.txt"
scontrol show job "$SLURM_JOB_ID" > "$RESULTS/job.txt"
trap 'status=$?; echo "$status" > "$RESULTS/exit-code.txt"; date -u +%FT%TZ > "$RESULTS/allocation-end.txt"' EXIT
test ! -e "$OUTPUT" || { echo "refusing to overwrite $OUTPUT" >&2; exit 2; }
srun --kill-on-bad-exit=1 --container-image="$CONTAINER_IMAGE" --container-mounts=/lustre:/lustre \
 --container-env=HF_HOME_SHARED,SOURCE_ARCHIVE,RESULTS,SNAPSHOT,OUTPUT --container-writable bash -c '
set -euo pipefail
src=$(mktemp -d /tmp/l10sft-src.XXXXXX); tar -xzf "$SOURCE_ARCHIVE" -C "$src"; cd "$src"
export PYTHONPATH="$src" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$HF_HOME_SHARED" PYTHONDONTWRITEBYTECODE=1
python -m cosmos_framework.scripts.convert_model_to_dcp -o "$OUTPUT.partial" --checkpoint-path "$SNAPSHOT" 2>&1 | tee "$RESULTS/convert.log"
mv "$OUTPUT.partial" "$OUTPUT"
'

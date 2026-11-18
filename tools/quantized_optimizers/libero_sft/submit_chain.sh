#!/usr/bin/env bash
# Submit one protocol run as a dependency chain: one allocation per 250 iterations.
# Usage: submit_chain.sh SOURCE_VERSION OPTIMIZER SEED RUN_NAME [FIRST_K=1] [LAST_K=8]
# Each link resumes natively from the run's latest checkpoint (written at 250k).
set -euo pipefail
# GPU partitions come only from SLURM_PARTITION (read from ~/config.env by the caller).
: "${SLURM_PARTITION:?set SLURM_PARTITION from ~/config.env; no default partition is used}"
[[ "$SLURM_PARTITION" =~ ^[A-Za-z0-9_,]+$ ]] || { echo "invalid SLURM_PARTITION" >&2; exit 2; }
version=$1 optimizer=$2 seed=$3 run=$4 first=${5:-1} last=${6:-8}
here=$(cd "$(dirname "$0")" && pwd)
. "$here/env.sh"
export SOURCE_ARCHIVE=$W/source-$version.tar.gz OPTIMIZER=$optimizer SEED=$seed RUN_NAME=$run
export OUTPUT_ROOT=$W/runs SAVE_ITER=250
export EXTRA_OVERRIDES="dataloader_train.max_samples_per_batch=64 trainer.grad_accum_iter=4"
dependency=""
for k in $(seq "$first" "$last"); do
  export MAX_ITER=$((250 * k)) RESULTS=$W/train/$run/link-$(printf %02d "$k")
  mkdir -p "$W/train/$run"
  id=$(sbatch --parsable --partition="$SLURM_PARTITION" $dependency --job-name="optimizer-l10sft-$run" \
       --output="$W/train/$run/link-$(printf %02d "$k")-%j.log" "$here/train.sbatch")
  echo "$run link $k max_iter $MAX_ITER job $id"
  dependency="--dependency=afterok:$id"
done

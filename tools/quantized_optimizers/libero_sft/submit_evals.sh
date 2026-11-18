#!/usr/bin/env bash
# Submit one closed-loop LIBERO-10 evaluation per published 250-step checkpoint that has
# not been evaluated (or queued) yet. Idempotent: safe to re-run as checkpoints appear.
# Usage: submit_evals.sh EVAL_SOURCE_VERSION RUN_NAME...
set -euo pipefail
version=$1; shift
here=$(cd "$(dirname "$0")" && pwd)
. "$here/env.sh"
export SOURCE_ARCHIVE=$W/source-$version.tar.gz TRIALS=50
for run in "$@"; do
  ckpts=$W/runs/cosmos3_action_libero/optimizer_qualification/$run/checkpoints
  # Only checkpoints published at or before latest_checkpoint.txt (written after a
  # completed save) are eligible; an in-progress save directory is never evaluated.
  [ -f "$ckpts/latest_checkpoint.txt" ] || continue
  latest=$(sed 's/iter_0*//' "$ckpts/latest_checkpoint.txt")
  for k in 1 2 3 4 5 6 7 8; do
    [ $((250 * k)) -le "$latest" ] || continue
    iter=$(printf "iter_%09d" $((250 * k)))
    [ -d "$ckpts/$iter/model" ] || continue
    out=$W/eval/$run/$iter
    [ -e "$out" ] && continue   # evaluated, running, or queued: never resubmit silently
    mkdir -p "$out"
    export RESULTS=$out CHECKPOINT=$ckpts/$iter
    id=$(sbatch --parsable --time=120 --job-name="optimizer-l10sft-eval-$run" \
         --output="$W/eval/$run/$iter-%j.log" "$here/eval.sbatch")
    echo "$run $iter job $id"
  done
done

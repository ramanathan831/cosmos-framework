#!/usr/bin/env bash
# Stage a fresh-process resume run from a published checkpoint of a reference run.
# Usage: stage_resume.sh SOURCE_RUN_DIR ITER NEW_RUN_DIR
# Hard-links the checkpoint (same Lustre filesystem; the source stays immutable because
# DCP writes new iteration directories rather than modifying existing files) and points
# latest_checkpoint.txt at it, so native auto-resume restores exactly that boundary.
set -euo pipefail
src=$1 iter=$(printf "iter_%09d" "$2") dst=$3
test -d "$src/checkpoints/$iter" || { echo "missing $src/checkpoints/$iter" >&2; exit 2; }
test ! -e "$dst" || { echo "refusing to reuse $dst" >&2; exit 2; }
mkdir -p "$dst/checkpoints"
cp -al "$src/checkpoints/$iter" "$dst/checkpoints/$iter"
printf "%s" "$iter" > "$dst/checkpoints/latest_checkpoint.txt"
echo "staged $dst/checkpoints/$iter"

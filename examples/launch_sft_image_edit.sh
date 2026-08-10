#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

# Generator-tower paired-image SFT for Cosmos3 Edge, Nano, or Super.
#
# Usage (inside the framework container):
#   DATASET_PATH=/data/uc3-paired \
#   BASE_CHECKPOINT_PATH=/checkpoints/Cosmos3-Edge \
#   WAN_VAE_PATH=/checkpoints/wan22_vae/Wan2.2_VAE.pth \
#   NPROC_PER_NODE=4 bash examples/launch_sft_image_edit.sh edge

set -uo pipefail

MODEL_VARIANT="${1:-edge}"
case "$MODEL_VARIANT" in
    edge|nano|super) ;;
    *) echo "ERROR: model must be one of: edge, nano, super" >&2; exit 2 ;;
esac
if (( $# > 0 )); then
    shift
fi
TAIL_OVERRIDES=("$@")

TOML_FILE="examples/toml/sft_config/image_edit_sft_${MODEL_VARIANT}.toml"
: "${DATASET_PATH:=examples/data/uc3-paired}"
case "$MODEL_VARIANT" in
    edge)  : "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Edge}" ;;
    nano)  : "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Nano}" ;;
    super) : "${BASE_CHECKPOINT_PATH:=examples/checkpoints/Cosmos3-Super}" ;;
esac

EXTRA_DATASET_CHECK='[[ -f "$DATASET_PATH/train.jsonl" ]] || { echo "ERROR: missing $DATASET_PATH/train.jsonl" >&2; exit 1; }; [[ -f "$DATASET_PATH/val.jsonl" ]] || { echo "ERROR: missing $DATASET_PATH/val.jsonl" >&2; exit 1; }'

if [[ "$MODEL_VARIANT" == "super" ]]; then
    export LD_LIBRARY_PATH=""
    export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
fi

source "$(dirname "${BASH_SOURCE[0]}")/_sft_launcher_common.sh"

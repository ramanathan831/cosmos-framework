#!/usr/bin/env bash
#SBATCH --job-name=optimizer-native-master
#SBATCH --partition=polar
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=40
#SBATCH --signal=B:TERM@60
#SBATCH --no-requeue
# Round-11 gate for the Joint PPO update-memory change: the round-10 ownership gate
# (affected tests + two-rank fused/nonpaged/paged reference/resume), plus the Ray
# lifecycle/topology suites that cannot run outside the pinned runtime.
set -euo pipefail
: "${SOURCE_ARCHIVE:?}" "${RESULTS:?}" "${CONTAINER_IMAGE:?}" "${MODEL_PYTHON:?}"
mkdir -p "$RESULTS"
date -u +%FT%TZ > "$RESULTS/allocation-start.txt"
scontrol show job "$SLURM_JOB_ID" > "$RESULTS/job.txt"
trap 'status=$?; echo "$status" > "$RESULTS/exit-code.txt"; date -u +%FT%TZ > "$RESULTS/allocation-end.txt"' EXIT
srun --kill-on-bad-exit=1 --container-image="$CONTAINER_IMAGE" --container-mounts=/lustre:/lustre \
 --container-env=SOURCE_ARCHIVE,RESULTS,MODEL_PYTHON --container-writable bash -c '
set -euo pipefail
source_dir=$(mktemp -d /tmp/optimizer-master.XXXXXX)
tar -xzf "$SOURCE_ARCHIVE" -C "$source_dir"
export PATH="$(dirname "$MODEL_PYTHON"):$PATH" PYTHONPATH="$source_dir" PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled OMP_NUM_THREADS=4 NO_ALBUMENTATIONS_UPDATE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=42 RAY_TMPDIR=/tmp/optim-ray HF_HOME="$RESULTS/hf"
mkdir -p "$RAY_TMPDIR"
cd "$source_dir"
nvidia-smi --query-gpu=name,memory.total,uuid --format=csv > "$RESULTS/devices.csv"
status=0
"$MODEL_PYTHON" -m pytest cosmos_framework/posttrain/tests/test_joint_ppo_update_memory.py \
 cosmos_framework/posttrain/tests/test_quantized_optimizer.py cosmos_framework/posttrain/tests/test_optimizer_master.py \
 cosmos_framework/posttrain/tests/test_roles_ppo_continuous.py cosmos_framework/posttrain/tests/test_roles_ppo.py \
 cosmos_framework/posttrain/tests/test_algorithm_joint_ppo_boundaries.py \
 cosmos_framework/posttrain/tests/test_algorithm_joint_ppo_verification.py \
 cosmos_framework/posttrain/tests/test_algorithm_joint_ppo_fixed_episode.py \
 cosmos_framework/posttrain/tests/test_utils_entry.py cosmos_framework/posttrain/tests/test_workers_worker_group.py \
 --confcutdir=cosmos_framework/posttrain/tests --override-ini=addopts= -p no:env -q --tb=short \
 > "$RESULTS/affected-tests.log" 2>&1 || status=1
CUDA_VISIBLE_DEVICES="" "$MODEL_PYTHON" -m pytest cosmos_framework/posttrain/tests/test_entrypoints_gr00t_lifecycle.py \
 cosmos_framework/posttrain/tests/test_roles_joint_topology.py \
 --confcutdir=cosmos_framework/posttrain/tests --override-ini=addopts= -p no:env -q --tb=short \
 > "$RESULTS/ray-lifecycle-tests.log" 2>&1 || status=1
for optimizer in adamw adamw_bnb_8bit paged_adamw_8bit; do
 for phase in reference resume; do
  "$MODEL_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=2 -m tools.quantized_optimizers.master_distributed \
   --optimizer "$optimizer" --phase "$phase" --output "$RESULTS/$optimizer" \
   > "$RESULTS/$optimizer-$phase.log" 2>&1 || status=1
 done
done
exit "$status"
'

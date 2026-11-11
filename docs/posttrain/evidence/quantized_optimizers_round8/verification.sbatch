#!/usr/bin/env bash
#SBATCH --job-name=optimizer-r8-verification
#SBATCH --partition=cpu_short
#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=45
#SBATCH --no-requeue
set -euo pipefail
: "${SOURCE:?}" "${SOURCE_ARCHIVE:?}" "${ASSETS:?}" "${RESULTS:?}" "${CONTAINER_IMAGE:?}" "${REFERENCE_CORPUS:?}" "${CANDIDATE_CORPUS:?}" "${FINAL_DATA:?}" "${PREPARATION_SLICE:?}"
mkdir -p "$RESULTS"
scontrol show job "$SLURM_JOB_ID" > "$RESULTS/job-$SLURM_JOB_ID.txt"
trap 'status=$?; echo "$status" > "$RESULTS/exit-code-$SLURM_JOB_ID.txt"' EXIT
srun --kill-on-bad-exit=1 --container-image="$CONTAINER_IMAGE" \
 --container-mounts=/lustre:/lustre --container-workdir="$SOURCE" --container-writable \
 --container-env=SOURCE,SOURCE_ARCHIVE,ASSETS,RESULTS,REFERENCE_CORPUS,CANDIDATE_CORPUS,FINAL_DATA,PREPARATION_SLICE \
 timeout --signal=TERM --kill-after=60 2520 bash -c '
 set -euo pipefail
 CAMPAIGN="$SOURCE"
 SOURCE=$(mktemp -d /tmp/cosmos-optimizer-verify.XXXXXX); export SOURCE
 tar -xzf "$SOURCE_ARCHIVE" -C "$SOURCE"
 export PATH="$ASSETS/environment/bin:$PATH" PYTHONPATH="$SOURCE" PYTHONDONTWRITEBYTECODE=1
 export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
 export SFT_MODEL_PATH="$ASSETS/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
 export SFT_DATA_ROOT="$CAMPAIGN/data"
 cd "$SOURCE"
 python -m pytest tools/quantized_optimizers/test_verify_published_corpus.py tools/quantized_optimizers/test_preparation_slice.py \
  --confcutdir=tools/quantized_optimizers --override-ini=addopts= -q > "$RESULTS/tests.txt" 2>&1
 for trial in tensors0 bytes0 bytes1 tensors1 tensors2 bytes2; do
  mode=${trial%[0-9]}
  python - "$mode" "$trial" << "PY"
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(os.environ["RESULTS"]); started=time.monotonic()
with (root/(sys.argv[2]+".log")).open("x") as log:
 result=subprocess.run([sys.executable,"docs/posttrain/evidence/quantized_optimizers_round8/verification_screen.py","--mode",sys.argv[1],"--output",str(root/(sys.argv[2]+".json"))],stdout=log,stderr=subprocess.STDOUT)
(root/(sys.argv[2]+".process.json")).write_text(json.dumps({"elapsed_seconds":time.monotonic()-started,"returncode":result.returncode})+"\n")
raise SystemExit(result.returncode)
PY
 done
 python -m tools.quantized_optimizers.parallel_prepare \
  --config cosmos_framework/posttrain/configs/sft_optimizer_deployment.py \
  --workers 4 --reference-corpus "$REFERENCE_CORPUS" --fixed-slice "$PREPARATION_SLICE" \
  --video-cache-bytes 67108864 --reference-verification bytes \
  --output "$RESULTS/fresh-slice" > "$RESULTS/fresh-slice.log" 2>&1
 echo complete > "$RESULTS/complete.txt"
 '

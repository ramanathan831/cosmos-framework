# Reproduce the bounded continuation

The cumulative allowance begins at `40885d9`; this continuation starts from clean
`582ee70`. The launch receipts retain exact Slurm arguments, nodes, paths and wall
limits. Use fresh owned outputs when reproducing. These commands are a record,
not permission for additional allocations. Original QLoRA assets are read-only.

The LR 8.75e-6 candidate failed its fused screen. Do not run its conditional
eight-bit or final comparison. The single subsequent objective-weighting probe
uses original LR 1e-5 and `rl.ce_exponent=0.95`; its protocol and outcome are separate.
Neither candidate changes the default native recipe.

```bash
export R=/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam
export ROOT=$R/cosmos-optimizer-r5-20261105
export OLD=$R/cosmos-optimizer-outcomes-20261104T1850
export SOURCE=$OLD ASSETS=$R/cosmos-qlora-wts-20261028T234324Z
export PRIOR=$R/cosmos-quantized-optimizers-20261103
export PREPARATION=$OLD/prepare-evaluation-v1
export CONTAINER_IMAGE=$R/cosmos-automl-focused-20261012/image-conversion/cosmos-framework-focused.sqsh
export TASK_DATA_ROOT=$OLD/data
export SOURCE_ARCHIVE=$ROOT/source-baseline.tgz
export TASK_SPEC=$ROOT/fused_lr875_probe.json RESULTS=$ROOT/lr875-fused
export MODE=study WORK_SECONDS=960
sbatch --parsable --export=ALL --time=18 --nodelist=batch-block4-1051 \
  --partition=interactive_singlenode,polar --output="$ROOT/lr875-fused-%j.log" \
  "$ROOT/deployment.sbatch"
# The second admitted shared factor uses the same source, limits and node:
export TASK_SPEC=$ROOT/fused_weight95_probe.json RESULTS=$ROOT/weight95-fused
sbatch --parsable --export=ALL --time=18 --nodelist=batch-block4-1051 \
  --partition=interactive_singlenode,polar --output="$ROOT/weight95-fused-%j.log" \
  "$ROOT/deployment.sbatch"
```

The specs contain exact native `torchrun` invocations, seeds 17/29/43, 15 updates,
40-update cosine horizon, FP32 trainables, BF16 compute, effective batch four,
full fused-baseline recomputation and unchanged mixed rows/clipping. The weighted
candidate changes only the native training exponent; validation remains token CE.
Both stages use archive `582ee70`, SHA-256
`987facf12879fabc0367f0e7f30f156b71fb937a8e325dcedbf1defda6d5216e`.

The third uninterrupted cold timing retains the original producer and worker
policy. It includes independent decode replay and full reference verification;
fresh output does not imply empty storage caches.

```bash
export SOURCE_ARCHIVE=$OLD/source-native-diagnostic.tgz
export MODE=full_one PREPARATION_PROFILE=0 DECODER_THREADS=
export REFERENCE_CORPUS=$PREPARATION/prepared-corpus
export FINAL_DATA=$OLD/evaluation-v1/validation.jsonl
export RESULTS=$ROOT/cold-replacement
sbatch --parsable --export=ALL --time=48 --mem=16G --partition=cpu_short \
  --nodelist=cpu-00008 --output="$ROOT/cold-replacement-%j.log" "$ROOT/prepare_parallel.sbatch"
```

Source is `e65f7eac8d7536171aa92ba7f7d019a8ea4b6998`, archive SHA-256
`83ad99a4feaf0cbe737f83cac9f08a18aeb21964f3801d4bf725406b37044325`.
The 48-minute effective wall limit plus 120-second termination grace reserves
3,000 CPU-wall seconds. Historical timeout/recovery costs remain charged.

Three counterbalanced shared-loading allocations use scoring source
`251dc2117590c2c4e492d8fbac2bb9b75deb1598`, archive SHA-256
`56e4dc82efc5405b1b22009a161d6983e515fde93764f40750906be36e999d28`.
Every score uses the same 32 development questions and immutable fused/eight-bit
seed-17 checkpoints. The serial control uses one verification worker; the shared
candidate uses four workers and overlapped checkpoint reading/restoration. Both
pin the full model SHA and retain fresh-process native restoration.

```bash
export SOURCE_ARCHIVE=$ROOT/source-loading.tgz MODE=study
export TASK_SPEC=$ROOT/loading_repeat0.json RESULTS=$ROOT/loading-repeat0 WORK_SECONDS=640
first=$(sbatch --parsable --export=ALL --time=12 --partition=interactive_singlenode,polar \
  --nodelist=batch-block4-0045 --output="$ROOT/loading0-%j.log" "$ROOT/deployment.sbatch")
export TASK_SPEC=$ROOT/loading_repeat1.json RESULTS=$ROOT/loading-repeat1 WORK_SECONDS=520
second=$(sbatch --parsable --export=ALL --time=10 --dependency="afterok:$first" \
  --partition=interactive_singlenode,polar --nodelist=batch-block4-0045 \
  --output="$ROOT/loading1-%j.log" "$ROOT/deployment.sbatch")
export TASK_SPEC=$ROOT/loading_repeat2.json RESULTS=$ROOT/loading-repeat2
sbatch --parsable --export=ALL --time=10 --dependency="afterok:$second" \
  --partition=interactive_singlenode,polar --nodelist=batch-block4-0045 \
  --output="$ROOT/loading2-%j.log" "$ROOT/deployment.sbatch"
```

The first allocation includes 15 pinned regression tests and a real-CUDA wrong-SHA
failure injection; nonzero return and no score/success receipt are required.
Explicit failure allowance applies only to that negative test. Do not discard
its allocation cost. The repetition order and statistics were locked before runs.

Collect bounded text evidence with the existing collector. It excludes weights,
optimizer arrays and prepared tensor payloads. Terminal accounting uses parent
job elapsed seconds once, not the sum of batch/extern/step duplicates.

```bash
python "$ROOT/collect_evidence.py" "$RESULTS" "$ROOT/result_raw.json.gz"
sacct -X -j 38042204,38042206,38042441,38042648,38042650,38042848 \
  --format=JobIDRaw,State,ElapsedRaw,TimelimitRaw,NodeList,ExitCode -P
PYTHONDONTWRITEBYTECODE=1 python docs/posttrain/evidence/quantized_optimizers_round5/analyze_screen.py \
  docs/posttrain/evidence/quantized_optimizers_round5/lr875_fused_raw.json.gz \
  fused_lr875 adamw --lr-ratio=.875 \
  --output=docs/posttrain/evidence/quantized_optimizers_round5/fused_lr875_summary.json
PYTHONDONTWRITEBYTECODE=1 python docs/posttrain/evidence/quantized_optimizers_round5/analyze_loading.py
```

No new final comparison is admitted unless both optimizer development stages
pass. Original matched-update, actual 60-second training, 390-second prepared
product and target-delivery results remain distinct. New shared-loading timings
are not substituted into historical optimizer end-to-end results.

The fused weighting screen passed, admitting its identical eight-bit stage. That
stage failed the substantive guard in seeds29/43, so no final comparison followed.
The recorded dependent diagnostic scores four existing seed-29 checkpoints on
development examples only; it adds no optimizer update.

```bash
export MODE=study WORK_SECONDS=960 SOURCE_ARCHIVE=$ROOT/source-baseline.tgz
export TASK_SPEC=$ROOT/eightbit_weight95_probe.json RESULTS=$ROOT/weight95-eightbit
sbatch --parsable --export=ALL --time=18 --nodelist=batch-block4-1051 \
  --partition=interactive_singlenode,polar --output="$ROOT/weight95-eightbit-%j.log" \
  "$ROOT/deployment.sbatch"
export WORK_SECONDS=520 SOURCE_ARCHIVE=$ROOT/source-loading.tgz
export TASK_SPEC=$ROOT/substantive_diagnosis_spec.json RESULTS=$ROOT/substantive-diagnosis
sbatch --parsable --export=ALL --time=10 --nodelist=batch-block4-1051 \
  --partition=interactive_singlenode,polar --dependency=afterok:38043623 \
  --output="$ROOT/substantive-diagnosis-%j.log" "$ROOT/deployment.sbatch"

PYTHONDONTWRITEBYTECODE=1 python docs/posttrain/evidence/quantized_optimizers_round5/analyze_screen.py \
  docs/posttrain/evidence/quantized_optimizers_round5/weight95_fused_raw.json.gz \
  fused_weight95 adamw --lr-ratio=1 \
  --output=docs/posttrain/evidence/quantized_optimizers_round5/fused_weight95_summary.json
PYTHONDONTWRITEBYTECODE=1 python docs/posttrain/evidence/quantized_optimizers_round5/analyze_screen.py \
  docs/posttrain/evidence/quantized_optimizers_round5/weight95_eightbit_raw.json.gz \
  eightbit_weight95 adamw_bnb_8bit --lr-ratio=1 \
  --output=docs/posttrain/evidence/quantized_optimizers_round5/eightbit_weight95_summary.json
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  docs/posttrain/evidence/quantized_optimizers_round5/analyze_cold.py --allocation-seconds=2121
PYTHONDONTWRITEBYTECODE=1 python docs/posttrain/evidence/quantized_optimizers_round5/analyze_substantive.py
```

The four-way reload verifies unchanged native task predictions and reports each
substantive answer's tokens and loss. It preserves the distinction between native
batch-four validation and batch-one descriptive scoring; the former remains the
acceptance metric. All target/mask definitions remain unchanged.

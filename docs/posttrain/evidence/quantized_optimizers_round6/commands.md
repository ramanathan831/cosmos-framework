# Round-six commands

These commands reproduce recorded work; they do not authorize new allocations.
Use fresh owned output paths. Local commits only; do not push/publish/merge or
modify the original QLoRA checkout. The QLoRA-named asset directory is read-only.

```bash
export R=/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam
export ROOT=$R/cosmos-optimizer-r6-20261109T1843
export OLD=$R/cosmos-optimizer-outcomes-20261104T1850
export SOURCE=$OLD ASSETS=$R/cosmos-qlora-wts-20261028T234324Z
export PRIOR=$R/cosmos-quantized-optimizers-20261103
export PREPARATION=$OLD/prepare-evaluation-v1
export CONTAINER_IMAGE=$R/cosmos-automl-focused-20261012/image-conversion/cosmos-framework-focused.sqsh
export TASK_DATA_ROOT=$OLD/data
export SOURCE_ARCHIVE=$R/cosmos-optimizer-r5-20261105/source-baseline.tgz
export TASK_SPEC=$ROOT/next_weight975_fused.json RESULTS=$ROOT/weight975-fused
export MODE=study WORK_SECONDS=960
sbatch --parsable --export=ALL --time=18 --nodelist=batch-block4-1051 \
  --partition=interactive_singlenode,polar --output="$ROOT/weight975-fused-%j.log" \
  "$ROOT/deployment.sbatch"
```

The committed spec retains seeds 17/29/43, LR1e-5, 15 actual updates, 40-update
schedule, effective batch four, FP32 trainables/BF16 compute, full recomputation,
original masks/rows/validation and original controls. Its only training change is
`rl.ce_exponent=0.975`. Archive582ee70 has SHA-256
`987facf12879fabc0367f0e7f30f156b71fb937a8e325dcedbf1defda6d5216e`.
The fused stage failed. **Do not submit the conditional eight-bit/final specs.**

```bash
python "$ROOT/collect_evidence.py" "$RESULTS" "$ROOT/weight975_fused_raw.json.gz"
PYTHONDONTWRITEBYTECODE=1 python docs/posttrain/evidence/quantized_optimizers_round5/analyze_screen.py \
  docs/posttrain/evidence/quantized_optimizers_round6/weight975_fused_raw.json.gz \
  fused_weight975 adamw --lr-ratio=1 \
  --output=docs/posttrain/evidence/quantized_optimizers_round6/fused_weight975_summary.json
```

The development preparation slice is fixed by `lock_slice.py` and the resulting
`preparation_slice_plan.json`. Selection uses training identities and native token
lengths, without task scores. It contains eight videos/15 targets (seven paired
choice/substantive videos, one substantive-only video); full corpus contracts
remain unchanged. The selected oracle payloads are fully hashed and exactly
compared. This subset check is not a full-source binding or full-corpus timing.

```bash
export SOURCE_ARCHIVE=$ROOT/source-profile.tgz MODE=slice
export REFERENCE_CORPUS=$PREPARATION/prepared-corpus
export FINAL_DATA=$OLD/evaluation-v1/validation.jsonl
export PREPARATION_PROFILE=1 DECODER_THREADS=
export PREPARATION_SLICE=$ROOT/preparation_slice_plan.json
export RESULTS=$ROOT/preparation-profile
sbatch --parsable --export=ALL --time=7 --mem=16G --partition=cpu_short \
  --nodelist=cpu-00008 --output="$ROOT/preparation-profile-%j.log" "$ROOT/prepare_parallel.sbatch"
# First allocation timed out: retained and charged. Same source/selection retry:
export RESULTS=$ROOT/preparation-profile-retry
sbatch --parsable --export=ALL --time=9 --mem=16G --partition=cpu_short \
  --nodelist=cpu-00008 --output="$ROOT/preparation-profile-retry-%j.log" "$ROOT/prepare_parallel.sbatch"
```

Profile sourcecc0f50c8, archive SHA-256
`2d3c7c83fcfa321dac600bdd8bd7bec08246988834783a81903b6f9fd56ffd5f`.
The profile runs four spawned workers/two Torch threads each, native decoder
settings, independent uncached replay, native publication and selected oracle
verification. Summed worker cumulative times overlap; profiler results are not
uninstrumented speed measurements. Process-tree PSS/RSS sampling is explicit.

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m pytest tools/quantized_optimizers/test_preparation_slice.py \
  --confcutdir=tools/quantized_optimizers --override-ini=addopts= -q
sacct -X -j 38278834,38279161,38279658 \
  --format=JobIDRaw,State,ElapsedRaw,TimelimitRaw,NodeList,ExitCode -P
```

Only these optimizer-project job IDs enter this continuation's ledger. Other
concurrent user allocations and checkouts are outside this project and untouched.

# Round-seven reproduction

Use fresh owned result directories and the configured scheduler routing. These
commands describe recorded work; they do not publish the branch or authorize a
new recipe sweep. The original QLoRA checkout and PR334 remain untouched.
Read-only model/environment assets retain their existing QLoRA-named directory.

```bash
export R=/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam
export ROOT=$R/cosmos-optimizer-r7-20261109T1945
export OLD=$R/cosmos-optimizer-outcomes-20261104T1850
export SOURCE=$OLD ASSETS=$R/cosmos-qlora-wts-20261028T234324Z
export PRIOR=$R/cosmos-quantized-optimizers-20261103
export PREPARATION=$OLD/prepare-evaluation-v1
export CONTAINER_IMAGE=$R/cosmos-automl-focused-20261012/image-conversion/cosmos-framework-focused.sqsh
export REFERENCE_CORPUS=$PREPARATION/prepared-corpus
export FINAL_DATA=$OLD/evaluation-v1/validation.jsonl
export SOURCE_ARCHIVE=$ROOT/source-accounted-final.tgz
export VIDEO_CACHE_BYTES=67108864 PREPARATION_PROFILE=0 DECODER_THREADS=
export MODE=full_one PREPARATION_SLICE=
export RESULTS=$ROOT/new-full-reproduction
mkdir -p "$RESULTS"
sbatch --parsable --export=ALL --partition=cpu_short --nodelist=cpu-00008 \
  --mem=16G --time=55 --output="$RESULTS/slurm-%j.log" \
  tools/quantized_optimizers/prepare_parallel.sbatch
```

The corrected implementation archive is from `75f02cfd`; exact SHA and launch
arguments are in `full0_launch_input.json`. The default CLI cache limit is zero.
For the locked paired slice use source `ff3b2ed1`, `MODE=slice_pairs`,
`PREPARATION_SLICE=$R/cosmos-optimizer-r6-20261109T1843/preparation_slice_plan.json`,
and a fresh result directory. `reuse_timed_launch.json` records the exact source
and25-minute allocation. Do not add the correctness-only retry to its timing set.
The initial inner-timer run and failed peak-equality assertion remain separate.

For the real CPU failure tests, the existing launcher copy
`accounting_check.sbatch` sets `SFT_VIDEO_REUSE_SLICE` to the pinned plan and runs:

```bash
python -m pytest tools/quantized_optimizers/test_video_reuse.py \
  tools/quantized_optimizers/test_preparation_slice.py \
  --confcutdir=tools/quantized_optimizers --override-ini=addopts= -q
```

The source media stays read-only. The test corrupts only a worker-local pixel
snapshot, and requires the independent native replay to reject it. The whole
15-target candidate corpus is then compared exactly with the qualified oracle.

The three-checkpoint GPU diagnostic uses the byte-for-byte committed
`substantive_spec.json` with source `71493cba`, `MODE=study`, `WORK_SECONDS=1680`,
and the existing `deployment.sbatch`. `substantive_launch.json` records the
30-minute A10080GB/8CPU/128GiB allocation and all source identities. Its wrapper
independently hashes each complete `model.pt`, then native restoration verifies
the pinned digest and provenance again before scoring. The native batch-four
loss must reproduce the recorded scalar; no final examples are scored.

Collect evidence on the remote host, then copy only the compressed archive:

```bash
python "$ROOT/collect_evidence.py" "$RESULTS" "$ROOT/new-raw.json.gz"
```

Reproduce the completed analyses locally from committed evidence:

```bash
export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python docs/posttrain/evidence/quantized_optimizers_round7/analyze_substantive.py
.venv/bin/python docs/posttrain/evidence/quantized_optimizers_round7/analyze_training_order.py
.venv/bin/python docs/posttrain/evidence/quantized_optimizers_round7/analyze_full_preparation.py
.venv/bin/python docs/posttrain/evidence/quantized_optimizers_round7/analyze_preparation.py \
  docs/posttrain/evidence/quantized_optimizers_round7/reuse_raw.json.gz
.venv/bin/python docs/posttrain/evidence/quantized_optimizers_round7/analyze_preparation.py \
  docs/posttrain/evidence/quantized_optimizers_round7/reuse_inner_raw.json.gz --inner-only
```

The training-order analysis uses the native cursor and the committed hashed
training metadata, with the saved cursor and qualified corpus manifest as
oracles. Recording groups use `recording_split.components` and the unchanged
full-inventory SHA `13477150f1dcd6326f5a7f5bf06ac1f1b972683fd9bedafd3bf03311494cd261`.
Directory names are not substituted for recording groups. No recipe selection,
threshold or evaluation denominator is changed by these descriptive analyses.

Full jobs `38284652`, `38286696` and `38288002` completed in 1826, 1965 and
2154 allocation-seconds. Their `full{0,1,2}_launch.json` records and compressed
raw archives bind the identical producer and exact output manifests. The full
analysis reads all three archives and the terminal allocation ledger; it refuses
to replace a missing or failed replication with an estimate. `terminal_check.json`
and `slurm_accounting.txt` account for all nine owned round-seven allocations.

The measured archive remains `75f02cfd`. A later launcher change only clarifies
that container startup is shared when multiple trials occupy one allocation;
each full single-trial allocation is charged its own startup. No raw timing is
altered by that wording correction. Package versions and metadata-file hashes
from the common runtime are in `runtime_identity.json`, with the post-run
snapshot limitation stated there. Original GPU runtime versions remain in the
raw diagnostic evidence.

The next quality diagnostic commands in `development_diagnosis.json` are
**proposed, not submitted**. They retain full updates 10 and 15 through native
`save_every=10`, `keep_optim_last=0`, `keep_model_last=0`, and a fresh run. Required
checkpoint attribution/state comparisons are part of the proposed cost; these
instrumented diagnostic times cannot replace product timings.

# Reproduce the additional development campaign

These commands describe completed/admitted work, not an instruction to launch a
new sweep. The 7.5e-6 candidate failed its fused screen; do not launch its eight-bit
stage or final comparison. The next 8.75e-6 probe is unsubmitted and needs a new
complete dependent-work reservation. Preserve all old results and use fresh output
paths when reproducing. Site credentials are supplied outside these commands.

```bash
export R=/lustre/fsw/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/rarunachalam
export OLD=$R/cosmos-optimizer-outcomes-20261104T1850
export ROUND4=$R/cosmos-optimizer-lr-20261105T1830
export SOURCE=$OLD
export SOURCE_ARCHIVE=$OLD/source-native-diagnostic.tgz
export ASSETS=$R/cosmos-qlora-wts-20261028T234324Z
export PRIOR=$R/cosmos-quantized-optimizers-20261103
export PREPARATION=$OLD/prepare-evaluation-v1
export CONTAINER_IMAGE=$R/cosmos-automl-focused-20261012/image-conversion/cosmos-framework-focused.sqsh
export TASK_DATA_ROOT=$OLD/data
export TASK_SPEC=$ROUND4/fused_lr_probe.json
export MODE=study WORK_SECONDS=960
export RESULTS=$ROUND4/fused-development
sbatch --parsable --export=ALL --time=18 --nodelist=batch-block4-0052 \
  --partition=interactive_singlenode,polar \
  --output="$ROUND4/optimizer-r4-lr75-fused-%j.log" "$ROUND4/deployment.sbatch"
```

The exact probe bytes are unchanged from round three. Its original status is
historical; [plan.json](plan.json) records this continuation's explicit admission.
The three native `torchrun` invocations are in [fused_lr_probe.json](fused_lr_probe.json).
All use 15 updates, the 40-update horizon, FP32 trainables, BF16 compute, effective
batch four, full recomputation, unchanged clipping and the shared mixed objective.

Three full CPU preparation replications use the same immutable source archive:

```bash
export MODE=full_one PREPARATION_PROFILE=0 DECODER_THREADS=
export REFERENCE_CORPUS=$PREPARATION/prepared-corpus
export FINAL_DATA=$OLD/evaluation-v1/validation.jsonl
for pair in 1:cpu-00016 2:cpu-00019 3:cpu-00020; do
  trial=${pair%%:*}
  node=${pair#*:}
  export RESULTS=$ROUND4/cold-preparation-$trial
  mkdir -p "$RESULTS"
  sbatch --parsable --export=ALL --time=39 --mem=16G --partition=cpu_short \
    --nodelist="$node" --output="$ROUND4/optimizer-r4-cold-$trial-%j.log" \
    "$ROUND4/prepare_parallel.sbatch"
done
```

Each reserves eight CPUs, runs four workers with two PyTorch threads each, performs
two independent native decodes for each target, and verifies exact equality with
the existing qualified corpus. No CUDA is initialized. Fresh outputs do not mean
empty filesystem caches. CPU nodes and GPU nodes share storage. Preserve actual
elapsed costs, Slurm rounding/grace and [the budget guard](cpu_budget_guard.json).

The two shared loading canaries each reserve ten GPU minutes and run four fresh
processes on the same 32 development questions, with fused/quantized order reversed:

```bash
export MODE=study WORK_SECONDS=520
export SOURCE_ARCHIVE=$ROUND4/source-runtime-candidate.tgz
export TASK_SPEC=$ROUND4/runtime_canary.json RESULTS=$ROUND4/runtime-canary
# Exact recorded paths/node/argv are in runtime_launch.json.
sbatch --parsable --export=ALL --time=10 --partition=interactive_singlenode,polar \
  --nodelist=batch-block4-0051 --output="$ROUND4/runtime-%j.log" "$ROUND4/deployment.sbatch"

export SOURCE_ARCHIVE=$ROUND4/source-checkpoint-overlap.tgz
export TASK_SPEC=$ROUND4/checkpoint_overlap_canary.json RESULTS=$ROUND4/checkpoint-overlap
sbatch --parsable --export=ALL --time=10 --partition=interactive_singlenode,polar \
  --nodelist=batch-block4-0102 --output="$ROUND4/overlap-%j.log" "$ROUND4/deployment.sbatch"
```

Runtime archive identities are in [runtime_source.json](runtime_source.json) and
[checkpoint_overlap_source.json](checkpoint_overlap_source.json); the immutable
commits can be rearchived with `git archive --format=tar.gz COMMIT`. Run metadata
and executed specs bind the actual input/output paths; prefer those receipts over
these explanatory examples. Native checkpoint provenance and full model checksums
are required before scores are published. Parallel source verification never uses
a stat-only content cache. Original source verification can be selected with
`--verification-workers 1`; checkpoint loading can be compared with
`--checkpoint-read-mode serial` and `overlap`.

Collect only bounded textual evidence using the existing collector; it excludes
model weights, optimizer states and prepared tensor payloads:

```bash
python "$ROUND4/collect_evidence.py" "$RESULTS" "$ROUND4/result-raw.json.gz"
sacct -X -j "$(awk '{print $1}' "$ROUND4/jobs.txt" | paste -sd,)" \
  --format=JobID,State,ElapsedRaw,TimelimitRaw,NodeList,ExitCode -P
```

Keep the original 60-second training, 390-second prepared-product and matched-update
endpoints separate. No new final examples are scored after the failed screen;
new loading timings must not be substituted into the old final end-to-end results.

The first full CPU allocation timed out; its within-budget time-limit extension
was denied by Slurm. [cpu_extension.json](cpu_extension.json) records the denial.
The separately admitted [verification batch](verify-published.sbatch) requests
three minutes plus a reserved two-minute termination grace and verifies every
payload/source byte with `verify_published_corpus.py`, without tensor imports.
It completed in 29 seconds. [preparation_recovery_plan.json](preparation_recovery_plan.json)
was locked before submission. The failed producer and recovery are separate
artifacts; do not turn the recovery receipt into an uninterrupted preparation.

```bash
sbatch --parsable --nodelist=cpu-00019 --job-name=optimizer-r4-verify \
  --output="$ROUND4/preparation-recovery/slurm-%j.log" "$ROUND4/verify-published.sbatch"
python docs/posttrain/evidence/quantized_optimizers_round4/analyze_runtime.py
```

The final ledger includes all eight allocations and their actual terminal costs.
All payload equality checks passed, but the prospective three-uninterrupted-run
cold timing interval is withheld. Neither a new recipe nor a replacement full
preparation is admitted by these reproduction examples.

Final review bound the parsed manifest bytes directly to the expected SHA-256.
Six integrity tests pass, including a changed parse read without a stat change.
The [corrected verification batch](verify-published-v2.sbatch) reserves two
minutes plus two-minute termination grace. Its fresh 25-second full recheck
passes, retains the original verification receipt, and brings the owned job
count to eight. This is a correctness rerun, not another cold timing replication.

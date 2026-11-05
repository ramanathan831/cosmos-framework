# Reproducing optimizer qualification

The implementation and support limits are in
[the design](../../docs/posttrain/quantized_optimizers.md). This directory contains
qualification tools, not an alternative training implementation. `native_probe.py`
calls the native SFT entrypoint and adds synchronized timing/shape hooks. Its
instrumented timings must not be presented as uninstrumented performance.

## Identities and data

The frozen mainline training source is
`0071b8fef340e6589eabe4836dd50bddc7b79044`, based on upstream
`e644a833bf909af9c23a02a5c1069967dc501111`. Model:
`Qwen/Qwen3-VL-8B-Instruct@0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
The native SFT environment uses Torch 2.11.0+cu128, Transformers 5.8.1 and
bitsandbytes 0.49.2. Per-run dependency inventories, device reports, resolved
configs and native metrics are retained in the compressed evidence archives.

`ASSETS` is the read-only QLoRA asset root containing the pinned model,
`environment/bin/python`, `data/canary` and `data/full`. `WTS_SOURCE_ROOT` is the separate original WTS media root, taken from the inventory paths.
Only campaign-owned output directories may be written. Reuse is authorized;
the evaluation pool has been inspected previously and is not historically untouched.

The original derived split had a recording shared across train and validation.
Its results remain in `native_development_overlap_raw.json.gz` and fail the data
integrity gate. The repaired campaign contains 173 train clips from 46 connected
recording groups and 56 development clips from 16 other groups. Grouping joins
both recording families and exact video SHA256 matches over the full inventory.

To regenerate from source, first remove overlap from the prior selection, then
derive the full-video multi-question rows. The derivation decodes existing
annotated option letters into answer text; it adds no generated explanations.

```sh
python -m tools.quantized_optimizers.recording_split \
  --inventory "$ASSETS/data/full/media_inventory.jsonl" \
  --train "$ASSETS/data/canary/train.jsonl" \
  --validation "$ASSETS/data/canary/validation.jsonl" \
  --output "$CAMPAIGN/repaired-prior"
python -m tools.quantized_optimizers.prepare_development \
  --raw "$WTS_SOURCE_ROOT/wts_train" --validation-raw "$WTS_SOURCE_ROOT/wts_eval" \
  --prior "$CAMPAIGN/repaired-prior" \
  --media-inventory "$ASSETS/data/full/media_inventory.jsonl" \
  --output "$CAMPAIGN/derived-clean"
```

For the exact recorded campaign, the preserved `recording-clean-v4` files have
SHA256 `dac521711ca5969fa0465fa461c959ab7687eb7f32433d34fba2a45149ffa487`
(train) and `52d6bd8cc4a53ab520e100e72359001c9ae1a976a521dab7dd7738a7567619a2`
(development). Check identities before training; source-path rebasing changes
JSONL hashes and must be recorded. The full inventory SHA256 is
`13477150f1dcd6326f5a7f5bf06ac1f1b972683fd9bedafd3bf03311494cd261`.

Final factorial task scoring uses 16 other recording groups, selected from the
approved full QLoRA evaluation pool by `select_final.py`. The selection is fixed
by metadata hashes and excludes all train/development components. Its manifest
is `final_recording_selection.json`; the old 16-row canary task score shares
development groups and is never promoted to this final comparison.

```sh
python -m tools.quantized_optimizers.select_final \
  --inventory "$ASSETS/data/full/media_inventory.jsonl" \
  --validation "$ASSETS/data/full/validation.jsonl" \
  --campaign-manifest "$CAMPAIGN/recording-clean-v4/manifest.json" \
  --output "$CAMPAIGN/final-selection"
```

## Mainline training and capacity jobs

Use the site's authorized Slurm account and container. The checked-in batch files
record this campaign's site defaults. Mainline paired runs use one A100-SXM4-80GB
with eight requested CPUs and 128 GiB host memory. Seeds 17/29 retain
`batch-block4-0052`; the entirely unstarted seed43 group may use another verified
A100 node. Every arm within a paired seed stays on one node; no trial is rerun
based on its result. See `execution_grouping_amendment.json`. The attended campaign also uses the
site-authorized `interactive_singlenode` partition (four-hour maximum, one running
job per user) alongside `polar`/`grizzly`. An explicit exclusion list restricts
unstarted groups to the previously established A100 node set; each allocation
still checks the actual GPU model. This changes queue eligibility, not trial
settings or the GPU-hour limit.
All three arms use effective batch four, eight processed video frames and the
same 40 updates, data order, precision, schedule and validation cadence.

```sh
export SOURCE="$CAMPAIGN"
export SOURCE_ARCHIVE="$CAMPAIGN/source-0e8e3a2.tgz"
git archive --format=tar.gz --output="$SOURCE_ARCHIVE" \
  0e8e3a2a0a10229f111fbebb389aece37697db68
sha256sum "$SOURCE_ARCHIVE"
export ASSETS=/shared/read-only-qlora-assets
export SFT_DATA_ROOT="$CAMPAIGN/recording-clean-v4"
export RESULTS="$CAMPAIGN/development-clean-v4"
export CONTAINER_IMAGE=/shared/pinned-runtime.sqsh
export CAPACITY_FILE="$CAMPAIGN/development-capacity-v3.json"
export DEVELOPMENT_DRIVER=1 DEPLOYMENT=1 TELEMETRY=1
export TRIAL_SEED=29 EXPECTED_GPU_NAME="NVIDIA A100-SXM4-80GB"
export TRAIN_KEYS='[layers.24.,layers.25.,layers.26.,layers.27.,layers.28.,layers.29.,layers.30.,layers.31.]'
export STEPS=40 FRAMES=8 VAL_EVERY=5 VAL_SAMPLES=16 VAL_BATCH=4
export VIDEO_PIXELS=1048576 VIDEO_MIN_PIXELS=262144 WORK_SECONDS=1350
sbatch --time=25 --partition=interactive_singlenode,polar --nodelist=batch-block4-0052 \
  --output="$RESULTS/slurm-%j.out" --error="$RESULTS/slurm-%j.err" \
  tools/quantized_optimizers/canary.sbatch
```

For the whole unstarted seed43 group, set `TRIAL_SEED=43`, `WORK_SECONDS=3150`
and reserve `--time=55`; remove the fixed node and use
`--partition=interactive_singlenode,polar,grizzly` with the campaign
`--exclude` list. Keep the 900-second per-arm admission guard. Seed29
needs only its remaining fused arm; completed eight-bit arms are preserved.

The initial two-update capacity calibration tries microbatches 4, 2, 1 for each
arm, keeping effective batch four. All three recorded arms fit microbatch four;
the same capacity result was reused after the data-only split repair. The driver
runs seeds 17, 29, 43 in rotated arm order and never overwrites a completed arm.
An incomplete arm requires inspection. `pending.json` is not completion: monitor
the Slurm job, inspect failures, and admit another serial allocation only within
the charged/reserved budget below.

The separate recomputation canary uses source
`9eac5c7ec60c771d43ba270096563f9dce8f4645` and `CAPACITY_FOLLOWUP=1`,
`WORK_SECONDS=1680`, `--time=30`, and trainable layers 16 through 31. Clear
`DEVELOPMENT_DRIVER`, retain the same substantive media/data settings, and use
a fresh output directory. Both fused and nonpaged optimizers try activation
checkpointing off and on, descending microbatch 4, 2, 1 at effective batch four.
Two actual updates test capacity and profile the mechanism; they cannot establish
task quality or a reproducible speed gain. Preserve OOMs and stop on other errors.
Set `TRAIN_KEYS` explicitly to blocks16–31 and unset `DEVELOPMENT_DRIVER` and
`TRIAL_SEED`. If the same allocation still admits work, the driver tests full-model
nonpaged/paged capacity at microbatch/effective batch one, with recomputation.
It does not submit extra allocations for this pressure pair.

The original full-model paged attempt completed two updates but OOMed during
validation while completed gradients remained live. A separate mainline fix
releases gradients at the completed boundary before native SFT validation. The first
scoped rerun (`db4ba624eef8c2809a59533633ec6b73197cc43f`) completed paged
training and validation, but failed the unchanged 64-supervised-token minimum
on its second training example. The final scoped pair uses
`dfce2fb5a0e1445976abd13ce2ec781d68a9b8c4`, `CAPACITY_PRESSURE_ONLY=1`,
`CAPACITY_FOLLOWUP=1`, `WORK_SECONDS=480`, `SEED=29` and a 10-minute wall limit
on the same A100 node. Seed 29 is the smallest existing campaign seed whose
first two training rows meet that minimum, selected by tokenizer counts without
evaluation outcomes. Pressure-arm admission reserves 240 seconds; the measured
previous maximum was 141 seconds. Other capacity searches retain 480 seconds.
Charged capacity time was 1,181 seconds; adding the 600-second reservation stays
within the unchanged 1,800-second ceiling. Only the full-model nonpaged/paged pair
is rerun; OOMs and the token-gate failure remain evidence.

Model loading in that final pair took substantially longer than the earlier
attempts. The nonpaged arm OOMed; the driver then deferred paging at its admission
guard. Preserve this incomplete pair. Source
`3641b40e070bc363f1b9635cf4eda9010f81227d` adds
`CAPACITY_PRESSURE_ARM=paged_adamw_8bit` in pressure-only mode to complete only the
deferred arm in a fresh output directory, on the same node, seed and controls.
Reserve 15 minutes with `WORK_SECONDS=780`. Three CPU workflow tests verify the
scope and unchanged token gate. Current mainline reserves eight minutes for every
capacity arm after observing the cold-load delay; a fourth CPU test covers that
admission guard. The frozen single-arm run had 13 minutes of work allowance and
completed in 152 allocated seconds. Before submission, ten unused mainline minutes
and five unassigned minutes moved to capacity; charged/reserved total28,076seconds
remained below28,800. The original control is never repeated or replaced.

## Disposable factorial integration

The primary feature does not contain the QLoRA branch. The separate integration
combines QLoRA `a7fd524242f4f06ac488a3d96d331b1f46e2b385` with the optimizer
feature at merge `a652405`. The integration-only checkpoint bridge and experiment
tools are available in that worktree and the zero-context `integration-only.patch`
artifact. Never apply the entire QLoRA branch to the primary optimizer branch.

Training uses the unchanged package/config/probe from
`ad88888d500903a36c89da37a84e4bcd8176f9d3` and wheel SHA256
`a1f8c9fcc4afca317bb8147a37345d42d269baf93da91293bdcf410f90b5b049`.
Workflow source `a02e75793e889aa8bb3fde26b0a9c578ee3f10d6` adds final scoring,
recovery, whole-seed scheduling and separate admission reserves: 900 seconds
for training, 420 seconds for scoring. Prior scoring took 122–161 seconds. Its archive SHA256 is
`ba8ee52b2971acf49e7c4bd67e7d1f2b5509ef52b0b94857ef4ed2f10b835a0e`.
The installed training wheel, configuration and probe retain the recorded training
identity. The archive also includes a dataclass field-order compatibility fix;
the immutable wheel takes precedence for runtime package imports.

From the disposable integration worktree, with the shared variables above:

```sh
export SOURCE_ARCHIVE="$CAMPAIGN/factorial-source-a02e757.tgz"
export RESULTS="$CAMPAIGN/factorial-clean-v4"
export SCORE_DATA="$CAMPAIGN/final-v5.jsonl"
export WHEEL="$CAMPAIGN/factorial-build-da43258/cosmos_framework-1.2.2-py3-none-any.whl"
export COSMOS_TRAINING_CODE_REVISION=ad88888d500903a36c89da37a84e4bcd8176f9d3
export COSMOS_TRAINING_WHEEL_SHA256=a1f8c9fcc4afca317bb8147a37345d42d269baf93da91293bdcf410f90b5b049
export FACTORIAL_CANARY=0 FACTORIAL_SEED_ONLY=29 WORK_SECONDS=2200
export EXPECTED_GPU_NAME="NVIDIA A100-SXM4-80GB"
sbatch --time=40 --partition=interactive_singlenode,grizzly --nodelist=batch-block1-2092 \
  --output="$RESULTS/slurm-%j.out" --error="$RESULTS/slurm-%j.err" \
  tools/quantized_optimizers/factorial.sbatch
```

The four-cell canary (`FACTORIAL_CANARY=1`) must pass first. Full comparisons use
three seeds and BF16/NF4 bases crossed with fused FP32/nonpaged eight-bit states.
Native full checkpoints at updates 8 and 16 enable fresh-process continuation;
the next eight updates, tensors, optimizer metadata, scheduler, cursor and RNG
are compared bitwise. Set `rl.keep_optim_last=0`: the default consolidated retention
prunes older optimizer components. Only correctness reference/recovery runs set this explicitly; quality/timing
trials retain the original pruning behavior for matched controls. Where
the original midpoint was already pruned, a separate correctness-only 16-update
reference replay must match the original final model/moments/data before its
retained midpoint is eligible. These repeats never replace quality/timing trials. Completed training is reused for final scoring. Old
development scores and their completion markers are preserved separately.
`FACTORIAL_SCORE_ONLY=1` runs frozen references, recovery and final scoring of
already-completed cells without starting new training. Run global preparation
and scoring serially. After both frozen references and exact recovery markers
are complete, different `FACTORIAL_SEED_ONLY` scopes may run on distinct nodes
within the two-GPU limit: shared artifacts are then read-only and outputs belong
to separate seeds. The scoped driver refuses incomplete shared preparation.
For the initial full comparison, leave `FACTORIAL_SEED_ONLY` unset until these
shared gates have been prepared; use one unscoped driver at a time.
Unset `FACTORIAL_SEED_ONLY` for this scoring phase and reserve 45 minutes
with `WORK_SECONDS=2550`. The whole seed43 factorial group also requires a
45-minute reservation, `FACTORIAL_SEED_ONLY=43`, and `FACTORIAL_SCORE_ONLY=0`.
Admit it only after unused time from completed allocations has been returned
to the ledger; a pending dependency alone does not reserve it within the budget.

## Budget, collection and analysis

The total ceiling is eight GPU-hours, including failures. Current reservations
are 220 mainline GPU-minutes, 180 factorial minutes, 45 capacity minutes,
1,017 seconds of earlier compatibility allocations and a conservative 15 local
GPU-minutes. Their sum is 476.95 minutes. After seed29 completed in 1,492 seconds, the final
seed43 allocation was admitted with 28,027 charged/reserved seconds against the
28,800-second total. Ten unused mainline minutes moved to the factorial phase;
the original trials and all thresholds are unchanged. Amendments in the plan files preserve
the original limits and reasons; no quality/performance threshold changed.
Final actual usage is 25,538 Slurm GPU-seconds plus a conservative 900-second local
allowance: 7h20m38s. All 49 recorded jobs are terminal, including canceled and failed
attempts. The paged completion passed; the complete mainline/factorial analyses
retain failed speed and task-NLL gates. The remaining 39m22s is insufficient for
the proposed 150-minute full deployment follow-up.
There are at most two concurrent phase allocations, on distinct nodes. Partially
completed paired seeds stay on their original node; only complete unstarted seed
groups move to another verified A100.

Before each submission, query `sacct -X` for owned job IDs. Charge terminal elapsed
GPU time, and reserve the **full** wall limits of running/pending allocations.
The proposed reservation must fit its phase and total ceilings. Returning unused
allocation time permits repacking the same prespecified trials, not new tuning.
Job IDs, failures, source archives and admission receipts are in `provenance.json`
and the raw archives. Stop at the cap; an unmet gate remains unmet.

```sh
python tools/quantized_optimizers/collect_evidence.py "$RESULTS" evidence.json.gz
python -m tools.quantized_optimizers.analyze_development \
  native_development_raw.json.gz /new/analysis
# Run the following in the disposable integration worktree:
python -m tools.quantized_optimizers.analyze_factorial \
  factorial_raw.json.gz media_inventory.jsonl /new/analysis/factorial_summary.json
```

The collector records each file's hash, retains bounded text/JSON telemetry, and
excludes binary checkpoints. Checkpoints remain at the exact recorded native
transaction paths. The analysis verifies matched controls and final recording
identities, retains missing cells, and plots quality/time and memory/throughput
with seed uncertainty. PyTorch peaks omit managed memory; process/device and host
measurements accompany them. Synthetic `diagnostic.py` and `paging_pressure.py`
provide kernel/allocation evidence, not task-quality evidence.

## Bounded deployment follow-up

The second campaign is separately preregistered in
[`quantized_optimizers_round2/plan.json`](../../docs/posttrain/evidence/quantized_optimizers_round2/plan.json).
It trains decoder blocks 16–31 with ordinary FP32 parameters/BF16 compute,
compares fused AdamW with activation checkpointing against nonpaged eight-bit
AdamW without it, and uses the same cosine schedule, microbatch four and 40
updates. Three restarts per paired seed measure timing variation; seeds remain
the statistical units. All checkpoints are retained, but final task scoring
uses only the prespecified restart zero.

`rl.prepared_corpus` optionally consumes immutable CPU inputs created by the
native packer. It preserves collation and label semantics, refuses missing or
changed entries, and binds the corpus digest into exact-resume identity.
Preparation rejects RNG-consuming preprocessing and verifies every row against
an independent raw replay. Ordinary recipes leave this setting empty.
The common preparation gain is not an optimizer gain: include preparation cost
when reporting a cold pipeline and identify prepared-corpus timing explicitly.

Use a **new** campaign directory. `PRIOR` below is the read-only first campaign;
`ASSETS` remains the read-only model/environment/data root. CPU preparation needs
no GPU and must use a CPU partition on this site. The recorded plan admits at
most 30 CPU-wall minutes across preparation attempts and 150 additional
GPU-minutes across local checks, canaries, failed attempts, training and scoring.
Refresh the ledger from Slurm before each admission; pending/running jobs count
at their full wall reservation. Do not launch full trials until the native
raw/prepared equality canary passes all six state comparisons.

```sh
export SOURCE=/shared/new-optimizer-deployment-campaign
export SOURCE_ARCHIVE="$SOURCE/source.tgz"
git archive --format=tar.gz --output="$SOURCE_ARCHIVE" f54620d
export ASSETS=/shared/read-only-qlora-assets
export PRIOR=/shared/read-only-first-optimizer-campaign
export CONTAINER_IMAGE=/shared/pinned-runtime.sqsh
export RESULTS="$SOURCE/prepare-v1"
mkdir -p "$RESULTS"
sbatch --output="$RESULTS/slurm-%j.log" tools/quantized_optimizers/prepare_deployment.sbatch

# After successful preparation (245 exact comparisons in the recorded dataset):
export PREPARATION="$RESULTS"
export RESULTS="$SOURCE/canary" MODE=canary WORK_SECONDS=650 TRIAL_SEED=17
mkdir -p "$RESULTS"
sbatch --time=12 --output="$RESULTS/slurm-%j.log" tools/quantized_optimizers/deployment.sbatch

# Only after canary complete.json and parity.json pass; repeat for seeds29,43,
# at most two GPUs on distinct nodes. All six runs for one seed stay in this job.
# The split stages avoid reserving a GPU for checkpoint-value hashing.
export RESULTS="$SOURCE/seed17" MODE=train_only WORK_SECONDS=1880 TRIAL_SEED=17
mkdir -p "$RESULTS"
TRAIN_JOB=$(sbatch --parsable --time=34 --output="$RESULTS/slurm-%j.log" \
  tools/quantized_optimizers/deployment.sbatch)
export TARGET="$RESULTS" RESULTS="$SOURCE/seed17-checkpoint-verification"
mkdir -p "$RESULTS"
VERIFY_JOB=$(sbatch --parsable --dependency="afterok:$TRAIN_JOB" \
  --output="$RESULTS/slurm-%j.log" tools/quantized_optimizers/verify_deployment_cpu.sbatch)
export RESULTS="$TARGET" MODE=score_only WORK_SECONDS=340
sbatch --time=7 --dependency="afterok:$VERIFY_JOB" \
  --output="$RESULTS/slurm-%j.log" tools/quantized_optimizers/deployment.sbatch
```

The recorded first two seeds used source `d35aa73` and combined training,
checkpoint verification and scoring in requested 43m20s allocations (Slurm rounded
these to 44 minutes). All twelve training
runs and all 48 repeat-checkpoint comparisons passed, but scoring startup exceeded
the work deadlines. Those failed attempts remain charged. Completed score commits
were checked and retained; only the missing score was run separately. The third
seed uses `f54620d` and the split stages above. A committed source-compatibility
receipt proves the native package tree, training-launch function and scoring
mathematics are identical. Sequential file reads before scoring's native loaders
change loading cost only. Every split-stage allocation must still fit the finite
campaign ledger; dependencies alone do not reserve budget or establish a pass.

Restrict each GPU submission to the site's verified A100-SXM4-80GB nodes using
its recorded node policy. The job resolves the allocated physical GPU UUID for
external NVML monitoring, rejects another device model, and launches native SFT
without the synchronizing probe used in the first development campaign. Final
dense task scoring restores `model` only through the native checkpoint manager;
it imports no QLoRA backend and restores no optimizer.

The direct training command used by each eight-bit run is:

```sh
export SFT_MODEL_PATH="$ASSETS/models/Qwen3-VL-8B-Instruct/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
export SFT_DATA_ROOT="$PRIOR/recording-clean-v4"
export SFT_PREPARED_CORPUS="$PREPARATION/prepared-corpus"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=42
export IMAGINAIRE_OUTPUT_ROOT=/shared/new-native-output

torchrun --standalone --nproc-per-node=1 -m cosmos_framework.posttrain.entrypoints.sft \
  --config cosmos_framework/posttrain/configs/sft_optimizer_deployment.py -- \
  rl.seed=17 rl.optimizer.name=adamw_bnb_8bit rl.optimizer.impl= \
  rl.optimizer.bnb_min_8bit_size=16384 model_hf.grad_ckpt=false
```

For the matched fused arm use `rl.optimizer.name=adamw`,
`rl.optimizer.impl=fused`, `rl.optimizer.bnb_min_8bit_size=4096`, and
`model_hf.grad_ckpt=true`. Preserve all other settings and data identities.
Do not interpret CPU corpus tests or an allocated/queued job as qualification.

Use the effective `TimeLimit` returned by Slurm for admission, not requested seconds:
Slurm rounded the campaign's fractional-minute requests upward. The retained audit
records an earlier seven-second excess in planned reservations; actual GPU spending
remained below the fixed cap. Subsequent examples use whole-minute requests.

The additional dense continuation launcher now sets `rl.keep_optim_last=0` before
its reference run, which saves at both 40 and 44. The resumed four-update phase uses
`rl.save_every=0` to avoid an unnecessary export. Its interrupted v10 attempt omitted
retention and was canceled before any training optimizer update. This extra large-workload
continuation did not fit the round-two remaining six-minute budget. The round-three
fused and nonpaged reference/resume pairs now pass every update41–44 comparison;
their raw evidence is separate from the historical timing runs. Existing small
native CUDA and full adapter integration recovery retain their original scope.

## Third campaign: actual budget checkpoints

The runtime archive for the locked final comparison is local commit `cb9d4f3`.
The protocol and expanded command lists are in
`docs/posttrain/evidence/quantized_optimizers_round3/final_protocol.json` and
`final_specs/seed{17,29,43}.json`. They preserve the original numerical gates,
use the shared development-selected `sft_optimizer_task.py` recipe and score
each actual committed checkpoint in a fresh process. Final scoring is part of
each GPU reservation. No analysis substitutes an update15/40 checkpoint for a
different elapsed budget.

Generate repaired and mixed training/development inputs using the authorized
annotations and the existing repaired recording split:

```sh
python -m tools.quantized_optimizers.prepare_task_diagnosis \
  --old-train "$OLD_TRAIN" --old-dev "$OLD_DEV" \
  --train-annotations "$TRAIN_ANNOTATIONS" --eval-annotations "$EVAL_ANNOTATIONS" \
  --full-eval "$FULL_EVAL" --inventory "$MEDIA_INVENTORY" \
  --fixed-final "$ROUND2_FINAL_JSONL" --output "$NEW_DATA_DIRECTORY"

python -m tools.quantized_optimizers.select_outcome_evaluation \
  --inventory "$MEDIA_INVENTORY" --full-eval "$FULL_EVAL" \
  --development-manifest "$NEW_DATA_DIRECTORY/manifest.json" \
  --previous-selection "$ROUND1_SELECTION" --previous-selection "$ROUND2_SELECTION" \
  --output "$NEW_EVALUATION_DIRECTORY"
```

Input/output identities are recorded in the third campaign's `data_manifest.json`
and `final_selection.json`. The original173-row train data are byte-identical;
mixed data add one original single-choice row per video. The fixed final manifest
contains128 questions across32 recording families, including both historical
16-question selections. It is explicitly a reused-pool comparison.

For exact replay on the recorded cluster, use the immutable existing input
artifacts below and a **new** output root. The supplied specs contain those input
paths; rebasing inputs elsewhere requires a separately identified protocol.
Archive the frozen runtime and expand the existing protocol before submission:

```sh
git archive --format=tar.gz -o /shared/source-final.tgz cb9d4f3
python -m tools.quantized_optimizers.plan_outcomes \
  docs/posttrain/evidence/quantized_optimizers_round3/final_protocol.json \
  /shared/new-final-specs
```

The following is the actual allocation pattern. `R` is the existing shared user
project directory; `ASSETS` remains read-only. The final GPU reservation is
3×4,500 seconds within the unchanged28,800-second campaign cap; at most two owned
GPUs run, on distinct nodes. Effective Slurm limits and failed allocations count.

```sh
ROOT="$R/cosmos-optimizer-outcomes-20261104T1850"
REPLAY="$R/new-optimizer-outcome-replay"
export SOURCE="$ROOT/recovery-launch" SOURCE_ARCHIVE=/shared/source-final.tgz
export ASSETS="$R/cosmos-qlora-wts-20261028T234324Z"
export PRIOR="$R/cosmos-quantized-optimizers-20261103"
export PREPARATION="$ROOT/prepare-evaluation-v1" TASK_DATA_ROOT="$ROOT/data"
export CONTAINER_IMAGE="$R/cosmos-automl-focused-20261012/image-conversion/cosmos-framework-focused.sqsh"
export MODE=study WORK_SECONDS=4380

export RESULTS="$REPLAY/final-seed17" TASK_SPEC=/shared/new-final-specs/seed17.json
mkdir -p "$RESULTS"
j17=$(sbatch --parsable --partition=interactive_singlenode \
  --nodelist=batch-block4-1034 --time=75 --export=ALL \
  --output="$RESULTS/slurm-%j.log" tools/quantized_optimizers/deployment.sbatch)

export RESULTS="$REPLAY/final-seed29" TASK_SPEC=/shared/new-final-specs/seed29.json
mkdir -p "$RESULTS"
sbatch --partition=interactive_singlenode --nodelist=batch-block4-1017 \
  --time=75 --export=ALL --output="$RESULTS/slurm-%j.log" \
  tools/quantized_optimizers/deployment.sbatch

export RESULTS="$REPLAY/final-seed43" TASK_SPEC=/shared/new-final-specs/seed43.json
mkdir -p "$RESULTS"
sbatch --partition=interactive_singlenode,polar --nodelist=batch-block4-0102 \
  --dependency="afterany:$j17" --time=75 --export=ALL \
  --output="$RESULTS/slurm-%j.log" tools/quantized_optimizers/deployment.sbatch
```

Seed43 was moved as a wholly unstarted allocation after its original node filled
and a second requested node became unavailable. It ran on `batch-block4-0102`;
no trial or score was discarded. The source archive, exact A100-SXM4-80GB device
check, eight-CPU allocation, dependency and frozen protocol were retained. The
[scheduling receipt](../../docs/posttrain/evidence/quantized_optimizers_round3/scheduling_notes.json)
records those changes. Queue delay is excluded from the declared product clock.

`run_task_study` records model-file hashes, complete setup clocks, source/dependency
identities, native task curves, actual checkpoint provenance and external device/
process/host memory for both training and final scoring. A target run stops only
after an observed post-update validation; a budget run records its actual completed
update and validation boundary. Late or absent states remain censored. The same
delivery-reserve calibration rule applies to both optimizers.

`prepare_parallel.sbatch` can run the qualified CPU canary or a full preparation
with `MODE=full_one`, `REFERENCE_CORPUS` pointing at the independently verified
corpus, and `FINAL_DATA` pointing at the fixed128-row JSONL. Workers use the native
packer and replay each unique target independently before publication. The full
run requests eight CPUs,16 GiB and40 minutes. The CPU cap is7,200 wall seconds;
the observed canary costs only permit one full measurement in this campaign,
not three full preparation timing repetitions.

After all dependent jobs terminate, collect textual evidence without model tensors:

```sh
python -m tools.quantized_optimizers.collect_evidence "$REPLAY" /shared/outcomes-raw.json.gz
python -m tools.quantized_optimizers.analyze_outcomes /shared/outcomes-raw.json.gz \
  docs/posttrain/evidence/quantized_optimizers_round3/final_protocol.json \
  docs/posttrain/evidence/quantized_optimizers_round3/final_selection.json \
  /shared/outcomes-analysis --cold-preparation-seconds "$QUALIFIED_FULL_CPU_ELAPSED"
```

The cold component is the completed qualified preparation's allocation time,
including setup. Its single measurement limits cold-pipeline replication claims;
it is not a claim about an empty OS page cache. The analysis preserves matched
updates, equal training time and equal product budget separately, uses three
paired quality seeds and bootstraps recording families, and retains both original
16-question summaries. This optimizer work makes no inference-latency claim.

### Corrected product-budget endpoint

The original endpoint remains censored for seed43: validation15 was pending in
its committed trainer journal. `pending_validation_inspection.json` binds that
journal, actual missing terminal metric and checkpoint provenance. Reproduce the
original analysis with `--pending-validation-inspections` pointing to that file;
do not treat the saved state as an eligible completed product.

`product_repair_protocol.json` repeats only the product-budget pair for all three
seeds. It preserves the original manifest, budget, reserve, recipe and numerical
gates. Source `f56a658` finishes required scheduled validation before an advance
budget notice. A real external signal retains immediate native preemption.
The native `cosmos_framework/` tree is identical to `cb9d4f3`.

Use the **current checked-in outer** `deployment.sbatch`, which exports and passes
`ALLOCATION_START_NS` into the container. `SOURCE` may be an existing directory
because the launcher extracts the immutable archive into a fresh temporary tree;
it does not authorize using an older outer script from that directory. Three
wrong-launcher preflight failures ran no training and remain charged (212 seconds).

With the same input/runtime variables above and a new result root:

```sh
git archive --format=tar.gz -o /shared/source-product-repair.tgz f56a658
export SOURCE_ARCHIVE=/shared/source-product-repair.tgz MODE=study WORK_SECONDS=960
export RESULTS="$REPLAY/product-repair-v2-seed17"
export TASK_SPEC=/shared/product-repair-specs/seed17.json
# Copy the committed product_repair_specs/seed*.json files byte-for-byte first.
mkdir -p "$RESULTS"
p17=$(sbatch --parsable --partition=interactive_singlenode,polar \
  --nodelist=batch-block4-0102 --time=18 --export=ALL \
  --output="$RESULTS/slurm-%j.log" tools/quantized_optimizers/deployment.sbatch)

export RESULTS="$REPLAY/product-repair-v2-seed29"
export TASK_SPEC=/shared/product-repair-specs/seed29.json
mkdir -p "$RESULTS"
sbatch --partition=interactive_singlenode,polar --nodelist=batch-block4-1017 \
  --dependency="afterok:$p17" --time=18 --export=ALL \
  --output="$RESULTS/slurm-%j.log" tools/quantized_optimizers/deployment.sbatch

export RESULTS="$REPLAY/product-repair-v2-seed43"
export TASK_SPEC=/shared/product-repair-specs/seed43.json
mkdir -p "$RESULTS"
sbatch --partition=interactive_singlenode,polar --nodelist=batch-block4-0102 \
  --dependency="afterok:$p17" --time=18 --export=ALL \
  --output="$RESULTS/slurm-%j.log" tools/quantized_optimizers/deployment.sbatch
```

Reserve all three effective 18-minute limits before admission. Each allocation
includes both native runs and both dependent fixed-manifest scores. The first
seed is the launcher canary; the other two cannot start if it fails. A job
completing successfully does not imply every state meets the product deadline.

Keep original and corrected raw archives separate, with keys prefixed by their
actual `final-seed*` or `product-repair-v2-seed*` directories. The corrected
analysis checks unchanged case arguments, actual exposure, source archive,
scoring contracts and required validations, and reuses other original endpoints:

```sh
python -m tools.quantized_optimizers.analyze_outcomes outcomes_raw.json.gz \
  docs/posttrain/evidence/quantized_optimizers_round3/final_protocol.json \
  docs/posttrain/evidence/quantized_optimizers_round3/final_selection.json \
  /new/corrected-product-analysis --cold-preparation-seconds 1990 \
  --product-repair-raw product_repair_raw.json.gz \
  --product-repair-protocol \
  docs/posttrain/evidence/quantized_optimizers_round3/product_repair_protocol.json
```

The separate `native_arithmetic_probe.json` is development-only: one seed, three
identical full-recomputation native runs, then CUDA comparison of cumulative
parameter updates. Its all-FP32 bitsandbytes control sets the supported minimum
state size above every trainable tensor and asserts zero quantized elements;
it is not a new public optimizer name. Its 25-minute reservation follows both
corrected seed jobs, with no new final scoring or automatic precision changes.

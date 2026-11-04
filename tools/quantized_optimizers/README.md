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
on fixed node `batch-block4-0052`, eight requested CPUs and 128 GiB host memory.
All three arms use effective batch four, eight processed video frames and the
same 40 updates, data order, precision, schedule and validation cadence.

```sh
export SOURCE="$CAMPAIGN"
export SOURCE_ARCHIVE="$CAMPAIGN/source-0071b8f.tgz"
git archive --format=tar.gz --output="$SOURCE_ARCHIVE" \
  0071b8fef340e6589eabe4836dd50bddc7b79044
sha256sum "$SOURCE_ARCHIVE"
export ASSETS=/shared/read-only-qlora-assets
export SFT_DATA_ROOT="$CAMPAIGN/recording-clean-v4"
export RESULTS="$CAMPAIGN/development-clean-v4"
export CONTAINER_IMAGE=/shared/pinned-runtime.sqsh
export CAPACITY_FILE="$CAMPAIGN/development-capacity-v3.json"
export DEVELOPMENT_DRIVER=1 DEPLOYMENT=1 TELEMETRY=1
export TRAIN_KEYS='[layers.24.,layers.25.,layers.26.,layers.27.,layers.28.,layers.29.,layers.30.,layers.31.]'
export STEPS=40 FRAMES=8 VAL_EVERY=5 VAL_SAMPLES=16 VAL_BATCH=4
export VIDEO_PIXELS=1048576 VIDEO_MIN_PIXELS=262144 WORK_SECONDS=2550
sbatch --time=45 --nodelist=batch-block4-0052 \
  --output="$RESULTS/slurm-%j.out" --error="$RESULTS/slurm-%j.err" \
  tools/quantized_optimizers/canary.sbatch
```

The initial two-update capacity calibration tries microbatches 4, 2, 1 for each
arm, keeping effective batch four. All three recorded arms fit microbatch four;
the same capacity result was reused after the data-only split repair. The driver
runs seeds 17, 29, 43 in rotated arm order and never overwrites a completed arm.
An incomplete arm requires inspection. `pending.json` is not completion: monitor
the Slurm job, inspect failures, and admit another serial allocation only within
the charged/reserved budget below.

The separate recomputation canary uses source
`2e81f9d0cf5633f00a6874be7cb8c8cca4dcd812` and `CAPACITY_FOLLOWUP=1`,
`WORK_SECONDS=1680`, `--time=30`, and trainable layers 16 through 31. Clear
`DEVELOPMENT_DRIVER`, retain the same substantive media/data settings, and use
a fresh output directory. Both fused and nonpaged optimizers try activation
checkpointing off and on, descending microbatch 4, 2, 1 at effective batch four.
Two actual updates test capacity and profile the mechanism; they cannot establish
task quality or a reproducible speed gain. Preserve OOMs and stop on other errors.

## Disposable factorial integration

The primary feature does not contain the QLoRA branch. The separate integration
combines QLoRA `a7fd524242f4f06ac488a3d96d331b1f46e2b385` with the optimizer
feature at merge `a652405`. The integration-only checkpoint bridge and experiment
tools are available in that worktree and the zero-context `integration-only.patch`
artifact. Never apply the entire QLoRA branch to the primary optimizer branch.

Training uses the unchanged package/config/probe from
`ad88888d500903a36c89da37a84e4bcd8176f9d3` and wheel SHA256
`a1f8c9fcc4afca317bb8147a37345d42d269baf93da91293bdcf410f90b5b049`.
Workflow source `2762d6d30d4703e156faea7c967d3545ba989d3d` changes scoring and
recovery scheduling only; its archive SHA256 is
`7fb7087c049c48582e3df1ce7a87d2a62e360f14c9bc30c529cc69b9d4672d8d`.
Keeping the training build identity is justified by the byte-identical runtime
package, configuration and probe, not by relaxing resume identity validation.

From the disposable integration worktree, with the shared variables above:

```sh
export SOURCE_ARCHIVE="$CAMPAIGN/factorial-source-2762d6d.tgz"
export RESULTS="$CAMPAIGN/factorial-clean-v4"
export SCORE_DATA="$CAMPAIGN/final-v5.jsonl"
export WHEEL="$CAMPAIGN/factorial-build-da43258/cosmos_framework-1.2.2-py3-none-any.whl"
export COSMOS_TRAINING_CODE_REVISION=ad88888d500903a36c89da37a84e4bcd8176f9d3
export COSMOS_TRAINING_WHEEL_SHA256=a1f8c9fcc4afca317bb8147a37345d42d269baf93da91293bdcf410f90b5b049
export FACTORIAL_CANARY=0 WORK_SECONDS=2200
sbatch --time=40 --nodelist=batch-block1-2092 \
  --output="$RESULTS/slurm-%j.out" --error="$RESULTS/slurm-%j.err" \
  tools/quantized_optimizers/factorial.sbatch
```

The four-cell canary (`FACTORIAL_CANARY=1`) must pass first. Full comparisons use
three seeds and BF16/NF4 bases crossed with fused FP32/nonpaged eight-bit states.
Native full checkpoints at updates 8 and 16 enable fresh-process continuation;
the next eight updates, tensors, optimizer metadata, scheduler, cursor and RNG
are compared bitwise. Completed training is reused for final scoring. Old
development scores and their completion markers are preserved separately.

## Budget, collection and analysis

The total ceiling is eight GPU-hours, including failures. Current reservations
are 240 mainline GPU-minutes, 170 factorial minutes, 30 capacity minutes,
1,017 seconds of earlier compatibility allocations and a conservative 15 local
GPU-minutes. Their sum is 471.95 minutes. Amendments in the plan files preserve
the original limits and reasons; no quality/performance threshold changed.
There are at most two concurrent phase allocations, on distinct fixed nodes.

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

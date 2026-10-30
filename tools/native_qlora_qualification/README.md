# Native QLoRA qualification tools

These tools orchestrate the native Cosmos SFT entrypoint and inspect its artifacts. They do not contain another training loop, submit scheduler jobs, or declare release support. Use the sealed runtime and pinned model described in [the native QLoRA guide](../../docs/post_training/native_qlora.md). Run accelerator commands inside a one-GPU A100 80GB allocation. Keep the tested wheel overlay first on `PYTHONPATH`, the workflow source root second, and the working directory outside the source checkout.

## Data and selection

`split.py` checks the prepared WTS conversion hashes, joins related recording families and identical video content, and records earlier evaluation exposure. Supply every previous score file and a JSON list of additional probe IDs. It excludes all validation-linked and selected development groups from new training. Development groups may have been used by historical training; every new model must start from the pinned base.

```bash
python -m tools.native_qlora_qualification.split \
  --data-root "$PREPARED_WTS" --exposed-scores "$OLD_SCORES" \
  --exposed-ids "$PROBE_IDS" --exposed-validation-prefix 64 \
  --development-groups 12 --seed 20261030 --output "$NEW_SPLITS"
```

Keep generated rows and media private. Preserve the split manifests and their SHA256 digests before scoring. The original 256-row selection remains a regression set. A single untouched recording group is labeled `untouched_diagnostic`; it cannot establish independent-cluster uncertainty. Do not promote it to a final release set. `score.py --all-rows` requires the exact split manifest and external digest. Final and untouched-diagnostic scoring additionally require a sealed selection receipt binding the chosen native configuration and development evidence.

`quality.py` verifies paired row/input/mask/media identities and computes token-weighted NLL, restricted-choice accuracy, frozen-base comparisons, task slices and a deterministic recording-group bootstrap for one training seed. `quality_confirmation.analyze_confirmation` separately requires all three paired seeds (17, 42 and 123), both fixed frozen references and the sealed connected-recording labels. It reports a seed-then-group hierarchical bootstrap plus a crossed sensitivity that shares group draws across seeds. Both use 10,000 replicates, RNG seed 2026 and the unchanged 2% NLL, one-percentage-point accuracy and positive-adaptation bounds. Both must pass before the conditional statistical criteria pass. Slice intervals remain exploratory and no difficult examples are removed. Artifact/checkpoint identity validation is an independent prerequisite; this statistical function never declares release qualification. An exposed development or regression result is not untouched final confirmation.

## Native experiments

`protocol.py` defines the strict JSON `Experiment` schema and bounded allocation accounting. The two precision arms use the same native overrides, differing in their base recipe and run name. Training, adapter initialization and validation seeds are explicit. Parameter, arithmetic and optimizer precision remain separate.

`run.py kernel-gate` verifies the wheel overlay and runs all 14 real CUDA tests without skips. `run.py experiment` requires that gate for the same build, then measures native training, portable adapter publication, fresh DCP/portable probes, exact reload comparison and sealed-split scoring. Each output directory is exclusive; unsuccessful attempts retain commands, logs and exit status. A successful completion receipt is execution evidence, not a quality-gate pass.

```bash
python "$WORKFLOW_SOURCE/tools/native_qlora_qualification/run.py" kernel-gate \
  --build "$BUILD" --output "$NEW_GATE" --time-binary "$GNU_TIME"

python "$WORKFLOW_SOURCE/tools/native_qlora_qualification/run.py" experiment \
  --spec "$EXPERIMENT_JSON" --build "$BUILD" --kernel-gate "$NEW_GATE/completion.json" \
  --model "$PINNED_MODEL" --data-root "$TRAINING_DATA" \
  --score-data "$NEW_SPLITS/development.jsonl" \
  --split-manifest "$NEW_SPLITS/development.manifest.json" --split development \
  --output "$NEW_RUN" --time-binary "$GNU_TIME"
```

`TRAINING_DATA` contains the sealed fresh `train.jsonl` and a `validation.jsonl` copy of the development data. Export `PYTHONHASHSEED` equal to the experiment seed and `CUBLAS_WORKSPACE_CONFIG=:4096:8` before launching. The runner records input digests and the complete native dotlist. Pair precision arms sequentially within the same scheduler allocation, counterbalance their order across repeats, and record physical CPU placement. Account for failed allocations as well as successful ones. The 2026-10-30 continuation uses a predeclared maximum of 48 GPU allocation-hours, four concurrent GPUs, and 64 core-hours for separate CPU diagnostics.

`lifecycle.py` runs a separate four-update recovery qualification for each selected numerical configuration. It compares uninterrupted execution with both a completed-boundary interruption and process loss after a real accumulating microbatch. Only explicit checkpoints from that attempt may resume. The immediate next update and final update must match model, optimizer, RNG, cursor, controls and numerical metrics exactly. Fresh multimodal DCP and portable probes must also match exactly. Filter experiment output/horizon fields before passing repeated `--override` arguments: the recovery runner owns its output names and fixes horizon four, warmup one and save cadence one.

## Runtime and memory diagnostics

`worker_memory.py` compares the inherited worker environment with a fresh-process BLAS-thread intervention. Use sequential AB/BA pairs on matched CPU placement. It checks exact processed tensor and RNG identities, live-cache accounting, process-tree memory and thread/allocator provenance. Its end-only garbage collection and allocator trim are disposable diagnostics, not a production policy.

`workloads.py` prepares actual processor-measured video workloads, including 3,840–4,096-token examples with at least 1,024 supervised tokens and a larger 16-frame input. The long task is an explicitly synthetic textual ledger alongside real media. These fixtures establish numerical, recovery and resource behavior; they do not establish WTS task quality. No token padding or truncation is used to manufacture sequence lengths.

`monitor.py` samples owned device-process memory and host process-tree RSS/PSS. Sampled peaks are lower bounds; summed RSS may count shared pages twice. Native CUDA phase peaks end at their recorded event and reset afterward. Preprocessing processing/wait counters overlap training and each other and must not be added to elapsed wall time. Report loading, first-step/warmup, training, validation, checkpointing, publication, reload and scoring costs separately, including complete training-to-evaluation elapsed time.

Multi-GPU training, other model families, other quantizers and other adapter methods remain outside this qualification.

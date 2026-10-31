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

`run.py kernel-gate` verifies the wheel overlay and runs all 14 real CUDA tests plus the existing higher-precision native CPU regression without skips. `run.py experiment` requires that gate for the same build, then measures native training, portable adapter publication, fresh DCP/portable probes, exact reload comparison and sealed-split scoring. Each output directory is exclusive; unsuccessful attempts retain commands, logs and exit status. A successful completion receipt is execution evidence, not a quality-gate pass. Fresh `train-eval` and `frozen` actions reject an external checkpoint; `evaluate` requires one explicitly.

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

## Selection and comparison

`selection.py` validates all eight predeclared development candidates against their exact specifications, final checkpoints, scorer identities and complete development rows. It retains failed candidates and refuses selection when a precision baseline is incomplete. Development selection does not authorize untouched scoring.

`execution_selection.py` separately validates the short-video full/no-checkpointing calibration: each precision starts at microbatch one and advances to two or four only after the preceding cell satisfies the observed 24 GiB device-process budget. It recomputes native timing and memory evidence, verifies accepted-boundary recovery and rejects incomplete calibration. The sealed rule chooses the lowest mean update time after warmup, then lower observed peak, smaller microbatch and full checkpointing for exact ties. It combines those execution choices with the development-selected learning rate, rank and alpha to emit six 300-update confirmation specifications at seeds 17, 42 and 123. Long/media fixtures remain separate scope tests; this short screening result is not a full-run speed or quality result.

```bash
python -m tools.native_qlora_qualification.execution_selection \
  --inputs "$PINNED_EXECUTION_INPUTS" --inputs-sha256 "$INPUTS_SHA256" \
  --evidence-root "$QUALIFICATION_EVIDENCE" --output "$NEW_SELECTION"
```

The input manifest binds all consumed files and declares any recorded-root aliases needed for a local evidence mirror. Incomplete calibration produces only a diagnostic decision; complete decisions produce the locked specifications and execution receipt. Keep executable receipts private when they contain filesystem paths.

`performance.py` keeps controlled comparisons strict. A separately tuned `best_fit` comparison requires an externally pinned v2 pair plan binding the confirmation/development plans and both selection decisions. Only selected learning rate, rank, alpha, microbatch and checkpointing mode may differ between precision arms. All three declared seeds, paired placement and valid memory evidence remain required. An over-budget BF16 control stays visible in controlled comparisons; it cannot qualify as a 24 GiB best-fit candidate. The observed budget does not establish support on a physical 24 GiB accelerator.

## Runtime and memory diagnostics

`worker_memory.py` compares the inherited worker environment with a fresh-process BLAS-thread intervention. Use sequential AB/BA pairs on matched CPU placement. It checks exact processed tensor and RNG identities, live-cache accounting, process-tree memory and thread/allocator provenance. Its end-only garbage collection and allocator trim are disposable diagnostics, not a production policy.

The opt-in periodic diagnostic uses `--trim-every-rows 0` for a matched sham and `--trim-every-rows 128` for worker-local `malloc_trim(0)`. Both modes keep inherited thread settings, verify the same live cache and RNG at every drained 128-row boundary, and perform no explicit GC or cache eviction. Both receive the same separately timed end-only trim. Omitting the option preserves the original thread comparison and end-only GC/trim behavior. Periodic comparisons require one complete sham and one complete treatment with identical placement, environment, processed tensors, cursor and cache identities, plus clean child exits. Post-result shutdown stamps distinguish pipeline close, monitor join and leaving the child main.

Run from a tools-only staging directory against the verified wheel overlay and prepared processor/data bundle; the launcher must verify package bytes and import resolution before these commands. Use a disposable CPU allocation with eight physical cores for the recorded configuration. Keep the inherited thread environment identical within each pair and use a fresh output root for every repeat:

```bash
for interval in 0 128; do
  python -m tools.native_qlora_qualification.worker_memory run \
    --config cosmos_framework/posttrain/configs/sft_qwen3_vl_8b_qlora.py \
    --model-path "$SFT_MODEL_PATH" --data-root "$WTS_ACCEPTED_DATA" \
    --build-receipt "$BUILD_RECEIPT" --output "$CPU_EVIDENCE/trim-$interval" \
    --thread-policy inherited --trim-every-rows "$interval" \
    --rows 1000 --seed 42 --repeat 0 --expected-physical-cpus 8 \
    --timeout-seconds 1720
done
python -m tools.native_qlora_qualification.worker_memory compare \
  --run "$CPU_EVIDENCE/trim-0" --run "$CPU_EVIDENCE/trim-128" \
  --output "$CPU_EVIDENCE/comparison.json"
```

Counterbalance with `128 0`, `--repeat 1` and another fresh output root. The timeout covers startup, replay and shutdown; allocation wall time must cover both children and setup. The [completed CPU evidence](../../docs/post_training/evidence/native_qlora_worker_memory_20261030.json) has SHA256 `72fbd667c01c5b7bab357de3b6eff4061dd5f0019b7e58a24a9ca6374cc7837a`. Its two clean pairs reduced observed boundary worker PSS by about 1.08 and 1.11 GiB while preserving all processed outputs and state. The failed initial attempt remains descriptive-only evidence. Replay time increased by about 12 and 210 seconds respectively; order, cache warmth and system load remain confounded, so no speed improvement is claimed.

The promoted periodic replay, maintenance and shutdown functions preserve the qualified diagnostic's syntax-tree semantics; [source-provenance tests](tests/test_worker_memory_periodic.py) pin those functions and the legacy replay separately. Promotion adds explicit mode dispatch, rejects mixed protocols or incomplete exits, and requires at least one periodic boundary. The evidence is CPU preprocessing replay only: no model, production reclamation policy, live-training memory bound or larger-media benefit is established. Direct trim-call time excludes deferred page-fault and training costs.

`workloads.py` prepares actual processor-measured video workloads, including 3,840–4,096-token examples with at least 1,024 supervised tokens and a larger 16-frame input. The long task is an explicitly synthetic textual ledger alongside real media. These fixtures establish numerical, recovery and resource behavior; they do not establish WTS task quality. No token padding or truncation is used to manufacture sequence lengths.

`monitor.py` samples owned device-process memory and host process-tree RSS/PSS. Sampled peaks are lower bounds; summed RSS may count shared pages twice. Native CUDA phase peaks end at their recorded event and reset afterward. Preprocessing processing/wait counters overlap training and each other and must not be added to elapsed wall time. Report loading, first-step/warmup, training, validation, checkpointing, publication, reload and scoring costs separately, including complete training-to-evaluation elapsed time.

`monitor_host.py` records raw process identity, memory reads, errors and bounded retry attempts. Its version 3 receipt uses admitted ancestry, PID, birth time and a held process pidfd as identity; mutable `/proc` inode ownership remains diagnostic metadata. Known descendants remain tracked after reparenting. A zombie leader alone does not prove whole-process exit. Exit-compatible unreadability may wait at most 350 ms on the same pidfd, retaining the initial poll and reserving 250 ms for a complete retry within the original 750 ms budget. Permission errors, unexplained reads, PID reuse and stage races remain incomplete; a still-live group never becomes zero memory. Synchronous proc reads and scheduling are not hard real-time bounded. Legacy receipts retain their original interpretation and incomplete flags. The [actual 61-test host proof](../../docs/post_training/evidence/native_qlora_collector_wait_94289a2.json) does not change model arithmetic, repair historical measurements or qualify a new training configuration.

Multi-GPU training, other model families, other quantizers and other adapter methods remain outside this qualification.

# Round-eight commands

All results use fresh owned directories. No aggregate GPU/CPU ceilings apply.
Scheduler routing, realistic wall limits, termination grace and actual cost
accounting remain in force. Nothing here publishes the branch or touches PR334.

The authoritative replay commands, environment and immutable source are in
`replay_spec.json` and `replay_launch.json`. Run the committed `deployment.sbatch`
with `MODE=study`, that exact task spec and a fresh `RESULTS` directory. The
spec retains native updates10/15 (`save_every=10`, both retention limits zero),
records read-only attribution10..15, compares update15 state with the corresponding
stored run, then reloads/scorers both endpoints in fresh processes. No final
examples are loaded. Diagnostic timings do not enter product comparisons.

The preparation verification screen is reproduced with `verification.sbatch`
and the exact source/environment in `verification_launch.json`. It runs three
counterbalanced pairs of fresh verifier processes on existing full corpora,
then one fresh preparation of the pinned development slice. These are verification
phase measurements, not full cold replications. The original decoder and all
independent uncached target replays remain in the producer.

The optional implementation switch is:

```bash
python -m tools.quantized_optimizers.parallel_prepare \
  --config cosmos_framework/posttrain/configs/sft_optimizer_deployment.py \
  --workers 4 --reference-corpus "$REFERENCE_CORPUS" \
  --fixed-slice "$PREPARATION_SLICE" --video-cache-bytes 67108864 \
  --reference-verification bytes --output "$FRESH_OUTPUT"
```

The default remains `--reference-verification tensors`. Byte verification requires
exact serialized payload identities and fails closed on equivalent tensors whose
serialization differs; it never falls back or skips a hash. Native source and
processor binding remains mandatory. The complete preparation success receipt
still follows reference verification.

Affected local checks:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m pytest \
  tools/quantized_optimizers/test_verify_published_corpus.py \
  tools/quantized_optimizers/test_preparation_slice.py \
  tools/quantized_optimizers/test_late_observer.py \
  tools/quantized_optimizers/test_substantive_diagnostic.py \
  --confcutdir=tools/quantized_optimizers --override-ini=addopts= -q
```

Collect terminal raw evidence using the existing `collect_evidence.py`; do not
copy dense checkpoints locally. `analyze_replay.py` refuses analysis until the
full replay-parity receipt exists. Recipe selection remains conditional on that
evidence and the unchanged three-seed development guards.

The first shared recipe trial is locked in `interleave_protocol.json` and
`fused_interleave_spec.json`. `fused_interleave_launch.json` records the submitted
archive (`d0948599`), environment and exact `sbatch` argv. The native overrides
are `rl.data.interleave_id_prefix=mcq:` and `rl.data.interleave_window=20`;
`rl.ce_exponent=1.0` and LR 1e-5 retain their original values. Seed17 keeps the
update10 checkpoint and resumes it in a fresh process. The final verification
case compares actual next inputs, every resumed update and all final components.
Do not use the earlier incomplete `03051145` archive: its newly added cursor was
caught under the repository's generic `env/` ignore rule before submission.

After collecting a terminal study with the existing collector:

```bash
.venv/bin/python docs/posttrain/evidence/quantized_optimizers_round8/analyze_screen.py \
  docs/posttrain/evidence/quantized_optimizers_round8/fused_interleave_raw.json.gz \
  fused_interleave adamw --lr-ratio 1 \
  --output docs/posttrain/evidence/quantized_optimizers_round8/fused_interleave_summary.json
```

The analogous eight-bit spec is conditional and must not run before all fused
guards pass. Analysis uses the original matching controls, fixed evaluation
identities and exact exposure within each five-update window. Diagnostic state
hashing and midpoint saving are correctness work, excluded from product timing.

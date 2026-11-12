# Round-nine reproduction commands

Use fresh output directories and the exact source archives, dependencies and
environment recorded in each launch receipt. No aggregate resource caps apply;
configured routing, allocation walls, termination grace and actual accounting
remain mandatory. These commands neither publish the branch nor touch QLoRA.

`permutation_spec.json` uses the native task launcher for a same-order seed-43
reference, full checkpoint/curve parity against its retained original, and a
window-four permutation. The actual four-row update sets are unchanged. Observed
steps are 1/13/14/15; only step 1 has identical initial parameters/moments and a
full elementwise clipped-gradient comparison. Later observations describe
diverged trajectories. Extra diagnostics and saves exclude these runs from
product timing claims.

`full_seed17_spec.json`, `full_seed29_spec.json` and `full_seed43_spec.json` use:

```bash
torchrun --standalone --nproc-per-node=1 \
  -m tools.quantized_optimizers.task_development \
  --config cosmos_framework/posttrain/configs/sft_optimizer_full_exposure.py -- \
  rl.seed=17 rl.optimizer.name=adamw rl.optimizer.impl=fused \
  model_hf.grad_ckpt=true rl.optimizer.lr=1e-5 rl.ce_exponent=1.0 \
  rl.label_smoothing=0.0 rl.resume=off
```

The specs supply model/data/prepared-corpus paths, diagnostic configuration and
fresh native output roots. For the corresponding eight-bit arm they select
`rl.optimizer.name=adamw_bnb_8bit`, the recipe’s empty `impl` selector, and
`model_hf.grad_ckpt=false`. This is a memory-feasible deployment comparison,
not identical-execution optimizer isolation. The locked endpoint is update 174,
after exactly two complete 346-row passes; each pass ends with a real two-row
batch. The partial-epoch policy supports one rank and accumulation one. Both
seed-17 arms retain update 87 and resume it in fresh processes through update 174.

`source_hash_launch.json` and `source_hash.sbatch` run the bounded CPU-only
source-identity screen. Native opt-in flags are `rl.data.media_hash_workers=4`
and `rl.validation.data.media_hash_workers=4`; the task-development diagnostic
also accepts `TASK_SOURCE_HASH_WORKERS=4`. Fresh-process scoring accepts
`--source-hash-workers 4`. All default to serial full hashing. The independent
prepared-corpus verifier and its required checks remain in place.

Affected local checks (the existing local environment is read-only):

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest \
  cosmos_framework/posttrain/tests/test_sft_epoch_tail.py \
  cosmos_framework/posttrain/tests/test_sft_parallel_media_identity.py \
  tools/quantized_optimizers/test_training_order_observer.py \
  --confcutdir=cosmos_framework/posttrain/tests --override-ini=addopts= -p no:env -q
```

The CPU preflight additionally checks all 60 actual prepared targets, both
collation orders, individual labels/masks/counts and concatenated media tensors
under the pinned cluster environment. It does not establish accelerator parity.

After terminal status, collect bounded text evidence with the existing
`collect_evidence.py`, retain its per-file hashes, then run:

```bash
PYTHONPATH=. .venv/bin/python docs/posttrain/evidence/quantized_optimizers_round9/analyze_permutation.py
PYTHONPATH=. .venv/bin/python docs/posttrain/evidence/quantized_optimizers_round9/analyze_full_exposure.py
PYTHONPATH=. .venv/bin/python docs/posttrain/evidence/quantized_optimizers_round9/analyze_source_hash.py
```

Analysis requires the respective complete receipts; queued or partial runs are
not outcomes. The original short-horizon and final protocols remain unchanged.
The full-exposure study has its own prospective development guards and never
selects a best checkpoint retrospectively.

The effective GPU route is the configured `polar` pool, with no fixed node,
eight CPUs, 128 GiB host memory and one checked A100-SXM4-80GB. Both arms of a
seed share one allocation. `routing_update.json` retains the removal of the
unrequested stale node pin; no new account, pool or QoS was introduced.
The copied cluster batch script matches `tools/quantized_optimizers/deployment.sbatch`
exactly (SHA256 `7454235f87f610242250cb3226407d0fdba4adbc81849d9e8b18ae75c420eb36`).

For example, from a checkout on the configured Slurm host, this submits the
recorded seed-29 pair to a new output directory, retaining its exact immutable
source archive, specification, model and prepared data. Run seed groups serially
and monitor each through completion; do not overwrite existing results.

```bash
python - <<'PY'
import json, os, subprocess
from pathlib import Path
receipt = json.loads(Path('docs/posttrain/evidence/quantized_optimizers_round9/full_seed29_launch.json').read_text())
env = os.environ.copy()
env.update(receipt['environment'])
env['RESULTS'] += '-reproduction'
assert not Path(env['RESULTS']).exists(), 'fresh output directory required'
subprocess.run([
    'sbatch', '--parsable', '--export=ALL', '--time=45', '--partition=polar',
    '--job-name=optimizer-r9-full-seed29-reproduction',
    '--output=' + env['RESULTS'] + '-%j.log',
    'tools/quantized_optimizers/deployment.sbatch',
], env=env, check=True)
PY
```

The full study uses source `8858db8a8799888d8874922144ea51ecb409d82b`,
archive SHA256 `36e8a31ee699e303dfb393acbdea0794fa586cd3b86b2fd89f1d0d0c5a32d422`.
It deliberately retains serial source hashing in both arms. The separate shared
hashing screen uses `0cc7ffc2f1bf593ca753b6621ecf718682292d15`; its CPU result is
not retroactively included in these GPU timings. The corrected permutation
archive uses `29aaf9dd`; launch receipts bind its complete revision and archive hash.

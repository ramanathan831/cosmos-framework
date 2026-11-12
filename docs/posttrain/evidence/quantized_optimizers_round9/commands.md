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

# AutoML

Cosmos Framework owns the workload integration; `nvidia-automl-core` owns the
repository-neutral search and execution controller. Users do not need a TAO
checkout, SDK, model registry or skills installation.

This initial implementation tunes **fixed-budget reasoner SFT** against a
held-out native reasoner evaluation. It supports sequential random/Bayesian
search in a local Linux Python environment or Docker container. It is not yet
a general optimizer for every Framework training/inference workflow.

## Installation and release status

The independently built `nvidia-automl-core` 0.1.0 wheel must be supplied by the
maintainer: **this change does not publish it to an index**. Install the wheel
first, then install the extra into your already-prepared Framework environment:

```bash
python -m pip install /path/to/nvidia_automl_core-0.1.0-py3-none-any.whl
python -m pip install -e '.[automl,workflows]'
cosmos-automl --help
```

Normal `pip install 'cosmos-framework[automl]'` (and resolving `uv --all-extras`)
requires the core wheel to be available on your package index or supplied via
a wheelhouse. Publishing that dependency and updating the resolver lock is a
release gate, not something the helper silently works around with a source
checkout. The engine's staging source/build instructions live in the separate
AutoML project at `packages/automl-core`; none of the commands here read it.

The controller is CPU-only. The execution environment must separately have
Framework's training and reasoner evaluation prerequisites, GPU resources,
checkpoints and datasets. See [setup](setup.md), [post-training](../.agents/skills/cosmos3-post-training/SKILL.md)
and the [reasoner guide](../tools/cosmos_workflows/models/cosmos3-reasoner/guide.md).
The trial handoff also uses `tomli-w` (included by the AutoML extra). A container
only needs these workload dependencies, not the controller or TAO SDK.

Preflight both training **and the selected evaluator** in that environment.
The example evaluation selects `vision.video_decoder = "torchcodec-cuda-on-demand"`,
which uses Framework's CUDA TorchCodec preprocessor. Omitting the decoder selects
the evaluator's PyNvVideoCodec path and requires its separate dependencies;
a successful TorchCodec probe does not validate that path. For NVIDIA Docker
GPU video decoding, expose `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`
in the image or export it and include its name in `execution.environment`.
Also verify that the non-root container has a valid user identity (for example,
forward `USER` and `LOGNAME`) and writable per-trial caches.

## Prepare and review

Start with [reasoner.yaml](../examples/automl/reasoner.yaml),
[training.toml](../examples/automl/training.toml) and
[evaluation.toml](../examples/automl/evaluation.toml). These are **planning
templates**, not launch-ready data/checkpoint selections. Replace the training
recipe with one already prepared for your dataset and matching checkpoint;
the example preset selects LLaVA data, not an arbitrary local JSONL.

Specify:

- Metric path and direction, e.g. `overall.accuracy` / `maximize`. Confirm that
  the selected task actually emits that metric. Validation data must be
  disjoint from training; keep a final test split untouched by search.
- Search domains and seed. Current supported axes are `optimizer.lr`,
  `optimizer.weight_decay`, `trainer.grad_accum_iter`, and
  `dataloader_train.max_samples_per_batch`. Include each field in the base TOML.
  Other axes need explicit adapter support; accepted TOML fields are not
  automatically safe search knobs (some are ignored for VLM).
- Fixed positive `trainer.max_iter` and `checkpoint.save_iter` that divides
  it. Epoch-based training/checkpoint schedules are not part of this adapter.
  The evaluator refuses an earlier checkpoint even after training exits zero.
- Initial weights follow native VLM semantics: set `checkpoint.load_path` to
  a DCP checkpoint for a warm start, or explicitly set it to `""` to load HF
  weights from an explicit `model.backbone.model_name` (and, when separate,
  `model.backbone.safetensors_path`). An HF directory is not a DCP load path.
- Maximum trials, per-trial timeout (including checkpoint export/evaluation),
  total elapsed timeout, chosen GPU IDs and `nproc_per_node`. The total clock
  starts at experiment creation, includes downtime, and is not reset on resume.
  Failed trials count against the budget. There is no automatic baseline run.
- Native evaluation TOML for `mcq`, `bcq`, `binary`, `its_directionality`, or
  `metropolis_sgd`. Set training `model.lora_enabled` explicitly so evaluation
  never guesses the mode inherited from a preset. The first adapter scores a nonempty classification task;
  freeform/captioning metrics and video-generator quality need separate result
  contracts. LoRA recipes retain native LoRA handling and must supply
  evaluation `model.base_model_path`; no PBT weight transplant is implied.
- Immutable training/validation/code revisions in `search.provenance`.
  The manifest binds recipe/evaluation contents, but external datasets, recipe
  Python code, checkpoints and mutable credential-backed resources are not
  automatically content-hashed. Do not change them during a search.

Manifest recipe/evaluation/cwd/mount paths resolve relative to the manifest.
Paths *inside* the native TOMLs retain Framework semantics (usually execution
cwd); use absolute mounted paths for assets. Hydra `${oc.env:NAME}` stays in
the recipe; declare NAME in `execution.environment` and export it in the
session. Do not put credentials in manifests or command arguments. Environment
values are not saved in the ledger. Logs remain the workload's responsibility.

```bash
cosmos-automl plan --manifest examples/automl/reasoner.yaml --workspace outputs/automl
# After resolving template fields and reviewing the plan/budget:
cosmos-automl run --manifest examples/automl/reasoner.yaml --workspace outputs/automl --approve
```

`python -m cosmos_framework.scripts.automl` is the equivalent module entrypoint.
Planning validates the structured SFT schema and domain endpoints without
writing files or importing training/GPU modules. It does **not** Hydra-compose
the preset, validate all cross-parameter combinations, check model memory,
check dataset overlap, download assets, or establish GPU compatibility.
Each actual candidate is validated again before launch. Native training still
performs its own full configuration and dependency checks.

If using Docker, replace the execution block explicitly:

```yaml
execution:
  kind: docker
  python: /opt/venv/bin/python
  image: your-reviewed-framework-image@sha256:your-digest
  gpu_devices: '0'
  mounts: [/absolute/checkpoints, /absolute/datasets]
  environment: [BASE_CHECKPOINT_PATH]
```

The image must already exist locally. No implicit registry login or pull occurs.
The Framework cwd and extra assets are mounted read-only at identical paths;
only the trial directory is writable. Containers run as the controller's UID/GID
with a writable per-trial HOME, so output ownership permits host-side collection;
the prepared image/interpreter must support that UID. Venv execution is not a sandbox. Both
adapters require trusted foreground commands; remote scheduler submission is
not part of this first execution contract. `nproc_per_node` must agree with
the recipe's parallelism and allocated GPUs. Evaluation currently uses one GPU.

## Results, monitoring and recovery

Each trial runs native `torch.distributed.run -m cosmos_framework.scripts.train
--sft-toml=...`, with a separate output root and fixed training seed. It starts
fresh from the declared initial checkpoint (`load_training_state=false`). The
trial then locates the committed DCP pointer, checks the expected iteration,
and invokes the existing reasoner evaluator. Native checkpoint preparation
handles DCP-to-HF export; no new exporter or inference implementation is added.

```bash
cosmos-automl status --experiment-id reasoner-lr-search --workspace outputs/automl
cosmos-automl logs --experiment-id reasoner-lr-search --workspace outputs/automl
cosmos-automl resume --manifest search.yaml --workspace outputs/automl --approve
cosmos-automl cancel --experiment-id reasoner-lr-search --workspace outputs/automl --approve
```

Resume uses the same manifest and ID. Inspection/cancellation can use just the
workspace and ID even if source manifests have moved. `status` consults native
process/container state for active trials; logs prints log paths. Cancellation
stops execution and prevents future trials. Resume checks manifest identity and
the venv package inventory or Docker image ID and reconciles durable worker
receipts. It never guesses “latest experiment.” An `UNKNOWN` launch stops the
controller rather than submitting another job. Inspect its log and native
process/container; if unrecoverable, stop it explicitly and use a new ID.

`<workspace>/<experiment_id>/experiment.json` is authoritative; `best.json`
is the best successful observation, with sampled config and paths to its
native TOML, DCP and metrics. A trial needs successful commands, a finite metric,
nonempty evaluation and valid trial-local artifacts. No successful trial means
`best=null` and a nonzero CLI exit. A best observed validation score is not final
test evidence. Training configs, checkpoints, containers and logs are retained;
no automatic cleanup deletes them.

## Scope and checks

The shared engine reuses TAO's GP/EI numerical implementation, not its SDK runner
or model selector. Other existing TAO algorithms are unchanged in that consumer.
ASHA/Hyperband fidelity, population checkpoint reuse, parallel search,
LLM/GEPA suggestions and remote executors are deferred pending real capability
contracts. They are not advertised by the Framework skill.

Run adapter tests with an installed core wheel (no GPU imports required):

```bash
python -m pytest -o addopts='' --confcutdir=tests/automl tests/automl
python -m ruff check cosmos_framework/automl cosmos_framework/scripts/automl.py tests/automl
```

These validate planning, native paths, schema guards, discovery and checkpoint
handoffs using fixtures. They are not end-to-end GPU training/evaluation tests.

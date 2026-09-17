# Cosmos workflow skills

This directory owns 29 Cosmos and supporting skills migrated from TAO Skill Bank. It is
self-contained: no Skill Bank checkout, plugin, home-directory installation, or
source overlay is required. The existing five `cosmos3-*` skills remain the
entrypoints for native setup, code navigation, debugging, generation, and recipe
post-training. These additional skills cover managed reasoner training and
evaluation, retrieval, synthetic data, and iterative improvement workflows.

## Discovery and routing

The canonical packages live in `skills/<category>/<name>`. Relative symlinks in
the repository's `.agents/skills` and `.claude/skills` expose the same packages to
both agents without duplicated implementations. Resolve symlinks when reading
skill-relative scripts and references. Do not copy individual skill directories:
the shared scripts, templates, schemas, and `versions.yaml` are part of the bundle.

| Request | Skill |
| --- | --- |
| Native Cosmos3 generation, Ray/Gradio, or recipe development | Existing `cosmos3-inference`, `cosmos3-post-training`, and navigation skills |
| Reasoner SFT, dense/PEFT, evaluation, checkpoint preparation, or backend comparison | `cosmos3-reasoner` |
| Video/text embeddings, retrieval, or embedder fine-tuning | `cosmos-embed` |
| Synthetic video generation with PAIDF | `cosmos-predict` |
| Cosmos3-based AnomalyGenNext fine-tuning and synthetic defects | `cosmos-finetune-anomalygennext`, `cosmos-prepare-anomalygennext-inputs`, `cosmos-generate-od-defects` |
| Image embeddings for anomaly preparation or mining | `cosmos-generate-image-embeddings` |
| DAFT-to-Cosmos dataset conversion and validation | `cosmos-convert-dataset-format`, `cosmos-validate-dataset-format` |
| Video captioning and reasoning QA annotations | `cosmos-annotate-videos` |
| Inspection improvement with proxy/benchmark isolation | `cosmos-deft-aoi` |
| Traffic-video improvement with embedding-based mining | `cosmos-deft-traffic` |
| Explicit hyperparameter or prompt optimization | `cosmos-automl` |
| Containerized inference endpoint | `cosmos-inference-service` |
| Platform selection, lifecycle, and result tracking | `cosmos-launch-workflow` plus `cosmos-run-on-<platform>` |

The supporting data, artifact, host-setup, and platform skills are bundled because
these workflows invoke them. Unrelated TAO model skills, plugin installers,
customer-specific evaluation jobs, and marketplace hooks are not included.
The RT-DETR-specific DEFT application is outside this bundle; its Cosmos3-based
AnomalyGenNext leaves and their embedding dependency are included independently.

## Use from a checkout

From the Cosmos Framework repository root:

```bash
source tools/cosmos_skills/env.sh
python "$COSMOS_SKILLS_ROOT/scripts/list_tao_capabilities.py" --format text
python "$COSMOS_SKILLS_ROOT/scripts/resolve_tao_model.py" \
  --model nvidia/Cosmos3-Nano --action train --workload training \
  --backend cosmos-framework
```

Shared command paths such as `scripts/resolve_tao_model.py`, `skills/...`, and
`templates/...` are relative to this bundle, not the framework repository root.
For those examples, first `cd "$COSMOS_SKILLS_ROOT"`. A skill's own `scripts/`,
`references/`, or `assets/` paths remain relative to that skill's directory.
Python shared helpers also find the bundle from their own location, so their
absolute paths work from any working directory without setting an environment
variable. `COSMOS_SKILLS_ROOT` is an explicit override for a relocated bundle.

Resolve the model/action before selecting a backend or image. Preserve explicit
backend choices; moving the workflows here does not change the existing
Cosmos-RL/Framework selection policy. The launch skill owns platform selection,
preflight, review, and the `submit/status/logs/cancel` contract. Creating a plan
does not authorize GPU jobs, downloads, registry operations, or system changes.

## Runtime compatibility

Public skill names use the Cosmos namespace. Existing `tao_*` helper filenames,
container commands, `TAO_*` runtime variables, schemas, and `.tao` job records
remain where required by the packaged runtimes. They do not imply a dependency
on the deprecated Skill Bank repository. `migration.json` records the original
commit, skill-name mapping, source paths, and hashes; it is provenance, not a
runtime dependency or an integrity seal for the maintained destination files.

Container images and optional AutoML wheels remain pinned in `versions.yaml`.
The model containers, TAO Data Services, PAIDF/AnomalyGenNext, and the optional
`nvidia-tao-automl` and `nvidia-tao-daft` packages are still external runtime dependencies. AutoML retains
its transitive SDK dependency; the ordinary platform execution paths use native
CLIs. Some inherited images require registry access. This migration does not
publish replacement images or change those runtime contracts.

## CPU validation

These tests use small fixtures and mocked platform commands; they do not require
model weights, a GPU, registry credentials, or an installed framework runtime.

```bash
cd tools/cosmos_skills
python -m pip install -r requirements-test.txt
python -m pytest --confcutdir=. -q
python scripts/stamp_versions.py --check
```

The local pytest configuration isolates these tests from the framework's GPU
fixtures. Live training and serving remain separate validation steps requiring
the selected platform and model assets.

Imported sources retain their license headers and per-skill license metadata;
see `LICENSE` and `NOTICE`.

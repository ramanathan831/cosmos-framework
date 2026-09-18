# Cosmos workflow skills

This directory owns 18 Cosmos and supporting skills migrated from TAO Skill Bank. It is
self-contained: no Skill Bank checkout, plugin, home-directory installation, or
source overlay is required. The existing five `cosmos3-*` skills remain the
entrypoints for native setup, code navigation, debugging, generation, and recipe
post-training. These additional skills cover managed reasoner training and
evaluation, retrieval, video generation, and training-data preparation.

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
| DAFT-to-Cosmos dataset conversion and validation | `cosmos-convert-dataset-format`, `cosmos-validate-dataset-format` |
| Video captioning and reasoning QA annotations | `cosmos-annotate-videos` |
| Containerized inference endpoint | `cosmos-inference-service` |
| Platform selection, lifecycle, and result tracking | `cosmos-launch-workflow` plus `cosmos-run-on-<platform>` |

The supporting artifact, data-I/O, host-setup, and platform skills are bundled
because these workflows invoke them. Docker, SLURM, Kubernetes, Brev, and
virtualenv execution remain available; the selected model/action contract still
determines which platforms fit. Host setup retains the GPU compatibility checks,
and the inference-service skill supplies the optional annotation endpoint.

## Deferred scope

DEFT AOI/traffic loops, anomaly generation and AnomalyGenNext, image embedding
and neighbor mining, gap analysis, and AutoML/HPO are not included in this core
port. Their full migration is preserved on
[`archive/cosmos-skills-full-migration`](https://github.com/ramanathan831/cosmos-framework/tree/archive/cosmos-skills-full-migration/tools/cosmos_skills)
for separate, reviewable follow-ups. Do not route tasks to those absent skills
or infer runnable AutoML support from model schema tuning metadata.

Unrelated TAO model skills, plugin installers, customer-specific evaluation
jobs, and marketplace hooks are also excluded. The native five skills and the
parent branch's framework changes are untouched by this scope reduction.

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

Container images remain pinned in `versions.yaml`. The model containers, TAO
Data Services, PAIDF, and optional `nvidia-tao-daft` package remain external
runtime dependencies. Platform execution uses native CLIs; this core bundle
does not require the TAO SDK or AutoML wheels. Some inherited images require
registry access. This migration does not
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

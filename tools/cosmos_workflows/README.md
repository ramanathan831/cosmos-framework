# Cosmos workflow support

These are supporting references and CPU-only orchestration helpers for the
framework's existing skills, not a second skill bank or plugin. Native Python
training/inference implementations remain in `cosmos_framework/`.

## Ownership and routing

| User task | Skill entrypoint | Supporting resources here |
| --- | --- | --- |
| Recipe SFT, reasoner video-QA SFT/evaluation, checkpoint preparation | `cosmos3-post-training` | `models/cosmos3-reasoner/` |
| Native generation, reasoner inference, container endpoints | `cosmos3-inference` | `inference-service/`, reasoner evaluation contracts |
| Installation, managed launch, credentials, platform/GPU checks | `cosmos3-setup` | `execution/`, `scripts/`, `templates/` |
| Environment and managed-job failures | `cosmos3-env-troubleshoot` | Execution retry and reasoner error references |
| Video-text embeddings and retrieval | `cosmos-embed` | `models/cosmos-embed/` |
| Explicit PAIDF video generation | `cosmos-predict` | `data/cosmos-predict/` |
| Multi-stage caption/description/reasoning-QA generation | `cosmos-annotate-videos` | `data/cosmos-annotate-videos/` |

Only the last three are new skills, exposed through relative discovery symlinks
in `.agents/skills` and `.claude/skills`. The four existing skill files are
extended in both directories. Setup and launch references are read on demand,
not loaded for unrelated code edits or ordinary native recipe questions.

Framework requests explicitly select `--backend cosmos-framework`; requested
Cosmos-RL jobs and explicit backend comparisons remain supported. The imported
resolver's historical `auto` behavior is retained for compatibility, not used
to silently replace a native Framework workflow. All five execution options
(Docker, SLURM, Kubernetes, Brev, virtualenv) remain as supporting references;
the model/action contract determines which fit.

## Helpers

From the repository root:

```bash
source tools/cosmos_workflows/env.sh
python "$COSMOS_WORKFLOWS_ROOT/scripts/resolve_tao_model.py" \
  --model nvidia/Cosmos3-Nano --action train --backend cosmos-framework
```

Shared command paths in the references are relative to this directory. Use
`cd "$COSMOS_WORKFLOWS_ROOT"` for those examples. A component's `scripts/`,
`references/`, and `assets/` paths are relative to that component. Helpers find
their own resources from `__file__`; they need no home-directory skill install.
`tao_*` filenames, runtime commands, image pins, `.tao` records, and backend
contracts are compatibility interfaces, not dependencies on the old repository.

The launch reference owns review/approval, nested specs, record-before-submit
ordering, and backend monitoring. Planning does not authorize jobs, registry
operations, downloads, or host mutations. Model containers, TAO Data Services,
PAIDF, and optional DAFT tooling remain external runtime dependencies.

## Scope and provenance

The full original migration is preserved on
[`archive/cosmos-skills-full-migration`](https://github.com/ramanathan831/cosmos-framework/tree/archive/cosmos-skills-full-migration/tools/cosmos_skills).
DEFT, anomaly generation, mining, and AutoML/HPO remain deferred. The interim
18-skill version is preserved on `archive/cosmos-skills-pre-integration`.

This integration removes the parallel setup/capability registry and AutoML
catalog/schema-generation tooling. Runtime request/job schemas and tests stay
with the helpers that use them. Domain prompt examples already available in
the annotation image are not duplicated here. `migration.json` records original
source hashes and destination paths, not current-file integrity hashes. Imported
license headers and metadata remain authoritative; see `LICENSE` and `NOTICE`.

## CPU validation

```bash
cd tools/cosmos_workflows
python -m pip install -r requirements-test.txt
python -m pytest --confcutdir=. -q
python scripts/stamp_versions.py --check --strict-strays
```

Tests cover model/backend/image resolution, sealed plans, checkpoint/evaluation
handoffs, execution contracts, relocation, and agent integration without model
weights, GPUs, registry access, or framework GPU fixtures. They do not substitute
for live training or inference validation.

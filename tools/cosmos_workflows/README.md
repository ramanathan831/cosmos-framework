# Cosmos workflow support

These are supporting references and CPU-only orchestration helpers for the
framework's existing skills. Native Python
training/inference implementations remain in `cosmos_framework/`.

## Ownership and routing

| User task | Skill entrypoint | Supporting resources here |
| --- | --- | --- |
| Recipe SFT, reasoner video-QA SFT/evaluation, checkpoint preparation | `cosmos3-post-training` | `models/cosmos3-reasoner/` |
| Native generation, reasoner inference, container endpoints | `cosmos3-inference` | `inference-service/`, reasoner evaluation contracts |
| Installation, managed launch, credentials, platform/GPU checks | `cosmos3-setup` | `execution/`, `scripts/`, `templates/` |
| Environment and managed-job failures | `cosmos3-env-troubleshoot` | Execution retry and reasoner error references |
| Explicit PAIDF video generation | `cosmos-predict` | `data/cosmos-predict/` |
| Multi-stage caption/description/reasoning-QA generation | `cosmos-annotate-videos` | `data/cosmos-annotate-videos/` |

Only the last two are new skills, exposed through relative discovery symlinks
in `.agents/skills` and `.claude/skills`. The four existing skill files are
extended in both directories. Setup and launch references are read on demand,
not loaded for unrelated code edits or ordinary native recipe questions.

Framework requests explicitly select `--backend cosmos-framework`; requested
Cosmos-RL jobs and explicit backend comparisons remain supported. The
resolver's `auto` behavior is retained for compatibility, not used
to silently replace a native Framework workflow. All five execution options
(Docker, SLURM, Kubernetes, Brev, virtualenv) remain as supporting references;
the model/action contract determines which fit.

## Helpers

From the repository root:

```bash
source tools/cosmos_workflows/env.sh
python "$COSMOS_WORKFLOWS_ROOT/scripts/resolve_cosmos_model.py" \
  --model nvidia/Cosmos3-Nano --action train --backend cosmos-framework
```

Shared command paths in the references are relative to this directory. Use
`cd "$COSMOS_WORKFLOWS_ROOT"` for those examples. A component's `scripts/`,
`references/`, and `assets/` paths are relative to that component. Helpers find
their own resources from `__file__`; they need no home-directory skill install.
Framework-owned helpers use Cosmos names, with job records under
`~/.cosmos/jobs` by default (`COSMOS_STATE_DIR` overrides the state root).
Render commands take `--cosmos-job-id`; workflow metadata uses `cosmos_job_id`
and `terminal_runtime_status`. Kubernetes selection uses `COSMOS_K8S_CONTEXT`
and `COSMOS_K8S_NAMESPACE`. Existing state directories are not moved or deleted.
External image URIs, package imports, and runtime protocols must match the
selected backend; changing their spelling does not port their implementation.

The launch reference owns review/approval, nested specs, record-before-submit
ordering, and backend monitoring. Planning does not authorize jobs, registry
operations, downloads, or host mutations. Dataset validation/conversion,
annotation, reasoner actions, checkpoint handling, service adapters, and lifecycle
logging now have implementations in `cosmos_framework/`. Install the optional
`workflows` extra for these capabilities. GPU dependencies remain separate from
the CPU orchestration helpers.

Build images from this checkout: the root `Dockerfile` for Framework,
`docker/cosmos-rl.Dockerfile` for the optional native Cosmos-RL integration,
`docker/quantize.Dockerfile` for its isolated quantization dependencies, and
`docker/predict-service.Dockerfile` to extend an official Predict 2.5 image.
The `*:local` values in `versions.yaml` are local build targets, not published
images. Remote platforms require a user-built registry image/digest or an
explicitly materialized SLURM SQSH. Native Cosmos-RL and Predict are independent
optional backends; PAIDF remains a separate explicitly selected service.

## Scope

Runtime request/job schemas and tests stay with the helpers that use them.
Domain prompts live with the native annotation implementation, not in a second
copy beside the skill. License headers and metadata remain authoritative; see
`LICENSE` and `NOTICE`. File attribution is recorded in `migration.json`.

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

Additional runtime contract tests are in `tests/workflow_runtime/`; run them
with `pytest --confcutdir=tests/workflow_runtime -c /dev/null tests/workflow_runtime`.
The optional Cosmos-RL integration expects the native APIs checked by its
preflight. Framework owns the conversation, collation, validation-cache,
decoder-worker, and status extensions. GPU builds, distributed execution,
quantization, model serving, and paid API calls require separate validation.

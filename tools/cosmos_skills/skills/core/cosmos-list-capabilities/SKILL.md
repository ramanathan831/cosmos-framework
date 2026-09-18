---
name: cosmos-list-capabilities
description: List the managed Cosmos models, workflow skills, platforms, and actions bundled with Cosmos Framework. Use for capability and supported-action questions about this workflow bundle; native generation and code navigation use the existing cosmos3 skills.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
allowed-tools: Read Bash
---

# Cosmos workflow capabilities

Use the packaged manifests as evidence. Resolve this checkout's
`tools/cosmos_skills` directory with `cosmos-workflow-setup`, then run:

```bash
python "$COSMOS_SKILLS_ROOT/scripts/list_tao_capabilities.py" --format text
python "$COSMOS_SKILLS_ROOT/scripts/list_tao_models.py" --scope all --format text
```

These are read-only helpers. They discover only this bundle's application,
data, model, and platform packages. They do not describe all capabilities of
the Cosmos Framework Python package; use the native `cosmos3-*` skills for
generation modalities, Ray/Gradio, and recipe development.

For a specific model/action, run `scripts/resolve_tao_model.py` with the
requested `--model`, `--action`, `--workload`, and explicit backend if given.
Report the selected backend and its rationale. A schema describes an action's
configuration; the selected backend contract determines whether it is
supported. Do not infer backend parity from a top-level action list.

AutoML/HPO and DEFT workflows are not bundled. Retained model schemas may carry
tuning metadata for compatibility, but that does not make the deferred AutoML
runner available. Report only the installed workflows.

---
name: cosmos-workflow-setup
description: Initialize the Cosmos workflow bundle in a Cosmos Framework checkout. Use before managed Cosmos reasoner, Embed, Predict, DEFT, or AutoML workflows to resolve local helpers, runtime dependencies, and platform routing. Native framework installation uses cosmos3-setup.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
allowed-tools: Read Bash
---

# Cosmos workflow setup

The bundle is maintained at `tools/cosmos_skills` inside Cosmos Framework.
Resolve this skill's real path: the bundle is three directories above its skill
directory. The repository's `.agents/skills` and `.claude/skills` entries are
discovery symlinks. Do not install a Skill Bank plugin or rewrite global agent
configuration.

From the framework repository root, source:

```bash
source tools/cosmos_skills/env.sh
python "$COSMOS_SKILLS_ROOT/scripts/list_tao_capabilities.py" --format text
```

For another working directory, use the absolute path to `env.sh`. Shell
environment does not persist across agent tool calls; source it in each call
that consumes it. Shared commands and paths beginning with `skills/` or
`templates/` resolve against `COSMOS_SKILLS_ROOT`; a skill's own references
and scripts resolve against its real directory.

## Route the task

- Native framework installation, generation, Ray/Gradio, recipe SFT, or source
  changes: use the existing `cosmos3-setup`, `cosmos3-inference`,
  `cosmos3-post-training`, `cosmos3-env-troubleshoot`, or
  `cosmos3-codebase-nav` skill.
- Managed reasoner train/evaluate and checkpoint preparation: use
  `cosmos3-reasoner`. Resolve the requested model, action, and workload with
  `scripts/resolve_tao_model.py`, then its packaged backend planner.
- Retrieval: `cosmos-embed`; synthetic videos: `cosmos-predict`; video QA
  annotation: `cosmos-annotate-videos`.
- Inspection or traffic improvement: `cosmos-deft-aoi` or
  `cosmos-deft-traffic`. Explicit HPO uses `cosmos-automl`.
- Container endpoints: `cosmos-inference-service`.

## Prepare execution

Read the model/action contract before selecting its image or runtime.
Preserve the explicit backend selection and packaged model/action defaults.
For several supported platforms, ask once if none was selected; a workflow
supporting only one has already selected its platform. Read that platform's
`cosmos-run-on-<platform>` skill and run its read-only preflight. Local Docker
is not a prerequisite for a remote SLURM workflow.

Invoke `cosmos-launch-workflow` before launching. Keep specs nested, create the
job record before native submission, and monitor the backend through
`submit/status/logs/cancel`. Resolve data staging through `cosmos-data-io`
only when the selected compute cannot already read the inputs.

Use `cosmos-setup-gpu-host` only for host setup requested by the user or a
missing host prerequisite; inspect first and obtain authorization before
system changes. Model-specific GPU requirements override platform defaults.

## Credentials and dependencies

Check only the presence of credentials needed by the selected operation.
Never print values or read credential-file contents. Use session variables or
a user-approved env file in the same shell as the consuming command. Registry
login, pulls, downloads, paid services, and GPU launches require authorization
for the concrete operation; reuse authorization already given for that scope.

The bundle needs no Cosmos workflow bundle install. Model containers, TAO Data Services,
PAIDF, native platform CLIs, and optional AutoML wheels remain external
dependencies. Image and wheel pins live in `versions.yaml`. Ordinary execution
uses native CLIs; AutoML alone uses `nvidia-tao-automl` and its transitive SDK.
Do not change a model's image or install heavyweight runtimes merely to run
read-only planners or CPU tests.

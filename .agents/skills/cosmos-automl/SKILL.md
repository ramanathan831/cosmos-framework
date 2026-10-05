---
name: cosmos-automl
description: Tune Cosmos Framework reasoner SFT hyperparameters with bounded random or Bayesian AutoML, native TOML recipes, DCP checkpoints, and held-out evaluation. Use when asked to search learning rate, weight decay, batch size or gradient accumulation, or to monitor/resume an AutoML experiment.
---

# Cosmos Framework AutoML

Read [docs/automl.md](../../../docs/automl.md) before planning a search. It owns
the installation/release prerequisites, manifest contract, result semantics
and current limits. Use the native `cosmos-automl` CLI; do not call a TAO runner
or introduce model/runtime selection. The engine is an optional wheel, not a
source-checkout dependency.

Reuse **cosmos3-setup** for environment preparation and
**cosmos3-post-training** for a prepared reasoner recipe and dataset/checkpoint
compatibility. AutoML starts from that explicit recipe, not a guessed model ID.

- Collect held-out data, objective **and direction**, domains, fixed training
  budget, seed, immutable data/code revisions, GPU allocation, and both trial
  count/time limits. Check training/validation separation. Keep final test data
  out of the optimization loop.
- Ask for venv versus Docker when not selected; neither is a default. Images
  and assets must be prepared separately with approval. Do not silently pull,
  download, increase resources, or fall back to another execution environment.
- Run `cosmos-automl plan --manifest <yaml> --workspace <directory>`. This is a
  CPU-only structural check, not a full Hydra/GPU preflight. Review the actual
  native commands, resource allocation, mounts, metric, budgets and prerequisites.
  Check the selected evaluator's decoder and export dependencies as well as
  training; a training-only GPU probe does not establish evaluation readiness.
- Obtain approval before `run ... --approve` or `resume ... --approve`.
  Approval covers the reviewed search budget, not unbounded trial retries.
- Use `status` / `logs` for evidence. `cancel ... --approve` stops native jobs.
  An `UNKNOWN` launch is an investigation gate, never permission to resubmit.
- Hand off `best.json`, the winning native TOML, DCP and held-out metrics, or
  report that no trial succeeded. Do not call validation improvement final test
  performance or describe fixture checks as GPU end-to-end verification.

Current workflow: sequential fixed-budget reasoner SFT with supported scored
tasks. No video-generator metric contract, ASHA/Hyperband, PBT checkpoint reuse,
LLM/GEPA spend or remote scheduler executor is implemented. Route ordinary
training/evaluation to the existing skills instead of forcing it through AutoML.

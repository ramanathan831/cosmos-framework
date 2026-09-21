# Cosmos workflow support

Read `README.md` for ownership and scope. This directory supports the existing
framework skills; do not recreate a parallel setup, launch, or capability skill
registry. Only Embed, Predict, and reasoning-QA annotation have new entrypoints.

- Keep native training and inference implementation in `cosmos_framework/`.
- Resolve helper paths here, never through a deprecated Skill Bank install.
- Preserve explicit backends, nested specs, launch approval, record-before-submit
  ordering, and backend-reported terminal state. Runtime `tao_*` names remain
  compatibility interfaces.
- Read only the selected model/platform references. Native recipes need not use
  the managed-job layer; framework tasks must not silently select Cosmos-RL.
- Keep helper tests CPU-only: `python -m pytest --confcutdir=. -q` here.
- Update both existing skill copies for routing changes; new skill symlinks in
  `.agents` and `.claude` must resolve to the same canonical directory.
- Keep source provenance in `migration.json` when moving or removing resources.

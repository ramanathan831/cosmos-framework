# Cosmos workflow bundle

This directory is the maintained home of the Cosmos workflows formerly packaged
in TAO Skill Bank. Read `README.md` for scope, routing, and CPU validation.

- Use the bundle here. Never install or locate the deprecated Skill Bank plugin.
- Canonical skills are under `skills/<category>/<name>`; `.agents/skills` and
  `.claude/skills` in the framework repository link to these same directories.
- Shared paths resolve against this directory. Source `env.sh` to set
  `COSMOS_SKILLS_ROOT` when following shell examples; skill-local references
  resolve against the real skill directory after following symlinks.
- Preserve the model/backend contracts, nested specs, preflight, user launch
  authorization, record-before-launch ordering, and native backend monitoring.
- Preserve runtime commands, image pins, and job formats unless explicitly
  changing their contracts. `tao_*` helper names are compatibility interfaces.
- Keep these helpers independent of heavyweight framework imports. Run CPU
  tests from this directory with `python -m pytest --confcutdir=. -q`.
- Update both discovery symlinks when adding a skill. Keep provenance in
  `migration.json`; it records original sources, not current-file checksums.

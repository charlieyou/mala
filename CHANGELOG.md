# Changelog

## Unreleased

### Cerberus v2 upgrade (BREAKING)

- `reviewer_type: cerberus` now invokes the v2 `cerberus` Go binary on `$PATH`. Users must install cerberus v2.
- Plugin auto-discovery from `.claude/plugins/cache/cerberus/...` was removed; cerberus must be on `$PATH`.
- `MalaConfig.cerberus_bin_path` was removed. Constructing `MalaConfig(cerberus_bin_path=...)` raises `TypeError`.
- Added `MalaConfig.cerberus_state_root` and `MalaConfig.cerberus_project_key` fields for factory-injected defaults and optional overrides.
- `--max-rounds 0` is rejected by v2; embedded v1-only flags in `code_review.cerberus.spawn_args` will break.
- Internal renames: `cerberus_gate_cli.py` -> `cerberus_cli.py`; `review_output_parser.py` -> `cerberus_output_parser.py`. No shims; downstream consumers must update imports.

### Changed

- Migrated validation evidence to the unified `MALA_EVIDENCE name=<name> exit=<code> log=<path>` protocol for built-in and custom validation commands. This removes marker-based custom-command evidence (`[custom:<name>:start|pass|fail|timeout]`) and consolidates the legacy split evidence fields into `ValidationEvidence.commands`. See `docs/validation.md` for the full evidence protocol and breaking behavior notes.

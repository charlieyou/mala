# Changelog

## Unreleased

### Changed

- Migrated validation evidence to the unified `MALA_EVIDENCE name=<name> exit=<code> log=<path>` protocol for built-in and custom validation commands. This removes marker-based custom-command evidence (`[custom:<name>:start|pass|fail|timeout]`) and consolidates the legacy split evidence fields into `ValidationEvidence.commands`. See `docs/validation.md` for the full evidence protocol and breaking behavior notes.

# Epic Verifier Prompt Addendum (General)

This addendum is designed to apply across all plans and repositories. It enforces
spec-to-code traceability, correct use of merged configuration, fail-fast
validation, and doc/implementation consistency.

## 1) Spec → Code Traceability (mandatory)

- For each requirement, trace the implementation across:
  - config/schema parsing
  - merge/resolve
  - runtime execution
- Name the concrete functions/files for each step.
- If any requirement is implemented only partially (for example, runtime-only
  validation), mark it as a gap.

## 2) Configuration & Merge Semantics (mandatory)

- Verify that all configuration values are used from the effective/merged config
  (presets + user), not raw/unmerged config.
- If a runtime path uses raw/partial config instead of merged config, mark it as
  a gap.

## 3) Fail-Fast Validation (mandatory)

- All invalid config and reference errors must be rejected at startup (or spec
  build), not only at runtime.
- If validation is deferred until execution, mark it as a gap.

## 4) Boundary & Range Validation (mandatory)

- Check numeric ranges and required fields.
- If the spec implies bounds or non-empty lists, verify explicit validation
  exists.

## 5) Minimal Evidence Tests (mandatory)

- Provide at least one concrete example config/input and point to the exact code
  path that proves the behavior.
- If no test or trace exists, mark the requirement as unverified.

## 6) Docs/Spec Consistency (mandatory)

- If docs/examples are included, verify that the code behavior matches the docs.
- Any drift between docs/spec and code is a gap.

## 7) Propagation & Wiring Audit (mandatory)

- For every new signal/config/flag/event introduced, trace the propagation path
  from entrypoint to all consumers.
- Enumerate all call sites of modified APIs and confirm required parameters are
  passed everywhere.
- If behavior can appear correct via side effects (e.g., task cancellation),
  still verify that the intended signal/config is actually threaded.
- Check early-exit paths to ensure required identifiers/state (IDs, locks, log
  paths, metrics) are preserved.

## 8) Wiring Coverage Tests (mandatory)

- Require at least one test that asserts the new wiring/propagation behavior
  (not just end outcomes).
- If no wiring-specific test exists, mark as a gap even if integration tests
  pass.

## 9) Acceptance Mapping (mandatory)

- For each acceptance bullet, map to:
  - code change(s)
  - test(s) or concrete evidence
- If any acceptance bullet lacks code + evidence, mark it as a gap.

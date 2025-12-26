# Quality Hardening Plan (Strict-by-Default)

Goal: maximize the probability that `mala run` produces correct, working code.
Constraint: no budget limits; all checks ON by default with explicit CLI opt-outs.

This document starts with data types and architecture, then lays out the
validation pipeline, CLI defaults, and an execution plan. It is intentionally
strict but modular and not overengineered.

## 1) Data Types (Core, Minimal, Extensible)

These are small dataclasses intended to keep responsibilities clear without
creating a heavyweight framework.

### ValidationSpec
Defines what to run.

```
ValidationSpec:
  commands: list[ValidationCommand]
  require_clean_git: bool
  require_pytest_for_code_changes: bool
  allow_lint_only_for_non_code: bool
  coverage: CoverageConfig
  e2e: E2EConfig
  scope: ValidationScope  # per-issue or run-level
```

### ValidationCommand
Represents a single command in a validation pipeline.

```
ValidationCommand:
  name: str
  command: list[str]
  kind: "deps" | "lint" | "format" | "typecheck" | "test" | "e2e"
  use_test_mutex: bool
  allow_fail: bool  # default false
```

### ValidationContext
Immutable context for a single validation run.

```
ValidationContext:
  issue_id: str | None
  repo_path: Path
  commit_hash: str
  log_path: Path | None  # claude session log
  changed_files: list[str]
  scope: ValidationScope
```

### ValidationResult
Output from a validation run.

```
ValidationResult:
  passed: bool
  failures: list[str]
  command_results: list[CommandResult]
  coverage_result: CoverageResult | None
  e2e_result: E2EResult | None
  duration_seconds: float
  artifacts: ValidationArtifacts
```

### CommandResult
Captured execution result for one command.

```
CommandResult:
  name: str
  exit_code: int
  duration_seconds: float
  stdout_path: Path | None
  stderr_path: Path | None
```

### CoverageConfig / CoverageResult
Minimal coverage config plus result fields.

```
CoverageConfig:
  enabled: bool
  min_percent: float  # default 85.0
  branch: bool  # default true
  report_path: Path | None

CoverageResult:
  percent: float
  passed: bool
  report_path: Path | None
```

### E2EConfig / E2EResult
True fixture-repo E2E (LLM-backed).

```
E2EConfig:
  enabled: bool
  fixture_root: Path
  command: list[str]  # e.g. ["uv", "run", "mala", "run", ...]
  required_env: list[str]  # e.g. MORPH_API_KEY, CLAUDE auth presence

E2EResult:
  passed: bool
  failure_reason: str | None
```

### WorktreePlan / WorktreeResult
Encapsulates clean-room validation.

```
WorktreePlan:
  root_dir: Path
  keep_on_failure: bool

WorktreeResult:
  path: Path
  removed: bool
```

### GateDecision
Result of an individual gate in the chain.

```
GateDecision:
  name: str
  passed: bool
  reasons: list[str]
  retryable: bool
```

## 2) Architecture Overview (Clean + Modular)

### Existing flow (today)
Agent -> Evidence gate (log-based) -> Optional Codex review -> Orchestrator close.

### Proposed flow (strict-by-default)
Agent -> Evidence gate -> Post-commit clean-room validation -> Codex review
-> Orchestrator close -> Run-level validation.

Key properties:
- Evidence gate remains fast and enforces compliance.
- Clean-room validation is the source of truth.
- Run-level validation verifies combined state across issues.
- Codex review stays as a correctness backstop.

### New modules (minimal set)
```
src/validation/spec.py      # build ValidationSpec from CLI/config
src/validation/runner.py    # run ValidationCommands, capture results
src/validation/worktree.py  # create/cleanup git worktrees
src/validation/coverage.py  # parse coverage reports
src/validation/e2e.py       # fixture repo creation + mala run
```

Existing modules updated:
- `src/orchestrator.py`: add validation stages + retry handling.
- `src/quality_gate.py`: treat as EvidenceGate only (log parsing).
- `src/logging/run_metadata.py`: record new validation results.

No large framework; each module is a small utility that the orchestrator calls.

## 3) Validation Pipeline (Gate Chain)

### Gate 1: EvidenceGate (existing)
- Requires `bd-<issue>` commit evidence and validation commands in logs.
- Updated to require full suite execution (including slow tests) by default.
- Still uses log offsets for same-session retries.

### Gate 2: Post-Commit Clean-Room Validation (new, per issue)
- Create git worktree at commit hash.
- Run full suite in the worktree.
- Fail closed if environment prerequisites are missing unless explicitly disabled.

### Gate 3: Codex Review (existing, default ON)
- Run after clean-room validation (decision: correctness first, review second).
- On failure: same-session re-entry, then re-run Gate 1 + Gate 2 + Gate 3.

### Gate 4: Run-Level Validation (new, after all issues)
- Run full suite at current HEAD (all commits).
- If failure: run ends with non-zero exit, record metadata, and create a
  follow-up bead (see Failure Handling).

## 4) CLI Defaults (Strict by Default, Opt-Outs Only)

Defaults:
- post-commit validation: ON
- run-level validation: ON
- slow tests: ON
- coverage: ON (min 85%)
- E2E fixture repo: ON
- codex review: ON

New CLI flags (disable-only):
- `--no-post-validate`
- `--no-run-level-validate`
- `--no-slow-tests`
- `--no-e2e`
- `--no-coverage`
- `--coverage-threshold <float>` (default 85.0)
- `--no-codex-review` (flip current default)

## 5) E2E Fixture Repo Design (True End-to-End)

Objective: run the real `mala run` command against a tiny fixture repo that
contains one trivial issue. This exercises the agent SDK, locks, gate, and
quality checks end-to-end.

Fixture repo plan:
1. Create temp dir under a dedicated root (e.g. `~/.config/mala/e2e-fixtures/`).
2. Initialize git repo, add minimal Python package skeleton.
3. `bd init` and create one tiny issue (e.g. "Add file hello.txt with content").
4. Run:
   ```
   uv run mala run --max-agents 1 --max-issues 1 <fixture-path>
   ```
5. Verify:
   - mala exits 0
   - issue closed
   - expected artifact created
6. Delete fixture directory unless `--keep-e2e-fixture` is passed.

Preconditions:
- Claude CLI authenticated
- MORPH_API_KEY present
If missing and `--no-e2e` not set, fail closed.

## 6) Coverage (High but Reasonable)

Default threshold: **85%** line + branch coverage.
Rationale: strong signal without being unrealistic for early runs.

Command (in clean worktree):
```
uv run pytest --cov=src --cov-branch --cov-report=term-missing --cov-fail-under=85
```

Notes:
- Coverage measured in post-commit validation and run-level validation.
- No diff-coverage yet (can be a later enhancement).

## 7) Failure Handling & Retries

Per-issue retries (same session):
- EvidenceGate failure -> re-entry
- Post-commit validation failure -> re-entry
- Codex review failure -> re-entry

Proposed retry limits:
- reuse `max_gate_retries` for EvidenceGate
- add `max_post_validate_retries` (default 2)
- reuse `max_review_retries` for Codex review

Run-level validation failure:
- Exit non-zero.
- Create a follow-up beads issue in the target repo (e.g. "Run-level validation failed").
- Record failure details in run metadata.

## 8) Observability / Run Metadata

Extend `RunMetadata` with:
- per-issue post-commit validation result (command outcomes + coverage/e2e)
- run-level validation result
- paths to validation logs/artifacts

Store raw stdout/stderr per command (in `~/.config/mala/validation/`).

## 9) Testing Plan

Unit tests:
- ValidationSpec building (defaults ON, disable flags work)
- Worktree creation and cleanup
- Coverage parsing and failure logic
- E2E config precondition checks

Integration tests:
- Orchestrator flow with mock ValidationRunner
- Retry loops for post-commit validation failures
- Run-level validation triggers and failure handling

Slow tests (optional, marked):
- Real fixture E2E run (requires API keys)

## 10) Rollout Phases

Phase 0: Data types + spec/runner scaffolding (no behavior changes).
Phase 1: Post-commit validation in worktree (strict ON).
Phase 2: Run-level validation at end of run (strict ON).
Phase 3: Coverage gate (85% default).
Phase 4: True fixture E2E run (strict ON).
Phase 5: Documentation updates (README, prompts).

## 11) Open Decisions (if you want to adjust)

- Confirm coverage threshold 85% (default proposed).
- Whether to keep Codex review before or after clean-room validation.
- Whether run-level failure should mark all issues `needs-followup`
  or only create a single follow-up issue (current plan: one follow-up issue).


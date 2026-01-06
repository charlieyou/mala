# Quality Hardening Plan v5 (Strict-by-Default)

Goal: maximize the probability that `mala run` produces correct, working code.
Constraint: no budget limits; all checks ON by default with explicit CLI opt-outs.

This document starts with data types and architecture, then lays out the
validation pipeline, CLI defaults, and an execution plan. It is intentionally
strict but modular and not overengineered.

## 1) Data Types (Core, Minimal, Extensible)

### ValidationSpec
Defines what to run for a given scope.

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

### ValidationScope
Explicit scope identifier.

```
ValidationScope:
  "per_issue" | "run_level"
```

### ValidationArtifacts
Minimal record of validation outputs for observability.

```
ValidationArtifacts:
  log_dir: Path
  worktree_path: Path | None
  worktree_state: "kept" | "removed" | None
  coverage_report: Path | None
  e2e_fixture_path: Path | None
```

### IssueResolution
Records how an issue was resolved.

```
IssueResolution:
  outcome: "success" | "no_change" | "obsolete"
  rationale: str
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
  required_env: list[str]  # e.g. CLAUDE auth presence (MORPH_API_KEY is optional)

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
Agent -> EvidenceGate -> Post-commit clean-room validation -> Codex review
-> Run-level validation -> Orchestrator close.

Key properties:
- EvidenceGate remains fast and enforces compliance.
- Clean-room validation is the source of truth for per-issue correctness.
- Run-level validation verifies combined state across all issues.
- Codex review remains a correctness backstop.
- Issues do not close until run-level validation passes.

### Minimal module split
```
src/validation/spec.py      # build ValidationSpec from CLI/config
src/validation/runner.py    # run ValidationCommands, capture results
src/validation/worktree.py  # create/cleanup git worktrees
src/validation/coverage.py  # parse coverage reports
src/validation/e2e.py       # fixture repo creation + mala run
```

Existing modules updated:
- `src/orchestrator.py`: add validation stages + retry handling + run-level close gate
- `src/quality_gate.py`: EvidenceGate only (log parsing)
- `src/logging/run_metadata.py`: record new validation results

## 3) Validation Pipeline (Gate Chain)

### Gate 1: EvidenceGate (existing)
- Requires `bd-<issue>` commit evidence and validation commands in logs,
  unless the no-op/obsolete path applies (see below).
- Expected evidence is derived from ValidationSpec by scope:
  - Per-issue EvidenceGate expects per-issue commands only (no E2E).
  - Run-level E2E evidence is only required at run-level validation.
- Must honor CLI opt-outs (e.g., `--disable integration-tests`).
- No-op / obsolete issue path (no commit required):
  - Agent logs an explicit marker: `ISSUE_NO_CHANGE` or `ISSUE_OBSOLETE`.
  - Include a short rationale in the log line.
  - Working tree must be clean (`git status --porcelain` empty).
  - EvidenceGate records `IssueResolution.outcome` and skips Gate 2 and Gate 3
    for this issue.

### Gate 2: Post-Commit Clean-Room Validation (per issue)
- Create git worktree at commit hash for this attempt.
- Run full suite in the worktree, including mandatory deps bootstrap:
  - `uv sync --all-extras`
- Worktree details:
  - Unique path per issue+attempt:
    `~/.config/mala/worktrees/<run_id>/<issue_id>/<attempt>`
  - Cleanup on pass/fail unless `--keep-worktrees` is set.
  - Repo-wide commands still use the global test mutex (shared LOCK_DIR).

### Gate 3: Codex Review (default ON)
- Run after clean-room validation (do not burn review cycles on broken builds).
- On failure: same-session re-entry, then re-run Gate 1 + Gate 2 + Gate 3.
- Commit hash per retry:
  - Each attempt re-reads the latest `bd-<issue>` commit hash.
  - Gate 2 uses the commit hash found in the current attempt.

### Gate 4: Run-Level Validation (after all issues)
- Run full suite at current HEAD, in a clean worktree.
- Includes `uv sync --all-extras` + full test/coverage/E2E defaults.
- Must pass before closing issues (strict-by-default).
- On failure: run ends non-zero, metadata recorded, follow-up issue created
  unless `--disable followup-on-run-validate-fail` is set.

## 4) CLI Defaults (Strict by Default, Opt-Outs Only)

Defaults:
- post-commit validation: ON
- run-level validation: ON
- integration tests: ON
- coverage: ON (min 85%)
- E2E fixture repo: ON (run-level only)
- codex review: ON

Disable list (repeatable or comma-separated):
- `--disable <value>` (repeatable, e.g., `--disable integration-tests --disable coverage`)
- `--disable <csv>` (comma-separated, e.g., `--disable "integration-tests,coverage"`)
  - values: `post-validate`, `run-level-validate`, `integration-tests`, `coverage`,
    `e2e`, `review`, `followup-on-run-validate-fail`

Other flags:
- `--keep-worktrees`

Opt-in exceptions:
- `--lint-only-for-docs`
- `--skip-e2e-if-no-keys`

## 5) E2E Fixture Repo Design (True End-to-End)

Objective: run the real `mala run` command against a tiny fixture repo that
contains one trivial issue. This exercises the agent SDK, locks, gate, and
quality checks end-to-end.

Fixture repo plan (run-level only):
1. Create temp dir under `~/.config/mala/e2e-fixtures/`.
2. Initialize git repo, add minimal Python package skeleton.
3. `bd init` and create one tiny issue (e.g., "Add file hello.txt with content").
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
- `bd` installed and available
- MORPH_API_KEY (optional; enables MorphLLM MCP tools when present)

Strict default: fail closed if missing and E2E is enabled
(i.e., `--disable` does not include `e2e`).
Optional dev flag: `--skip-e2e-if-no-keys` (warn and skip instead of failing).

## 6) Coverage (High but Reasonable)

Default threshold: 85% line + branch coverage.
If current coverage is lower, set `coverage.min_percent` in `mala.yaml` temporarily and
ratchet up over time.

Command (in clean worktree):
```
uv run pytest --cov=src --cov-branch --cov-report=term-missing --cov-report=xml --cov-fail-under=<threshold>
```

Where `<threshold>` is `CoverageConfig.min_percent` (default 85.0), configured
via `coverage.min_percent` in `mala.yaml`.

## 7) Policy Enforcement Details

### Clean git enforcement
`require_clean_git` is enforced at:
- Per-issue acceptance (before marking the issue success in orchestrator).
- Run-level validation start (before Gate 4).

### Code change classification
Defaults for code changes (require tests + coverage):
- Paths: `src/**`, `tests/**`, `commands/**`, `src/scripts/**`
- Files: `pyproject.toml`, `uv.lock`, `.env` templates
- Extensions: `.py`, `.sh`, `.toml`, `.yml`, `.yaml`, `.json`

Configuration (optional, not required for v1):
- Allow overrides in `pyproject.toml` or `.mala.toml`:
  - `code_paths`, `doc_paths`, `code_extensions`

### require_pytest_for_code_changes
If a change is classified as code and this flag is true:
- ValidationSpec must include pytest + coverage commands.
- `--lint-only-for-docs` is ignored.

### --lint-only-for-docs
If enabled and a change is classified as docs-only:
- Per-issue validation may skip tests + coverage + E2E.
- Run-level validation still runs the full suite unless explicitly disabled.

### EvidenceGate opt-out mapping
- `--disable integration-tests` => EvidenceGate excludes `@pytest.mark.integration` tests.
- `--disable coverage` => coverage evidence not required.
- `--disable e2e` => E2E evidence not required.

## 8) Worktree + venv notes

- Each issue+attempt uses a separate worktree; `uv sync` uses global cache,
  so repeated dependency installs are faster but still deterministic.
- Optional future enhancement: shared venv per run (deferred).

## 9) Failure Handling & Retries

Per-issue retries (same session):
- EvidenceGate failure -> re-entry
- Post-commit validation failure -> re-entry
- Codex review failure -> re-entry

Proposed retry limits:
- reuse `max_gate_retries` for EvidenceGate
- add `max_post_validate_retries` (default 2)
- reuse `max_review_retries` for Codex review

Retry exhaustion behavior:
- If a per-issue retry limit is hit, mark the issue failed and continue.
- Run-level validation still runs and will likely fail if issues are broken.

No-op / obsolete issues:
- Do not require commits or per-issue validation.
- Close only after run-level validation passes (strict-by-default).

Run-level validation failure:
- Exit non-zero.
- Create a follow-up beads issue unless disabled.
- Leave issues unclosed.

## 10) Observability / Run Metadata

Extend `RunMetadata` with:
- per-issue post-commit validation results (commands + coverage/E2E)
- run-level validation result
- artifacts (worktree state, logs, coverage reports)
- per-issue `IssueResolution` (success / no_change / obsolete)

Store raw stdout/stderr per command in `~/.config/mala/validation/`.

## 11) Testing Plan

Unit tests:
- ValidationSpec building (defaults ON, disable flags work)
- Worktree creation and cleanup
- Coverage parsing and failure logic
- E2E precondition checks

Integration tests:
- Orchestrator flow with mock ValidationRunner
- Retry loops for post-commit validation failures
- Run-level validation triggers and failure handling

Slow tests (optional, marked):
- Real fixture E2E run (requires API keys)

## 12) Rollout Phases

Phase 0: Data types + spec/runner scaffolding (no behavior changes).
Phase 1: Post-commit validation in worktree (strict ON).
Phase 2: Run-level validation in worktree (strict ON).
Phase 3: Coverage gate (85% default).
Phase 4: True fixture E2E run (strict ON).
Phase 5: Documentation updates (README, prompts).

## 13) Deferred / Future (Non-Blocking)

- Checkpoint/resume (`mala resume <run_id>`) using run metadata.
- Configurable code classification via `pyproject.toml` or `.mala.toml`.
- Shared venv per run for faster worktree validation.

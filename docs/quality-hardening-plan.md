# Quality Validation Reference

Date: 2025-12-27 (updated from original plan)

This document describes the validation system architecture. Originally a planning
document, it now serves as a reference for the implemented validation pipeline.

## 1) Data Types

### ValidationSpec
Defines what validations run for a given scope.

```
ValidationSpec:
  commands: list[ValidationCommand]
  require_clean_git: bool
  require_pytest_for_code_changes: bool
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
Coverage configuration and result fields.

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
Fixture-repo E2E configuration (LLM-backed).

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

## 2) Architecture Overview

### Validation Flow
Agent -> EvidenceGate -> Post-commit clean-room validation -> External review
-> Run-level validation -> Orchestrator close.

Key properties:
- EvidenceGate is fast and enforces compliance via log parsing.
- Clean-room validation is the source of truth for per-issue correctness.
- Run-level validation verifies combined state across all issues.
- External review (Cerberus review-gate) is a correctness backstop.
- Issues do not close until run-level validation passes.

### Module Structure
```
src/validation/spec.py      # build ValidationSpec from CLI/config
src/validation/runner.py    # run ValidationCommands, capture results
src/validation/worktree.py  # create/cleanup git worktrees
src/validation/coverage.py  # parse coverage reports
src/validation/e2e.py       # fixture repo creation + mala run
```

Related modules:
- `src/orchestrator.py`: validation stages + retry handling + run-level close gate
- `src/quality_gate.py`: EvidenceGate only (log parsing)
- `src/logging/run_metadata.py`: record validation results

## 3) Validation Pipeline (Gate Chain)

### Gate 1: EvidenceGate
- Requires `bd-<issue>` commit evidence and validation commands in logs,
  unless the no-op/obsolete path applies.
- Expected evidence is derived from ValidationSpec by scope:
  - Per-issue EvidenceGate expects per-issue commands only (no E2E).
  - Run-level E2E evidence is only required at run-level validation.
- Honors CLI opt-outs (e.g., `--disable-validations=integration-tests`).
- No-op / obsolete issue path (no commit required):
  - Agent logs an explicit marker: `ISSUE_NO_CHANGE` or `ISSUE_OBSOLETE`.
  - Include a short rationale in the log line.
  - Working tree must be clean (`git status --porcelain` empty).
  - EvidenceGate records `IssueResolution.outcome` and skips Gate 2 and Gate 3.

### Gate 2: Post-Commit Clean-Room Validation (per issue)
- Creates git worktree at commit hash for this attempt.
- Runs full suite in the worktree, including mandatory deps bootstrap:
  - `uv sync --all-extras` (skipped if pyproject.toml/uv.lock unchanged)
- Worktree details:
  - Unique path per issue+attempt:
    `~/.config/mala/worktrees/<run_id>/<issue_id>/<attempt>`
  - Cleanup on pass/fail unless `--keep-worktrees` is set.

### Gate 3: External Review (default ON)
- Uses Cerberus review-gate with multiple external reviewers (Codex, Gemini, Claude).
- Runs after clean-room validation (avoids burning review cycles on broken builds).
- Requires unanimous consensus from all available reviewers.
- On failure: same-session re-entry, then re-run Gate 1 + Gate 2 + Gate 3.
- Commit hash per retry:
  - Each attempt re-reads the latest `bd-<issue>` commit hash.
  - Gate 2 uses the commit hash found in the current attempt.

### Gate 4: Run-Level Validation (after all issues)
- Runs full suite at current HEAD, in a clean worktree.
- Includes `uv sync --all-extras` + full test/coverage/E2E defaults.
- Must pass before closing issues (strict-by-default).
- On failure: run ends non-zero, metadata recorded, follow-up issue created
  unless `--disable-validations=followup-on-run-validate-fail` is set.

## 4) CLI Defaults (Strict by Default)

Defaults (all ON):
- post-commit validation: ON
- run-level validation: ON
- integration tests: ON
- coverage: ON (min 85%)
- E2E fixture repo: ON (run-level only)
- external review: ON

Disable list (comma-separated via `--disable-validations <csv>`):
- `post-validate`: Skip test commands entirely
- `run-level-validate`: Skip run-level validation
- `integration-tests`: Exclude integration tests from pytest
- `coverage`: Disable coverage checking
- `e2e`: Disable E2E fixture repo test
- `review`: Disable external review (Cerberus review-gate)
- `followup-on-run-validate-fail`: Skip follow-up issue creation

Other flags:
- `--coverage-threshold <float>` (default 85.0)
- `--keep-worktrees`

## 5) E2E Fixture Repo Design

Objective: run the real `mala run` command against a tiny fixture repo that
contains one trivial issue. This exercises the agent SDK, locks, gate, and
quality checks end-to-end.

Fixture repo (run-level only):
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

## 6) Coverage

Default threshold: 85% line + branch coverage.
Override with `--coverage-threshold` if needed.

Command (in clean worktree):
```
uv run pytest --cov=src --cov-branch --cov-report=term-missing --cov-report=xml --cov-fail-under=<threshold>
```

## 7) Policy Enforcement Details

### Clean git enforcement
`require_clean_git` is enforced at:
- Per-issue acceptance (before marking the issue success in orchestrator).
- Run-level validation start (before Gate 4).

### Code change classification
All changes run the full validation suite. Classification is used for future
extensibility but does not currently affect validation behavior.

Code paths/files:
- Paths: `src/**`, `tests/**`, `commands/**`, `src/scripts/**`
- Files: `pyproject.toml`, `uv.lock`, `.env` templates
- Extensions: `.py`, `.sh`, `.toml`, `.yml`, `.yaml`, `.json`

Doc extensions: `.md`, `.rst`, `.txt`

### EvidenceGate opt-out mapping
- `--disable-validations=post-validate` => No pytest evidence required.
- `--disable-validations=integration-tests` => EvidenceGate expects unit-only pytest.
- `--disable-validations=coverage` => Coverage evidence not required.
- `--disable-validations=e2e` => E2E evidence not required.

## 8) Worktree Notes

- Each issue+attempt uses a separate worktree.
- `uv sync` uses global cache, so repeated dependency installs are fast.

## 9) Failure Handling & Retries

Per-issue retries (same session):
- EvidenceGate failure -> re-entry
- Post-commit validation failure -> re-entry
- External review failure -> re-entry

Retry limits:
- `max_gate_retries` for EvidenceGate (default 3)
- `max_review_retries` for external review (default 3)

Retry exhaustion behavior:
- If a per-issue retry limit is hit, mark the issue failed and continue.
- Run-level validation still runs and will likely fail if issues are broken.

No-op / obsolete issues:
- Do not require commits or per-issue validation.
- Close only after run-level validation passes.

Run-level validation failure:
- Exit non-zero.
- Create a follow-up beads issue unless disabled.
- Leave issues unclosed.

## 10) Observability / Run Metadata

`RunMetadata` includes:
- per-issue post-commit validation results (commands + coverage/E2E)
- run-level validation result
- artifacts (worktree state, logs, coverage reports)
- per-issue `IssueResolution` (success / no_change / obsolete)

Raw stdout/stderr stored in `~/.config/mala/validation/`.

## 11) Migration Notes (Cerberus Review-Gate)

### Breaking Changes (v0.x → Cerberus integration)

**CLI flag changes:**
- `--disable-validations=codex-review` → `--disable-validations=review`
- `--codex-thinking-mode` flag removed (reasoning effort now configured in Cerberus)

**Metadata field renames:**
- `codex_review_log_path` → `review_log_path` (in `IssueRun`)
- `codex_review` → `review_enabled` (in `RunConfig`)
- `codex_review_enabled` → `review_enabled` (in `LifecycleConfig`)

**Impact:** Historical run metadata files (in `~/.config/mala/runs/`) with the old field names will fail to parse. Old run data is not migrated—accept this as a clean break since historical metadata is non-critical.

**Code module changes:**
- `src/codex_review.py` deleted, replaced by `src/cerberus_review.py`
- `CodexReviewResult` → `ReviewResult` (new dataclass with Cerberus-compatible fields)
- `run_codex_review()` → `run_cerberus_review()` (new adapter function)

## 12) Deferred / Future

- Checkpoint/resume (`mala resume <run_id>`) using run metadata.
- Configurable code classification via `pyproject.toml` or `.mala.toml`.
- Shared venv per run for faster worktree validation.

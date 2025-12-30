# Cerberus Review Integration Plan

## Goal
Replace the custom Codex-only review flow in mala with Cerberus review-gate, using unanimous external reviewers and injecting the issue description into review prompts.

## Decisions
- Consensus is unanimous among spawned/available reviewers.
- No backward compatibility for Codex-specific flags or naming.
- Issue description must be injected into reviewer prompts.
- Cerberus CLI is the integration surface; mala does not embed Cerberus code.
- mala owns review retry policy; Cerberus "wait" is single-pass.

## Cerberus Contract (to implement)

### Commands

#### 1. Spawn command
```bash
review-gate spawn-code-review --diff <baseline>..<head> \
  [--context-file <path>] \
  [--codex-reasoning <low|medium|high>]
```

**Inputs:**
- `--diff <baseline>..<head>` (required): Git diff range to review
- `--context-file <path>` (optional): File containing issue description for reviewer context
- `--codex-reasoning <level>` (optional): Reasoning effort for Codex reviewer (default: high)

**Output (stdout):**
```json
{
  "session_key": "uuid-string",
  "reviewers_spawned": ["codex", "gemini", "claude"]
}
```

**Exit codes:** 0 = spawned successfully, 1 = spawn failed

#### 2. Wait command
```bash
review-gate wait --json [--timeout <sec>] [--session-key <key>]
```

**Inputs:**
- `--json` (required): Output structured JSON
- `--timeout <sec>` (optional): Max wait time in seconds (default: 300)
- `--session-key <key>` (optional): Session from spawn; defaults to most recent

**Output (stdout):** See JSON schema below

### JSON schema (minimum fields)
```json
{
  "status": "resolved|timeout|error",
  "consensus": {
    "verdict": "PASS|FAIL|NEEDS_WORK|no_reviewers",
    "iteration": 1
  },
  "reviewers": {
    "codex": { "verdict": "PASS|FAIL", "summary": "...", "issues": [...], "error": null },
    "gemini": { "verdict": "PASS|FAIL", "summary": "...", "issues": [...], "error": null }
  },
  "issues": [
    {
      "reviewer": "codex",
      "file": "src/foo.py",
      "line_start": 42,
      "line_end": 45,
      "priority": 1,
      "title": "Missing null check",
      "body": "The function does not handle None input..."
    }
  ],
  "parse_errors": ["gemini: malformed JSON response"]
}
```

**Issue schema fields** (required for mala's `ReviewIssue` protocol):
- `reviewer`: string — which reviewer found this issue
- `file`: string — file path where issue was found
- `line_start`: int — starting line number
- `line_end`: int — ending line number
- `priority`: int | null — 0=P0, 1=P1, 2=P2, 3=P3, or null
- `title`: string — short issue title
- `body`: string — detailed description

### Exit codes (consolidated reference)

| Exit Code | Meaning | mala Behavior |
|-----------|---------|---------------|
| 0 | PASS — all reviewers agree, no issues | `passed=True` |
| 1 | FAIL/NEEDS_WORK — legitimate review failure | `passed=False`, agent should fix |
| 2 | Parse error — malformed reviewer output | `passed=False, parse_error="..."`, retry review |
| 3 | Timeout — reviewers didn't respond in time | `passed=False, parse_error="timeout"`, retry review |
| 4 | No reviewers — no reviewer CLIs available | `passed=False, fatal_error=True`, abort |
| 5 | Internal error — unexpected failure | `passed=False, fatal_error=True`, abort |

**Rationale**: Exit code 1 vs 2 distinction allows mala to differentiate between:
- Exit 1: Reviewers found real issues → agent should attempt fix
- Exit 2: Tool/integration failure → retry review without agent changes

**Disambiguation rule for mala adapter:**
1. On exit code 1: Parse JSON, populate `issues` list, set `parse_error=None`. Agent gets actionable feedback.
2. On exit code 2: Parse JSON if possible to get error message from `parse_errors[]`; set `parse_error="<message>"` and `issues=[]`. Mala retries the review tool.
3. Key distinction: Exit 1 always has valid JSON with `issues`; exit 2 may have partial/missing JSON.

### Context injection
- mala will use `--context-file <path>` to pass issue description.
- The context file contains the issue title and body for reviewer prompts.
- The context file must have a unique name per issue to support parallel processing (e.g., `/tmp/claude/review-context-{issue_id}.txt`).

### Reasoning effort configuration

**Migration path for `--codex-thinking-mode`:**
1. mala currently passes `thinking_mode` through `CodeReviewer.__call__` (protocols.py:267)
2. After migration:
   - mala removes its `--codex-thinking-mode` CLI flag entirely
   - Cerberus accepts `--codex-reasoning <low|medium|high>` on `spawn-code-review`
   - Default reasoning effort is "high" (matching current mala default)
   - If mala needs to control reasoning effort in the future, it can pass `--codex-reasoning` to spawn

**Decision:** Reasoning effort configuration moves from mala CLI to Cerberus spawn command. mala will not expose this setting initially; Cerberus uses "high" by default.

## mala Implementation Plan
### 1) Code changes

#### 1.1) Replace `src/codex_review.py` with `src/cerberus_review.py`
- Delete `src/codex_review.py` entirely (585 lines of Codex-specific JSON parsing).
- Create new `src/cerberus_review.py` as a thin adapter:
  - `spawn_review(diff_range: str, context_file: Path, session_key: str) -> None`
  - `wait_for_review(session_key: str, timeout: int) -> ReviewResult`
  - Parse Cerberus JSON directly (no complex extraction needed since Cerberus returns structured output).

#### 1.2) Update `src/protocols.py` with new result type

**Key constraint:** `ReviewResult` must satisfy the `ReviewOutcome` protocol in `lifecycle.py`:
```python
class ReviewOutcome(Protocol):
    passed: bool
    parse_error: str | None    # NOTE: str | None, not bool!
    fatal_error: bool
    issues: list[ReviewIssue]  # NOTE: must be named "issues", not "findings"
```

**Field mapping from `CodexReviewResult` → `ReviewResult`:**

| Old field (CodexReviewResult) | New field (ReviewResult) | Mapping from Cerberus JSON |
|-------------------------------|--------------------------|----------------------------|
| `passed: bool` | `passed: bool` | `consensus.verdict == "PASS"` |
| `issues: list[ReviewIssue]` | `issues: list[ReviewIssue]` | `issues[]` array (keep same name!) |
| `raw_output: str \| None` | *(removed)* | Not needed; Cerberus returns structured JSON |
| `parse_error: str \| None` | `parse_error: str \| None` | Error message from exit code 2/3, or None |
| `fatal_error: bool` | `fatal_error: bool` | Exit code 4 or 5 |
| `attempt: int` | *(removed)* | mala tracks retry count externally |
| `session_log_path: Path \| None` | `review_log_path: Path \| None` | Cerberus session directory path |

**New `ReviewResult` dataclass (satisfies `ReviewOutcome` protocol):**
```python
@dataclass
class ReviewResult:
    passed: bool                     # consensus.verdict == "PASS"
    issues: list[ReviewIssue]        # aggregated from all reviewers (MUST be named "issues")
    parse_error: str | None          # error message if exit code 2/3, else None
    fatal_error: bool                # True if exit code 4 or 5
    review_log_path: Path | None     # Cerberus session directory
```

**`ReviewIssue` dataclass (matches Cerberus JSON schema):**
```python
@dataclass
class ReviewIssue:
    file: str           # file path
    line_start: int     # starting line number
    line_end: int       # ending line number
    priority: int | None  # 0=P0, 1=P1, 2=P2, 3=P3, or None
    title: str          # short issue title
    body: str           # detailed description
    reviewer: str       # which reviewer found this (NEW field)
```

**Dropped field:** `confidence_score: float` from old `CodexReviewResult` is intentionally removed. This field was Codex-specific and not used by lifecycle/runner logic. Cerberus reviewers don't provide confidence scores.

**Update `ReviewIssue` protocol in `lifecycle.py` (lines 80-116):**
- Add `reviewer: str` property to the protocol to match the new Cerberus schema
- This ensures structural typing compatibility when Cerberus returns issues with reviewer attribution

**Update `CodeReviewer` protocol (lines 247-284):**
- Change return type from `CodexReviewResult` to `ReviewResult`
- Remove `thinking_mode: str` parameter (reasoning effort moves to Cerberus config)
- Updated signature:
  ```python
  class CodeReviewer(Protocol):
      def __call__(
          self,
          diff_range: str,
          context_file: Path,
          timeout: int = 300,
      ) -> ReviewResult: ...
  ```

#### 1.3) Update `src/orchestrator.py`

**Rename class:** `DefaultCodeReviewer` → `DefaultReviewer` (or keep name, just update internals)

**Constructor change:** Add `repo_path: Path` to constructor. The adapter needs the repo path to execute `review-gate` CLI commands in the correct directory.

```python
@dataclass
class DefaultReviewer:
    repo_path: Path  # Required for CLI execution context

    async def __call__(self, diff_range: str, context_file: Path, timeout: int = 300) -> ReviewResult:
        ...
```

**Updated implementation flow:**
1. Write issue description to unique temp context file: `/tmp/claude/review-context-{issue_id}.txt`
2. Call `review-gate spawn-code-review --diff baseline..HEAD --context-file <path>` (in `repo_path` directory)
3. Parse spawn output JSON to get `session_key`
4. **Spawn error handling:** If spawn exits with code 1, return `ReviewResult(passed=False, issues=[], parse_error="spawn failed: <stderr>", fatal_error=False)`
5. Call `review-gate wait --json --session-key <key> --timeout <sec>`
6. Parse wait JSON output and map to `ReviewResult`

**Empty diff handling:** If diff is empty on the **initial review** (not a retry), return `ReviewResult(passed=True, issues=[], parse_error=None, fatal_error=False, ...)` without spawning.

**Important:** This PASS short-circuit only applies to the initial review. On retry attempts, an empty diff indicates the agent made no progress addressing review feedback — this should be treated as a failure by `AgentSessionRunner`'s existing no-progress detection logic, not by the review adapter.

**Exit code mapping (for `wait` command):**
- 0 → `passed=True, parse_error=None`
- 1 → `passed=False, parse_error=None` (legitimate failure, agent should fix)
- 2 → `passed=False, parse_error="<error message from JSON>"` (retryable tool failure)
- 3 → `passed=False, parse_error="timeout"` (retryable)
- 4 → `passed=False, fatal_error=True` (no reviewers available)
- 5 → `passed=False, fatal_error=True` (internal error)

#### 1.4) Update `src/lifecycle.py`

**Protocol compatibility:** The new `ReviewResult` dataclass satisfies `ReviewOutcome` protocol exactly:
- `passed: bool` → `ReviewResult.passed`
- `parse_error: str | None` → `ReviewResult.parse_error`
- `fatal_error: bool` → `ReviewResult.fatal_error`
- `issues: list[ReviewIssue]` → `ReviewResult.issues`

No changes to `ReviewOutcome` protocol itself are needed.

**Critical behavior change for parse_error handling:**
When `parse_error` is set (exit code 2/3), lifecycle should return `Effect.RUN_REVIEW` to retry the review tool directly, **not** `Effect.SEND_REVIEW_RETRY` which would prompt the agent. This implements the "retry review without agent changes" behavior:
- `parse_error` is set → `Effect.RUN_REVIEW` (loop back to review, don't prompt agent)
- `parse_error` is None and `issues` exist → `Effect.SEND_REVIEW_RETRY` (prompt agent with issues)

**Specific renames in lifecycle.py:**
- Line ~31: Update `Effect.RUN_REVIEW` docstring from "Codex review" → "External review"
- Line ~119-145: Update `ReviewOutcome` docstring to remove Codex references (line 123: "Callers (orchestrator) pass infra CodexReviewResult objects" → "...pass ReviewResult objects")
- Line ~180: Rename `LifecycleConfig.codex_review_enabled` → `review_enabled`
- Update all docstrings referencing "Codex" to say "external reviewers"

#### 1.5) Update `src/pipeline/agent_session_runner.py`

**This is a critical update** — the pipeline runner has extensive Codex-specific code (lines ~162–235, ~676–783).

**Import changes:**
- Remove: `from src.codex_review import CodexReviewResult, format_review_issues, run_codex_review`
- Add: `from src.cerberus_review import ReviewResult, format_review_issues, run_cerberus_review`

**Field renames:**
- `codex_review_enabled` → `review_enabled` (config field)
- `codex_review_log_path` → `review_log_path` (output field in `AgentSessionOutput` dataclass, line 234)
- `CodexReviewResult` → `ReviewResult` (type annotations)

**Function call updates:**
- Replace `run_codex_review(...)` calls with `run_cerberus_review(...)`
- `format_review_issues(result.issues)` call remains unchanged (field name is still `issues`)
- Update review retry logic to use `ReviewResult` fields
- Update `parse_error` usage: now `str | None` instead of `bool` — check `if result.parse_error:` still works

**Review follow-up prompt formatting:**
- The `issues` field structure is compatible; `format_review_issues` function moves to `cerberus_review.py` with same signature

#### 1.5.1) Update `src/pipeline/review_runner.py`

**This module orchestrates review execution** and has Codex-specific references:

**Import changes:**
- Line 26: Remove `from src.codex_review import CodexReviewResult`
- Add: `from src.cerberus_review import ReviewResult`

**Type annotation updates:**
- `ReviewOutput.result: CodexReviewResult` → `ReviewOutput.result: ReviewResult` (line 76)
- Update docstrings referencing "CodexReviewResult" and "codex"

**Config changes in `ReviewRunnerConfig` (lines 31-43):**
- Remove `thinking_mode: str | None = None` (no longer passed to reviewer)
- Update docstring: remove "reasoning effort level for reviewer"

**Method changes in `ReviewRunner.run_review()` (lines 141-148):**
- Remove `max_retries=2` parameter (Cerberus doesn't retry internally; mala owns retry)
- Remove `thinking_mode=self.config.thinking_mode` parameter
- Update call signature to match new `CodeReviewer` protocol:
  ```python
  result = await self.code_reviewer(
      diff_range=f"{input.baseline_commit or 'HEAD~1'}..{input.commit_sha}",
      context_file=context_file_path,  # Write issue_description to temp file
      timeout=300,
  )
  ```

#### 1.6) Update `src/prompts/review_followup.md`
- Change heading from "## Codex Review Failed" → "## External Review Failed".
- Update body text to reference "external reviewers" instead of "Codex".

### 2) CLI/config changes

#### 2.1) Update `src/cli.py`
- In `VALID_DISABLE_VALUES`: rename `codex-review` → `review`
- Usage changes from `--disable-validations=codex-review` to `--disable-validations=review`
- Remove `--codex-thinking-mode` flag entirely
- Add `--review-timeout <sec>` optional flag (default: 300s)

#### 2.2) Update `src/config.py`
- Check if `MalaConfig` has `codex_review_enabled` or `codex_thinking_mode` fields
- Rename `codex_review_enabled` → `review_enabled` if present
- Remove `codex_thinking_mode` if present

#### 2.3) Update runtime flag enforcement
Files that check for `"codex-review"` in disabled validations:
- `src/orchestrator.py`: Update `if "codex-review" in disabled:` → `if "review" in disabled:`
- `src/pipeline/agent_session_runner.py`: Update any disabled validation checks
- `src/lifecycle.py`: Update `LifecycleConfig.codex_review_enabled` → `review_enabled`

**All places using `codex_review_enabled` must be renamed to `review_enabled`:**
- `src/lifecycle.py` (LifecycleConfig)
- `src/pipeline/agent_session_runner.py` (config field)
- `src/event_sink.py` (RunConfig)
- `src/logging/run_metadata.py` (RunConfig)

### 3) Telemetry/logging

#### 3.1) Update `src/event_sink.py`

**Note:** The event methods (`on_review_started`, `on_review_passed`, `on_review_retry` at lines 254-295) already use generic "review" naming — no changes needed there.

**Changes needed in `RunConfig` dataclass (lines 18-38):**
- Rename `codex_review_enabled` → `review_enabled`
- Remove `codex_thinking_mode` field (reasoning effort no longer tracked in mala)
- Update any docstrings/comments referencing "Codex" to say "external reviewers"

#### 3.2) Update `src/logging/run_metadata.py`
- In `IssueRun` (line 72):
  - Rename `codex_review_log_path` → `review_log_path`
- In `RunConfig` (line 88):
  - Rename `codex_review` → `review_enabled`
- Note: Historical metadata files with old field names will fail to parse. Add migration note to rollout section.

### 4) Tests

#### 4.1) Replace `tests/test_codex_review.py`
- Delete existing Codex-specific tests.
- Create `tests/test_cerberus_review.py` with:
  - JSON parsing tests for Cerberus response format
  - Exit code mapping tests (0, 1, 2, 3, 4, 5)
  - Timeout handling tests
  - Empty diff short-circuit test

#### 4.2) Update `tests/test_agent_session_runner.py`
- Update imports from `codex_review` → `cerberus_review`
- Rename `codex_review_enabled` → `review_enabled` in config fixtures
- Rename `codex_review_log_path` → `review_log_path` in assertions
- Update mock return types to use new `ReviewResult`

#### 4.3) Update `tests/test_event_sink.py`
- Rename `codex_review_enabled` → `review_enabled` in test fixtures
- Remove `codex_thinking_mode` from test data

#### 4.4) Update `tests/test_run_metadata.py`
- Rename `codex_review_log_path` → `review_log_path` in test fixtures
- Rename `codex_review` → `review_enabled` in RunConfig tests

#### 4.5) Update `tests/test_lifecycle.py`
- Update `ReviewOutcome` mock objects to use new `ReviewResult` type
- Update tests for `parse_error` handling to verify new `Effect.RUN_REVIEW` behavior
- Rename `codex_review_enabled` → `review_enabled` in test fixtures
- Update Codex-specific docstring references in test descriptions

#### 4.5.1) Update `tests/test_orchestrator.py`
- Update `DefaultCodeReviewer` tests to use new `DefaultReviewer` with `repo_path` constructor
- Update mock Cerberus adapter calls
- Rename `codex_review_enabled` → `review_enabled` in test fixtures
- Update `"codex-review"` → `"review"` in disable validation tests

#### 4.5.2) Update `tests/test_review_runner.py`
- Update imports from `CodexReviewResult` → `ReviewResult`
- Update `ReviewRunnerConfig` fixtures: remove `thinking_mode` field
- Update `run_review()` mock expectations: remove `max_retries` and `thinking_mode` params
- Update mock reviewer signatures to match new `CodeReviewer` protocol

#### 4.6) Update CLI tests
- Update `--disable-validations=codex-review` tests to use `--disable-validations=review`
- Remove `--codex-thinking-mode` tests

### 5) Docs and naming cleanup

#### 5.1) Documentation updates
- Update `README.md`, `docs/quality-hardening-plan.md`, and coordination docs to reference Cerberus review-gate.

#### 5.2) Codex reference sweep
Perform a global search for "codex" (case-insensitive) and update all references:
- **Docstrings:** Update to say "external reviewers" or "review-gate"
- **Log messages:** Update `src/pipeline/agent_session_runner.py` log messages referencing "Codex"
- **Comments:** Update inline comments in all modified files
- **Variable names:** Already covered in field renames above

## Open Edge Cases to Validate
- Reviewer availability: missing CLIs should lead to `no_reviewers` (fail closed).
- Timeout behavior: map to retryable failure until `max_review_retries` exhausted.
- Merge commits or empty diffs: detect empty diff and skip review.

## Test Plan
- Unit: JSON mapping, exit code mapping, empty diff short-circuit.
- Integration: spawn/wait flow with a single reviewer available.
- Failure: simulate timeout and parse error outputs.

## Rollout
1. **Phase 1: Cerberus CLI** - Implement `review-gate spawn` command with diff args and `--context-file` support.
2. **Phase 2: mala adapter** - Create `src/cerberus_review.py`, update protocols and orchestrator.
3. **Phase 3: Pipeline update** - Update `agent_session_runner.py` with new imports and field names.
4. **Phase 4: Cleanup** - Delete `src/codex_review.py`, update CLI flags, telemetry, and tests.
5. **Phase 5: Docs** - Update all documentation references.
6. **Phase 6: Validation** - Run full test suite and validate on a small issue batch.

### Migration notes
- Historical metadata files with `codex_review_log_path` or `codex_review` fields will fail to parse after the rename. Options:
  - Accept breakage for old runs (recommended: old data is not critical).
  - Add field aliases in Pydantic models for backward-compatible parsing.

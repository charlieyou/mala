# Implementation Plan: Session Resume Flag

## Context & Goals
- **Spec**: N/A — derived from user description and interview decisions
- Enhance existing `--resume/-r` flag to also resume Claude SDK sessions for WIP issues
- When `--resume` is used, WIP issues should automatically attempt to resume their prior sessions
- Add `--strict` flag that fails issues when no prior session is found (or session is stale)
- Improves context efficiency and continuity for users iteratively working on issues

## Scope & Non-Goals
- **In Scope**
  - Extend `--resume` to trigger session resumption for WIP issues (tied together)
  - Session ID lookup from previous run metadata (scan recent runs, most-recent-first)
  - Pass session ID to SDK via existing `SDKClientFactory.with_resume()` pattern
  - New `--strict` flag to control failure behavior when no session found or stale
  - Logging of session resumption (verbose mode only)

- **Out of Scope (Non-Goals)**
  - Modifying how sessions are stored (reuse existing RunMetadata)
  - Changes to review loop itself (already has resume pattern)
  - Creating a separate session index file (scan approach is sufficient)
  - Per-issue resume control (global for run)
  - Per-issue granularity for the strict setting (global flag only)
  - Resuming sessions for issues not in WIP status (feature is tied to `--resume` workflow)

## Assumptions & Constraints
- `RunMetadata` files (`~/.mala/runs/{repo}/*.json`) already contain `session_id` in their `IssueRun` records
- Session resumption uses Claude SDK's `resume=session_id` option via `SDKClientFactory.with_resume()`
- Existing `with_resume()` pattern in `SDKClientFactory` is reusable (already used by `idle_retry_policy.py` and `context_pressure_handler.py`)
- `--resume` flag currently means "prioritize WIP" — we extend it to also resume sessions
- WIP issues are identified by `status: in_progress` in issue data (already used by `prioritize_wip`)

### Implementation Constraints
- No flag naming conflict: extend existing `--resume` behavior rather than adding new flag
- Must integrate with existing `AgentSessionRunner._initialize_session()` flow
- Session lookup: scan RunMetadata JSONs by most-recent-first (timestamp prefix in filename), stop at first match for issue_id
- Session resume logging only shown with `--verbose`
- "Strict" failure must apply to the specific issue being processed, not crash the entire orchestrator run

### Testing Constraints
- Unit tests required for session lookup helper (`lookup_prior_session`)
- Unit tests for strict mode behavior (fail when no session found)
- Integration/Contract tests to verify `resume_session_id` is correctly passed to the runner
- Must not break existing `--resume` (prioritize WIP) functionality

## Prerequisites
- [x] `RunMetadata` capture logic exists and is saving `session_id` (verified: `IssueRun.session_id` field exists)
- [x] `SDKClientFactory.with_resume()` signature supports `session_id` injection (verified in `src/infra/sdk_adapter.py`)
- [x] No blockers — existing patterns are reusable

## High-Level Approach

When `--resume` flag is used, the orchestrator will:
1. Prioritize WIP issues (existing behavior, unchanged)
2. For each WIP issue, look up the most recent session ID from prior `RunMetadata` files
3. Pass the session ID to `AgentSessionRunner` which uses `SDKClientFactory.with_resume()`
4. If `--strict` and no session found (or SDK rejects stale session), fail that issue only
5. If not strict (default), fall back to creating a new session

The session lookup scans `~/.mala/runs/{repo}/*.json` files in reverse chronological order (most recent first, using timestamp prefix in filename) and returns the first `session_id` found for the target issue. This is a simple approach that avoids maintaining a separate index.

## Technical Design

### Architecture

```
CLI (--resume, --strict)
    │
    ▼
OrchestratorConfig
    │ prioritize_wip: bool (existing)
    │ strict_resume: bool (new)
    ▼
MalaOrchestrator
    │ On issue start: lookup_prior_session(repo_path, issue_id) -> session_id | None
    ▼
AgentSessionInput
    │ resume_session_id: str | None (new field)
    ▼
AgentSessionRunner._initialize_session()
    │ If resume_session_id provided: sdk_client_factory.with_resume(options, session_id)
    ▼
Claude SDK (resume=session_id)
```

**Session Lookup Flow:**
1. `get_repo_runs_dir(repo_path)` → `~/.mala/runs/{repo}/`
2. Glob `*.json`, load each and extract `started_at` timestamp
3. Sort by parsed `started_at` descending (most recent first); fallback to filename if parsing fails
4. For each file: check `issues[issue_id].session_id`
5. Return first non-None session_id, or None if not found

**Stale Session Handling (Orchestrator-level):**
The orchestrator wraps the session run in a try/catch. If the SDK raises an error indicating the session cannot be resumed (e.g., `SessionNotFoundError`, `InvalidSessionError`, or similar SDK-specific exceptions):
- **Lenient mode (default)**: Log warning, clear `resume_session_id`, retry with fresh session
- **Strict mode**: Mark issue as failed with message "Session resumption failed: {error}", continue to next issue

This keeps the runner simple (it just uses `with_resume()`) while the orchestrator handles policy decisions.

### Data Model
- No new data models needed
- Reuses existing `RunMetadata`, `IssueRun.session_id` field
- Add helper function: `lookup_prior_session(repo_path: Path, issue_id: str) -> str | None`

### API/Interface Design

**New config field (src/orchestration/types.py - OrchestratorConfig):**
```python
strict_resume: bool = False  # Fail issue if no prior session (with --resume)

# Note: session resumption is implicit when prioritize_wip=True (--resume flag)
```

**New CLI flag (src/cli/cli.py):**
```python
strict: Annotated[
    bool,
    typer.Option(
        "--strict",
        help="With --resume: fail issues that have no prior session to resume",
        rich_help_panel="Scope & Ordering",
    ),
] = False,
```

**CLI validation:** If `--strict` is passed without `--resume`, emit error and exit:
```python
if strict and not resume:
    raise typer.BadParameter("--strict requires --resume flag")
```

**Updated CLI help text for --resume:**
```python
help="Prioritize in_progress issues AND attempt to resume their prior Claude sessions"
```

**New helper function (src/infra/io/log_output/run_metadata.py):**
```python
def lookup_prior_session(repo_path: Path, issue_id: str) -> str | None:
    """Find the most recent session_id for an issue from prior runs.

    Scans run metadata files in reverse chronological order.
    Returns the first matching session_id for the given issue_id, or None if not found.

    Sorting strategy (robust ordering):
    1. Load each metadata file and parse `started_at` timestamp
    2. Sort by parsed timestamp descending (most recent first)
    3. Fall back to filename sort only if timestamp parsing fails

    Handles:
    - Missing run directory (returns None)
    - Corrupt/invalid JSON files (skips, continues scanning)
    - Issues without session_id (skips, continues scanning)
    """
```

**AgentSessionInput extension (src/pipeline/agent_session_runner.py):**
```python
@dataclass
class AgentSessionInput:
    issue_id: str
    prompt: str
    baseline_commit: str | None = None
    issue_description: str | None = None
    agent_id: str | None = None
    resume_session_id: str | None = None  # Session to resume (if any) - NEW
```

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/cli/cli.py` | Exists | Add `--strict` flag, pass to OrchestratorConfig |
| `src/orchestration/types.py` | Exists | Add `strict_resume: bool` field to OrchestratorConfig |
| `src/infra/io/log_output/run_metadata.py` | Exists | Add `lookup_prior_session()` helper function |
| `src/orchestration/orchestrator.py` | Exists | Call `lookup_prior_session()` before running issue, pass to AgentSessionInput |
| `src/pipeline/agent_session_runner.py` | Exists | Add `resume_session_id` field to `AgentSessionInput`, use in `_initialize_session()` |
| `tests/unit/infra/io/log_output/test_run_metadata.py` | Exists | Add tests for `lookup_prior_session()` (sorting, matching, edge cases) |
| `tests/unit/pipeline/test_agent_session_runner.py` | Exists | Add tests for resume session handling |

## Risks, Edge Cases & Breaking Changes

### Edge Cases & Failure Modes
- **No prior runs for repo**: `lookup_prior_session()` returns `None`. Lenient: creates new session. Strict: fails issue.
- **Prior run exists but issue not in it**: Continue scanning older runs. If none found, same as "no prior runs".
- **Prior run has issue but no session_id**: Skip that run, continue scanning older runs.
- **Corrupt metadata files**: `lookup_prior_session()` should gracefully handle JSON errors (log warning, skip file, continue).
- **Stale/expired session**: SDK may reject old session_id with an error.
  - **Error detection**: Catch SDK exceptions that indicate unresumable sessions (e.g., `SessionNotFoundError`, `InvalidSessionError`, HTTP 404/410 responses). Transient errors (rate limits, network timeouts, auth failures) should NOT trigger fallback/failure—they should propagate normally.
  - **Error handling location**: Orchestrator level (wraps the session run call), not inside the runner.
  - **Lenient mode**: Catch resumption-specific error, log warning, clear `resume_session_id`, retry with fresh session.
  - **Strict mode**: Catch resumption-specific error, mark issue as failed with message "Session resumption failed: {error}", continue to next issue.
- **Multiple runs with same issue**: Use most recent run that contains the issue (by file timestamp).

### Breaking Changes & Compatibility
- **Behavior change for `--resume`**: This flag now also attempts session resumption (previously only prioritized WIP). This is additive—WIP prioritization still works. Document in CLI help text and release notes.
- `--strict` is a new flag with no prior behavior to break.
- Performance consideration: Scanning JSON files could be slow with many runs. Mitigated by:
  - Stopping at first match (most common case: recent session exists)
  - Sorting by timestamp (newest first)
  - Most repos won't have thousands of run files
  - **Optional optimization**: Cache session lookups per-run (keyed by `issue_id`) to avoid repeated scans when multiple WIP issues exist. Implementation: `_session_cache: dict[str, str | None]` in orchestrator, populated on first lookup.

## Testing & Validation Strategy

- **Unit Tests (`tests/unit/infra/io/log_output/test_run_metadata.py`)**
  - `test_lookup_finds_recent_session`: Setup mock metadata files, verify most recent is returned
  - `test_lookup_returns_none_if_missing`: Verify `None` when no file matches the issue
  - `test_lookup_ignores_runs_without_session_id`: Verify skips entries with null session_id
  - `test_lookup_handles_corrupt_json`: Verify graceful handling of malformed files
  - `test_lookup_sorts_by_started_at_timestamp`: Verify sorting uses `started_at` field, not filename
  - `test_lookup_fallback_to_filename_sort`: Verify fallback when `started_at` parsing fails

- **Unit Tests (`tests/unit/pipeline/test_agent_session_runner.py`)**
  - `test_resume_session_id_applied`: Verify `with_resume()` called when `resume_session_id` is set
  - `test_resume_session_id_none_skipped`: Verify normal flow when `resume_session_id` is None

- **Integration Tests**
  - Mock `RunMetadata` scanning in orchestrator tests
  - Verify `strict=True` fails issue when mock lookup returns `None`
  - Verify `strict=True` fails issue when SDK rejects stale session
  - Verify `--strict` without `--resume` raises `BadParameter` error
  - Verify stale session error in lenient mode triggers fallback (not failure)

- **Manual Verification**
  - Run an issue: `mala run ...`
  - Keep issue in WIP state (e.g., gate failure or ctrl-c)
  - Run `mala run --resume --verbose ...` and check logs for "Resuming session..."
  - Run `mala run --resume --strict ...` on a fresh issue and verify it fails with clear error

### Acceptance Criteria Coverage
| Spec AC | Covered By |
|---------|------------|
| AC #1: --resume attempts session resume for WIP issues | Orchestrator logic calling `lookup_prior_session()` |
| AC #2: Most recent session_id used | `lookup_prior_session()` unit tests (sorting by `started_at` timestamp) |
| AC #3: Default lenient behavior (create new if not found OR resumption fails) | Orchestrator fallback logic: if session_id is None or SDK rejects, proceed with fresh session |
| AC #4: --strict fails that issue only if no prior session found OR session resumption fails | Orchestrator conditional check, marks issue failed, continues run |
| AC #5: Verbose-only logging | Logging in orchestrator/runner gated by verbose flag |
| AC #6: --strict requires --resume | CLI validation raises error if --strict without --resume |

## Open Questions
- (Resolved) Flag naming: Extend `--resume`, add `--strict` — decided via interview
- (Resolved) Multiple runs: Most recent first by filename timestamp — decided via interview
- (Resolved) Stale sessions: Controlled by `--strict` flag — decided via interview
- (Resolved) Fail scope: Fail that issue only, not entire run — decided via interview
- (Resolved) WIP identification: Uses existing `status: in_progress` check from `prioritize_wip` logic
- (Resolved) Session lookup scope: Same repo only (via `get_repo_runs_dir(repo_path)`)

## Architectural Note: Session Discovery

The `cli/logs.py` module already has session discovery logic (`_discover_run_files`, `_extract_sessions`, `_sort_sessions`). The initial implementation (T001-T004) will create a parallel `lookup_prior_session()` in `run_metadata.py`.

**Follow-up refactor (mala-twoa.5)**: After initial implementation, extract shared session discovery to `run_metadata.py` so both CLI and orchestrator use the same code. This maintains proper layering (cli → infra, orchestration → infra) and eliminates duplication.

## Next Steps
After this plan is approved, run `/create-tasks` to generate:
- `--beads` → Beads issues with dependencies for multi-agent execution
- (default) → TODO.md checklist for simpler tracking

# Implementation Plan: Resume with Final Review Feedback

## Context & Goals

- **Spec**: N/A — derived from user description
- **Problem**: When an agent hits max review iterations and fails, resuming with `--resume` doesn't provide the final reviewer feedback to the agent. The agent continues without knowing what issues caused the failure.
- **Root Cause**: On max retry exceeded, session ends with `COMPLETE_FAILURE` but no `pending_query` is sent to agent. The review issues exist in `ctx.last_review_result` but aren't injected into the resumed session.
- **Goal**: Ensure resumed agents receive the review feedback that caused the previous failure

## Scope & Non-Goals

### In Scope
- **Option A**: Persist full review issues in run metadata
- On resume, use `review_followup` template as initial prompt (same format as normal retries)
- All P0/P1 review issues included in the prompt
- Backward compatible metadata schema changes

### Out of Scope (Non-Goals)
- Changing the max retry limit behavior itself
- Modifying how reviewers generate feedback
- Changing the review_followup prompt template content
- Gate failure resume handling (only review failures)
- Resume behavior for non-review failures (e.g., gate failures, timeouts)
- Feature flag gating (ship always-on)
- Remote/CI metadata backends (only local `~/.mala/runs/` supported)

## Assumptions & Constraints

### Implementation Constraints
- SDK session resume continues to work as-is; we modify the initial prompt only
- Metadata schema change must handle old records without `last_review_issues`
- Must not break existing non-resume flows (fresh starts)
- Uses existing `review_followup` template and `format_review_issues()` formatter
- P2+ issues are excluded from resume prompt (only P0/P1 stored)

### Metadata Compatibility Contract
The run metadata JSON format follows these rules for forward/backward compatibility:
- **Unknown keys ignored**: The codebase uses manual dict parsing with `.get()` calls (not dacite), which naturally ignores unknown keys
- **Missing keys defaulted**: All new fields have `None` defaults; manual parsing uses `dict.get("key")` which returns `None` for missing keys
- **No schema version field**: We rely on tolerant parsing rather than explicit versioning

**Cross-version behavior:**
- Older mala reading new metadata: Unknown `last_review_issues` key is silently ignored by `.get()` calls
- New mala reading old metadata: Missing key defaults to `None`, uses normal `implementer_prompt`
- No breakage in either direction; graceful degradation to current behavior

**Note**: The codebase uses manual dict access (see `extract_session_from_run` at line ~912-956 and `_from_dict` at line ~430-445), not dacite.

### Testing Constraints
- Unit tests for metadata serialization/deserialization (including cross-version)
- Unit test that loads JSON blob with new key using old parsing path to verify no crash
- Unit tests for prompt building logic
- Unit test asserting P2+ issues are excluded from storage
- Integration test for full resume-with-review-context flow
- Existing tests must pass

## Prerequisites

- [x] Research complete: understood lifecycle, metadata, prompt structures
- [x] User decisions: Option A, all P0/P1 issues, same template, replace prompt
- [x] `format_review_issues` utility exists in `src/pipeline/review_formatter.py`

## High-Level Approach

**Selected: Option A - Persist Final Review Issues in Metadata**

When an agent session fails due to max review iterations:
1. Extract P0/P1 issues from `lifecycle_ctx.last_review_result` in `AgentSessionRunner._build_session_output()` (single filtering point)
2. Flow issues through `AgentSessionOutput` → `IssueResult` → `IssueRun` for persistence

On resume via `--prioritize-wip`:
3. Detect prior session had review failure (check metadata for stored issues)
4. Build prompt using `review_followup` template with stored issues (attempt=1, fresh start)
5. Use this as the initial prompt instead of `implementer_prompt`
6. SDK resumes the session; agent sees review feedback and continues fixing

## Technical Design

### Architecture

```
Session 1 (fails at max review iterations):
┌─────────────────────────────────────────────────────────────────┐
│ Review runs → finds issues → ctx.last_review_result populated  │
│ can_retry=False (hit max) → COMPLETE_FAILURE                   │
│ AgentSessionRunner._build_session_output():                    │
│   - Filter to P0/P1 only (single choke point)                  │
│   - Convert to dict format matching ReviewIssueProtocol        │
│   - Set AgentSessionOutput.last_review_issues                  │
│ Orchestrator: Pass issues to IssueResult                       │
│ IssueFinalizer: Persist issues in IssueRun metadata            │
│ Session ends                                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
Session 2 (resume with --prioritize-wip):
┌─────────────────────────────────────────────────────────────────┐
│ lookup_prior_session_info() → finds prior session metadata     │
│ SessionInfo includes last_review_issues from metadata          │
│ _build_resume_prompt():                                        │
│   - Convert dicts to StoredReviewIssue typed adapter           │
│   - Format with format_review_issues()                         │
│   - Log info: "Using review_followup for resume (N issues)"    │
│ Use review_followup prompt AS initial prompt                   │
│ SDK resumes session → agent sees review feedback               │
│ Agent addresses issues → review runs again (attempt=1)         │
└─────────────────────────────────────────────────────────────────┘
```

**Data Flow**:
```
lifecycle_ctx.last_review_result
    → [AgentSessionRunner: filter P0/P1, convert to dict]
    → AgentSessionOutput.last_review_issues
    → IssueResult.last_review_issues
    → IssueRun.last_review_issues (persisted as JSON)
    → SessionInfo.last_review_issues (on resume lookup)
    → [_build_resume_prompt: convert to StoredReviewIssue]
    → format_review_issues()
    → review_followup template
```

**P0/P1 Filtering Location**: The single source of truth for filtering is `AgentSessionRunner._build_session_output()`. This ensures consistent filtering regardless of how the data flows to persistence.

### SDK Resume Prompt Interaction

**How resume prompts work:**
- SDK resume restores the conversation history from the prior session
- The `prompt` parameter passed to `sdk_client.run()` becomes a **new user message** appended to the restored history
- On resume with review issues: `review_followup` becomes this new user message
- On resume without issues: `implementer_prompt` becomes this new user message (unchanged behavior)

**No duplicate prompts:** The resumed session's history already contains prior exchanges. The new prompt (whether `review_followup` or `implementer_prompt`) is a continuation message, not a replacement of history. The agent sees the review feedback as the latest instruction to act on.

### Data Model

**Add typed adapter for stored review issues** (`src/orchestration/orchestrator.py`):

```python
@dataclass(frozen=True)
class StoredReviewIssue:
    """Typed adapter for review issues loaded from metadata.

    Matches ReviewIssueProtocol fields exactly for format_review_issues().
    Provides validation and explicit defaults for missing/extra keys.

    Default values are safe for format_review_issues():
    - file="unknown": displayed as-is (no path normalization issues)
    - line_start=0, line_end=0: formatter handles gracefully
    - body="": formatter skips empty bodies
    - reviewer="unknown": displayed in brackets
    """
    file: str
    line_start: int
    line_end: int
    priority: int | None
    title: str
    body: str
    reviewer: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StoredReviewIssue:
        """Create from dict with validation and defaults.

        Never raises for missing keys; provides safe defaults.
        Coerces types where needed (strings to ints for line numbers).
        """
        # Coerce line numbers to int (handle string values from JSON)
        def to_int(val: Any, default: int) -> int:
            if val is None:
                return default
            try:
                return int(val)
            except (ValueError, TypeError):
                return default

        line_start = to_int(d.get("line_start"), 0)
        return cls(
            file=str(d.get("file", "unknown")),
            line_start=line_start,
            line_end=to_int(d.get("line_end"), line_start),  # Default to line_start
            priority=d.get("priority"),  # None is valid
            title=str(d.get("title", "Unknown issue")),
            body=str(d.get("body", "")),
            reviewer=str(d.get("reviewer", "unknown")),
        )
```

**Add to `AgentSessionOutput`** (`src/pipeline/agent_session_runner.py`):

```python
@dataclass
class AgentSessionOutput:
    # ... existing fields ...

    # NEW: P0/P1 review issues from last failed review (for resume)
    # Stored as list of dicts matching ReviewIssueProtocol fields
    last_review_issues: list[dict[str, Any]] | None = None
```

**Add to `IssueResult`** (`src/pipeline/issue_result.py`):

```python
@dataclass
class IssueResult:
    # ... existing fields ...

    # NEW: P0/P1 review issues from last failed review (for resume)
    last_review_issues: list[dict[str, Any]] | None = None
```

**Add to `IssueRun` dataclass** (`src/infra/io/log_output/run_metadata.py`):

```python
@dataclass
class IssueRun:
    # ... existing fields ...

    # NEW: P0/P1 review issues from last failed review (for resume)
    # Stored as list of dicts for JSON serialization
    last_review_issues: list[dict[str, Any]] | None = None
```

**Review issue dict structure** (matches `ReviewIssueProtocol` fields exactly):
```python
{
    "file": str,           # File path where issue was found
    "line_start": int,     # Starting line number
    "line_end": int,       # Ending line number (required, not optional)
    "priority": int | None,  # 0=P0, 1=P1 (only P0/P1 stored)
    "title": str,          # Issue title
    "body": str,           # Issue description (NOT "description")
    "reviewer": str,       # Which reviewer found this issue
}
```

**Add to `SessionInfo`** (`src/infra/io/log_output/run_metadata.py`):
```python
@dataclass
class SessionInfo:
    # ... existing fields ...

    # NEW: P0/P1 review issues from failed session (for resume prompt)
    last_review_issues: list[dict[str, Any]] | None = None
```

### API/Interface Design

**New imports in `src/orchestration/orchestrator.py`**:

```python
from src.domain.prompts import build_custom_commands_section
from src.pipeline.review_formatter import format_review_issues
```

**New function in `src/orchestration/orchestrator.py`**:

```python
def _build_resume_prompt(
    prior_session: SessionInfo,
    prompts: PromptProvider,
    validation_commands: PromptValidationCommands,
    issue_id: str,
    max_review_retries: int,
    repo_path: Path,
) -> str | None:
    """Build resume prompt if prior session failed with review issues.

    Returns review_followup prompt if issues exist, None otherwise.
    When resuming, we treat it as attempt 1 (fresh retry counter).
    """
    if not prior_session.last_review_issues:
        return None

    # Convert dicts to typed adapter for format_review_issues()
    issues = [
        StoredReviewIssue.from_dict(d)
        for d in prior_session.last_review_issues
    ]

    # Log for observability
    logger.info(
        "Using review_followup for resume: %d issues from prior run %s",
        len(issues),
        prior_session.run_id,
    )

    review_issues_text = format_review_issues(issues, base_path=repo_path)

    # Use review_followup template (attempt=1 for fresh session start)
    return prompts.review_followup_prompt.format(
        attempt=1,
        max_attempts=max_review_retries,
        review_issues=review_issues_text,
        issue_id=issue_id,
        lint_command=validation_commands.lint,
        format_command=validation_commands.format,
        typecheck_command=validation_commands.typecheck,
        test_command=validation_commands.test,
        custom_commands_section=build_custom_commands_section(
            validation_commands.custom_commands
        ),
    )
```

**Modify `run_implementer` in orchestrator.py**:

```python
# After getting prior_session from lookup_prior_session_info()
if prior_session:
    resume_prompt = _build_resume_prompt(
        prior_session, self._prompts, self._prompt_validation_commands,
        issue_id, self.config.max_review_retries, self._repo_path
    )
    if resume_prompt:
        prompt = resume_prompt  # Use instead of implementer_prompt
    # else: use normal implementer_prompt (no review issues)
```

**Modify `_build_session_output` in `AgentSessionRunner`**:

```python
def _build_session_output(self, ...) -> AgentSessionOutput:
    # ... existing code ...

    # Extract P0/P1 issues for resume (single filtering point)
    # Only populate when:
    #   1. Session failed (not success)
    #   2. Review ran (last_review_result exists)
    #   3. Review found blocking issues (P0/P1)
    # This condition ensures we only store issues for review-specific failures,
    # not gate failures or timeouts that happen to have a stale last_review_result.
    last_review_issues = None
    if (
        not state.lifecycle_ctx.success
        and state.lifecycle_ctx.last_review_result is not None
        and state.lifecycle_ctx.retry_state.review_attempt > 0  # Review actually ran
    ):
        blocking_issues = [
            issue for issue in state.lifecycle_ctx.last_review_result.issues
            if issue.priority is not None and issue.priority <= 1
        ]
        if blocking_issues:
            last_review_issues = [
                {
                    "file": issue.file,
                    "line_start": issue.line_start,
                    "line_end": issue.line_end,
                    "priority": issue.priority,
                    "title": issue.title,
                    "body": issue.body,
                    "reviewer": issue.reviewer,
                }
                for issue in blocking_issues
            ]

    return AgentSessionOutput(
        # ... existing fields ...
        last_review_issues=last_review_issues,
    )
```

**Review failure condition**: The predicate `retry_state.review_attempt > 0` ensures we only store issues when the review phase actually ran. This distinguishes review failures from gate failures or timeouts that might have stale `last_review_result` from a previous lifecycle iteration.

**Modify IssueResult construction in `orchestrator.py:807-821`**:

```python
# In _create_issue_result() or equivalent
return IssueResult(
    # ... existing fields ...
    last_review_issues=output.last_review_issues,  # NEW: pass through from AgentSessionOutput
)
```

**IssueFinalizer persistence path**: `IssueFinalizer` reads `IssueResult.last_review_issues` (passed from orchestrator) and writes to `IssueRun.last_review_issues` when recording the final session result. This only happens once per issue completion (not intermediate attempts).

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/infra/io/log_output/run_metadata.py` | Exists | Add `last_review_issues` to `IssueRun` and `SessionInfo`; update `_from_dict()` (~line 430) and `extract_session_from_run()` (~line 912) to populate new field |
| `src/orchestration/orchestrator.py` | Exists | Add `StoredReviewIssue`, `_build_resume_prompt()`; add imports; modify `run_implementer()`; update IssueResult construction (~line 807-821) to pass `last_review_issues=output.last_review_issues` |
| `src/pipeline/issue_finalizer.py` | Exists | Read `IssueResult.last_review_issues` and write to `IssueRun.last_review_issues` when recording final session |
| `src/pipeline/agent_session_runner.py` | Exists | Add `last_review_issues` to `AgentSessionOutput`; P0/P1 filtering in `_build_session_output()` with review_attempt > 0 check |
| `src/pipeline/issue_result.py` | Exists | Add `last_review_issues` field to pass through data flow |
| `src/domain/lifecycle.py` | Exists | No changes needed (already has `last_review_result`) |
| `src/pipeline/review_formatter.py` | Exists | (Read-only) Tolerant of unknown paths and line numbers; defaults in StoredReviewIssue are safe |
| `tests/unit/infra/test_run_metadata.py` | Exists | Add tests for new field serialization and cross-version compat |
| `tests/unit/orchestration/test_orchestrator.py` | Exists | Add tests for `StoredReviewIssue` and resume prompt building |
| `tests/unit/pipeline/test_agent_session_runner.py` | Exists | Add tests for P0/P1 filtering and review_attempt > 0 condition |
| `tests/unit/pipeline/test_issue_result.py` | Exists | Add tests for new field |

## Risks, Edge Cases & Breaking Changes

### Backward Compatibility
- **Old metadata without `last_review_issues`**: Field defaults to `None`, resume uses normal `implementer_prompt` (current behavior preserved)
- **New code reading old metadata**: Manual `.get()` parsing handles missing field gracefully with `None` default
- **Old code reading new metadata**: Unknown `last_review_issues` key is silently ignored by `.get()` calls
- **Exact unchanged behavior for old metadata**: No fallback attempts to SDK/logs; simply uses original prompt

### Edge Cases
- **Multiple consecutive resumes**: Each resume gets the same stored review issues. After successful fix, new session won't have issues stored (success path clears).
- **Review issues change between runs**: Unlikely since we store from the exact failed review. The agent still has the specific issues from its failure.
- **Session fails for non-review reasons**: `last_review_issues` will be `None` because `review_attempt > 0` check prevents storage; falls back to `implementer_prompt`
- **Non-review failure with stale last_review_result**: If a session ran review successfully but then timed out, `last_review_result` exists but `review_attempt > 0` and success status will correctly exclude storage (only review-caused failures store issues)
- **Stale SDK session**: If SDK session is stale and we retry fresh, the review_followup prompt still works (doesn't depend on session history)
- **Empty issue list after filtering**: If `last_review_result` exists but has no P0/P1 issues (all were P2+), treat as no issues present and use `implementer_prompt`.
- **Resume after resume**: Retry counter resets to attempt=1 on each resume; issues do not accumulate across multiple resume cycles.
- **Missing dict fields**: `StoredReviewIssue.from_dict()` provides safe defaults and type coercion for any missing/malformed keys

### Breaking Changes
- **None**: Schema additions are backward compatible; behavior only changes when resuming failed-review sessions

## Testing & Validation Strategy

### Unit Tests
1. **Metadata serialization**: `IssueRun` with `last_review_issues` serializes/deserializes correctly
2. **Cross-version compat**: Load JSON blob with new key using tolerant parsing; verify no crash
3. **Cross-version compat**: Load JSON blob without new key; verify `None` default
4. **SessionInfo population**: `lookup_prior_session_info` includes `last_review_issues` from metadata
5. **Resume prompt building**: `_build_resume_prompt` formats correctly with sample issues
6. **StoredReviewIssue adapter**: Test `from_dict()` with missing/extra keys, verify defaults
7. **P0/P1 filtering**: `_build_session_output` excludes P2+ issues; test with mixed priority list
8. **Data flow**: `AgentSessionOutput` and `IssueResult` correctly pass issues through

### Integration Tests
1. **Full resume flow**: Session fails at max review → metadata saved with issues → resume → assert the resumed run adds exactly one new user message to restored history and that its content matches `review_followup` template (not `implementer_prompt`)
2. **Fresh start unchanged**: New session without prior failure uses `implementer_prompt` as the initial user message

### Manual Validation
1. Run agent on issue that will fail review
2. Let it hit max iterations
3. Verify log message: "Using review_followup for resume: N issues from prior run X"
4. Resume with `--prioritize-wip`
5. Verify agent sees review issues and attempts to fix them

### Acceptance Criteria Coverage

| Criterion | Approach |
|-----------|----------|
| Resumed agent sees prior review issues | review_followup prompt with stored issues |
| Works with SDK resume | Prompt is new user message appended to restored history |
| Works with stale session fallback | Review issues in metadata, not session dependent |
| Backward compatible | Optional field with None default; tolerant parsing |
| Only review failures affected | Only populate `last_review_issues` for review failures |
| P0/P1 only | Single filtering point in `_build_session_output()` |
| Observable | Info-level log when resume uses review prompt |

## Open Questions

*Resolved during synthesis and review:*
1. ~~Which approach?~~ → Option A (persist in metadata)
2. ~~How much context?~~ → All P0/P1 issues only
3. ~~Resume prompt format?~~ → Same as review_followup template
4. ~~Explicit resume mention?~~ → No, use standard template
5. ~~Reset retry counters?~~ → Yes, fresh session starts at attempt=1
6. ~~Accumulate issues across retries?~~ → No, each resume uses only the stored issues from prior failure
7. ~~Feature flag?~~ → No, ship always-on
8. ~~Backend support?~~ → Only local `~/.mala/runs/`
9. ~~Old metadata behavior?~~ → Exact unchanged (None default, no fallback)
10. ~~Cross-version compat?~~ → Tolerant parsing (ignore unknown keys, default missing keys)
11. ~~P0/P1 filtering location?~~ → `AgentSessionRunner._build_session_output()` (single choke point)
12. ~~SDK resume interaction?~~ → Review prompt is new user message appended to history

*Remaining:*
- None

## Next Steps

After this plan is approved, run `/create-tasks` to generate:
- `--beads` → Beads issues with dependencies for multi-agent execution
- (default) → TODO.md checklist for simpler tracking

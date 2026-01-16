# Implementation Plan: Remove Context Pressure Tracking

## Context & Goals
- **Spec**: N/A — derived from user description
- **Problem**: SDK reports incorrect token usage values when prompt caching is enabled, making manual context pressure calculation unreliable
- Remove all manual context pressure tracking code
- Rely on SDK's built-in auto-compaction (happens automatically when context fills up)
- Add PreCompact hook to archive transcript before compaction
- **Who this helps**: All mala users running long sessions; removes unreliable manual tracking in favor of SDK-native compaction

## Scope & Non-Goals
- **In Scope**
  - Delete ContextPressureHandler and ContextPressureError entirely
  - Remove manual token tracking from MessageStreamProcessor
  - Add PreCompact hook that archives full transcript before SDK compaction
  - Update AgentRuntimeBuilder to register PreCompact hook
  - Remove config options: `context_restart_threshold`, `context_limit` (breaking change)
- **Out of Scope (Non-Goals)**
  - Custom checkpoint extraction (SDK's compaction summary handles this)
  - Changing how SDK compaction works
  - Manual compaction triggering
  - User-visible compaction notifications (future enhancement; just log for now)
  - Feature flags for staged rollout (immediate behavior change acceptable)

## Assumptions & Constraints
- SDK auto-compacts automatically when context window fills (no explicit enable needed)
- SDK compaction keeps session alive and summarizes older messages in-place
- This is different from our current "restart fresh session with checkpoint" approach
- PreCompact hook fires with `trigger: "auto"` or `trigger: "manual"` when SDK compacts
- Archive both auto and manual compaction triggers (treat identically)

### Implementation Constraints
- No way to inject additional context into compaction summary from PreCompact hook
- PreCompact hook can only archive, log, or return a systemMessage
- **SDK contract**: PreCompact hook available in claude-agent-sdk (see [hooks documentation](https://platform.claude.com/docs/en/agent-sdk/hooks))
  - Input fields: `session_id`, `transcript_path`, `trigger`, `cwd`, `custom_instructions`
  - Hook key must be `"PreCompact"` (PascalCase, matching `PreToolUse`/`PostToolUse`/`Stop` convention)
- PreCompact hook signature: `async def hook(input_data, tool_use_id, context) -> dict` (matches existing hooks)
- Hook failures must not block compaction—log errors and continue
- **Defensive parsing**: Treat `hook_input` as opaque dict; tolerate missing keys gracefully

### Testing Constraints
- Testing auto-compaction requires very long conversations or mock SDK
- Integration tests should verify PreCompact hook is registered and called
- Unit tests should verify archive file creation and error handling

## Integration Analysis

### Existing Mechanisms Considered

| Existing Mechanism | Could Serve Feature? | Decision | Rationale |
|--------------------|---------------------|----------|-----------|
| SDK PreCompact hook | Yes | Extend | SDK has built-in hook system for this exact purpose |
| AgentRuntimeBuilder | Yes | Extend | Already registers PreToolUse/PostToolUse/Stop hooks |
| SDK auto-compaction | Yes | Use as-is | SDK handles triggering automatically |

### Integration Approach
Extend existing hook infrastructure in AgentRuntimeBuilder to add PreCompact hook. The hook archives the full transcript (via `transcript_path` in hook input) before SDK compacts it. SDK handles all triggering and session continuation—no manual intervention needed.

### Transcript Path Source
The `transcript_path` field in PreCompact hook input is provided by the SDK, pointing to the SDK's internal transcript file. This is the authoritative source:
- **Who writes it**: SDK writes transcript to its internal storage location
- **When flushed**: SDK flushes before firing PreCompact hook (transcript is complete at hook time)
- **Stability**: Path is valid only during hook execution; archive immediately
- **Format**: SDK-defined (may be `.md`, `.jsonl`, or other); archive preserves original extension

If `transcript_path` is missing or file doesn't exist (SDK version mismatch, race condition), the hook logs a warning and skips archiving—compaction proceeds unblocked. This is best-effort archiving.

### Repo Identity & Archive Path
The archive directory uses `get_repo_runs_dir(repo_path)` which derives a unique key from the repo path:
- **Keying scheme**: Repo path with `/` replaced by `-` (e.g., `/home/cyou/mala` → `-home-cyou-mala`)
- **Full archive path**: `~/.config/mala/runs/{repo-key}/archives/{session_id}_{timestamp}_transcript{ext}`
- **Collision prevention**: Timestamp ensures uniqueness; path-based keying prevents cross-repo collisions
- **Unit test**: Add test for `get_repo_runs_dir()` path derivation to prevent regressions

## Prerequisites
- [x] Confirm current SDK version supports PreCompact hook (confirmed in spec context)
- [x] Identify where transcript archives should be stored → `~/.config/mala/runs/{repo}/archives/`
- [ ] **Verify SDK integration in-repo**: Before implementing, inspect `src/infra/agent_runtime.py` and `src/infra/sdk_adapter.py` to confirm:
  1. How hook events are dispatched to callbacks
  2. The exact input dict shape passed to PreCompact hooks
  3. Whether mala's adapter transforms or wraps SDK hook inputs
  - **Implementation rule**: Use the *exact* callable signature used by the adapter's hook dispatcher. Do not assume `tool_use_id` or `stderr` fields exist for PreCompact (lifecycle hooks don't have tool context). Base tests on the adapter-dispatched callable shape, not external SDK docs alone.
- [ ] No external dependencies or approvals required

## High-Level Approach

1. **Create PreCompact hook** that archives transcript before compaction
2. **Register hook** in AgentRuntimeBuilder alongside existing hooks
3. **Delete** ContextPressureHandler, ContextPressureError, and all manual tracking
4. **Remove config options** context_restart_threshold and context_limit (with user-friendly error message)
5. **Update tests** to remove pressure tracking tests, add PreCompact hook tests
6. **Update docs** (`docs/project-config.md`) to reflect removed config fields

## Technical Design

### Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    SDK Auto-Compaction                       │
│  (triggers when context window fills, no explicit enable)   │
└─────────────────────┬───────────────────────────────────────┘
                      │ fires
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   PreCompact Hook                            │
│  Input: session_id, transcript_path, trigger='auto|manual'  │
│  Action: Archive transcript to ~/.config/mala/runs/{repo}/  │
│          archives/{session_id}_{timestamp}_transcript{ext}  │
│          (preserves original extension from transcript_path)│
│  Output: {} (allow compaction to proceed)                   │
└─────────────────────────────────────────────────────────────┘
                      │ returns
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                SDK Compaction                                │
│  - Summarizes older messages                                │
│  - Keeps session alive                                      │
│  - Continues conversation with summarized context           │
└─────────────────────────────────────────────────────────────┘
```

### Data Model
N/A - no new data models. ContextPressureError removed.

### API/Interface Design

**PreCompact hook callback** (in `src/infra/hooks/precompact.py`):
```python
from pathlib import Path
import shutil
import logging
from typing import Any

from src.infra.tools.env import get_repo_runs_dir

logger = logging.getLogger(__name__)


def make_precompact_hook(repo_path: Path) -> Callable:
    """Create a PreCompact hook that archives transcripts before compaction.

    NOTE: Actual signature determined by adapter inspection (see prerequisites).
    PreCompact is a lifecycle hook - likely no tool_use_id parameter.
    """

    async def precompact_hook(
        hook_input: Any,  # SDK type - {"session_id", "transcript_path", "trigger", ...}
        *args: Any,  # Accept additional args to match adapter signature
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Archive transcript before SDK compacts the conversation."""
        session_id = hook_input.get("session_id", "unknown")
        transcript_path = hook_input.get("transcript_path")
        trigger = hook_input.get("trigger", "unknown")  # "auto" or "manual"

        if not transcript_path:
            logger.warning(f"PreCompact: no transcript_path provided for {session_id}")
            return {}

        source = Path(transcript_path)
        if not source.exists():
            logger.warning(f"PreCompact: transcript not found at {transcript_path}")
            return {}

        try:
            # Archive to repo-specific runs directory (consistent with other run data)
            archive_dir = get_repo_runs_dir(repo_path) / "archives"
            archive_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Preserve original extension (SDK may use .md, .jsonl, etc.)
            ext = source.suffix or ".txt"
            dest = archive_dir / f"{session_id}_{timestamp}_transcript{ext}"
            shutil.copy2(source, dest)
            dest.chmod(0o600)  # Owner read/write only - may contain sensitive data
            archive_dir.chmod(0o700)  # Ensure directory permissions regardless of umask

            # Log archive size for observability (helps detect runaway disk usage)
            file_size_kb = dest.stat().st_size / 1024
            logger.info(f"PreCompact: archived transcript ({trigger=}) to {dest} ({file_size_kb:.1f} KB)")
        except OSError as e:
            # Don't block compaction on archive failure
            logger.error(f"PreCompact: failed to archive transcript: {e}")

        return {}  # Allow compaction to proceed

    return precompact_hook
```

**Hook registration** (in `src/infra/agent_runtime.py`):
```python
from src.infra.hooks.precompact import make_precompact_hook

# In AgentRuntimeBuilder.build():
hooks = {
    "PreToolUse": [...],
    "PostToolUse": [...],
    "Stop": [...],
    "PreCompact": [HookMatcher(hooks=[make_precompact_hook(repo_path)])],  # NEW
}
```

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/pipeline/context_pressure_handler.py` | **Delete** | Remove entire file |
| `tests/unit/pipeline/test_context_pressure_handler.py` | **Delete** | Remove entire file |
| `src/pipeline/message_stream_processor.py` | Modify | Remove `_check_context_pressure()`, context_usage tracking |
| `src/pipeline/agent_session_runner.py` | Modify | Remove ContextPressureError handling, import cleanup |
| `src/infra/agent_runtime.py` | Modify | Add PreCompact hook registration |
| `src/domain/validation/config.py` | Modify | Remove `context_restart_threshold`, `context_limit` fields |
| `tests/conftest.py` | Modify | Remove `context_restart_threshold`, `context_limit` from fixtures |
| `tests/integration/pipeline/test_agent_session_runner.py` | Modify | Remove context restart tests |
| `tests/unit/infra/test_config_merger.py` | Modify | Remove context threshold tests |
| `src/infra/hooks/precompact.py` | **New** | PreCompact hook implementation |
| `tests/unit/infra/hooks/test_precompact.py` | **New** | Unit tests for PreCompact hook |
| `docs/project-config.md` | Modify | Remove `context_restart_threshold`, `context_limit` from schema docs |

## Risks, Edge Cases & Breaking Changes

### Breaking Changes & Compatibility
- **Config removal**: `context_restart_threshold` and `context_limit` removed from mala.yaml schema
  - **Current behavior**: Config validation in `config_loader.py` uses `_ALLOWED_TOP_LEVEL_FIELDS` frozenset (line 66-85). Unknown fields are checked at lines 205-211 and raise `ConfigError("Unknown field '{field}' in mala.yaml")`
  - **Implementation**: Remove fields from `_ALLOWED_TOP_LEVEL_FIELDS` and parsing code. The existing unknown-field check provides a user-friendly error: `"Unknown field 'context_restart_threshold' in mala.yaml"`
  - **Migration path**:
    1. Remove fields from `_ALLOWED_TOP_LEVEL_FIELDS` in `config_loader.py`
    2. Remove parsing code in `config.py` (lines ~986-1005 for threshold, ~1007-1025 for limit)
    3. Update `docs/project-config.md` to remove from schema documentation
    4. Add CHANGELOG entry explaining removal (SDK handles compaction)
  - **Regression test**: Add test with config containing removed fields, assert `ConfigError` with "Unknown field" message
  - Severity: Low impact—rarely configured manually, alpha tool
  - No backward-compatibility shims per CLAUDE.md code migration rules

### Edge Cases & Failure Modes
- **PreCompact hook failure**: If archiving fails (disk full, permissions), log error but don't block compaction
- **Missing transcript_path**: If SDK provides empty or nonexistent path, log warning and skip archive
- **Rapid compactions**: Archive naming includes timestamp to prevent overwrites (`{session_id}_{timestamp}_transcript.md`)
- **Manual vs auto triggers**: Both handled identically—archive and log
- **Unreadable transcript**: If transcript exists but is unreadable, log error and continue compaction

### Risks
- **Behavior change**: SDK in-place compaction vs our checkpoint-restart approach
  - Mitigation: SDK compaction is well-tested; should be more reliable than our manual tracking
- **Lost context**: SDK summarization may lose details our checkpoint preserved
  - Mitigation: Full transcript archived before compaction; monitor compacted sessions for quality issues
- **No checkpoint injection**: Cannot inject issue context into SDK's compaction summary
  - Mitigation: Accepted limitation; SDK summary is sufficient for session continuity

### Privacy & Retention
- **Transcript archives contain plaintext conversation history**, which may include sensitive data or secrets
- **Mitigations**:
  - Archive files created with 0600 permissions (owner read/write only)
  - Archive directory explicitly chmod'd to 0700 (regardless of umask)
  - Archives stored in user config dir (`~/.config/mala/`), not in repo
  - Archive size logged at INFO level (helps users notice runaway disk usage)
  - No automatic cleanup; users manage retention manually (treat as debug/audit data)
  - **Manual cleanup command**: `find ~/.config/mala/runs/*/archives -mtime +7 -delete` (delete archives older than 7 days)
- **Documentation note**: Add warning to docs that archives may contain sensitive content; users should treat `~/.config/mala/runs/` as sensitive

### Release / Operator Notes
**Behavior change**: Sessions no longer restart on context pressure; SDK compacts in-place.

**How to recognize compaction occurred**:
- Log line: `PreCompact: archived transcript (trigger=auto) to {path}` at INFO level
- Archive file appears in `~/.config/mala/runs/{repo-key}/archives/`

**Where to find archives**:
- `~/.config/mala/runs/{repo-key}/archives/{session_id}_{timestamp}_transcript{ext}`
- List recent archives: `ls -lt ~/.config/mala/runs/*/archives/ | head`

**Migration from older behavior**:
- Older sessions that relied on checkpoint-restart will now use SDK's in-place summarization
- No user action needed; sessions continue automatically after compaction
- Remove `context_restart_threshold` and `context_limit` from `mala.yaml` if present

## Testing & Validation Strategy

### Unit Tests
- **`test_precompact.py`**: Test PreCompact hook implementation
  - `test_precompact_hook_archives_transcript`: Verify hook copies transcript to archive location
  - `test_precompact_hook_creates_archive_dir`: Verify hook creates archive directory if missing
  - `test_precompact_hook_handles_missing_transcript`: Verify logs warning and returns {} (doesn't block)
  - `test_precompact_hook_handles_missing_path`: Verify handles empty `transcript_path` gracefully
  - `test_precompact_hook_handles_io_error`: Verify OSError logged but doesn't raise
  - `test_precompact_hook_timestamp_uniqueness`: Verify rapid calls produce unique filenames
  - `test_precompact_hook_file_permissions`: Verify archived file has 0600 permissions
  - `test_precompact_hook_preserves_extension`: Verify archive uses source file extension (.md, .jsonl, etc.)

### Integration Tests
- Verify PreCompact hook registered in AgentRuntimeBuilder hooks dict with key `"PreCompact"`
- Verify AgentRuntime passes PreCompact hook to SDK options correctly
- Verify hook is invoked when SDK adapter triggers compaction (mock SDK, verify hook called)
- Remove existing context pressure integration tests (no longer applicable)

### Regression Tests
- Ensure removal of ContextPressureError doesn't break existing error handling paths
- Verify config validation still works with removed fields absent
- **test_removed_config_fields_rejected**: Config containing `context_restart_threshold` or `context_limit` raises `ConfigError` with "Unknown field" message

### Manual Validation
- Run a long coding session that triggers compaction
- Verify conversation continues normally after compaction
- Check transcript archive exists at `~/.config/mala/runs/{repo}/archives/`
- Verify archived transcript contains full pre-compaction history

### Monitoring / Observability
- Log at INFO level when transcript archived successfully (includes file size in KB)
- Log at WARNING level when transcript path missing or unreadable
- Log at ERROR level when archive write fails
- Archive size in log helps operators notice runaway disk usage patterns

### Acceptance Criteria Coverage
| Criterion | Covered By |
|-----------|------------|
| Remove broken manual tracking | Delete ContextPressureHandler, remove from MessageStreamProcessor |
| SDK handles triggering | Remove our pressure detection; SDK auto-compacts |
| Archive before compaction | PreCompact hook saves full transcript; unit tests verify |
| Breaking config change | Remove fields from config.py; validation fails on old configs |
| Session continues after compaction | SDK in-place compaction; manual validation |

## Open Questions

**Resolved during synthesis:**

1. ~~Where should transcript archives be stored?~~
   - **Decision**: Use `~/.config/mala/runs/{repo}/archives/{session_id}_{timestamp}_transcript.md`
   - **Rationale**: Consistent with existing run metadata location (`get_repo_runs_dir()`); keeps all session data in central config location rather than polluting repo with `.mala/` directory

2. ~~Should we emit a notification when compaction happens?~~
   - **Decision**: No user-visible notification; log only
   - **Rationale**: Out of scope for this change; can be added as future enhancement. Users can check debug logs if needed.

**Remaining open questions:**
- None. All design decisions resolved.

## Next Steps
After this plan is approved, run `/create-tasks` to generate:
- `--beads` → Beads issues with dependencies for multi-agent execution
- (default) → TODO.md checklist for simpler tracking

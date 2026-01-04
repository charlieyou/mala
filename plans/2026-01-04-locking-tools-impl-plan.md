# Implementation Plan: Replace Locking Scripts with Named MCP Tools

## Context & Goals
- **Spec**: `plans/2026-01-04-locking-tools-design.md`
- Replace shell script-based file locking (`lock-try.sh`, `lock-wait.sh`, etc.) with named MCP tools exposed via Claude Agent SDK
- Provides first-class tool visibility, typed JSON parameters with schema validation, and structured per-file results
- Critically, enables deadlock detection to know exactly which files are acquired vs blocked (solving batched `&&` command problem where exit=1 gives no file granularity)

## Scope & Non-Goals

### In Scope
- Implement `locking_mcp.py` with **2 MCP tools**: `lock_acquire` and `lock_release`
- Add `wait_for_lock_async()` function to `locking.py` for non-blocking async wait using `asyncio.sleep`
- Extend `get_mcp_servers(repo_path, agent_id, emit_lock_event)` in `mcp.py` to accept `agent_id` and callback
- Update `AgentRuntimeBuilder` to pass `agent_id` and `emit_lock_event` callback to `get_mcp_servers()`
- Simplify deadlock hooks: PostToolUse only for ACQUIRED/RELEASED (WAITING emitted directly by tool handler)
- Remove bash-parsing logic from `deadlock.py` (tool hooks only)
- Update lock enforcement hook error messages to reference tools instead of scripts
- Update `implementer_prompt.md` to reference tools only (no parallel script documentation)
- Unit and integration tests for new MCP tools and hooks

### Out of Scope (Non-Goals)
- Removing shell scripts from disk (separate cleanup task)
- E2E tests with actual Claude agent (covered by existing infrastructure)
- Changes to DeadlockMonitor core logic (hooks emit same LockEvent types)
- Distributed/cross-machine locking (local filesystem only)
- Cross-process locking beyond file existence (relies on atomic directory creation)

## Assumptions & Constraints

### Assumptions
- `src/infra/tools/locking.py` contains authoritative locking logic using directory creation as atomic mutex
- Claude Agent SDK is already installed as project dependency (confirmed)
- The `mcp` module integration point is `get_mcp_servers()`
- Existing hook patterns use `async def hook(hook_input, stderr, context) -> dict`

### Implementation Constraints
- **Tool Handler Signature**: `async def handler(args: dict) -> dict`
- **Return Format**: `{"content": [{"type": "text", "text": json_string}]}` (SDK envelope with JSON-serialized result)
- **PostToolUse Parsing**: Hook parses `tool_response["content"][0]["text"]` as JSON to get structured result dict
- **JSON Schema**: Full schema with `oneOf` for mutually exclusive params (e.g., `filepaths` vs `all` in `lock_release`)
- **Canonicalization**: All paths must be processed via `canonicalize_path()` before locking
- **Sorting**: Multi-file operations must sort paths in canonical order to minimize deadlock risk
- **Idempotent Release**: `lock_release` returns list of released filepaths (includes paths not held - no-op success)
- **Empty Input Rejection**: Reject empty `filepaths` arrays with error, not empty results
- **No backward-compatibility shims** per CLAUDE.md rules

### Testing Constraints
- 85% coverage threshold enforced at quality gate
- Integration tests first for full tool→hook→monitor flow
- Parallel test execution with `-n auto`
- Must use `asyncio.sleep` (not `time.sleep`) to avoid blocking event loop

## Prerequisites
- [x] Core locking logic exists in `src/infra/tools/locking.py`
- [x] Agent Runtime supports MCP server registration via `get_mcp_servers()`
- [x] Claude Agent SDK is available in the environment (confirmed installed)
- [x] Hook patterns established in `src/infra/hooks/deadlock.py`

## High-Level Approach

The implementation follows a phased approach:

1. **Core Extension**: Add `wait_for_lock_async()` to `src/infra/tools/locking.py` using `asyncio.sleep` polling
2. **MCP Server Implementation**: Create `src/infra/tools/locking_mcp.py` with tool handlers that emit WAITING events via closure-captured callback
3. **Wiring**: Extend `get_mcp_servers(repo_path, agent_id, emit_lock_event)` in `mcp.py` and pass callback from `AgentRuntimeBuilder`
4. **Hook Simplification**: Remove PreToolUse hook; keep PostToolUse for ACQUIRED/RELEASED events only
5. **Prompt Update**: Replace shell script documentation in `implementer_prompt.md` with MCP tool documentation

## Technical Design

### Architecture

The locking MCP server is created per-agent with bound `agent_id` and `repo_namespace`:

```
AgentRuntimeBuilder.build()
  └── with_mcp()
        └── get_mcp_servers(repo_path, agent_id, emit_lock_event)
              └── Creates mala-locking server with closure-captured callback
```

**Simplified 2-tool API** (reduced from 5 tools):
- `lock_acquire` → `try_lock()` for each file; if blocked, emits WAITING via callback, then `wait_for_lock_async()` until **any** progress
- `lock_release` → `release_lock()` for specified files, or `cleanup_agent_locks()` when `all=true`

**Event emission strategy** (closure-based, not hook-based for WAITING):
- **WAITING**: Emitted directly by `lock_acquire` tool handler via closure-captured `emit_lock_event` callback when `try_lock()` fails
- **ACQUIRED/RELEASED**: Emitted by PostToolUse hook from structured tool results
- **Why**: PreToolUse cannot know which files are blocked (only discovered after `try_lock()` attempts). Closure injection verified to work with Claude Agent SDK's `@tool` decorator.

### Data Model

**Simplified 2-Tool API:**

```python
# ============================================
# lock_acquire - Try to acquire locks, wait if blocked until ANY progress
# ============================================

# Input
{
  "filepaths": ["src/main.py", "src/utils.py", "src/config.py"],
  "timeout_seconds": 30  # optional, default 30; use 0 for non-blocking
}

# Output (returns when ANY progress made, or all acquired, or timeout)
{
  "results": [
    {"filepath": "src/main.py", "acquired": True, "holder": None},
    {"filepath": "src/utils.py", "acquired": True, "holder": None},   # just became available
    {"filepath": "src/config.py", "acquired": False, "holder": "bd-43"}  # still blocked
  ],
  "all_acquired": False  # agent should call again for remaining files
}

# Behavior:
# 1. Try all files (sorted canonical order)
# 2. If all acquired → return immediately
# 3. If some blocked → emit WAITING via callback, then wait until ANY becomes available
#    - Spawns wait_for_lock_async() task per blocked file
#    - Uses asyncio.wait(..., return_when=FIRST_COMPLETED)
#    - Cancels remaining wait tasks when first completes (prevents unwanted acquisitions)
# 4. Acquire newly available file(s), return with current state
# 5. Agent calls again for remaining blocked files (incremental progress)
# 6. timeout_seconds=0 → non-blocking, return immediately after try
# 7. On timeout: keep already-acquired locks, return partial results
#
# WAITING emission: once per blocked filepath per call (not on each poll)
# "Progress" = at least one blocked file became available (not initial try-pass acquisitions)

# ============================================
# lock_release - Release specific files or all held locks
# ============================================

# Input (mutually exclusive - use oneOf in schema)
{"filepaths": ["src/main.py", "src/utils.py"]}  # Release specific files
# OR
{"all": true}  # Release ALL locks held by this agent

# Output (idempotent - lists requested paths even if not held)
{
  "released": ["src/main.py", "src/utils.py"],
  "count": 2
}

# Error if both filepaths and all provided, or neither
```

### API/Interface Design

**`src/infra/tools/locking.py`** (additions):
```python
async def wait_for_lock_async(
    filepath: str,
    agent_id: str,
    repo_namespace: str | None = None,
    timeout_seconds: float = 30.0,
    poll_interval_ms: int = 100,
) -> bool:
    """Async version of wait_for_lock using asyncio.sleep.
    
    Returns True when lock acquired, False on timeout.
    Designed to be spawned as tasks for asyncio.wait(return_when=FIRST_COMPLETED).
    """

def get_lock_holder(filepath: str, repo_namespace: str | None = None) -> str | None:
    """Read holder agent_id from lock metadata file. Returns None if not locked."""
```

**`src/infra/tools/locking_mcp.py`** (new file):
```python
def create_locking_mcp_server(
    agent_id: str,
    repo_namespace: str,
    emit_lock_event: Callable[[LockEvent], None],  # Closure-captured callback
) -> McpSdkServerConfig:
    """Create MCP server with 2 locking tools bound to agent context.
    
    The emit_lock_event callback is captured by tool handler closures,
    allowing WAITING events to be emitted during lock acquisition.
    """

# Tool handlers are closures that capture emit_lock_event:
@tool("lock_acquire", "...", {...})
async def lock_acquire(args: dict) -> dict:
    # emit_lock_event is captured from enclosing scope
    for filepath in blocked_files:
        emit_lock_event(LockEvent(WAITING, filepath, agent_id))
    # ... wait and retry logic
```

**`src/infra/mcp.py`** (extended signature):
```python
def get_mcp_servers(
    repo_path: Path,
    agent_id: str | None = None,
    emit_lock_event: Callable[[LockEvent], None] | None = None,  # Callback to DeadlockMonitor
) -> dict[str, Any]:
    """Get MCP servers configuration for agents.

    When agent_id and emit_lock_event are provided, includes the mala-locking server
    with WAITING event emission capability.
    
    repo_namespace is derived from repo_path.name (consistent with existing locking.py usage).
    emit_lock_event routes to DeadlockMonitor.handle_event() - must be thread-safe/async-safe.
    """
```

**`src/infra/hooks/deadlock.py`** (simplified - PostToolUse only):
```python
def make_lock_event_hook(...) -> PostToolUseHook:
    """PostToolUse hook for lock_acquire/lock_release - emit ACQUIRED/RELEASED.
    
    Note: WAITING events are emitted directly by the tool handler via
    closure-captured callback, not via hooks (PreToolUse cannot know
    which files are blocked before try_lock() attempts).
    """
```

### File Impact Summary

| Path | Status | Description |
|------|--------|-------------|
| `src/infra/tools/locking.py` | Exists | Add `wait_for_lock_async()` and `get_lock_holder()` |
| `src/infra/tools/locking_mcp.py` | **New** | MCP server + tool definitions with closure-captured callback |
| `src/infra/mcp.py` | Exists | Extend `get_mcp_servers()` to accept `agent_id` and `emit_lock_event` |
| `src/infra/agent_runtime.py` | Exists | Pass `agent_id` and `emit_lock_event` to `get_mcp_servers()` |
| `src/infra/hooks/deadlock.py` | Exists | Simplify to PostToolUse only (WAITING emitted by tool handler) |
| `src/infra/hooks/locking.py` | Exists | Update error message (scripts → tools) |
| `src/infra/hooks/__init__.py` | Exists | Export new hook factories |
| `src/prompts/implementer_prompt.md` | Exists | Replace script docs with tool docs |
| `tests/unit/infra/tools/test_locking_mcp.py` | **New** | Unit tests for MCP tool handlers |
| `tests/unit/infra/tools/test_locking.py` | Exists | Add tests for `wait_for_lock_async()` |
| `tests/unit/infra/hooks/test_deadlock_hook.py` | Exists | Update/replace tests for tool-based hooks |
| `tests/integration/infra/test_locking_mcp.py` | **New** | Integration tests for tool→hook→monitor flow |

## Risks, Edge Cases & Breaking Changes

### Risks
| Risk | Mitigation |
|------|------------|
| Event loop blocking in `lock_acquire` | Implement `wait_for_lock_async()` with `asyncio.sleep` not `time.sleep` |
| MCP server registration complexity | Follow existing SDK patterns in codebase (`get_mcp_servers`) |
| Hook matching edge cases | Use matcher `mcp__mala-locking__lock_acquire\|mcp__mala-locking__lock_release` (SDK tool naming verified) |
| Path canonicalization mismatch | Reuse existing `canonicalize_path()` in both tools and hooks |
| Tool result JSON parsing errors | PostToolUse hook parses JSON from `content[0].text`; use try/except with clear error |

### Edge Cases
- **Empty filepaths array**: Reject with error (user decision: not empty results)
- **Release lock not held**: Return success (idempotent, user decision)
- **Timeout in lock_acquire**: Keep already-acquired locks, return per-file results showing acquired vs blocked
- **Partial success**: Return per-file results, `all_acquired: False`, agent calls again
- **Multi-file operations**: Process in sorted canonical order to reduce deadlock risk
- **Stale locks from crash**: Existing `cleanup_agent_locks()` handles this; `lock_release(all=true)` exposes it
- **Both filepaths and all in lock_release**: Reject with error (mutually exclusive)
- **WAITING emission**: Once per blocked filepath per `lock_acquire` call (not per poll iteration)

### Breaking Changes & Compatibility
- **Prompt Change**: Agent will no longer see shell script documentation
  - *Impact*: Old conversations may reference scripts, but new runs use tools
  - *Mitigation*: Scripts remain on disk; this is intentional behavior change
- **Hook Logic**: Bash-parsing hooks removed entirely (user decision: tool hooks only)
  - *Impact*: Existing tests depending on bash parsing will need updates
  - *Mitigation*: Update tests as part of implementation

## Testing & Validation Strategy

### Test Priority: Integration First
Per user decision, integration tests are prioritized to validate the full tool→hook→monitor flow.

### Unit Tests (`tests/unit/infra/tools/test_locking_mcp.py`)
- Verify JSON schema validation (e.g., empty filepaths rejected)
- Verify return format structure matches spec
- Verify `wait_for_lock_async()` uses asyncio (mock time to avoid slow tests)
- Mock `locking.py` functions to verify wiring
- Test idempotent release behavior
- **Verify WAITING events emitted via closure callback when try_lock fails**

### Unit Tests (`tests/unit/infra/hooks/test_deadlock_hook.py`)
- Test PostToolUse hook matchers (no PreToolUse hooks needed)
- Verify ACQUIRED/RELEASED event emission from structured tool results
- Remove/update bash-parsing tests

### Integration Tests (`tests/integration/infra/test_locking_mcp.py`)
- **Real Filesystem**: Test actual directory creation/deletion
- **Concurrency**: Spawn background task to hold lock, verify `lock_acquire` returns on progress or timeout
- **Early Return**: Verify `lock_acquire` returns when ANY blocked file becomes available
- **Closure Callback**: Verify WAITING events emitted via callback during wait phase (not via hooks)
- **PostToolUse Hook**: Verify ACQUIRED/RELEASED events emitted from tool results
- **Multi-file Batches**: Verify sorted order and per-file granularity

### Manual Verification
- Run session where agent attempts to lock same file twice
- Verify `lock_acquire` returns holder info for blocked files
- Verify tools appear in Claude's tool list

### Acceptance Criteria Coverage
| Spec AC | Covered By |
|---------|------------|
| Tools exposed via SDK | `src/infra/mcp.py`, `src/infra/agent_runtime.py`, Integration tests |
| `lock_acquire` returns per-file results | Unit tests, Integration tests |
| `lock_acquire` with timeout=0 is non-blocking | Unit tests |
| `lock_acquire` returns on ANY progress | Integration tests (concurrent lock holder) |
| `lock_release` is idempotent | Unit tests |
| `lock_release` with `all=true` clears agent locks | Unit tests |
| PostToolUse hooks intercept MCP tools (not bash) | Updated `test_deadlock_hook.py` |
| WAITING emitted via closure callback during blocking | Integration tests with mock callback |
| Empty filepaths rejected | Unit tests |

## Open Questions

None. All key decisions resolved via user interview:
1. MCP Wiring: Option A - Extend `get_mcp_servers(repo_path, agent_id, emit_lock_event)` ✓
2. Prompt docs: Replace scripts with tools only ✓
3. Release semantics: Idempotent (released=True even if not held) ✓
4. Empty input: Reject with error ✓
5. Hook strategy: PostToolUse hooks only for ACQUIRED/RELEASED; WAITING via closure callback ✓
6. Test priority: Integration tests first ✓
7. SDK: Already installed ✓
8. WAITING event timing: Tool handler emits via closure-captured callback (PreToolUse impossible) ✓

## Next Steps

After this plan is approved, run `/create-tasks` to generate:
- `--beads` → Beads issues with dependencies for multi-agent execution
- (default) → TODO.md checklist for simpler tracking

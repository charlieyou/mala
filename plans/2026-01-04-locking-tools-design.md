# Design: Replace Locking Scripts with Named MCP Tools

**Date**: 2026-01-04  
**Status**: Draft (v2 - Oracle reviewed)  
**Author**: Agent  

## Summary

Replace shell script-based file locking (`lock-try.sh`, `lock-wait.sh`, etc.) with named MCP tools exposed via the Claude Agent SDK. This provides first-class tool visibility, typed parameters, cleaner hook integration, and **critically, structured multi-file locking results**.

## Motivation

### Current State

Agents acquire file locks by calling bash scripts:

```bash
lock-try.sh src/main.py      # Exit 0=acquired, 1=blocked
lock-wait.sh src/main.py 30  # Wait up to 30s
lock-release.sh src/main.py
```

**Problems:**

1. **Opaque to hooks**: Deadlock detection hook must regex-parse bash commands to detect lock operations
2. **No type safety**: Parameters are positional strings; easy to misuse
3. **Indirect invocation**: Scripts → Python module → actual lock logic
4. **Poor discoverability**: Agents must be told about scripts in prompts; not visible as tools
5. **CRITICAL: Batched commands break deadlock detection** (see T-019b86d3-dad0-739c-8ab3-ba98f3951465)

### The Batching Problem

When agents batch lock commands with `&&`:

```bash
lock-try.sh a.py && lock-try.sh b.py && lock-try.sh c.py
```

If exit code is 1 (contention), the hook **cannot determine which file failed**. The `&&` short-circuits on first failure, but we don't know if it was `a.py`, `b.py`, or `c.py`. This caused real deadlocks to go undetected.

**Root cause**: Bash exit codes provide no per-file granularity.

### Proposed State

Agents use named tools with **structured per-file results**:

```
Tool: lock_try
Input: {"filepaths": ["src/main.py", "src/utils.py", "src/config.py"]}
Output: {
  "results": [
    {"filepath": "src/main.py", "acquired": true, "holder": null},
    {"filepath": "src/utils.py", "acquired": false, "holder": "bd-43"},
    {"filepath": "src/config.py", "acquired": true, "holder": null}
  ],
  "all_acquired": false
}
```

**Benefits:**

1. **First-class tool visibility**: Tools appear in Claude's tool list with descriptions
2. **Typed parameters**: SDK validates inputs; clear schema in tool definition
3. **Clean hook integration**: `PreToolUse`/`PostToolUse` match by tool name, not regex
4. **Direct invocation**: Tool handler calls locking functions directly
5. **Per-file granularity**: Deadlock detection knows exactly which files are acquired vs blocked

## Design

### Tool Definitions

Create an MCP server registered under key `mala-locking`. Tool names exposed to Claude follow pattern `mcp__{server_key}__{tool_name}`.

| Tool Name | Description | Parameters | Returns |
|-----------|-------------|------------|---------|
| `lock_try` | Try to acquire locks on files | `filepaths: list[str]` | `{results: [{filepath, acquired, holder}], all_acquired: bool}` |
| `lock_wait` | Wait for locks on files with timeout | `filepaths: list[str], timeout_seconds?: float, poll_interval_ms?: int` | `{results: [{filepath, acquired, holder}], all_acquired: bool}` |
| `lock_release` | Release locks on files | `filepaths: list[str]` | `{results: [{filepath, released}], all_released: bool}` |
| `lock_status` | Check lock status for files | `filepaths: list[str]` | `{results: [{filepath, held_by_me, holder}]}` |
| `lock_release_all` | Release all locks held by current agent | (none) | `{count: int}` |

**Note**: All multi-file tools process files in **sorted canonical order** to reduce deadlock risk when multiple agents acquire overlapping sets.

### Implementation Location

```
src/infra/tools/
├── locking.py           # Existing lock functions (unchanged)
└── locking_mcp.py       # NEW: MCP tool definitions

src/infra/
├── mcp.py               # MODIFY: Add locking server to get_mcp_servers()
└── agent_runtime.py     # Uses AgentRuntimeBuilder.with_mcp()
```

### Tool Implementation (SDK-Correct)

**Important SDK constraints** (from Oracle review):
- Tool handler signature must be `async def handler(args: dict) -> dict`
- `input_schema` must be full JSON Schema with explicit `required` array
- Optional params need defaults in handler, not schema
- Return `{"content": [{"type": "text", "text": "..."}]}`

```python
# src/infra/tools/locking_mcp.py
import json
from claude_agent_sdk import tool, create_sdk_mcp_server

from .locking import (
    try_lock, wait_for_lock_async, release_lock, 
    get_lock_holder, cleanup_agent_locks, canonicalize_path
)

def create_locking_mcp_server(
    agent_id: str,
    repo_namespace: str | None = None,
) -> McpSdkServerConfig:
    """Create MCP server with locking tools bound to agent context.
    
    Note: Does not need LockManager injection; calls module functions directly.
    """

    def _canonical(filepath: str) -> str:
        """Canonicalize path for consistent deadlock graph nodes."""
        return canonicalize_path(filepath, repo_namespace)

    @tool(
        name="lock_try",
        description="Try to acquire locks on multiple files without blocking. Returns per-file results.",
        input_schema={
            "type": "object",
            "properties": {
                "filepaths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file paths to lock"
                }
            },
            "required": ["filepaths"]
        },
    )
    async def lock_try(args: dict) -> dict:
        filepaths = args["filepaths"]
        # Sort by canonical path to reduce deadlock risk
        sorted_paths = sorted(filepaths, key=_canonical)
        
        results = []
        for fp in sorted_paths:
            acquired = try_lock(fp, agent_id, repo_namespace)
            holder = None if acquired else get_lock_holder(fp, repo_namespace)
            results.append({
                "filepath": fp,
                "acquired": acquired,
                "holder": holder,
            })
        
        all_acquired = all(r["acquired"] for r in results)
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({"results": results, "all_acquired": all_acquired})
            }]
        }

    @tool(
        name="lock_wait",
        description="Wait for locks on multiple files with timeout. Processes in sorted order.",
        input_schema={
            "type": "object",
            "properties": {
                "filepaths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file paths to lock"
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Max seconds to wait per file (default: 30)"
                },
                "poll_interval_ms": {
                    "type": "integer",
                    "description": "Polling interval in ms (default: 100)"
                }
            },
            "required": ["filepaths"]
        },
    )
    async def lock_wait(args: dict) -> dict:
        filepaths = args["filepaths"]
        timeout = args.get("timeout_seconds", 30.0)
        poll_ms = args.get("poll_interval_ms", 100)
        
        sorted_paths = sorted(filepaths, key=_canonical)
        
        results = []
        for fp in sorted_paths:
            # Use async version to avoid blocking event loop
            acquired = await wait_for_lock_async(
                fp, agent_id, repo_namespace, timeout, poll_ms
            )
            holder = None if acquired else get_lock_holder(fp, repo_namespace)
            results.append({
                "filepath": fp,
                "acquired": acquired,
                "holder": holder,
            })
        
        all_acquired = all(r["acquired"] for r in results)
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({"results": results, "all_acquired": all_acquired})
            }]
        }

    @tool(
        name="lock_release",
        description="Release locks on multiple files. Idempotent (succeeds even if not held).",
        input_schema={
            "type": "object",
            "properties": {
                "filepaths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file paths to unlock"
                }
            },
            "required": ["filepaths"]
        },
    )
    async def lock_release(args: dict) -> dict:
        filepaths = args["filepaths"]
        
        results = []
        for fp in filepaths:
            released = release_lock(fp, agent_id, repo_namespace)
            results.append({"filepath": fp, "released": released})
        
        all_released = all(r["released"] for r in results)
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({"results": results, "all_released": all_released})
            }]
        }

    @tool(
        name="lock_status",
        description="Check lock status for multiple files.",
        input_schema={
            "type": "object",
            "properties": {
                "filepaths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of file paths to check"
                }
            },
            "required": ["filepaths"]
        },
    )
    async def lock_status(args: dict) -> dict:
        filepaths = args["filepaths"]
        
        results = []
        for fp in filepaths:
            holder = get_lock_holder(fp, repo_namespace)
            results.append({
                "filepath": fp,
                "held_by_me": holder == agent_id,
                "holder": holder,
            })
        
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({"results": results})
            }]
        }

    @tool(
        name="lock_release_all",
        description="Release all file locks held by this agent.",
        input_schema={"type": "object", "properties": {}, "required": []},
    )
    async def lock_release_all(args: dict) -> dict:
        count = cleanup_agent_locks(agent_id)
        return {
            "content": [{
                "type": "text",
                "text": json.dumps({"count": count})
            }]
        }

    return create_sdk_mcp_server(
        name="mala-locking",
        version="1.0.0",
        tools=[lock_try, lock_wait, lock_release, lock_status, lock_release_all],
    )
```

### Async Wait Implementation

Add to `src/infra/tools/locking.py`:

```python
import asyncio

async def wait_for_lock_async(
    filepath: str,
    agent_id: str,
    repo_namespace: str | None = None,
    timeout_seconds: float = 30.0,
    poll_interval_ms: int = 100,
) -> bool:
    """Async version of wait_for_lock. Uses asyncio.sleep to avoid blocking."""
    deadline = asyncio.get_event_loop().time() + timeout_seconds
    poll_seconds = poll_interval_ms / 1000.0
    
    while True:
        if try_lock(filepath, agent_id, repo_namespace):
            return True
        
        if asyncio.get_event_loop().time() >= deadline:
            return False
        
        await asyncio.sleep(poll_seconds)
```

### Integration with AgentRuntimeBuilder

Per Oracle review, MCP servers should be wired through `AgentRuntimeBuilder`, not a new method in `AgentSessionRunner`. The challenge is that locking tools need `agent_id`, which isn't available in the current `get_mcp_servers(repo_path)` signature.

**Option A**: Extend `get_mcp_servers()` to accept `agent_id`:

```python
# src/infra/mcp.py
def get_mcp_servers(
    repo_path: Path,
    agent_id: str | None = None,  # NEW
) -> dict[str, McpServerConfig]:
    servers = {}
    # ... existing MCP servers ...
    
    if agent_id:
        from src.infra.tools.locking_mcp import create_locking_mcp_server
        servers["mala-locking"] = create_locking_mcp_server(
            agent_id=agent_id,
            repo_namespace=str(repo_path),
        )
    
    return servers
```

**Option B**: Construct locking server directly in `AgentRuntimeBuilder.build()`:

```python
# src/infra/agent_runtime.py
class AgentRuntimeBuilder:
    def build(self) -> ClaudeAgentOptions:
        mcp_servers = get_mcp_servers(self.repo_path)
        
        # Add locking server (needs agent_id which we have here)
        from src.infra.tools.locking_mcp import create_locking_mcp_server
        mcp_servers["mala-locking"] = create_locking_mcp_server(
            agent_id=self.agent_id,
            repo_namespace=str(self.repo_path),
        )
        
        return ClaudeAgentOptions(mcp_servers=mcp_servers, ...)
```

**Recommendation**: Option A is cleaner (single source of truth in `mcp.py`).

### Hook Updates

**SDK Hook API** (corrected from Oracle review):
- `HookMatcher` uses `matcher: str` (regex pattern), not `tool_name`
- Post hook receives `hook_input["tool_response"]`, not `tool_result`

#### PreToolUse Hook (emit WAITING before lock_wait starts)

```python
# src/infra/hooks/deadlock.py

def make_lock_wait_pretool_hook(
    deadlock_monitor: DeadlockMonitor,
    agent_id: str,
    repo_namespace: str | None,
) -> HookMatcher:
    """Emit WAITING events before lock_wait starts (may block)."""
    
    async def on_lock_wait_pre(hook_input: dict) -> dict:
        tool_input = hook_input.get("tool_input", {})
        filepaths = tool_input.get("filepaths", [])
        
        for fp in filepaths:
            canonical = canonicalize_path(fp, repo_namespace)
            holder = get_lock_holder(fp, repo_namespace)
            if holder and holder != agent_id:
                deadlock_monitor.add_wait(agent_id, canonical, holder)
        
        return {"decision": "approve"}
    
    return HookMatcher(
        matcher="mcp__mala-locking__lock_wait",
        hooks=[on_lock_wait_pre],
    )
```

#### PostToolUse Hook (emit ACQUIRED/RELEASED based on results)

```python
def make_lock_event_posttool_hook(
    deadlock_monitor: DeadlockMonitor,
    agent_id: str,
    repo_namespace: str | None,
) -> HookMatcher:
    """Emit ACQUIRED/RELEASED events after lock tools complete."""
    
    async def on_lock_tool_complete(hook_input: dict) -> dict:
        tool_name = hook_input.get("tool_name", "")
        tool_response = hook_input.get("tool_response", {})
        
        # Parse structured result from tool output
        content = tool_response.get("content", [])
        if not content:
            return {}
        
        result = json.loads(content[0].get("text", "{}"))
        results = result.get("results", [])
        
        for r in results:
            fp = r["filepath"]
            canonical = canonicalize_path(fp, repo_namespace)
            
            if "lock_try" in tool_name or "lock_wait" in tool_name:
                if r.get("acquired"):
                    deadlock_monitor.add_hold(agent_id, canonical)
                else:
                    holder = r.get("holder")
                    if holder:
                        deadlock_monitor.add_wait(agent_id, canonical, holder)
            
            elif "lock_release" in tool_name:
                if r.get("released"):
                    deadlock_monitor.release(agent_id, canonical)
        
        return {}
    
    return HookMatcher(
        matcher="mcp__mala-locking__lock_.*",
        hooks=[on_lock_tool_complete],
    )
```

### Lock Enforcement Hook Update

Update `src/infra/hooks/locking.py` to reference tools instead of scripts:

```python
# Before:
"reason": f"File {file_path} is not locked. Acquire lock with: lock-try.sh {file_path}",

# After:
"reason": f"File {file_path} is not locked. Use lock_try tool with filepaths: [\"{file_path}\"]",
```

### Prompt Updates

Update `implementer_prompt.md` to reference tools instead of scripts:

```markdown
## File Locking

Use locking tools to coordinate file access with other agents:

| Tool | Description |
|------|-------------|
| `lock_try` | Try to acquire locks (returns immediately with per-file results) |
| `lock_wait` | Wait for locks with timeout |
| `lock_release` | Release locks you hold |
| `lock_status` | Check who holds locks |

### Workflow

1. Call `lock_try` with ALL files you plan to modify
2. Check results: work on acquired files, note blocked files and holders
3. For blocked files, call `lock_wait` with reasonable timeout
4. Edit files you have locked
5. Call `lock_release` when done

### Example

```
Tool: lock_try
Input: {"filepaths": ["src/main.py", "src/utils.py", "src/config.py"]}
→ {
    "results": [
      {"filepath": "src/main.py", "acquired": true, "holder": null},
      {"filepath": "src/utils.py", "acquired": false, "holder": "bd-43"},
      {"filepath": "src/config.py", "acquired": true, "holder": null}
    ],
    "all_acquired": false
  }

# Wait for blocked file
Tool: lock_wait
Input: {"filepaths": ["src/utils.py"], "timeout_seconds": 60}
→ {"results": [{"filepath": "src/utils.py", "acquired": true, "holder": null}], "all_acquired": true}

# Release when done
Tool: lock_release
Input: {"filepaths": ["src/main.py", "src/utils.py", "src/config.py"]}
→ {"results": [...], "all_released": true}
```
```

## Migration Plan

### Phase 1: Add Tools (Non-Breaking)

1. Implement `wait_for_lock_async()` in `locking.py`
2. Create `locking_mcp.py` with tool definitions
3. Extend `get_mcp_servers()` to accept `agent_id` and register locking server
4. Update prompts to mention both scripts and tools (parallel support)

### Phase 2: Update Hooks

1. Add new hook matchers for `mcp__mala-locking__*` tools
2. Keep existing bash-parsing hooks as fallback during transition
3. Test deadlock detection with both script and tool paths

### Phase 3: Deprecate Scripts

1. Remove script references from `implementer_prompt.md`
2. Update `src/infra/hooks/locking.py` error messages to reference tools
3. Monitor logs for any remaining script usage

### Phase 4: Cleanup

1. Remove bash scripts from `src/scripts/lock-*.sh`
2. Remove regex-based hook parsing from `src/infra/hooks/deadlock.py`
3. Remove CLI dispatch from `locking.py` (`_cli_main`, `COMMANDS`, etc.)
4. Remove `test-mutex.sh` (if not used elsewhere)

## Testing

### Unit Tests

- `lock_try`: Returns per-file results with correct acquired/holder
- `lock_wait`: Async polling works; respects timeout
- `lock_release`: Idempotent; returns per-file released status
- `lock_status`: Correctly identifies held_by_me vs other holder
- Path canonicalization in results matches deadlock monitor expectations

### Integration Tests

- Agent can acquire multiple locks via `lock_try`
- Contention scenario: Agent A holds file, Agent B gets `acquired: false` with holder
- Deadlock detection receives ACQUIRED/WAITING events from tool hooks
- PreToolUse `lock_wait` hook emits WAITING before blocking
- Lock enforcement hook blocks writes on unlocked files with correct error message

### E2E Tests

- Two agents coordinate via locking tools (no deadlock)
- Deadlock scenario: cycle detected and victim cancelled
- Mixed scenario: one agent uses tools, another uses scripts (transition period)

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Event loop blocking in `lock_wait` | Implement `wait_for_lock_async()` with `asyncio.sleep` |
| MCP server registration complexity | Follow SDK examples; test in isolation first |
| Tool naming conflicts | Use `mala-locking` namespace; document full tool names |
| Migration breaks existing agents | Phased rollout; keep scripts during transition |
| Hook matching edge cases | Regex `mcp__mala-locking__lock_.*` covers all tools |
| Path canonicalization mismatch | Reuse existing `canonicalize_path()` in both tools and hooks |

## Resolved Questions (from Oracle Review)

1. **Async handling**: Yes, implement `wait_for_lock_async()` with `asyncio.sleep()` polling. Do not use `time.sleep()` in async handlers.

2. **Tool handler signature**: Must be `async def handler(args: dict) -> dict`, not typed params.

3. **JSON Schema**: Use full schema with explicit `required` array. Optional params have defaults in handler.

4. **HookMatcher API**: Uses `matcher: str` (regex), not `tool_name`.

5. **Batching**: All tools are multi-file by default (`lock_try`, etc.). Process in sorted canonical order to reduce deadlock risk.

6. **Release semantics**: Idempotent—return `released: true` even if lock wasn't held (no-op success).

## References

- [Claude Agent SDK - Custom Tools](https://platform.claude.com/docs/en/agent-sdk/custom-tools)
- [Claude Agent SDK - Python Reference](https://platform.claude.com/docs/en/agent-sdk/python)
- Thread T-019b86d3-dad0-739c-8ab3-ba98f3951465: Batched lock commands break deadlock detection
- Current implementation: [src/infra/tools/locking.py](/home/cyou/mala/src/infra/tools/locking.py)
- Current scripts: [src/scripts/](/home/cyou/mala/src/scripts/)
- Deadlock hook: [src/infra/hooks/deadlock.py](/home/cyou/mala/src/infra/hooks/deadlock.py)
- MCP wiring: [src/infra/mcp.py](/home/cyou/mala/src/infra/mcp.py), [src/infra/agent_runtime.py](/home/cyou/mala/src/infra/agent_runtime.py)

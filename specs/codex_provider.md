# Codex Implementation Provider

## Overview

Add OpenAI Codex as an alternative implementation agent provider alongside Claude. This enables using Codex CLI (`codex exec --json`) to power the core coding loop instead of Claude Agent SDK. The provider is selected via CLI flag (`--provider codex`) with Claude remaining the default.

## Goals

- Enable Codex as an alternative implementation agent for perceived improved reasoning capability
- Maintain Claude as the default provider with zero regression
- Keep the existing Codex code review functionality completely separate and unchanged
- Provide clean abstraction that allows future provider additions

## Non-Goals (Out of Scope)

- Replacing Claude as the default provider
- Modifying the existing Codex review pipeline (`codex_review.py`)
- Provider fallback mechanisms (if Codex fails, the run fails)
- Hook support for Codex (file locking, lint caching)
- Per-issue model selection (all issues in a run use the same provider)
- Validation evidence parsing from Codex output (MVP scope - rely on commit checks only)

## User Stories

- As a user, I want to run `mala run --provider codex` to use Codex instead of Claude for implementation
- As a user, I want clear error messages when Codex fails or is unavailable
- As a user, I want the same retry/gate/review flow regardless of which provider I use

## Technical Design

### Architecture

The system already has a well-designed `SDKClientProtocol` in `src/pipeline/agent_session_runner.py:86-123` that abstracts SDK client interactions. We'll create a `CodexClient` that implements this protocol by wrapping `codex exec` CLI calls.

```
AgentSessionRunner
      |
      v
SDKClientProtocol  <-- ClaudeSDKClient (existing)
                   <-- CodexClient (new)
```

### Key Components

- **CodexClient** (`src/codex_client.py`): Implementation of `SDKClientProtocol` that wraps Codex CLI
  - Handles `query()` by spawning `codex exec --json` subprocess with stdin piping
  - Handles `receive_response()` by streaming JSONL events from stdout (skipping non-JSON header lines)
  - Captures stderr for diagnostics and error classification
  - Manages process lifecycle and synthetic session IDs

- **Provider selection** (`src/cli.py`): New `--provider` flag to choose provider
  - Values: `claude` (default), `codex`
  - Passed through to orchestrator configuration
  - Note: Using `--provider` (not `--model`) to avoid confusion with model selection within a provider

- **AgentSessionRunner changes** (`src/pipeline/agent_session_runner.py`):
  - Accept provider type in config (`self.config.provider`)
  - Branch client and options construction based on provider:
    - Claude: Build `ClaudeAgentOptions`, instantiate `ClaudeSDKClient`
    - Codex: Build `CodexClientOptions`, instantiate `CodexClient`
  - Skip hook setup when provider is Codex
  - Skip `WAIT_FOR_LOG` phase when provider is Codex (no session log to wait for)

- **Quality gate changes** (`src/quality_gate.py` / `AgentSessionRunner`):
  - When provider is Codex, set `check_validation_evidence=False` in gate config
  - Gate checks commit existence only (no log parsing for validation commands)

### Codex CLI Invocation

The exact CLI command for implementation (prompt via stdin to avoid arg length limits):

```bash
echo "<prompt>" | codex exec \
  --json \
  --dangerously-bypass-approvals-and-sandbox \
  -C <repo_path> \
  -c model_reasoning_effort=<thinking_mode> \
  -
```

Flags:
- `--json`: Output JSONL events to stdout (required for parsing)
- `--dangerously-bypass-approvals-and-sandbox`: Required for automated execution
- `-C <repo_path>`: Set working directory
- `-c model_reasoning_effort=<mode>`: Optional thinking mode (high/medium/low)
- `-`: Read prompt from stdin (avoids shell argument length limits)

### Codex Output Format (JSONL)

Codex `--json` outputs to stdout with:
1. **Header lines** (non-JSON): `session id: <uuid>`, separator lines, etc.
2. **JSONL events**: One JSON object per line

Key event types:

```jsonl
{"type": "message_start", "session_id": "...", ...}
{"type": "content_block_delta", "delta": {"type": "text_delta", "text": "..."}, ...}
{"type": "tool_use", "id": "...", "name": "bash", "input": {"command": "..."}, ...}
{"type": "tool_result", "tool_use_id": "...", "content": "...", ...}
{"type": "message_stop", ...}
```

**Important**: Parser must skip non-JSON lines gracefully (header output before JSONL starts).

### Message Mapping

Map Codex JSONL events to SDK-compatible types:

| Codex Event | SDK Type | Notes |
|-------------|----------|-------|
| `message_start` | (internal) | Extract session_id |
| `content_block_delta` (text) | `AssistantMessage` with `TextBlock` | Buffer text deltas, flush on tool_use or message_stop |
| `tool_use` | `AssistantMessage` with `ToolUseBlock` | Tool invocation (already executed by Codex) |
| `tool_result` | `AssistantMessage` with `ToolResultBlock` | Tool output (already received by Codex) |
| `message_stop` | `ResultMessage` | Session complete |
| (EOF without message_stop) | `ResultMessage` | Fallback: emit with buffered text and check exit code |

**Important - Tool Execution Ownership**: Codex CLI executes tools autonomously via `--dangerously-bypass-approvals-and-sandbox`. The `tool_use` and `tool_result` events are **observational** - they report what Codex already did, not requests for the runner to execute. `AgentSessionRunner` must NOT attempt to execute tools for the Codex provider; it only logs/observes them. This is the same behavior as with Claude SDK (which also executes tools internally).

### Session Log Handling (Provider-Specific)

**Claude**: Logs stored in `~/.claude/projects/{repo-hash}/{session-id}.jsonl`
- `WAIT_FOR_LOG` phase waits for this file
- Quality gate parses log for validation evidence

**Codex**: Logs stored in `~/.codex/sessions/YYYY/MM/DD/rollout-{datetime}-{session-id}.jsonl`
- **Skip `WAIT_FOR_LOG` phase entirely** - no log file needed for MVP
- Quality gate skips validation evidence parsing (`check_validation_evidence=False`)
- Only commit existence is verified

This eliminates the log path mismatch issue by not requiring Codex logs at all for MVP.

### Data Model

New config field in `AgentSessionConfig`:

```python
@dataclass
class AgentSessionConfig:
    # ... existing fields ...
    provider: Literal["claude", "codex"] = "claude"
```

### API Design

CodexClient will implement `SDKClientProtocol`:

```python
@dataclass
class CodexClientOptions:
    cwd: str
    env: dict[str, str] | None = None
    thinking_mode: str | None = None  # e.g., "high", "medium", "low"


class CodexClient:
    """Codex CLI wrapper implementing SDKClientProtocol."""

    def __init__(self, options: CodexClientOptions):
        self.options = options
        self._process: asyncio.subprocess.Process | None = None
        self._session_id: str | None = None
        self._accumulated_prompt: str = ""  # For retry context
        self._stderr_output: str = ""  # Captured stderr for diagnostics

    async def __aenter__(self) -> Self:
        """No setup needed - process created per query."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Terminate process if still running."""
        if self._process and self._process.returncode is None:
            self._process.terminate()
            await self._process.wait()

    async def query(self, prompt: str, session_id: str | None = None) -> None:
        """Spawn codex exec with prompt via stdin.

        Args:
            prompt: The prompt to send.
            session_id: Ignored for Codex (no native resume support).
                        For retries, prompt should include prior context.
        """
        # Accumulate prompts with clear delimiters for retry context
        if self._accumulated_prompt:
            self._accumulated_prompt += f"\n\n---\n\n[FOLLOW-UP]\n{prompt}"
        else:
            self._accumulated_prompt = prompt

        cmd = [
            "codex", "exec",
            "--json",
            "--dangerously-bypass-approvals-and-sandbox",
            "-C", self.options.cwd,
        ]
        if self.options.thinking_mode:
            cmd.extend(["-c", f"model_reasoning_effort={self.options.thinking_mode}"])
        cmd.append("-")  # Read from stdin

        self._process = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self.options.env,
        )
        # Write prompt to stdin and close
        assert self._process.stdin is not None
        self._process.stdin.write(self._accumulated_prompt.encode())
        await self._process.stdin.drain()
        self._process.stdin.close()

    async def receive_response(self) -> AsyncIterator[object]:
        """Stream SDK-compatible messages from Codex JSONL output.

        Skips non-JSON header lines gracefully.
        Also captures stderr for diagnostics and error classification.
        """
        if self._process is None:
            return
        if self._process.stdout is None:
            return

        # Start stderr reader task
        stderr_task = asyncio.create_task(self._read_stderr())

        text_buffer = ""

        try:
            async for line in self._process.stdout:
                line_str = line.decode().strip()
                if not line_str:
                    continue

                # Skip non-JSON header lines (e.g., "session id: ...", separators)
                if not line_str.startswith("{"):
                    continue

                try:
                    event = json.loads(line_str)
                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue

                if event.get("type") == "message_start":
                    self._session_id = event.get("session_id")

                elif event.get("type") == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text_buffer += delta.get("text", "")

                elif event.get("type") == "tool_use":
                    # Flush text buffer first
                    if text_buffer:
                        yield AssistantMessage(content=[TextBlock(text=text_buffer)])
                        text_buffer = ""
                    yield AssistantMessage(content=[
                        ToolUseBlock(id=event["id"], name=event["name"], input=event["input"])
                    ])

                elif event.get("type") == "tool_result":
                    yield AssistantMessage(content=[
                        ToolResultBlock(tool_use_id=event["tool_use_id"], content=event["content"])
                    ])

                elif event.get("type") == "message_stop":
                    if text_buffer:
                        yield AssistantMessage(content=[TextBlock(text=text_buffer)])
                        text_buffer = ""
                    yield ResultMessage(
                        session_id=self._session_id or f"codex-{uuid.uuid4().hex[:8]}",
                        result=None,  # Text already yielded above
                    )
                    return  # Clean exit

            # EOF reached without message_stop - fallback handling
            await self._process.wait()

            # Emit any remaining buffered text
            if text_buffer:
                yield AssistantMessage(content=[TextBlock(text=text_buffer)])

            # Always emit ResultMessage on EOF (success or failure)
            yield ResultMessage(
                session_id=self._session_id or f"codex-{uuid.uuid4().hex[:8]}",
                result=None,
            )

            # Check exit code and raise if non-zero
            if self._process.returncode != 0:
                raise CodexProcessError(
                    f"Codex exited with code {self._process.returncode}",
                    returncode=self._process.returncode,
                    stderr=self._stderr_output,
                )
        finally:
            await stderr_task


    async def _read_stderr(self) -> None:
        """Read stderr into buffer for diagnostics."""
        if self._process is None or self._process.stderr is None:
            return
        stderr_bytes = await self._process.stderr.read()
        self._stderr_output = stderr_bytes.decode(errors="replace")

    def is_auth_error(self) -> bool:
        """Check if the last error was an authentication failure.

        Returns True if stderr contains auth-related error patterns.
        """
        stderr_lower = self._stderr_output.lower()
        auth_patterns = [
            "authentication failed",
            "unauthorized",
            "invalid api key",
            "api key",
            "401",
            "auth error",
            "not authenticated",
        ]
        return any(pattern in stderr_lower for pattern in auth_patterns)

    def get_stderr(self) -> str:
        """Return captured stderr for logging/diagnostics."""
        return self._stderr_output


class CodexProcessError(Exception):
    """Raised when Codex process exits with non-zero code."""
    def __init__(self, message: str, returncode: int, stderr: str):
        super().__init__(message)
        self.returncode = returncode
        self.stderr = stderr
```

## User Experience

### Primary Flow

1. User runs `mala run --provider codex`
2. CLI validates `codex` command exists (upfront check before any work)
3. Orchestrator spawns agents using `CodexClient` instead of `ClaudeSDKClient`
4. Each issue:
   - CodexClient spawns `codex exec --json -` subprocess
   - Writes prompt to stdin (no arg length limits)
   - Streams JSONL events from stdout (skipping header lines), captures stderr
   - **Skips `WAIT_FOR_LOG`** (no session log file for Codex)
   - Quality gate runs with `check_validation_evidence=False` (commit checks only)
   - Codex review runs (unchanged - separate Codex invocation)
   - Success/failure determined by gate and review outcomes

### Error Handling

| Error Type | Detection | Behavior | Rationale |
|------------|-----------|----------|-----------|
| `codex` CLI not found | Upfront `which codex` check | Fatal: exit program | Pre-flight check fails |
| Codex auth error | `client.is_auth_error()` returns True | Fatal: exit program | All issues would fail anyway |
| Codex process exits non-zero | `CodexProcessError` raised after EOF | Fail issue, continue run | Per-issue transient failure |
| Codex JSONL parse error | `json.JSONDecodeError` on JSON lines | Skip line, continue parsing | Graceful degradation |
| Codex timeout | `asyncio.TimeoutError` | Fail issue, continue run | Existing timeout mechanism |
| EOF without message_stop | stdout ends without `message_stop` event | Emit fallback `ResultMessage`, check exit code | Graceful handling of crashes/truncation |

**Key distinction**: "Run fails" (exit program) vs "Issue fails" (continue with other issues).

**Stderr handling**: All stderr is captured via `client.get_stderr()` and logged for diagnostics. Auth errors are detected by pattern matching in stderr.

### Retry Behavior (No Session Resume)

Codex does not support session resume like Claude SDK. For gate/review retries:

1. **Context preservation**: `CodexClient` accumulates all prompts sent via `query()` in `_accumulated_prompt`
2. **Retry prompt**: Follow-up prompts (gate failures, review issues) are appended to accumulated context
3. **Fresh process**: Each `query()` spawns a new `codex exec` process with full accumulated prompt via stdin
4. **Implication**: Retry prompts are longer but have full context. This matches Claude's resumed session behavior semantically.

Example retry flow:
```
query(initial_prompt)         -> stdin: "initial_prompt"
query(gate_failure_followup)  -> stdin: "initial_prompt\n\n---\n\n[FOLLOW-UP]\ngate_failure_followup"
query(review_issues_followup) -> stdin: "initial_prompt\n\n---\n\n[FOLLOW-UP]\ngate_failure_followup\n\n---\n\n[FOLLOW-UP]\nreview_issues_followup"
```

The `---` separator and `[FOLLOW-UP]` header help Codex distinguish between the original task and subsequent feedback.

### Edge Cases

- **Synthetic session ID**: Codex provides session_id in `message_start`. If missing, generate `codex-{uuid}`.
- **Hook enforcement**: Skipped for Codex (Non-Goal). Trust Codex's file operations.
- **Non-JSON header lines**: Parser skips lines not starting with `{` (Codex outputs headers before JSONL).
- **Validation evidence**: MVP skips entirely. Gate checks commit existence only.

## Implementation Plan

1. [ ] Add upfront CLI validation: check `codex` command exists when `--provider codex` is used (fail fast)
2. [ ] Create `CodexClientOptions` dataclass in `src/codex_client.py`
3. [ ] Implement `CodexClient` class conforming to `SDKClientProtocol`
   - `__aenter__`/`__aexit__` for process lifecycle
   - `query()` to spawn `codex exec --json -` subprocess, write prompt to stdin
   - `receive_response()` to stream JSONL from stdout (skip non-JSON headers), capture stderr
   - `is_auth_error()` and `get_stderr()` for error classification and diagnostics
4. [ ] Add `provider` field to `AgentSessionConfig`
5. [ ] Add `--provider` CLI flag in `src/cli.py` (choices: claude, codex)
6. [ ] Modify `AgentSessionRunner.run_session()` to branch on provider:
   - **Claude path**: Build `ClaudeAgentOptions` with hooks, MCP servers, etc. Instantiate `ClaudeSDKClient`.
   - **Codex path**: Build `CodexClientOptions` (cwd, env, thinking_mode only). Instantiate `CodexClient`. Skip hook setup.
7. [ ] Modify `AgentSessionRunner` lifecycle handling for Codex:
   - Skip `WAIT_FOR_LOG` effect when provider is Codex
   - Pass `check_validation_evidence=False` to gate check callback when provider is Codex
8. [ ] Add provider to `MalaConfig` and thread through to orchestrator
9. [ ] Update orchestrator to pass provider config to `AgentSessionConfig`
10. [ ] Add unit tests for CodexClient (mocked subprocess, JSONL parsing with headers, auth error detection)
11. [ ] Add integration test for Codex provider (mocked CLI, verify log skip, verify gate behavior)

## Testing Strategy

- **Unit tests**:
  - `CodexClient` JSONL parsing with sample Codex output including header lines
  - Non-JSON line skipping (verify graceful handling)
  - Message mapping (Codex events -> SDK types)
  - Prompt accumulation for retry context
  - stdin piping (verify prompt written correctly)
  - Auth error detection from various stderr patterns
  - Provider selection logic

- **Integration tests**:
  - End-to-end run with mocked Codex CLI
  - Verify `WAIT_FOR_LOG` is skipped for Codex provider
  - Verify gate runs with `check_validation_evidence=False`
  - Verify gate/review retry flow with accumulated prompts
  - Verify auth error causes fatal exit

- **Manual testing**:
  - Run `mala run --provider codex` on a real issue
  - Verify Codex produces commits that pass quality gate
  - Verify separate Codex review still runs

## Decisions Made

- **`--provider` flag (not `--model`)**: Avoids confusion with model selection within a provider (e.g., `codex -m gpt-5`)
- **Codex CLI with `--json`**: Enables streaming JSONL parsing for SDK message compatibility
- **stdin piping for prompts**: Avoids OS command-line argument length limits that would break large/retry prompts
- **Skip non-JSON header lines**: Codex outputs headers before JSONL; parser skips lines not starting with `{`
- **Prompt accumulation with delimiters**: Use `---` separator and `[FOLLOW-UP]` header to help Codex distinguish retry context
- **Tool events are observational**: Codex executes tools internally; `tool_use`/`tool_result` events are for logging only, not execution
- **EOF fallback handling**: If `message_stop` is missing, emit `ResultMessage` anyway and check exit code
- **stderr capture and auth detection**: Enables fatal vs per-issue error classification and provides diagnostics
- **Skip `WAIT_FOR_LOG` for Codex**: Codex logs are in different location; MVP doesn't need them
- **Disable validation evidence for Codex**: Gate checks commit existence only; avoids log parsing dependency
- **Branch options construction**: Build `CodexClientOptions` vs `ClaudeAgentOptions` based on provider
- **No hooks**: User accepted no hook support for Codex to simplify MVP
- **No fallback**: Fatal errors exit program; per-issue errors continue run with remaining issues
- **Keep review separate**: Existing Codex review pipeline remains unchanged
- **Reuse prompts**: Same prompt templates work for both Claude and Codex
- **Upfront CLI check**: Validate `codex` command exists before starting any work

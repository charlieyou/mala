# Idle Timeout with Retry/Recovery

**Date**: 2025-12-31

## Context & Goals

### Problem Statement

Mala agent sessions can hang indefinitely when the Claude CLI subprocess gets stuck in a state where:
- The subprocess is alive (not crashed)
- The subprocess is not producing stdout output
- The subprocess is not closing its stdout pipe
- No `ResultMessage` is sent to signal completion

This was observed in run `bd0a654e` where session `90f81fe7` stopped producing output at 19:58:12 but the subprocess (PID 2449131) remained alive in `ep_poll` state indefinitely.

### Root Cause

The Claude CLI subprocess failed to send the expected completion signals:
1. No `ResultMessage` sent via SDK stdout to indicate response completion
2. Subprocess stayed alive with stdout pipe open but idle
3. The SDK's `receive_response()` iterator blocks indefinitely waiting for `ResultMessage`

### Current Mitigation (Idle Timeout)

The existing `idle_timeout_seconds` mechanism wraps `stream.__anext__()` with `asyncio.wait_for()`:

```python
msg = await asyncio.wait_for(stream.__anext__(), timeout=idle_timeout_seconds)
```

**The detection works correctly.** The problems are:
1. Subprocess is **not terminated** - remains alive, leaking resources
2. No retry/recovery - session just fails

### Goal

Enhance the existing idle timeout to terminate and **recover** when subprocess hangs:
1. Keep the current idle timeout detection (it works)
2. **Always** call `client.disconnect()` when `IdleTimeoutError` occurs (even if retry not possible)
3. Create a new subprocess and resume the session with the same context
4. Retry up to N times before failing

## Scope & Non-Goals

### In Scope
- Add retry/recovery logic for idle timeout
- Add `disconnect()` to `SDKClientProtocol` and `FakeSDKClient`
- Add config options for retry behavior (internal-only)
- Unit tests with mocked time

### Non-Goals
- Separate watchdog abstraction (deferred)
- Changes to the SDK itself
- Changes to the lifecycle state machine
- Exposing retry config to CLI/env vars

## Assumptions & Constraints

1. **SDK resume works**: The SDK's `session_id` parameter in `query()` loads full conversation context
2. **disconnect() is available**: Verified `ClaudeSDKClient.disconnect()` exists (client.py:362-367)
3. **Signature match**: SDK's `disconnect()` is `async def disconnect(self) -> None` (verified)
4. **Protocol update required**: `SDKClientProtocol` must be extended with `disconnect()` method

## Prerequisites

Before starting implementation:
1. Verify SDK client.py has `disconnect()` method - **VERIFIED** (line 362: `async def disconnect(self) -> None`)
2. Verify `query(session_id=)` supports resumption - **VERIFIED** (line 102)

## High-Level Approach

### The Fix

When idle timeout fires, instead of failing immediately:

1. **ALWAYS kill** the stuck subprocess via `client.disconnect()` (with 10s timeout) - even if retry is not possible
2. **Check** if retry is possible (retries remaining, no side effects without session context)
3. **Backoff** before retry (0s → 5s → 15s)
4. **Resume or restart**: If session_id exists, resume with context. If not AND no side effects occurred, start fresh.
5. **Retry** up to `max_idle_retries` times (default: 2)

**Key implementation insight**: The retry wrapper ONLY runs when a query needs to be sent. Non-query lifecycle phases (WAIT_FOR_LOG, RUN_GATE, RUN_REVIEW) do NOT trigger message iteration.

**First-turn hang handling**: If idle timeout occurs before any `ResultMessage`:
- If **no tool calls occurred**: Start fresh session (safe - no side effects)
- If **tool calls occurred**: Fail fast (side effects may have happened without context to resume)

## Detailed Implementation Tasks

### Task 1: Update SDKClientProtocol with disconnect()
**File**: `src/pipeline/agent_session_runner.py`
**Depends on**: None
**Complexity**: Low
**Lines changed**: ~5

Add `disconnect()` method to the protocol:

```python
@runtime_checkable
class SDKClientProtocol(Protocol):
    # ... existing methods (lines 80-117) ...

    async def disconnect(self) -> None:
        """Disconnect and terminate the subprocess.

        This should be called when an idle timeout is detected to ensure
        the hung subprocess is terminated before creating a new client.
        """
        ...
```

**Verification**: Type check passes (`uvx ty check`)

**Rollback**: Remove the method from protocol

---

### Task 2: Add config options (internal-only)
**File**: `src/pipeline/agent_session_runner.py`
**Depends on**: None
**Complexity**: Low
**Lines changed**: ~10

Add to `AgentSessionConfig` (after line 188):
```python
max_idle_retries: int = 2  # Retry up to 2 times on idle timeout
idle_retry_backoff: tuple[float, ...] = (0.0, 5.0, 15.0)  # Backoff per retry attempt
```

**Note on backoff semantics**: `idle_retry_count` tracks retries that have occurred:
- `idle_retry_count = 0`: First attempt (not a retry), no backoff
- `idle_retry_count = 1`: First retry, backoff index 0 → 0s
- `idle_retry_count = 2`: Second retry, backoff index 1 → 5s
- `idle_retry_count = 3+`: Third+ retry, backoff index 2 → 15s (capped)

**Verification**: Config can be set and read, env var override works

**Rollback**: Remove config fields and env var check

---

### Task 3: Implement idle timeout retry logic
**File**: `src/pipeline/agent_session_runner.py`
**Depends on**: Task 1, Task 2
**Complexity**: High
**Lines changed**: ~130

**Key insight**: The retry wrapper ONLY runs when `pending_query` is set. After successful message iteration, `pending_query` is cleared. Non-query phases (WAIT_FOR_LOG, RUN_GATE, RUN_REVIEW) do NOT set `pending_query`, so no query is sent.

**Current code structure** (lines 460-829):
```python
try:
    async with asyncio.timeout(self.config.timeout_seconds):  # Session timeout
        client = create_client()  # Once
        async with client:  # Single context
            lifecycle.start()
            await client.query(input.prompt)  # Initial query

            while not lifecycle.is_terminal:
                async for msg in _iter_messages():  # Message iteration
                    ...
                result = lifecycle.on_messages_complete(...)
                # Lifecycle transitions (WAIT_FOR_LOG, RUN_GATE, etc.)
                if result.effect == Effect.SEND_GATE_RETRY:
                    await client.query(followup, session_id)  # Gate retry query
                    continue
```

**Proposed structure**: Track `pending_query` and only query when set:

```python
# Add constant near existing REVIEW_FOLLOWUP_FILE (line 66)
IDLE_RESUME_PROMPT_FILE = _PROMPT_DIR / "idle_resume.md"

@functools.cache  # functools already imported at line 21
def _get_idle_resume_prompt() -> str:
    """Load idle resume prompt (cached on first use)."""
    return IDLE_RESUME_PROMPT_FILE.read_text()

# Constant for disconnect timeout
DISCONNECT_TIMEOUT = 10.0  # seconds

# Inside run_session(), restructure the main logic:

try:
    async with asyncio.timeout(self.config.timeout_seconds):  # Session timeout covers ALL retries
        lifecycle.start()
        if self.event_sink is not None:
            self.event_sink.on_lifecycle_state(input.issue_id, lifecycle.state.name)

        # Track state for query + message iteration
        # pending_query is Optional - None means no query needed this iteration
        pending_query: str | None = input.prompt
        pending_session_id: str | None = None
        idle_retry_count: int = 0
        tool_calls_this_turn: int = 0  # Track side effects for safe retry

        while not lifecycle.is_terminal:
            # === QUERY + MESSAGE ITERATION (only if pending_query is set) ===
            if pending_query is not None:
                message_iteration_complete = False
                tool_calls_this_turn = 0  # Reset for new query cycle

                while not message_iteration_complete:
                    # Backoff before retry (not on first attempt)
                    if idle_retry_count > 0:
                        backoff_idx = min(idle_retry_count - 1, len(self.config.idle_retry_backoff) - 1)
                        backoff = self.config.idle_retry_backoff[backoff_idx]
                        if backoff > 0:
                            logger.info(f"Idle retry {idle_retry_count}: waiting {backoff}s")
                            await asyncio.sleep(backoff)

                    # Create client for this attempt
                    if self.sdk_client_factory is not None:
                        client = self.sdk_client_factory.create(options)
                    else:
                        client = ClaudeSDKClient(options=options)

                    try:
                        async with client:
                            # Send query
                            await client.query(pending_query, session_id=pending_session_id)

                            # Define _iter_messages (captures client from scope)
                            async def _iter_messages() -> AsyncIterator[Any]:
                                stream = client.receive_response()
                                if idle_timeout_seconds is None:
                                    async for msg in stream:
                                        yield msg
                                    return
                                while True:
                                    try:
                                        msg = await asyncio.wait_for(
                                            stream.__anext__(),
                                            timeout=idle_timeout_seconds,
                                        )
                                    except StopAsyncIteration:
                                        break
                                    except TimeoutError as exc:
                                        raise IdleTimeoutError(
                                            f"SDK stream idle for {idle_timeout_seconds:.0f} seconds"
                                        ) from exc
                                    yield msg

                            try:
                                async for message in _iter_messages():
                                    # ... existing message processing (unchanged) ...

                                    # Track tool calls for side effect detection
                                    if isinstance(message, AssistantMessage):
                                        for block in message.content:
                                            if isinstance(block, ToolUseBlock):
                                                tool_calls_this_turn += 1

                                    if isinstance(message, ResultMessage):
                                        session_id = message.session_id
                                        lifecycle_ctx.session_id = session_id

                                # Success! Clear pending_query and exit retry loop
                                pending_query = None  # CRITICAL: Clear to prevent re-query
                                idle_retry_count = 0
                                message_iteration_complete = True

                            except IdleTimeoutError:
                                # === CRITICAL: Always disconnect first ===
                                logger.warning(
                                    f"Session {input.issue_id}: idle timeout, disconnecting subprocess"
                                )
                                try:
                                    await asyncio.wait_for(
                                        client.disconnect(),
                                        timeout=DISCONNECT_TIMEOUT
                                    )
                                except asyncio.TimeoutError:
                                    logger.warning("disconnect() timed out, subprocess abandoned")
                                except Exception as e:
                                    logger.debug(f"Error during disconnect: {e}")

                                # Check if we can retry
                                if idle_retry_count >= self.config.max_idle_retries:
                                    logger.error(
                                        f"Session {input.issue_id}: max idle retries "
                                        f"({self.config.max_idle_retries}) exceeded"
                                    )
                                    raise IdleTimeoutError(
                                        f"Max idle retries ({self.config.max_idle_retries}) exceeded"
                                    ) from None

                                # Prepare for retry
                                idle_retry_count += 1

                                # Determine resume strategy
                                resume_id = session_id or lifecycle_ctx.session_id
                                if resume_id is not None:
                                    # Have session context - resume with minimal prompt
                                    pending_session_id = resume_id
                                    pending_query = _get_idle_resume_prompt().format(
                                        issue_id=input.issue_id,
                                    )
                                    logger.info(
                                        f"Session {input.issue_id}: retrying with resume "
                                        f"(session_id={resume_id[:8]}..., attempt {idle_retry_count})"
                                    )
                                elif tool_calls_this_turn == 0:
                                    # No session context AND no side effects - safe to start fresh
                                    pending_session_id = None
                                    # Keep original pending_query (don't replace)
                                    logger.info(
                                        f"Session {input.issue_id}: retrying with fresh session "
                                        f"(no session_id, no side effects, attempt {idle_retry_count})"
                                    )
                                else:
                                    # No session context BUT side effects occurred - unsafe to retry
                                    logger.error(
                                        f"Session {input.issue_id}: cannot retry - "
                                        f"{tool_calls_this_turn} tool calls occurred without session_id"
                                    )
                                    raise IdleTimeoutError(
                                        f"Cannot retry: {tool_calls_this_turn} tool calls occurred "
                                        "without session context"
                                    ) from None

                                # Loop continues to retry

                    except IdleTimeoutError:
                        # Re-raise if we got here (means we gave up)
                        raise

            # === END QUERY + MESSAGE ITERATION ===

            # Lifecycle transitions (unchanged from current code)
            result = lifecycle.on_messages_complete(lifecycle_ctx, has_session_id=bool(session_id))
            # ... rest of lifecycle handling (WAIT_FOR_LOG, RUN_GATE, RUN_REVIEW, etc.) ...

            # If a followup query is needed, set pending_query for next iteration
            if result.effect == Effect.SEND_GATE_RETRY:
                # Build followup prompt (existing code)
                failure_text = "\n".join(f"- {r}" for r in gate_result.failure_reasons)
                pending_query = _get_gate_followup_prompt().format(...)
                pending_session_id = session_id
                idle_retry_count = 0  # Reset for new query cycle
                continue

            if result.effect == Effect.SEND_REVIEW_RETRY:
                # Build followup prompt (existing code)
                pending_query = _get_review_followup_prompt().format(...)
                pending_session_id = session_id
                idle_retry_count = 0  # Reset for new query cycle
                continue

            # For non-query phases (WAIT_FOR_LOG, RUN_GATE, RUN_REVIEW),
            # pending_query remains None, so next iteration skips query block
```

**Key implementation details**:
1. **Only query when pending_query is set**: The query block is gated by `if pending_query is not None`
2. **Clear pending_query after success**: Set to `None` after successful iteration
3. **Session timeout wraps all retries**: The outer `asyncio.timeout()` is authoritative
4. **Always disconnect on idle timeout**: Called BEFORE checking retry limits
5. **Track tool calls for side effects**: Fail fast if side effects occurred without session context
6. **Minimal resume prompt**: Just "Continue on issue {issue_id}" - SDK loads full context via session_id
7. **Uses Effect.SEND_GATE_RETRY**: Matches existing code style (line 641)

**Verification**:
- Unit test: Client hangs once then succeeds, verify retry works
- Unit test: Client hangs > max_retries times, verify gives up
- Unit test: First-turn hang with no tool calls, verify fresh session started
- Unit test: First-turn hang WITH tool calls, verify fail fast

**Rollback**: Revert to single-client structure

---

### Task 4: Add idle_resume.md prompt file
**File**: `src/prompts/idle_resume.md`
**Depends on**: None
**Complexity**: Low
**Lines changed**: ~10

Create minimal prompt file:
```markdown
Continue on issue {issue_id}.
```

**Verification**: File exists and is loadable

**Rollback**: Delete file

---

### Task 5: Update FakeSDKClient with disconnect()
**File**: `tests/test_agent_session_runner.py`
**Depends on**: Task 1
**Complexity**: Low
**Lines changed**: ~30

**Note**: Implement this BEFORE Task 3 to avoid intermediate type errors.

Add `disconnect()` to all fake client classes:

```python
class FakeSDKClient:
    def __init__(self, ...):
        # ... existing ...
        self.disconnect_called = False
        self.disconnect_delay: float = 0  # For testing timeout

    async def disconnect(self) -> None:
        """Disconnect the fake client."""
        if self.disconnect_delay > 0:
            await asyncio.sleep(self.disconnect_delay)
        self.disconnect_called = True

class HangingSDKClient(FakeSDKClient):
    # Inherits disconnect() from FakeSDKClient

class SlowSDKClient(FakeSDKClient):
    # Inherits disconnect() from FakeSDKClient
```

**Add SequencedSDKClientFactory** for retry tests (test infrastructure only):
```python
class SequencedSDKClientFactory:
    """Factory that returns different clients per create() call."""

    def __init__(self, clients: list[FakeSDKClient]):
        self.clients = clients
        self.create_calls: list[Any] = []
        self._index = 0

    def create(self, options: object) -> SDKClientProtocol:
        self.create_calls.append(options)
        client = self.clients[min(self._index, len(self.clients) - 1)]
        self._index += 1
        return client
```

**Verification**: Protocol conformance check passes

**Rollback**: Remove disconnect() method and SequencedSDKClientFactory

---

### Task 6: Add unit tests for idle timeout retry
**File**: `tests/test_agent_session_runner.py`
**Depends on**: Task 3, Task 5
**Complexity**: Medium
**Lines changed**: ~180

**Test strategy**:
- Use very short `idle_timeout_seconds` (0.01s) with `HangingSDKClient`
- Mock only `asyncio.sleep` for backoff delays (NOT `asyncio.wait_for`)
- Use `SequencedSDKClientFactory` to return different clients per attempt
- Mark all tests with `@pytest.mark.unit`

**Test cases:**

1. **`test_idle_timeout_retries_and_recovers`**:
   - Factory returns: [HangingClient (yields session_id then hangs), SuccessClient]
   - Verify: `disconnect()` called on first client
   - Verify: Session completes successfully with resume prompt

2. **`test_idle_timeout_gives_up_after_max_retries`**:
   - Factory always returns HangingClient
   - Verify: `IdleTimeoutError` raised after 3 attempts
   - Verify: `disconnect()` called on ALL clients (no leaks)

3. **`test_idle_timeout_always_disconnects_even_on_final_failure`**:
   - Verify: disconnect() called even when max retries exceeded

4. **`test_idle_timeout_first_turn_no_side_effects_starts_fresh`**:
   - Hang before any messages (no tool calls, no session_id)
   - Verify: Retry uses original prompt, not resume prompt

5. **`test_idle_timeout_first_turn_with_tool_calls_fails_fast`**:
   - Hang after tool calls but before ResultMessage
   - Verify: `IdleTimeoutError` raised with "tool calls occurred" message

6. **`test_idle_timeout_non_query_phases_skip_query_block`**:
   - Simulate WAIT_FOR_LOG phase (pending_query is None)
   - Verify: No client created, no query sent

7. **`test_idle_timeout_backoff_delays`**:
   - Track calls to mocked `asyncio.sleep`
   - Verify: Delays are 0s (retry 1), 5s (retry 2)

**Verification**: `uv run pytest tests/test_agent_session_runner.py -k idle_timeout -v -m unit`

**Rollback**: Delete tests

---

### Task 7: Update documentation
**File**: This file
**Depends on**: Task 3
**Complexity**: Low

Document:
- `max_idle_retries` config option (internal-only, default: 2)
- Side effect detection behavior

**Rollback**: Revert docs

## Risks, Edge Cases & Breaking Changes

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| `disconnect()` hangs | Low | Medium | Wrap with 10s timeout, abandon if times out |
| Resume loses context | Low | High | SDK `session_id` loads full conversation history |
| Retry loop creates infinite retries | Low | High | Hard cap at `max_idle_retries` (default 2) |
| Side effects without session context | Medium | High | Track tool calls, fail fast if side effects detected |
| Agent confused after resume | Medium | Low | SDK loads full context via session_id |

### Edge Cases

1. **First-turn hang, no side effects**: Start fresh session (safe).

2. **First-turn hang, tool calls occurred**: Fail fast (unsafe to retry without context).

3. **Later hang with session_id**: Resume with minimal prompt; SDK loads full context.

4. **Non-query lifecycle phases**: `pending_query` is None, query block is skipped entirely.

5. **disconnect() times out**: Log warning, proceed with retry anyway.

6. **Session timeout during backoff**: Expected behavior - session timeout is authoritative.

### Breaking Changes

**Behavioral change**: Sessions that previously failed immediately on idle timeout will now retry up to 2 times. This is the intended improvement.

## Testing & Validation

### Unit Tests (Task 6)
Tests map to risky areas:
- Retry success → validates recovery works
- Max retries exceeded → validates we don't loop forever
- Always disconnect → validates no resource leaks
- First-turn with tool calls → validates side effect protection
- Non-query phases → validates correct loop structure

### Manual Validation
```bash
# Run unit tests
uv run pytest tests/test_agent_session_runner.py -k idle_timeout -v -m unit

# Full test suite
uv run pytest -m "unit or integration"

# Type check
uvx ty check
```

## Plan-Level Rollback Strategy

If issues are discovered after deployment:

1. **Immediate**: Set `max_idle_retries=0` in code and deploy
2. **Short-term**: Revert tasks in reverse order (7 → 1)
3. **Verification**: Run full test suite after revert

## Open Questions

None - all P1 issues from reviews have been addressed:
1. ✅ Always disconnect on idle timeout
2. ✅ Loop structure only queries when pending_query is set
3. ✅ First-turn hang with side effects → fail fast
4. ✅ Minimal resume prompt (SDK loads context via session_id)

## References

- Incident analysis: Run `bd0a654e`, session `90f81fe7` (2025-12-31)
- Claude Agent SDK disconnect: `.venv/lib/python3.14/site-packages/claude_agent_sdk/client.py:362`
- SDK query with session_id: `.venv/lib/python3.14/site-packages/claude_agent_sdk/client.py:102`
- Current idle timeout: `src/pipeline/agent_session_runner.py:482-501`
- AgentSessionConfig: `src/pipeline/agent_session_runner.py:159-188`
- IdleTimeoutError: `src/pipeline/agent_session_runner.py:75-76`
- Effect enum usage: `src/pipeline/agent_session_runner.py:641` (uses `Effect.SEND_GATE_RETRY`)

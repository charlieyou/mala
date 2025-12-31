# Subprocess Watchdog Implementation Plan

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
1. No `stop_reason: "end_turn"` on the final assistant message (all messages had `stop_reason: null`)
2. No `ResultMessage` sent to indicate response completion
3. Subprocess stayed alive with stdout pipe open but idle

### Current Mitigation (Idle Timeout)

The existing `idle_timeout_seconds` mechanism wraps `stream.__anext__()` with `asyncio.wait_for()`:

```python
msg = await asyncio.wait_for(stream.__anext__(), timeout=idle_timeout_seconds)
```

**Problems with this approach:**
1. Default timeout is 12 minutes (20% of 60-min session timeout, clamped to 300-900s)
2. Operates at SDK message level, not subprocess level
3. When timeout fires, raises `IdleTimeoutError` but **does not terminate the subprocess**
4. The subprocess can remain alive as a zombie, leaking resources

### Goal

Replace the idle timeout with a subprocess watchdog that:
1. Monitors the Claude CLI subprocess directly by PID
2. Terminates the subprocess gracefully when idle too long
3. Ensures proper cleanup of all resources

## Scope & Non-Goals

### In Scope
- Create `SubprocessWatchdog` class for external process monitoring
- Integrate watchdog into `AgentSessionRunner.run_session()`
- Add `watchdog_timeout_seconds` and `watchdog_enabled` config options
- Phased deprecation of `idle_timeout_seconds`
- Logging and metrics for watchdog events

### Non-Goals
- Modifying the Claude Agent SDK itself
- Detecting why the subprocess got stuck (that's a CLI bug to investigate separately)
- Auto-recovery or retry logic after watchdog kills a session

## Assumptions & Constraints

### Assumptions
1. **SDK PID Access (Verified)**: The Claude Agent SDK exposes the subprocess PID via `client._transport._process.pid`. This is accessible through the private `_transport` attribute of `ClaudeSDKClient`, which holds a `SubprocessCLITransport` instance with a `_process: anyio.abc.Process` property.
2. We are running on Linux (production) or macOS (local dev)
3. The subprocess being idle (no stdout writes) for 3+ minutes indicates a stuck state

### Constraints
1. **Architectural Contract**: The watchdog module will be placed in `src/pipeline/` which:
   - Cannot import from CLI, orchestration, or domain layers
   - Cannot import external SDKs directly (must use protocols)
   - Must remain acyclic with other pipeline modules
2. The watchdog cannot use `psutil` due to the "SDK confined to infra" contract (pipeline cannot add new external SDK dependencies)
3. Must use portable process monitoring (`os.kill(pid, 0)`) that works on both Linux and macOS

## Prerequisites

Before implementation:

1. **Verify PID Access**: Write a quick e2e test to confirm we can access `client._transport._process.pid` after `client.connect()`:
   ```python
   async with ClaudeSDKClient(options) as client:
       pid = client._transport._process.pid
       assert pid is not None and pid > 0
   ```
   - **If this fails**: Fall back to enhanced activity tracking with explicit transport termination

2. **Verify Process Monitoring Works**: Confirm `os.kill(pid, 0)` works for detecting process existence on the target platform.

## High-Level Approach

**Primary: External Process Monitor (Option C)** with a concrete **Activity Tracking Fallback (Option B)**.

### Why Option C is Preferred

Option C directly monitors the subprocess via PID and `os.kill(pid, 0)`, allowing us to:
1. Kill the subprocess explicitly when stuck (current idle timeout cannot do this)
2. Detect stuck state even if the SDK stream iterator itself is broken
3. Release file handles, locks, and other resources held by the subprocess

### Concrete Fallback (Option B) When PID Unavailable

When PID access fails (e.g., SDK update breaks `_transport._process.pid`), the fallback will:
1. Create an activity-tracking async task that monitors message timestamps
2. When timeout is reached, **explicitly call `client.disconnect()`** which triggers `SubprocessCLITransport.close()`
3. The transport's `close()` method calls `process.terminate()` on the subprocess (verified in SDK source)
4. This differs from current idle_timeout which just raises an exception without termination

The key difference from current idle_timeout: Option B actively terminates the transport, not just raising an exception.

### Design

```
┌─────────────────────────────────────────────────────────────┐
│                    AgentSessionRunner                        │
│                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────┐ │
│  │ ClaudeSDK    │────▶│ Message Loop │────▶│ Lifecycle   │ │
│  │ Client       │     │              │     │ Manager     │ │
│  └──────────────┘     └──────────────┘     └─────────────┘ │
│         │                    │                     │        │
│         │                    ▼                     │        │
│         │           ┌─────────────────┐            │        │
│         └──────────▶│ SubprocessWatch │◀───────────┘        │
│             (pid)   │ dog (TaskGroup) │  (cancels on        │
│                     │                 │   lifecycle end)    │
│                     └─────────────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

### Watchdog Behavior

1. **Start**: After SDK `client.connect()`, extract PID from `client._transport._process.pid`
2. **Monitor**: Run async task using `os.kill(pid, 0)` every `poll_interval` seconds (portable across Linux/macOS)
3. **Activity Detection**: Track last observed stdout activity timestamp (updated when messages received)
4. **Timeout**: If no activity for `watchdog_timeout_seconds`:
   - Send SIGTERM to process, wait 5s grace period
   - Send SIGKILL if still running
   - Set cancellation flag for session loop
5. **Cleanup**: Watchdog task is cancelled when lifecycle reaches terminal state or session ends normally

### Stream Integration Details

When the watchdog kills the subprocess:
1. The subprocess receives SIGTERM → SIGKILL
2. The subprocess stdout pipe closes
3. `SubprocessCLITransport._read_messages_impl()` reaches end of async iterator
4. `client.receive_response()` stops yielding messages
5. The `async for message in _iter_messages():` loop exits
6. Lifecycle transitions to terminal state
7. Watchdog cancellation is detected and `SubprocessWatchdogTimeout` is raised

### Timeout Propagation Pattern

The watchdog runs in a `TaskGroup` with the message processing loop. When the watchdog detects a timeout:

```python
async with asyncio.TaskGroup() as tg:
    watchdog_task = tg.create_task(watchdog.monitor())
    try:
        async for message in _iter_messages():
            watchdog.notify_activity()
            # ... process message ...
    except* SubprocessWatchdogTimeout:
        # Watchdog killed the process
        raise
```

Using `TaskGroup` ensures exceptions from the watchdog task are properly propagated to the main loop, unlike standalone `create_task()` which would log unhandled exceptions but not propagate them.

## Detailed Tasks

### Task 1: Create SubprocessWatchdog class
**File**: `src/pipeline/watchdog.py` (new file)
**Dependencies**: None
**Estimated complexity**: Medium

Create the watchdog class using portable `os.kill(pid, 0)`:

```python
"""Subprocess watchdog for detecting and killing stuck Claude CLI processes."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class SubprocessWatchdogTimeout(Exception):
    """Raised when subprocess is killed due to inactivity."""
    pass


@dataclass
class WatchdogConfig:
    """Configuration for subprocess watchdog."""
    timeout_seconds: float = 180.0  # 3 minutes
    poll_interval_seconds: float = 10.0
    kill_grace_seconds: float = 5.0


class SubprocessWatchdog:
    """Monitors a subprocess and kills it if idle too long.

    Uses os.kill(pid, 0) for portable process existence checking
    across Linux and macOS.
    """

    def __init__(self, pid: int, config: WatchdogConfig | None = None):
        self.pid = pid
        self.config = config or WatchdogConfig()
        self._last_activity = asyncio.get_event_loop().time()
        self._stopped = False

    def notify_activity(self) -> None:
        """Call this when subprocess produces output."""
        self._last_activity = asyncio.get_event_loop().time()

    def stop(self) -> None:
        """Signal the watchdog to stop monitoring."""
        self._stopped = True

    async def monitor(self) -> None:
        """Monitor loop - checks process and kills if idle.

        Raises SubprocessWatchdogTimeout when subprocess is killed.
        This method is designed to run in a TaskGroup alongside the
        message processing loop, ensuring proper exception propagation.
        """
        while not self._stopped:
            await asyncio.sleep(self.config.poll_interval_seconds)

            if not self._is_process_alive():
                return  # Process exited normally

            idle_time = asyncio.get_event_loop().time() - self._last_activity
            if idle_time > self.config.timeout_seconds:
                await self._terminate_process()
                raise SubprocessWatchdogTimeout(
                    f"Subprocess {self.pid} idle for {idle_time:.0f}s, killed"
                )

    def _is_process_alive(self) -> bool:
        """Check if process is still running using os.kill(pid, 0).

        This is portable across Linux and macOS, unlike /proc which
        is Linux-only.
        """
        try:
            os.kill(self.pid, 0)
            return True
        except ProcessLookupError:
            return False  # Process doesn't exist
        except PermissionError:
            return True  # Process exists but we can't signal it

    async def _terminate_process(self) -> None:
        """Gracefully terminate, then force kill subprocess."""
        logger.warning(
            f"Watchdog killing subprocess {self.pid} after "
            f"{self.config.timeout_seconds}s idle"
        )

        try:
            os.kill(self.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return  # Already dead

        # Wait grace period for graceful shutdown
        for _ in range(int(self.config.kill_grace_seconds * 10)):
            await asyncio.sleep(0.1)
            if not self._is_process_alive():
                logger.info(f"Subprocess {self.pid} terminated gracefully")
                return

        # Force kill
        try:
            os.kill(self.pid, signal.SIGKILL)
            logger.warning(f"Subprocess {self.pid} required SIGKILL")
        except (ProcessLookupError, PermissionError):
            pass
```

**Verification**:
- Unit test: Watchdog kills spawned subprocess after timeout
- Unit test: Watchdog resets timer on `notify_activity()`
- Unit test: Watchdog sends SIGTERM then SIGKILL
- Unit test: Watchdog handles already-dead process gracefully

**Rollback**: Delete `src/pipeline/watchdog.py`

---

### Task 2: Add config options
**File**: `src/pipeline/agent_session_runner.py` (AgentSessionConfig at line 159)
**Dependencies**: None
**Estimated complexity**: Low

Add to `AgentSessionConfig`:

```python
@dataclass
class AgentSessionConfig:
    """Configuration for agent session execution."""

    # ... existing fields ...

    # Watchdog configuration (Phase 1: alongside idle_timeout)
    watchdog_enabled: bool = True
    watchdog_timeout_seconds: float = 180.0  # 3 minutes
    watchdog_poll_interval: float = 10.0
    watchdog_kill_grace_seconds: float = 5.0

    # Existing idle_timeout (deprecated in Phase 2, removed in Phase 3)
    idle_timeout_seconds: float | None = None
```

**Verification**: Import test passes, config can be instantiated

**Rollback**: Revert changes to `src/pipeline/agent_session_runner.py`

---

### Task 3: Add SDKClientProtocol.get_subprocess_pid() method
**File**: `src/pipeline/agent_session_runner.py` (SDKClientProtocol at line 80)
**Dependencies**: None
**Estimated complexity**: Low

Extend `SDKClientProtocol` with optional PID access:

```python
class SDKClientProtocol(Protocol):
    """Protocol for SDK client interactions."""

    # ... existing methods ...

    def get_subprocess_pid(self) -> int | None:
        """Return the subprocess PID if available, None otherwise.

        This is optional - implementations may return None if they don't
        have subprocess access (e.g., test fakes).
        """
        ...
```

The real `ClaudeSDKClient` doesn't implement this directly, so we'll wrap access in the integration:

```python
def _get_pid_from_client(client: SDKClientProtocol) -> int | None:
    """Extract subprocess PID from client, with fallback for protocol types."""
    # Try protocol method first (for test fakes)
    if hasattr(client, 'get_subprocess_pid'):
        try:
            return client.get_subprocess_pid()
        except (AttributeError, NotImplementedError):
            pass

    # Try direct SDK access (for real ClaudeSDKClient)
    try:
        return client._transport._process.pid  # type: ignore[attr-defined]
    except AttributeError:
        return None
```

**Verification**: Protocol still type-checks, existing tests pass

**Rollback**: Revert protocol changes

---

### Task 4: Integrate watchdog into AgentSessionRunner
**File**: `src/pipeline/agent_session_runner.py`
**Dependencies**: Tasks 1, 2, 3
**Estimated complexity**: High

Modify `run_session()` to use TaskGroup for proper exception propagation:

```python
# After client connect, extract PID
pid = _get_pid_from_client(client)
watchdog: SubprocessWatchdog | None = None

if pid and self.config.watchdog_enabled:
    watchdog = SubprocessWatchdog(
        pid,
        WatchdogConfig(
            timeout_seconds=self.config.watchdog_timeout_seconds,
            poll_interval_seconds=self.config.watchdog_poll_interval,
            kill_grace_seconds=self.config.watchdog_kill_grace_seconds,
        ),
    )
    logger.info(f"Watchdog started for PID {pid}, timeout={self.config.watchdog_timeout_seconds}s")
elif self.config.watchdog_enabled:
    logger.warning("Could not access subprocess PID, watchdog disabled")

try:
    if watchdog:
        # Use TaskGroup for proper exception propagation
        async with asyncio.TaskGroup() as tg:
            tg.create_task(watchdog.monitor())
            try:
                async for message in _iter_messages():
                    watchdog.notify_activity()
                    # ... existing message processing ...
            finally:
                watchdog.stop()
    else:
        # No watchdog - use existing loop
        async for message in _iter_messages():
            # ... existing message processing ...
except* SubprocessWatchdogTimeout as eg:
    # Extract the single exception from the group
    raise eg.exceptions[0] from None
```

**Verification**:
- Integration test with real subprocess that hangs
- Verify subprocess is killed and session terminates
- Verify logs show watchdog timeout

**Rollback**: Revert changes to `agent_session_runner.py`

---

### Task 5: Add deprecation warning for idle_timeout_seconds
**File**: `src/pipeline/agent_session_runner.py` (around line 453-458)
**Dependencies**: Task 2
**Estimated complexity**: Low

Add deprecation warning where `idle_timeout_seconds` is processed:

```python
idle_timeout_seconds = self.config.idle_timeout_seconds
if idle_timeout_seconds is not None:
    logger.warning(
        "idle_timeout_seconds is deprecated and will be removed. "
        "Use watchdog_timeout_seconds instead."
    )
    derived = self.config.timeout_seconds * 0.2
    idle_timeout_seconds = min(900.0, max(300.0, derived))
# ... rest of idle_timeout logic ...
```

Add environment variable support for watchdog in `MalaConfig.from_env()`:

```python
# In src/config.py, add to from_env():
# Parse watchdog settings
watchdog_enabled_raw = os.environ.get("MALA_WATCHDOG_ENABLED", "").lower()
watchdog_enabled = watchdog_enabled_raw not in ("0", "false", "no", "off")

watchdog_timeout_raw = os.environ.get("MALA_WATCHDOG_TIMEOUT")
watchdog_timeout = None
if watchdog_timeout_raw:
    try:
        watchdog_timeout = float(watchdog_timeout_raw)
    except ValueError:
        parse_errors.append(f"MALA_WATCHDOG_TIMEOUT: invalid float '{watchdog_timeout_raw}'")
```

Note: No `mala.toml` parsing exists in the codebase, so watchdog config is env-var only.

**Verification**: Unit test config loading from env

**Rollback**: Revert changes

---

### Task 6: Add logging and metrics
**File**: `src/pipeline/agent_session_runner.py`
**Dependencies**: Task 4
**Estimated complexity**: Low

Add structured logging for watchdog events:
- `watchdog.started` with PID and config
- `watchdog.activity` (debug level) on each activity ping
- `watchdog.timeout` with idle duration
- `watchdog.killed` with SIGTERM/SIGKILL distinction

For Braintrust integration, log watchdog timeouts to the span:
```python
if tracer:
    tracer.log_event("watchdog_timeout", {"pid": pid, "idle_seconds": idle_time})
```

**Verification**: Run session, verify logs appear

**Rollback**: Revert logging changes

---

### Task 7: Add unit tests
**File**: `tests/test_watchdog.py` (new file)
**Dependencies**: Task 1
**Estimated complexity**: Medium

Test cases using real spawned subprocesses:

```python
import asyncio
import subprocess
import pytest
from src.pipeline.watchdog import SubprocessWatchdog, WatchdogConfig, SubprocessWatchdogTimeout


@pytest.mark.unit
async def test_watchdog_kills_idle_subprocess():
    """Watchdog kills a subprocess that stops producing output."""
    # Spawn a real subprocess that sleeps forever
    proc = subprocess.Popen(["sleep", "3600"])

    try:
        watchdog = SubprocessWatchdog(
            proc.pid,
            WatchdogConfig(timeout_seconds=0.5, poll_interval_seconds=0.1),
        )

        with pytest.raises(SubprocessWatchdogTimeout):
            await watchdog.monitor()

        # Verify process was killed
        assert proc.poll() is not None  # Process has exited
    finally:
        # Cleanup in case test failed
        proc.kill()
        proc.wait()


@pytest.mark.unit
async def test_watchdog_resets_on_activity():
    """Watchdog timer resets when activity is notified."""
    proc = subprocess.Popen(["sleep", "3600"])

    try:
        watchdog = SubprocessWatchdog(
            proc.pid,
            WatchdogConfig(timeout_seconds=0.5, poll_interval_seconds=0.1),
        )

        # Keep notifying activity
        async def keep_alive():
            for _ in range(10):
                await asyncio.sleep(0.1)
                watchdog.notify_activity()
            watchdog.stop()

        # Both tasks run - watchdog should not timeout
        await asyncio.gather(watchdog.monitor(), keep_alive())

        # Process should still be alive (we stopped watchdog, not killed process)
        assert proc.poll() is None
    finally:
        proc.kill()
        proc.wait()


@pytest.mark.unit
async def test_watchdog_handles_already_dead_process():
    """Watchdog exits gracefully if process is already dead."""
    proc = subprocess.Popen(["true"])  # Exits immediately
    proc.wait()  # Wait for it to exit

    watchdog = SubprocessWatchdog(
        proc.pid,
        WatchdogConfig(timeout_seconds=1.0, poll_interval_seconds=0.1),
    )

    # Should exit without raising
    await watchdog.monitor()


@pytest.mark.unit
def test_get_pid_from_client_with_fake():
    """Test PID extraction from fake client."""
    class FakeClient:
        def get_subprocess_pid(self) -> int | None:
            return 12345

    from src.pipeline.agent_session_runner import _get_pid_from_client
    assert _get_pid_from_client(FakeClient()) == 12345


@pytest.mark.unit
def test_get_pid_from_client_returns_none_when_unavailable():
    """Test PID extraction returns None for clients without PID access."""
    class FakeClient:
        pass

    from src.pipeline.agent_session_runner import _get_pid_from_client
    assert _get_pid_from_client(FakeClient()) is None
```

**Verification**: `pytest tests/test_watchdog.py -v`

**Rollback**: Delete test file (no production impact)

---

### Task 8: Add integration test
**File**: `tests/test_agent_session_runner.py`
**Dependencies**: Tasks 4, 7
**Estimated complexity**: Medium

Add test using a subprocess that actually hangs:

```python
import asyncio
import subprocess
import pytest


class HangingSDKClient:
    """Fake SDK client that simulates a stuck subprocess."""

    def __init__(self):
        # Spawn a real subprocess that sleeps
        self._proc = subprocess.Popen(["sleep", "3600"])

    def get_subprocess_pid(self) -> int:
        return self._proc.pid

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self._proc.kill()
        self._proc.wait()

    async def query(self, prompt: str) -> None:
        pass

    async def receive_response(self):
        # Yield one message then hang forever
        yield AssistantMessage(content=[TextBlock(text="Starting...")])
        await asyncio.sleep(float('inf'))  # Simulate stuck


@pytest.mark.integration
async def test_watchdog_kills_stuck_session():
    """Verify watchdog kills session when subprocess is stuck."""
    config = AgentSessionConfig(
        repo_path=Path("/tmp"),
        timeout_seconds=300,
        watchdog_enabled=True,
        watchdog_timeout_seconds=1.0,  # Short timeout for test
        watchdog_poll_interval=0.1,
    )

    factory = FakeSDKClientFactory(HangingSDKClient)
    runner = AgentSessionRunner(config, sdk_client_factory=factory)

    input = AgentSessionInput(issue_id="test-1", prompt="test")

    with pytest.raises(SubprocessWatchdogTimeout):
        await runner.run_session(input)

    # Verify the subprocess was killed
    assert factory.last_client._proc.poll() is not None
```

**Verification**: `pytest tests/test_agent_session_runner.py::test_watchdog_kills_stuck_session -v`

**Rollback**: Revert test additions

---

### Task 9: Update pipeline exports
**File**: `src/pipeline/__init__.py`
**Dependencies**: Task 1
**Estimated complexity**: Low

Add exports for watchdog module:

```python
from src.pipeline.watchdog import (
    SubprocessWatchdog,
    SubprocessWatchdogTimeout,
    WatchdogConfig,
)

__all__ = [
    # ... existing exports ...
    "SubprocessWatchdog",
    "SubprocessWatchdogTimeout",
    "WatchdogConfig",
]
```

**Verification**: Import test passes

**Rollback**: Revert exports

---

### Task 10: Update documentation
**File**: `README.md` or `docs/configuration.md` (if exists)
**Dependencies**: Task 5
**Estimated complexity**: Low

Document:
- `MALA_WATCHDOG_ENABLED` and `MALA_WATCHDOG_TIMEOUT` environment variables
- Migration path from `idle_timeout_seconds`
- How to disable watchdog in emergency

**Verification**: Documentation renders correctly

**Rollback**: Revert docs

## Risks, Edge Cases & Breaking Changes

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Watchdog kills healthy session | Low | High | Conservative 3min default, activity tracking resets timer on every message |
| PID access breaks with SDK update | Medium | Medium | Fallback detection + warning log if PID unavailable |
| `os.kill(pid, 0)` fails unexpectedly | Low | Medium | Catch PermissionError, treat as "process exists" |
| Race between normal completion and watchdog | Low | Low | Stop watchdog before processing final result |

### Edge Cases

1. **Very long tool execution**: A legitimate tool call (e.g., running tests) may take >3 minutes. The watchdog only fires if no *messages* are received. If the CLI sends progress messages, the timer resets. If truly silent, 3 minutes is still reasonable.

2. **Multiple concurrent sessions**: Each session has its own watchdog with its own PID. No shared state.

3. **Process already dead**: Watchdog uses `os.kill(pid, 0)` which raises `ProcessLookupError` for dead processes. Handles gracefully.

4. **Zombie processes**: On Linux, a process becomes a zombie after termination until reaped. `os.kill(pid, 0)` still succeeds for zombies, so the graceful termination loop may wait for the full grace period. This is acceptable (just slightly inefficient).

5. **macOS development**: `os.kill(pid, 0)` is portable and works on macOS, unlike `/proc` which is Linux-only.

### Breaking Changes

- **Phase 2 (after one release)**: `idle_timeout_seconds` will log deprecation warnings
- **Phase 3 (after two releases)**: `idle_timeout_seconds` config option will be removed

Config migration path:
```yaml
# Before (deprecated)
idle_timeout_seconds: 600

# After (environment variable)
MALA_WATCHDOG_TIMEOUT=180
```

## Testing & Validation

### Test Strategy Matrix

| Area | Test Type | Coverage |
|------|-----------|----------|
| Watchdog core logic | Unit | Task 7 |
| PID extraction (real + fake) | Unit | Task 7 |
| Process termination (real subprocess) | Unit | Task 7 |
| Fallback when no PID | Unit | Task 7 |
| Full session with watchdog | Integration | Task 8 |
| Stuck subprocess killed | Integration | Task 8 |
| Config loading from env | Unit | Task 5 |

### Verification Commands

```bash
# Unit tests
uv run pytest tests/test_watchdog.py -v

# Integration tests
uv run pytest tests/test_agent_session_runner.py -k watchdog -v

# Full test suite
uv run pytest -m "unit or integration"
```

## Rollout Plan

### Phase 1: Add Watchdog Alongside Idle Timeout (This PR)
- Add watchdog as new mechanism
- Keep idle_timeout active as backup
- Both can fire; watchdog takes precedence in cleanup
- **Feature flag**: `MALA_WATCHDOG_ENABLED` (default: enabled)
- **How to disable in emergency**: Set `MALA_WATCHDOG_ENABLED=false`
- **Metrics to watch**:
  - Count of `watchdog.timeout` log events
  - Count of `idle_timeout.fired` log events
  - Compare: watchdog should fire before idle timeout in stuck cases

### Phase 2: Deprecate Idle Timeout (Next Release)
- Log deprecation warning when `idle_timeout_seconds` is set
- Increase idle_timeout default to 2x watchdog (backup only)
- **Rollback**: Revert to Phase 1 by removing deprecation warnings

### Phase 3: Remove Idle Timeout (Release After)
- Remove `idle_timeout_seconds` config option
- Remove idle timeout code from `_iter_messages()`
- **Rollback**: Re-add idle_timeout code if watchdog proves unreliable

## Plan-Level Rollback Strategy

If the watchdog feature causes issues after deployment:

1. **Immediate mitigation**: Set `MALA_WATCHDOG_ENABLED=false` in environment
2. **Rollback PR**: Revert all commits from this implementation
3. **Fallback behavior**: System reverts to idle_timeout mechanism (still present in Phase 1)

To identify watchdog issues vs SDK bugs:
- Watchdog kills are logged with: `Watchdog killing subprocess {pid} after {seconds}s idle`
- SDK/CLI bugs would show different error patterns (crashes, exceptions)
- Check session logs for last message received before watchdog fired

## Open Questions

*All questions resolved during planning:*

1. ~~Should we report stuck subprocess incidents to telemetry?~~ → Yes, log to Braintrust span
2. ~~What's the right default timeout?~~ → 180s (3 minutes), justified below
3. ~~Should watchdog timeout be configurable via CLI/env?~~ → Yes, via `MALA_WATCHDOG_TIMEOUT`
4. ~~Do we need to enhance claude-agent-sdk to expose subprocess PID?~~ → No, already accessible via `client._transport._process.pid`
5. ~~How to handle macOS (no /proc)?~~ → Use portable `os.kill(pid, 0)` instead
6. ~~How to propagate watchdog exceptions to main loop?~~ → Use `asyncio.TaskGroup`

### Justification for 180s Default Timeout

The 180s (3 minute) default is based on:
1. **Observed incident data**: Run `bd0a654e` showed subprocess hung indefinitely with no messages
2. **Normal message frequency**: Active Claude sessions typically produce messages every few seconds
3. **Conservative buffer**: Even a slow tool execution that produces no progress messages rarely takes >3 minutes of complete silence
4. **Comparison to idle_timeout**: Current idle_timeout is 300-900s (5-15 minutes), but that's for message-level timeout. Subprocess-level monitoring can be more aggressive since we're detecting true hangs, not just slow responses.

## References

- Incident analysis: Run `bd0a654e`, session `90f81fe7` (2025-12-31)
- Claude Agent SDK: `.venv/lib/python3.14/site-packages/claude_agent_sdk/_internal/transport/subprocess_cli.py`
- SDK client: `.venv/lib/python3.14/site-packages/claude_agent_sdk/client.py`
- Current idle timeout: `src/pipeline/agent_session_runner.py:453-500`
- Existing graceful termination pattern: `src/tools/command_runner.py:238-280`
- AgentSessionConfig: `src/pipeline/agent_session_runner.py:159-188`
- SDKClientProtocol: `src/pipeline/agent_session_runner.py:80-100`

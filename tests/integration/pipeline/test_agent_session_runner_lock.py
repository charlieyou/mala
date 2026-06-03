"""Integration tests for the lock park/resume loop in execute_iteration.

These tests exercise :meth:`IdleTimeoutRetryPolicy.execute_iteration`'s
between-turn lock park/resume behavior: when a Claude agent yields on contended
file locks via the ``lock_wait`` MCP tool, the policy keeps the SDK client
connected, parks the session until the contended paths free (or a deadline),
resumes the agent on the *same* client to re-acquire and finalize, and only then
returns. The gate runs once at the true end.

A self-contained fake :class:`~src.core.protocols.sdk.SDKClientProtocol` drives
the scenarios so no real SDK subprocess is spawned, and a fake
:class:`~src.pipeline.idle_retry_policy.LockWaitProbe` decides which paths remain
blocked so no real lock files are touched. The fakes resolve instantly, so the
real ``asyncio.timeout`` budgets never fire without patching the clock.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import pytest

from src.core.protocols.agent_event import (
    AgentResultEvent,
    AgentToolResultEvent,
    AgentToolUseEvent,
)
from src.domain.lifecycle import LifecycleContext
from src.domain.validation.config_types import LockWaitConfig
from src.infra.io.base_sink import BaseEventSink
from src.pipeline import idle_retry_policy
from src.pipeline.idle_retry_policy import (
    IdleTimeoutRetryPolicy,
    RetryConfig,
)
from src.pipeline.message_stream_processor import (
    MessageIterationState,
    MessageStreamProcessor,
    StreamProcessorCallbacks,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence
    from typing import Self

    from src.core.protocols.sdk import SDKClientFactoryProtocol
    from src.pipeline.idle_retry_policy import IterationResult
    from src.pipeline.message_stream_processor import LintCacheProtocol


_LOCK_WAIT_TOOL_USE_ID = "toolu_lock_wait"
_LOCK_WAIT_TOOL = "mcp__mala-locking__lock_wait"
_WAIT_PATH = "/repo/hot_file.py"


class _FakeLintCache:
    """Minimal lint cache: detects nothing, records nothing."""

    def detect_lint_command(self, command: str) -> str | None:
        del command
        return None

    def mark_success(self, lint_type: str, command: str) -> None:
        del lint_type, command


def _park_turn_events(
    wait_paths: Sequence[str] = (_WAIT_PATH,),
    tool_use_id: str = _LOCK_WAIT_TOOL_USE_ID,
) -> list[object]:
    """A turn that parks on contended locks via ``lock_wait`` and ends ok.

    Emits the ``lock_wait`` tool_use plus a ``parked: true`` tool_result so the
    stream processor populates ``pending_lock_waits`` exactly as a real run does.
    """
    return [
        AgentToolUseEvent(
            id=tool_use_id,
            name=_LOCK_WAIT_TOOL,
            input={"filepaths": list(wait_paths)},
        ),
        AgentToolResultEvent(
            tool_use_id=tool_use_id,
            content=json.dumps({"parked": True, "wait_paths": list(wait_paths)}),
        ),
        AgentResultEvent(session_id="sess-1", subtype="success"),
    ]


def _finalize_turn_events() -> list[object]:
    """A normal resume turn that ends with no further lock park."""
    return [
        AgentResultEvent(session_id="sess-1", subtype="success"),
    ]


@dataclass
class _FakeSDKClient:
    """Fake SDK client implementing the lock park/resume surface.

    ``per_turn_responses`` is a queue of message lists; each ``query`` consumes
    the next list, which ``receive_response`` then yields.
    """

    per_turn_responses: list[list[object]]
    supports_bg: bool = True
    queries: list[str] = field(default_factory=list)
    _current_turn: list[object] = field(default_factory=list)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def query(self, prompt: str, session_id: str | None = None) -> None:
        del session_id
        self.queries.append(prompt)
        self._current_turn = (
            self.per_turn_responses.pop(0) if self.per_turn_responses else []
        )

    async def receive_response(self) -> AsyncIterator[object]:
        for message in self._current_turn:
            yield message

    def supports_background_tasks(self) -> bool:
        return self.supports_bg

    async def receive_messages(self) -> AsyncIterator[object]:
        # Not used by the lock loop (it polls via the probe), but part of the
        # protocol surface the background loop touches.
        if False:  # pragma: no cover - never yields
            yield None

    async def stop_task(self, task_id: str) -> None:
        del task_id

    async def disconnect(self) -> None:
        return None


@dataclass
class _FakeClientFactory:
    """Returns a pre-built fake client; records resume tokens."""

    client: _FakeSDKClient
    resume_calls: list[str | None] = field(default_factory=list)

    def create(self, runtime: object) -> _FakeSDKClient:
        del runtime
        return self.client

    def with_resume(self, runtime: object, resume: str | None) -> object:
        self.resume_calls.append(resume)
        return runtime


@dataclass
class _FakeLockWaitProbe:
    """Stateful probe: reports blocked until ``free_after`` polls elapse.

    ``free_after=0`` frees immediately; a large value never frees within the
    deadline. ``poll_count`` records how many times the loop probed.
    """

    blocked: list[str]
    free_after: int = 0
    poll_count: int = 0
    raise_on_poll: int | None = None

    def blocked_paths(self, canonical_paths: Sequence[str]) -> list[str]:
        if self.raise_on_poll is not None and self.poll_count == self.raise_on_poll:
            raise asyncio.CancelledError
        self.poll_count += 1
        if self.poll_count > self.free_after:
            return []
        return [p for p in canonical_paths if p in self.blocked]


def _make_policy(
    client: _FakeSDKClient,
    *,
    event_sink: BaseEventSink | None = None,
) -> IdleTimeoutRetryPolicy:
    factory = _FakeClientFactory(client=client)

    def _stream_processor_factory() -> MessageStreamProcessor:
        return MessageStreamProcessor(callbacks=StreamProcessorCallbacks())

    return IdleTimeoutRetryPolicy(
        sdk_client_factory=cast("SDKClientFactoryProtocol", factory),
        stream_processor_factory=_stream_processor_factory,
        config=RetryConfig(max_idle_retries=0),
        event_sink=event_sink,
    )


async def _run(
    client: _FakeSDKClient,
    *,
    lock_wait: LockWaitConfig | None,
    probe: _FakeLockWaitProbe | None,
    drain_event: asyncio.Event | None = None,
    event_sink: BaseEventSink | None = None,
    state: MessageIterationState | None = None,
) -> IterationResult:
    policy = _make_policy(client, event_sink=event_sink)
    return await policy.execute_iteration(
        query="initial prompt",
        issue_id="ISSUE-1",
        runtime=object(),
        state=state if state is not None else MessageIterationState(),
        lifecycle_ctx=LifecycleContext(),
        lint_cache=cast("LintCacheProtocol", _FakeLintCache()),
        idle_timeout_seconds=None,
        lock_wait=lock_wait,
        lock_resume_template=("resume {issue_id}: status={status} paths={wait_paths}"),
        lock_wait_probe=cast("idle_retry_policy.LockWaitProbe", probe),
        hard_timeout_seconds=300.0,
        drain_event=drain_event,
    )


@pytest.mark.asyncio
async def test_locks_free_mid_wait_resumes_with_status_free() -> None:
    """Locks freeing within the budget triggers one resume with status ``free``."""
    client = _FakeSDKClient(
        per_turn_responses=[_park_turn_events(), _finalize_turn_events()],
    )
    probe = _FakeLockWaitProbe(blocked=[_WAIT_PATH], free_after=1)
    result = await _run(
        client,
        lock_wait=LockWaitConfig(poll_interval_ms=1),
        probe=probe,
    )

    assert result.success is True
    # Two queries: the original park + one resume.
    assert len(client.queries) == 2
    assert "resume ISSUE-1" in client.queries[1]
    assert "status=free" in client.queries[1]
    assert _WAIT_PATH in client.queries[1]


@pytest.mark.asyncio
async def test_never_frees_resumes_with_status_unavailable() -> None:
    """Exhausting the budget still resumes the agent with status ``unavailable``."""
    client = _FakeSDKClient(
        per_turn_responses=[_park_turn_events(), _finalize_turn_events()],
    )
    # free_after far beyond the deadline -> never frees within max_wait_seconds.
    probe = _FakeLockWaitProbe(blocked=[_WAIT_PATH], free_after=10_000)
    result = await _run(
        client,
        # Tiny budget + tiny poll so the deadline elapses immediately.
        lock_wait=LockWaitConfig(max_wait_seconds=0, poll_interval_ms=1),
        probe=probe,
    )

    # Still resumes (to wrap up cleanly) with the unavailable branch.
    assert result.success is True
    assert len(client.queries) == 2
    assert "status=unavailable" in client.queries[1]


@pytest.mark.asyncio
async def test_drain_interrupts_wait_and_abandons_state() -> None:
    """A drain signal ends the park promptly and proceeds to the gate.

    With no fix the wait would spin until the budget elapsed; the drain event
    breaks it, no resume is issued, success is returned, and the parked state is
    cleared so a later gate-retry iteration does not re-wait.
    """
    client = _FakeSDKClient(per_turn_responses=[_park_turn_events()])
    # free_after far beyond the deadline -> the path never frees on its own, so
    # only the pre-set drain signal can end the park.
    probe = _FakeLockWaitProbe(blocked=[_WAIT_PATH], free_after=10_000)
    drain = asyncio.Event()
    drain.set()
    state = MessageIterationState()
    result = await _run(
        client,
        lock_wait=LockWaitConfig(max_wait_seconds=600, poll_interval_ms=1),
        probe=probe,
        drain_event=drain,
        state=state,
    )

    assert result.success is True  # proceeds to the gate
    assert len(client.queries) == 1  # no resume issued
    # The abandon path cleared the parked state for later gate-retry iterations.
    assert state.pending_lock_waits == set()
    assert state.lock_wait_request_ids == set()


@pytest.mark.asyncio
async def test_repark_runs_a_second_cycle() -> None:
    """A resume turn that parks again drives a second park/resume cycle."""
    second_path = "/repo/other_file.py"
    client = _FakeSDKClient(
        per_turn_responses=[
            _park_turn_events(),
            # The resume re-parks on a *different* contended file.
            _park_turn_events(
                wait_paths=(second_path,), tool_use_id="toolu_lock_wait_2"
            ),
            _finalize_turn_events(),
        ],
    )
    # Both paths are reported free on the first probe of each wait.
    probe = _FakeLockWaitProbe(blocked=[_WAIT_PATH, second_path], free_after=0)
    result = await _run(
        client,
        lock_wait=LockWaitConfig(max_resume_cycles=3, poll_interval_ms=1),
        probe=probe,
    )

    assert result.success is True
    # Original park + two resumes (one per cycle).
    assert len(client.queries) == 3
    assert "status=free" in client.queries[1]
    assert second_path in client.queries[2]


@pytest.mark.asyncio
async def test_max_resume_cycles_exhausted_proceeds_to_gate() -> None:
    """Re-parking every resume turn stops after max_resume_cycles as success."""
    client = _FakeSDKClient(
        # Every turn re-parks (distinct ids), so the loop keeps re-parking.
        per_turn_responses=[
            _park_turn_events(tool_use_id=f"toolu_lock_wait_{i}") for i in range(5)
        ],
    )
    probe = _FakeLockWaitProbe(blocked=[_WAIT_PATH], free_after=0)
    state = MessageIterationState()
    result = await _run(
        client,
        lock_wait=LockWaitConfig(max_resume_cycles=2, poll_interval_ms=1),
        probe=probe,
        state=state,
    )

    assert result.success is True
    # Original park + exactly max_resume_cycles resumes, then proceed.
    assert len(client.queries) == 3
    # Exhaustion abandons the still-parked state.
    assert state.pending_lock_waits == set()


@pytest.mark.asyncio
async def test_cancelled_error_propagates() -> None:
    """A deadlock-victim cancellation must propagate, not become success.

    The probe raises ``CancelledError`` on its first poll (simulating the monitor
    cancelling this parked task). The lock loop must let it propagate so the
    deadlock actually resolves, rather than swallowing it into a success result.
    """
    client = _FakeSDKClient(per_turn_responses=[_park_turn_events()])
    probe = _FakeLockWaitProbe(blocked=[_WAIT_PATH], free_after=10_000, raise_on_poll=0)

    with pytest.raises(asyncio.CancelledError):
        await _run(
            client,
            lock_wait=LockWaitConfig(poll_interval_ms=1),
            probe=probe,
        )

    # No resume was issued before the cancellation propagated.
    assert len(client.queries) == 1


@pytest.mark.asyncio
async def test_no_background_support_skips_lock_loop() -> None:
    """A client with supports_background_tasks()=False never enters the loop.

    Even with ``pending_lock_waits`` set by the park turn, an Amp/Codex-style
    one-shot client must not reach the same-client resume path: no second query,
    no probing.
    """
    client = _FakeSDKClient(
        per_turn_responses=[_park_turn_events()],
        supports_bg=False,
    )
    probe = _FakeLockWaitProbe(blocked=[_WAIT_PATH], free_after=0)
    result = await _run(
        client,
        lock_wait=LockWaitConfig(poll_interval_ms=1),
        probe=probe,
    )

    assert result.success is True
    # No resume issued and the probe was never consulted.
    assert len(client.queries) == 1
    assert probe.poll_count == 0


@pytest.mark.asyncio
async def test_lock_wait_disabled_skips_loop() -> None:
    """When lock_wait is disabled, the park/resume loop is skipped."""
    client = _FakeSDKClient(per_turn_responses=[_park_turn_events()])
    probe = _FakeLockWaitProbe(blocked=[_WAIT_PATH], free_after=0)
    result = await _run(
        client,
        lock_wait=LockWaitConfig(enabled=False, poll_interval_ms=1),
        probe=probe,
    )

    assert result.success is True
    assert len(client.queries) == 1
    assert probe.poll_count == 0


@pytest.mark.asyncio
async def test_no_probe_skips_loop() -> None:
    """A None probe (e.g. unconfigured) skips the loop even when enabled."""
    client = _FakeSDKClient(per_turn_responses=[_park_turn_events()])
    result = await _run(
        client,
        lock_wait=LockWaitConfig(poll_interval_ms=1),
        probe=None,
    )

    assert result.success is True
    assert len(client.queries) == 1


class _RecordingSink(BaseEventSink):
    """Event sink that records lock-wait progress emissions."""

    def __init__(self) -> None:
        self.waits: list[tuple[str, float, float, int]] = []

    def on_lock_wait(
        self,
        agent_id: str,
        elapsed_seconds: float,
        budget_seconds: float,
        blocked_count: int,
    ) -> None:
        self.waits.append((agent_id, elapsed_seconds, budget_seconds, blocked_count))


@pytest.mark.asyncio
async def test_progress_emitted_while_parked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The otherwise-silent park emits periodic progress on the event sink."""
    monkeypatch.setattr(idle_retry_policy, "BACKGROUND_WAIT_PROGRESS_INTERVAL", 0.01)
    client = _FakeSDKClient(
        per_turn_responses=[_park_turn_events(), _finalize_turn_events()],
    )
    # Stay blocked for a few polls so at least one progress tick fires.
    probe = _FakeLockWaitProbe(blocked=[_WAIT_PATH], free_after=5)
    sink = _RecordingSink()
    result = await _run(
        client,
        lock_wait=LockWaitConfig(max_wait_seconds=600, poll_interval_ms=5),
        probe=probe,
        event_sink=sink,
    )

    assert result.success is True
    assert sink.waits, "expected at least one lock-wait progress event"
    # blocked_count reflects the single contended path.
    assert sink.waits[0][3] == 1
    assert sink.waits[0][2] == 600

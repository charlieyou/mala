"""Integration tests for the background wait/resume loop in execute_iteration.

These tests exercise :meth:`IdleTimeoutRetryPolicy.execute_iteration`'s
keep-connected wait/resume behavior (Option S): when a Claude agent backgrounds
work via ``Bash(run_in_background=true)`` and yields, the policy keeps the SDK
client connected, waits for the task's completion notification on the continuous
``receive_messages()`` stream, resumes the agent on the *same* client to
finalize, and only then returns. The gate runs once at the true end (the runner
calls ``on_messages_complete`` after ``execute_iteration`` returns).

A self-contained fake :class:`~src.core.protocols.sdk.SDKClientProtocol` drives
the scenarios so no real SDK subprocess is spawned. The fakes yield finite
event streams and never block, so the real ``asyncio.timeout`` budgets never
fire — the waits resolve instantly without patching the clock.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import pytest

from src.core.protocols.agent_event import (
    AgentResultEvent,
    AgentTaskCompletedEvent,
    AgentTaskStartedEvent,
    AgentToolResultEvent,
    AgentToolUseEvent,
)
from src.domain.lifecycle import LifecycleContext
from src.domain.validation.config_types import LongRunningConfig
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
    from collections.abc import AsyncIterator
    from typing import Self

    from src.core.protocols.sdk import SDKClientFactoryProtocol
    from src.pipeline.idle_retry_policy import IterationResult
    from src.pipeline.message_stream_processor import LintCacheProtocol


_LAUNCH_TOOL_USE_ID = "toolu_bg_launch"
_TASK_ID = "task_123"


class _FakeLintCache:
    """Minimal lint cache: detects nothing, records nothing."""

    def detect_lint_command(self, command: str) -> str | None:
        del command
        return None

    def mark_success(self, lint_type: str, command: str) -> None:
        del lint_type, command


def _launch_turn_events(tool_use_id: str = _LAUNCH_TOOL_USE_ID) -> list[object]:
    """A turn that backgrounds a Bash command and ends successfully.

    ``tool_use_id`` defaults to the shared launch id; pass a distinct id when a
    scenario backgrounds work across several turns (real launches each get a
    unique tool-use id).
    """
    return [
        AgentToolUseEvent(
            id=tool_use_id,
            name="Bash",
            input={"command": "long_job.sh", "run_in_background": True},
        ),
        AgentResultEvent(session_id="sess-1", subtype="success"),
    ]


def _auto_background_turn_events(tool_use_id: str = _LAUNCH_TOOL_USE_ID) -> list[object]:
    """A Bash command that Claude auto-backgrounds after launch."""
    return [
        AgentToolUseEvent(
            id=tool_use_id,
            name="Bash",
            input={"command": "long_job.sh"},
        ),
        AgentTaskStartedEvent(task_id=_TASK_ID, tool_use_id=tool_use_id),
        AgentResultEvent(session_id="sess-1", subtype="success"),
    ]


def _auto_background_ack_turn_events(
    tool_use_id: str = _LAUNCH_TOOL_USE_ID,
) -> list[object]:
    """A Bash command whose tool result supplies the background task id."""
    return [
        AgentToolUseEvent(
            id=tool_use_id,
            name="Bash",
            input={"command": "long_job.sh"},
        ),
        AgentToolResultEvent(
            tool_use_id=tool_use_id,
            content=f"Command was auto-backgrounded with ID: {_TASK_ID}.",
        ),
        AgentResultEvent(session_id="sess-1", subtype="success"),
    ]


def _finalize_turn_events() -> list[object]:
    """A normal resume turn that ends with no further background launch."""
    return [
        AgentResultEvent(session_id="sess-1", subtype="success"),
    ]


@dataclass
class _FakeSDKClient:
    """Fake SDK client implementing the background-task surface.

    ``per_turn_responses`` is a queue of message lists; each ``query`` consumes
    the next list, which ``receive_response`` then yields. ``notifications`` is
    the list of messages ``receive_messages`` yields during the between-turn
    wait (e.g. an ``AgentTaskCompletedEvent``).
    """

    per_turn_responses: list[list[object]]
    notifications: list[object] = field(default_factory=list)
    supports_bg: bool = True
    queries: list[str] = field(default_factory=list)
    stopped_task_ids: list[str] = field(default_factory=list)
    receive_messages_calls: int = 0
    # When set, ``receive_messages`` blocks on this event before yielding —
    # simulating the real "completion notification never arrives" hang so a
    # drain signal is the only way out.
    block_event: asyncio.Event | None = None
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
        self.receive_messages_calls += 1
        if self.block_event is not None:
            await self.block_event.wait()
        for message in self.notifications:
            yield message

    async def stop_task(self, task_id: str) -> None:
        self.stopped_task_ids.append(task_id)

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


def _make_policy(
    client: _FakeSDKClient,
) -> tuple[IdleTimeoutRetryPolicy, _FakeClientFactory]:
    factory = _FakeClientFactory(client=client)

    def _stream_processor_factory() -> MessageStreamProcessor:
        return MessageStreamProcessor(callbacks=StreamProcessorCallbacks())

    policy = IdleTimeoutRetryPolicy(
        sdk_client_factory=cast("SDKClientFactoryProtocol", factory),
        stream_processor_factory=_stream_processor_factory,
        config=RetryConfig(max_idle_retries=0),
    )
    return policy, factory


async def _run(
    client: _FakeSDKClient,
    *,
    long_running: LongRunningConfig | None,
) -> IterationResult:
    policy, _factory = _make_policy(client)
    state = MessageIterationState()
    lifecycle_ctx = LifecycleContext()
    return await policy.execute_iteration(
        query="initial prompt",
        issue_id="ISSUE-1",
        runtime=object(),
        state=state,
        lifecycle_ctx=lifecycle_ctx,
        lint_cache=cast("LintCacheProtocol", _FakeLintCache()),
        idle_timeout_seconds=None,
        long_running=long_running,
        await_resume_template=(
            "resume {issue_id}: status={status} file={output_file} sum={summary}"
        ),
        hard_timeout_seconds=300.0,
    )


@pytest.mark.asyncio
async def test_completed_background_task_resumes_then_completes() -> None:
    """A completed task triggers exactly one resume, then a normal completion.

    The fakes yield finite event streams and never block, so the real
    ``asyncio.timeout`` budgets simply never fire — no patching needed.
    """
    client = _FakeSDKClient(
        per_turn_responses=[_launch_turn_events(), _finalize_turn_events()],
        notifications=[
            AgentTaskStartedEvent(task_id=_TASK_ID, tool_use_id=_LAUNCH_TOOL_USE_ID),
            AgentTaskCompletedEvent(
                task_id=_TASK_ID,
                tool_use_id=_LAUNCH_TOOL_USE_ID,
                status="completed",
                summary="exit code 0",
                output_file="/tmp/out.log",
            ),
        ],
    )
    result = await _run(client, long_running=LongRunningConfig())

    assert result.success is True
    # Two queries: the original launch + one resume.
    assert len(client.queries) == 2
    assert "resume ISSUE-1" in client.queries[1]
    assert "status=completed" in client.queries[1]
    assert "/tmp/out.log" in client.queries[1]
    # The wait read the continuous stream exactly once.
    assert client.receive_messages_calls == 1
    assert client.stopped_task_ids == []


@pytest.mark.asyncio
async def test_auto_backgrounded_task_resumes_before_success() -> None:
    """Claude auto-backgrounded Bash tasks wait/resume before validation can run."""
    client = _FakeSDKClient(
        per_turn_responses=[_auto_background_turn_events(), _finalize_turn_events()],
        notifications=[
            AgentTaskCompletedEvent(
                task_id=_TASK_ID,
                tool_use_id=_LAUNCH_TOOL_USE_ID,
                status="completed",
                summary="exit code 0",
                output_file="/tmp/out.log",
            ),
        ],
    )
    result = await _run(client, long_running=LongRunningConfig())

    assert result.success is True
    assert len(client.queries) == 2
    assert "resume ISSUE-1" in client.queries[1]
    assert client.receive_messages_calls == 1


@pytest.mark.asyncio
async def test_auto_background_ack_resumes_before_success() -> None:
    """Ack-only auto-background metadata also drives wait/resume."""
    client = _FakeSDKClient(
        per_turn_responses=[
            _auto_background_ack_turn_events(),
            _finalize_turn_events(),
        ],
        notifications=[
            AgentTaskCompletedEvent(
                task_id=_TASK_ID,
                tool_use_id=_LAUNCH_TOOL_USE_ID,
                status="completed",
                summary="exit code 0",
                output_file="/tmp/out.log",
            ),
        ],
    )
    result = await _run(client, long_running=LongRunningConfig())

    assert result.success is True
    assert len(client.queries) == 2
    assert "resume ISSUE-1" in client.queries[1]
    assert client.receive_messages_calls == 1


@pytest.mark.asyncio
async def test_failed_background_task_still_resumes() -> None:
    """A failed/stopped task still resumes the agent (to diagnose, not commit)."""
    client = _FakeSDKClient(
        per_turn_responses=[_launch_turn_events(), _finalize_turn_events()],
        notifications=[
            AgentTaskCompletedEvent(
                task_id=_TASK_ID,
                tool_use_id=_LAUNCH_TOOL_USE_ID,
                status="failed",
                summary="exit code 1",
                output_file="/tmp/out.log",
            ),
        ],
    )
    result = await _run(client, long_running=LongRunningConfig())

    # The agent is resumed even on failure; stop_task is NOT called (only a
    # wait timeout stops the task).
    assert len(client.queries) == 2
    assert "status=failed" in client.queries[1]
    assert client.stopped_task_ids == []
    # The resume turn ended successfully (the agent handled the failure).
    assert result.success is True


@pytest.mark.asyncio
async def test_wait_timeout_stops_task_then_proceeds_to_gate() -> None:
    """No completion notification -> stop the task and proceed to the gate.

    The continuous stream yields the ``task_started`` event (so the policy
    learns the ``task_id``) but never a matching ``task_completed`` and then
    ends. ``_wait_for_background_completion`` returns an outcome with no
    completion, which the policy treats like a ``max_wait`` deadline: it calls
    ``stop_task`` and returns success so the gate (the real arbiter) runs rather
    than failing the agent's otherwise-successful turn.
    """
    client = _FakeSDKClient(
        per_turn_responses=[_launch_turn_events()],
        notifications=[
            AgentTaskStartedEvent(task_id=_TASK_ID, tool_use_id=_LAUNCH_TOOL_USE_ID),
        ],
    )
    result = await _run(
        client,
        long_running=LongRunningConfig(max_wait_seconds=4),
    )

    # Proceeds to the gate as success (the gate re-validates), but still stops
    # the runaway task and issues no resume query.
    assert result.success is True
    assert client.stopped_task_ids == [_TASK_ID]
    assert len(client.queries) == 1


@pytest.mark.asyncio
async def test_max_resume_cycles_exhausted_proceeds_to_gate() -> None:
    """When every resume turn relaunches in background, proceed after max cycles.

    Each turn backgrounds a *new* launch (distinct tool-use id, as real
    launches do), so the pending set keeps refilling. After ``max_resume_cycles``
    resumes a launch is still pending; the policy best-effort stops the
    still-pending task and proceeds to the gate as success (the gate arbitrates)
    rather than failing the agent's successful turns.
    """
    launch_ids = ["bg-0", "bg-1", "bg-2"]

    def _launch_with_started(tid: str) -> list[object]:
        # Mirror a real launch turn: the tool_use plus the task_started event
        # the SDK emits (recorded into state.background_task_ids) so the still-
        # pending final launch can be stopped on exhaustion by its known id.
        return [
            AgentToolUseEvent(
                id=tid,
                name="Bash",
                input={"command": "long_job.sh", "run_in_background": True},
            ),
            AgentTaskStartedEvent(task_id=f"task-{tid}", tool_use_id=tid),
            AgentResultEvent(session_id="sess-1", subtype="success"),
        ]

    client = _FakeSDKClient(
        per_turn_responses=[_launch_with_started(tid) for tid in launch_ids],
        notifications=[
            AgentTaskCompletedEvent(
                task_id=f"task-{tid}",
                tool_use_id=tid,
                status="completed",
                summary="exit code 0",
                output_file="/tmp/out.log",
            )
            for tid in launch_ids
        ],
    )
    result = await _run(
        client,
        long_running=LongRunningConfig(max_resume_cycles=2),
    )

    # Original launch + exactly max_resume_cycles resumes, then proceed.
    assert len(client.queries) == 3
    assert result.success is True
    # The still-pending final launch (bg-2) was stopped via its known task id.
    assert "task-bg-2" in client.stopped_task_ids


@pytest.mark.asyncio
async def test_no_background_support_skips_wait() -> None:
    """Codex/Amp-style clients (supports_background_tasks False) never wait."""
    client = _FakeSDKClient(
        per_turn_responses=[_launch_turn_events()],
        notifications=[
            AgentTaskCompletedEvent(
                task_id=_TASK_ID,
                tool_use_id=_LAUNCH_TOOL_USE_ID,
                status="completed",
                summary="exit code 0",
                output_file="/tmp/out.log",
            ),
        ],
        supports_bg=False,
    )
    result = await _run(client, long_running=LongRunningConfig())

    assert result.success is True
    # No resume, no continuous-stream read.
    assert len(client.queries) == 1
    assert client.receive_messages_calls == 0
    assert client.stopped_task_ids == []


@pytest.mark.asyncio
async def test_abandoned_wait_clears_pending_background_state() -> None:
    """Giving up on a launch leaves no stale pending id in the shared state.

    Regression: ``MessageIterationState`` is reused across the gate-retry
    ``execute_iteration`` calls in ``_run_lifecycle_loop`` and is not reset at
    the top of ``execute_iteration``. A launch left in
    ``pending_background_tool_ids`` after a timeout/abandon would be re-reported
    by the next turn and force another wait for a completion that can never
    arrive on that turn's fresh client. The abandon paths must clear it.
    """
    client = _FakeSDKClient(
        per_turn_responses=[_launch_turn_events()],
        notifications=[
            AgentTaskStartedEvent(task_id=_TASK_ID, tool_use_id=_LAUNCH_TOOL_USE_ID),
        ],
    )
    policy, _factory = _make_policy(client)
    state = MessageIterationState()
    result = await policy.execute_iteration(
        query="initial prompt",
        issue_id="ISSUE-1",
        runtime=object(),
        state=state,
        lifecycle_ctx=LifecycleContext(),
        lint_cache=cast("LintCacheProtocol", _FakeLintCache()),
        idle_timeout_seconds=None,
        long_running=LongRunningConfig(max_wait_seconds=4),
        await_resume_template="resume {issue_id}",
        hard_timeout_seconds=300.0,
    )

    # Proceeds to the gate as success, and the shared state is clean so the next
    # iteration does not re-wait on the abandoned launch.
    assert result.success is True
    assert state.pending_background_tool_ids == set()
    assert state.background_task_ids == {}


class _RecordingSink(BaseEventSink):
    """Event sink that records background-wait progress emissions."""

    def __init__(self) -> None:
        self.waits: list[tuple[str, str, float, float]] = []

    def on_background_wait(
        self,
        agent_id: str,
        tool_use_id: str,
        elapsed_seconds: float,
        budget_seconds: float,
    ) -> None:
        self.waits.append((agent_id, tool_use_id, elapsed_seconds, budget_seconds))


@pytest.mark.asyncio
async def test_drain_interrupts_wait_and_emits_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The 1st Ctrl-C (drain) breaks the wait promptly; progress is emitted.

    ``receive_messages`` blocks forever (the real "notification never arrives"
    hang). With no fix the iteration would block until ``max_wait_seconds``.
    Here the wait emits periodic progress (so it is not silent) and a drain
    signal ends it, proceeding to the gate as success without a resume.
    """
    monkeypatch.setattr(idle_retry_policy, "BACKGROUND_WAIT_PROGRESS_INTERVAL", 0.02)
    drain = asyncio.Event()
    block = asyncio.Event()  # never set -> receive_messages blocks forever
    client = _FakeSDKClient(
        per_turn_responses=[_launch_turn_events()],
        notifications=[],
        block_event=block,
    )
    sink = _RecordingSink()
    factory = _FakeClientFactory(client=client)
    policy = IdleTimeoutRetryPolicy(
        sdk_client_factory=cast("SDKClientFactoryProtocol", factory),
        stream_processor_factory=lambda: MessageStreamProcessor(
            callbacks=StreamProcessorCallbacks()
        ),
        config=RetryConfig(max_idle_retries=0),
        event_sink=sink,
    )
    state = MessageIterationState()
    lifecycle_ctx = LifecycleContext()
    task = asyncio.ensure_future(
        policy.execute_iteration(
            query="initial prompt",
            issue_id="ISSUE-1",
            runtime=object(),
            state=state,
            lifecycle_ctx=lifecycle_ctx,
            lint_cache=cast("LintCacheProtocol", _FakeLintCache()),
            idle_timeout_seconds=None,
            long_running=LongRunningConfig(max_wait_seconds=600),
            await_resume_template="resume {issue_id}",
            hard_timeout_seconds=300.0,
            drain_event=drain,
        )
    )

    # Let the wait spin so at least one progress tick fires.
    for _ in range(50):
        if sink.waits:
            break
        await asyncio.sleep(0.02)

    assert sink.waits, "expected at least one background-wait progress event"
    assert sink.waits[0][1] == _LAUNCH_TOOL_USE_ID

    # First Ctrl-C (drain) must end the wait promptly.
    drain.set()
    result = await asyncio.wait_for(task, timeout=2.0)

    assert result.success is True  # proceeds to the gate
    assert len(client.queries) == 1  # no resume issued
    assert client.stopped_task_ids == []  # interrupt path leaves teardown to do it


@pytest.mark.asyncio
async def test_long_running_disabled_skips_wait() -> None:
    """When long_running is disabled, the wait/resume loop is skipped."""
    client = _FakeSDKClient(
        per_turn_responses=[_launch_turn_events()],
        notifications=[
            AgentTaskCompletedEvent(
                task_id=_TASK_ID,
                tool_use_id=_LAUNCH_TOOL_USE_ID,
                status="completed",
                summary="exit code 0",
                output_file="/tmp/out.log",
            ),
        ],
    )
    result = await _run(
        client,
        long_running=LongRunningConfig(enabled=False),
    )

    assert result.success is True
    assert len(client.queries) == 1
    assert client.receive_messages_calls == 0

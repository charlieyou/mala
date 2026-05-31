"""Unit tests for background Bash launch detection in MessageStreamProcessor.

Covers plan Step 3 (Option S): the stream processor records the tool_use_id of a
``Bash(run_in_background=true)`` launch on :class:`MessageIterationState`, clears it
when a matching ``task_completed`` event arrives within the same turn, and surfaces
the still-pending set on :class:`MessageIterationResult`.

These tests branch only on ``event.kind`` (the processor's own contract), so the
fake events here mirror the normalized ``AgentEvent`` vocabulary produced by
``src.core.protocols.agent_event`` without importing the SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

from src.domain.lifecycle import LifecycleContext
from src.pipeline.message_stream_processor import (
    MessageIterationState,
    MessageStreamProcessor,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from src.pipeline.message_stream_processor import MessageIterationResult


# --- Fake events mirroring the normalized AgentEvent shapes --------------------


@dataclass(frozen=True)
class _ToolUseEvent:
    id: str
    name: str
    input: dict[str, Any]
    kind: str = "tool_use"


@dataclass(frozen=True)
class _ToolResultEvent:
    tool_use_id: str
    is_error: bool = False
    content: object = None
    kind: str = "tool_result"


@dataclass(frozen=True)
class _TaskStartedEvent:
    task_id: str = ""
    tool_use_id: str = ""
    kind: str = "task_started"


@dataclass(frozen=True)
class _TaskCompletedEvent:
    task_id: str = ""
    tool_use_id: str = ""
    status: str = ""
    summary: str = ""
    output_file: str = ""
    kind: str = "task_completed"


@dataclass(frozen=True)
class _ResultEvent:
    session_id: str | None = "sess-1"
    result: str = "done"
    subtype: str = "success"
    is_error: bool = False
    kind: str = "result"


# --- Fakes for the processor's collaborators ----------------------------------


@dataclass
class _FakeLintCache:
    """Lint cache that never matches, so lint tracking is inert in these tests."""

    successes: list[tuple[str, str]] = field(default_factory=list)

    def detect_lint_command(self, command: str) -> str | None:
        del command
        return None

    def mark_success(self, lint_type: str, command: str) -> None:
        self.successes.append((lint_type, command))


@dataclass
class _FakeLifecycleContext(LifecycleContext):
    session_id: str | None = None
    final_result: str = ""


async def _stream(events: list[object]) -> AsyncIterator[object]:
    for event in events:
        yield event


async def _run(
    events: list[object],
) -> tuple[MessageIterationState, MessageIterationResult]:
    processor = MessageStreamProcessor()
    state = MessageIterationState()
    result = await processor.process_stream(
        stream=_stream(events),
        issue_id="issue-1",
        state=state,
        lifecycle_ctx=_FakeLifecycleContext(),
        lint_cache=_FakeLintCache(),
        query_start=0.0,
        tracer=None,
    )
    return state, result


# --- Tests --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_background_bash_launch_recorded_as_pending() -> None:
    events: list[object] = [
        _ToolUseEvent(
            id="tool-bg",
            name="Bash",
            input={"command": "long_job.sh", "run_in_background": True},
        ),
        _ResultEvent(),
    ]

    state, result = await _run(events)

    assert state.pending_background_tool_ids == {"tool-bg"}
    assert state.background_launch_commands == {"tool-bg": "long_job.sh"}
    assert result.pending_background_tool_ids == frozenset({"tool-bg"})


@pytest.mark.asyncio
async def test_background_detection_is_case_insensitive() -> None:
    events: list[object] = [
        _ToolUseEvent(
            id="tool-bg",
            name="bash",
            input={"command": "x", "run_in_background": True},
        ),
        _ResultEvent(),
    ]

    state, result = await _run(events)

    assert state.pending_background_tool_ids == {"tool-bg"}
    assert result.pending_background_tool_ids == frozenset({"tool-bg"})


@pytest.mark.asyncio
async def test_normal_bash_not_recorded() -> None:
    events: list[object] = [
        _ToolUseEvent(id="tool-fg", name="Bash", input={"command": "echo hi"}),
        _ResultEvent(),
    ]

    state, result = await _run(events)

    assert state.pending_background_tool_ids == set()
    assert state.background_launch_commands == {}
    assert result.pending_background_tool_ids == frozenset()


@pytest.mark.asyncio
async def test_run_in_background_false_not_recorded() -> None:
    events: list[object] = [
        _ToolUseEvent(
            id="tool-fg",
            name="Bash",
            input={"command": "echo hi", "run_in_background": False},
        ),
        _ResultEvent(),
    ]

    state, result = await _run(events)

    assert state.pending_background_tool_ids == set()
    assert result.pending_background_tool_ids == frozenset()


@pytest.mark.asyncio
async def test_non_bash_tool_not_recorded() -> None:
    events: list[object] = [
        _ToolUseEvent(
            id="tool-edit",
            name="Edit",
            input={"run_in_background": True},
        ),
        _ResultEvent(),
    ]

    state, result = await _run(events)

    assert state.pending_background_tool_ids == set()
    assert result.pending_background_tool_ids == frozenset()


@pytest.mark.asyncio
async def test_within_turn_task_completed_clears_pending() -> None:
    events: list[object] = [
        _ToolUseEvent(
            id="tool-bg",
            name="Bash",
            input={"command": "quick.sh", "run_in_background": True},
        ),
        _TaskStartedEvent(task_id="task-1", tool_use_id="tool-bg"),
        _TaskCompletedEvent(
            task_id="task-1",
            tool_use_id="tool-bg",
            status="completed",
            summary="exit 0",
            output_file="/tmp/out.log",
        ),
        _ResultEvent(),
    ]

    state, result = await _run(events)

    assert state.pending_background_tool_ids == set()
    assert result.pending_background_tool_ids == frozenset()


@pytest.mark.asyncio
async def test_task_completed_for_unknown_tool_id_is_ignored() -> None:
    events: list[object] = [
        _ToolUseEvent(
            id="tool-bg",
            name="Bash",
            input={"command": "long.sh", "run_in_background": True},
        ),
        _TaskCompletedEvent(
            task_id="task-other",
            tool_use_id="tool-other",
            status="completed",
        ),
        _ResultEvent(),
    ]

    state, result = await _run(events)

    # The unrelated completion must not clear our pending launch.
    assert state.pending_background_tool_ids == {"tool-bg"}
    assert result.pending_background_tool_ids == frozenset({"tool-bg"})


@pytest.mark.asyncio
async def test_multiple_launches_one_completes_within_turn() -> None:
    events: list[object] = [
        _ToolUseEvent(
            id="tool-a",
            name="Bash",
            input={"command": "a.sh", "run_in_background": True},
        ),
        _ToolUseEvent(
            id="tool-b",
            name="Bash",
            input={"command": "b.sh", "run_in_background": True},
        ),
        _TaskCompletedEvent(task_id="task-a", tool_use_id="tool-a", status="completed"),
        _ResultEvent(),
    ]

    state, result = await _run(events)

    assert state.pending_background_tool_ids == {"tool-b"}
    assert result.pending_background_tool_ids == frozenset({"tool-b"})


@pytest.mark.asyncio
async def test_task_started_alone_does_not_clear_pending() -> None:
    events: list[object] = [
        _ToolUseEvent(
            id="tool-bg",
            name="Bash",
            input={"command": "long.sh", "run_in_background": True},
        ),
        _TaskStartedEvent(task_id="task-1", tool_use_id="tool-bg"),
        _ResultEvent(),
    ]

    state, result = await _run(events)

    assert state.pending_background_tool_ids == {"tool-bg"}
    assert result.pending_background_tool_ids == frozenset({"tool-bg"})


# --- Inline retrieval clears only on terminal result or no-task-found error ----

_LAUNCH_ACK = (
    "Command running in background with ID: task-1. Output is being written to: "
    "/tmp/mala/task-1.output. You will be notified when it completes."
)


def _bash_bg_launch(tool_use_id: str = "tool-bg") -> _ToolUseEvent:
    return _ToolUseEvent(
        id=tool_use_id,
        name="Bash",
        input={"command": "long.sh", "run_in_background": True},
    )


@pytest.mark.asyncio
async def test_taskoutput_terminal_result_clears_pending() -> None:
    """TaskOutput(task_id) clears the launch once its result reports completion."""
    events: list[object] = [
        _bash_bg_launch(),
        _TaskStartedEvent(task_id="task-1", tool_use_id="tool-bg"),
        _ToolUseEvent(
            id="tool-out", name="TaskOutput", input={"task_id": "task-1", "block": True}
        ),
        _ToolResultEvent(
            tool_use_id="tool-out",
            content="<status>completed</status>\n<exit_code>0</exit_code>",
        ),
        _ResultEvent(),
    ]

    state, result = await _run(events)

    assert state.pending_background_tool_ids == set()
    assert result.pending_background_tool_ids == frozenset()


@pytest.mark.asyncio
async def test_taskoutput_terminal_result_correlates_via_ack_task_id() -> None:
    """The launch-ack tool_result supplies the task_id for correlation.

    Even without a ``task_started`` event, the ack's "ID: task-1" lets a later
    TaskOutput(task_id=task-1) terminal result clear the launch.
    """
    events: list[object] = [
        _bash_bg_launch(),
        _ToolResultEvent(tool_use_id="tool-bg", content=_LAUNCH_ACK),
        _ToolUseEvent(
            id="tool-out", name="TaskOutput", input={"task_id": "task-1", "block": True}
        ),
        _ToolResultEvent(tool_use_id="tool-out", content="<status>completed</status>"),
        _ResultEvent(),
    ]

    state, result = await _run(events)

    assert state.pending_background_tool_ids == set()
    assert result.pending_background_tool_ids == frozenset()


@pytest.mark.asyncio
async def test_taskoutput_request_without_result_keeps_pending() -> None:
    """A TaskOutput request alone (no result yet) does not clear the launch."""
    events: list[object] = [
        _bash_bg_launch(),
        _TaskStartedEvent(task_id="task-1", tool_use_id="tool-bg"),
        _ToolUseEvent(
            id="tool-out", name="TaskOutput", input={"task_id": "task-1", "block": True}
        ),
        _ResultEvent(),
    ]

    state, result = await _run(events)

    assert state.pending_background_tool_ids == {"tool-bg"}
    assert result.pending_background_tool_ids == frozenset({"tool-bg"})


@pytest.mark.asyncio
async def test_taskoutput_running_result_keeps_pending() -> None:
    """A block=False poll reporting ``running`` must NOT clear the launch."""
    events: list[object] = [
        _bash_bg_launch(),
        _TaskStartedEvent(task_id="task-1", tool_use_id="tool-bg"),
        _ToolUseEvent(
            id="tool-out",
            name="TaskOutput",
            input={"task_id": "task-1", "block": False},
        ),
        _ToolResultEvent(tool_use_id="tool-out", content="<status>running</status>"),
        _ResultEvent(),
    ]

    state, result = await _run(events)

    assert state.pending_background_tool_ids == {"tool-bg"}
    assert result.pending_background_tool_ids == frozenset({"tool-bg"})


@pytest.mark.asyncio
async def test_taskoutput_generic_error_result_keeps_pending() -> None:
    """A generic errored retrieval leaves the launch pending."""
    events: list[object] = [
        _bash_bg_launch(),
        _TaskStartedEvent(task_id="task-1", tool_use_id="tool-bg"),
        _ToolUseEvent(
            id="tool-out", name="TaskOutput", input={"task_id": "task-1", "block": True}
        ),
        _ToolResultEvent(
            tool_use_id="tool-out",
            is_error=True,
            content="<status>completed</status>",
        ),
        _ResultEvent(),
    ]

    state, result = await _run(events)

    assert state.pending_background_tool_ids == {"tool-bg"}
    assert result.pending_background_tool_ids == frozenset({"tool-bg"})


@pytest.mark.parametrize(
    ("tool_name", "id_field"),
    [("TaskOutput", "task_id"), ("BashOutput", "bash_id")],
)
@pytest.mark.asyncio
async def test_background_output_no_task_found_error_clears_pending(
    tool_name: str, id_field: str
) -> None:
    """A no-task-found poll is terminal for the correlated launch."""
    events: list[object] = [
        _bash_bg_launch(),
        _ToolResultEvent(tool_use_id="tool-bg", content=_LAUNCH_ACK),
        _ToolUseEvent(
            id="tool-out",
            name=tool_name,
            input={id_field: "task-1", "block": False},
        ),
        _ToolResultEvent(
            tool_use_id="tool-out",
            is_error=True,
            content="<tool_use_error>No task found with ID: task-1</tool_use_error>",
        ),
        _ResultEvent(),
    ]

    state, result = await _run(events)

    assert state.pending_background_tool_ids == set()
    assert result.pending_background_tool_ids == frozenset()


@pytest.mark.asyncio
async def test_unrelated_taskoutput_does_not_clear() -> None:
    """A TaskOutput for a different task leaves the launch pending."""
    events: list[object] = [
        _bash_bg_launch(),
        _TaskStartedEvent(task_id="task-1", tool_use_id="tool-bg"),
        _ToolUseEvent(
            id="tool-out",
            name="TaskOutput",
            input={"task_id": "task-999", "block": True},
        ),
        _ToolResultEvent(tool_use_id="tool-out", content="<status>completed</status>"),
        _ResultEvent(),
    ]

    state, result = await _run(events)

    assert state.pending_background_tool_ids == {"tool-bg"}
    assert result.pending_background_tool_ids == frozenset({"tool-bg"})


@pytest.mark.asyncio
async def test_read_does_not_clear_pending() -> None:
    """A ``Read`` is only an interim check, never a consumption signal.

    The SDK explicitly suggests Read for interim output, so a Read of the task's
    output file must not clear the launch (it may be partial / still running).
    """
    events: list[object] = [
        _bash_bg_launch(),
        _TaskStartedEvent(task_id="task-1", tool_use_id="tool-bg"),
        _ToolUseEvent(
            id="tool-read", name="Read", input={"file_path": "/tmp/mala/task-1.output"}
        ),
        _ToolResultEvent(tool_use_id="tool-read", content="partial output..."),
        _ResultEvent(),
    ]

    state, result = await _run(events)

    assert state.pending_background_tool_ids == {"tool-bg"}
    assert result.pending_background_tool_ids == frozenset({"tool-bg"})

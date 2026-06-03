"""Unit tests for ``lock_wait`` park detection in MessageStreamProcessor.

Covers plan Step 2 (Detection): the stream processor records the tool_use_id of a
``lock_wait`` MCP call (``mcp__mala-locking__lock_wait``) on
:class:`MessageIterationState`, then — when that call's tool_result reports
``parked: true`` — moves its ``wait_paths`` into ``pending_lock_waits`` and surfaces
the snapshot on :class:`MessageIterationResult`. A ``parked: false`` result records
nothing.

These tests branch only on ``event.kind`` (the processor's own contract), so the
fake events here mirror the normalized ``AgentEvent`` vocabulary produced by
``src.core.protocols.agent_event`` without importing the SDK.
"""

from __future__ import annotations

import json
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


_LOCK_WAIT_TOOL = "mcp__mala-locking__lock_wait"


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


def _lock_wait_call(tool_use_id: str = "tool-lw") -> _ToolUseEvent:
    return _ToolUseEvent(
        id=tool_use_id,
        name=_LOCK_WAIT_TOOL,
        input={"filepaths": ["src/a.py", "src/b.py"]},
    )


# --- Tests --------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parked_lock_wait_populates_pending_lock_waits() -> None:
    events: list[object] = [
        _lock_wait_call(),
        _ToolResultEvent(
            tool_use_id="tool-lw",
            content=json.dumps(
                {"parked": True, "wait_paths": ["/repo/src/a.py", "/repo/src/b.py"]}
            ),
        ),
        _ResultEvent(),
    ]

    state, result = await _run(events)

    assert state.pending_lock_waits == {"/repo/src/a.py", "/repo/src/b.py"}
    assert result.pending_lock_waits == frozenset({"/repo/src/a.py", "/repo/src/b.py"})
    # The request id is consumed once resolved.
    assert state.lock_wait_request_ids == set()


@pytest.mark.asyncio
async def test_not_parked_lock_wait_records_nothing() -> None:
    events: list[object] = [
        _lock_wait_call(),
        _ToolResultEvent(
            tool_use_id="tool-lw",
            content=json.dumps({"parked": False, "all_acquired_or_free": True}),
        ),
        _ResultEvent(),
    ]

    state, result = await _run(events)

    assert state.pending_lock_waits == set()
    assert result.pending_lock_waits == frozenset()
    assert state.lock_wait_request_ids == set()


@pytest.mark.asyncio
async def test_lock_wait_content_as_blocks_is_flattened() -> None:
    """The MCP result content may arrive as a list of text blocks."""
    events: list[object] = [
        _lock_wait_call(),
        _ToolResultEvent(
            tool_use_id="tool-lw",
            content=[
                {
                    "type": "text",
                    "text": json.dumps(
                        {"parked": True, "wait_paths": ["/repo/src/a.py"]}
                    ),
                }
            ],
        ),
        _ResultEvent(),
    ]

    state, result = await _run(events)

    assert state.pending_lock_waits == {"/repo/src/a.py"}
    assert result.pending_lock_waits == frozenset({"/repo/src/a.py"})


@pytest.mark.asyncio
async def test_lock_wait_request_without_result_records_nothing() -> None:
    """A lock_wait call whose result never arrives leaves the snapshot empty."""
    events: list[object] = [
        _lock_wait_call(),
        _ResultEvent(),
    ]

    state, result = await _run(events)

    assert state.pending_lock_waits == set()
    assert result.pending_lock_waits == frozenset()
    # The request id is still outstanding (no result resolved it).
    assert state.lock_wait_request_ids == {"tool-lw"}


@pytest.mark.asyncio
async def test_unrelated_tool_result_does_not_populate() -> None:
    """A tool_result for an id we never tracked must not park anything."""
    events: list[object] = [
        _lock_wait_call(),
        _ToolResultEvent(
            tool_use_id="tool-other",
            content=json.dumps({"parked": True, "wait_paths": ["/repo/x.py"]}),
        ),
        _ResultEvent(),
    ]

    state, result = await _run(events)

    assert state.pending_lock_waits == set()
    assert result.pending_lock_waits == frozenset()
    assert state.lock_wait_request_ids == {"tool-lw"}

"""Processor consumes ``AgentEvent.kind`` — no provider class-name branches.

This module asserts the A1 contract: ``MessageStreamProcessor`` reaches its
text / tool_use / tool_result / result side effects purely from
``event.kind``, regardless of the adapter that produced the event. The
``[integration-path-test]`` exercises the processor's full event-handling
path with both Claude (Anthropic SDK) and Amp (synthetic-event) adapter
outputs translated through :func:`src.core.protocols.agent_event.to_agent_events`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pytest

from src.core.protocols.agent_event import (
    AgentResultEvent,
    AgentTextEvent,
    AgentToolResultEvent,
    AgentToolUseEvent,
    to_agent_events,
)
from src.domain.lifecycle import LifecycleContext
from src.pipeline.message_stream_processor import (
    MessageIterationState,
    MessageStreamProcessor,
    StreamProcessorCallbacks,
)
from tests.fakes.lint_cache import FakeLintCache

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@dataclass
class _Observations:
    text_calls: list[tuple[str, str]] = field(default_factory=list)
    tool_calls: list[tuple[str, str, dict[str, Any] | None]] = field(
        default_factory=list
    )
    final_session_id: str | None = None
    final_result: str = ""
    tool_calls_this_turn: int = 0
    pending_tool_ids: set[str] = field(default_factory=set)


async def _stream(events: list[Any]) -> AsyncIterator[Any]:
    for ev in events:
        yield ev


async def _drive(events: list[Any]) -> _Observations:
    obs = _Observations()
    callbacks = StreamProcessorCallbacks(
        on_tool_use=lambda issue_id, name, inp: obs.tool_calls.append(
            (issue_id, name, inp)
        ),
        on_agent_text=lambda issue_id, text: obs.text_calls.append((issue_id, text)),
    )
    processor = MessageStreamProcessor(callbacks=callbacks)
    state = MessageIterationState()
    lifecycle_ctx = LifecycleContext()
    lint_cache = FakeLintCache()

    await processor.process_stream(
        stream=_stream(events),
        issue_id="issue-42",
        state=state,
        lifecycle_ctx=lifecycle_ctx,
        lint_cache=lint_cache,
        query_start=0.0,
        tracer=None,
    )

    obs.final_session_id = lifecycle_ctx.session_id
    obs.final_result = lifecycle_ctx.final_result
    obs.tool_calls_this_turn = state.tool_calls_this_turn
    obs.pending_tool_ids = set(state.pending_tool_ids)
    return obs


# ---------------------------------------------------------------------------
# kind discrimination — processor reaches each branch from kind alone
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_text_kind_invokes_text_callback() -> None:
    obs = await _drive(
        [
            AgentTextEvent(text="hi"),
            AgentResultEvent(session_id="sess-x", result="done"),
        ]
    )
    assert obs.text_calls == [("issue-42", "hi")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_text_deltas_are_coalesced_before_callback() -> None:
    obs = await _drive(
        [
            AgentTextEvent(text="I", is_delta=True),
            AgentTextEvent(text="'ll", is_delta=True),
            AgentTextEvent(text=" implement", is_delta=True),
            AgentToolUseEvent(id="t-1", name="Bash", input={"command": "pwd"}),
            AgentResultEvent(session_id="sess-x", result="done"),
        ]
    )
    assert obs.text_calls == [("issue-42", "I'll implement")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_text_deltas_flush_when_stream_raises() -> None:
    obs = _Observations()
    callbacks = StreamProcessorCallbacks(
        on_agent_text=lambda issue_id, text: obs.text_calls.append((issue_id, text)),
    )
    processor = MessageStreamProcessor(callbacks=callbacks)

    async def broken_stream() -> AsyncIterator[Any]:
        yield AgentTextEvent(text="partial", is_delta=True)
        raise RuntimeError("stream failed")

    with pytest.raises(RuntimeError, match="stream failed"):
        await processor.process_stream(
            stream=broken_stream(),
            issue_id="issue-42",
            state=MessageIterationState(),
            lifecycle_ctx=LifecycleContext(),
            lint_cache=FakeLintCache(),
            query_start=0.0,
            tracer=None,
        )

    assert obs.text_calls == [("issue-42", "partial")]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_use_kind_increments_counter_and_tracks_pending() -> None:
    obs = await _drive(
        [
            AgentToolUseEvent(id="t-1", name="Read", input={"path": "/x"}),
            AgentResultEvent(session_id="sess-x", result=None),
        ]
    )
    assert obs.tool_calls == [("issue-42", "Read", {"path": "/x"})]
    assert obs.tool_calls_this_turn == 1
    assert obs.pending_tool_ids == {"t-1"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tool_result_kind_clears_pending_match() -> None:
    obs = await _drive(
        [
            AgentToolUseEvent(id="t-1", name="Read", input={}),
            AgentToolResultEvent(tool_use_id="t-1", is_error=False, content="ok"),
            AgentResultEvent(session_id="sess-x", result="done"),
        ]
    )
    assert obs.pending_tool_ids == set()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_result_kind_populates_lifecycle() -> None:
    obs = await _drive(
        [
            AgentResultEvent(session_id="sess-final", result="done"),
        ]
    )
    assert obs.final_session_id == "sess-final"
    assert obs.final_result == "done"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unknown_kind_is_silently_ignored() -> None:
    @dataclass(frozen=True)
    class _Mystery:
        kind: str = "unrecognised"

    obs = await _drive(
        [
            _Mystery(),
            AgentResultEvent(session_id="sess-x", result="done"),
        ]
    )
    # No callbacks fire, no pending tools, no crash.
    assert obs.text_calls == []
    assert obs.tool_calls == []
    assert obs.pending_tool_ids == set()


# ---------------------------------------------------------------------------
# integration-path-test — Claude + Amp adapter outputs reach the processor
# through to_agent_events and produce identical observations
# ---------------------------------------------------------------------------


def _claude_messages() -> list[Any]:
    """Anthropic SDK messages with one of each block kind plus a result."""
    from claude_agent_sdk import (
        AssistantMessage as ClaudeAssistantMessage,
    )
    from claude_agent_sdk import (
        ResultMessage as ClaudeResultMessage,
    )
    from claude_agent_sdk import (
        TextBlock as ClaudeTextBlock,
    )
    from claude_agent_sdk import (
        ToolResultBlock as ClaudeToolResultBlock,
    )
    from claude_agent_sdk import (
        ToolUseBlock as ClaudeToolUseBlock,
    )

    return [
        ClaudeAssistantMessage(
            content=[
                ClaudeTextBlock(text="hello"),
                ClaudeToolUseBlock(id="t-1", name="Bash", input={"command": "ls"}),
                ClaudeToolResultBlock(tool_use_id="t-1", content="ok", is_error=False),
            ],
            model="claude-test",
        ),
        ClaudeResultMessage(
            subtype="result",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="sess-xyz",
            result="done",
        ),
    ]


def _amp_events() -> list[Any]:
    """Pre-translated AgentEvents matching what AmpClient now emits."""
    return [
        AgentTextEvent(text="hello"),
        AgentToolUseEvent(id="t-1", name="Bash", input={"command": "ls"}),
        AgentToolResultEvent(tool_use_id="t-1", content="ok", is_error=False),
        AgentResultEvent(session_id="sess-xyz", result="done", subtype="result"),
    ]


async def _drive_via_translator(messages: list[Any]) -> _Observations:
    """Run inputs through ``to_agent_events`` then the processor."""
    obs = _Observations()
    callbacks = StreamProcessorCallbacks(
        on_tool_use=lambda issue_id, name, inp: obs.tool_calls.append(
            (issue_id, name, inp)
        ),
        on_agent_text=lambda issue_id, text: obs.text_calls.append((issue_id, text)),
    )
    processor = MessageStreamProcessor(callbacks=callbacks)
    state = MessageIterationState()
    lifecycle_ctx = LifecycleContext()
    lint_cache = FakeLintCache()

    await processor.process_stream(
        stream=to_agent_events(_stream(messages)),
        issue_id="issue-42",
        state=state,
        lifecycle_ctx=lifecycle_ctx,
        lint_cache=lint_cache,
        query_start=0.0,
        tracer=None,
    )
    obs.final_session_id = lifecycle_ctx.session_id
    obs.final_result = lifecycle_ctx.final_result
    obs.tool_calls_this_turn = state.tool_calls_this_turn
    obs.pending_tool_ids = set(state.pending_tool_ids)
    return obs


@pytest.mark.unit
@pytest.mark.asyncio
async def test_claude_and_amp_adapter_outputs_produce_identical_observations() -> None:
    """[integration-path-test] Claude SDK messages and Amp events round-trip
    through ``to_agent_events`` to the same processor observations — proving
    the processor's path is coder-agnostic."""
    claude_obs = await _drive_via_translator(_claude_messages())
    amp_obs = await _drive_via_translator(_amp_events())

    assert amp_obs.text_calls == claude_obs.text_calls == [("issue-42", "hello")]
    assert (
        amp_obs.tool_calls
        == claude_obs.tool_calls
        == [("issue-42", "Bash", {"command": "ls"})]
    )
    assert amp_obs.tool_calls_this_turn == claude_obs.tool_calls_this_turn == 1
    assert amp_obs.pending_tool_ids == claude_obs.pending_tool_ids == set()
    assert amp_obs.final_session_id == claude_obs.final_session_id == "sess-xyz"
    assert amp_obs.final_result == claude_obs.final_result == "done"


# ---------------------------------------------------------------------------
# Regression: no source line in src/pipeline/ may key on Anthropic class
# names. Pin the contract here (the verification grep is an extra layer).
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_pipeline_processor_has_no_anthropic_classname_branches() -> None:
    """The processor does not branch on ``type(...).__name__`` against
    Anthropic class names. Captures AC #15 — drift would silently regress
    coder-agnostic event handling."""
    from pathlib import Path

    src = Path(__file__).resolve().parents[3] / "src" / "pipeline"
    forbidden = ("AssistantMessage", "ToolUseBlock", "ToolResultBlock")
    offenders: list[tuple[Path, str]] = []
    for path in src.rglob("*.py"):
        text = path.read_text()
        if "type(" not in text or "__name__" not in text:
            continue
        for line in text.splitlines():
            if (
                "type(" in line
                and "__name__" in line
                and any(cls in line for cls in forbidden)
            ):
                offenders.append((path, line))
    assert offenders == [], f"class-name duck typing leaked into pipeline: {offenders}"

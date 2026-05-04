"""Unit tests for synthetic Amp message dataclasses.

These dataclasses must be drop-in compatible with the Claude SDK message
types as far as `MessageStreamProcessor` is concerned: the processor keys
off `type(message).__name__` and reads fields via `getattr`. The round-trip
test asserts that feeding a synthetic `AssistantMessage` (with all three
block types) plus a synthetic `ResultMessage` into the real processor
produces the same observable callbacks and state mutations as feeding the
Claude SDK equivalents.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import pytest
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

from src.domain.lifecycle import LifecycleContext
from src.infra.clients.amp_messages import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from src.pipeline.message_stream_processor import (
    MessageIterationState,
    MessageStreamProcessor,
    StreamProcessorCallbacks,
)
from tests.fakes.lint_cache import FakeLintCache

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# ---------------------------------------------------------------------------
# Class-name and field-name contract assertions
# ---------------------------------------------------------------------------


def test_class_names_match_processor_keys() -> None:
    """`MessageStreamProcessor` reads `type(msg).__name__`; names must match."""
    assert AssistantMessage.__name__ == "AssistantMessage"
    assert ResultMessage.__name__ == "ResultMessage"
    assert TextBlock.__name__ == "TextBlock"
    assert ToolUseBlock.__name__ == "ToolUseBlock"
    assert ToolResultBlock.__name__ == "ToolResultBlock"


@pytest.mark.parametrize(
    ("cls", "expected_fields"),
    [
        (TextBlock, {"text"}),
        (ToolUseBlock, {"id", "name", "input"}),
        (ToolResultBlock, {"tool_use_id", "is_error", "content"}),
        (AssistantMessage, {"content"}),
        (ResultMessage, {"session_id", "result"}),
    ],
)
def test_field_names_match_processor_getattr_keys(
    cls: type, expected_fields: set[str]
) -> None:
    """Fields read by the processor via `getattr` must exist by exact name."""
    actual = {f.name for f in dataclasses.fields(cls)}
    assert actual == expected_fields, (
        f"{cls.__name__} fields {actual} do not match expected {expected_fields}"
    )


# ---------------------------------------------------------------------------
# Round-trip: synthetic vs Claude SDK messages produce identical observations
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Observations:
    text_calls: list[tuple[str, str]] = dataclasses.field(default_factory=list)
    tool_calls: list[tuple[str, str, dict[str, Any] | None]] = dataclasses.field(
        default_factory=list
    )
    final_session_id: str | None = None
    final_result: str = ""
    tool_calls_this_turn: int = 0
    pending_tool_ids: set[str] = dataclasses.field(default_factory=set)


async def _stream(messages: list[Any]) -> AsyncIterator[Any]:
    for msg in messages:
        yield msg


async def _drive_processor(messages: list[Any]) -> _Observations:
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
        stream=_stream(messages),
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


def _claude_messages() -> list[Any]:
    return [
        ClaudeAssistantMessage(
            content=[
                ClaudeTextBlock(text="hello"),
                ClaudeToolUseBlock(id="tool-1", name="Bash", input={"command": "ls"}),
                ClaudeToolResultBlock(
                    tool_use_id="tool-1", content="ok", is_error=False
                ),
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


def _amp_messages() -> list[Any]:
    return [
        AssistantMessage(
            content=[
                TextBlock(text="hello"),
                ToolUseBlock(id="tool-1", name="Bash", input={"command": "ls"}),
                ToolResultBlock(tool_use_id="tool-1", content="ok", is_error=False),
            ]
        ),
        ResultMessage(session_id="sess-xyz", result="done"),
    ]


@pytest.mark.asyncio
async def test_round_trip_matches_claude_sdk_observable_output() -> None:
    """Driving the processor with synthetic Amp messages produces the same
    observable callbacks and state changes as driving it with Claude SDK
    messages constructed with identical payloads."""
    claude_obs = await _drive_processor(_claude_messages())
    amp_obs = await _drive_processor(_amp_messages())

    assert amp_obs.text_calls == claude_obs.text_calls == [("issue-42", "hello")]
    assert (
        amp_obs.tool_calls
        == claude_obs.tool_calls
        == [("issue-42", "Bash", {"command": "ls"})]
    )
    assert amp_obs.tool_calls_this_turn == claude_obs.tool_calls_this_turn == 1
    # ToolResultBlock with matching tool_use_id should clear the pending set.
    assert amp_obs.pending_tool_ids == claude_obs.pending_tool_ids == set()
    assert amp_obs.final_session_id == claude_obs.final_session_id == "sess-xyz"
    assert amp_obs.final_result == claude_obs.final_result == "done"


@pytest.mark.asyncio
async def test_round_trip_preserves_pending_tool_when_result_missing() -> None:
    """If no ToolResultBlock arrives, the tool id stays pending — same as
    the Claude path."""
    claude_obs = await _drive_processor(
        [
            ClaudeAssistantMessage(
                content=[
                    ClaudeToolUseBlock(
                        id="tool-9", name="Read", input={"path": "/tmp"}
                    ),
                ],
                model="claude-test",
            ),
            ClaudeResultMessage(
                subtype="result",
                duration_ms=1,
                duration_api_ms=1,
                is_error=False,
                num_turns=1,
                session_id="sess-9",
                result=None,
            ),
        ]
    )
    amp_obs = await _drive_processor(
        [
            AssistantMessage(
                content=[
                    ToolUseBlock(id="tool-9", name="Read", input={"path": "/tmp"}),
                ]
            ),
            ResultMessage(session_id="sess-9", result=None),
        ]
    )
    assert amp_obs.pending_tool_ids == claude_obs.pending_tool_ids == {"tool-9"}
    assert amp_obs.final_result == claude_obs.final_result == ""


# ---------------------------------------------------------------------------
# Duck-typing regression: the processor never calls isinstance, but the
# class-name lookup is what makes synthetic objects substitutable. Guard
# against a refactor that swaps `type(...).__name__` for `isinstance(...)`.
# ---------------------------------------------------------------------------


def test_synthetic_classes_are_distinct_from_claude_sdk() -> None:
    """Synthetic dataclasses are *not* subclasses of the Claude SDK types.
    Substitutability rests on duck-typing by class name + field name, so a
    future refactor of the processor to use `isinstance` would silently
    break Amp. This test pins the current contract."""
    assert not issubclass(AssistantMessage, ClaudeAssistantMessage)
    assert not issubclass(ResultMessage, ClaudeResultMessage)
    assert not issubclass(TextBlock, ClaudeTextBlock)
    assert not issubclass(ToolUseBlock, ClaudeToolUseBlock)
    assert not issubclass(ToolResultBlock, ClaudeToolResultBlock)

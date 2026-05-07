"""Coder-agnostic agent event protocol.

`AgentEvent` is the cross-coder event shape consumed by the pipeline (chiefly
`MessageStreamProcessor`). Each adapter (Claude, Amp, Codex) translates its
provider-specific stream into a flat sequence of these events. A `kind`
discriminator lets consumers branch on event type without provider-specific
class-name duck typing.

Two surfaces are exposed: ``AgentEvent`` is a ``runtime_checkable`` protocol
for ``isinstance`` checks, and ``AgentEventValue`` is the concrete union used
for static return annotations (avoids variance friction with frozen
dataclasses whose ``kind`` is a narrow Literal).

The :func:`to_agent_events` translator lives here too: it flattens Anthropic
SDK ``AssistantMessage`` / ``ResultMessage`` instances (and pass-through
``AgentEvent``s) into the canonical event stream. Hosting it on the protocol
side keeps the pipeline free of SDK-import contracts (``src.pipeline``
cannot transitively reach ``claude_agent_sdk``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


AgentEventKind = Literal["text", "tool_use", "tool_result", "result"]


@runtime_checkable
class AgentEvent(Protocol):
    """Cross-coder event flowing into pipeline consumers."""

    kind: AgentEventKind


@dataclass(frozen=True)
class AgentTextEvent:
    """Assistant emitted plain text."""

    text: str = ""
    kind: Literal["text"] = "text"


@dataclass(frozen=True)
class AgentToolUseEvent:
    """Assistant invoked a tool."""

    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)
    kind: Literal["tool_use"] = "tool_use"


@dataclass(frozen=True)
class AgentToolResultEvent:
    """Tool execution returned a result for a prior tool_use.

    `content` is intentionally typed as `object` — callers cast to the shape
    they expect for the specific tool.
    """

    tool_use_id: str = ""
    is_error: bool = False
    content: object = None
    kind: Literal["tool_result"] = "tool_result"


@dataclass(frozen=True)
class AgentResultEvent:
    """Terminal event closing a turn.

    `session_id` is the provider's session/thread id (Anthropic
    `ResultMessage.session_id` / Amp `result.session_id`). `result` carries the
    provider's terminal payload as an opaque object — callers introspect
    based on the provider/subtype.
    """

    session_id: str = ""
    is_error: bool = False
    subtype: str = ""
    result: object = None
    kind: Literal["result"] = "result"


AgentEventValue = (
    AgentTextEvent | AgentToolUseEvent | AgentToolResultEvent | AgentResultEvent
)
"""Concrete union of all `AgentEvent` variants for static typing."""


_AGENT_EVENT_KINDS = frozenset({"text", "tool_use", "tool_result", "result"})


async def to_agent_events(
    stream: AsyncIterator[object],
) -> AsyncIterator[AgentEventValue]:
    """Translate a provider stream into a flat sequence of ``AgentEvent``s.

    Three input shapes are accepted:

    * Anthropic SDK ``AssistantMessage`` (and synthetic look-alikes used by
      tests): each content block is flattened into one event keyed by block
      class name (``TextBlock`` / ``ToolUseBlock`` / ``ToolResultBlock``).
    * Anthropic SDK ``ResultMessage`` (and synthetic look-alikes): mapped to
      a single ``AgentResultEvent``.
    * Anything already carrying a recognised ``kind`` discriminator (i.e. an
      ``AgentEvent``) is yielded verbatim, so providers that emit events
      directly (Amp) pass through unchanged.

    Other shapes are dropped silently — matching the previous duck-typing
    behaviour in ``MessageStreamProcessor``, which simply skipped messages
    whose class name did not match.
    """

    async for message in stream:
        kind = getattr(message, "kind", None)
        if isinstance(kind, str) and kind in _AGENT_EVENT_KINDS:
            yield cast("AgentEventValue", message)
            continue

        msg_class = type(message).__name__
        if msg_class == "AssistantMessage":
            for block in getattr(message, "content", []) or []:
                event = _block_to_event(block)
                if event is not None:
                    yield event
        elif msg_class == "ResultMessage":
            yield _result_to_event(message)


def _block_to_event(block: object) -> AgentEventValue | None:
    """Translate a single content block to an ``AgentEvent`` or skip it."""
    block_type = type(block).__name__
    if block_type == "TextBlock":
        return AgentTextEvent(text=str(getattr(block, "text", "") or ""))
    if block_type == "ToolUseBlock":
        block_input = getattr(block, "input", {}) or {}
        if not isinstance(block_input, dict):
            block_input = {}
        return AgentToolUseEvent(
            id=str(getattr(block, "id", "") or ""),
            name=str(getattr(block, "name", "") or ""),
            input=dict(block_input),
        )
    if block_type == "ToolResultBlock":
        return AgentToolResultEvent(
            tool_use_id=str(getattr(block, "tool_use_id", "") or ""),
            is_error=bool(getattr(block, "is_error", False)),
            content=getattr(block, "content", None),
        )
    return None


def _result_to_event(message: object) -> AgentResultEvent:
    """Translate an Anthropic-shaped ``ResultMessage`` to ``AgentResultEvent``.

    `agent_sdk_review` consumes Anthropic ``ResultMessage`` instances directly
    and is responsible for any structured-output handling — this translator
    is the pipeline path only and just forwards ``result`` verbatim.
    """
    session_id = str(getattr(message, "session_id", "") or "")
    subtype = str(getattr(message, "subtype", "") or "")
    is_error = bool(getattr(message, "is_error", False))
    raw_result = getattr(message, "result", None)
    return AgentResultEvent(
        session_id=session_id,
        is_error=is_error,
        subtype=subtype,
        result=raw_result,
    )

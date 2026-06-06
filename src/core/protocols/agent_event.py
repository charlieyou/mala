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

import html
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


AgentEventKind = Literal[
    "text", "tool_use", "tool_result", "result", "task_started", "task_completed"
]


@runtime_checkable
class AgentEvent(Protocol):
    """Cross-coder event flowing into pipeline consumers."""

    kind: AgentEventKind


@dataclass(frozen=True)
class AgentTextEvent:
    """Assistant emitted plain text."""

    text: str = ""
    is_delta: bool = False
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


@dataclass(frozen=True)
class AgentTaskStartedEvent:
    """A backgrounded task started (e.g. ``Bash(run_in_background=True)``).

    Surfaced from the Claude SDK ``TaskStartedMessage``. ``task_id`` is the
    SDK's task handle; ``tool_use_id`` ties the task back to the originating
    ``Bash`` tool-use block so consumers can correlate the launch.
    """

    task_id: str = ""
    tool_use_id: str = ""
    kind: Literal["task_started"] = "task_started"


@dataclass(frozen=True)
class AgentTaskCompletedEvent:
    """A backgrounded task finished and the SDK delivered a notification.

    Surfaced from the Claude SDK ``TaskNotificationMessage``, which the SDK
    pushes proactively on the continuous message stream after the launching
    turn ends. ``status`` is the SDK's terminal status (e.g.
    ``completed``/``failed``/``stopped``); ``summary`` carries the
    human-readable summary (including exit code) and ``output_file`` the path
    to the captured task output.
    """

    task_id: str = ""
    tool_use_id: str = ""
    status: str = ""
    summary: str = ""
    output_file: str = ""
    kind: Literal["task_completed"] = "task_completed"


AgentEventValue = (
    AgentTextEvent
    | AgentToolUseEvent
    | AgentToolResultEvent
    | AgentResultEvent
    | AgentTaskStartedEvent
    | AgentTaskCompletedEvent
)
"""Concrete union of all `AgentEvent` variants for static typing."""


_AGENT_EVENT_KINDS = frozenset(
    {
        "text",
        "tool_use",
        "tool_result",
        "result",
        "task_started",
        "task_completed",
    }
)

_TASK_NOTIFICATION_RE = re.compile(
    r"<task-notification\b[^>]*>(?P<body>.*?)</task-notification>",
    re.IGNORECASE | re.DOTALL,
)


def agent_task_completed_from_notification_text(
    text: str,
) -> AgentTaskCompletedEvent | None:
    """Parse Claude's queued ``<task-notification>`` prompt, when present.

    Recent Claude Code builds can deliver background-task completion as a queued
    command / attachment prompt instead of a typed SDK ``TaskNotificationMessage``.
    The XML-ish prompt carries the same fields, so adapters can recover the
    canonical ``task_completed`` event from the text without depending on SDK
    internals.
    """
    match = _TASK_NOTIFICATION_RE.search(text)
    if match is None:
        return None
    body = match.group("body")
    task_id = _task_notification_tag(body, "task-id")
    tool_use_id = _task_notification_tag(body, "tool-use-id")
    if not task_id and not tool_use_id:
        return None
    return AgentTaskCompletedEvent(
        task_id=task_id,
        tool_use_id=tool_use_id,
        status=_task_notification_tag(body, "status"),
        summary=_task_notification_tag(body, "summary"),
        output_file=_task_notification_tag(body, "output-file"),
    )


def _task_notification_tag(body: str, name: str) -> str:
    match = re.search(
        rf"<{re.escape(name)}\b[^>]*>\s*(.*?)\s*</{re.escape(name)}>",
        body,
        re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return ""
    return html.unescape(match.group(1).strip())


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
        elif msg_class == "UserMessage":
            for event in _user_message_to_events(message):
                yield event
        elif msg_class == "ResultMessage":
            yield _result_to_event(message)
        elif msg_class == "TaskStartedMessage":
            yield _task_started_to_event(message)
        elif msg_class == "TaskNotificationMessage":
            yield _task_notification_to_event(message)


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


def _user_message_to_events(message: object) -> list[AgentEventValue]:
    """Translate SDK ``UserMessage`` blocks relevant to pipeline state.

    Tool results are user-role messages in Claude SDK streams. The pipeline must
    observe them to clear pending tool ids, learn background launch ids from Bash
    acknowledgements, and mark successful lint commands. Plain user prompt text
    is intentionally ignored, except for Claude queued task notifications, which
    are provider-generated control prompts and should surface as task completion.
    """
    content = getattr(message, "content", None)
    if isinstance(content, str):
        event = agent_task_completed_from_notification_text(content)
        return [event] if event is not None else []
    if not isinstance(content, list):
        return []

    events: list[AgentEventValue] = []
    for block in content:
        if isinstance(block, dict):
            block_type = str(block.get("type") or "")
            if block_type == "tool_result":
                events.append(
                    AgentToolResultEvent(
                        tool_use_id=str(block.get("tool_use_id") or ""),
                        is_error=bool(block.get("is_error", False)),
                        content=block.get("content"),
                    )
                )
                continue
            if block_type == "text":
                text = block.get("text")
                if isinstance(text, str):
                    event = agent_task_completed_from_notification_text(text)
                    if event is not None:
                        events.append(event)
                continue

        block_type = type(block).__name__
        if block_type == "ToolResultBlock":
            events.append(
                AgentToolResultEvent(
                    tool_use_id=str(getattr(block, "tool_use_id", "") or ""),
                    is_error=bool(getattr(block, "is_error", False)),
                    content=getattr(block, "content", None),
                )
            )
            continue
        if block_type == "TextBlock":
            event = agent_task_completed_from_notification_text(
                str(getattr(block, "text", "") or "")
            )
            if event is not None:
                events.append(event)
    return events


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


def _task_started_to_event(message: object) -> AgentTaskStartedEvent:
    """Translate a Claude SDK ``TaskStartedMessage`` to ``AgentTaskStartedEvent``.

    Detected by class name (not ``isinstance``) so this module keeps its
    no-``claude_agent_sdk``-import contract, mirroring the ``AssistantMessage``
    / ``ResultMessage`` handling above.
    """
    return AgentTaskStartedEvent(
        task_id=str(getattr(message, "task_id", "") or ""),
        tool_use_id=str(getattr(message, "tool_use_id", "") or ""),
    )


def _task_notification_to_event(message: object) -> AgentTaskCompletedEvent:
    """Translate a Claude SDK ``TaskNotificationMessage`` to a completed event.

    Maps ``status`` / ``summary`` / ``output_file`` / ``task_id`` /
    ``tool_use_id`` across. ``None`` fields coerce to ``""`` so consumers can
    rely on string-typed fields.
    """
    return AgentTaskCompletedEvent(
        task_id=str(getattr(message, "task_id", "") or ""),
        tool_use_id=str(getattr(message, "tool_use_id", "") or ""),
        status=str(getattr(message, "status", "") or ""),
        summary=str(getattr(message, "summary", "") or ""),
        output_file=str(getattr(message, "output_file", "") or ""),
    )

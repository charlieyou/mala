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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable


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

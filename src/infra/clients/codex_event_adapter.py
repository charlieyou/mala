"""Codex notification → :class:`AgentEvent` translator (Phase D, T011).

Pure-stateful translator: takes one ``codex_app_server`` notification at a
time and emits zero or more provider-neutral :class:`AgentEvent` values.
The adapter is instance-scoped because two D1 cases need cross-message
context: streamed ``item/agentMessage/delta`` notifications and the
matching ``ItemCompletedNotification`` carrying an
``AgentMessageThreadItem`` both target the same logical text. The plan
table (``plans/2026-05-07-codex-provider-plan.md`` L749-L762) requires
deduplication so the pipeline does not see the same text twice; the
``_seen_message_deltas`` set tracks which item ids already streamed at
least one delta so the completed item is suppressed in that case.

Mapping table — implements plan L749-L762 verbatim:

  * ``item/agentMessage/delta`` → :class:`AgentTextEvent` (D1).
  * ``item/completed`` carrying ``AgentMessageThreadItem`` →
    :class:`AgentTextEvent` only when no prior delta was observed for
    that item id (D1 dedup).
  * ``item/started`` carrying ``CommandExecutionThreadItem`` →
    :class:`AgentToolUseEvent` (``name="bash"``) so
    :meth:`LintCacheProtocol.detect_lint_command` recognises bash
    commands the same way it does for Claude/Amp (D2).
  * ``item/completed`` carrying ``CommandExecutionThreadItem`` →
    paired :class:`AgentToolResultEvent` keyed by item id (D2).
  * ``item/started`` carrying ``FileChangeThreadItem`` →
    :class:`AgentToolUseEvent` (``name="file_change"``); paired
    completed event uses :class:`PatchApplyStatus` to set
    ``is_error`` (D3).
  * ``item/started`` carrying ``McpToolCallThreadItem`` →
    :class:`AgentToolUseEvent` (``name="<server>.<tool>"``); paired
    completed event surfaces :class:`McpToolCallResult` /
    :class:`McpToolCallError` (D4).
  * ``item/reasoning/*`` and ``item/{started,completed}`` carrying
    :class:`ReasoningThreadItem` → no runtime event (D5, decision #12).
  * ``turn/completed`` → :class:`AgentResultEvent`
    (``subtype="completed"``); ``is_error`` derives from
    :class:`TurnStatus` (D6).
  * ``error`` → :class:`AgentResultEvent` (``is_error=True``) with
    ``subtype`` classified via :func:`_classify_codex_error` as one of
    ``"error_auth"`` / ``"error_rate_limit"`` / ``"error_overload"``,
    falling back to ``"error"`` when no signal matches. The mid-turn
    provider error surfaces as a terminal failure (D7) rather than an
    idle hang.

Unknown notification methods are silently ignored — the SDK is still
churning and a future addition must not crash a running turn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from src.core.protocols.agent_event import (
    AgentResultEvent,
    AgentTextEvent,
    AgentToolResultEvent,
    AgentToolUseEvent,
)

if TYPE_CHECKING:
    from src.core.protocols.agent_event import AgentEventValue


_TERMINAL_COMMAND_STATUSES = frozenset({"completed", "failed", "declined"})
"""``CommandExecutionStatus`` values that close a tool-use with a result."""

_TERMINAL_PATCH_STATUSES = frozenset({"completed", "failed", "declined"})
"""``PatchApplyStatus`` values that close a file_change tool-use."""

_TERMINAL_MCP_STATUSES = frozenset({"completed", "failed"})
"""``McpToolCallStatus`` values that close an MCP tool-use."""


# D7 — Codex error classification (plan L762).
#
# ``CodexErrorInfoValue`` is a string enum on the SDK side
# (``codex_app_server/generated/v2_all.py:301``); structured variants
# (``HttpConnectionFailedCodexErrorInfo`` and friends, same module
# L321-L380) carry an ``http_status_code``. We classify both shapes plus a
# substring fallback on ``error.message`` so a payload that omits the
# structured info still surfaces a useful subtype to the lifecycle
# (``MessageStreamProcessor._handle_result_event`` keys off ``error_*``).

_AUTH_ENUM_VALUES = frozenset({"unauthorized"})
_RATE_LIMIT_ENUM_VALUES = frozenset({"usageLimitExceeded"})
_OVERLOAD_ENUM_VALUES = frozenset({"serverOverloaded"})

_AUTH_HTTP_STATUSES = frozenset({401, 403})
_RATE_LIMIT_HTTP_STATUSES = frozenset({429})
_OVERLOAD_HTTP_STATUSES = frozenset({500, 502, 503, 504, 529})

_AUTH_MESSAGE_MARKERS: tuple[str, ...] = (
    "unauthorized",
    "401",
    "forbidden",
    "invalid api key",
)
_RATE_LIMIT_MESSAGE_MARKERS: tuple[str, ...] = (
    "rate limit",
    "rate-limit",
    "ratelimit",
    "usage limit",
    "429",
)
_OVERLOAD_MESSAGE_MARKERS: tuple[str, ...] = (
    "overloaded",
    "server busy",
    "service unavailable",
    "503",
    "529",
)


def _enum_value(obj: object) -> str | None:
    """Best-effort coerce a status enum / pydantic ``RootModel`` to its string.

    ``codex_app_server`` exposes status fields as ``Enum`` subclasses
    (``CommandExecutionStatus`` etc.) whose ``value`` is the string. Tests
    use ``SimpleNamespace(value=...)`` look-alikes; either shape works.
    Returns ``None`` if no string can be extracted.
    """
    if obj is None:
        return None
    value = getattr(obj, "value", None)
    if isinstance(value, str):
        return value
    if isinstance(obj, str):
        return obj
    return None


def _unwrap_item(raw_item: object) -> object:
    """Dereference ``ThreadItem.root`` if present (pydantic ``RootModel``)."""
    return getattr(raw_item, "root", raw_item)


def _extract_http_status(obj: object, depth: int = 3) -> int | None:
    """Walk a ``CodexErrorInfo`` variant looking for ``http_status_code``.

    The structured variants (``HttpConnectionFailedCodexErrorInfo`` and
    friends, ``codex_app_server/generated/v2_all.py:321-380``) nest the
    status one level deep under variant-specific attribute names. Walking
    ``__dict__`` (or ``dict`` items) keeps us robust to SDK schema drift
    without binding to each variant's exact attribute name. The depth
    bound prevents pathological cycles in test fakes.
    """
    if obj is None or depth < 0:
        return None
    code = getattr(obj, "http_status_code", None)
    if isinstance(code, int):
        return code
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in ("http_status_code", "httpStatusCode") and isinstance(
                value, int
            ):
                return value
            found = _extract_http_status(value, depth - 1)
            if found is not None:
                return found
        return None
    inner = getattr(obj, "__dict__", None)
    if isinstance(inner, dict):
        for value in inner.values():
            found = _extract_http_status(value, depth - 1)
            if found is not None:
                return found
    return None


def _classify_codex_error(error_payload: object) -> str:
    """Return the ``AgentResultEvent.subtype`` for a Codex ``TurnError``.

    Resolves auth / rate-limit / overload from (in order):

      1. ``codex_error_info.root`` enum value (``CodexErrorInfoValue``).
      2. ``http_status_code`` on a structured ``CodexErrorInfo`` variant.
      3. Substring match on ``error.message``.

    Falls back to plain ``"error"`` when no signal matches. Subtypes
    starting with ``"error_"`` are recognised as terminal failures by
    :meth:`MessageStreamProcessor._handle_result_event`, so any of the
    classified values still drives the same lifecycle treatment as a
    bare error — the classification is metadata for diagnostics and
    retry logic, not a behavioural switch.
    """
    info = _unwrap_item(getattr(error_payload, "codex_error_info", None))
    enum_value = _enum_value(info)
    if enum_value is not None:
        if enum_value in _AUTH_ENUM_VALUES:
            return "error_auth"
        if enum_value in _RATE_LIMIT_ENUM_VALUES:
            return "error_rate_limit"
        if enum_value in _OVERLOAD_ENUM_VALUES:
            return "error_overload"

    status = _extract_http_status(info)
    if status is not None:
        if status in _AUTH_HTTP_STATUSES:
            return "error_auth"
        if status in _RATE_LIMIT_HTTP_STATUSES:
            return "error_rate_limit"
        if status in _OVERLOAD_HTTP_STATUSES:
            return "error_overload"

    message = str(getattr(error_payload, "message", "") or "")
    lowered = message.lower()
    if any(marker in lowered for marker in _AUTH_MESSAGE_MARKERS):
        return "error_auth"
    if any(marker in lowered for marker in _RATE_LIMIT_MESSAGE_MARKERS):
        return "error_rate_limit"
    if any(marker in lowered for marker in _OVERLOAD_MESSAGE_MARKERS):
        return "error_overload"

    return "error"


class CodexEventAdapter:
    """Stateful translator from Codex notifications to AgentEvents.

    State is per-instance because D1 dedup requires remembering which
    ``AgentMessageThreadItem`` ids streamed at least one delta; one
    adapter is owned by one :class:`CodexClient` (one Codex turn).
    """

    def __init__(self) -> None:
        self._seen_message_deltas: set[str] = set()

    def to_events(self, notification: object) -> list[AgentEventValue]:
        """Translate a single Codex notification into 0+ ``AgentEvent`` values."""
        method = getattr(notification, "method", None)
        payload = getattr(notification, "payload", None)
        if not isinstance(method, str) or payload is None:
            return []

        if method == "item/agentMessage/delta":
            return self._handle_message_delta(payload)
        if method == "item/started":
            return self._handle_item_started(payload)
        if method == "item/completed":
            return self._handle_item_completed(payload)
        if method == "turn/completed":
            return self._handle_turn_completed(payload)
        if method == "error":
            return self._handle_error(payload)

        # ``item/reasoning/*`` notifications, ``thread/*`` lifecycle
        # notifications, and any future SDK additions fall through to
        # an empty list — the pipeline never sees them. Reasoning is
        # tee'd to an evidence log by Phase F (T013); the adapter
        # discards them here per decision #12.
        return []

    # ------------------------------------------------------------------
    # D1 — agent message
    # ------------------------------------------------------------------

    def _handle_message_delta(self, payload: object) -> list[AgentEventValue]:
        delta = getattr(payload, "delta", "") or ""
        item_id = str(getattr(payload, "item_id", "") or "")
        if item_id:
            self._seen_message_deltas.add(item_id)
        return [AgentTextEvent(text=str(delta))]

    def _handle_message_completed(
        self, item: object, item_id: str
    ) -> list[AgentEventValue]:
        if item_id and item_id in self._seen_message_deltas:
            # Streaming deltas already covered the text; emitting the
            # completed item again would double-print for consumers
            # that simply concatenate text events.
            return []
        text = getattr(item, "text", "") or ""
        if not text:
            return []
        return [AgentTextEvent(text=str(text))]

    # ------------------------------------------------------------------
    # item/started + item/completed dispatch
    # ------------------------------------------------------------------

    def _handle_item_started(self, payload: object) -> list[AgentEventValue]:
        item = _unwrap_item(getattr(payload, "item", None))
        if item is None:
            return []
        item_type = getattr(item, "type", None)
        if item_type == "commandExecution":
            return self._command_started(item)
        if item_type == "fileChange":
            return self._file_change_started(item)
        if item_type == "mcpToolCall":
            return self._mcp_started(item)
        # ``agentMessage`` / ``reasoning`` / unknown item types do not
        # produce a tool-use start; agentMessage text is delivered via
        # ``item/agentMessage/delta`` + completed-item dedup.
        return []

    def _handle_item_completed(self, payload: object) -> list[AgentEventValue]:
        item = _unwrap_item(getattr(payload, "item", None))
        if item is None:
            return []
        item_id = str(getattr(item, "id", "") or "")
        item_type = getattr(item, "type", None)
        if item_type == "agentMessage":
            return self._handle_message_completed(item, item_id)
        if item_type == "commandExecution":
            return self._command_completed(item, item_id)
        if item_type == "fileChange":
            return self._file_change_completed(item, item_id)
        if item_type == "mcpToolCall":
            return self._mcp_completed(item, item_id)
        # ``reasoning`` items: drop per decision #12.
        return []

    # ------------------------------------------------------------------
    # D2 — command execution
    # ------------------------------------------------------------------

    def _command_started(self, item: object) -> list[AgentEventValue]:
        item_id = str(getattr(item, "id", "") or "")
        command = getattr(item, "command", "") or ""
        cwd = getattr(item, "cwd", "")
        # ``cwd`` may be a pydantic ``RootModel[str]``; coerce to plain str.
        cwd_str = str(getattr(cwd, "root", cwd)) if cwd is not None else ""
        return [
            AgentToolUseEvent(
                id=item_id,
                name="bash",
                input={"command": str(command), "cwd": cwd_str},
            )
        ]

    def _command_completed(self, item: object, item_id: str) -> list[AgentEventValue]:
        status = _enum_value(getattr(item, "status", None))
        if status not in _TERMINAL_COMMAND_STATUSES:
            return []
        # ``status == "completed"`` means the command ran to exit; we
        # still consult ``exit_code`` so a non-zero shell return marks
        # the result as an error, which is what the lint cache needs to
        # NOT mark the lint run as successful (see
        # ``MessageStreamProcessor._handle_tool_result_event``).
        exit_code = getattr(item, "exit_code", None)
        if status != "completed":
            is_error = True
        else:
            is_error = exit_code not in (None, 0)
        content = getattr(item, "aggregated_output", None)
        return [
            AgentToolResultEvent(
                tool_use_id=item_id,
                is_error=is_error,
                content=content,
            )
        ]

    # ------------------------------------------------------------------
    # D3 — file change
    # ------------------------------------------------------------------

    def _file_change_started(self, item: object) -> list[AgentEventValue]:
        item_id = str(getattr(item, "id", "") or "")
        changes = getattr(item, "changes", None) or []
        normalised: list[dict[str, Any]] = []
        for change in changes:
            normalised.append(
                {
                    "path": str(getattr(change, "path", "") or ""),
                    "diff": str(getattr(change, "diff", "") or ""),
                    "kind": _enum_value(getattr(change, "kind", None))
                    or str(getattr(change, "kind", "") or ""),
                }
            )
        return [
            AgentToolUseEvent(
                id=item_id,
                name="file_change",
                input={"changes": normalised},
            )
        ]

    def _file_change_completed(
        self, item: object, item_id: str
    ) -> list[AgentEventValue]:
        status = _enum_value(getattr(item, "status", None))
        if status not in _TERMINAL_PATCH_STATUSES:
            return []
        is_error = status != "completed"
        return [
            AgentToolResultEvent(
                tool_use_id=item_id,
                is_error=is_error,
                content=getattr(item, "changes", None),
            )
        ]

    # ------------------------------------------------------------------
    # D4 — MCP tool call
    # ------------------------------------------------------------------

    def _mcp_started(self, item: object) -> list[AgentEventValue]:
        item_id = str(getattr(item, "id", "") or "")
        server = str(getattr(item, "server", "") or "")
        tool = str(getattr(item, "tool", "") or "")
        arguments = getattr(item, "arguments", None)
        input_dict: dict[str, Any]
        if isinstance(arguments, dict):
            input_dict = dict(arguments)
        else:
            input_dict = {"arguments": arguments}
        return [
            AgentToolUseEvent(
                id=item_id,
                name=f"{server}.{tool}",
                input=input_dict,
            )
        ]

    def _mcp_completed(self, item: object, item_id: str) -> list[AgentEventValue]:
        status = _enum_value(getattr(item, "status", None))
        if status not in _TERMINAL_MCP_STATUSES:
            return []
        is_error = status != "completed"
        # Surface the structured ``McpToolCallResult`` on success, the
        # ``McpToolCallError`` on failure — matches the plan's
        # "content=McpToolCallResult" note (D4).
        content: object
        error = getattr(item, "error", None)
        if is_error and error is not None:
            content = error
        else:
            content = getattr(item, "result", None)
        return [
            AgentToolResultEvent(
                tool_use_id=item_id,
                is_error=is_error,
                content=content,
            )
        ]

    # ------------------------------------------------------------------
    # D6 — turn completed
    # ------------------------------------------------------------------

    def _handle_turn_completed(self, payload: object) -> list[AgentEventValue]:
        thread_id = str(getattr(payload, "thread_id", "") or "")
        turn_obj = getattr(payload, "turn", None)
        status_value = _enum_value(getattr(turn_obj, "status", None))
        is_error = bool(status_value) and status_value != "completed"
        # ``MessageStreamProcessor._handle_result_event`` copies
        # ``result`` into ``lifecycle_ctx.final_result`` and exposes it
        # as ``AgentSessionOutput.summary`` for retry author context.
        # Surfacing the ``TurnStatus`` string keeps that summary human
        # readable; passing the whole ``Turn`` object would render as
        # an opaque pydantic ``repr``.
        return [
            AgentResultEvent(
                session_id=thread_id,
                is_error=is_error,
                subtype="completed",
                result=status_value or "",
            )
        ]

    # ------------------------------------------------------------------
    # D7 — error notification
    # ------------------------------------------------------------------

    def _handle_error(self, payload: object) -> list[AgentEventValue]:
        thread_id = str(getattr(payload, "thread_id", "") or "")
        error = getattr(payload, "error", None)
        return [
            AgentResultEvent(
                session_id=thread_id,
                is_error=True,
                subtype=_classify_codex_error(error),
                result=error,
            )
        ]

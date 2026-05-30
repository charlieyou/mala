"""MessageStreamProcessor: AgentEvent stream processing component.

Consumes the coder-agnostic ``AgentEvent`` stream emitted by adapters
(Claude via :func:`src.core.protocols.agent_event.to_agent_events`, Amp via
:meth:`AmpClient.receive_response`) and drives the lifecycle/lint side
effects. This module handles:
- Wrapping event streams with idle timeout detection.
- Iterating and processing :class:`AgentEvent`s (text/tool_use/tool_result/result).
- Tracking tool calls and lint cache updates.

Design principles:
- Branch on ``event.kind`` only — no provider-specific class-name duck typing.
- Explicit state management via :class:`MessageIterationState`.
- Callbacks for external operations (text/tool notifications).
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    TypeVar,
    cast,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import Self

    from src.domain.lifecycle import LifecycleContext
    from src.infra.telemetry import TelemetrySpan


class LintCacheProtocol(Protocol):
    """Protocol for lint cache operations used by stream processor."""

    def detect_lint_command(self, command: str) -> str | None:
        """Detect if command is a lint command and return lint type."""
        ...

    def mark_success(self, lint_type: str, command: str) -> None:
        """Mark a lint command as successful."""
        ...


logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# Parse the background-launch acknowledgement tool_result, e.g.
# "Command running in background with ID: bu1538lol. ..." — used to learn the
# launch's task_id when the ``task_started`` system event was not observed.
_BG_LAUNCH_ID_RE = re.compile(r"background with ID:\s*(\S+)")

# Parse the ``<status>...</status>`` field a TaskOutput/BashOutput retrieval
# tool_result reports. Only a terminal status proves the task actually finished
# (block=False polling reports ``running``/``pending`` and must NOT clear the wait).
_TASK_RESULT_STATUS_RE = re.compile(r"<status>\s*(\w+)\s*</status>", re.IGNORECASE)
_TERMINAL_TASK_STATUSES = frozenset({"completed", "failed", "stopped"})


def _strip_trailing_punct(value: str) -> str:
    """Drop sentence punctuation the SDK appends after ids/paths in prose."""
    return value.strip().rstrip(".")


def _extract_tool_result_text(content: object) -> str:
    """Best-effort plain-text view of a tool_result's ``content``.

    The Anthropic SDK delivers ``ToolResultBlock.content`` either as a string or
    as a list of content blocks (dicts/objects carrying ``text``). Both shapes
    are flattened so launch-acknowledgement parsing works regardless.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = cast("dict[str, Any]", item).get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


class IdleTimeoutError(Exception):
    """Raised when the SDK response stream is idle for too long."""


class IdleTimeoutStream(Generic[_T]):
    """Wrap an async iterator with idle timeout detection.

    Raises IdleTimeoutError if no message received within timeout,
    unless pending_tool_ids is non-empty (tool execution in progress).
    """

    def __init__(
        self,
        stream: AsyncIterator[_T],
        timeout_seconds: float | None,
        pending_tool_ids: set[str],
    ) -> None:
        self._stream: AsyncIterator[_T] = stream
        self._timeout_seconds = timeout_seconds
        self._pending_tool_ids = pending_tool_ids

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> _T:
        if self._timeout_seconds is None:
            return await self._stream.__anext__()
        # Disable timeout if tools are pending (execution in progress)
        current_timeout = None if self._pending_tool_ids else self._timeout_seconds
        try:
            return await asyncio.wait_for(
                self._stream.__anext__(),
                timeout=current_timeout,
            )
        except TimeoutError as exc:
            raise IdleTimeoutError(
                f"SDK stream idle for {self._timeout_seconds:.0f} seconds"
            ) from exc


@dataclass
class MessageIterationState:
    """Mutable state for message iteration within a session.

    Used to track state that evolves during SDK message streaming
    and idle retry handling.

    Attributes:
        session_id: SDK session ID (updated when ResultMessage received).
        pending_session_id: Session ID to use for resuming after idle timeout.
        tool_calls_this_turn: Number of tool calls in the current turn.
        idle_retry_count: Number of idle timeout retries attempted.
        pending_tool_ids: Set of tool IDs awaiting results.
        pending_lint_commands: Map of tool_use_id to (lint_type, command).
        first_message_received: Whether any message was received in current turn.
        pending_background_tool_ids: tool_use_ids of Bash launches started with
            ``run_in_background=true`` that the agent has not yet handled.
            Populated in :meth:`_handle_tool_use_event` and cleared when a matching
            ``task_completed`` event is observed, or when the agent retrieves the
            task inline via ``TaskOutput``/``BashOutput`` *and* that retrieval's
            tool_result reports a terminal status (completed/failed/stopped).
        background_launch_commands: Map of tool_use_id to the launch command text,
            recorded for background Bash launches when cheaply available.
        background_task_ids: Map of tool_use_id to SDK task_id, recorded when a
            ``task_started`` event is observed so the wait/resume loop can stop the
            task on timeout even when the started event was consumed mid-turn.
        pending_bg_retrievals: Map of an in-flight ``TaskOutput``/``BashOutput``
            retrieval's tool_use_id to the background launch's tool_use_id it
            targets. Recorded when the agent *requests* a retrieval; resolved when
            the retrieval's tool_result arrives — the launch is cleared from
            ``pending_background_tool_ids`` only if that result reports a terminal
            status (so block=False polling that says ``running`` does not skip a
            still-needed wait).
    """

    session_id: str | None = None
    pending_session_id: str | None = None
    tool_calls_this_turn: int = 0
    idle_retry_count: int = 0
    pending_tool_ids: set[str] = field(default_factory=set)
    pending_lint_commands: dict[str, tuple[str, str]] = field(default_factory=dict)
    first_message_received: bool = False
    pending_background_tool_ids: set[str] = field(default_factory=set)
    background_launch_commands: dict[str, str] = field(default_factory=dict)
    background_task_ids: dict[str, str] = field(default_factory=dict)
    pending_bg_retrievals: dict[str, str] = field(default_factory=dict)


@dataclass
class MessageIterationResult:
    """Result from a message iteration.

    Attributes:
        success: Whether the iteration completed successfully.
        session_id: Updated session ID (if received).
        error: Error text when the SDK reports an unsuccessful result.
        pending_query: Next query to send (for retries), or None if complete.
        pending_session_id: Session ID to use for next query.
        idle_retry_count: Updated idle retry count.
        pending_background_tool_ids: tool_use_ids of background Bash launches whose
            ``task_completed`` event was not observed before the turn ended. The
            wait/resume loop in ``execute_iteration`` reads this to decide whether to
            keep the SDK client connected and wait for the task notification.
        background_task_ids: Map of tool_use_id to SDK task_id seen on
            ``task_started`` events during the turn, so the wait/resume loop can
            stop a task on timeout even if no completion event arrives.
    """

    success: bool
    session_id: str | None = None
    error: str | None = None
    pending_query: str | None = None
    pending_session_id: str | None = None
    idle_retry_count: int = 0
    pending_background_tool_ids: frozenset[str] = frozenset()
    background_task_ids: dict[str, str] = field(default_factory=dict)


# Callbacks for SDK message streaming events
ToolUseCallback = Callable[[str, str, dict[str, Any] | None], None]
AgentTextCallback = Callable[[str, str], None]


@dataclass
class StreamProcessorConfig:
    """Configuration for MessageStreamProcessor."""

    pass


@dataclass
class StreamProcessorCallbacks:
    """Callbacks for stream processing events.

    Attributes:
        on_tool_use: Called for each tool_use AgentEvent.
        on_agent_text: Called for each text AgentEvent.
    """

    on_tool_use: ToolUseCallback | None = None
    on_agent_text: AgentTextCallback | None = None


class MessageStreamProcessor:
    """Processes ``AgentEvent`` streams.

    Iterates an :class:`AgentEvent` stream, tracking tool calls, updating
    the lint cache, and detecting idle timeouts. Branches on
    ``event.kind`` — no provider-specific class-name checks.

    Usage:
        processor = MessageStreamProcessor(config, callbacks)
        result = await processor.process_stream(
            stream, issue_id, state, lifecycle_ctx, lint_cache, query_start, tracer
        )
    """

    def __init__(
        self,
        config: StreamProcessorConfig | None = None,
        callbacks: StreamProcessorCallbacks | None = None,
    ) -> None:
        self.config = config or StreamProcessorConfig()
        self.callbacks = callbacks or StreamProcessorCallbacks()

    async def process_stream(
        self,
        stream: AsyncIterator[Any],
        issue_id: str,
        state: MessageIterationState,
        lifecycle_ctx: LifecycleContext,
        lint_cache: LintCacheProtocol,
        query_start: float,
        tracer: TelemetrySpan | None,
    ) -> MessageIterationResult:
        """Process an ``AgentEvent`` stream and update state.

        Updates state.session_id, state.tool_calls_this_turn, state.pending_tool_ids,
        and lint_cache on successful lint commands.

        Args:
            stream: The ``AgentEvent`` stream to process.
            issue_id: Issue ID for logging.
            state: Mutable state for the iteration.
            lifecycle_ctx: Lifecycle context for session state.
            lint_cache: Cache for lint command results.
            query_start: Timestamp when query was sent.
            tracer: Optional telemetry span context.

        Returns:
            MessageIterationResult with success status.
        """
        result_error: str | None = None
        pending_text_delta = ""
        try:
            async for event in stream:
                if not state.first_message_received:
                    state.first_message_received = True
                    latency = time.time() - query_start
                    logger.debug(
                        "Session %s: first message after %.1fs",
                        issue_id,
                        latency,
                    )
                if tracer is not None:
                    tracer.log_message(event)

                kind = getattr(event, "kind", None)
                if kind == "text":
                    pending_text_delta = self._handle_text_event(
                        event, issue_id, pending_text_delta
                    )
                elif kind == "tool_use":
                    self._flush_text_delta(issue_id, pending_text_delta)
                    pending_text_delta = ""
                    self._handle_tool_use_event(event, issue_id, state, lint_cache)
                elif kind == "tool_result":
                    self._flush_text_delta(issue_id, pending_text_delta)
                    pending_text_delta = ""
                    self._handle_tool_result_event(event, state, lint_cache)
                elif kind == "task_started":
                    self._handle_task_started_event(event, issue_id, state)
                elif kind == "task_completed":
                    self._handle_task_completed_event(event, issue_id, state)
                elif kind == "result":
                    self._flush_text_delta(issue_id, pending_text_delta)
                    pending_text_delta = ""
                    result_error = self._handle_result_event(
                        event, issue_id, state, lifecycle_ctx
                    )
        finally:
            self._flush_text_delta(issue_id, pending_text_delta)

        # Success
        stream_duration = time.time() - query_start
        logger.debug(
            "Session %s: stream complete after %.1fs, %d tool calls",
            issue_id,
            stream_duration,
            state.tool_calls_this_turn,
        )
        return MessageIterationResult(
            success=result_error is None,
            session_id=state.session_id,
            error=result_error,
            idle_retry_count=0,
            pending_background_tool_ids=frozenset(state.pending_background_tool_ids),
            background_task_ids=dict(state.background_task_ids),
        )

    def _handle_text_event(
        self, event: object, issue_id: str, pending_text_delta: str
    ) -> str:
        if self.callbacks.on_agent_text is None:
            return ""
        text = getattr(event, "text", "")
        if bool(getattr(event, "is_delta", False)):
            return pending_text_delta + str(text)
        self._flush_text_delta(issue_id, pending_text_delta)
        self.callbacks.on_agent_text(issue_id, text)
        return ""

    def _flush_text_delta(self, issue_id: str, pending_text_delta: str) -> None:
        if not pending_text_delta or self.callbacks.on_agent_text is None:
            return
        self.callbacks.on_agent_text(issue_id, pending_text_delta)

    def _handle_tool_use_event(
        self,
        event: object,
        issue_id: str,
        state: MessageIterationState,
        lint_cache: LintCacheProtocol,
    ) -> None:
        state.tool_calls_this_turn += 1
        block_id = getattr(event, "id", "")
        state.pending_tool_ids.add(block_id)
        name = getattr(event, "name", "")
        block_input = getattr(event, "input", {}) or {}
        if self.callbacks.on_tool_use is not None:
            self.callbacks.on_tool_use(issue_id, name, block_input)
        if isinstance(name, str) and name.lower() == "bash":
            cmd = (
                block_input.get("command", "") if isinstance(block_input, dict) else ""
            )
            lint_type = lint_cache.detect_lint_command(cmd)
            if lint_type:
                state.pending_lint_commands[block_id] = (lint_type, cmd)
            if (
                isinstance(block_input, dict)
                and block_input.get("run_in_background") is True
                and block_id
            ):
                state.pending_background_tool_ids.add(block_id)
                if cmd:
                    state.background_launch_commands[block_id] = str(cmd)
                logger.debug(
                    "Session %s: background Bash launch detected (tool_use_id=%s)",
                    issue_id,
                    block_id,
                )
        elif isinstance(name, str) and name.lower() in ("taskoutput", "bashoutput"):
            # The agent is retrieving a backgrounded task. Record which launch
            # this retrieval targets, but do NOT clear the launch yet: only the
            # retrieval's tool_result proves the task actually finished. Clearing
            # on the request would wrongly skip the wait for a block=False poll
            # that merely reports the task is still running.
            self._record_bg_retrieval_request(state, block_id, block_input)

    def _handle_tool_result_event(
        self,
        event: object,
        state: MessageIterationState,
        lint_cache: LintCacheProtocol,
    ) -> None:
        tool_use_id = getattr(event, "tool_use_id", None)
        if tool_use_id:
            state.pending_tool_ids.discard(tool_use_id)
            if tool_use_id in state.pending_background_tool_ids:
                self._record_background_launch_metadata(
                    state,
                    tool_use_id,
                    _extract_tool_result_text(getattr(event, "content", None)),
                )
            if tool_use_id in state.pending_bg_retrievals:
                self._resolve_bg_retrieval(
                    state,
                    tool_use_id,
                    is_error=bool(getattr(event, "is_error", False)),
                    result_text=_extract_tool_result_text(
                        getattr(event, "content", None)
                    ),
                )
        if tool_use_id in state.pending_lint_commands:
            lint_type, cmd = state.pending_lint_commands.pop(tool_use_id)
            if not getattr(event, "is_error", False):
                lint_cache.mark_success(lint_type, cmd)

    def _record_background_launch_metadata(
        self, state: MessageIterationState, tool_use_id: str, ack_text: str
    ) -> None:
        """Capture the launch's task_id from its acknowledgement tool_result.

        The launch tool_result reads like "Command running in background with ID:
        <id>. ..." Recording the id (keyed by the launch's tool_use_id) lets
        later retrieval detection correlate ``TaskOutput(task_id=...)`` back to
        the launch even when the ``task_started`` system event was not observed.
        """
        if not ack_text:
            return
        id_match = _BG_LAUNCH_ID_RE.search(ack_text)
        if id_match:
            state.background_task_ids.setdefault(
                tool_use_id, _strip_trailing_punct(id_match.group(1))
            )

    def _record_bg_retrieval_request(
        self,
        state: MessageIterationState,
        retrieval_tool_use_id: str,
        block_input: dict[str, Any],
    ) -> None:
        """Map a TaskOutput/BashOutput retrieval to the launch it targets.

        No clearing happens here — the launch stays pending until the retrieval's
        tool_result confirms a terminal status (see :meth:`_resolve_bg_retrieval`).
        """
        if not retrieval_tool_use_id:
            return
        task_id = str(block_input.get("task_id") or block_input.get("bash_id") or "")
        if not task_id:
            return
        for launch_id in state.pending_background_tool_ids:
            if state.background_task_ids.get(launch_id) == task_id:
                state.pending_bg_retrievals[retrieval_tool_use_id] = launch_id
                return

    def _resolve_bg_retrieval(
        self,
        state: MessageIterationState,
        retrieval_tool_use_id: str,
        *,
        is_error: bool,
        result_text: str,
    ) -> None:
        """Clear a launch only when its retrieval returned a terminal result.

        The SDK delivers a backgrounded task's completion only as a queued
        next-turn prompt (never a standalone notification on
        ``receive_messages()``), so a launch the agent successfully retrieved
        inline will never surface a notification to wait on — drop it. But a
        ``block=False`` poll that reports ``running`` (or an errored retrieval)
        must keep the launch pending so the wait still runs.
        """
        launch_id = state.pending_bg_retrievals.pop(retrieval_tool_use_id, None)
        if launch_id is None:
            return
        if is_error:
            return
        status_match = _TASK_RESULT_STATUS_RE.search(result_text)
        status = status_match.group(1).lower() if status_match else ""
        if status in _TERMINAL_TASK_STATUSES:
            state.pending_background_tool_ids.discard(launch_id)
            logger.debug(
                "Background launch %s cleared via terminal inline retrieval "
                "(status=%s)",
                launch_id,
                status,
            )

    def _handle_task_started_event(
        self, event: object, issue_id: str, state: MessageIterationState
    ) -> None:
        """Process a ``task_started`` AgentEvent.

        Emitted by the Claude SDK when a backgrounded task begins. The launch is
        already recorded from the originating ``tool_use`` event; additionally
        record the ``tool_use_id -> task_id`` mapping so the wait/resume loop can
        stop the task on timeout even when this event was consumed during the
        launch turn (and so never reappears on the continuous stream).
        """
        task_id = str(getattr(event, "task_id", "") or "")
        tool_use_id = str(getattr(event, "tool_use_id", "") or "")
        if tool_use_id and task_id:
            state.background_task_ids[tool_use_id] = task_id
        logger.debug(
            "Session %s: background task started (task_id=%s, tool_use_id=%s)",
            issue_id,
            task_id,
            tool_use_id,
        )

    def _handle_task_completed_event(
        self,
        event: object,
        issue_id: str,
        state: MessageIterationState,
    ) -> None:
        """Process a ``task_completed`` AgentEvent.

        When a backgrounded task finishes within the same turn that launched it
        (a short task), the SDK pushes the notification before the turn ends.
        Clear the matching launch from the pending set so the wait/resume loop in
        ``execute_iteration`` does not block on a task that is already done.
        """
        tool_use_id = str(getattr(event, "tool_use_id", "") or "")
        task_id = str(getattr(event, "task_id", "") or "")
        status = str(getattr(event, "status", "") or "")
        if tool_use_id and tool_use_id in state.pending_background_tool_ids:
            state.pending_background_tool_ids.discard(tool_use_id)
            logger.debug(
                "Session %s: background task completed within turn "
                "(tool_use_id=%s, task_id=%s, status=%s)",
                issue_id,
                tool_use_id,
                task_id,
                status,
            )

    def _handle_result_event(
        self,
        event: object,
        issue_id: str,
        state: MessageIterationState,
        lifecycle_ctx: LifecycleContext,
    ) -> str | None:
        """Process a result AgentEvent.

        Returns an error string when the result reports an unsuccessful turn.
        Amp exposes transient stream failures as result events (for example
        ``error_during_execution``) rather than Python exceptions; callers
        must not treat those as completed agent turns.
        """
        del issue_id  # logged via state mutation; reserved for future tracing
        state.session_id = getattr(event, "session_id", None)
        lifecycle_ctx.session_id = state.session_id
        result = getattr(event, "result", "") or ""
        lifecycle_ctx.final_result = result

        subtype = str(getattr(event, "subtype", "") or "")
        result_text = str(result)
        subtype_lower = subtype.lower()
        result_text_lower = result_text.lower()
        is_error = bool(getattr(event, "is_error", False))
        if (
            is_error
            or subtype_lower.startswith("error_")
            or result_text_lower.startswith("error_")
        ):
            return result_text or subtype or "SDK result reported an error"
        return None

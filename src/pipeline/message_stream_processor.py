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
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    Protocol,
    TypeVar,
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
    """

    session_id: str | None = None
    pending_session_id: str | None = None
    tool_calls_this_turn: int = 0
    idle_retry_count: int = 0
    pending_tool_ids: set[str] = field(default_factory=set)
    pending_lint_commands: dict[str, tuple[str, str]] = field(default_factory=dict)
    first_message_received: bool = False


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
    """

    success: bool
    session_id: str | None = None
    error: str | None = None
    pending_query: str | None = None
    pending_session_id: str | None = None
    idle_retry_count: int = 0


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
                self._handle_text_event(event, issue_id)
            elif kind == "tool_use":
                self._handle_tool_use_event(event, issue_id, state, lint_cache)
            elif kind == "tool_result":
                self._handle_tool_result_event(event, state, lint_cache)
            elif kind == "result":
                result_error = self._handle_result_event(
                    event, issue_id, state, lifecycle_ctx
                )

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
        )

    def _handle_text_event(self, event: object, issue_id: str) -> None:
        if self.callbacks.on_agent_text is None:
            return
        text = getattr(event, "text", "")
        self.callbacks.on_agent_text(issue_id, text)

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

    def _handle_tool_result_event(
        self,
        event: object,
        state: MessageIterationState,
        lint_cache: LintCacheProtocol,
    ) -> None:
        tool_use_id = getattr(event, "tool_use_id", None)
        if tool_use_id:
            state.pending_tool_ids.discard(tool_use_id)
        if tool_use_id in state.pending_lint_commands:
            lint_type, cmd = state.pending_lint_commands.pop(tool_use_id)
            if not getattr(event, "is_error", False):
                lint_cache.mark_success(lint_type, cmd)

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

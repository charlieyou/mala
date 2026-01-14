"""MessageStreamProcessor: SDK stream processing component.

Extracted from AgentSessionRunner to separate stream iteration logic from
session lifecycle management. This module handles:
- Wrapping SDK streams with idle timeout detection
- Iterating and processing SDK messages (AssistantMessage, ResultMessage)
- Tracking tool calls and lint cache updates
- Context pressure detection from usage data

Design principles:
- Protocol-based message/block checks for testability (no SDK imports at runtime)
- Explicit state management via MessageIterationState
- Callbacks for external operations (text/tool notifications)
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


class ContextPressureError(Exception):
    """Raised when context usage exceeds the restart threshold.

    This exception signals that the agent session should be checkpointed
    and restarted with a fresh context to avoid context exhaustion.

    Attributes:
        session_id: SDK session ID for checkpoint query.
        input_tokens: Current input token count.
        output_tokens: Current output token count.
        cache_read_tokens: Current cache read token count.
        pressure_ratio: Ratio of usage to limit (e.g., 0.92 = 92%).
    """

    def __init__(
        self,
        session_id: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        pressure_ratio: float,
    ) -> None:
        self.session_id = session_id
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_tokens = cache_read_tokens
        self.pressure_ratio = pressure_ratio
        super().__init__(
            f"Context pressure {pressure_ratio:.1%} exceeds threshold "
            f"(input={input_tokens}, output={output_tokens}, session={session_id})"
        )


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
        pending_query: Next query to send (for retries), or None if complete.
        pending_session_id: Session ID to use for next query.
        idle_retry_count: Updated idle retry count.
    """

    success: bool
    session_id: str | None = None
    pending_query: str | None = None
    pending_session_id: str | None = None
    idle_retry_count: int = 0


# Callbacks for SDK message streaming events
ToolUseCallback = Callable[[str, str, dict[str, Any] | None], None]
AgentTextCallback = Callable[[str, str], None]


@dataclass
class StreamProcessorConfig:
    """Configuration for MessageStreamProcessor.

    Attributes:
        context_limit: Maximum context tokens for pressure detection.
        context_restart_threshold: Ratio (0.0-1.0) at which to raise ContextPressureError.
    """

    context_limit: int = 100_000
    context_restart_threshold: float = 0.70


@dataclass
class StreamProcessorCallbacks:
    """Callbacks for stream processing events.

    Attributes:
        on_tool_use: Called when ToolUseBlock is encountered.
        on_agent_text: Called when TextBlock is encountered.
    """

    on_tool_use: ToolUseCallback | None = None
    on_agent_text: AgentTextCallback | None = None


class MessageStreamProcessor:
    """Processes SDK message streams.

    Handles iteration over SDK streams, tracking tool calls, updating lint cache,
    and detecting context pressure. Uses duck typing for SDK message types to
    avoid SDK imports at runtime.

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
        """Process SDK message stream and update state.

        Updates state.session_id, state.tool_calls_this_turn, state.pending_tool_ids,
        and lint_cache on successful lint commands.

        Args:
            stream: The message stream to process.
            issue_id: Issue ID for logging.
            state: Mutable state for the iteration.
            lifecycle_ctx: Lifecycle context for session state.
            lint_cache: Cache for lint command results.
            query_start: Timestamp when query was sent.
            tracer: Optional telemetry span context.

        Returns:
            MessageIterationResult with success status.

        Raises:
            ContextPressureError: If context pressure exceeds threshold.
        """
        # Use duck typing to avoid SDK imports - check type name instead of isinstance
        async for message in stream:
            if not state.first_message_received:
                state.first_message_received = True
                latency = time.time() - query_start
                logger.debug(
                    "Session %s: first message after %.1fs",
                    issue_id,
                    latency,
                )
            if tracer is not None:
                tracer.log_message(message)

            msg_type = type(message).__name__
            if msg_type == "AssistantMessage":
                self._process_assistant_message(message, issue_id, state, lint_cache)

            elif msg_type == "ResultMessage":
                self._process_result_message(message, issue_id, state, lifecycle_ctx)

        # Success
        stream_duration = time.time() - query_start
        logger.debug(
            "Session %s: stream complete after %.1fs, %d tool calls",
            issue_id,
            stream_duration,
            state.tool_calls_this_turn,
        )
        return MessageIterationResult(
            success=True,
            session_id=state.session_id,
            idle_retry_count=0,
        )

    def _process_assistant_message(
        self,
        message: object,
        issue_id: str,
        state: MessageIterationState,
        lint_cache: LintCacheProtocol,
    ) -> None:
        """Process an AssistantMessage, handling text/tool blocks."""
        content = getattr(message, "content", [])
        for block in content:
            block_type = type(block).__name__
            if block_type == "TextBlock":
                text = getattr(block, "text", "")
                if self.callbacks.on_agent_text is not None:
                    self.callbacks.on_agent_text(issue_id, text)
            elif block_type == "ToolUseBlock":
                state.tool_calls_this_turn += 1
                block_id = getattr(block, "id", "")
                state.pending_tool_ids.add(block_id)
                name = getattr(block, "name", "")
                block_input = getattr(block, "input", {})
                if self.callbacks.on_tool_use is not None:
                    self.callbacks.on_tool_use(issue_id, name, block_input)
                if name.lower() == "bash":
                    cmd = block_input.get("command", "")
                    lint_type = lint_cache.detect_lint_command(cmd)
                    if lint_type:
                        state.pending_lint_commands[block_id] = (
                            lint_type,
                            cmd,
                        )
            elif block_type == "ToolResultBlock":
                tool_use_id = getattr(block, "tool_use_id", None)
                if tool_use_id:
                    state.pending_tool_ids.discard(tool_use_id)
                if tool_use_id in state.pending_lint_commands:
                    lint_type, cmd = state.pending_lint_commands.pop(tool_use_id)
                    if not getattr(block, "is_error", False):
                        lint_cache.mark_success(lint_type, cmd)

    def _process_result_message(
        self,
        message: object,
        issue_id: str,
        state: MessageIterationState,
        lifecycle_ctx: LifecycleContext,
    ) -> None:
        """Process a ResultMessage, extracting session ID and usage.

        Raises:
            ContextPressureError: If context pressure exceeds threshold.
        """
        state.session_id = getattr(message, "session_id", None)
        lifecycle_ctx.session_id = state.session_id
        lifecycle_ctx.final_result = getattr(message, "result", "") or ""

        # Extract token usage from SDK for context pressure detection
        usage = getattr(message, "usage", None)
        if usage is not None:
            # Handle both dict and object forms of usage
            if isinstance(usage, dict):
                input_tokens = usage.get("input_tokens", 0) or 0
                output_tokens = usage.get("output_tokens", 0) or 0
                cache_read = usage.get("cache_read_input_tokens", 0) or 0
            else:
                input_tokens = getattr(usage, "input_tokens", 0) or 0
                output_tokens = getattr(usage, "output_tokens", 0) or 0
                cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

            # Accumulate usage across turns
            lifecycle_ctx.context_usage.add_turn(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read,
            )

            # Check context pressure threshold
            pressure = lifecycle_ctx.context_usage.pressure_ratio(
                self.config.context_limit
            )
            logger.debug(
                "Context usage: input=%d output=%d cache_read=%d limit=%d pressure=%.1f%%",
                lifecycle_ctx.context_usage.input_tokens,
                lifecycle_ctx.context_usage.output_tokens,
                lifecycle_ctx.context_usage.cache_read_tokens,
                self.config.context_limit,
                pressure * 100,
            )
            if pressure >= self.config.context_restart_threshold:
                # session_id was already extracted above
                raise ContextPressureError(
                    session_id=state.session_id or "",
                    input_tokens=lifecycle_ctx.context_usage.input_tokens,
                    output_tokens=lifecycle_ctx.context_usage.output_tokens,
                    cache_read_tokens=lifecycle_ctx.context_usage.cache_read_tokens,
                    pressure_ratio=pressure,
                )
        else:
            logger.warning(
                "Session %s: ResultMessage missing usage field, "
                "context pressure tracking disabled",
                issue_id,
            )
            lifecycle_ctx.context_usage.disable_tracking()

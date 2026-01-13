"""Idle timeout retry policy for AgentSessionRunner.

This module contains the IdleTimeoutRetryPolicy class which encapsulates
idle timeout handling with backoff semantics, extracted from AgentSessionRunner.

Design principles:
- Protocol-based dependencies for testability
- Explicit state management via MessageIterationState
- Preserves exact async behavior including asyncio.shield patterns
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.pipeline.message_stream_processor import (
    IdleTimeoutError,
    IdleTimeoutStream,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.core.protocols.sdk import SDKClientFactoryProtocol, SDKClientProtocol
    from src.domain.lifecycle import LifecycleContext
    from src.infra.telemetry import TelemetrySpan
    from src.pipeline.message_stream_processor import (
        LintCacheProtocol,
        MessageIterationResult,
        MessageIterationState,
        MessageStreamProcessor,
    )


logger = logging.getLogger(__name__)

# Timeout for disconnect() call
DISCONNECT_TIMEOUT = 10.0


@dataclass
class RetryConfig:
    """Configuration for idle timeout retry behavior.

    The backoff sequence should have entries for each retry attempt.
    If fewer entries are provided than max_idle_retries, the last
    entry will be reused for subsequent retries.

    Attributes:
        max_idle_retries: Maximum number of idle timeout retries.
        idle_retry_backoff: Tuple of backoff delays in seconds for each retry.
        idle_resume_prompt: Template for idle resume prompts.
    """

    max_idle_retries: int = 0
    idle_retry_backoff: tuple[float, ...] = (0.0, 5.0, 15.0)
    idle_resume_prompt: str = ""

    def __post_init__(self) -> None:
        """Validate configuration invariants."""
        if self.max_idle_retries < 0:
            raise ValueError("max_idle_retries must be non-negative")
        if self.max_idle_retries > 0 and not self.idle_resume_prompt:
            raise ValueError(
                "idle_resume_prompt must be non-empty when max_idle_retries > 0"
            )


@dataclass
class IterationResult:
    """Result from a single session iteration."""

    success: bool
    session_id: str | None = None
    should_continue: bool = True
    error_message: str | None = None


class IdleTimeoutRetryPolicy:
    """Encapsulates idle timeout handling with backoff semantics.

    This policy manages the retry logic for SDK message iterations when
    idle timeouts occur. It handles:
    - Backoff delays between retry attempts
    - Preparing state for retry (session ID, tool IDs)
    - Processing message streams with timeout wrappers
    - Graceful client disconnection on timeout

    Usage:
        policy = IdleTimeoutRetryPolicy(
            sdk_client_factory=factory,
            stream_processor_factory=lambda: create_processor(),
            config=RetryConfig(max_idle_retries=2, idle_resume_prompt="Continue {issue_id}"),
        )
        result = await policy.execute_iteration(
            issue_id="ISSUE-123",
            query="...",
            options=sdk_options,
            state=msg_state,
            lifecycle_ctx=lifecycle_ctx,
            lint_cache=lint_cache,
            idle_timeout_seconds=300.0,
        )
    """

    def __init__(
        self,
        sdk_client_factory: SDKClientFactoryProtocol,
        stream_processor_factory: Callable[[], MessageStreamProcessor],
        config: RetryConfig,
    ) -> None:
        """Initialize the retry policy.

        Args:
            sdk_client_factory: Factory for creating SDK clients.
            stream_processor_factory: Factory that creates fresh MessageStreamProcessor
                instances. Called at the start of each execute_iteration to ensure
                current config/callbacks are used.
            config: Retry configuration (max retries, backoff).
        """
        self._sdk_client_factory = sdk_client_factory
        self._stream_processor_factory = stream_processor_factory
        self._config = config

    async def execute_iteration(
        self,
        query: str,
        issue_id: str,
        options: object,
        state: MessageIterationState,
        lifecycle_ctx: LifecycleContext,
        lint_cache: LintCacheProtocol,
        idle_timeout_seconds: float | None,
        tracer: TelemetrySpan | None = None,
    ) -> IterationResult:
        """Run a single message iteration with idle retry handling.

        Sends a query to the SDK and processes the response stream.
        Handles idle timeouts with automatic retry logic.

        Args:
            query: The query to send to the agent.
            issue_id: Issue ID for logging.
            options: SDK client options.
            state: Mutable state for the iteration.
            lifecycle_ctx: Lifecycle context for session state.
            lint_cache: Cache for lint command results.
            idle_timeout_seconds: Idle timeout (None to disable).
            tracer: Optional telemetry span context.

        Returns:
            IterationResult with success status and session ID.

        Raises:
            IdleTimeoutError: If max idle retries exceeded.
        """
        # Create fresh stream processor to capture current config/callbacks
        stream_processor = self._stream_processor_factory()

        pending_query: str | None = query
        state.tool_calls_this_turn = 0
        state.first_message_received = False
        state.pending_tool_ids.clear()
        state.pending_lint_commands.clear()

        while pending_query is not None:
            # Backoff before retry (not on first attempt)
            if state.idle_retry_count > 0:
                await self._apply_retry_backoff(state.idle_retry_count)

            # Create options with resume if we have a session to resume
            effective_options = options
            if state.pending_session_id is not None:
                effective_options = self._sdk_client_factory.with_resume(
                    options, state.pending_session_id
                )

            # Create client for this attempt
            client = self._sdk_client_factory.create(effective_options)

            try:
                async with client:
                    # Send query
                    query_start = time.time()
                    if state.pending_session_id is not None:
                        logger.debug(
                            "Session %s: sending query with resume=%s...",
                            issue_id,
                            state.pending_session_id[:8],
                        )
                    else:
                        logger.debug(
                            "Session %s: sending query (new session)",
                            issue_id,
                        )
                    await client.query(pending_query)

                    # Wrap stream with idle timeout handling
                    stream = IdleTimeoutStream(
                        client.receive_response(),
                        idle_timeout_seconds,
                        state.pending_tool_ids,
                    )

                    try:
                        result = await self._process_message_stream(
                            stream_processor,
                            stream,
                            issue_id,
                            state,
                            lifecycle_ctx,
                            lint_cache,
                            query_start,
                            tracer,
                        )
                        return IterationResult(
                            success=result.success,
                            session_id=result.session_id,
                        )

                    except IdleTimeoutError:
                        # Disconnect on idle timeout
                        idle_duration = time.time() - query_start
                        logger.warning(
                            f"Session {issue_id}: idle timeout after "
                            f"{idle_duration:.1f}s, first_msg={state.first_message_received}, "
                            f"{state.tool_calls_this_turn} tool calls, disconnecting subprocess"
                        )
                        await self._disconnect_client_safely(client, issue_id)

                        # Prepare state for retry (may raise IdleTimeoutError)
                        retry_query = self._prepare_idle_retry(
                            state, lifecycle_ctx, issue_id
                        )
                        # Empty string means keep original query
                        if retry_query:
                            pending_query = retry_query

            except IdleTimeoutError:
                raise

        # Should not reach here
        return IterationResult(success=False)

    async def _apply_retry_backoff(self, retry_count: int) -> None:
        """Apply backoff delay before an idle retry attempt.

        Args:
            retry_count: Current retry count (1-based).
        """
        if self._config.idle_retry_backoff:
            backoff_idx = min(
                retry_count - 1,
                len(self._config.idle_retry_backoff) - 1,
            )
            backoff = self._config.idle_retry_backoff[backoff_idx]
        else:
            backoff = 0.0
        if backoff > 0:
            logger.info(f"Idle retry {retry_count}: waiting {backoff}s")
            await asyncio.sleep(backoff)

    async def _disconnect_client_safely(
        self, client: SDKClientProtocol, issue_id: str
    ) -> None:
        """Disconnect SDK client with timeout, logging any failures."""
        try:
            await asyncio.wait_for(
                client.disconnect(),
                timeout=DISCONNECT_TIMEOUT,
            )
        except TimeoutError:
            logger.warning("disconnect() timed out, subprocess abandoned")
        except Exception as e:
            logger.debug(f"Error during disconnect: {e}")

    def _prepare_idle_retry(
        self,
        state: MessageIterationState,
        lifecycle_ctx: LifecycleContext,
        issue_id: str,
    ) -> str:
        """Prepare state for idle retry and return the next query.

        Updates state.idle_retry_count, state.pending_session_id, and clears
        state.pending_tool_ids and state.pending_lint_commands.

        Raises:
            IdleTimeoutError: If retry is not possible (max retries exceeded,
                or tool calls occurred without session context).

        Returns:
            The query to use for the retry attempt.
        """
        # Check if we can retry
        if state.idle_retry_count >= self._config.max_idle_retries:
            logger.error(
                f"Session {issue_id}: max idle retries "
                f"({self._config.max_idle_retries}) exceeded"
            )
            raise IdleTimeoutError(
                f"Max idle retries ({self._config.max_idle_retries}) exceeded"
            )

        # Prepare for retry
        state.idle_retry_count += 1
        # Clear pending state from previous attempt to avoid
        # hanging on stale tool IDs (they won't resolve on new stream)
        state.pending_tool_ids.clear()
        state.pending_lint_commands.clear()
        state.first_message_received = False
        resume_id = state.session_id or lifecycle_ctx.session_id

        if resume_id is not None:
            state.pending_session_id = resume_id
            pending_query = self._config.idle_resume_prompt.format(issue_id=issue_id)
            logger.info(
                f"Session {issue_id}: retrying with resume "
                f"(session_id={resume_id[:8]}..., "
                f"attempt {state.idle_retry_count})"
            )
            # Reset tool calls after decision to preserve safety check
            state.tool_calls_this_turn = 0
            return pending_query
        elif state.tool_calls_this_turn == 0:
            state.pending_session_id = None
            # Keep original query - caller must provide it
            logger.info(
                f"Session {issue_id}: retrying with fresh session "
                f"(no session_id, no side effects, "
                f"attempt {state.idle_retry_count})"
            )
            # Return empty string to signal caller to keep original query
            return ""
        else:
            logger.error(
                f"Session {issue_id}: cannot retry - "
                f"{state.tool_calls_this_turn} tool calls "
                "occurred without session_id"
            )
            raise IdleTimeoutError(
                f"Cannot retry: {state.tool_calls_this_turn} tool calls "
                "occurred without session context"
            )

    async def _process_message_stream(
        self,
        stream_processor: MessageStreamProcessor,
        stream: IdleTimeoutStream,
        issue_id: str,
        state: MessageIterationState,
        lifecycle_ctx: LifecycleContext,
        lint_cache: LintCacheProtocol,
        query_start: float,
        tracer: TelemetrySpan | None,
    ) -> MessageIterationResult:
        """Process SDK message stream and update state.

        Delegates to MessageStreamProcessor for stream iteration logic.

        Args:
            stream_processor: Processor for the message stream.
            stream: The message stream to process.
            issue_id: Issue ID for logging.
            state: Mutable state for the iteration.
            lifecycle_ctx: Lifecycle context for session state.
            lint_cache: Cache for lint command results.
            query_start: Timestamp when query was sent.
            tracer: Optional telemetry span context.

        Returns:
            MessageIterationResult with success status.
        """
        return await stream_processor.process_stream(
            stream, issue_id, state, lifecycle_ctx, lint_cache, query_start, tracer
        )

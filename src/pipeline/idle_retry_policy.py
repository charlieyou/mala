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
from typing import TYPE_CHECKING, Literal

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

# Timeout for disconnect() call. Keep this above CodexClient's bounded teardown
# budget (interrupt + close) so the shared retry path does not cancel Codex
# cleanup before close gets its own chance to run.
DISCONNECT_TIMEOUT = 30.0

_AGENT_SUBPROCESS_EXIT_MARKER = "agent subprocess exited with code 1"
_RETRYABLE_SUBPROCESS_EXIT_MARKERS = (
    _AGENT_SUBPROCESS_EXIT_MARKER,
    "amp subprocess exited with code 1",
)


def _is_retryable_subprocess_exit(exc: Exception) -> bool:
    """Return True for transient agent subprocess exits seen in the stream."""
    error_text = str(exc).lower()
    return any(marker in error_text for marker in _RETRYABLE_SUBPROCESS_EXIT_MARKERS)


def _client_session_id(client: object) -> str | None:
    """Best-effort session/thread id exposed by clients such as AmpClient."""
    session_id = getattr(client, "session_id", None)
    if session_id is None:
        return None
    session_text = str(session_id)
    return session_text or None


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
    max_subprocess_exit_attempts: int = 5
    subprocess_exit_retry_backoff: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)

    def __post_init__(self) -> None:
        """Validate configuration invariants."""
        if self.max_idle_retries < 0:
            raise ValueError("max_idle_retries must be non-negative")
        if self.max_subprocess_exit_attempts < 1:
            raise ValueError("max_subprocess_exit_attempts must be at least 1")
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
    """Encapsulates agent iteration retry handling with backoff semantics.

    This policy manages the retry logic for SDK message iterations when
    idle timeouts or known transient subprocess exits occur. It handles:
    - Backoff delays between retry attempts
    - Preparing state for retry (session ID, tool IDs)
    - Processing message streams with timeout wrappers
    - Graceful client disconnection on retryable failures

    Usage:
        policy = IdleTimeoutRetryPolicy(
            sdk_client_factory=factory,
            stream_processor_factory=lambda: create_processor(),
            config=RetryConfig(max_idle_retries=2, idle_resume_prompt="Continue {issue_id}"),
        )
        result = await policy.execute_iteration(
            issue_id="ISSUE-123",
            query="...",
            runtime=coder_runtime,
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
        runtime: object,
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
            runtime: Opaque coder-shaped runtime forwarded verbatim to
                ``client_factory.create(runtime)`` /
                ``client_factory.with_resume(runtime, ...)``. The pipeline
                never inspects its shape.
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
        pending_retry_kind: Literal["idle", "subprocess"] | None = None
        subprocess_exit_retries = 0

        while pending_query is not None:
            # Backoff before retry (not on first attempt)
            if pending_retry_kind == "idle":
                await self._apply_retry_backoff(state.idle_retry_count)
            elif pending_retry_kind == "subprocess":
                await self._apply_subprocess_exit_backoff(subprocess_exit_retries)

            # Create runtime with resume if we have a session to resume
            effective_runtime = runtime
            if state.pending_session_id is not None:
                effective_runtime = self._sdk_client_factory.with_resume(
                    runtime, state.pending_session_id
                )

            # Create client for this attempt
            client = self._sdk_client_factory.create(effective_runtime)

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

                    # Wrap the provider's AgentEvent stream with idle timeout
                    # handling. Both Claude (via the wrapped SDK client in
                    # ``src.infra.sdk_adapter``) and Amp emit ``AgentEvent``s
                    # directly, so no translation is needed at the pipeline
                    # layer.
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
                        if not result.success:
                            error_text = result.error or "SDK result reported an error"
                            exc = RuntimeError(
                                f"{_AGENT_SUBPROCESS_EXIT_MARKER}: {error_text}"
                            )
                            retry_query = self._prepare_subprocess_retry(
                                state,
                                lifecycle_ctx,
                                issue_id,
                                exc,
                                subprocess_exit_retries=subprocess_exit_retries,
                                resume_id=_client_session_id(client)
                                or result.session_id
                                or state.pending_session_id
                                or state.session_id
                                or lifecycle_ctx.session_id,
                            )
                            if retry_query is None:
                                raise exc
                            subprocess_exit_retries += 1
                            pending_retry_kind = "subprocess"
                            if retry_query:
                                pending_query = retry_query
                            continue
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
                        # Capture client-known session id BEFORE disconnect: Amp's
                        # thread id arrives on the system init event and lives on
                        # the client until ResultMessage. If the stream goes idle
                        # mid-turn (no ResultMessage yet), state.session_id is
                        # still None and only the client knows the resume id.
                        idle_resume_id = (
                            _client_session_id(client)
                            or state.pending_session_id
                            or state.session_id
                            or lifecycle_ctx.session_id
                        )
                        await self._disconnect_client_safely(client, issue_id)

                        # Prepare state for retry (may raise IdleTimeoutError)
                        retry_query = self._prepare_idle_retry(
                            state, lifecycle_ctx, issue_id, resume_id=idle_resume_id
                        )
                        pending_retry_kind = "idle"
                        # Empty string means keep original query
                        if retry_query:
                            pending_query = retry_query

                    except Exception as exc:
                        if not _is_retryable_subprocess_exit(exc):
                            raise

                        elapsed = time.time() - query_start
                        logger.warning(
                            "Session %s: agent subprocess exited after %.1fs, "
                            "first_msg=%s, %d tool calls, disconnecting subprocess",
                            issue_id,
                            elapsed,
                            state.first_message_received,
                            state.tool_calls_this_turn,
                        )
                        await self._disconnect_client_safely(client, issue_id)

                        retry_query = self._prepare_subprocess_retry(
                            state,
                            lifecycle_ctx,
                            issue_id,
                            exc,
                            subprocess_exit_retries=subprocess_exit_retries,
                            resume_id=_client_session_id(client)
                            or state.pending_session_id
                            or state.session_id
                            or lifecycle_ctx.session_id,
                        )
                        if retry_query is None:
                            raise
                        subprocess_exit_retries += 1
                        pending_retry_kind = "subprocess"
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
            logger.info(f"Agent retry {retry_count}: waiting {backoff}s")
            await asyncio.sleep(backoff)

    async def _apply_subprocess_exit_backoff(self, retry_count: int) -> None:
        """Apply exponential backoff before retrying subprocess exit failures."""
        if self._config.subprocess_exit_retry_backoff:
            backoff_idx = min(
                retry_count - 1,
                len(self._config.subprocess_exit_retry_backoff) - 1,
            )
            backoff = self._config.subprocess_exit_retry_backoff[backoff_idx]
        else:
            backoff = 0.0
        if backoff > 0:
            logger.info(
                "Agent subprocess retry %d/%d: waiting %.1fs",
                retry_count + 1,
                self._config.max_subprocess_exit_attempts,
                backoff,
            )
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
        resume_id: str | None = None,
    ) -> str:
        """Prepare state for idle retry and return the next query.

        Updates state.idle_retry_count, state.pending_session_id, and clears
        state.pending_tool_ids and state.pending_lint_commands.

        Args:
            resume_id: Caller-provided session id to resume from. Pass the
                client-known id (e.g. Amp thread id) so mid-turn idle timeouts
                can resume even when no ResultMessage populated state.session_id.

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

        retry_query = self._prepare_retry_state(
            state,
            lifecycle_ctx,
            issue_id,
            retry_reason="idle timeout",
            resume_id=resume_id,
        )
        if retry_query is None:
            logger.error(
                f"Session {issue_id}: cannot retry - "
                f"{state.tool_calls_this_turn} tool calls "
                "occurred without session_id"
            )
            raise IdleTimeoutError(
                f"Cannot retry: {state.tool_calls_this_turn} tool calls "
                "occurred without session context"
            )
        return retry_query

    def _prepare_subprocess_retry(
        self,
        state: MessageIterationState,
        lifecycle_ctx: LifecycleContext,
        issue_id: str,
        exc: Exception,
        subprocess_exit_retries: int,
        resume_id: str | None,
    ) -> str | None:
        """Prepare a safe retry after a transient subprocess exit.

        Returns None when retrying would exceed the retry budget or risk
        duplicating tool-side effects; the caller should then re-raise ``exc``.
        """
        max_retries = self._config.max_subprocess_exit_attempts - 1
        if subprocess_exit_retries >= max_retries:
            logger.error(
                "Session %s: max subprocess attempts (%d) exceeded after: %s",
                issue_id,
                self._config.max_subprocess_exit_attempts,
                exc,
            )
            return None

        retry_query = self._prepare_retry_state(
            state,
            lifecycle_ctx,
            issue_id,
            retry_reason="agent subprocess exit",
            resume_id=resume_id,
            increment_idle_retry_count=False,
            attempt_number=subprocess_exit_retries + 1,
        )
        if retry_query is None:
            logger.error(
                "Session %s: cannot retry subprocess exit after %d tool calls "
                "without session_id: %s",
                issue_id,
                state.tool_calls_this_turn,
                exc,
            )
        return retry_query

    def _prepare_retry_state(
        self,
        state: MessageIterationState,
        lifecycle_ctx: LifecycleContext,
        issue_id: str,
        retry_reason: str,
        resume_id: str | None = None,
        *,
        increment_idle_retry_count: bool = True,
        attempt_number: int | None = None,
    ) -> str | None:
        """Update iteration state for a safe retry.

        Returns a replacement query, an empty string to keep the original query,
        or None when retrying would risk duplicating tool-side effects.
        """
        if resume_id is None:
            resume_id = (
                state.pending_session_id or state.session_id or lifecycle_ctx.session_id
            )
        if resume_id is None and state.tool_calls_this_turn != 0:
            return None

        if increment_idle_retry_count:
            state.idle_retry_count += 1
            logged_attempt = state.idle_retry_count
        elif attempt_number is not None:
            logged_attempt = attempt_number
        else:
            logged_attempt = 0
        # Clear pending state from previous attempt to avoid
        # hanging on stale tool IDs (they won't resolve on new stream)
        state.pending_tool_ids.clear()
        state.pending_lint_commands.clear()
        state.first_message_received = False

        if resume_id is not None:
            state.pending_session_id = resume_id
            pending_query = self._config.idle_resume_prompt.format(issue_id=issue_id)
            logger.info(
                "Session %s: retrying after %s with resume "
                "(session_id=%s..., attempt %d)",
                issue_id,
                retry_reason,
                resume_id[:8],
                logged_attempt,
            )
            # Reset tool calls after decision to preserve safety check
            state.tool_calls_this_turn = 0
            return pending_query
        state.pending_session_id = None
        state.tool_calls_this_turn = 0
        # Keep original query - caller must provide it
        logger.info(
            "Session %s: retrying after %s with fresh session "
            "(no session_id, no side effects, attempt %d)",
            issue_id,
            retry_reason,
            logged_attempt,
        )
        # Return empty string to signal caller to keep original query
        return ""

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

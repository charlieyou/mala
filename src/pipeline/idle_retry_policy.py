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
from typing import TYPE_CHECKING, Literal, cast

from src.pipeline.message_stream_processor import (
    IdleTimeoutError,
    IdleTimeoutStream,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.core.protocols.agent_event import AgentTaskCompletedEvent
    from src.core.protocols.events.sink import MalaEventSink
    from src.core.protocols.sdk import SDKClientFactoryProtocol, SDKClientProtocol
    from src.domain.lifecycle import LifecycleContext
    from src.domain.validation.config_types import LongRunningConfig
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

# Cadence for surfacing the otherwise-silent background wait to the event sink.
BACKGROUND_WAIT_PROGRESS_INTERVAL = 30.0

# Sentinel returned by the wrapped stream reader when the continuous message
# stream is exhausted (StopAsyncIteration cannot cross a Future boundary).
_STREAM_END = object()

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


@dataclass
class _BgWaitOutcome:
    """Outcome of waiting for a single backgrounded task's completion.

    Distinguishes the three terminal states so the wait/resume loop can act
    appropriately: a completion arrived, the wait was interrupted by a
    drain/abort signal, or the wait gave up (timeout / stream end).
    """

    completed: AgentTaskCompletedEvent | None = None
    task_id: str | None = None
    interrupted: bool = False


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
        event_sink: MalaEventSink | None = None,
    ) -> None:
        """Initialize the retry policy.

        Args:
            sdk_client_factory: Factory for creating SDK clients.
            stream_processor_factory: Factory that creates fresh MessageStreamProcessor
                instances. Called at the start of each execute_iteration to ensure
                current config/callbacks are used.
            config: Retry configuration (max retries, backoff).
            event_sink: Optional event sink used to surface the between-turn
                background wait as periodic progress (so the run does not look
                hung). None disables progress emission.
        """
        self._sdk_client_factory = sdk_client_factory
        self._stream_processor_factory = stream_processor_factory
        self._config = config
        self._event_sink = event_sink

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
        *,
        long_running: LongRunningConfig | None = None,
        await_resume_template: str = "",
        hard_timeout_seconds: float | None = None,
        drain_event: asyncio.Event | None = None,
        interrupt_event: asyncio.Event | None = None,
    ) -> IterationResult:
        """Run a single message iteration with idle retry handling.

        Sends a query to the SDK and processes the response stream.
        Handles idle timeouts with automatic retry logic.

        When the agent backgrounds work via ``Bash(run_in_background=true)`` and
        yields, the turn ends with a still-pending background launch. If the
        client supports background tasks (Claude only) and ``long_running`` is
        enabled, this method keeps the SDK client connected, waits for the
        task's completion notification on the continuous stream, resumes the
        agent on the same client to finalize, and only then returns — so the
        gate runs once at the true end.

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
            long_running: Background wait/resume configuration. When None or
                disabled, the wait/resume loop is skipped (immediate return).
            await_resume_template: Template rendered into the resume prompt
                after a backgrounded task finishes (see
                ``format_await_resume_prompt``).
            hard_timeout_seconds: Per-turn hard timeout that bounds the agent's
                active compute. Each message-stream read is wrapped in
                ``asyncio.timeout(hard_timeout_seconds)`` so the (possibly
                multi-hour) background wait is not killed by the per-task
                timeout. None disables the wrapper.
            drain_event: Set on the first Ctrl-C (drain). When set during the
                between-turn background wait, the wait stops promptly (best-effort
                ``stop_task``) and the iteration proceeds rather than blocking.
            interrupt_event: Set on abort (second Ctrl-C). Treated like
                ``drain_event`` for breaking the background wait.

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
                    # Bound query startup (and any SDK connect/resume work it
                    # triggers) by the hard per-task timeout, restoring the
                    # coverage the previous outer timeout gave. The between-turn
                    # background wait in ``_wait_and_resume_background`` is
                    # intentionally left unbounded.
                    if hard_timeout_seconds is not None:
                        async with asyncio.timeout(hard_timeout_seconds):
                            await client.query(pending_query)
                    else:
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
                        result = await self._run_message_stream_turn(
                            stream_processor,
                            stream,
                            issue_id,
                            state,
                            lifecycle_ctx,
                            lint_cache,
                            query_start,
                            tracer,
                            hard_timeout_seconds,
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
                        # Successful turn. If the agent backgrounded work and
                        # yielded, keep this client connected, wait for the
                        # task's completion notification, and resume the agent
                        # on the same client to finalize before returning.
                        return await self._wait_and_resume_background(
                            client,
                            result,
                            stream_processor,
                            issue_id,
                            state,
                            lifecycle_ctx,
                            lint_cache,
                            tracer,
                            idle_timeout_seconds=idle_timeout_seconds,
                            long_running=long_running,
                            await_resume_template=await_resume_template,
                            hard_timeout_seconds=hard_timeout_seconds,
                            drain_event=drain_event,
                            interrupt_event=interrupt_event,
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

    async def _run_message_stream_turn(
        self,
        stream_processor: MessageStreamProcessor,
        stream: IdleTimeoutStream,
        issue_id: str,
        state: MessageIterationState,
        lifecycle_ctx: LifecycleContext,
        lint_cache: LintCacheProtocol,
        query_start: float,
        tracer: TelemetrySpan | None,
        hard_timeout_seconds: float | None,
    ) -> MessageIterationResult:
        """Process one turn's message stream, bounding active compute.

        The hard per-task timeout wraps each per-turn stream read here (rather
        than the whole ``execute_iteration``) so the background wait between
        turns is not killed by the per-task timeout. ``IdleTimeoutStream``
        still applies its own idle timeout within the turn.
        """
        if hard_timeout_seconds is None:
            return await self._process_message_stream(
                stream_processor,
                stream,
                issue_id,
                state,
                lifecycle_ctx,
                lint_cache,
                query_start,
                tracer,
            )
        async with asyncio.timeout(hard_timeout_seconds):
            return await self._process_message_stream(
                stream_processor,
                stream,
                issue_id,
                state,
                lifecycle_ctx,
                lint_cache,
                query_start,
                tracer,
            )

    async def _wait_and_resume_background(
        self,
        client: SDKClientProtocol,
        result: MessageIterationResult,
        stream_processor: MessageStreamProcessor,
        issue_id: str,
        state: MessageIterationState,
        lifecycle_ctx: LifecycleContext,
        lint_cache: LintCacheProtocol,
        tracer: TelemetrySpan | None,
        *,
        idle_timeout_seconds: float | None,
        long_running: LongRunningConfig | None,
        await_resume_template: str,
        hard_timeout_seconds: float | None,
        drain_event: asyncio.Event | None = None,
        interrupt_event: asyncio.Event | None = None,
    ) -> IterationResult:
        """Keep the client connected to finish backgrounded work, then return.

        Loops while the just-finished turn left a still-pending background
        launch (one the agent did *not* retrieve inline), the client supports
        background tasks (Claude only), and the long-running feature is enabled.
        Each cycle waits for the task's completion notification on
        ``client.receive_messages()``, resumes the agent on the *same* client
        with the await-resume prompt, and processes the resulting turn.

        The wait is interruptible (``drain_event``/``interrupt_event``) and
        emits periodic progress so it is never a silent hang. When the wait is
        interrupted, times out, or the resume-cycle budget is exhausted, the
        iteration **proceeds to the gate as success** rather than failing — the
        agent's launch turn succeeded, and the gate re-runs validation and is
        the real arbiter of whether the backgrounded work produced what the
        issue needs. Only a genuine *resume-turn* error fails the iteration.
        """
        from src.domain.prompts import format_await_resume_prompt

        if (
            long_running is None
            or not long_running.enabled
            or not client.supports_background_tasks()
        ):
            return IterationResult(
                success=result.success,
                session_id=result.session_id,
            )

        # Completions for *other* pending launches can arrive on the shared
        # continuous stream while we wait for this one. Buffer them by
        # tool_use_id so a later cycle finds them instead of waiting until
        # timeout. ``started_task_ids`` is seeded from task_started events seen
        # during the launch turn (recorded on shared state) so stop_task works
        # on timeout even if no completion ever arrives on the stream.
        completion_buffer: dict[str, AgentTaskCompletedEvent] = {}
        started_task_ids: dict[str, str] = dict(state.background_task_ids)
        cycles = 0
        while (
            result.success
            and result.pending_background_tool_ids
            and cycles < long_running.max_resume_cycles
        ):
            tool_use_id = next(iter(result.pending_background_tool_ids))
            buffered = completion_buffer.pop(tool_use_id, None)
            if buffered is not None:
                outcome = _BgWaitOutcome(
                    completed=buffered,
                    task_id=buffered.task_id or started_task_ids.get(tool_use_id),
                )
            else:
                outcome = await self._wait_for_background_completion(
                    client,
                    tool_use_id,
                    long_running.max_wait_seconds,
                    issue_id,
                    completion_buffer=completion_buffer,
                    started_task_ids=started_task_ids,
                    drain_event=drain_event,
                    interrupt_event=interrupt_event,
                )

            if outcome.interrupted:
                # Drain/abort during the wait: stop blocking and proceed. The SDK
                # client is torn down when this iteration returns (killing the
                # task's process group), so we do not stop_task here — keeping the
                # first Ctrl-C responsive matters more than a tidy stop.
                logger.info(
                    "Session %s: background wait interrupted; proceeding to gate",
                    issue_id,
                )
                self._abandon_background_state(state)
                return IterationResult(
                    success=result.success,
                    session_id=result.session_id,
                )

            if outcome.completed is None:
                # ``state.background_task_ids`` is live shared state updated by
                # the stream processor, so a task_started consumed by a resume
                # turn's ``receive_response`` is found here even though the
                # pre-loop ``started_task_ids`` snapshot predates it.
                task_id = (
                    outcome.task_id
                    or started_task_ids.get(tool_use_id)
                    or state.background_task_ids.get(tool_use_id)
                )
                logger.warning(
                    "Session %s: background task (tool_use_id=%s) did not "
                    "complete within %.0fs; stopping it and proceeding to gate",
                    issue_id,
                    tool_use_id,
                    long_running.max_wait_seconds,
                )
                if task_id:
                    await self._safe_stop_task(client, task_id, issue_id)
                self._abandon_background_state(state)
                return IterationResult(
                    success=result.success,
                    session_id=result.session_id,
                )

            completed = outcome.completed
            # The completion arrived on the continuous stream (not the turn's
            # ``receive_response``), so the stream processor never cleared this
            # launch from the shared iteration state. Discard it here so the
            # resume turn's result reports only launches started *during* that
            # turn — otherwise this finished launch would be re-reported and the
            # loop would spin until ``max_resume_cycles``.
            state.pending_background_tool_ids.discard(tool_use_id)

            resume_prompt = format_await_resume_prompt(
                await_resume_template,
                issue_id=issue_id,
                output_file=completed.output_file,
                status=completed.status,
                summary=completed.summary,
            )
            logger.info(
                "Session %s: background task completed (status=%s); resuming "
                "agent to finalize (cycle %d/%d)",
                issue_id,
                completed.status,
                cycles + 1,
                long_running.max_resume_cycles,
            )
            # Reset per-turn state before the resume turn, mirroring the
            # initial-turn setup in execute_iteration. The launch turn can end
            # on a background Bash tool_use with no matching tool_result, leaving
            # that id in pending_tool_ids — which would suppress the resume
            # turn's idle timeout until the hard timeout. Start the resume turn
            # clean.
            state.tool_calls_this_turn = 0
            state.first_message_received = False
            state.pending_tool_ids.clear()
            state.pending_lint_commands.clear()

            # Bound the resume query startup by the hard per-task timeout too
            # (the resume turn is active agent compute, not the background wait).
            if hard_timeout_seconds is not None:
                async with asyncio.timeout(hard_timeout_seconds):
                    await client.query(resume_prompt)
            else:
                await client.query(resume_prompt)
            resume_start = time.time()
            stream = IdleTimeoutStream(
                client.receive_response(),
                idle_timeout_seconds,
                state.pending_tool_ids,
            )
            result = await self._run_message_stream_turn(
                stream_processor,
                stream,
                issue_id,
                state,
                lifecycle_ctx,
                lint_cache,
                resume_start,
                tracer,
                hard_timeout_seconds,
            )
            cycles += 1

        # Resume-cycle budget exhausted while a launch is still pending: the
        # agent kept backgrounding work without finishing. Best-effort stop the
        # runaway tasks, then proceed to the gate as success and let it arbitrate
        # rather than failing an otherwise-successful turn.
        if result.success and result.pending_background_tool_ids:
            for pending_id in result.pending_background_tool_ids:
                stop_id = started_task_ids.get(
                    pending_id
                ) or state.background_task_ids.get(pending_id)
                if stop_id:
                    await self._safe_stop_task(client, stop_id, issue_id)
            logger.warning(
                "Session %s: background work still pending after %d resume "
                "cycles; proceeding to gate",
                issue_id,
                long_running.max_resume_cycles,
            )
            self._abandon_background_state(state)
            return IterationResult(
                success=result.success,
                session_id=result.session_id,
            )

        return IterationResult(
            success=result.success,
            session_id=result.session_id,
            error_message=None if result.success else result.error,
        )

    @staticmethod
    def _abandon_background_state(state: MessageIterationState) -> None:
        """Drop all pending background tracking before returning success.

        The interrupt, wait-timeout, and resume-exhaustion paths give up on a
        still-pending launch and proceed to the gate. ``MessageIterationState``
        is shared across the gate-retry ``execute_iteration`` calls in
        ``_run_lifecycle_loop`` (and is not reset at the top of
        ``execute_iteration``), so a launch left in ``pending_background_tool_ids``
        would be re-reported by the next turn and force another wait for a
        completion that can never arrive on that turn's fresh client. Clearing
        the launch ids and their metadata here prevents that stale carry-over.
        """
        state.pending_background_tool_ids.clear()
        state.background_task_ids.clear()
        state.pending_bg_retrievals.clear()
        state.background_launch_commands.clear()

    async def _safe_stop_task(
        self, client: SDKClientProtocol, task_id: str, issue_id: str
    ) -> None:
        """Best-effort ``stop_task`` that never raises into the wait loop."""
        try:
            await client.stop_task(task_id)
        except Exception as exc:  # best-effort cleanup
            logger.debug("Session %s: stop_task(%s) failed: %s", issue_id, task_id, exc)

    async def _wait_for_background_completion(
        self,
        client: SDKClientProtocol,
        tool_use_id: str,
        max_wait_seconds: float,
        issue_id: str,
        *,
        completion_buffer: dict[str, AgentTaskCompletedEvent] | None = None,
        started_task_ids: dict[str, str] | None = None,
        drain_event: asyncio.Event | None = None,
        interrupt_event: asyncio.Event | None = None,
    ) -> _BgWaitOutcome:
        """Wait for the completion notification of a backgrounded launch.

        Reads the client's continuous (turn-spanning) message stream until an
        ``AgentTaskCompletedEvent`` for ``tool_use_id`` arrives, bounded by
        ``max_wait_seconds``. The read is raced against ``drain_event`` /
        ``interrupt_event`` so the first Ctrl-C ends the wait promptly, and
        periodic progress is emitted on the event sink so the wait is never a
        silent hang.

        Returns a :class:`_BgWaitOutcome`: ``completed`` set on success,
        ``interrupted`` set when a drain/abort signal fired, or both unset on
        timeout / stream end. ``task_id`` is captured from the matching
        completion event, falling back to a ``task_started`` event seen for the
        same launch (so the caller can stop the task even without a completion).
        """
        task_id: str | None = None
        loop = asyncio.get_event_loop()
        deadline = loop.time() + max_wait_seconds
        next_progress = loop.time() + BACKGROUND_WAIT_PROGRESS_INTERVAL

        events = [ev for ev in (drain_event, interrupt_event) if ev is not None]
        signal_tasks = [asyncio.ensure_future(ev.wait()) for ev in events]
        next_task: asyncio.Future[object] | None = None
        aiter = client.receive_messages().__aiter__()

        def _interrupted() -> bool:
            return any(ev.is_set() for ev in events)

        try:
            if _interrupted():
                return _BgWaitOutcome(task_id=task_id, interrupted=True)
            while True:
                now = loop.time()
                if now >= deadline:
                    return _BgWaitOutcome(task_id=task_id)
                if next_task is None:
                    next_task = asyncio.ensure_future(self._next_stream_event(aiter))
                slice_timeout = min(deadline - now, next_progress - now)
                slice_timeout = max(slice_timeout, 0.0)
                done, _pending = await asyncio.wait(
                    {next_task, *signal_tasks},
                    timeout=slice_timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if _interrupted():
                    next_task.cancel()
                    return _BgWaitOutcome(task_id=task_id, interrupted=True)
                if next_task not in done:
                    # Progress tick: the in-flight read is intentionally left
                    # running (cancelling it would corrupt the async generator).
                    self._emit_background_wait_progress(
                        issue_id, tool_use_id, max_wait_seconds, deadline, loop
                    )
                    next_progress = loop.time() + BACKGROUND_WAIT_PROGRESS_INTERVAL
                    continue

                event = next_task.result()
                next_task = None
                if event is _STREAM_END:
                    logger.debug(
                        "Session %s: background stream ended without completion for %s",
                        issue_id,
                        tool_use_id,
                    )
                    return _BgWaitOutcome(task_id=task_id)

                kind = getattr(event, "kind", None)
                event_tool_use_id = str(getattr(event, "tool_use_id", "") or "")
                if kind == "task_started":
                    ev_task_id = str(getattr(event, "task_id", "") or "")
                    if started_task_ids is not None and event_tool_use_id:
                        started_task_ids[event_tool_use_id] = ev_task_id
                    if event_tool_use_id == tool_use_id:
                        task_id = ev_task_id or task_id
                    continue
                if kind == "task_completed":
                    completed = cast("AgentTaskCompletedEvent", event)
                    if event_tool_use_id == tool_use_id:
                        event_task_id = str(getattr(event, "task_id", "") or "")
                        return _BgWaitOutcome(
                            completed=completed, task_id=event_task_id or task_id
                        )
                    # Completion for a *different* pending launch — buffer it so a
                    # later cycle finds it instead of waiting until timeout (the
                    # SDK pushes all task events onto one stream).
                    if completion_buffer is not None and event_tool_use_id:
                        completion_buffer[event_tool_use_id] = completed
        finally:
            # Cancel any in-flight read and the signal waiters. The surrounding
            # ``async with client`` tears the stream down, so abandoning an
            # in-flight ``__anext__`` is safe.
            if next_task is not None and not next_task.done():
                next_task.cancel()
            for sig in signal_tasks:
                sig.cancel()

    @staticmethod
    async def _next_stream_event(aiter: object) -> object:
        """Pull the next event from a message stream, or ``_STREAM_END``.

        StopAsyncIteration cannot cross the ``asyncio.Future`` boundary used to
        race the read against signals, so it is converted to a sentinel.
        """
        try:
            return await aiter.__anext__()  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]
        except StopAsyncIteration:
            return _STREAM_END

    def _emit_background_wait_progress(
        self,
        issue_id: str,
        tool_use_id: str,
        max_wait_seconds: float,
        deadline: float,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Surface the otherwise-silent background wait to the event sink."""
        if self._event_sink is None:
            return
        elapsed = max_wait_seconds - max(deadline - loop.time(), 0.0)
        self._event_sink.on_background_wait(
            issue_id, tool_use_id, elapsed, max_wait_seconds
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

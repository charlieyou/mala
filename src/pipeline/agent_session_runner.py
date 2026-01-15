"""AgentSessionRunner: Agent session execution pipeline stage.

Extracted from MalaOrchestrator to separate SDK-specific session handling from
orchestration logic. This module handles:
- SDK streaming and message processing
- Session lifecycle transitions via ImplementerLifecycle
- Session metadata tracking (session ID, log paths)

The AgentSessionRunner receives explicit inputs and returns explicit outputs,
making it testable without SDK dependencies when using the SDKClientProtocol.

Design principles:
- Protocol-based SDK client for testability
- Explicit input/output types for clarity
- Lifecycle state machine drives policy decisions
- Protocol interfaces for external operations (gate checks, reviews, lifecycle)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

from src.infra.agent_runtime import AgentRuntimeBuilder
from src.infra.sigint_guard import FlowInterruptedError, InterruptGuard
from src.domain.lifecycle import (
    Effect,
    ImplementerLifecycle,
    LifecycleConfig,
    LifecycleContext,
    LifecycleState,
)
from src.pipeline.idle_retry_policy import (
    IdleTimeoutRetryPolicy,
    RetryConfig,
)
from src.pipeline.lifecycle_effect_handler import (
    LifecycleEffectHandler,
    _count_blocking_issues,
)
from src.core.session_end_result import SessionEndRetryState
from src.pipeline.message_stream_processor import (
    IdleTimeoutError,
    MessageIterationState,
    MessageStreamProcessor,
    StreamProcessorCallbacks,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from src.core.protocols.lifecycle import ISessionLifecycle
    from src.core.protocols.review import IReviewRunner
    from src.core.protocols.validation import IGateRunner
    from src.core.models import IssueResolution
    from src.core.protocols.events import MalaEventSink
    from src.core.protocols.review import ReviewIssueProtocol
    from src.core.protocols.sdk import McpServerFactory, SDKClientFactoryProtocol
    from src.domain.deadlock import DeadlockMonitor
    from src.domain.lifecycle import (
        GateOutcome,
        RetryState,
        ReviewOutcome,
        TransitionResult,
    )
    from src.domain.validation.config import PromptValidationCommands
    from src.infra.hooks import LintCache
    from src.infra.telemetry import TelemetrySpan
    from src.core.session_end_result import SessionEndResult


# Module-level logger for idle retry messages
logger = logging.getLogger(__name__)


def _is_stale_session_error(exc: Exception) -> bool:
    """Check if an exception indicates an unresumable/stale session.

    Catches SDK errors that indicate the session cannot be resumed:
    - SessionNotFoundError or InvalidSessionError (if SDK defines these)
    - HTTP 404/410 response errors wrapped in SDK exceptions
    - Any error with "session" + "not found"/"invalid"/"expired" in message

    Note: This heuristic may catch some auth-related errors if they mention
    "session expired". This is acceptable as the fallback behavior (retry
    without resume) is safe for both stale sessions and auth issues.
    """
    exc_name = type(exc).__name__
    if exc_name in ("SessionNotFoundError", "InvalidSessionError"):
        return True

    msg = str(exc).lower()
    if "session" in msg:
        stale_keywords = ("not found", "invalid", "expired", "404", "410")
        if any(kw in msg for kw in stale_keywords):
            return True

    return False


@dataclass
class SessionConfig:
    """Derived configuration for session execution.

    Computed from AgentSessionConfig during initialization.

    Attributes:
        agent_id: Unique agent ID for this session.
        options: SDK client options.
        lint_cache: Lint command result cache.
        log_file_wait_timeout: Timeout for log file availability.
        log_file_poll_interval: Poll interval for log file.
        idle_timeout_seconds: Idle timeout for SDK stream.
    """

    agent_id: str
    options: object
    lint_cache: LintCache
    log_file_wait_timeout: float
    log_file_poll_interval: float
    idle_timeout_seconds: float | None


@dataclass
class SessionExecutionState:
    """Mutable state for session execution.

    Bundles all session state that evolves during execution, including
    lifecycle context, session identifiers, and log paths.

    Attributes:
        lifecycle: Lifecycle state machine.
        lifecycle_ctx: Lifecycle context with retry state.
        session_id: SDK session ID (updated when ResultMessage received).
        log_path: Path to session log file.
        final_result: Final result text from session.
        cerberus_review_log_path: Path to Cerberus review log (if any).
        msg_state: Message iteration state.
    """

    lifecycle: ImplementerLifecycle
    lifecycle_ctx: LifecycleContext
    session_id: str | None = None
    log_path: Path | None = None
    final_result: str = ""
    cerberus_review_log_path: str | None = None
    msg_state: MessageIterationState = field(default_factory=MessageIterationState)


@dataclass
class SessionPrompts:
    """Prompt templates for agent session execution.

    Holds prompt templates loaded from files. This keeps file I/O at the
    orchestration boundary and allows tests to inject custom prompts.

    Attributes:
        gate_followup: Template for gate failure follow-up prompts.
        review_followup: Template for review issues follow-up prompts.
        idle_resume: Template for idle timeout resume prompts.
        checkpoint_request: Prompt to request checkpoint from agent.
        continuation: Template for continuation prompt with checkpoint.
    """

    gate_followup: str
    review_followup: str
    idle_resume: str
    checkpoint_request: str = ""
    continuation: str = ""


@dataclass
class AgentSessionConfig:
    """Configuration for agent session execution.

    Bundles all configuration needed to run an agent session.

    Attributes:
        repo_path: Path to the repository.
        timeout_seconds: Session timeout in seconds.
        prompts: Provider for session prompts (gate, review, idle resume).
        max_gate_retries: Maximum gate retry attempts.
        max_review_retries: Maximum review retry attempts.
        review_enabled: Whether Cerberus external review is enabled.
            When enabled, code changes are reviewed after the gate passes.
        log_file_wait_timeout: Seconds to wait for log file after session
            completes. Default 60s allows time for SDK to flush logs under load.
        idle_timeout_seconds: Seconds to wait for SDK message when not waiting
            for tool execution. If None, defaults to min(900, max(300, timeout_seconds * 0.2))
            which scales with session timeout (300-900s range). During tool execution
            (after ToolUseBlock, before result), timeout is disabled. Set to 0 to
            disable timeout entirely.
        lint_tools: Set of lint tool names for LintCache. If None, uses default
            lint tools. Populated from ValidationSpec commands.
        prompt_validation_commands: Validation commands for prompt templates.
            If None, uses default Python/uv commands.
        strict_resume: When True and session resumption fails (stale session),
            fail the session instead of retrying with a fresh session.
        setting_sources: Optional list of Claude settings sources to use.
            When None, uses SDK defaults (["local", "project"]).
    """

    repo_path: Path
    timeout_seconds: int
    prompts: SessionPrompts
    max_gate_retries: int = 3
    max_review_retries: int = 3
    review_enabled: bool = True
    log_file_wait_timeout: float = 60.0
    idle_timeout_seconds: float | None = None
    max_idle_retries: int = 2
    idle_retry_backoff: tuple[float, ...] = (0.0, 5.0, 15.0)
    lint_tools: frozenset[str] | None = None
    prompt_validation_commands: PromptValidationCommands | None = None
    deadlock_monitor: DeadlockMonitor | None = None
    mcp_server_factory: McpServerFactory | None = None
    strict_resume: bool = False
    setting_sources: list[str] | None = None


@dataclass
class AgentSessionInput:
    """Input for running an agent session.

    Bundles all data needed to start a session for an issue.

    Attributes:
        issue_id: The issue ID being worked on.
        prompt: The initial prompt to send to the agent.
        issue_description: Issue description for scope verification.
        agent_id: Optional pre-generated agent ID for lock management.
        resume_session_id: Optional session ID to resume from a prior run.
        flow: Flow identifier for structured logging (e.g., "implementer", "epic_remediation").
        baseline_timestamp: Optional baseline timestamp to persist across resumes.
    """

    issue_id: str
    prompt: str
    issue_description: str | None = None
    agent_id: str | None = None
    resume_session_id: str | None = None
    flow: str = "implementer"
    baseline_timestamp: int | None = None


@dataclass
class AgentSessionOutput:
    """Output from an agent session.

    Contains all results and metadata from a completed session.

    Attributes:
        success: Whether the session completed successfully.
        summary: Human-readable summary of the outcome.
        session_id: Claude SDK session ID (if available).
        log_path: Path to the session log file (if available).
        gate_attempts: Number of gate retry attempts.
        review_attempts: Number of review retry attempts.
        resolution: Issue resolution outcome (if any).
        duration_seconds: Total session duration.
        agent_id: The agent ID used for this session.
        review_log_path: Path to Cerberus review session log (if any).
        low_priority_review_issues: P2/P3 review issues to track as beads issues.
        interrupted: Whether the session was interrupted by SIGINT.
        baseline_timestamp: Baseline timestamp used for commit freshness checks.
        session_end_result: Session end validation result (if available).
    """

    success: bool
    summary: str
    session_id: str | None = None
    log_path: Path | None = None
    gate_attempts: int = 1
    review_attempts: int = 0
    resolution: IssueResolution | None = None
    duration_seconds: float = 0.0
    agent_id: str = ""
    review_log_path: str | None = None
    low_priority_review_issues: list[ReviewIssueProtocol] | None = None
    interrupted: bool = False
    baseline_timestamp: int | None = None
    last_review_issues: list[dict[str, Any]] | None = None
    session_end_result: SessionEndResult | None = None


@dataclass
class AgentSessionRunner:
    """Runs agent sessions with lifecycle management.

    This class encapsulates the SDK session execution logic that was previously
    inline in MalaOrchestrator.run_implementer. It manages:
    - SDK client creation and message streaming
    - Lifecycle state transitions
    - Hook setup (lock enforcement, lint cache, etc.)
    - Message logging and telemetry

    The runner uses protocol interfaces for external operations (gate checks,
    reviews, lifecycle operations) to decouple from orchestrator internals.

    Usage:
        runner = AgentSessionRunner(
            config=AgentSessionConfig(repo_path=repo_path, ...),
            sdk_client_factory=SDKClientFactory(),
            gate_runner=gate_runner_impl,
            review_runner=review_runner_impl,
            session_lifecycle=session_lifecycle_impl,
        )
        output = await runner.run_session(input)

    Attributes:
        config: Session configuration.
        sdk_client_factory: Factory for creating SDK clients (required).
        event_sink: Optional event sink for structured logging.
        gate_runner: Protocol for gate checking operations (required).
        review_runner: Protocol for review operations (required).
        session_lifecycle: Protocol for session lifecycle operations (required).
    """

    config: AgentSessionConfig
    sdk_client_factory: SDKClientFactoryProtocol
    gate_runner: IGateRunner
    review_runner: IReviewRunner
    session_lifecycle: ISessionLifecycle
    event_sink: MalaEventSink | None = None
    _retry_policy: IdleTimeoutRetryPolicy = field(init=False, repr=False)
    _effect_handler: LifecycleEffectHandler = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Initialize derived components."""
        # Initialize retry policy with stream processor factory
        retry_config = RetryConfig(
            max_idle_retries=self.config.max_idle_retries,
            idle_retry_backoff=self.config.idle_retry_backoff,
            idle_resume_prompt=self.config.prompts.idle_resume,
        )
        self._retry_policy = IdleTimeoutRetryPolicy(
            sdk_client_factory=self.sdk_client_factory,
            stream_processor_factory=self._get_stream_processor,
            config=retry_config,
        )
        # Initialize lifecycle effect handler
        self._effect_handler = LifecycleEffectHandler(
            config=self.config,
            event_sink=self.event_sink,
            gate_runner=self.gate_runner,
            review_runner=self.review_runner,
            session_lifecycle=self.session_lifecycle,
        )

    def _initialize_session(
        self,
        input: AgentSessionInput,
        agent_id: str | None = None,
    ) -> tuple[SessionConfig, SessionExecutionState]:
        """Initialize session components and state.

        Creates lifecycle, hooks, SDK options, and mutable state for session.

        Args:
            input: Session input with issue_id, prompt, etc.
            agent_id: Optional agent ID. If None, generates a new one.
                Pass an existing ID to preserve lock continuity across restarts.

        Returns:
            Tuple of (SessionConfig, SessionExecutionState).
        """
        # Generate agent ID if not provided
        if agent_id is None:
            agent_id = f"{input.issue_id}-{uuid.uuid4().hex[:8]}"

        # Initialize lifecycle
        lifecycle_config = LifecycleConfig(
            max_gate_retries=self.config.max_gate_retries,
            max_review_retries=self.config.max_review_retries,
            review_enabled=self.config.review_enabled,
        )
        lifecycle = ImplementerLifecycle(lifecycle_config)
        lifecycle_ctx = LifecycleContext()
        baseline_timestamp = input.baseline_timestamp
        if baseline_timestamp is None or baseline_timestamp <= 0:
            baseline_timestamp = int(time.time())
        lifecycle_ctx.retry_state.baseline_timestamp = baseline_timestamp

        # Build session components using AgentRuntimeBuilder
        runtime = (
            AgentRuntimeBuilder(
                self.config.repo_path,
                agent_id,
                self.sdk_client_factory,
                mcp_server_factory=self.config.mcp_server_factory,
                setting_sources=self.config.setting_sources,
            )
            .with_hooks(deadlock_monitor=self.config.deadlock_monitor)
            .with_env(extra={"MALA_SDK_FLOW": input.flow})
            .with_mcp()
            .with_disallowed_tools()
            .with_lint_tools(self.config.lint_tools)
            .build()
        )

        # Calculate idle timeout
        idle_timeout_seconds = self.config.idle_timeout_seconds
        if idle_timeout_seconds is None:
            derived = self.config.timeout_seconds * 0.2
            idle_timeout_seconds = min(900.0, max(300.0, derived))
        if idle_timeout_seconds <= 0:
            idle_timeout_seconds = None

        # Apply session resumption if resume_session_id is set
        options = runtime.options
        if input.resume_session_id:
            options = self.sdk_client_factory.with_resume(
                options, input.resume_session_id
            )

        session_config = SessionConfig(
            agent_id=agent_id,
            options=options,
            lint_cache=runtime.lint_cache,
            log_file_wait_timeout=self.config.log_file_wait_timeout,
            log_file_poll_interval=0.5,
            idle_timeout_seconds=idle_timeout_seconds,
        )

        exec_state = SessionExecutionState(
            lifecycle=lifecycle,
            lifecycle_ctx=lifecycle_ctx,
        )

        return session_config, exec_state

    async def _run_lifecycle_loop(
        self,
        input: AgentSessionInput,
        session_cfg: SessionConfig,
        state: SessionExecutionState,
        tracer: TelemetrySpan | None = None,
    ) -> None:
        """Run the main lifecycle loop.

        Executes the message iteration, gate, and review loop until
        terminal state is reached.

        Args:
            input: Session input with issue_id, prompt, etc.
            session_cfg: Derived session configuration.
            state: Mutable session execution state.
            tracer: Optional telemetry span context.
        """
        lifecycle = state.lifecycle
        lifecycle_ctx = state.lifecycle_ctx

        # Start lifecycle
        lifecycle.start()
        if self.event_sink is not None:
            self.event_sink.on_lifecycle_state(input.issue_id, lifecycle.state.name)

        pending_query: str | None = input.prompt
        result: TransitionResult | None = None

        while not lifecycle.is_terminal:
            # Check for run abort before each major stage
            abort_event = self._get_abort_event()
            if abort_event is not None and abort_event.is_set():
                logger.info(
                    "Session %s: aborting due to run abort signal", input.issue_id
                )
                lifecycle_ctx.final_result = "Session interrupted: run_aborted"
                break

            # === QUERY + MESSAGE ITERATION ===
            if pending_query is not None:
                iter_result = await self._retry_policy.execute_iteration(
                    query=pending_query,
                    issue_id=input.issue_id,
                    options=session_cfg.options,
                    state=state.msg_state,
                    lifecycle_ctx=lifecycle_ctx,
                    lint_cache=session_cfg.lint_cache,
                    idle_timeout_seconds=session_cfg.idle_timeout_seconds,
                    tracer=tracer,
                )
                if iter_result.session_id is not None:
                    state.session_id = iter_result.session_id
                pending_query = None

                result = lifecycle.on_messages_complete(
                    lifecycle_ctx, has_session_id=bool(state.session_id)
                )
                if self.event_sink is not None:
                    self.event_sink.on_lifecycle_state(
                        input.issue_id, lifecycle.state.name
                    )
                if result.effect == Effect.COMPLETE_FAILURE:
                    state.final_result = lifecycle_ctx.final_result
                    break
            else:
                assert result is not None, (
                    "Bug: entered loop without pending_query but result not set"
                )

            # Handle WAIT_FOR_LOG
            if result.effect == Effect.WAIT_FOR_LOG:
                state.log_path, result = await self._handle_log_waiting(
                    state.session_id,
                    input.issue_id,
                    state.log_path,
                    lifecycle,
                    lifecycle_ctx,
                    session_cfg.log_file_wait_timeout,
                    session_cfg.log_file_poll_interval,
                )
                if lifecycle.state == LifecycleState.FAILED:
                    state.final_result = lifecycle_ctx.final_result
                    break

            # Handle RUN_GATE
            if result.effect == Effect.RUN_GATE:
                # Check abort before starting gate
                if abort_event is not None and abort_event.is_set():
                    lifecycle_ctx.final_result = "Session interrupted: run_aborted"
                    break
                # Emit validation started BEFORE the gate check
                self._effect_handler.process_gate_check(input, lifecycle, lifecycle_ctx)

                assert state.log_path is not None
                gate_result, new_offset = await self._run_gate_check(
                    input.issue_id, state.log_path, lifecycle_ctx.retry_state
                )

                retry_query, should_break, gate_trans = (
                    self._effect_handler.process_gate_effect(
                        input, gate_result, lifecycle, lifecycle_ctx, new_offset
                    )
                )
                if should_break:
                    if lifecycle.is_terminal:
                        state.final_result = lifecycle_ctx.final_result
                        break
                    result = gate_trans
                elif retry_query is not None:
                    if state.session_id is None:
                        raise IdleTimeoutError(
                            "Cannot retry gate: session_id not received from SDK"
                        )
                    logger.debug(
                        "Session %s: queueing gate retry prompt (%d chars, session_id=%s)",
                        input.issue_id,
                        len(retry_query),
                        state.session_id[:8],
                    )
                    pending_query = retry_query
                    result = gate_trans
                    state.msg_state.pending_session_id = state.session_id
                    state.msg_state.idle_retry_count = 0
                    continue
                else:
                    result = gate_trans

            # Handle RUN_SESSION_END
            if result.effect == Effect.RUN_SESSION_END:
                # Check abort before starting session_end
                if abort_event is not None and abort_event.is_set():
                    lifecycle_ctx.final_result = "Session interrupted: run_aborted"
                    break
                # Emit session_end started event BEFORE the check
                self._effect_handler.process_session_end_check(input, lifecycle_ctx)

                # Check if session_end is configured
                if not self._has_session_end_check():
                    from src.core.session_end_result import SessionEndResult

                    session_end_result = SessionEndResult(
                        status="skipped", reason="not_configured"
                    )
                    # No log offset update when skipped without callback
                    new_offset = lifecycle_ctx.retry_state.log_offset
                else:
                    assert state.log_path is not None
                    # Build SessionEndRetryState from lifecycle context
                    session_end_retry_state = SessionEndRetryState(
                        attempt=lifecycle_ctx.retry_state.session_end_attempt,
                        max_retries=lifecycle.config.max_session_end_retries,
                        log_offset=lifecycle_ctx.retry_state.log_offset,
                        previous_commit_hash=lifecycle_ctx.retry_state.previous_commit_hash,
                    )
                    session_end_result = await self._run_session_end_check(
                        input.issue_id,
                        state.log_path,
                        session_end_retry_state,
                    )

                    # Get new log offset after callback execution
                    new_offset = self._get_log_offset(
                        state.log_path, lifecycle_ctx.retry_state.log_offset
                    )

                _, should_break, session_end_trans = (
                    self._effect_handler.process_session_end_effect(
                        input,
                        session_end_result,
                        lifecycle,
                        lifecycle_ctx,
                        new_offset,
                    )
                )

                if should_break:
                    if lifecycle.is_terminal:
                        state.final_result = lifecycle_ctx.final_result
                        break
                    result = session_end_trans
                else:
                    result = session_end_trans

            # Handle RUN_REVIEW
            if result.effect == Effect.RUN_REVIEW:
                # Check abort before starting review
                if abort_event is not None and abort_event.is_set():
                    lifecycle_ctx.final_result = "Session interrupted: run_aborted"
                    break
                assert state.log_path is not None
                cerberus_log_path: str | None = None

                # Emit gate passed events when first entering review
                self._effect_handler.process_review_check(input, lifecycle_ctx)

                # Check no-progress before running review
                if no_progress_result := self._effect_handler.check_review_no_progress(
                    input, state.log_path, lifecycle, lifecycle_ctx, cerberus_log_path
                ):
                    if no_progress_result.cerberus_log_path is not None:
                        state.cerberus_review_log_path = (
                            no_progress_result.cerberus_log_path
                        )
                    result = no_progress_result.transition_result
                    if lifecycle.is_terminal:
                        state.final_result = lifecycle_ctx.final_result
                        break
                    continue

                # Emit review_started event
                self._effect_handler.emit_review_started(input, lifecycle_ctx)

                logger.debug(
                    "Session %s: starting review (attempt %d/%d, session_id=%s)",
                    input.issue_id,
                    lifecycle_ctx.retry_state.review_attempt,
                    self.config.max_review_retries,
                    lifecycle_ctx.session_id[:8] if lifecycle_ctx.session_id else None,
                )
                review_start = time.time()
                author_context = None
                previous_findings: Sequence[ReviewIssueProtocol] | None = None
                if lifecycle_ctx.retry_state.review_attempt > 1:
                    author_context = lifecycle_ctx.final_result or None
                    # Pass previous findings so reviewer knows what was disputed
                    if lifecycle_ctx.last_review_result is not None:
                        previous_findings = cast(
                            "Sequence[ReviewIssueProtocol]",
                            lifecycle_ctx.last_review_result.issues,
                        )
                review_result = await self._run_review_check(
                    input.issue_id,
                    input.issue_description,
                    lifecycle_ctx.session_id,
                    lifecycle_ctx.retry_state,
                    author_context,
                    previous_findings,
                    lifecycle_ctx.last_session_end_result,
                )
                review_duration = time.time() - review_start
                issue_count = len(review_result.issues) if review_result.issues else 0
                blocking_count = _count_blocking_issues(review_result.issues)
                logger.debug(
                    "Session %s: review completed in %.1fs "
                    "(passed=%s, issues=%d, blocking=%d, parse_error=%s)",
                    input.issue_id,
                    review_duration,
                    review_result.passed,
                    issue_count,
                    blocking_count,
                    review_result.parse_error,
                )

                # Check for fatal error
                if review_result.fatal_error:
                    self._on_abort(
                        review_result.parse_error or "Unrecoverable review error"
                    )

                # Capture Cerberus review log if available
                log_attr = getattr(review_result, "review_log_path", None)
                if log_attr is not None:
                    cerberus_log_path = str(log_attr)

                # Get new log offset
                new_offset = self._get_log_offset(
                    state.log_path, lifecycle_ctx.retry_state.log_offset
                )

                # Process review result via effect handler
                review_effect = self._effect_handler.process_review_effect(
                    input,
                    review_result,
                    lifecycle,
                    lifecycle_ctx,
                    new_offset,
                    cerberus_log_path,
                )

                if review_effect.cerberus_log_path is not None:
                    state.cerberus_review_log_path = review_effect.cerberus_log_path
                result = review_effect.transition_result
                if review_effect.should_break:
                    if lifecycle.is_terminal:
                        state.final_result = lifecycle_ctx.final_result
                        break
                    continue
                if review_effect.pending_query is not None:
                    if state.session_id is None:
                        raise IdleTimeoutError(
                            "Cannot retry review: session_id not received from SDK"
                        )
                    logger.debug(
                        "Session %s: queueing review retry prompt (%d chars, session_id=%s)",
                        input.issue_id,
                        len(review_effect.pending_query),
                        state.session_id[:8],
                    )
                    pending_query = review_effect.pending_query
                    state.msg_state.pending_session_id = state.session_id
                    state.msg_state.idle_retry_count = 0
                continue

        state.final_result = lifecycle_ctx.final_result

    def _build_session_output(
        self,
        session_cfg: SessionConfig,
        state: SessionExecutionState,
        duration: float,
        interrupted: bool = False,
    ) -> AgentSessionOutput:
        """Build session output from execution state.

        Args:
            session_cfg: Session configuration with agent_id.
            state: Session execution state.
            duration: Total session duration in seconds.
            interrupted: Whether the session was interrupted by SIGINT.

        Returns:
            AgentSessionOutput with all results and metadata.
        """
        # Filter P0/P1 issues when session failed due to review
        last_review_issues: list[dict[str, Any]] | None = None
        if (
            not state.lifecycle_ctx.success
            and state.lifecycle_ctx.last_review_result is not None
            and state.lifecycle_ctx.retry_state.review_attempt > 0
        ):
            blocking_issues = [
                issue
                for issue in state.lifecycle_ctx.last_review_result.issues
                if issue.priority is not None and issue.priority <= 1
            ]
            if blocking_issues:
                last_review_issues = [
                    {
                        "file": issue.file,
                        "line_start": issue.line_start,
                        "line_end": issue.line_end,
                        "priority": issue.priority,
                        "title": issue.title,
                        "body": issue.body,
                        "reviewer": issue.reviewer,
                    }
                    for issue in blocking_issues
                ]

        return AgentSessionOutput(
            success=state.lifecycle_ctx.success,
            summary=state.final_result,
            session_id=state.session_id,
            log_path=state.log_path,
            gate_attempts=state.lifecycle_ctx.retry_state.gate_attempt,
            review_attempts=state.lifecycle_ctx.retry_state.review_attempt,
            resolution=state.lifecycle_ctx.resolution,
            duration_seconds=duration,
            agent_id=session_cfg.agent_id,
            review_log_path=state.cerberus_review_log_path,
            low_priority_review_issues=cast(
                "list[ReviewIssueProtocol] | None",
                state.lifecycle_ctx.low_priority_review_issues or None,
            ),
            interrupted=interrupted,
            baseline_timestamp=state.lifecycle_ctx.retry_state.baseline_timestamp,
            last_review_issues=last_review_issues,
            session_end_result=state.lifecycle_ctx.last_session_end_result,
        )

    async def _handle_log_waiting(
        self,
        session_id: str | None,
        issue_id: str,
        log_path: Path | None,
        lifecycle: ImplementerLifecycle,
        lifecycle_ctx: LifecycleContext,
        log_file_wait_timeout: float,
        log_file_poll_interval: float = 0.5,
    ) -> tuple[Path | None, TransitionResult]:
        """Handle WAIT_FOR_LOG effect - wait for log file to become available.

        Args:
            session_id: Current SDK session ID.
            issue_id: Issue ID for logging.
            log_path: Current log path (may be None).
            lifecycle: Lifecycle state machine.
            lifecycle_ctx: Lifecycle context.
            log_file_wait_timeout: Max seconds to wait for log file.
            log_file_poll_interval: Seconds between poll attempts.

        Returns:
            Tuple of (updated log_path, TransitionResult from log ready/timeout).
        """
        if session_id is None:
            raise ValueError("session_id must be set before waiting for log")
        new_log_path = self._get_log_path(session_id)

        # Reset log_offset if log file changed (new session started)
        # This prevents using a stale offset from a previous session's
        # larger log file when parsing a new, smaller log file
        if log_path is not None and new_log_path != log_path:
            logger.info(
                "Session %s: log path changed from %s to %s, "
                "resetting log_offset from %d to 0",
                issue_id,
                log_path.name,
                new_log_path.name,
                lifecycle_ctx.retry_state.log_offset,
            )
            lifecycle_ctx.retry_state.log_offset = 0

        log_path = new_log_path
        if self.event_sink is not None:
            self.event_sink.on_log_waiting(issue_id)

        # Wait for log file
        wait_elapsed = 0.0
        while not log_path.exists():
            if wait_elapsed >= log_file_wait_timeout:
                result = lifecycle.on_log_timeout(lifecycle_ctx, str(log_path))
                if self.event_sink is not None:
                    self.event_sink.on_log_timeout(issue_id, str(log_path))
                return log_path, result
            await asyncio.sleep(log_file_poll_interval)
            wait_elapsed += log_file_poll_interval

        if log_path.exists():
            if self.event_sink is not None:
                self.event_sink.on_log_ready(issue_id)
        result = lifecycle.on_log_ready(lifecycle_ctx)
        return log_path, result

    # ===== Protocol Bridge Methods =====
    # These methods delegate to protocol interfaces

    def _get_abort_event(self) -> asyncio.Event | None:
        """Get abort event from session lifecycle protocol."""
        return self.session_lifecycle.get_abort_event()

    def _get_log_path(self, session_id: str) -> Path:
        """Get log path from session lifecycle protocol."""
        return self.session_lifecycle.get_log_path(session_id)

    def _get_log_offset(self, log_path: Path, start_offset: int) -> int:
        """Get log offset from session lifecycle protocol."""
        return self.session_lifecycle.get_log_offset(log_path, start_offset)

    def _on_abort(self, reason: str) -> None:
        """Call on_abort from session lifecycle protocol."""
        self.session_lifecycle.on_abort(reason)

    async def _run_gate_check(
        self, issue_id: str, log_path: Path, retry_state: RetryState
    ) -> tuple[GateOutcome, int]:
        """Run gate check via protocol."""
        return await self.gate_runner.run_gate_check(issue_id, log_path, retry_state)

    def _has_session_end_check(self) -> bool:
        """Check if session_end check is available.

        Always returns True since gate_runner is now required.
        The actual run_session_end_check call may still return skipped
        if the gate_runner doesn't support session_end.
        """
        return True

    async def _run_session_end_check(
        self, issue_id: str, log_path: Path, retry_state: SessionEndRetryState
    ) -> SessionEndResult:
        """Run session_end check via protocol.

        If gate_runner's run_session_end_check raises NotImplementedError
        or AttributeError (method not present), returns skipped result.
        """
        from src.core.session_end_result import SessionEndResult

        try:
            return await self.gate_runner.run_session_end_check(
                issue_id, log_path, retry_state
            )
        except (NotImplementedError, AttributeError):
            # gate_runner doesn't support session_end
            return SessionEndResult(status="skipped", reason="not_configured")

    async def _run_review_check(
        self,
        issue_id: str,
        description: str | None,
        session_id: str | None,
        retry_state: RetryState,
        author_context: str | None,
        previous_findings: Sequence[ReviewIssueProtocol] | None,
        session_end_result: SessionEndResult | None,
    ) -> ReviewOutcome:
        """Run review via protocol."""
        return await self.review_runner.run_review(
            issue_id,
            description,
            session_id,
            retry_state,
            author_context,
            previous_findings,
            session_end_result,
        )

    def _get_stream_processor(self) -> MessageStreamProcessor:
        """Create a MessageStreamProcessor with protocol callbacks."""
        callbacks = StreamProcessorCallbacks(
            on_tool_use=self.session_lifecycle.on_tool_use,
            on_agent_text=self.session_lifecycle.on_agent_text,
        )
        return MessageStreamProcessor(callbacks=callbacks)

    async def run_session(
        self,
        input: AgentSessionInput,
        tracer: TelemetrySpan | None = None,
        interrupt_event: asyncio.Event | None = None,
    ) -> AgentSessionOutput:
        """Run an agent session for the given input.

        This method manages the full lifecycle of an agent session:
        1. Creates SDK client with appropriate options
        2. Sends initial prompt and streams responses
        3. Handles lifecycle transitions (gate, review, retries)
        4. Returns session output with results and metadata

        Args:
            input: AgentSessionInput with issue_id, prompt, etc.
            tracer: Optional telemetry span context.
            interrupt_event: Optional event to check for SIGINT interrupts.

        Returns:
            AgentSessionOutput with success, summary, session_id, etc.
        """
        guard = InterruptGuard(interrupt_event)

        # Use provided agent_id or generate one to preserve lock continuity across restarts
        # Computed early so it's available for early interrupt returns
        agent_id = input.agent_id or f"{input.issue_id}-{uuid.uuid4().hex[:8]}"

        # Check for early interrupt before starting
        if guard.is_interrupted():
            return AgentSessionOutput(
                success=False,
                summary="Session interrupted before start",
                agent_id=agent_id,
                interrupted=True,
                baseline_timestamp=input.baseline_timestamp,
            )

        start_time = asyncio.get_event_loop().time()
        current_prompt = input.prompt
        # Only use resume_session_id on first iteration; clear after to avoid resuming
        # on stale session retry
        current_resume_session_id = input.resume_session_id
        # Track whether we've already retried after a stale session error
        stale_session_retried = False
        # Track whether session was interrupted
        was_interrupted = False

        # Initialize state outside loop for interrupt handling before first iteration
        session_cfg: SessionConfig | None = None
        state: SessionExecutionState | None = None

        while True:
            # Check for interrupt at loop start
            if guard.is_interrupted():
                was_interrupted = True
                break

            # Calculate remaining time to enforce overall session timeout
            loop = asyncio.get_event_loop()
            elapsed = loop.time() - start_time
            remaining = self.config.timeout_seconds - elapsed

            # Create fresh lifecycle for each iteration
            session_input = AgentSessionInput(
                issue_id=input.issue_id,
                prompt=current_prompt,
                issue_description=input.issue_description,
                resume_session_id=current_resume_session_id,
                baseline_timestamp=input.baseline_timestamp,
            )
            session_cfg, state = self._initialize_session(session_input, agent_id)

            try:
                # Check timeout inside try block so on_timeout cleanup runs
                if remaining <= 0:
                    raise TimeoutError("Session timeout exceeded across restarts")
                async with asyncio.timeout(remaining):
                    await self._run_lifecycle_loop(
                        session_input, session_cfg, state, tracer
                    )
                # Normal completion - exit loop
                break
            except FlowInterruptedError:
                # Session was interrupted by SIGINT
                was_interrupted = True
                if state is not None:
                    state.final_result = "Session interrupted by SIGINT"
                break
            except IdleTimeoutError as e:
                state.lifecycle.on_error(state.lifecycle_ctx, e)
                state.final_result = state.lifecycle_ctx.final_result
                break
            except TimeoutError:
                timeout_mins = self.config.timeout_seconds // 60
                state.lifecycle.on_timeout(state.lifecycle_ctx, timeout_mins)
                state.final_result = state.lifecycle_ctx.final_result
                break
            except Exception as e:
                # Handle stale session errors when resuming
                if (
                    session_input.resume_session_id
                    and not stale_session_retried
                    and _is_stale_session_error(e)
                ):
                    # Stale session on first attempt
                    if self.config.strict_resume:
                        # Strict mode: fail instead of retrying
                        logger.warning(
                            "Stale session %s for %s in strict mode, failing: %s",
                            session_input.resume_session_id,
                            input.issue_id,
                            e,
                        )
                        state.lifecycle.on_error(state.lifecycle_ctx, e)
                        state.final_result = (
                            f"Session resumption failed (strict mode): {e}"
                        )
                        break
                    # Lenient mode: clear resume and retry once
                    logger.warning(
                        "Stale session %s for %s, retrying without resume: %s",
                        session_input.resume_session_id,
                        input.issue_id,
                        e,
                    )
                    current_resume_session_id = None
                    stale_session_retried = True
                    # Continue loop to retry with fresh session
                    continue
                state.lifecycle.on_error(state.lifecycle_ctx, e)
                state.final_result = state.lifecycle_ctx.final_result
                break

        duration = asyncio.get_event_loop().time() - start_time

        # Handle early interrupt before state was initialized
        if session_cfg is None or state is None:
            return AgentSessionOutput(
                success=False,
                summary="Session interrupted before initialization",
                agent_id=agent_id,
                duration_seconds=duration,
                interrupted=was_interrupted,
            )

        return self._build_session_output(session_cfg, state, duration, was_interrupted)

"""LifecycleEffectHandler: Encapsulates gate/review side-effect processing.

Extracted from AgentSessionRunner to separate lifecycle effect handling from
session execution. This module handles:
- Gate check processing and event emission
- Review check processing and event emission
- Gate effect handling (retry prompts, events)
- Review effect handling (retry prompts, no-progress detection, events)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from src.domain.lifecycle import Effect
from src.domain.prompts import (
    build_custom_commands_section,
    get_default_validation_commands as _get_default_validation_commands,
)
from src.infra.sigint_guard import InterruptGuard
from src.pipeline.review_formatter import format_review_issues

if TYPE_CHECKING:
    from src.core.protocols.lifecycle import ISessionLifecycle
    from src.core.protocols.review import IReviewRunner
    from src.core.protocols.validation import IGateRunner
    import asyncio
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from src.core.protocols.events import MalaEventSink
    from src.domain.lifecycle import (
        GateOutcome,
        ImplementerLifecycle,
        LifecycleContext,
        ReviewIssue,
        ReviewOutcome,
        TransitionResult,
    )
    from src.domain.validation.config import PromptValidationCommands
    from src.core.session_end_result import SessionEndResult

    from .agent_session_runner import (
        AgentSessionConfig,
        AgentSessionInput,
    )

T = TypeVar("T")


@dataclass
class SyntheticReviewOutcome:
    """A synthetic review outcome for no-progress scenarios.

    This is a simple dataclass that satisfies the ReviewResultProtocol
    used by the lifecycle state machine. It allows the pipeline to create
    synthetic failed reviews without importing from infra.clients.
    """

    passed: bool
    issues: list[ReviewIssue]
    parse_error: str | None = None
    fatal_error: bool = False
    review_log_path: Path | None = None
    interrupted: bool = False


@dataclass
class ReviewEffectResult:
    """Result from review effect processing.

    Encapsulates the multi-value return from LifecycleEffectHandler's
    review processing methods with named fields for clarity.
    """

    pending_query: str | None
    """Query to send for review retry, or None if no retry needed."""

    should_break: bool
    """Whether the caller should break out of the message iteration loop."""

    cerberus_log_path: str | None
    """Path to Cerberus review log file, if captured."""

    transition_result: TransitionResult
    """Lifecycle transition result."""


logger = logging.getLogger(__name__)


def _emit_review_result_events(
    event_sink: MalaEventSink | None,
    input: AgentSessionInput,
    result: TransitionResult,
    review_result: ReviewOutcome,
    lifecycle_ctx: LifecycleContext,
    max_review_retries: int,
    blocking_count: int,
) -> None:
    """Emit events based on review result transition.

    Handles event emission for:
    - COMPLETE_SUCCESS: on_review_passed
    - RUN_REVIEW (parse error): on_warning
    - SEND_REVIEW_RETRY: on_review_retry with blocking_count
    """
    if event_sink is None:
        return

    if result.effect == Effect.COMPLETE_SUCCESS:
        event_sink.on_review_passed(
            input.issue_id,
            issue_id=input.issue_id,
        )
        return

    if result.effect == Effect.RUN_REVIEW:
        error_detail = review_result.parse_error or "unknown error"
        event_sink.on_warning(
            f"Review tool error: {error_detail}; retrying",
            agent_id=input.issue_id,
        )
        return

    if result.effect == Effect.SEND_REVIEW_RETRY:
        logger.debug(
            "Session %s: SEND_REVIEW_RETRY triggered "
            "(attempt %d/%d, %d blocking issues)",
            input.issue_id,
            lifecycle_ctx.retry_state.review_attempt,
            max_review_retries,
            blocking_count,
        )
        event_sink.on_review_retry(
            input.issue_id,
            lifecycle_ctx.retry_state.review_attempt,
            max_review_retries,
            error_count=blocking_count or None,
            parse_error=review_result.parse_error,
            issue_id=input.issue_id,
        )


def _emit_gate_passed_events(
    event_sink: MalaEventSink | None,
    issue_id: str,
    review_attempt: int,
) -> None:
    """Emit gate passed events when first entering review.

    Only emits events on the first review attempt (review_attempt == 1),
    indicating the gate has just passed.

    Args:
        event_sink: Event sink to emit to, or None to skip emission.
        issue_id: The issue identifier.
        review_attempt: Current review attempt number (1-based).
    """
    if review_attempt != 1 or event_sink is None:
        return

    event_sink.on_gate_passed(
        issue_id,
        issue_id=issue_id,
    )
    event_sink.on_validation_result(
        issue_id,
        passed=True,
        issue_id=issue_id,
    )


def _count_blocking_issues(issues: list[ReviewIssue] | None) -> int:
    """Count issues with priority <= 1 (P0 or P1).

    Args:
        issues: List of review issues, or None.

    Returns:
        Number of blocking (high-priority) issues.
    """
    if not issues:
        return 0
    return sum(1 for i in issues if i.priority is not None and i.priority <= 1)


def _make_review_effect_result(
    effect: Effect,
    cerberus_log_path: str | None,
    transition_result: TransitionResult,
    pending_query: str | None = None,
) -> ReviewEffectResult:
    """Build ReviewEffectResult based on effect type.

    Args:
        effect: The lifecycle effect to handle.
        cerberus_log_path: Path to Cerberus review log.
        transition_result: Lifecycle transition result.
        pending_query: Query string for SEND_REVIEW_RETRY effect.

    Returns:
        ReviewEffectResult with appropriate should_break flag.
    """
    should_break = effect in (Effect.COMPLETE_SUCCESS, Effect.COMPLETE_FAILURE)
    return ReviewEffectResult(
        pending_query=pending_query,
        should_break=should_break,
        cerberus_log_path=cerberus_log_path,
        transition_result=transition_result,
    )


def _build_review_retry_prompt(
    review_result: ReviewOutcome,
    lifecycle_ctx: LifecycleContext,
    issue_id: str,
    repo_path: Path,
    max_review_retries: int,
    review_followup_template: str,
    validation_commands: PromptValidationCommands,
) -> str:
    """Build the follow-up prompt for review retry.

    Args:
        review_result: The review outcome with issues to address.
        lifecycle_ctx: Lifecycle context with retry state.
        issue_id: The issue identifier.
        repo_path: Repository path for formatting issue paths.
        max_review_retries: Maximum number of review retries.
        review_followup_template: Template for review follow-up prompts.
        validation_commands: Validation commands for the prompt.

    Returns:
        Formatted prompt string for the agent to address review issues.
    """
    review_issues_text = format_review_issues(
        review_result.issues,  # type: ignore[arg-type]  # ReviewIssue ⊂ ReviewIssueProtocol
        base_path=repo_path,
    )
    custom_commands_section = build_custom_commands_section(
        validation_commands.custom_commands
    )
    return review_followup_template.format(
        attempt=lifecycle_ctx.retry_state.review_attempt,
        max_attempts=max_review_retries,
        review_issues=review_issues_text,
        issue_id=issue_id,
        lint_command=validation_commands.lint,
        format_command=validation_commands.format,
        typecheck_command=validation_commands.typecheck,
        custom_commands_section=custom_commands_section,
        test_command=validation_commands.test,
    )


@dataclass
class LifecycleEffectHandler:
    """Handles gate/review side-effect processing.

    Encapsulates all gate/review effect handling logic extracted from
    AgentSessionRunner. Manages event emission, retry prompt building,
    and lifecycle state transitions for gate and review operations.

    The handler uses protocol interfaces for external operations (gate checks,
    reviews, lifecycle operations) to decouple from orchestrator internals.

    Usage:
        handler = LifecycleEffectHandler(
            config=config,
            event_sink=event_sink,
            gate_runner=gate_runner_impl,
            review_runner=review_runner_impl,
            session_lifecycle=session_lifecycle_impl,
        )
        handler.process_gate_check(input, lifecycle, lifecycle_ctx)
        retry_query, should_break, trans = handler.process_gate_effect(...)
        review_effect = handler.process_review_effect(...)

    Attributes:
        config: Session configuration with prompts and retry limits.
        event_sink: Optional event sink for structured logging.
        gate_runner: Protocol for gate checking operations (required).
        review_runner: Protocol for review operations (required).
        session_lifecycle: Protocol for session lifecycle operations (required).
    """

    config: AgentSessionConfig
    gate_runner: IGateRunner
    review_runner: IReviewRunner
    session_lifecycle: ISessionLifecycle
    event_sink: MalaEventSink | None = None

    def process_gate_check(
        self,
        input: AgentSessionInput,
        lifecycle: ImplementerLifecycle,
        lifecycle_ctx: LifecycleContext,
    ) -> None:
        """Emit events before gate check.

        Called before running the gate check to emit validation_started
        and gate_started events.

        Args:
            input: Session input with issue_id.
            lifecycle: Lifecycle state machine (unused, for interface consistency).
            lifecycle_ctx: Lifecycle context with retry state.
        """
        if self.event_sink is not None:
            self.event_sink.on_validation_started(
                input.issue_id, issue_id=input.issue_id
            )
            self.event_sink.on_gate_started(
                input.issue_id,
                lifecycle_ctx.retry_state.gate_attempt,
                self.config.max_gate_retries,
                issue_id=input.issue_id,
            )

    def process_gate_effect(
        self,
        input: AgentSessionInput,
        gate_result: GateOutcome,
        lifecycle: ImplementerLifecycle,
        lifecycle_ctx: LifecycleContext,
        new_offset: int,
    ) -> tuple[str | None, bool, TransitionResult]:
        """Handle RUN_GATE effect - process gate result and emit events.

        Args:
            input: Session input with issue_id.
            gate_result: Result from gate check callback.
            lifecycle: Lifecycle state machine.
            lifecycle_ctx: Lifecycle context.
            new_offset: New log offset after gate check.

        Returns:
            Tuple of (pending_query for retry or None, should_break, transition_result).
        """
        result = lifecycle.on_gate_result(lifecycle_ctx, gate_result, new_offset)

        if result.effect == Effect.COMPLETE_SUCCESS:
            _emit_gate_passed_events(self.event_sink, input.issue_id, review_attempt=1)
            return None, True, result  # break

        if result.effect == Effect.COMPLETE_FAILURE:
            if self.event_sink is not None:
                self.event_sink.on_gate_failed(
                    input.issue_id,
                    lifecycle_ctx.retry_state.gate_attempt,
                    self.config.max_gate_retries,
                    issue_id=input.issue_id,
                )
                self.event_sink.on_gate_result(
                    input.issue_id,
                    passed=False,
                    failure_reasons=list(gate_result.failure_reasons),
                    issue_id=input.issue_id,
                )
                self.event_sink.on_validation_result(
                    input.issue_id,
                    passed=False,
                    issue_id=input.issue_id,
                )
            return None, True, result  # break

        if result.effect == Effect.SEND_GATE_RETRY:
            if self.event_sink is not None:
                self.event_sink.on_gate_retry(
                    input.issue_id,
                    lifecycle_ctx.retry_state.gate_attempt,
                    self.config.max_gate_retries,
                    issue_id=input.issue_id,
                )
                self.event_sink.on_gate_result(
                    input.issue_id,
                    passed=False,
                    failure_reasons=list(gate_result.failure_reasons),
                    issue_id=input.issue_id,
                )
                # Emit validation_result before retry so every
                # on_validation_started has a corresponding result
                self.event_sink.on_validation_result(
                    input.issue_id,
                    passed=False,
                    issue_id=input.issue_id,
                )
            # Build follow-up prompt
            failure_text = "\n".join(f"- {r}" for r in gate_result.failure_reasons)
            # Get validation commands or use defaults
            cmds = (
                self.config.prompt_validation_commands
                or _get_default_validation_commands()
            )
            pending_query = self.config.prompts.gate_followup.format(
                attempt=lifecycle_ctx.retry_state.gate_attempt,
                max_attempts=self.config.max_gate_retries,
                failure_reasons=failure_text,
                issue_id=input.issue_id,
                lint_command=cmds.lint,
                format_command=cmds.format,
                typecheck_command=cmds.typecheck,
                test_command=cmds.test,
            )
            return pending_query, False, result  # continue with retry

        # RUN_REVIEW or other effects - pass through
        return None, False, result

    def process_session_end_check(
        self,
        input: AgentSessionInput,
        lifecycle_ctx: LifecycleContext,
    ) -> None:
        """Emit events before session_end check.

        Called before running the session_end check to emit gate_passed and
        session_end_started events. Gate passed is emitted on first session_end
        attempt since session_end only runs after the gate passes.

        Args:
            input: Session input with issue_id.
            lifecycle_ctx: Lifecycle context with retry state.
        """
        # Emit gate_passed on first session_end attempt (gate just passed)
        if lifecycle_ctx.retry_state.session_end_attempt == 1:
            _emit_gate_passed_events(self.event_sink, input.issue_id, review_attempt=1)
        if self.event_sink is not None:
            self.event_sink.on_session_end_started(input.issue_id)

    def process_session_end_effect(
        self,
        input: AgentSessionInput,
        session_end_result: SessionEndResult,
        lifecycle: ImplementerLifecycle,
        lifecycle_ctx: LifecycleContext,
        new_offset: int,
    ) -> tuple[str | None, bool, TransitionResult]:
        """Handle RUN_SESSION_END effect - process session_end result and emit events.

        Args:
            input: Session input with issue_id.
            session_end_result: Result from session_end callback.
            lifecycle: Lifecycle state machine.
            lifecycle_ctx: Lifecycle context.
            new_offset: New log offset after session_end check.

        Returns:
            Tuple of (pending_query for retry or None, should_break, transition_result).
        """
        # Process result through lifecycle state machine
        # can_remediate=False for now (T011 will add proper remediation support)
        result = lifecycle.on_session_end_result(
            lifecycle_ctx, session_end_result, new_offset, can_remediate=False
        )

        # Emit session_end completed/skipped events
        if self.event_sink is not None:
            if session_end_result.status == "skipped":
                self.event_sink.on_session_end_skipped(
                    input.issue_id, session_end_result.reason or "unknown"
                )
            else:
                self.event_sink.on_session_end_completed(
                    input.issue_id, session_end_result.status
                )

        # Session_end never blocks - always proceed to review or success
        should_break = result.effect in (
            Effect.COMPLETE_SUCCESS,
            Effect.COMPLETE_FAILURE,
        )
        return None, should_break, result

    def process_review_check(
        self,
        input: AgentSessionInput,
        lifecycle_ctx: LifecycleContext,
    ) -> None:
        """Emit gate passed events when first entering review.

        Called at the start of review processing to emit gate passed events
        on the first review attempt. Skips emission if session_end already ran
        (session_end_attempt > 0) since gate_passed was already emitted there.

        Args:
            input: Session input with issue_id.
            lifecycle_ctx: Lifecycle context with retry state.
        """
        # Skip if session_end already emitted gate_passed
        if lifecycle_ctx.retry_state.session_end_attempt > 0:
            return
        _emit_gate_passed_events(
            self.event_sink, input.issue_id, lifecycle_ctx.retry_state.review_attempt
        )

    def check_review_no_progress(
        self,
        input: AgentSessionInput,
        log_path: Path,
        lifecycle: ImplementerLifecycle,
        lifecycle_ctx: LifecycleContext,
        cerberus_review_log_path: str | None,
    ) -> ReviewEffectResult | None:
        """Check if review retry has made no progress and should be skipped.

        Args:
            input: Session input with issue_id.
            log_path: Path to log file.
            lifecycle: Lifecycle state machine.
            lifecycle_ctx: Lifecycle context.
            cerberus_review_log_path: Path to Cerberus review log, if any.

        Returns:
            ReviewEffectResult if no progress detected (caller should return early),
            None if review should proceed normally.
        """
        # Only check on retry attempts (attempt > 1)
        if lifecycle_ctx.retry_state.review_attempt <= 1:
            return None

        current_commit = (
            lifecycle_ctx.last_gate_result.commit_hash
            if lifecycle_ctx.last_gate_result
            else None
        )

        no_progress = self.review_runner.check_no_progress(
            log_path,
            lifecycle_ctx.retry_state.log_offset,
            lifecycle_ctx.retry_state.previous_commit_hash,
            current_commit,
        )

        if not no_progress:
            return None

        # Emit event for no-progress skip
        if self.event_sink is not None:
            self.event_sink.on_review_skipped_no_progress(input.issue_id)

        # Create synthetic failed review
        synthetic = SyntheticReviewOutcome(passed=False, issues=[])

        new_offset = self.session_lifecycle.get_log_offset(
            log_path, lifecycle_ctx.retry_state.log_offset
        )
        no_progress_result = lifecycle.on_review_result(
            lifecycle_ctx,
            synthetic,
            new_offset,
            no_progress=True,
        )
        return ReviewEffectResult(
            pending_query=None,
            should_break=True,
            cerberus_log_path=cerberus_review_log_path,
            transition_result=no_progress_result,
        )

    def emit_review_started(
        self,
        input: AgentSessionInput,
        lifecycle_ctx: LifecycleContext,
    ) -> None:
        """Emit review_started event.

        Args:
            input: Session input with issue_id.
            lifecycle_ctx: Lifecycle context with retry state.
        """
        if self.event_sink is not None:
            self.event_sink.on_review_started(
                input.issue_id,
                lifecycle_ctx.retry_state.review_attempt,
                self.config.max_review_retries,
                issue_id=input.issue_id,
            )

    def process_review_effect(
        self,
        input: AgentSessionInput,
        review_result: ReviewOutcome,
        lifecycle: ImplementerLifecycle,
        lifecycle_ctx: LifecycleContext,
        new_offset: int,
        cerberus_log_path: str | None,
    ) -> ReviewEffectResult:
        """Handle review result and build ReviewEffectResult.

        Args:
            input: Session input with issue_id.
            review_result: The review outcome.
            lifecycle: Lifecycle state machine.
            lifecycle_ctx: Lifecycle context.
            new_offset: New log offset after review.
            cerberus_log_path: Path to Cerberus review log, if any.

        Returns:
            ReviewEffectResult with pending_query, should_break, cerberus_log_path,
            and transition_result.
        """
        blocking = _count_blocking_issues(review_result.issues)

        result = lifecycle.on_review_result(lifecycle_ctx, review_result, new_offset)

        # Emit appropriate events based on transition result
        _emit_review_result_events(
            self.event_sink,
            input,
            result,
            review_result,
            lifecycle_ctx,
            self.config.max_review_retries,
            blocking,
        )

        # Build pending_query only for SEND_REVIEW_RETRY
        pending_query = None
        if result.effect == Effect.SEND_REVIEW_RETRY:
            cmds = (
                self.config.prompt_validation_commands
                or _get_default_validation_commands()
            )
            pending_query = _build_review_retry_prompt(
                review_result,
                lifecycle_ctx,
                input.issue_id,
                self.config.repo_path,
                self.config.max_review_retries,
                self.config.prompts.review_followup,
                cmds,
            )

        return _make_review_effect_result(
            result.effect,
            cerberus_log_path,
            result,
            pending_query,
        )

    async def run_effects(
        self,
        effect_fns: list[Callable[[], Awaitable[T]]],
        interrupt_event: asyncio.Event | None = None,
    ) -> tuple[list[T], bool]:
        """Run a list of effect functions, stopping if interrupted.

        This method iterates over effect functions and executes each one,
        checking for interrupt between each effect. If interrupted, returns
        partial results with remaining effects skipped.

        Args:
            effect_fns: List of async callables that produce effect results.
            interrupt_event: Optional event to check for SIGINT interrupts.

        Returns:
            Tuple of (completed_results, was_interrupted) where:
            - completed_results: List of results from successfully run effects.
            - was_interrupted: True if iteration was stopped due to interrupt.
        """
        guard = InterruptGuard(interrupt_event)
        results: list[T] = []

        for effect_fn in effect_fns:
            # Check interrupt before each effect
            if guard.is_interrupted():
                return (results, True)

            result = await effect_fn()
            results.append(result)

        return (results, False)

"""Implementer lifecycle state machine for orchestrator control flow.

Extracts the retry/gate/review policy as a pure state machine that can be
tested without Claude SDK or subprocess dependencies.

The state machine is data-in/data-out: it receives events and returns
effects (actions the orchestrator should take). This separation allows
the orchestrator to remain responsible for I/O while the lifecycle
handles all policy decisions.

This module provides the canonical RetryState and lifecycle state machine:

1. **Testing policy in isolation**: Use ImplementerLifecycle directly to verify
   retry/gate/review transitions without mocking SDK or subprocesses.

2. **Integration with orchestrator**: The orchestrator imports RetryState from
   this module and uses lifecycle_ctx.retry_state directly for gate checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .validation.spec import ResolutionOutcome

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.core.session_end_result import SessionEndResult

    from .validation.spec import IssueResolution

# Resolution outcomes that skip review (no new code to review)
_SKIP_REVIEW_OUTCOMES = frozenset(
    {
        ResolutionOutcome.NO_CHANGE,
        ResolutionOutcome.OBSOLETE,
        ResolutionOutcome.ALREADY_COMPLETE,
        ResolutionOutcome.DOCS_ONLY,
    }
)


# ---------------------------------------------------------------------------
# Local outcome protocols: define the interface lifecycle needs from infra
# ---------------------------------------------------------------------------


@runtime_checkable
class GateOutcome(Protocol):
    """Protocol defining what lifecycle needs from a gate result.

    Callers (orchestrator) pass infra GateResult objects that satisfy this
    protocol. Lifecycle only accesses these fields.
    """

    @property
    def passed(self) -> bool:
        """Whether the gate check passed."""
        ...

    @property
    def failure_reasons(self) -> list[str]:
        """Reasons for failure (empty if passed)."""
        ...

    @property
    def commit_hash(self) -> str | None:
        """Commit hash if a commit was found."""
        ...

    @property
    def no_progress(self) -> bool:
        """Whether no progress was detected since last attempt."""
        ...

    @property
    def resolution(self) -> IssueResolution | None:
        """Issue resolution if a resolution marker was found."""
        ...


class ReviewIssue(Protocol):
    """Protocol for a single issue from an external review.

    Matches the fields lifecycle needs from review issues for building
    failure messages. Uses Protocol to allow structural subtyping with
    cerberus_review.ReviewIssue.
    """

    @property
    def file(self) -> str:
        """File path where the issue was found."""
        ...

    @property
    def line_start(self) -> int:
        """Starting line number of the issue."""
        ...

    @property
    def line_end(self) -> int:
        """Ending line number of the issue."""
        ...

    @property
    def priority(self) -> int | None:
        """Priority level: 0=P0, 1=P1, 2=P2, 3=P3, or None if unknown."""
        ...

    @property
    def title(self) -> str:
        """Short title of the issue."""
        ...

    @property
    def body(self) -> str:
        """Detailed description of the issue."""
        ...

    @property
    def reviewer(self) -> str:
        """Identifier of the reviewer that found this issue."""
        ...


@runtime_checkable
class ReviewOutcome(Protocol):
    """Protocol defining what lifecycle needs from an external review result.

    Callers (orchestrator) pass infra review result objects (e.g., ReviewResult
    from cerberus_review) that satisfy this protocol. Lifecycle only accesses
    these fields.
    """

    @property
    def passed(self) -> bool:
        """Whether the review passed."""
        ...

    @property
    def parse_error(self) -> str | None:
        """Parse error message if JSON parsing failed."""
        ...

    @property
    def fatal_error(self) -> bool:
        """Whether the review failure is unrecoverable."""
        ...

    @property
    def issues(self) -> list[ReviewIssue]:
        """List of issues found during review."""
        ...

    @property
    def interrupted(self) -> bool:
        """Whether the review was interrupted by SIGINT."""
        ...


class LifecycleState(Enum):
    """States in the implementer lifecycle."""

    # Initial state - agent session starting
    INITIAL = auto()
    # Processing messages from SDK stream
    PROCESSING = auto()
    # Waiting for log file to appear
    AWAITING_LOG = auto()
    # Running quality gate check
    RUNNING_GATE = auto()
    # Running session_end trigger (after gate passed, before review)
    RUNNING_SESSION_END = auto()
    # Remediating session_end issues (agent fixing problems)
    SESSION_END_REMEDIATING = auto()
    # Running external review (after gate passed)
    RUNNING_REVIEW = auto()
    # Terminal: success
    SUCCESS = auto()
    # Terminal: failed
    FAILED = auto()


class Effect(Enum):
    """Effects/actions the orchestrator should perform.

    These are returned by state transitions to tell the orchestrator
    what I/O actions to take.
    """

    # Continue processing SDK messages
    CONTINUE = auto()
    # Wait for log file to appear
    WAIT_FOR_LOG = auto()
    # Run quality gate check
    RUN_GATE = auto()
    # Run session_end trigger
    RUN_SESSION_END = auto()
    # Run external review
    RUN_REVIEW = auto()
    # Send gate retry follow-up prompt to SDK
    SEND_GATE_RETRY = auto()
    # Send session_end retry follow-up prompt to SDK
    SEND_SESSION_END_RETRY = auto()
    # Send review retry follow-up prompt to SDK
    SEND_REVIEW_RETRY = auto()
    # Complete with success
    COMPLETE_SUCCESS = auto()
    # Complete with failure
    COMPLETE_FAILURE = auto()


@dataclass
class LifecycleConfig:
    """Configuration for lifecycle behavior."""

    max_gate_retries: int = 3
    max_session_end_retries: int = 3
    max_review_retries: int = 3
    session_end_enabled: bool = True
    review_enabled: bool = True


@dataclass
class RetryState:
    """Tracks retry attempts for gate, session_end, and review.

    This is mutable state that the lifecycle updates during transitions.
    The orchestrator can read it to format follow-up prompts.
    """

    gate_attempt: int = 1
    session_end_attempt: int = 0
    review_attempt: int = 0
    log_offset: int = 0
    previous_commit_hash: str | None = None
    baseline_timestamp: int = 0


@dataclass
class LifecycleContext:
    """Context passed to and updated by state transitions.

    This bundles the mutable state and accumulated results that
    the lifecycle tracks across transitions.
    """

    retry_state: RetryState = field(default_factory=RetryState)
    session_id: str | None = None
    final_result: str = ""
    success: bool = False
    # Last gate result for building failure messages
    last_gate_result: GateOutcome | None = None
    # Last session_end result for review evidence
    last_session_end_result: SessionEndResult | None = None
    # Last review result for building follow-up prompts
    last_review_result: ReviewOutcome | None = None
    # Resolution for issue (no-op, obsolete, etc.)
    resolution: IssueResolution | None = None
    # P2/P3 issues from review to create as tracking issues
    # These are low-priority issues that don't block the review but should be tracked
    low_priority_review_issues: list[ReviewIssue] = field(default_factory=list)


@dataclass
class TransitionResult:
    """Result of a state transition.

    Contains the new state and the effect the orchestrator should perform.
    """

    state: LifecycleState
    effect: Effect
    # Optional message explaining the transition (for logging)
    message: str | None = None


class ImplementerLifecycle:
    """Pure state machine for implementer agent lifecycle.

    This class encapsulates all policy decisions about:
    - When to run gate vs review
    - When to retry vs fail
    - How to track attempt counts

    It does NOT perform any I/O - the orchestrator handles that based
    on the Effect returned by each transition.
    """

    def __init__(self, config: LifecycleConfig):
        self.config = config
        self._state = LifecycleState.INITIAL

    @property
    def state(self) -> LifecycleState:
        """Current lifecycle state."""
        return self._state

    @property
    def is_terminal(self) -> bool:
        """Whether the lifecycle has reached a terminal state."""
        return self._state in (LifecycleState.SUCCESS, LifecycleState.FAILED)

    def start(self) -> TransitionResult:
        """Begin the lifecycle - transition from INITIAL to PROCESSING."""
        if self._state != LifecycleState.INITIAL:
            raise ValueError(f"Cannot start from state {self._state}")
        self._state = LifecycleState.PROCESSING
        logger.info("Lifecycle started: state=%s", self._state.name)
        return TransitionResult(
            state=self._state,
            effect=Effect.CONTINUE,
            message="Agent session started",
        )

    def on_messages_complete(
        self, ctx: LifecycleContext, has_session_id: bool
    ) -> TransitionResult:
        """Handle completion of message processing.

        Called when the SDK receive_response iterator completes.
        Transitions to AWAITING_LOG if we have a session ID.
        """
        if self._state != LifecycleState.PROCESSING:
            raise ValueError(f"Unexpected state for messages_complete: {self._state}")

        if not has_session_id:
            # No session ID means no log to check
            ctx.final_result = "No session ID received from agent"
            ctx.success = False
            self._state = LifecycleState.FAILED
            return TransitionResult(
                state=self._state,
                effect=Effect.COMPLETE_FAILURE,
                message="No session ID",
            )

        self._state = LifecycleState.AWAITING_LOG
        logger.debug("Messages complete: effect=%s", Effect.WAIT_FOR_LOG.name)
        return TransitionResult(
            state=self._state,
            effect=Effect.WAIT_FOR_LOG,
        )

    def on_log_ready(self, ctx: LifecycleContext) -> TransitionResult:
        """Handle log file becoming available.

        Transitions to RUNNING_GATE.
        """
        if self._state != LifecycleState.AWAITING_LOG:
            raise ValueError(f"Unexpected state for log_ready: {self._state}")

        self._state = LifecycleState.RUNNING_GATE
        return TransitionResult(
            state=self._state,
            effect=Effect.RUN_GATE,
        )

    def on_log_timeout(self, ctx: LifecycleContext, log_path: str) -> TransitionResult:
        """Handle timeout waiting for log file.

        Transitions to FAILED.
        """
        if self._state != LifecycleState.AWAITING_LOG:
            raise ValueError(f"Unexpected state for log_timeout: {self._state}")

        ctx.final_result = f"Session log missing after timeout: {log_path}"
        ctx.success = False
        self._state = LifecycleState.FAILED
        return TransitionResult(
            state=self._state,
            effect=Effect.COMPLETE_FAILURE,
            message="Log file timeout",
        )

    def on_gate_result(
        self, ctx: LifecycleContext, gate_result: GateOutcome, new_log_offset: int
    ) -> TransitionResult:
        """Handle quality gate result.

        Decides whether to:
        - Proceed to review (if gate passed and review enabled)
        - Complete with success (if gate passed and review disabled)
        - Retry gate (if failed but retries remain)
        - Fail (if no retries left or no progress)
        """
        if self._state != LifecycleState.RUNNING_GATE:
            raise ValueError(f"Unexpected state for gate_result: {self._state}")

        ctx.last_gate_result = gate_result
        ctx.resolution = gate_result.resolution

        logger.info(
            "Gate result: outcome=%s attempt=%d state=%s",
            "passed" if gate_result.passed else "failed",
            ctx.retry_state.gate_attempt,
            self._state.name,
        )

        if gate_result.passed:
            # Gate passed - should we run session_end?
            # Skip session_end for resolutions with no new code
            # (no_change, obsolete, already_complete, docs_only)
            resolution_skips_session_end = (
                gate_result.resolution is not None
                and gate_result.resolution.outcome in _SKIP_REVIEW_OUTCOMES
            )
            if (
                self.config.session_end_enabled
                and gate_result.commit_hash
                and not resolution_skips_session_end
            ):
                # Initialize session_end_attempt if not already started
                if ctx.retry_state.session_end_attempt == 0:
                    ctx.retry_state.session_end_attempt = 1
                self._state = LifecycleState.RUNNING_SESSION_END
                return TransitionResult(
                    state=self._state,
                    effect=Effect.RUN_SESSION_END,
                    message=f"Gate passed, running session_end (attempt {ctx.retry_state.session_end_attempt}/{self.config.max_session_end_retries})",
                )
            else:
                # No session_end needed - proceed to review or success
                return self._proceed_to_review_or_success(ctx, gate_result)

        # Gate failed - can we retry?
        can_retry = (
            ctx.retry_state.gate_attempt < self.config.max_gate_retries
            and not gate_result.no_progress
        )

        if can_retry:
            # Prepare for retry
            ctx.retry_state.gate_attempt += 1
            ctx.retry_state.log_offset = new_log_offset
            ctx.retry_state.previous_commit_hash = gate_result.commit_hash
            self._state = LifecycleState.PROCESSING
            logger.debug(
                "Retry triggered: reason=gate_failed attempt=%d/%d",
                ctx.retry_state.gate_attempt,
                self.config.max_gate_retries,
            )
            return TransitionResult(
                state=self._state,
                effect=Effect.SEND_GATE_RETRY,
                message=f"Gate retry {ctx.retry_state.gate_attempt}/{self.config.max_gate_retries}",
            )

        # No retries left or no progress - fail
        ctx.final_result = (
            f"Quality gate failed: {'; '.join(gate_result.failure_reasons)}"
        )
        ctx.success = False
        self._state = LifecycleState.FAILED
        logger.info(
            "Lifecycle terminal: state=%s message=%s",
            self._state.name,
            "Gate failed, no retries left",
        )
        return TransitionResult(
            state=self._state,
            effect=Effect.COMPLETE_FAILURE,
            message="Gate failed, no retries left",
        )

    def _proceed_to_review_or_success(
        self, ctx: LifecycleContext, gate_result: GateOutcome
    ) -> TransitionResult:
        """Proceed to review or complete with success.

        Helper method called when gate passed and session_end is skipped/passed.
        """
        # Skip review for resolutions with no new code to review
        resolution_skips_review = (
            gate_result.resolution is not None
            and gate_result.resolution.outcome in _SKIP_REVIEW_OUTCOMES
        )
        if (
            self.config.review_enabled
            and gate_result.commit_hash
            and not resolution_skips_review
        ):
            # Only initialize review_attempt if not already started
            # (preserves count across gate re-runs after review retry)
            if ctx.retry_state.review_attempt == 0:
                ctx.retry_state.review_attempt = 1
            self._state = LifecycleState.RUNNING_REVIEW
            return TransitionResult(
                state=self._state,
                effect=Effect.RUN_REVIEW,
                message=f"Running review (attempt {ctx.retry_state.review_attempt}/{self.config.max_review_retries})",
            )
        else:
            # No review needed - success!
            ctx.success = True
            self._state = LifecycleState.SUCCESS
            return TransitionResult(
                state=self._state,
                effect=Effect.COMPLETE_SUCCESS,
                message="No review required",
            )

    def on_session_end_result(
        self,
        ctx: LifecycleContext,
        session_end_result: SessionEndResult,
        new_log_offset: int,
        no_progress: bool = False,
        can_remediate: bool = False,
    ) -> TransitionResult:
        """Handle session_end trigger result.

        Decides whether to:
        - Proceed to review (if session_end passed or no blocking issues)
        - Retry via agent prompt (if failed but can_remediate=True and retries remain)
        - Complete with success (if review disabled)

        Args:
            ctx: Lifecycle context with retry state.
            session_end_result: The session_end outcome to process.
            new_log_offset: Updated log offset for next attempt.
            no_progress: If True, the agent made no progress since last attempt.
            can_remediate: If True and retries remain, trigger remediation loop.
                Set by handler based on trigger config failure_mode=remediate.
        """
        if self._state != LifecycleState.RUNNING_SESSION_END:
            raise ValueError(f"Unexpected state for session_end_result: {self._state}")

        ctx.last_session_end_result = session_end_result

        logger.info(
            "Session_end result: status=%s attempt=%d state=%s",
            session_end_result.status,
            ctx.retry_state.session_end_attempt,
            self._state.name,
        )

        # Session_end passed - proceed to review or success
        if session_end_result.status == "pass":
            # Use gate_result from context to determine review behavior
            gate_result = ctx.last_gate_result
            if gate_result is None:
                # Shouldn't happen, but handle gracefully
                ctx.success = True
                self._state = LifecycleState.SUCCESS
                return TransitionResult(
                    state=self._state,
                    effect=Effect.COMPLETE_SUCCESS,
                    message="Session_end passed, no gate result",
                )
            return self._proceed_to_review_or_success(ctx, gate_result)

        # Session_end skipped - proceed directly to review
        if session_end_result.status == "skipped":
            gate_result = ctx.last_gate_result
            if gate_result is None:
                ctx.success = True
                self._state = LifecycleState.SUCCESS
                return TransitionResult(
                    state=self._state,
                    effect=Effect.COMPLETE_SUCCESS,
                    message="Session_end skipped, no gate result",
                )
            return self._proceed_to_review_or_success(ctx, gate_result)

        # Session_end interrupted - proceed to review with interrupted status
        if session_end_result.status == "interrupted":
            gate_result = ctx.last_gate_result
            if gate_result is None:
                ctx.success = True
                self._state = LifecycleState.SUCCESS
                return TransitionResult(
                    state=self._state,
                    effect=Effect.COMPLETE_SUCCESS,
                    message="Session_end interrupted, no gate result",
                )
            return self._proceed_to_review_or_success(ctx, gate_result)

        # Session_end failed or timed out - check if we can retry
        can_retry = (
            ctx.retry_state.session_end_attempt < self.config.max_session_end_retries
            and not no_progress
            and can_remediate
        )

        if can_retry:
            # Transition to remediation state
            ctx.retry_state.session_end_attempt += 1
            ctx.retry_state.log_offset = new_log_offset
            self._state = LifecycleState.SESSION_END_REMEDIATING
            logger.debug(
                "Session_end retry triggered: attempt=%d/%d",
                ctx.retry_state.session_end_attempt,
                self.config.max_session_end_retries,
            )
            return TransitionResult(
                state=self._state,
                effect=Effect.SEND_SESSION_END_RETRY,
                message=f"Session_end retry {ctx.retry_state.session_end_attempt}/{self.config.max_session_end_retries}",
            )

        # No retries or can't remediate - proceed to review anyway
        # Per spec, session_end failures don't block review - they're informational
        gate_result = ctx.last_gate_result
        if gate_result is None:
            ctx.success = True
            self._state = LifecycleState.SUCCESS
            return TransitionResult(
                state=self._state,
                effect=Effect.COMPLETE_SUCCESS,
                message="Session_end failed, no gate result",
            )
        return self._proceed_to_review_or_success(ctx, gate_result)

    def on_session_end_remediation_complete(
        self, ctx: LifecycleContext
    ) -> TransitionResult:
        """Handle completion of session_end remediation (agent finished fixing).

        Transitions back to RUNNING_SESSION_END to re-run the trigger.
        """
        if self._state != LifecycleState.SESSION_END_REMEDIATING:
            raise ValueError(
                f"Unexpected state for session_end_remediation_complete: {self._state}"
            )

        self._state = LifecycleState.RUNNING_SESSION_END
        return TransitionResult(
            state=self._state,
            effect=Effect.RUN_SESSION_END,
            message=f"Session_end remediation complete, re-running (attempt {ctx.retry_state.session_end_attempt}/{self.config.max_session_end_retries})",
        )

    def on_review_result(
        self,
        ctx: LifecycleContext,
        review_result: ReviewOutcome,
        new_log_offset: int,
        no_progress: bool = False,
    ) -> TransitionResult:
        """Handle external review result.

        Decides whether to:
        - Complete with success (if review passed)
        - Re-run review (if parse_error and retries remain)
        - Retry via agent prompt (if failed with issues but retries remain)
        - Fail (if no retries left or no progress)

        Args:
            ctx: Lifecycle context with retry state.
            review_result: The review outcome to process.
            new_log_offset: Updated log offset for next attempt.
            no_progress: If True, the agent made no progress since last attempt
                (same commit, no new validation evidence). Triggers fail-fast.
        """
        if self._state != LifecycleState.RUNNING_REVIEW:
            raise ValueError(f"Unexpected state for review_result: {self._state}")

        ctx.last_review_result = review_result

        logger.info(
            "Review result: outcome=%s attempt=%d state=%s",
            "passed" if review_result.passed else "failed",
            ctx.retry_state.review_attempt,
            self._state.name,
        )

        # Check for blocking issues (P0/P1 only). P2/P3 issues are acceptable
        # and can be tracked as beads issues later.
        # Issues with None priority are treated as non-blocking (default to P3).
        blocking_issues = [
            i
            for i in review_result.issues
            if i.priority is not None and i.priority <= 1
        ]

        # Parse errors are always blocking - we can't determine if there are issues
        has_parse_error = review_result.parse_error is not None

        if review_result.passed or (not blocking_issues and not has_parse_error):
            ctx.success = True
            self._state = LifecycleState.SUCCESS
            # Collect P2/P3 issues for tracking - include issues with priority > 1
            # or issues with None priority (treated as P3 for tracking purposes).
            # This ensures no review feedback is lost.
            low_pri_issues = [
                i for i in review_result.issues if i.priority is None or i.priority > 1
            ]
            ctx.low_priority_review_issues = low_pri_issues
            # Include P2/P3 count in message if any exist
            low_pri_count = len(low_pri_issues)
            if low_pri_count > 0:
                msg = f"Review passed ({low_pri_count} P2/P3 issues noted for later)"
            else:
                msg = "Review passed"
            return TransitionResult(
                state=self._state,
                effect=Effect.COMPLETE_SUCCESS,
                message=msg,
            )

        if review_result.parse_error and review_result.fatal_error:
            ctx.final_result = f"External review failed: {review_result.parse_error}"
            ctx.success = False
            self._state = LifecycleState.FAILED
            return TransitionResult(
                state=self._state,
                effect=Effect.COMPLETE_FAILURE,
                message="Review failed, unrecoverable error",
            )

        # Parse error (non-fatal): re-run review tool directly, not agent prompt.
        # This handles infrastructure issues with the reviewer (e.g., malformed JSON).
        if review_result.parse_error:
            can_retry = ctx.retry_state.review_attempt < self.config.max_review_retries
            if can_retry:
                ctx.retry_state.review_attempt += 1
                # Stay in RUNNING_REVIEW state - orchestrator re-runs external review
                return TransitionResult(
                    state=self._state,
                    effect=Effect.RUN_REVIEW,
                    message=f"Review parse error, re-running review (attempt {ctx.retry_state.review_attempt}/{self.config.max_review_retries})",
                )
            # No retries left
            ctx.final_result = f"External review failed: {review_result.parse_error}"
            ctx.success = False
            self._state = LifecycleState.FAILED
            return TransitionResult(
                state=self._state,
                effect=Effect.COMPLETE_FAILURE,
                message="Review failed, no retries left",
            )

        # Review failed with blocking issues - can we retry via agent prompt?
        can_retry = (
            ctx.retry_state.review_attempt < self.config.max_review_retries
            and not no_progress
        )

        if can_retry:
            # Prepare for retry - update offset and increment counter
            ctx.retry_state.log_offset = new_log_offset
            if ctx.last_gate_result:
                ctx.retry_state.previous_commit_hash = ctx.last_gate_result.commit_hash
            ctx.retry_state.review_attempt += 1
            self._state = LifecycleState.PROCESSING
            return TransitionResult(
                state=self._state,
                effect=Effect.SEND_REVIEW_RETRY,
                message=f"Review retry {ctx.retry_state.review_attempt}/{self.config.max_review_retries}",
            )

        # No retries left - fail with review error details
        if no_progress:
            ctx.final_result = "External review failed: No progress (commit unchanged, no working tree changes)"
            failure_message = "Review failed, no progress detected"
        else:
            # Format P0/P1 issues (these are blocking)
            critical_msgs = [
                f"{i.file}:{i.line_start}: {i.title}" for i in blocking_issues[:3]
            ]
            if critical_msgs:
                ctx.final_result = f"External review failed: {'; '.join(critical_msgs)}"
            else:
                ctx.final_result = "External review failed: Unknown reason"
            failure_message = "Review failed, no retries left"
        ctx.success = False
        self._state = LifecycleState.FAILED
        return TransitionResult(
            state=self._state,
            effect=Effect.COMPLETE_FAILURE,
            message=failure_message,
        )

    def on_timeout(
        self, ctx: LifecycleContext, timeout_minutes: int
    ) -> TransitionResult:
        """Handle session timeout.

        This can be called from any non-terminal state.
        """
        if self.is_terminal:
            return TransitionResult(
                state=self._state,
                effect=Effect.COMPLETE_FAILURE,
                message="Timeout after terminal state",
            )

        ctx.final_result = f"Timeout after {timeout_minutes} minutes"
        ctx.success = False
        self._state = LifecycleState.FAILED
        return TransitionResult(
            state=self._state,
            effect=Effect.COMPLETE_FAILURE,
            message=f"Timeout after {timeout_minutes} minutes",
        )

    def on_error(self, ctx: LifecycleContext, error: Exception) -> TransitionResult:
        """Handle unexpected error.

        This can be called from any non-terminal state.
        """
        if self.is_terminal:
            return TransitionResult(
                state=self._state,
                effect=Effect.COMPLETE_FAILURE,
                message=f"Error after terminal state: {error}",
            )

        ctx.final_result = str(error)
        ctx.success = False
        self._state = LifecycleState.FAILED
        return TransitionResult(
            state=self._state,
            effect=Effect.COMPLETE_FAILURE,
            message=f"Error: {error}",
        )

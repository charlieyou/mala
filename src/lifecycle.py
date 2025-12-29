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

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from src.validation.spec import ResolutionOutcome

if TYPE_CHECKING:
    from src.validation.spec import IssueResolution

# Resolution outcomes that skip codex review (no new code to review)
_SKIP_REVIEW_OUTCOMES = frozenset(
    {
        ResolutionOutcome.NO_CHANGE,
        ResolutionOutcome.OBSOLETE,
        ResolutionOutcome.ALREADY_COMPLETE,
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
    """Protocol for a single issue from a code review.

    Matches the fields lifecycle needs from review issues for building
    failure messages. Uses Protocol to allow structural subtyping with
    codex_review.ReviewIssue.
    """

    @property
    def file(self) -> str:
        """File path where the issue was found."""
        ...

    @property
    def line(self) -> int | None:
        """Line number, or None for file-level issues."""
        ...

    @property
    def severity(self) -> str:
        """Severity level: 'error', 'warning', or 'info'."""
        ...

    @property
    def message(self) -> str:
        """Description of the issue."""
        ...


@runtime_checkable
class ReviewOutcome(Protocol):
    """Protocol defining what lifecycle needs from a review result.

    Callers (orchestrator) pass infra CodexReviewResult objects that satisfy
    this protocol. Lifecycle only accesses these fields.
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
    def issues(self) -> list[ReviewIssue]:
        """List of issues found during review."""
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
    # Running Codex review (after gate passed)
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
    # Run Codex review
    RUN_REVIEW = auto()
    # Send gate retry follow-up prompt to SDK
    SEND_GATE_RETRY = auto()
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
    max_review_retries: int = 2
    codex_review_enabled: bool = True


@dataclass
class RetryState:
    """Tracks retry attempts for gate and review.

    This is mutable state that the lifecycle updates during transitions.
    The orchestrator can read it to format follow-up prompts.
    """

    gate_attempt: int = 1
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
    # Last review result for building follow-up prompts
    last_review_result: ReviewOutcome | None = None
    # Resolution for issue (no-op, obsolete, etc.)
    resolution: IssueResolution | None = None


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

        if gate_result.passed:
            # Gate passed - should we run review?
            # Skip review for resolutions with no new code (no_change, obsolete, already_complete)
            resolution_skips_review = (
                gate_result.resolution is not None
                and gate_result.resolution.outcome in _SKIP_REVIEW_OUTCOMES
            )
            if (
                self.config.codex_review_enabled
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
                    message=f"Gate passed, running review (attempt {ctx.retry_state.review_attempt}/{self.config.max_review_retries})",
                )
            else:
                # No review needed - success!
                ctx.success = True
                self._state = LifecycleState.SUCCESS
                return TransitionResult(
                    state=self._state,
                    effect=Effect.COMPLETE_SUCCESS,
                    message="Gate passed, no review required",
                )

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
        return TransitionResult(
            state=self._state,
            effect=Effect.COMPLETE_FAILURE,
            message="Gate failed, no retries left",
        )

    def on_review_result(
        self,
        ctx: LifecycleContext,
        review_result: ReviewOutcome,
        new_log_offset: int,
        no_progress: bool = False,
    ) -> TransitionResult:
        """Handle Codex review result.

        Decides whether to:
        - Complete with success (if review passed)
        - Retry review (if failed but retries remain and progress made)
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

        if review_result.passed:
            ctx.success = True
            self._state = LifecycleState.SUCCESS
            return TransitionResult(
                state=self._state,
                effect=Effect.COMPLETE_SUCCESS,
                message="Review passed",
            )

        # Review failed - can we retry?
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
            ctx.final_result = "Codex review failed: No progress (commit unchanged, no working tree changes)"
            failure_message = "Review failed, no progress detected"
        elif review_result.parse_error:
            ctx.final_result = f"Codex review failed: {review_result.parse_error}"
            failure_message = "Review failed, no retries left"
        else:
            error_msgs = [
                f"{i.file}:{i.line or 0}: {i.message}"
                for i in review_result.issues
                if i.severity == "error"
            ]
            ctx.final_result = f"Codex review failed: {'; '.join(error_msgs[:3])}"
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

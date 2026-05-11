"""Pure state machine for run validation trigger flow.

This module models the policy branch points in
``RunCoordinator._run_trigger_validation_loop`` without importing the
coordinator or performing I/O. Callers translate real command/review outcomes
into events, then execute the returned effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class RunValidationState(Enum):
    """States in run validation processing."""

    PROCESSING_COMMANDS = auto()
    AWAITING_REMEDIATION = auto()
    RUNNING_CODE_REVIEW = auto()
    REMEDIATING_REVIEW = auto()
    EMITTING_TERMINAL_EVENT = auto()


class RunValidationEvent(Enum):
    """Events accepted by the run validation state machine."""

    QUEUE_EMPTY = auto()
    COMMANDS_PASSED = auto()
    COMMAND_FAILED_ABORT = auto()
    COMMAND_FAILED_CONTINUE = auto()
    COMMAND_FAILED_REMEDIATE = auto()
    REMEDIATION_PASSED = auto()
    REMEDIATION_FAILED = auto()
    REMEDIATION_ABORTED = auto()
    CODE_REVIEW_DISABLED = auto()
    CODE_REVIEW_SKIPPED = auto()
    CODE_REVIEW_PASSED = auto()
    CODE_REVIEW_FAILED_ABORT = auto()
    CODE_REVIEW_FAILED_CONTINUE = auto()
    CODE_REVIEW_FAILED_REMEDIATE = auto()
    CODE_REVIEW_ERROR = auto()
    REVIEW_REMEDIATION_PASSED = auto()
    REVIEW_REMEDIATION_FAILED = auto()
    REVIEW_REMEDIATION_ABORTED = auto()
    VALIDATION_INTERRUPTED = auto()
    TERMINAL_EVENT_EMITTED = auto()


class RunValidationEffect(Enum):
    """Effects/actions the coordinator should perform."""

    RECORD_RUN_VALIDATION = auto()
    RECORD_FAILURE = auto()
    RUN_COMMAND_REMEDIATION = auto()
    RUN_CODE_REVIEW = auto()
    RUN_CODE_REVIEW_REMEDIATION = auto()
    EMIT_CODE_REVIEW_SKIPPED = auto()
    EMIT_CODE_REVIEW_PASSED = auto()
    EMIT_CODE_REVIEW_FAILED = auto()
    EMIT_CODE_REVIEW_ERROR = auto()
    COMPLETE_TRIGGER = auto()
    COMPLETE_PASSED = auto()
    COMPLETE_FAILED = auto()
    COMPLETE_ABORTED = auto()


class TerminalStatus(Enum):
    """Terminal result selected before terminal event emission."""

    PASSED = auto()
    FAILED = auto()
    ABORTED = auto()


@dataclass(frozen=True)
class FailureRecord:
    """The latest non-aborting failure that affects final run status."""

    ref: str
    trigger_type: str


@dataclass(frozen=True)
class RunValidationContext:
    """State-machine context carried between pure transitions."""

    last_failure: FailureRecord | None = None
    active_failure: FailureRecord | None = None
    terminal_status: TerminalStatus | None = None
    terminal_details: str | None = None


@dataclass(frozen=True)
class TransitionResult:
    """Result of one run validation transition."""

    state: RunValidationState
    context: RunValidationContext
    effects: tuple[RunValidationEffect, ...] = ()
    message: str | None = None


_TERMINAL_EFFECTS = {
    TerminalStatus.PASSED: RunValidationEffect.COMPLETE_PASSED,
    TerminalStatus.FAILED: RunValidationEffect.COMPLETE_FAILED,
    TerminalStatus.ABORTED: RunValidationEffect.COMPLETE_ABORTED,
}


def transition(
    state: RunValidationState,
    event: RunValidationEvent,
    context: RunValidationContext | None = None,
    failure: FailureRecord | None = None,
    details: str | None = None,
) -> TransitionResult:
    """Apply one pure run validation state transition.

    Raises:
        ValueError: if the event is not valid for the current state.
    """

    ctx = context or RunValidationContext()

    if event == RunValidationEvent.VALIDATION_INTERRUPTED:
        effects: tuple[RunValidationEffect, ...] = ()
        if state == RunValidationState.AWAITING_REMEDIATION:
            effects = (RunValidationEffect.RECORD_RUN_VALIDATION,)
        elif state == RunValidationState.REMEDIATING_REVIEW:
            effects = (
                RunValidationEffect.RECORD_RUN_VALIDATION,
                RunValidationEffect.EMIT_CODE_REVIEW_ERROR,
            )
        return TransitionResult(
            state=RunValidationState.EMITTING_TERMINAL_EVENT,
            context=RunValidationContext(
                last_failure=ctx.active_failure or ctx.last_failure,
                terminal_status=TerminalStatus.ABORTED,
                terminal_details=details or "Validation interrupted by SIGINT",
            ),
            effects=effects,
        )

    if state == RunValidationState.PROCESSING_COMMANDS:
        return _transition_processing_commands(event, ctx, failure, details)
    if state == RunValidationState.AWAITING_REMEDIATION:
        return _transition_awaiting_remediation(event, ctx, details)
    if state == RunValidationState.RUNNING_CODE_REVIEW:
        return _transition_running_code_review(event, ctx, failure, details)
    if state == RunValidationState.REMEDIATING_REVIEW:
        return _transition_remediating_review(event, ctx, details)
    if state == RunValidationState.EMITTING_TERMINAL_EVENT:
        return _transition_emitting_terminal_event(event, ctx)

    raise ValueError(f"Unknown run validation state: {state}")


def _transition_processing_commands(
    event: RunValidationEvent,
    ctx: RunValidationContext,
    failure: FailureRecord | None,
    details: str | None,
) -> TransitionResult:
    if event == RunValidationEvent.QUEUE_EMPTY:
        return _terminal_from_last_failure(ctx, effects=())
    if event == RunValidationEvent.COMMANDS_PASSED:
        return TransitionResult(
            state=RunValidationState.RUNNING_CODE_REVIEW,
            context=ctx,
            effects=(
                RunValidationEffect.RECORD_RUN_VALIDATION,
                RunValidationEffect.RUN_CODE_REVIEW,
            ),
        )
    if event == RunValidationEvent.COMMAND_FAILED_CONTINUE:
        failure = _require_failure(event, failure)
        return TransitionResult(
            state=RunValidationState.RUNNING_CODE_REVIEW,
            context=RunValidationContext(last_failure=failure),
            effects=(
                RunValidationEffect.RECORD_FAILURE,
                RunValidationEffect.RECORD_RUN_VALIDATION,
                RunValidationEffect.RUN_CODE_REVIEW,
            ),
        )
    if event == RunValidationEvent.COMMAND_FAILED_REMEDIATE:
        failure = _require_failure(event, failure)
        return TransitionResult(
            state=RunValidationState.AWAITING_REMEDIATION,
            context=RunValidationContext(
                last_failure=ctx.last_failure,
                active_failure=failure,
            ),
            effects=(RunValidationEffect.RUN_COMMAND_REMEDIATION,),
        )
    if event == RunValidationEvent.COMMAND_FAILED_ABORT:
        failure = _require_failure(event, failure)
        return TransitionResult(
            state=RunValidationState.EMITTING_TERMINAL_EVENT,
            context=RunValidationContext(
                last_failure=failure,
                terminal_status=TerminalStatus.ABORTED,
                terminal_details=_failure_details(failure, details),
            ),
            effects=(
                RunValidationEffect.RECORD_FAILURE,
                RunValidationEffect.RECORD_RUN_VALIDATION,
            ),
        )

    raise ValueError(f"Unexpected event for PROCESSING_COMMANDS: {event}")


def _transition_awaiting_remediation(
    event: RunValidationEvent,
    ctx: RunValidationContext,
    details: str | None,
) -> TransitionResult:
    if event == RunValidationEvent.REMEDIATION_PASSED:
        return TransitionResult(
            state=RunValidationState.PROCESSING_COMMANDS,
            context=RunValidationContext(last_failure=ctx.last_failure),
        )
    if event == RunValidationEvent.REMEDIATION_FAILED:
        failure = _require_active_failure(event, ctx)
        return TransitionResult(
            state=RunValidationState.EMITTING_TERMINAL_EVENT,
            context=RunValidationContext(
                last_failure=failure,
                terminal_status=TerminalStatus.ABORTED,
                terminal_details=_failure_details(failure, details),
            ),
            effects=(
                RunValidationEffect.RECORD_FAILURE,
                RunValidationEffect.RECORD_RUN_VALIDATION,
            ),
        )
    if event == RunValidationEvent.REMEDIATION_ABORTED:
        return TransitionResult(
            state=RunValidationState.EMITTING_TERMINAL_EVENT,
            context=RunValidationContext(
                terminal_status=TerminalStatus.ABORTED,
                terminal_details=details or "Command remediation aborted",
            ),
            effects=(RunValidationEffect.RECORD_RUN_VALIDATION,),
        )

    raise ValueError(f"Unexpected event for AWAITING_REMEDIATION: {event}")


def _transition_running_code_review(
    event: RunValidationEvent,
    ctx: RunValidationContext,
    failure: FailureRecord | None,
    details: str | None,
) -> TransitionResult:
    if event == RunValidationEvent.CODE_REVIEW_DISABLED:
        return TransitionResult(
            state=RunValidationState.PROCESSING_COMMANDS,
            context=RunValidationContext(last_failure=ctx.last_failure),
            effects=(RunValidationEffect.COMPLETE_TRIGGER,),
        )
    if event == RunValidationEvent.CODE_REVIEW_SKIPPED:
        return TransitionResult(
            state=RunValidationState.PROCESSING_COMMANDS,
            context=RunValidationContext(last_failure=ctx.last_failure),
            effects=(RunValidationEffect.EMIT_CODE_REVIEW_SKIPPED,),
        )
    if event == RunValidationEvent.CODE_REVIEW_PASSED:
        return TransitionResult(
            state=RunValidationState.PROCESSING_COMMANDS,
            context=RunValidationContext(last_failure=ctx.last_failure),
            effects=(RunValidationEffect.EMIT_CODE_REVIEW_PASSED,),
        )
    if event == RunValidationEvent.CODE_REVIEW_FAILED_ABORT:
        failure = _require_failure(event, failure)
        return TransitionResult(
            state=RunValidationState.EMITTING_TERMINAL_EVENT,
            context=RunValidationContext(
                last_failure=failure,
                terminal_status=TerminalStatus.ABORTED,
                terminal_details=_failure_details(failure, details),
            ),
            effects=(
                RunValidationEffect.EMIT_CODE_REVIEW_FAILED,
                RunValidationEffect.RECORD_FAILURE,
            ),
        )
    if event == RunValidationEvent.CODE_REVIEW_FAILED_CONTINUE:
        failure = _require_failure(event, failure)
        return TransitionResult(
            state=RunValidationState.PROCESSING_COMMANDS,
            context=RunValidationContext(last_failure=failure),
            effects=(
                RunValidationEffect.EMIT_CODE_REVIEW_FAILED,
                RunValidationEffect.RECORD_FAILURE,
                RunValidationEffect.COMPLETE_TRIGGER,
            ),
        )
    if event == RunValidationEvent.CODE_REVIEW_FAILED_REMEDIATE:
        failure = _require_failure(event, failure)
        return TransitionResult(
            state=RunValidationState.REMEDIATING_REVIEW,
            context=RunValidationContext(
                last_failure=ctx.last_failure,
                active_failure=failure,
            ),
            effects=(RunValidationEffect.RUN_CODE_REVIEW_REMEDIATION,),
        )
    if event == RunValidationEvent.CODE_REVIEW_ERROR:
        return TransitionResult(
            state=RunValidationState.EMITTING_TERMINAL_EVENT,
            context=RunValidationContext(
                terminal_status=TerminalStatus.ABORTED,
                terminal_details=details or "Code review execution failed",
            ),
            effects=(RunValidationEffect.EMIT_CODE_REVIEW_ERROR,),
        )

    raise ValueError(f"Unexpected event for RUNNING_CODE_REVIEW: {event}")


def _transition_remediating_review(
    event: RunValidationEvent,
    ctx: RunValidationContext,
    details: str | None,
) -> TransitionResult:
    if event == RunValidationEvent.REVIEW_REMEDIATION_PASSED:
        return TransitionResult(
            state=RunValidationState.PROCESSING_COMMANDS,
            context=RunValidationContext(last_failure=ctx.last_failure),
            effects=(RunValidationEffect.EMIT_CODE_REVIEW_PASSED,),
        )
    if event == RunValidationEvent.REVIEW_REMEDIATION_FAILED:
        failure = _require_active_failure(event, ctx)
        return TransitionResult(
            state=RunValidationState.PROCESSING_COMMANDS,
            context=RunValidationContext(last_failure=failure),
            effects=(
                RunValidationEffect.RECORD_FAILURE,
                RunValidationEffect.EMIT_CODE_REVIEW_FAILED,
                RunValidationEffect.COMPLETE_TRIGGER,
            ),
        )
    if event == RunValidationEvent.REVIEW_REMEDIATION_ABORTED:
        return TransitionResult(
            state=RunValidationState.EMITTING_TERMINAL_EVENT,
            context=RunValidationContext(
                terminal_status=TerminalStatus.ABORTED,
                terminal_details=details or "Code review remediation aborted",
            ),
            effects=(
                RunValidationEffect.RECORD_RUN_VALIDATION,
                RunValidationEffect.EMIT_CODE_REVIEW_ERROR,
            ),
        )

    raise ValueError(f"Unexpected event for REMEDIATING_REVIEW: {event}")


def _transition_emitting_terminal_event(
    event: RunValidationEvent,
    ctx: RunValidationContext,
) -> TransitionResult:
    if event != RunValidationEvent.TERMINAL_EVENT_EMITTED:
        raise ValueError(f"Unexpected event for EMITTING_TERMINAL_EVENT: {event}")
    if ctx.terminal_status is None:
        raise ValueError("Cannot emit terminal event without terminal status")

    return TransitionResult(
        state=RunValidationState.EMITTING_TERMINAL_EVENT,
        context=ctx,
        effects=(_TERMINAL_EFFECTS[ctx.terminal_status],),
    )


def _terminal_from_last_failure(
    ctx: RunValidationContext,
    *,
    effects: tuple[RunValidationEffect, ...],
) -> TransitionResult:
    if ctx.last_failure is None:
        terminal_status = TerminalStatus.PASSED
        terminal_details = None
    else:
        terminal_status = TerminalStatus.FAILED
        terminal_details = _failure_details(ctx.last_failure)

    return TransitionResult(
        state=RunValidationState.EMITTING_TERMINAL_EVENT,
        context=RunValidationContext(
            last_failure=ctx.last_failure,
            terminal_status=terminal_status,
            terminal_details=terminal_details,
        ),
        effects=effects,
    )


def _failure_details(failure: FailureRecord, details: str | None = None) -> str:
    if details is not None:
        return details
    return f"Command '{failure.ref}' failed in trigger {failure.trigger_type}"


def _require_failure(
    event: RunValidationEvent,
    failure: FailureRecord | None,
) -> FailureRecord:
    if failure is None:
        raise ValueError(f"Event {event.name} requires failure details")
    return failure


def _require_active_failure(
    event: RunValidationEvent,
    ctx: RunValidationContext,
) -> FailureRecord:
    if ctx.active_failure is None:
        raise ValueError(f"Event {event.name} requires active failure details")
    return ctx.active_failure

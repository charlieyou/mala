"""State/event matrix tests for run validation state machine."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from itertools import product

import pytest

from src.pipeline.run_validation_state import (
    FailureRecord,
    RunValidationContext,
    RunValidationEffect,
    RunValidationEvent,
    RunValidationState,
    TerminalStatus,
    TransitionResult,
    transition,
)

_COMMAND_FAILURE = FailureRecord(
    ref="command_failure",
    trigger_type="run_validation",
)
_CODE_REVIEW_FAILURE = FailureRecord(
    ref="code_review_findings",
    trigger_type="run_validation",
)
_EXISTING_FAILURE = FailureRecord(ref="lint", trigger_type="run_end")

_CLEAN_CONTEXT = RunValidationContext()
_FAILED_CONTEXT = RunValidationContext(last_failure=_EXISTING_FAILURE)
_TERMINAL_PASSED_CONTEXT = RunValidationContext(terminal_status=TerminalStatus.PASSED)
_TERMINAL_FAILED_CONTEXT = RunValidationContext(terminal_status=TerminalStatus.FAILED)
_TERMINAL_ABORTED_CONTEXT = RunValidationContext(terminal_status=TerminalStatus.ABORTED)

_EXPECTED: dict[
    tuple[RunValidationState, RunValidationEvent],
    tuple[RunValidationState, tuple[RunValidationEffect, ...], RunValidationContext],
] = {
    (
        RunValidationState.PROCESSING_COMMANDS,
        RunValidationEvent.COMMANDS_PASSED,
    ): (
        RunValidationState.RUNNING_CODE_REVIEW,
        (
            RunValidationEffect.RECORD_RUN_VALIDATION,
            RunValidationEffect.RUN_CODE_REVIEW,
        ),
        _CLEAN_CONTEXT,
    ),
    (
        RunValidationState.PROCESSING_COMMANDS,
        RunValidationEvent.COMMAND_FAILED_ABORT,
    ): (
        RunValidationState.EMITTING_TERMINAL_EVENT,
        (
            RunValidationEffect.RECORD_FAILURE,
            RunValidationEffect.RECORD_RUN_VALIDATION,
        ),
        RunValidationContext(
            last_failure=_COMMAND_FAILURE,
            terminal_status=TerminalStatus.ABORTED,
            terminal_details="Command failed with abort failure mode",
        ),
    ),
    (
        RunValidationState.PROCESSING_COMMANDS,
        RunValidationEvent.COMMAND_FAILED_CONTINUE,
    ): (
        RunValidationState.RUNNING_CODE_REVIEW,
        (
            RunValidationEffect.RECORD_FAILURE,
            RunValidationEffect.RECORD_RUN_VALIDATION,
            RunValidationEffect.RUN_CODE_REVIEW,
        ),
        RunValidationContext(last_failure=_COMMAND_FAILURE),
    ),
    (
        RunValidationState.PROCESSING_COMMANDS,
        RunValidationEvent.COMMAND_FAILED_REMEDIATE,
    ): (
        RunValidationState.AWAITING_REMEDIATION,
        (RunValidationEffect.RUN_COMMAND_REMEDIATION,),
        _CLEAN_CONTEXT,
    ),
    (
        RunValidationState.AWAITING_REMEDIATION,
        RunValidationEvent.REMEDIATION_PASSED,
    ): (RunValidationState.PROCESSING_COMMANDS, (), _CLEAN_CONTEXT),
    (
        RunValidationState.AWAITING_REMEDIATION,
        RunValidationEvent.REMEDIATION_FAILED,
    ): (
        RunValidationState.EMITTING_TERMINAL_EVENT,
        (
            RunValidationEffect.RECORD_FAILURE,
            RunValidationEffect.RECORD_RUN_VALIDATION,
        ),
        RunValidationContext(
            last_failure=_COMMAND_FAILURE,
            terminal_status=TerminalStatus.ABORTED,
            terminal_details="Command remediation failed",
        ),
    ),
    (
        RunValidationState.AWAITING_REMEDIATION,
        RunValidationEvent.REMEDIATION_ABORTED,
    ): (
        RunValidationState.EMITTING_TERMINAL_EVENT,
        (RunValidationEffect.RECORD_RUN_VALIDATION,),
        RunValidationContext(
            terminal_status=TerminalStatus.ABORTED,
            terminal_details="Command remediation aborted",
        ),
    ),
    (
        RunValidationState.RUNNING_CODE_REVIEW,
        RunValidationEvent.CODE_REVIEW_DISABLED,
    ): (
        RunValidationState.EMITTING_TERMINAL_EVENT,
        (),
        RunValidationContext(terminal_status=TerminalStatus.PASSED),
    ),
    (
        RunValidationState.RUNNING_CODE_REVIEW,
        RunValidationEvent.CODE_REVIEW_SKIPPED,
    ): (
        RunValidationState.EMITTING_TERMINAL_EVENT,
        (RunValidationEffect.EMIT_CODE_REVIEW_SKIPPED,),
        RunValidationContext(terminal_status=TerminalStatus.PASSED),
    ),
    (
        RunValidationState.RUNNING_CODE_REVIEW,
        RunValidationEvent.CODE_REVIEW_PASSED,
    ): (
        RunValidationState.EMITTING_TERMINAL_EVENT,
        (RunValidationEffect.EMIT_CODE_REVIEW_PASSED,),
        RunValidationContext(terminal_status=TerminalStatus.PASSED),
    ),
    (
        RunValidationState.RUNNING_CODE_REVIEW,
        RunValidationEvent.CODE_REVIEW_FAILED_ABORT,
    ): (
        RunValidationState.EMITTING_TERMINAL_EVENT,
        (
            RunValidationEffect.RECORD_FAILURE,
            RunValidationEffect.EMIT_CODE_REVIEW_FAILED,
        ),
        RunValidationContext(
            last_failure=_CODE_REVIEW_FAILURE,
            terminal_status=TerminalStatus.ABORTED,
            terminal_details="Code review findings exceeded threshold",
        ),
    ),
    (
        RunValidationState.RUNNING_CODE_REVIEW,
        RunValidationEvent.CODE_REVIEW_FAILED_CONTINUE,
    ): (
        RunValidationState.EMITTING_TERMINAL_EVENT,
        (
            RunValidationEffect.RECORD_FAILURE,
            RunValidationEffect.EMIT_CODE_REVIEW_FAILED,
        ),
        RunValidationContext(
            last_failure=_CODE_REVIEW_FAILURE,
            terminal_status=TerminalStatus.FAILED,
            terminal_details="Code review findings exceeded threshold",
        ),
    ),
    (
        RunValidationState.RUNNING_CODE_REVIEW,
        RunValidationEvent.CODE_REVIEW_FAILED_REMEDIATE,
    ): (
        RunValidationState.REMEDIATING_REVIEW,
        (RunValidationEffect.RUN_CODE_REVIEW_REMEDIATION,),
        _CLEAN_CONTEXT,
    ),
    (
        RunValidationState.RUNNING_CODE_REVIEW,
        RunValidationEvent.CODE_REVIEW_ERROR,
    ): (
        RunValidationState.EMITTING_TERMINAL_EVENT,
        (RunValidationEffect.EMIT_CODE_REVIEW_ERROR,),
        RunValidationContext(
            terminal_status=TerminalStatus.ABORTED,
            terminal_details="Code review execution failed",
        ),
    ),
    (
        RunValidationState.REMEDIATING_REVIEW,
        RunValidationEvent.REVIEW_REMEDIATION_PASSED,
    ): (
        RunValidationState.EMITTING_TERMINAL_EVENT,
        (RunValidationEffect.EMIT_CODE_REVIEW_PASSED,),
        RunValidationContext(terminal_status=TerminalStatus.PASSED),
    ),
    (
        RunValidationState.REMEDIATING_REVIEW,
        RunValidationEvent.REVIEW_REMEDIATION_FAILED,
    ): (
        RunValidationState.EMITTING_TERMINAL_EVENT,
        (
            RunValidationEffect.RECORD_FAILURE,
            RunValidationEffect.EMIT_CODE_REVIEW_FAILED,
        ),
        RunValidationContext(
            last_failure=_CODE_REVIEW_FAILURE,
            terminal_status=TerminalStatus.FAILED,
            terminal_details="Code review remediation failed",
        ),
    ),
    (
        RunValidationState.REMEDIATING_REVIEW,
        RunValidationEvent.REVIEW_REMEDIATION_ABORTED,
    ): (
        RunValidationState.EMITTING_TERMINAL_EVENT,
        (RunValidationEffect.EMIT_CODE_REVIEW_ERROR,),
        RunValidationContext(
            terminal_status=TerminalStatus.ABORTED,
            terminal_details="Code review remediation aborted",
        ),
    ),
    (
        RunValidationState.EMITTING_TERMINAL_EVENT,
        RunValidationEvent.TERMINAL_EVENT_EMITTED,
    ): (
        RunValidationState.EMITTING_TERMINAL_EVENT,
        (RunValidationEffect.COMPLETE_PASSED,),
        _TERMINAL_PASSED_CONTEXT,
    ),
}

_CONTEXT_OVERRIDES = {
    (
        RunValidationState.RUNNING_CODE_REVIEW,
        RunValidationEvent.CODE_REVIEW_DISABLED,
    ): _CLEAN_CONTEXT,
    (
        RunValidationState.RUNNING_CODE_REVIEW,
        RunValidationEvent.CODE_REVIEW_SKIPPED,
    ): _CLEAN_CONTEXT,
    (
        RunValidationState.RUNNING_CODE_REVIEW,
        RunValidationEvent.CODE_REVIEW_PASSED,
    ): _CLEAN_CONTEXT,
    (
        RunValidationState.REMEDIATING_REVIEW,
        RunValidationEvent.REVIEW_REMEDIATION_PASSED,
    ): _CLEAN_CONTEXT,
    (
        RunValidationState.EMITTING_TERMINAL_EVENT,
        RunValidationEvent.TERMINAL_EVENT_EMITTED,
    ): _TERMINAL_PASSED_CONTEXT,
}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("state", "event"),
    list(product(RunValidationState, RunValidationEvent)),
)
def test_state_event_matrix_is_explicit(
    state: RunValidationState,
    event: RunValidationEvent,
) -> None:
    """Every state/event pair either has an expected result or rejects."""

    context = _CONTEXT_OVERRIDES.get((state, event), _CLEAN_CONTEXT)
    expected = _EXPECTED.get((state, event))

    if expected is None:
        with pytest.raises(ValueError):
            transition(state, event, context)
        return

    next_state, effects, next_context = expected
    result = transition(state, event, context)

    assert result == TransitionResult(
        state=next_state,
        context=next_context,
        effects=effects,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("event", "effect", "context"),
    [
        (
            RunValidationEvent.CODE_REVIEW_DISABLED,
            RunValidationEffect.COMPLETE_FAILED,
            _FAILED_CONTEXT,
        ),
        (
            RunValidationEvent.CODE_REVIEW_SKIPPED,
            RunValidationEffect.COMPLETE_FAILED,
            _FAILED_CONTEXT,
        ),
        (
            RunValidationEvent.CODE_REVIEW_PASSED,
            RunValidationEffect.COMPLETE_FAILED,
            _FAILED_CONTEXT,
        ),
        (
            RunValidationEvent.REVIEW_REMEDIATION_PASSED,
            RunValidationEffect.COMPLETE_FAILED,
            _FAILED_CONTEXT,
        ),
        (
            RunValidationEvent.TERMINAL_EVENT_EMITTED,
            RunValidationEffect.COMPLETE_FAILED,
            _TERMINAL_FAILED_CONTEXT,
        ),
        (
            RunValidationEvent.TERMINAL_EVENT_EMITTED,
            RunValidationEffect.COMPLETE_ABORTED,
            _TERMINAL_ABORTED_CONTEXT,
        ),
    ],
)
def test_context_controls_deferred_terminal_status(
    event: RunValidationEvent,
    effect: RunValidationEffect,
    context: RunValidationContext,
) -> None:
    if event == RunValidationEvent.TERMINAL_EVENT_EMITTED:
        result = transition(RunValidationState.EMITTING_TERMINAL_EVENT, event, context)
    elif event == RunValidationEvent.REVIEW_REMEDIATION_PASSED:
        result = transition(RunValidationState.REMEDIATING_REVIEW, event, context)
        result = transition(
            result.state,
            RunValidationEvent.TERMINAL_EVENT_EMITTED,
            result.context,
        )
    else:
        result = transition(RunValidationState.RUNNING_CODE_REVIEW, event, context)
        result = transition(
            result.state,
            RunValidationEvent.TERMINAL_EVENT_EMITTED,
            result.context,
        )

    assert result.effects == (effect,)


@pytest.mark.unit
def test_terminal_effects_are_only_emitted_from_terminal_state() -> None:
    terminal_effects = {
        RunValidationEffect.COMPLETE_PASSED,
        RunValidationEffect.COMPLETE_FAILED,
        RunValidationEffect.COMPLETE_ABORTED,
    }

    for state, event in product(RunValidationState, RunValidationEvent):
        context = _CONTEXT_OVERRIDES.get((state, event), _CLEAN_CONTEXT)
        try:
            result = transition(state, event, context)
        except ValueError:
            continue

        if result.effects and terminal_effects.intersection(result.effects):
            assert state == RunValidationState.EMITTING_TERMINAL_EVENT


@pytest.mark.unit
def test_transition_context_is_frozen() -> None:
    result = transition(
        RunValidationState.PROCESSING_COMMANDS,
        RunValidationEvent.COMMANDS_PASSED,
    )

    with pytest.raises(FrozenInstanceError):
        setattr(result.context, "terminal_status", TerminalStatus.PASSED)

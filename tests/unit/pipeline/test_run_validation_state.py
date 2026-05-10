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
    ref="lint",
    trigger_type="run_end",
)
_CODE_REVIEW_FAILURE = FailureRecord(
    ref="code_review_findings",
    trigger_type="run_end",
)
_EXISTING_FAILURE = FailureRecord(ref="unit", trigger_type="run_start")

_CLEAN_CONTEXT = RunValidationContext()
_FAILED_CONTEXT = RunValidationContext(last_failure=_EXISTING_FAILURE)
_COMMAND_REMEDIATION_CONTEXT = RunValidationContext(active_failure=_COMMAND_FAILURE)
_REVIEW_REMEDIATION_CONTEXT = RunValidationContext(active_failure=_CODE_REVIEW_FAILURE)
_TERMINAL_PASSED_CONTEXT = RunValidationContext(terminal_status=TerminalStatus.PASSED)
_TERMINAL_FAILED_CONTEXT = RunValidationContext(terminal_status=TerminalStatus.FAILED)
_TERMINAL_ABORTED_CONTEXT = RunValidationContext(terminal_status=TerminalStatus.ABORTED)

_EXPECTED: dict[
    tuple[RunValidationState, RunValidationEvent],
    tuple[RunValidationState, tuple[RunValidationEffect, ...], RunValidationContext],
] = {
    (
        RunValidationState.PROCESSING_COMMANDS,
        RunValidationEvent.QUEUE_EMPTY,
    ): (
        RunValidationState.EMITTING_TERMINAL_EVENT,
        (),
        RunValidationContext(terminal_status=TerminalStatus.PASSED),
    ),
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
            terminal_details="Command 'lint' failed in trigger run_end",
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
        _COMMAND_REMEDIATION_CONTEXT,
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
            terminal_details="Command 'lint' failed in trigger run_end",
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
        RunValidationState.PROCESSING_COMMANDS,
        (RunValidationEffect.COMPLETE_TRIGGER,),
        _CLEAN_CONTEXT,
    ),
    (
        RunValidationState.RUNNING_CODE_REVIEW,
        RunValidationEvent.CODE_REVIEW_SKIPPED,
    ): (
        RunValidationState.PROCESSING_COMMANDS,
        (RunValidationEffect.EMIT_CODE_REVIEW_SKIPPED,),
        _CLEAN_CONTEXT,
    ),
    (
        RunValidationState.RUNNING_CODE_REVIEW,
        RunValidationEvent.CODE_REVIEW_PASSED,
    ): (
        RunValidationState.PROCESSING_COMMANDS,
        (RunValidationEffect.EMIT_CODE_REVIEW_PASSED,),
        _CLEAN_CONTEXT,
    ),
    (
        RunValidationState.RUNNING_CODE_REVIEW,
        RunValidationEvent.CODE_REVIEW_FAILED_ABORT,
    ): (
        RunValidationState.EMITTING_TERMINAL_EVENT,
        (
            RunValidationEffect.EMIT_CODE_REVIEW_FAILED,
            RunValidationEffect.RECORD_FAILURE,
        ),
        RunValidationContext(
            last_failure=_CODE_REVIEW_FAILURE,
            terminal_status=TerminalStatus.ABORTED,
            terminal_details="Command 'code_review_findings' failed in trigger run_end",
        ),
    ),
    (
        RunValidationState.RUNNING_CODE_REVIEW,
        RunValidationEvent.CODE_REVIEW_FAILED_CONTINUE,
    ): (
        RunValidationState.PROCESSING_COMMANDS,
        (
            RunValidationEffect.EMIT_CODE_REVIEW_FAILED,
            RunValidationEffect.RECORD_FAILURE,
            RunValidationEffect.COMPLETE_TRIGGER,
        ),
        RunValidationContext(last_failure=_CODE_REVIEW_FAILURE),
    ),
    (
        RunValidationState.RUNNING_CODE_REVIEW,
        RunValidationEvent.CODE_REVIEW_FAILED_REMEDIATE,
    ): (
        RunValidationState.REMEDIATING_REVIEW,
        (RunValidationEffect.RUN_CODE_REVIEW_REMEDIATION,),
        _REVIEW_REMEDIATION_CONTEXT,
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
        RunValidationState.PROCESSING_COMMANDS,
        (RunValidationEffect.EMIT_CODE_REVIEW_PASSED,),
        _CLEAN_CONTEXT,
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
            terminal_details="Command 'code_review_findings' failed in trigger run_end",
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

for _state in RunValidationState:
    _EXPECTED[(_state, RunValidationEvent.VALIDATION_INTERRUPTED)] = (
        RunValidationState.EMITTING_TERMINAL_EVENT,
        (),
        RunValidationContext(
            terminal_status=TerminalStatus.ABORTED,
            terminal_details="Validation interrupted by SIGINT",
        ),
    )

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
        RunValidationState.AWAITING_REMEDIATION,
        RunValidationEvent.REMEDIATION_PASSED,
    ): _COMMAND_REMEDIATION_CONTEXT,
    (
        RunValidationState.REMEDIATING_REVIEW,
        RunValidationEvent.REVIEW_REMEDIATION_PASSED,
    ): _CLEAN_CONTEXT,
    (
        RunValidationState.AWAITING_REMEDIATION,
        RunValidationEvent.REMEDIATION_FAILED,
    ): _COMMAND_REMEDIATION_CONTEXT,
    (
        RunValidationState.REMEDIATING_REVIEW,
        RunValidationEvent.REVIEW_REMEDIATION_FAILED,
    ): _REVIEW_REMEDIATION_CONTEXT,
    (
        RunValidationState.EMITTING_TERMINAL_EVENT,
        RunValidationEvent.TERMINAL_EVENT_EMITTED,
    ): _TERMINAL_PASSED_CONTEXT,
}

_FAILURE_OVERRIDES = {
    (
        RunValidationState.PROCESSING_COMMANDS,
        RunValidationEvent.COMMAND_FAILED_ABORT,
    ): _COMMAND_FAILURE,
    (
        RunValidationState.PROCESSING_COMMANDS,
        RunValidationEvent.COMMAND_FAILED_CONTINUE,
    ): _COMMAND_FAILURE,
    (
        RunValidationState.PROCESSING_COMMANDS,
        RunValidationEvent.COMMAND_FAILED_REMEDIATE,
    ): _COMMAND_FAILURE,
    (
        RunValidationState.RUNNING_CODE_REVIEW,
        RunValidationEvent.CODE_REVIEW_FAILED_ABORT,
    ): _CODE_REVIEW_FAILURE,
    (
        RunValidationState.RUNNING_CODE_REVIEW,
        RunValidationEvent.CODE_REVIEW_FAILED_CONTINUE,
    ): _CODE_REVIEW_FAILURE,
    (
        RunValidationState.RUNNING_CODE_REVIEW,
        RunValidationEvent.CODE_REVIEW_FAILED_REMEDIATE,
    ): _CODE_REVIEW_FAILURE,
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
    failure = _FAILURE_OVERRIDES.get((state, event))
    expected = _EXPECTED.get((state, event))

    if expected is None:
        with pytest.raises(ValueError):
            transition(state, event, context)
        return

    next_state, effects, next_context = expected
    result = transition(state, event, context, failure=failure)

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
            RunValidationEvent.QUEUE_EMPTY,
            result.context,
        )
        result = transition(
            result.state,
            RunValidationEvent.TERMINAL_EVENT_EMITTED,
            result.context,
        )
    else:
        result = transition(RunValidationState.RUNNING_CODE_REVIEW, event, context)
        result = transition(
            result.state,
            RunValidationEvent.QUEUE_EMPTY,
            result.context,
        )
        result = transition(
            result.state,
            RunValidationEvent.TERMINAL_EVENT_EMITTED,
            result.context,
        )

    assert result.effects == (effect,)


@pytest.mark.unit
def test_failure_events_require_real_failure_details() -> None:
    with pytest.raises(ValueError, match="requires failure details"):
        transition(
            RunValidationState.PROCESSING_COMMANDS,
            RunValidationEvent.COMMAND_FAILED_CONTINUE,
        )


@pytest.mark.unit
def test_command_failure_details_are_preserved_for_terminal_result() -> None:
    result = transition(
        RunValidationState.PROCESSING_COMMANDS,
        RunValidationEvent.COMMAND_FAILED_ABORT,
        failure=FailureRecord(ref="mypy", trigger_type="run_start"),
    )

    assert result.context.last_failure == FailureRecord(
        ref="mypy",
        trigger_type="run_start",
    )
    assert result.context.terminal_details == (
        "Command 'mypy' failed in trigger run_start"
    )


@pytest.mark.unit
def test_code_review_continue_returns_to_command_processing() -> None:
    result = transition(
        RunValidationState.RUNNING_CODE_REVIEW,
        RunValidationEvent.CODE_REVIEW_FAILED_CONTINUE,
        failure=FailureRecord(ref="code_review_findings", trigger_type="run_end"),
    )

    assert result.state == RunValidationState.PROCESSING_COMMANDS
    assert result.context == RunValidationContext(
        last_failure=FailureRecord(
            ref="code_review_findings",
            trigger_type="run_end",
        )
    )
    assert result.effects == (
        RunValidationEffect.EMIT_CODE_REVIEW_FAILED,
        RunValidationEffect.RECORD_FAILURE,
        RunValidationEffect.COMPLETE_TRIGGER,
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "event",
    [
        RunValidationEvent.CODE_REVIEW_DISABLED,
        RunValidationEvent.CODE_REVIEW_SKIPPED,
        RunValidationEvent.CODE_REVIEW_PASSED,
    ],
)
def test_non_aborting_review_completion_waits_for_queue_empty(
    event: RunValidationEvent,
) -> None:
    result = transition(
        RunValidationState.RUNNING_CODE_REVIEW,
        event,
        RunValidationContext(last_failure=_COMMAND_FAILURE),
    )

    assert result.state == RunValidationState.PROCESSING_COMMANDS
    assert result.context == RunValidationContext(last_failure=_COMMAND_FAILURE)

    terminal = transition(
        result.state,
        RunValidationEvent.QUEUE_EMPTY,
        result.context,
    )
    assert terminal.state == RunValidationState.EMITTING_TERMINAL_EVENT
    assert terminal.context.terminal_status == TerminalStatus.FAILED


@pytest.mark.unit
def test_custom_terminal_details_are_preserved() -> None:
    result = transition(
        RunValidationState.RUNNING_CODE_REVIEW,
        RunValidationEvent.CODE_REVIEW_FAILED_ABORT,
        failure=FailureRecord(ref="code_review_findings", trigger_type="run_end"),
        details="Code review findings exceed threshold (P1) in trigger run_end",
    )

    assert result.context.terminal_details == (
        "Code review findings exceed threshold (P1) in trigger run_end"
    )


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

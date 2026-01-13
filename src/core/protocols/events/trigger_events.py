"""Trigger validation and code review event protocols.

This module defines protocols for trigger validation, trigger code review,
and session end lifecycle events.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TriggerEvents(Protocol):
    """Protocol for trigger validation and code review events.

    These events track trigger-based validation, code review, and session end.
    """

    # -------------------------------------------------------------------------
    # Trigger validation lifecycle
    # -------------------------------------------------------------------------

    def on_trigger_validation_queued(
        self, trigger_type: str, trigger_context: str
    ) -> None:
        """Called when a trigger is queued for validation.

        Args:
            trigger_type: Type of trigger (e.g., "epic_completion", "session_end").
            trigger_context: Context info for the trigger (e.g., issue_id).
        """
        ...

    def on_trigger_validation_started(
        self, trigger_type: str, commands: list[str]
    ) -> None:
        """Called when trigger validation begins execution.

        Args:
            trigger_type: Type of trigger being validated.
            commands: List of command refs to be executed.
        """
        ...

    def on_trigger_command_started(
        self, trigger_type: str, command_ref: str, index: int, total_commands: int
    ) -> None:
        """Called when a trigger command starts execution.

        Args:
            trigger_type: Type of trigger.
            command_ref: Reference name of the command.
            index: 0-based index of the command in the trigger's command list.
            total_commands: Total number of commands in this validation.
        """
        ...

    def on_trigger_command_completed(
        self,
        trigger_type: str,
        command_ref: str,
        index: int,
        total_commands: int,
        passed: bool,
        duration_seconds: float,
    ) -> None:
        """Called when a trigger command completes.

        Args:
            trigger_type: Type of trigger.
            command_ref: Reference name of the command.
            index: 0-based index of the command.
            total_commands: Total number of commands in this validation.
            passed: Whether the command passed.
            duration_seconds: Execution duration.
        """
        ...

    def on_trigger_validation_passed(
        self, trigger_type: str, duration_seconds: float
    ) -> None:
        """Called when all commands for a trigger pass.

        Args:
            trigger_type: Type of trigger that passed.
            duration_seconds: Total duration for the trigger's validation.
        """
        ...

    def on_trigger_validation_failed(
        self, trigger_type: str, failed_command: str, failure_mode: str
    ) -> None:
        """Called when a trigger validation fails.

        Args:
            trigger_type: Type of trigger that failed.
            failed_command: The command ref that failed.
            failure_mode: How the failure was handled ("abort", "continue", "remediate").
        """
        ...

    def on_trigger_validation_skipped(self, trigger_type: str, reason: str) -> None:
        """Called when a trigger validation is skipped.

        Args:
            trigger_type: Type of trigger that was skipped.
            reason: Reason for skipping (e.g., "sigint", "run_aborted").
        """
        ...

    def on_trigger_remediation_started(
        self, trigger_type: str, attempt: int, max_retries: int
    ) -> None:
        """Called when remediation begins for a failed trigger command.

        Args:
            trigger_type: Type of trigger being remediated.
            attempt: Current attempt number (1-indexed).
            max_retries: Maximum number of retry attempts.
        """
        ...

    def on_trigger_remediation_succeeded(self, trigger_type: str, attempt: int) -> None:
        """Called when trigger remediation succeeds.

        Args:
            trigger_type: Type of trigger that was remediated.
            attempt: The attempt number that succeeded.
        """
        ...

    def on_trigger_remediation_exhausted(
        self, trigger_type: str, attempts: int
    ) -> None:
        """Called when trigger remediation is exhausted without success.

        Args:
            trigger_type: Type of trigger.
            attempts: Total number of attempts made.
        """
        ...

    # -------------------------------------------------------------------------
    # Trigger code review lifecycle
    # -------------------------------------------------------------------------

    def on_trigger_code_review_started(self, trigger_type: str) -> None:
        """Called when trigger code review starts.

        Args:
            trigger_type: Type of trigger (e.g., "run_end", "epic_completion").
        """
        ...

    def on_trigger_code_review_skipped(self, trigger_type: str, reason: str) -> None:
        """Called when trigger code review is skipped.

        Args:
            trigger_type: Type of trigger.
            reason: Reason for skipping (e.g., "disabled", "no_changes").
        """
        ...

    def on_trigger_code_review_passed(self, trigger_type: str) -> None:
        """Called when trigger code review passes.

        Args:
            trigger_type: Type of trigger.
        """
        ...

    def on_trigger_code_review_failed(
        self, trigger_type: str, blocking_count: int
    ) -> None:
        """Called when trigger code review fails.

        Args:
            trigger_type: Type of trigger.
            blocking_count: Number of blocking issues found.
        """
        ...

    def on_trigger_code_review_error(self, trigger_type: str, error: str) -> None:
        """Called when trigger code review encounters an error.

        Args:
            trigger_type: Type of trigger.
            error: Error message.
        """
        ...

    # -------------------------------------------------------------------------
    # Session end lifecycle
    # -------------------------------------------------------------------------

    def on_session_end_started(self, issue_id: str) -> None:
        """Called when session_end processing starts for an issue.

        Args:
            issue_id: The issue ID for which session_end processing started.
        """
        ...

    def on_session_end_completed(self, issue_id: str, result: str) -> None:
        """Called when session_end processing completes for an issue.

        Args:
            issue_id: The issue ID for which session_end processing completed.
            result: The result (pass, fail, timeout, interrupted).
        """
        ...

    def on_session_end_skipped(self, issue_id: str, reason: str) -> None:
        """Called when session_end processing is skipped for an issue.

        Args:
            issue_id: The issue ID for which session_end processing was skipped.
            reason: The reason for skipping (gate_failed, not_configured).
        """
        ...

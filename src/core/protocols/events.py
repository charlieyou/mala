"""Event sink protocols for orchestrator event handling.

This module defines protocols and dataclasses for the orchestrator event system,
enabling decoupled presentation (console, logging, metrics) from coordination logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .lifecycle import DeadlockInfoProtocol


@dataclass
class TriggerSummary:
    """Summary of a single trigger configuration for logging.

    Attributes:
        enabled: Whether this trigger is configured.
        failure_mode: The failure mode (e.g., "abort", "continue", "remediate").
        command_count: Number of commands configured for this trigger.
        command_names: Names of commands configured for this trigger.
    """

    enabled: bool = False
    failure_mode: str | None = None
    command_count: int = 0
    command_names: tuple[str, ...] = ()


@dataclass
class ValidationTriggersSummary:
    """Summary of all validation triggers for logging.

    Provides a lightweight summary of trigger configuration for CLI output
    without importing the full ValidationConfig machinery.

    Attributes:
        epic_completion: Summary for epic completion trigger.
        session_end: Summary for session end trigger.
        periodic: Summary for periodic trigger.
        run_end: Summary for run end trigger.
    """

    epic_completion: TriggerSummary | None = None
    session_end: TriggerSummary | None = None
    periodic: TriggerSummary | None = None
    run_end: TriggerSummary | None = None

    def has_any_enabled(self) -> bool:
        """Return True if any trigger is enabled."""
        return any(
            t is not None and t.enabled
            for t in [
                self.epic_completion,
                self.session_end,
                self.periodic,
                self.run_end,
            ]
        )


@dataclass
class EventRunConfig:
    """Configuration snapshot for a run, passed to on_run_started.

    Mirrors the relevant fields from MalaOrchestrator for event reporting.
    """

    repo_path: str
    max_agents: int | None
    timeout_minutes: int | None
    max_issues: int | None
    max_gate_retries: int
    max_review_retries: int
    epic_id: str | None = None
    only_ids: list[str] | None = None
    review_enabled: bool = True  # Cerberus code review enabled
    review_disabled_reason: str | None = None
    include_wip: bool = False
    orphans_only: bool = False
    cli_args: dict[str, object] | None = None
    validation_triggers: ValidationTriggersSummary | None = None


@runtime_checkable
class MalaEventSink(Protocol):
    """Protocol for receiving orchestrator events.

    Implementations handle presentation (console, logging, metrics) while
    the orchestrator focuses on coordination logic. Each method corresponds
    to a semantic event in the orchestration flow.

    All methods are synchronous and should be non-blocking. Implementations
    that need async behavior should queue events internally.
    """

    # -------------------------------------------------------------------------
    # Run lifecycle
    # -------------------------------------------------------------------------

    def on_run_started(self, config: EventRunConfig) -> None:
        """Called when the orchestrator run begins.

        Args:
            config: Run configuration snapshot.
        """
        ...

    def on_run_completed(
        self,
        success_count: int,
        total_count: int,
        run_validation_passed: bool | None,
        abort_reason: str | None = None,
        *,
        validation_ran: bool = True,
    ) -> None:
        """Called when the orchestrator run completes.

        Args:
            success_count: Number of issues completed successfully.
            total_count: Total number of issues processed.
            run_validation_passed: Whether global validation passed, or None if skipped.
            abort_reason: If run was aborted, the reason string.
            validation_ran: Whether validation was attempted. False if skipped due to
                no issues completed. Used to distinguish between skipped (exit 0) and
                interrupted (exit 130) when run_validation_passed is None.
        """
        ...

    def on_ready_issues(self, issue_ids: list[str]) -> None:
        """Called when ready issues are fetched.

        Args:
            issue_ids: List of issue IDs ready for processing.
        """
        ...

    def on_waiting_for_agents(self, count: int) -> None:
        """Called when waiting for agents to complete.

        Args:
            count: Number of active agents being waited on.
        """
        ...

    def on_no_more_issues(self, reason: str) -> None:
        """Called when there are no more issues to process.

        Args:
            reason: Reason string (e.g., "limit_reached", "none_ready").
        """
        ...

    # -------------------------------------------------------------------------
    # Agent lifecycle
    # -------------------------------------------------------------------------

    def on_agent_started(self, agent_id: str, issue_id: str) -> None:
        """Called when an agent is spawned for an issue.

        Args:
            agent_id: Unique agent identifier.
            issue_id: Issue being worked on.
        """
        ...

    def on_agent_completed(
        self,
        agent_id: str,
        issue_id: str,
        success: bool,
        duration_seconds: float,
        summary: str,
    ) -> None:
        """Called when an agent completes (success or failure).

        Args:
            agent_id: Unique agent identifier.
            issue_id: Issue that was worked on.
            success: Whether the agent succeeded.
            duration_seconds: Total execution time.
            summary: Result summary or error message.
        """
        ...

    def on_claim_failed(self, agent_id: str, issue_id: str) -> None:
        """Called when claiming an issue fails.

        Args:
            agent_id: Agent that attempted the claim.
            issue_id: Issue that could not be claimed.
        """
        ...

    # -------------------------------------------------------------------------
    # SDK message streaming
    # -------------------------------------------------------------------------

    def on_tool_use(
        self,
        agent_id: str,
        tool_name: str,
        description: str = "",
        arguments: dict[str, Any] | None = None,
    ) -> None:
        """Called when an agent invokes a tool.

        Args:
            agent_id: Agent invoking the tool.
            tool_name: Name of the tool being called.
            description: Brief description of the action.
            arguments: Tool arguments (may be truncated for display).
        """
        ...

    def on_agent_text(self, agent_id: str, text: str) -> None:
        """Called when an agent emits text output.

        Args:
            agent_id: Agent emitting text.
            text: Text content (may be truncated for display).
        """
        ...

    # -------------------------------------------------------------------------
    # Quality gate events
    # -------------------------------------------------------------------------

    def on_gate_started(
        self,
        agent_id: str | None,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        """Called when a quality gate check begins.

        Args:
            agent_id: Agent ID (None for global gate).
            attempt: Current attempt number (1-indexed).
            max_attempts: Maximum retry attempts.
            issue_id: Issue being validated (for display).
        """
        ...

    def on_gate_passed(
        self,
        agent_id: str | None,
        issue_id: str | None = None,
    ) -> None:
        """Called when a quality gate passes.

        Args:
            agent_id: Agent ID (None for global gate).
            issue_id: Issue being validated (for display).
        """
        ...

    def on_gate_failed(
        self,
        agent_id: str | None,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        """Called when a quality gate fails after all retries.

        Args:
            agent_id: Agent ID (None for global gate).
            attempt: Final attempt number.
            max_attempts: Maximum retry attempts.
            issue_id: Issue being validated (for display).
        """
        ...

    def on_gate_retry(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        """Called when retrying a quality gate after failure.

        Args:
            agent_id: Agent being retried.
            attempt: Current retry attempt number.
            max_attempts: Maximum retry attempts.
            issue_id: Issue being validated (for display).
        """
        ...

    def on_gate_result(
        self,
        agent_id: str | None,
        passed: bool,
        failure_reasons: list[str] | None = None,
        issue_id: str | None = None,
    ) -> None:
        """Called when a quality gate check completes with its result.

        This provides the detailed gate result including failure reasons,
        complementing the simpler on_gate_passed/on_gate_failed events.

        Args:
            agent_id: Agent ID (None for global gate).
            passed: Whether the gate passed.
            failure_reasons: List of failure reasons (if failed).
            issue_id: Issue being validated (for display).
        """
        ...

    # -------------------------------------------------------------------------
    # Codex review events
    # -------------------------------------------------------------------------

    def on_review_started(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        """Called when a Codex review begins.

        Args:
            agent_id: Agent being reviewed.
            attempt: Current attempt number.
            max_attempts: Maximum review attempts.
            issue_id: Issue being reviewed (for display).
        """
        ...

    def on_review_passed(
        self,
        agent_id: str,
        issue_id: str | None = None,
    ) -> None:
        """Called when a Codex review passes.

        Args:
            agent_id: Agent that passed review.
            issue_id: Issue being reviewed (for display).
        """
        ...

    def on_review_retry(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        error_count: int | None = None,
        parse_error: str | None = None,
        issue_id: str | None = None,
    ) -> None:
        """Called when retrying a Codex review after issues found.

        Args:
            agent_id: Agent being retried.
            attempt: Current retry attempt number.
            max_attempts: Maximum review attempts.
            error_count: Number of errors found (if available).
            parse_error: Parse error message (if review failed to parse).
            issue_id: Issue being reviewed (for display).
        """
        ...

    def on_review_warning(
        self,
        message: str,
        agent_id: str | None = None,
        issue_id: str | None = None,
    ) -> None:
        """Called for review-related warnings (e.g., verdict mismatch).

        Args:
            message: Warning message.
            agent_id: Associated agent (if any).
            issue_id: Issue being reviewed (for display).
        """
        ...

    # -------------------------------------------------------------------------
    # Fixer agent events
    # -------------------------------------------------------------------------

    def on_fixer_started(
        self,
        attempt: int,
        max_attempts: int,
    ) -> None:
        """Called when a fixer agent is spawned.

        Args:
            attempt: Current fixer attempt number.
            max_attempts: Maximum fixer attempts.
        """
        ...

    def on_fixer_completed(self, result: str) -> None:
        """Called when a fixer agent completes.

        Args:
            result: Brief result description.
        """
        ...

    def on_fixer_failed(self, reason: str) -> None:
        """Called when a fixer agent fails.

        Args:
            reason: Failure reason (e.g., "timeout", "error").
        """
        ...

    # -------------------------------------------------------------------------
    # Issue lifecycle
    # -------------------------------------------------------------------------

    def on_issue_closed(self, agent_id: str, issue_id: str) -> None:
        """Called when an issue is closed after successful completion.

        Args:
            agent_id: Agent that completed the issue.
            issue_id: Issue that was closed.
        """
        ...

    def on_issue_completed(
        self,
        agent_id: str,
        issue_id: str,
        success: bool,
        duration_seconds: float,
        summary: str,
    ) -> None:
        """Called when an issue implementation completes (success or failure).

        This is the primary issue completion event, distinct from on_agent_completed
        which tracks the agent lifecycle. Use this for issue-level tracking.

        Args:
            agent_id: Agent that worked on the issue.
            issue_id: Issue that was completed.
            success: Whether the issue was successfully implemented.
            duration_seconds: Total time spent on the issue.
            summary: Result summary or error message.
        """
        ...

    def on_epic_closed(self, agent_id: str) -> None:
        """Called when a parent epic is auto-closed.

        Args:
            agent_id: Agent that triggered the epic closure.
        """
        ...

    def on_validation_started(
        self,
        agent_id: str,
        issue_id: str | None = None,
    ) -> None:
        """Called when per-session validation begins.

        Args:
            agent_id: Agent being validated.
            issue_id: Issue being validated (for display).
        """
        ...

    def on_validation_result(
        self,
        agent_id: str,
        passed: bool,
        issue_id: str | None = None,
    ) -> None:
        """Called when per-session validation completes.

        Args:
            agent_id: Agent that was validated.
            passed: Whether validation passed.
            issue_id: Issue being validated (for display).
        """
        ...

    def on_validation_step_running(
        self,
        step_name: str,
        agent_id: str | None = None,
    ) -> None:
        """Called when a validation step starts.

        Args:
            step_name: Name of the validation step (e.g., "ruff", "pytest").
            agent_id: Associated agent (if any).
        """
        ...

    def on_validation_step_skipped(
        self,
        step_name: str,
        reason: str,
        agent_id: str | None = None,
    ) -> None:
        """Called when a validation step is skipped.

        Args:
            step_name: Name of the validation step.
            reason: Reason for skipping (e.g., "cache hit", "no changes").
            agent_id: Associated agent (if any).
        """
        ...

    def on_validation_step_passed(
        self,
        step_name: str,
        duration_seconds: float,
        agent_id: str | None = None,
    ) -> None:
        """Called when a validation step succeeds.

        Args:
            step_name: Name of the validation step.
            duration_seconds: Time taken to complete the step.
            agent_id: Associated agent (if any).
        """
        ...

    def on_validation_step_failed(
        self,
        step_name: str,
        exit_code: int,
        agent_id: str | None = None,
    ) -> None:
        """Called when a validation step fails.

        Args:
            step_name: Name of the validation step.
            exit_code: Exit code from the step.
            agent_id: Associated agent (if any).
        """
        ...

    # -------------------------------------------------------------------------
    # Warnings and diagnostics
    # -------------------------------------------------------------------------

    def on_warning(self, message: str, agent_id: str | None = None) -> None:
        """Called for warning conditions.

        Args:
            message: Warning message.
            agent_id: Associated agent (if any).
        """
        ...

    def on_log_timeout(self, agent_id: str, log_path: str) -> None:
        """Called when waiting for a log file times out.

        Args:
            agent_id: Agent waiting for the log.
            log_path: Path to the missing log file.
        """
        ...

    def on_locks_cleaned(self, agent_id: str, count: int) -> None:
        """Called when stale locks are cleaned up for an agent.

        Args:
            agent_id: Agent whose locks were cleaned.
            count: Number of locks cleaned.
        """
        ...

    def on_locks_released(self, count: int) -> None:
        """Called when remaining locks are released at run end.

        Args:
            count: Number of locks released.
        """
        ...

    def on_issues_committed(self) -> None:
        """Called when .beads/issues.jsonl is committed."""
        ...

    def on_run_metadata_saved(self, path: str) -> None:
        """Called when run metadata is saved.

        Args:
            path: Path to the saved metadata file.
        """
        ...

    def on_global_validation_disabled(self) -> None:
        """Called when global validation is disabled."""
        ...

    def on_abort_requested(self, reason: str) -> None:
        """Called when a fatal error triggers a run abort.

        Args:
            reason: Description of the fatal error.
        """
        ...

    def on_tasks_aborting(self, count: int, reason: str) -> None:
        """Called when active tasks are being aborted.

        Args:
            count: Number of active tasks being aborted.
            reason: Reason for the abort.
        """
        ...

    # -------------------------------------------------------------------------
    # SIGINT escalation lifecycle
    # -------------------------------------------------------------------------

    def on_drain_started(self, active_task_count: int) -> None:
        """Called when drain mode starts (1st Ctrl-C).

        Args:
            active_task_count: Number of active tasks being drained.
        """
        ...

    def on_abort_started(self) -> None:
        """Called when graceful abort starts (2nd Ctrl-C)."""
        ...

    def on_force_abort(self) -> None:
        """Called when hard abort starts (3rd Ctrl-C)."""
        ...

    # -------------------------------------------------------------------------
    # Epic verification lifecycle
    # -------------------------------------------------------------------------

    def on_epic_verification_started(self, epic_id: str) -> None:
        """Called when epic verification begins.

        Args:
            epic_id: The epic being verified.
        """
        ...

    def on_epic_verification_passed(self, epic_id: str, confidence: float) -> None:
        """Called when epic verification passes.

        Args:
            epic_id: The epic that passed verification.
            confidence: Confidence score (0.0 to 1.0).
        """
        ...

    def on_epic_verification_failed(
        self,
        epic_id: str,
        unmet_count: int,
        remediation_ids: list[str],
    ) -> None:
        """Called when epic verification fails with unmet criteria.

        Args:
            epic_id: The epic that failed verification.
            unmet_count: Number of unmet criteria.
            remediation_ids: IDs of created remediation issues.
        """
        ...

    def on_epic_remediation_created(
        self,
        epic_id: str,
        issue_id: str,
        criterion: str,
    ) -> None:
        """Called when a remediation issue is created for an unmet criterion.

        Args:
            epic_id: The epic the remediation is for.
            issue_id: The created issue ID.
            criterion: The unmet criterion text (may be truncated).
        """
        ...

    # -------------------------------------------------------------------------
    # Pipeline module events
    # -------------------------------------------------------------------------

    def on_lifecycle_state(self, agent_id: str, state: str) -> None:
        """Called when lifecycle state changes (verbose/debug).

        Args:
            agent_id: Agent whose lifecycle changed.
            state: New lifecycle state name.
        """
        ...

    def on_log_waiting(self, agent_id: str) -> None:
        """Called when waiting for session log file.

        Args:
            agent_id: Agent waiting for log.
        """
        ...

    def on_log_ready(self, agent_id: str) -> None:
        """Called when session log file is ready.

        Args:
            agent_id: Agent whose log is ready.
        """
        ...

    def on_review_skipped_no_progress(self, agent_id: str) -> None:
        """Called when review is skipped due to no progress.

        Args:
            agent_id: Agent whose review was skipped.
        """
        ...

    def on_fixer_text(self, attempt: int, text: str) -> None:
        """Called when fixer agent emits text output.

        Args:
            attempt: Current fixer attempt number.
            text: Text content.
        """
        ...

    def on_fixer_tool_use(
        self,
        attempt: int,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        """Called when fixer agent invokes a tool.

        Args:
            attempt: Current fixer attempt number.
            tool_name: Name of the tool being called.
            arguments: Tool arguments.
        """
        ...

    def on_deadlock_detected(self, info: DeadlockInfoProtocol) -> None:
        """Called when a deadlock is detected and resolved.

        Args:
            info: Information about the detected deadlock, including the cycle
                of agents, the victim selected for cancellation, and the
                blocker holding the needed resource.
        """
        ...

    def on_watch_idle(self, wait_seconds: float, issues_blocked: int | None) -> None:
        """Called when watch mode enters idle sleep.

        Args:
            wait_seconds: Duration of the upcoming sleep.
            issues_blocked: Count of blocked issues, or None if unknown.
        """
        ...

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
        self, trigger_type: str, command_ref: str, index: int
    ) -> None:
        """Called when a trigger command starts execution.

        Args:
            trigger_type: Type of trigger.
            command_ref: Reference name of the command.
            index: 0-based index of the command in the trigger's command list.
        """
        ...

    def on_trigger_command_completed(
        self,
        trigger_type: str,
        command_ref: str,
        index: int,
        passed: bool,
        duration_seconds: float,
    ) -> None:
        """Called when a trigger command completes.

        Args:
            trigger_type: Type of trigger.
            command_ref: Reference name of the command.
            index: 0-based index of the command.
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

"""Fake event sink for testing.

FakeEventSink captures all MalaEventSink events for assertion in tests.
"""

from dataclasses import dataclass, field
from typing import Any

from src.core.protocols.events import EventRunConfig, MalaEventSink
from src.core.protocols.lifecycle import DeadlockInfoProtocol


@dataclass
class RecordedEvent:
    """A captured event with its type and arguments."""

    event_type: str
    kwargs: dict[str, Any] = field(default_factory=dict)


class FakeEventSink(MalaEventSink):
    """In-memory event sink that records all events for testing.

    Usage:
        sink = FakeEventSink()
        # ... code that emits events ...
        assert sink.has_event("run_started")
        events = sink.get_events("agent_completed")
    """

    def __init__(self) -> None:
        self._events: list[RecordedEvent] = []

    def _record(self, event_type: str, **kwargs: object) -> None:
        """Record an event with its arguments."""
        self._events.append(RecordedEvent(event_type=event_type, kwargs=kwargs))

    @property
    def events(self) -> list[RecordedEvent]:
        """All recorded events."""
        return self._events

    def has_event(self, event_type: str) -> bool:
        """Check if an event of the given type was recorded."""
        return any(e.event_type == event_type for e in self._events)

    def get_events(self, event_type: str) -> list[RecordedEvent]:
        """Get all events of the given type."""
        return [e for e in self._events if e.event_type == event_type]

    def clear(self) -> None:
        """Clear all recorded events."""
        self._events.clear()

    # -------------------------------------------------------------------------
    # Run lifecycle
    # -------------------------------------------------------------------------

    def on_run_started(self, config: EventRunConfig) -> None:
        self._record("run_started", config=config)

    def on_run_completed(
        self,
        success_count: int,
        total_count: int,
        run_validation_passed: bool | None,
        abort_reason: str | None = None,
        *,
        validation_ran: bool = True,
    ) -> None:
        self._record(
            "run_completed",
            success_count=success_count,
            total_count=total_count,
            run_validation_passed=run_validation_passed,
            abort_reason=abort_reason,
            validation_ran=validation_ran,
        )

    def on_ready_issues(self, issue_ids: list[str]) -> None:
        self._record("ready_issues", issue_ids=issue_ids)

    def on_waiting_for_agents(self, count: int) -> None:
        self._record("waiting_for_agents", count=count)

    def on_no_more_issues(self, reason: str) -> None:
        self._record("no_more_issues", reason=reason)

    # -------------------------------------------------------------------------
    # Agent lifecycle
    # -------------------------------------------------------------------------

    def on_agent_started(self, agent_id: str, issue_id: str) -> None:
        self._record("agent_started", agent_id=agent_id, issue_id=issue_id)

    def on_agent_completed(
        self,
        agent_id: str,
        issue_id: str,
        success: bool,
        duration_seconds: float,
        summary: str,
    ) -> None:
        self._record(
            "agent_completed",
            agent_id=agent_id,
            issue_id=issue_id,
            success=success,
            duration_seconds=duration_seconds,
            summary=summary,
        )

    def on_claim_failed(self, agent_id: str, issue_id: str) -> None:
        self._record("claim_failed", agent_id=agent_id, issue_id=issue_id)

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
        self._record(
            "tool_use",
            agent_id=agent_id,
            tool_name=tool_name,
            description=description,
            arguments=arguments,
        )

    def on_agent_text(self, agent_id: str, text: str) -> None:
        self._record("agent_text", agent_id=agent_id, text=text)

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
        self._record(
            "gate_started",
            agent_id=agent_id,
            attempt=attempt,
            max_attempts=max_attempts,
            issue_id=issue_id,
        )

    def on_gate_passed(
        self,
        agent_id: str | None,
        issue_id: str | None = None,
    ) -> None:
        self._record("gate_passed", agent_id=agent_id, issue_id=issue_id)

    def on_gate_failed(
        self,
        agent_id: str | None,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        self._record(
            "gate_failed",
            agent_id=agent_id,
            attempt=attempt,
            max_attempts=max_attempts,
            issue_id=issue_id,
        )

    def on_gate_retry(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        self._record(
            "gate_retry",
            agent_id=agent_id,
            attempt=attempt,
            max_attempts=max_attempts,
            issue_id=issue_id,
        )

    def on_gate_result(
        self,
        agent_id: str | None,
        passed: bool,
        failure_reasons: list[str] | None = None,
        issue_id: str | None = None,
    ) -> None:
        self._record(
            "gate_result",
            agent_id=agent_id,
            passed=passed,
            failure_reasons=failure_reasons,
            issue_id=issue_id,
        )

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
        self._record(
            "review_started",
            agent_id=agent_id,
            attempt=attempt,
            max_attempts=max_attempts,
            issue_id=issue_id,
        )

    def on_review_passed(
        self,
        agent_id: str,
        issue_id: str | None = None,
    ) -> None:
        self._record("review_passed", agent_id=agent_id, issue_id=issue_id)

    def on_review_retry(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        error_count: int | None = None,
        parse_error: str | None = None,
        issue_id: str | None = None,
    ) -> None:
        self._record(
            "review_retry",
            agent_id=agent_id,
            attempt=attempt,
            max_attempts=max_attempts,
            error_count=error_count,
            parse_error=parse_error,
            issue_id=issue_id,
        )

    def on_review_warning(
        self,
        message: str,
        agent_id: str | None = None,
        issue_id: str | None = None,
    ) -> None:
        self._record(
            "review_warning",
            message=message,
            agent_id=agent_id,
            issue_id=issue_id,
        )

    # -------------------------------------------------------------------------
    # Fixer agent events
    # -------------------------------------------------------------------------

    def on_fixer_started(
        self,
        attempt: int,
        max_attempts: int,
    ) -> None:
        self._record("fixer_started", attempt=attempt, max_attempts=max_attempts)

    def on_fixer_completed(self, result: str) -> None:
        self._record("fixer_completed", result=result)

    def on_fixer_failed(self, reason: str) -> None:
        self._record("fixer_failed", reason=reason)

    # -------------------------------------------------------------------------
    # Issue lifecycle
    # -------------------------------------------------------------------------

    def on_issue_closed(self, agent_id: str, issue_id: str) -> None:
        self._record("issue_closed", agent_id=agent_id, issue_id=issue_id)

    def on_issue_completed(
        self,
        agent_id: str,
        issue_id: str,
        success: bool,
        duration_seconds: float,
        summary: str,
    ) -> None:
        self._record(
            "issue_completed",
            agent_id=agent_id,
            issue_id=issue_id,
            success=success,
            duration_seconds=duration_seconds,
            summary=summary,
        )

    def on_epic_closed(self, agent_id: str) -> None:
        self._record("epic_closed", agent_id=agent_id)

    def on_validation_started(
        self,
        agent_id: str,
        issue_id: str | None = None,
    ) -> None:
        self._record("validation_started", agent_id=agent_id, issue_id=issue_id)

    def on_validation_result(
        self,
        agent_id: str,
        passed: bool,
        issue_id: str | None = None,
    ) -> None:
        self._record(
            "validation_result",
            agent_id=agent_id,
            passed=passed,
            issue_id=issue_id,
        )

    def on_validation_step_running(
        self,
        step_name: str,
        agent_id: str | None = None,
    ) -> None:
        self._record("validation_step_running", step_name=step_name, agent_id=agent_id)

    def on_validation_step_skipped(
        self,
        step_name: str,
        reason: str,
        agent_id: str | None = None,
    ) -> None:
        self._record(
            "validation_step_skipped",
            step_name=step_name,
            reason=reason,
            agent_id=agent_id,
        )

    def on_validation_step_passed(
        self,
        step_name: str,
        duration_seconds: float,
        agent_id: str | None = None,
    ) -> None:
        self._record(
            "validation_step_passed",
            step_name=step_name,
            duration_seconds=duration_seconds,
            agent_id=agent_id,
        )

    def on_validation_step_failed(
        self,
        step_name: str,
        exit_code: int,
        agent_id: str | None = None,
    ) -> None:
        self._record(
            "validation_step_failed",
            step_name=step_name,
            exit_code=exit_code,
            agent_id=agent_id,
        )

    # -------------------------------------------------------------------------
    # Warnings and diagnostics
    # -------------------------------------------------------------------------

    def on_warning(self, message: str, agent_id: str | None = None) -> None:
        self._record("warning", message=message, agent_id=agent_id)

    def on_log_timeout(self, agent_id: str, log_path: str) -> None:
        self._record("log_timeout", agent_id=agent_id, log_path=log_path)

    def on_locks_cleaned(self, agent_id: str, count: int) -> None:
        self._record("locks_cleaned", agent_id=agent_id, count=count)

    def on_locks_released(self, count: int) -> None:
        self._record("locks_released", count=count)

    def on_issues_committed(self) -> None:
        self._record("issues_committed")

    def on_run_metadata_saved(self, path: str) -> None:
        self._record("run_metadata_saved", path=path)

    def on_global_validation_disabled(self) -> None:
        self._record("global_validation_disabled")

    def on_abort_requested(self, reason: str) -> None:
        self._record("abort_requested", reason=reason)

    def on_tasks_aborting(self, count: int, reason: str) -> None:
        self._record("tasks_aborting", count=count, reason=reason)

    # -------------------------------------------------------------------------
    # SIGINT escalation lifecycle
    # -------------------------------------------------------------------------

    def on_drain_started(self, active_task_count: int) -> None:
        self._record("drain_started", active_task_count=active_task_count)

    def on_abort_started(self) -> None:
        self._record("abort_started")

    def on_force_abort(self) -> None:
        self._record("force_abort")

    # -------------------------------------------------------------------------
    # Epic verification lifecycle
    # -------------------------------------------------------------------------

    def on_epic_verification_started(
        self, epic_id: str, *, reviewer_type: str = "agent_sdk"
    ) -> None:
        self._record(
            "epic_verification_started", epic_id=epic_id, reviewer_type=reviewer_type
        )

    def on_epic_verification_passed(
        self, epic_id: str, *, reviewer_type: str = "agent_sdk"
    ) -> None:
        self._record(
            "epic_verification_passed",
            epic_id=epic_id,
            reviewer_type=reviewer_type,
        )

    def on_epic_verification_failed(
        self,
        epic_id: str,
        unmet_count: int,
        remediation_ids: list[str],
        *,
        reason: str | None = None,
        reviewer_type: str = "agent_sdk",
    ) -> None:
        self._record(
            "epic_verification_failed",
            epic_id=epic_id,
            unmet_count=unmet_count,
            remediation_ids=remediation_ids,
            reason=reason,
            reviewer_type=reviewer_type,
        )

    def on_epic_remediation_created(
        self,
        epic_id: str,
        issue_id: str,
        criterion: str,
    ) -> None:
        self._record(
            "epic_remediation_created",
            epic_id=epic_id,
            issue_id=issue_id,
            criterion=criterion,
        )

    # -------------------------------------------------------------------------
    # Pipeline module events
    # -------------------------------------------------------------------------

    def on_lifecycle_state(self, agent_id: str, state: str) -> None:
        self._record("lifecycle_state", agent_id=agent_id, state=state)

    def on_log_waiting(self, agent_id: str) -> None:
        self._record("log_waiting", agent_id=agent_id)

    def on_log_ready(self, agent_id: str) -> None:
        self._record("log_ready", agent_id=agent_id)

    def on_review_skipped_no_progress(self, agent_id: str) -> None:
        self._record("review_skipped_no_progress", agent_id=agent_id)

    def on_fixer_text(self, attempt: int, text: str) -> None:
        self._record("fixer_text", attempt=attempt, text=text)

    def on_fixer_tool_use(
        self,
        attempt: int,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        self._record(
            "fixer_tool_use",
            attempt=attempt,
            tool_name=tool_name,
            arguments=arguments,
        )

    def on_deadlock_detected(self, info: DeadlockInfoProtocol) -> None:
        self._record("deadlock_detected", info=info)

    def on_watch_idle(self, wait_seconds: float, issues_blocked: int | None) -> None:
        self._record(
            "watch_idle",
            wait_seconds=wait_seconds,
            issues_blocked=issues_blocked,
        )

    # -------------------------------------------------------------------------
    # Trigger validation lifecycle
    # -------------------------------------------------------------------------

    def on_trigger_validation_queued(
        self, trigger_type: str, trigger_context: str
    ) -> None:
        self._record(
            "trigger_validation_queued",
            trigger_type=trigger_type,
            trigger_context=trigger_context,
        )

    def on_trigger_validation_started(
        self, trigger_type: str, commands: list[str]
    ) -> None:
        self._record(
            "trigger_validation_started",
            trigger_type=trigger_type,
            commands=commands,
        )

    def on_trigger_command_started(
        self, trigger_type: str, command_ref: str, index: int, total_commands: int
    ) -> None:
        self._record(
            "trigger_command_started",
            trigger_type=trigger_type,
            command_ref=command_ref,
            index=index,
            total_commands=total_commands,
        )

    def on_trigger_command_completed(
        self,
        trigger_type: str,
        command_ref: str,
        index: int,
        total_commands: int,
        passed: bool,
        duration_seconds: float,
    ) -> None:
        self._record(
            "trigger_command_completed",
            trigger_type=trigger_type,
            command_ref=command_ref,
            index=index,
            total_commands=total_commands,
            passed=passed,
            duration_seconds=duration_seconds,
        )

    def on_trigger_validation_passed(
        self, trigger_type: str, duration_seconds: float
    ) -> None:
        self._record(
            "trigger_validation_passed",
            trigger_type=trigger_type,
            duration_seconds=duration_seconds,
        )

    def on_trigger_validation_failed(
        self, trigger_type: str, failed_command: str, failure_mode: str
    ) -> None:
        self._record(
            "trigger_validation_failed",
            trigger_type=trigger_type,
            failed_command=failed_command,
            failure_mode=failure_mode,
        )

    def on_trigger_validation_skipped(self, trigger_type: str, reason: str) -> None:
        self._record(
            "trigger_validation_skipped",
            trigger_type=trigger_type,
            reason=reason,
        )

    def on_trigger_remediation_started(
        self, trigger_type: str, attempt: int, max_retries: int
    ) -> None:
        self._record(
            "trigger_remediation_started",
            trigger_type=trigger_type,
            attempt=attempt,
            max_retries=max_retries,
        )

    def on_trigger_remediation_succeeded(self, trigger_type: str, attempt: int) -> None:
        self._record(
            "trigger_remediation_succeeded",
            trigger_type=trigger_type,
            attempt=attempt,
        )

    def on_trigger_remediation_exhausted(
        self, trigger_type: str, attempts: int
    ) -> None:
        self._record(
            "trigger_remediation_exhausted",
            trigger_type=trigger_type,
            attempts=attempts,
        )

    # -------------------------------------------------------------------------
    # Trigger code review lifecycle
    # -------------------------------------------------------------------------

    def on_trigger_code_review_started(self, trigger_type: str) -> None:
        self._record("trigger_code_review_started", trigger_type=trigger_type)

    def on_trigger_code_review_skipped(self, trigger_type: str, reason: str) -> None:
        self._record(
            "trigger_code_review_skipped", trigger_type=trigger_type, reason=reason
        )

    def on_trigger_code_review_passed(self, trigger_type: str) -> None:
        self._record("trigger_code_review_passed", trigger_type=trigger_type)

    def on_trigger_code_review_failed(
        self, trigger_type: str, blocking_count: int
    ) -> None:
        self._record(
            "trigger_code_review_failed",
            trigger_type=trigger_type,
            blocking_count=blocking_count,
        )

    def on_trigger_code_review_error(self, trigger_type: str, error: str) -> None:
        self._record(
            "trigger_code_review_error", trigger_type=trigger_type, error=error
        )

    # -------------------------------------------------------------------------
    # Session end lifecycle
    # -------------------------------------------------------------------------

    def on_session_end_started(self, issue_id: str) -> None:
        self._record("session_end_started", issue_id=issue_id)

    def on_session_end_completed(self, issue_id: str, result: str) -> None:
        self._record("session_end_completed", issue_id=issue_id, result=result)

    def on_session_end_skipped(self, issue_id: str, reason: str) -> None:
        self._record("session_end_skipped", issue_id=issue_id, reason=reason)

"""Base event sink implementation with no-op defaults.

Provides BaseEventSink with default no-op implementations of all protocol
methods, and NullEventSink for testing.
"""

from typing import Any

from src.core.protocols.events import EventRunConfig, MalaEventSink
from src.core.protocols.lifecycle import DeadlockInfoProtocol


class BaseEventSink:
    """Base event sink with no-op implementations of all protocol methods.

    Provides default no-op implementations for all MalaEventSink protocol
    methods. Subclasses can override only the methods they need to handle.

    This eliminates the need to implement all 51 methods when creating a
    new event sink - just inherit from BaseEventSink and override what
    you need.

    Example:
        class MyEventSink(BaseEventSink):
            def on_agent_started(self, agent_id: str, issue_id: str) -> None:
                print(f"Agent {agent_id} started on {issue_id}")
            # All other methods are no-ops by default
    """

    # -------------------------------------------------------------------------
    # Run lifecycle
    # -------------------------------------------------------------------------

    def on_run_started(self, config: EventRunConfig) -> None:
        pass

    def on_run_completed(
        self,
        success_count: int,
        total_count: int,
        run_validation_passed: bool | None,
        abort_reason: str | None = None,
        *,
        validation_ran: bool = True,
    ) -> None:
        pass

    def on_ready_issues(self, issue_ids: list[str]) -> None:
        pass

    def on_waiting_for_agents(self, count: int) -> None:
        pass

    def on_no_more_issues(self, reason: str) -> None:
        pass

    # -------------------------------------------------------------------------
    # Agent lifecycle
    # -------------------------------------------------------------------------

    def on_agent_started(self, agent_id: str, issue_id: str) -> None:
        pass

    def on_agent_completed(
        self,
        agent_id: str,
        issue_id: str,
        success: bool,
        duration_seconds: float,
        summary: str,
    ) -> None:
        pass

    def on_claim_failed(self, agent_id: str, issue_id: str) -> None:
        pass

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
        pass

    def on_agent_text(self, agent_id: str, text: str) -> None:
        pass

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
        pass

    def on_gate_passed(
        self,
        agent_id: str | None,
        issue_id: str | None = None,
    ) -> None:
        pass

    def on_gate_failed(
        self,
        agent_id: str | None,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        pass

    def on_gate_retry(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        pass

    def on_gate_result(
        self,
        agent_id: str | None,
        passed: bool,
        failure_reasons: list[str] | None = None,
        issue_id: str | None = None,
    ) -> None:
        pass

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
        pass

    def on_review_passed(
        self,
        agent_id: str,
        issue_id: str | None = None,
    ) -> None:
        pass

    def on_review_retry(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        error_count: int | None = None,
        parse_error: str | None = None,
        issue_id: str | None = None,
    ) -> None:
        pass

    def on_review_warning(
        self,
        message: str,
        agent_id: str | None = None,
        issue_id: str | None = None,
    ) -> None:
        pass

    # -------------------------------------------------------------------------
    # Fixer agent events
    # -------------------------------------------------------------------------

    def on_fixer_started(
        self,
        attempt: int,
        max_attempts: int,
    ) -> None:
        pass

    def on_fixer_completed(self, result: str) -> None:
        pass

    def on_fixer_failed(self, reason: str) -> None:
        pass

    # -------------------------------------------------------------------------
    # Issue lifecycle
    # -------------------------------------------------------------------------

    def on_issue_closed(self, agent_id: str, issue_id: str) -> None:
        pass

    def on_issue_completed(
        self,
        agent_id: str,
        issue_id: str,
        success: bool,
        duration_seconds: float,
        summary: str,
    ) -> None:
        pass

    def on_epic_closed(self, agent_id: str) -> None:
        pass

    def on_validation_started(
        self,
        agent_id: str,
        issue_id: str | None = None,
    ) -> None:
        pass

    def on_validation_result(
        self,
        agent_id: str,
        passed: bool,
        issue_id: str | None = None,
    ) -> None:
        pass

    def on_validation_step_running(
        self,
        step_name: str,
        agent_id: str | None = None,
    ) -> None:
        pass

    def on_validation_step_skipped(
        self,
        step_name: str,
        reason: str,
        agent_id: str | None = None,
    ) -> None:
        pass

    def on_validation_step_passed(
        self,
        step_name: str,
        duration_seconds: float,
        agent_id: str | None = None,
    ) -> None:
        pass

    def on_validation_step_failed(
        self,
        step_name: str,
        exit_code: int,
        agent_id: str | None = None,
    ) -> None:
        pass

    # -------------------------------------------------------------------------
    # Warnings and diagnostics
    # -------------------------------------------------------------------------

    def on_warning(self, message: str, agent_id: str | None = None) -> None:
        pass

    def on_log_timeout(self, agent_id: str, log_path: str) -> None:
        pass

    def on_locks_cleaned(self, agent_id: str, count: int) -> None:
        pass

    def on_locks_released(self, count: int) -> None:
        pass

    def on_issues_committed(self) -> None:
        pass

    def on_run_metadata_saved(self, path: str) -> None:
        pass

    def on_global_validation_disabled(self) -> None:
        pass

    def on_abort_requested(self, reason: str) -> None:
        pass

    def on_tasks_aborting(self, count: int, reason: str) -> None:
        pass

    # -------------------------------------------------------------------------
    # SIGINT escalation lifecycle
    # -------------------------------------------------------------------------

    def on_drain_started(self, active_task_count: int) -> None:
        pass

    def on_abort_started(self) -> None:
        pass

    def on_force_abort(self) -> None:
        pass

    # -------------------------------------------------------------------------
    # Epic verification lifecycle
    # -------------------------------------------------------------------------

    def on_epic_verification_started(
        self, epic_id: str, *, reviewer_type: str = "agent_sdk"
    ) -> None:
        pass

    def on_epic_verification_passed(
        self, epic_id: str, *, reviewer_type: str = "agent_sdk"
    ) -> None:
        pass

    def on_epic_verification_failed(
        self,
        epic_id: str,
        unmet_count: int,
        remediation_ids: list[str],
        *,
        reason: str | None = None,
        reviewer_type: str = "agent_sdk",
    ) -> None:
        pass

    def on_epic_remediation_created(
        self,
        epic_id: str,
        issue_id: str,
        criterion: str,
    ) -> None:
        pass

    # -------------------------------------------------------------------------
    # Pipeline module events
    # -------------------------------------------------------------------------

    def on_lifecycle_state(self, agent_id: str, state: str) -> None:
        pass

    def on_log_waiting(self, agent_id: str) -> None:
        pass

    def on_log_ready(self, agent_id: str) -> None:
        pass

    def on_review_skipped_no_progress(self, agent_id: str) -> None:
        pass

    def on_fixer_text(self, attempt: int, text: str) -> None:
        pass

    def on_fixer_tool_use(
        self,
        attempt: int,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        pass

    def on_deadlock_detected(self, info: DeadlockInfoProtocol) -> None:
        pass

    def on_watch_idle(self, wait_seconds: float, issues_blocked: int | None) -> None:
        pass

    # -------------------------------------------------------------------------
    # Trigger validation lifecycle
    # -------------------------------------------------------------------------

    def on_trigger_validation_queued(
        self, trigger_type: str, trigger_context: str
    ) -> None:
        pass

    def on_trigger_validation_started(
        self, trigger_type: str, commands: list[str]
    ) -> None:
        pass

    def on_trigger_command_started(
        self, trigger_type: str, command_ref: str, index: int, total_commands: int
    ) -> None:
        pass

    def on_trigger_command_completed(
        self,
        trigger_type: str,
        command_ref: str,
        index: int,
        total_commands: int,
        passed: bool,
        duration_seconds: float,
    ) -> None:
        pass

    def on_trigger_validation_passed(
        self, trigger_type: str, duration_seconds: float
    ) -> None:
        pass

    def on_trigger_validation_failed(
        self, trigger_type: str, failed_command: str, failure_mode: str
    ) -> None:
        pass

    def on_trigger_validation_skipped(self, trigger_type: str, reason: str) -> None:
        pass

    def on_trigger_remediation_started(
        self, trigger_type: str, attempt: int, max_retries: int
    ) -> None:
        pass

    def on_trigger_remediation_succeeded(self, trigger_type: str, attempt: int) -> None:
        pass

    def on_trigger_remediation_exhausted(
        self, trigger_type: str, attempts: int
    ) -> None:
        pass

    # -------------------------------------------------------------------------
    # Trigger code review lifecycle
    # -------------------------------------------------------------------------

    def on_trigger_code_review_started(self, trigger_type: str) -> None:
        pass

    def on_trigger_code_review_skipped(self, trigger_type: str, reason: str) -> None:
        pass

    def on_trigger_code_review_passed(self, trigger_type: str) -> None:
        pass

    def on_trigger_code_review_failed(
        self, trigger_type: str, blocking_count: int
    ) -> None:
        pass

    def on_trigger_code_review_error(self, trigger_type: str, error: str) -> None:
        pass

    # -------------------------------------------------------------------------
    # Session end lifecycle
    # -------------------------------------------------------------------------

    def on_session_end_started(self, issue_id: str) -> None:
        pass

    def on_session_end_completed(self, issue_id: str, result: str) -> None:
        pass

    def on_session_end_skipped(self, issue_id: str, reason: str) -> None:
        pass


class NullEventSink(BaseEventSink):
    """No-op event sink for testing.

    Inherits all no-op implementations from BaseEventSink. This class exists
    for backward compatibility and semantic clarity - use NullEventSink when
    you explicitly want no side effects (e.g., in tests).

    Example:
        from src.orchestration.factory import create_orchestrator, OrchestratorDependencies

        sink = NullEventSink()
        deps = OrchestratorDependencies(event_sink=sink)
        orchestrator = create_orchestrator(config, deps=deps)
        await orchestrator.run()  # No console output
    """

    pass


# Protocol assertions to verify implementation compliance
assert isinstance(BaseEventSink(), MalaEventSink)
assert isinstance(NullEventSink(), MalaEventSink)

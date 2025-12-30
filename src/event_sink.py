"""Event sink protocol for MalaOrchestrator.

Defines the MalaEventSink protocol to decouple orchestration logic from
presentation concerns. The orchestrator emits semantic events through the
sink, allowing different implementations for:
- Console output (ConsoleEventSink - follow-up task)
- Testing (NullEventSink - no side effects)
- Structured logging (future: JSON/file-based sinks)

This module defines the protocol and provides NullEventSink for testing.
"""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


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
    braintrust_enabled: bool = False
    braintrust_disabled_reason: str | None = None  # e.g., "add BRAINTRUST_API_KEY..."
    codex_review_enabled: bool = True
    codex_thinking_mode: str | None = None
    morph_enabled: bool = True
    morph_disallowed_tools: list[str] | None = None
    morph_disabled_reason: str | None = (
        None  # e.g., "--no-morph", "MORPH_API_KEY not set"
    )
    prioritize_wip: bool = False
    cli_args: dict[str, object] | None = None


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
        run_validation_passed: bool,
        abort_reason: str | None = None,
    ) -> None:
        """Called when the orchestrator run completes.

        Args:
            success_count: Number of issues completed successfully.
            total_count: Total number of issues processed.
            run_validation_passed: Whether Gate 4 (run-level validation) passed.
            abort_reason: If run was aborted, the reason string.
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
    ) -> None:
        """Called when a quality gate check begins.

        Args:
            agent_id: Agent ID (None for run-level gate).
            attempt: Current attempt number (1-indexed).
            max_attempts: Maximum retry attempts.
        """
        ...

    def on_gate_passed(self, agent_id: str | None) -> None:
        """Called when a quality gate passes.

        Args:
            agent_id: Agent ID (None for run-level gate).
        """
        ...

    def on_gate_failed(
        self,
        agent_id: str | None,
        attempt: int,
        max_attempts: int,
    ) -> None:
        """Called when a quality gate fails after all retries.

        Args:
            agent_id: Agent ID (None for run-level gate).
            attempt: Final attempt number.
            max_attempts: Maximum retry attempts.
        """
        ...

    def on_gate_retry(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
    ) -> None:
        """Called when retrying a quality gate after failure.

        Args:
            agent_id: Agent being retried.
            attempt: Current retry attempt number.
            max_attempts: Maximum retry attempts.
        """
        ...

    def on_gate_result(
        self,
        agent_id: str | None,
        passed: bool,
        failure_reasons: list[str] | None = None,
    ) -> None:
        """Called when a quality gate check completes with its result.

        This provides the detailed gate result including failure reasons,
        complementing the simpler on_gate_passed/on_gate_failed events.

        Args:
            agent_id: Agent ID (None for run-level gate).
            passed: Whether the gate passed.
            failure_reasons: List of failure reasons (if failed).
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
    ) -> None:
        """Called when a Codex review begins.

        Args:
            agent_id: Agent being reviewed.
            attempt: Current attempt number.
            max_attempts: Maximum review attempts.
        """
        ...

    def on_review_passed(self, agent_id: str) -> None:
        """Called when a Codex review passes.

        Args:
            agent_id: Agent that passed review.
        """
        ...

    def on_review_retry(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        error_count: int | None = None,
        parse_error: str | None = None,
    ) -> None:
        """Called when retrying a Codex review after issues found.

        Args:
            agent_id: Agent being retried.
            attempt: Current retry attempt number.
            max_attempts: Maximum review attempts.
            error_count: Number of errors found (if available).
            parse_error: Parse error message (if review failed to parse).
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

    def on_validation_result(
        self,
        agent_id: str,
        passed: bool,
    ) -> None:
        """Called when per-issue validation completes.

        Args:
            agent_id: Agent that was validated.
            passed: Whether validation passed.
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

    def on_run_level_validation_disabled(self) -> None:
        """Called when run-level validation is disabled."""
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


class NullEventSink:
    """No-op event sink for testing.

    All methods are implemented as no-ops, allowing tests to run the
    orchestrator without producing console output or other side effects.

    Example:
        sink = NullEventSink()
        orchestrator = MalaOrchestrator(..., event_sink=sink)
        await orchestrator.run()  # No console output
    """

    def on_run_started(self, config: EventRunConfig) -> None:
        pass

    def on_run_completed(
        self,
        success_count: int,
        total_count: int,
        run_validation_passed: bool,
        abort_reason: str | None = None,
    ) -> None:
        pass

    def on_ready_issues(self, issue_ids: list[str]) -> None:
        pass

    def on_waiting_for_agents(self, count: int) -> None:
        pass

    def on_no_more_issues(self, reason: str) -> None:
        pass

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

    def on_gate_started(
        self,
        agent_id: str | None,
        attempt: int,
        max_attempts: int,
    ) -> None:
        pass

    def on_gate_passed(self, agent_id: str | None) -> None:
        pass

    def on_gate_failed(
        self,
        agent_id: str | None,
        attempt: int,
        max_attempts: int,
    ) -> None:
        pass

    def on_gate_retry(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
    ) -> None:
        pass

    def on_gate_result(
        self,
        agent_id: str | None,
        passed: bool,
        failure_reasons: list[str] | None = None,
    ) -> None:
        pass

    def on_review_started(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
    ) -> None:
        pass

    def on_review_passed(self, agent_id: str) -> None:
        pass

    def on_review_retry(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        error_count: int | None = None,
        parse_error: str | None = None,
    ) -> None:
        pass

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

    def on_validation_result(
        self,
        agent_id: str,
        passed: bool,
    ) -> None:
        pass

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

    def on_run_level_validation_disabled(self) -> None:
        pass

    def on_abort_requested(self, reason: str) -> None:
        pass

    def on_tasks_aborting(self, count: int, reason: str) -> None:
        pass

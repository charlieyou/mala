"""Event sink protocol for MalaOrchestrator.

Defines the MalaEventSink protocol to decouple orchestration logic from
presentation concerns. The orchestrator emits semantic events through the
sink, allowing different implementations for:
- Console output (ConsoleEventSink - follow-up task)
- Testing (NullEventSink - no side effects)
- Structured logging (future: JSON/file-based sinks)

This module defines the protocol and provides NullEventSink for testing.
"""

import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .log_output.console import (
    Colors,
    log,
    log_agent_text,
    log_tool,
    log_verbose,
    truncate_text,
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
    braintrust_enabled: bool = False
    braintrust_disabled_reason: str | None = None  # e.g., "add BRAINTRUST_API_KEY..."
    review_enabled: bool = True  # Cerberus code review enabled
    review_disabled_reason: str | None = None
    morph_enabled: bool = True
    morph_disallowed_tools: list[str] | None = None
    morph_disabled_reason: str | None = (
        None  # e.g., "--no-morph", "MORPH_API_KEY not set"
    )
    prioritize_wip: bool = False
    orphans_only: bool = False
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
        issue_id: str | None = None,
    ) -> None:
        """Called when a quality gate check begins.

        Args:
            agent_id: Agent ID (None for run-level gate).
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
            agent_id: Agent ID (None for run-level gate).
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
            agent_id: Agent ID (None for run-level gate).
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
            agent_id: Agent ID (None for run-level gate).
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
        """Called when per-issue validation begins.

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
        """Called when per-issue validation completes.

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

    def on_epic_verification_human_review(
        self,
        epic_id: str,
        reason: str,
        review_issue_id: str,
    ) -> None:
        """Called when epic verification requires human review.

        Args:
            epic_id: The epic requiring review.
            reason: Why human review is needed.
            review_issue_id: ID of the created review issue.
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


class BaseEventSink:
    """Base event sink with no-op implementations of all protocol methods.

    Provides default no-op implementations for all MalaEventSink protocol
    methods. Subclasses can override only the methods they need to handle.

    This eliminates the need to implement all 46 methods when creating a
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

    def on_run_level_validation_disabled(self) -> None:
        pass

    def on_abort_requested(self, reason: str) -> None:
        pass

    def on_tasks_aborting(self, count: int, reason: str) -> None:
        pass

    # -------------------------------------------------------------------------
    # Epic verification lifecycle
    # -------------------------------------------------------------------------

    def on_epic_verification_started(self, epic_id: str) -> None:
        pass

    def on_epic_verification_passed(self, epic_id: str, confidence: float) -> None:
        pass

    def on_epic_verification_failed(
        self,
        epic_id: str,
        unmet_count: int,
        remediation_ids: list[str],
    ) -> None:
        pass

    def on_epic_verification_human_review(
        self,
        epic_id: str,
        reason: str,
        review_issue_id: str,
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


class ConsoleEventSink(BaseEventSink):
    """Event sink that outputs to the console using existing log helpers.

    Implements all MalaEventSink methods, delegating to log(), log_tool(),
    and log_agent_text() from src/infra/io/log_output/console.py.

    Example:
        from src.orchestration.factory import create_orchestrator, OrchestratorDependencies

        sink = ConsoleEventSink()
        deps = OrchestratorDependencies(event_sink=sink)
        orchestrator = create_orchestrator(config, deps=deps)
        await orchestrator.run()  # Produces console output
    """

    # -------------------------------------------------------------------------
    # Run lifecycle
    # -------------------------------------------------------------------------

    def on_run_started(self, config: EventRunConfig) -> None:
        """Log run configuration at startup."""
        print()
        log("●", "mala orchestrator", Colors.MAGENTA)
        log("◐", f"repo: {config.repo_path}", Colors.MUTED)

        self._log_limits(config)
        self._log_review_config(config)
        self._log_filters(config)
        self._log_braintrust_config(config)
        self._log_morph_config(config)
        self._log_cli_args(config)
        print()

    def _log_limits(self, config: EventRunConfig) -> None:
        """Log limit-related settings (max-issues, max-agents, timeout, gate-retries)."""
        limit_str = (
            str(config.max_issues) if config.max_issues is not None else "unlimited"
        )
        agents_str = (
            str(config.max_agents) if config.max_agents is not None else "unlimited"
        )
        timeout_str = f"{config.timeout_minutes}m" if config.timeout_minutes else "none"
        log(
            "◐",
            f"max-agents: {agents_str}, timeout: {timeout_str}, max-issues: {limit_str}",
            Colors.MUTED,
        )
        log("◐", f"gate-retries: {config.max_gate_retries}", Colors.MUTED)

    def _log_review_config(self, config: EventRunConfig) -> None:
        """Log Cerberus review configuration."""
        if config.review_enabled:
            log(
                "◐",
                f"review: enabled (max-retries: {config.max_review_retries})",
                Colors.CYAN,
            )
        elif config.review_disabled_reason:
            log(
                "◐",
                f"review: disabled ({config.review_disabled_reason})",
                Colors.YELLOW,
            )

    def _log_filters(self, config: EventRunConfig) -> None:
        """Log filter settings (epic_id, only_ids, prioritize_wip, orphans_only)."""
        if config.epic_id:
            log("◐", f"epic filter: {config.epic_id}", Colors.CYAN)
        if config.only_ids:
            log(
                "◐",
                f"only processing: {', '.join(sorted(config.only_ids))}",
                Colors.CYAN,
            )
        if config.prioritize_wip:
            log("◐", "wip: prioritizing in_progress issues", Colors.CYAN)
        if config.orphans_only:
            log(
                "◐",
                "orphans-only: processing only issues without parent epic",
                Colors.CYAN,
            )

    def _log_braintrust_config(self, config: EventRunConfig) -> None:
        """Log Braintrust configuration."""
        if config.braintrust_enabled:
            log("◐", "braintrust: enabled (LLM spans auto-traced)", Colors.CYAN)
        elif config.braintrust_disabled_reason:
            log(
                "◐",
                f"braintrust: disabled ({config.braintrust_disabled_reason})",
                Colors.MUTED,
            )

    def _log_morph_config(self, config: EventRunConfig) -> None:
        """Log Morph configuration."""
        if config.morph_enabled:
            log(
                "◐",
                "morph: enabled (edit_file, warpgrep_codebase_search)",
                Colors.CYAN,
            )
            if config.morph_disallowed_tools:
                log(
                    "◐",
                    f"morph: blocked tools: {', '.join(config.morph_disallowed_tools)}",
                    Colors.MUTED,
                )
        elif config.morph_disabled_reason:
            log("◐", f"morph: disabled ({config.morph_disabled_reason})", Colors.MUTED)

    def _log_cli_args(self, config: EventRunConfig) -> None:
        """Log CLI argument reconstruction."""
        if not config.cli_args:
            return
        cli_parts = []
        if config.cli_args.get("disable_validations"):
            cli_parts.append(
                f"--disable-validations={config.cli_args['disable_validations']}"
            )
        if config.cli_args.get("coverage_threshold") is not None:
            cli_parts.append(
                f"--coverage-threshold={config.cli_args['coverage_threshold']}"
            )
        if config.cli_args.get("wip"):
            cli_parts.append("--wip")
        if config.cli_args.get("max_issues") is not None:
            cli_parts.append(f"--max-issues={config.cli_args['max_issues']}")
        if config.cli_args.get("max_gate_retries") is not None:
            cli_parts.append(
                f"--max-gate-retries={config.cli_args['max_gate_retries']}"
            )
        if config.cli_args.get("max_review_retries") is not None:
            cli_parts.append(
                f"--max-review-retries={config.cli_args['max_review_retries']}"
            )
        if config.cli_args.get("review_timeout") is not None:
            cli_parts.append(f"--review-timeout={config.cli_args['review_timeout']}")
        spawn_args = config.cli_args.get("cerberus_spawn_args")
        if isinstance(spawn_args, list) and spawn_args:
            spawn_strs = [arg for arg in spawn_args if isinstance(arg, str)]
            if spawn_strs:
                cli_parts.append(f"--cerberus-spawn-args={' '.join(spawn_strs)}")
        wait_args = config.cli_args.get("cerberus_wait_args")
        if isinstance(wait_args, list) and wait_args:
            wait_strs = [arg for arg in wait_args if isinstance(arg, str)]
            if wait_strs:
                cli_parts.append(f"--cerberus-wait-args={' '.join(wait_strs)}")
        env_vars = config.cli_args.get("cerberus_env")
        if isinstance(env_vars, dict) and env_vars:
            env_str = ",".join(f"{key}={value}" for key, value in env_vars.items())
            cli_parts.append(f"--cerberus-env={env_str}")
        if config.cli_args.get("braintrust"):
            cli_parts.append("--braintrust")
        if cli_parts:
            log("◐", f"cli: {' '.join(cli_parts)}", Colors.MUTED)

    def on_run_completed(
        self,
        success_count: int,
        total_count: int,
        run_validation_passed: bool,
        abort_reason: str | None = None,
    ) -> None:
        """Log run completion summary."""
        print()
        if abort_reason:
            log("○", f"Run aborted: {abort_reason}", Colors.RED)
        elif success_count == total_count and total_count > 0 and run_validation_passed:
            log("●", f"Completed: {success_count}/{total_count} issues", Colors.GREEN)
        elif success_count > 0:
            if not run_validation_passed:
                log(
                    "◐",
                    f"Completed: {success_count}/{total_count} issues (Gate 4 failed)",
                    Colors.YELLOW,
                )
            else:
                log(
                    "◐",
                    f"Completed: {success_count}/{total_count} issues",
                    Colors.YELLOW,
                )
        elif total_count == 0:
            log("○", "No issues to process", Colors.GRAY)
        else:
            log("○", f"Completed: {success_count}/{total_count} issues", Colors.RED)

    def on_ready_issues(self, issue_ids: list[str]) -> None:
        """Log list of ready issues."""
        if issue_ids:
            log("◌", f"Ready issues: {', '.join(issue_ids)}", Colors.MUTED)

    def on_waiting_for_agents(self, count: int) -> None:
        """Log when waiting for agents to complete."""
        log("◌", f"Waiting for {count} agent(s)...", Colors.MUTED)

    def on_no_more_issues(self, reason: str) -> None:
        """Log when no more issues are available."""
        if reason.startswith("limit_reached"):
            # Extract limit from reason like "limit_reached (5)"
            log("○", f"Issue {reason.replace('_', ' ')}", Colors.GRAY)
        else:
            log("○", "No more issues to process", Colors.GRAY)

    # -------------------------------------------------------------------------
    # Agent lifecycle
    # -------------------------------------------------------------------------

    def on_agent_started(self, agent_id: str, issue_id: str) -> None:
        """Log agent spawn."""
        log("▶", "Agent started", Colors.BLUE, agent_id=issue_id)

    def on_agent_completed(
        self,
        agent_id: str,
        issue_id: str,
        success: bool,
        duration_seconds: float,
        summary: str,
    ) -> None:
        """Log agent completion."""
        status = "✓" if success else "✗"
        color = Colors.GREEN if success else Colors.RED
        duration_str = f"{duration_seconds:.1f}s"
        log(
            status,
            f"Agent completed ({duration_str}): {summary}",
            color,
            agent_id=agent_id,
        )

    def on_claim_failed(self, agent_id: str, issue_id: str) -> None:
        """Log when claiming an issue fails."""
        log("⚠", f"Failed to claim {issue_id}", Colors.YELLOW, agent_id=issue_id)

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
        """Log tool usage."""
        log_tool(tool_name, description, agent_id=agent_id, arguments=arguments)

    def on_agent_text(self, agent_id: str, text: str) -> None:
        """Log agent text output."""
        log_agent_text(text, agent_id)

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
        """Log quality gate check start."""
        scope = "run-level " if agent_id is None else ""
        log(
            "→",
            f"Quality gate {scope}check (attempt {attempt}/{max_attempts})",
            Colors.MUTED,
            agent_id=agent_id,
            issue_id=issue_id,
        )

    def on_gate_passed(
        self,
        agent_id: str | None,
        issue_id: str | None = None,
    ) -> None:
        """Log quality gate passed."""
        scope = "Run-level gate" if agent_id is None else "Gate"
        log("✓", f"{scope} passed", Colors.GREEN, agent_id=agent_id, issue_id=issue_id)

    def on_gate_failed(
        self,
        agent_id: str | None,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        """Log quality gate failed after all retries.

        Args:
            agent_id: Agent ID (None for run-level gate).
            attempt: Final attempt number.
            max_attempts: Maximum retry attempts.
            issue_id: Issue being validated (for display).
        """
        scope = "Run-level gate" if agent_id is None else "Gate"
        log(
            "✗",
            f"{scope} failed after {attempt}/{max_attempts} attempts",
            Colors.RED,
            agent_id=agent_id,
            issue_id=issue_id,
        )

    def on_gate_retry(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        """Log quality gate retry."""
        log(
            "↻",
            f"Retrying gate (attempt {attempt}/{max_attempts})",
            Colors.YELLOW,
            agent_id=agent_id,
            issue_id=issue_id,
        )

    def on_gate_result(
        self,
        agent_id: str | None,
        passed: bool,
        failure_reasons: list[str] | None = None,
        issue_id: str | None = None,
    ) -> None:
        """Log detailed gate result.

        For failures, logs the failure reasons at normal log level so users
        can see what went wrong without needing verbose mode.

        Args:
            agent_id: Agent ID (None for run-level gate).
            passed: Whether the gate passed.
            failure_reasons: List of failure reasons (if failed).
            issue_id: Issue being validated (for display).
        """
        if passed:
            return  # on_gate_passed already handles success
        if failure_reasons:
            # Log failure reasons at normal level (not verbose) so users can
            # see what went wrong - this matches the previous inline logging
            reasons_str = "; ".join(failure_reasons)
            log(
                "  ",
                f"Failures: {reasons_str}",
                Colors.RED,
                agent_id=agent_id,
                issue_id=issue_id,
            )

    # -------------------------------------------------------------------------
    # Review events
    # -------------------------------------------------------------------------

    def on_review_started(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        """Log review start."""
        log(
            "→",
            f"Review (attempt {attempt}/{max_attempts})",
            Colors.MUTED,
            agent_id=agent_id,
            issue_id=issue_id,
        )

    def on_review_passed(
        self,
        agent_id: str,
        issue_id: str | None = None,
    ) -> None:
        """Log review passed."""
        log("✓", "Review passed", Colors.GREEN, agent_id=agent_id, issue_id=issue_id)

    def on_review_retry(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        error_count: int | None = None,
        parse_error: str | None = None,
        issue_id: str | None = None,
    ) -> None:
        """Log review retry."""
        if parse_error:
            detail = f"parse error: {parse_error}"
        elif error_count is not None:
            detail = f"{error_count} errors found"
        else:
            detail = "issues found"
        log(
            "↻",
            f"Retrying review ({detail}, attempt {attempt}/{max_attempts})",
            Colors.YELLOW,
            agent_id=agent_id,
            issue_id=issue_id,
        )

    def on_review_warning(
        self,
        message: str,
        agent_id: str | None = None,
        issue_id: str | None = None,
    ) -> None:
        """Log review-related warning."""
        log("!", message, Colors.YELLOW, agent_id=agent_id, issue_id=issue_id)

    # -------------------------------------------------------------------------
    # Fixer agent events
    # -------------------------------------------------------------------------

    def on_fixer_started(
        self,
        attempt: int,
        max_attempts: int,
    ) -> None:
        """Log fixer agent spawn."""
        log(
            "▶", f"Spawning fixer agent (attempt {attempt}/{max_attempts})", Colors.CYAN
        )

    def on_fixer_completed(self, result: str) -> None:
        """Log fixer agent completion."""
        log("✓", f"Fixer completed: {truncate_text(result, 50)}", Colors.GREEN)

    def on_fixer_failed(self, reason: str) -> None:
        """Log fixer agent failure."""
        log("✗", f"Fixer failed: {reason}", Colors.RED)

    # -------------------------------------------------------------------------
    # Issue lifecycle
    # -------------------------------------------------------------------------

    def on_issue_closed(self, agent_id: str, issue_id: str) -> None:
        """Log issue closure."""
        log("✓", f"Closed {issue_id}", Colors.GREEN, agent_id=agent_id)

    def on_issue_completed(
        self,
        agent_id: str,
        issue_id: str,
        success: bool,
        duration_seconds: float,
        summary: str,
    ) -> None:
        """Log issue completion."""
        status = "✓" if success else "✗"
        color = Colors.GREEN if success else Colors.RED
        duration_str = f"{duration_seconds:.1f}s"
        log(
            status,
            f"Issue {issue_id} completed ({duration_str}): {summary}",
            color,
            agent_id=agent_id,
        )

    def on_epic_closed(self, agent_id: str) -> None:
        """Log parent epic auto-closure."""
        log("✓", "Parent epic auto-closed", Colors.GREEN, agent_id=agent_id)

    def on_validation_started(
        self,
        agent_id: str,
        issue_id: str | None = None,
    ) -> None:
        """Log validation start."""
        log(
            "◐",
            "Starting validation...",
            Colors.MUTED,
            agent_id=agent_id,
            issue_id=issue_id,
        )

    def on_validation_result(
        self,
        agent_id: str,
        passed: bool,
        issue_id: str | None = None,
    ) -> None:
        """Log per-issue validation result."""
        if passed:
            log(
                "✓",
                "Validation passed",
                Colors.GREEN,
                agent_id=agent_id,
                issue_id=issue_id,
            )
        else:
            log(
                "✗",
                "Validation failed",
                Colors.RED,
                agent_id=agent_id,
                issue_id=issue_id,
            )

    def on_validation_step_running(
        self,
        step_name: str,
        agent_id: str | None = None,
    ) -> None:
        """Log validation step start."""
        log("▸", f"{step_name} running...", Colors.CYAN, agent_id=agent_id)

    def on_validation_step_skipped(
        self,
        step_name: str,
        reason: str,
        agent_id: str | None = None,
    ) -> None:
        """Log validation step skipped."""
        log("○", f"{step_name} skipped: {reason}", Colors.CYAN, agent_id=agent_id)

    def on_validation_step_passed(
        self,
        step_name: str,
        duration_seconds: float,
        agent_id: str | None = None,
    ) -> None:
        """Log validation step success."""
        log(
            "✓",
            f"{step_name} passed ({duration_seconds:.1f}s)",
            Colors.GREEN,
            agent_id=agent_id,
        )

    def on_validation_step_failed(
        self,
        step_name: str,
        exit_code: int,
        agent_id: str | None = None,
    ) -> None:
        """Log validation step failure."""
        log(
            "✗",
            f"{step_name} failed (exit {exit_code})",
            Colors.RED,
            agent_id=agent_id,
        )

    # -------------------------------------------------------------------------
    # Warnings and diagnostics
    # -------------------------------------------------------------------------

    def on_warning(self, message: str, agent_id: str | None = None) -> None:
        """Log warning condition."""
        log("!", message, Colors.YELLOW, agent_id=agent_id)

    def on_log_timeout(self, agent_id: str, log_path: str) -> None:
        """Log when waiting for a log file times out."""
        log("!", f"Log file not found: {log_path}", Colors.YELLOW, agent_id=agent_id)

    def on_locks_cleaned(self, agent_id: str, count: int) -> None:
        """Log stale lock cleanup."""
        log(
            "🧹",
            f"Cleaned {count} locks for {agent_id[:8]}",
            Colors.MUTED,
        )

    def on_locks_released(self, count: int) -> None:
        """Log remaining locks released at run end."""
        if count > 0:
            log("🧹", f"Released {count} remaining locks", Colors.MUTED)

    def on_issues_committed(self) -> None:
        """Log issues.jsonl commit."""
        log("◐", "Committed .beads/issues.jsonl", Colors.MUTED)

    def on_run_metadata_saved(self, path: str) -> None:
        """Log run metadata save."""
        log("◐", f"Run metadata: {path}", Colors.MUTED)

    def on_run_level_validation_disabled(self) -> None:
        """Log when run-level validation is disabled."""
        log_verbose("◦", "Run-level validation disabled", Colors.MUTED)

    def on_abort_requested(self, reason: str) -> None:
        """Log fatal error triggering run abort."""
        log("✗", f"Fatal error: {reason}. Aborting run.", Colors.RED)

    def on_tasks_aborting(self, count: int, reason: str) -> None:
        """Log when active tasks are being aborted."""
        log("✗", f"Aborting {count} active task(s): {reason}", Colors.RED)

    # -------------------------------------------------------------------------
    # Epic verification lifecycle
    # -------------------------------------------------------------------------

    def on_epic_verification_started(self, epic_id: str) -> None:
        """Log epic verification start."""
        log("🔍", f"Verifying epic {epic_id}", Colors.CYAN)

    def on_epic_verification_passed(self, epic_id: str, confidence: float) -> None:
        """Log epic verification passed."""
        log(
            "✓", f"Epic {epic_id} verified (confidence: {confidence:.0%})", Colors.GREEN
        )

    def on_epic_verification_failed(
        self,
        epic_id: str,
        unmet_count: int,
        remediation_ids: list[str],
    ) -> None:
        """Log epic verification failed with unmet criteria."""
        ids_str = ", ".join(remediation_ids) if remediation_ids else "none"
        log(
            "✗",
            f"Epic {epic_id} failed verification: {unmet_count} unmet criteria, remediation issues: [{ids_str}]",
            Colors.RED,
        )

    def on_epic_verification_human_review(
        self,
        epic_id: str,
        reason: str,
        review_issue_id: str,
    ) -> None:
        """Log epic flagged for human review."""
        log(
            "👁",
            f"Epic {epic_id} requires human review: {reason} (issue: {review_issue_id})",
            Colors.YELLOW,
        )

    def on_epic_remediation_created(
        self,
        epic_id: str,
        issue_id: str,
        criterion: str,
    ) -> None:
        """Log remediation issue creation."""
        # Sanitize and truncate criterion for display
        sanitized_criterion = re.sub(r"\s+", " ", criterion.strip())
        crit_display = (
            sanitized_criterion[:60] + "..."
            if len(sanitized_criterion) > 60
            else sanitized_criterion
        )
        log_verbose(
            "◐",
            f"Created remediation {issue_id} for epic {epic_id}: {crit_display}",
            Colors.MUTED,
        )

    # -------------------------------------------------------------------------
    # Pipeline module events
    # -------------------------------------------------------------------------

    def on_lifecycle_state(self, agent_id: str, state: str) -> None:
        """Log lifecycle state change (verbose)."""
        log_verbose("->", f"State: {state}", agent_id=agent_id)

    def on_log_waiting(self, agent_id: str) -> None:
        """Log waiting for session log (verbose only)."""
        log_verbose("o", "Waiting for session log...", Colors.MUTED, agent_id=agent_id)

    def on_log_ready(self, agent_id: str) -> None:
        """Log session log ready (verbose only)."""
        log_verbose("v", "Log file ready", Colors.GREEN, agent_id=agent_id)

    def on_review_skipped_no_progress(self, agent_id: str) -> None:
        """Log review skipped due to no progress."""
        log(
            "x",
            "Review skipped: No progress (commit unchanged, no working tree changes)",
            Colors.RED,
            agent_id=agent_id,
        )

    def on_fixer_text(self, attempt: int, text: str) -> None:
        """Log fixer agent text output."""
        log_agent_text(text, f"fixer-{attempt}")

    def on_fixer_tool_use(
        self,
        attempt: int,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> None:
        """Log fixer agent tool usage."""
        log_tool(tool_name, agent_id=f"fixer-{attempt}", arguments=arguments)


# Verify all event sinks implement MalaEventSink at import time
assert isinstance(BaseEventSink(), MalaEventSink)
assert isinstance(NullEventSink(), MalaEventSink)
assert isinstance(ConsoleEventSink(), MalaEventSink)

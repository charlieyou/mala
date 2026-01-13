"""Issue lifecycle and fixer event protocols.

This module defines protocols for issue lifecycle events, validation steps,
and fixer agent events.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class IssueLifecycleEvents(Protocol):
    """Protocol for issue lifecycle and fixer events.

    These events track issue completion, validation, and fixer agent operations.
    """

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

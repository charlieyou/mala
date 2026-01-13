"""Run lifecycle event protocols.

This module defines protocols for run-level events in the orchestrator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .dataclasses import EventRunConfig


@runtime_checkable
class RunLifecycleEvents(Protocol):
    """Protocol for run lifecycle events.

    These events track the overall orchestrator run from start to completion.
    """

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

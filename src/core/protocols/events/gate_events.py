"""Quality gate and review event protocols.

This module defines protocols for quality gate checks and code review events.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class GateEvents(Protocol):
    """Protocol for quality gate and code review events.

    These events track quality gate checks, retries, and Codex code reviews.
    """

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

    def on_review_skipped_no_progress(self, agent_id: str) -> None:
        """Called when review is skipped due to no progress.

        Args:
            agent_id: Agent whose review was skipped.
        """
        ...

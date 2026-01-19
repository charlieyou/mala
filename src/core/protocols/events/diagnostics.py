"""Diagnostics and system event protocols.

This module defines protocols for warnings, abort handling, SIGINT escalation,
epic verification, and pipeline state events.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.core.protocols.lifecycle import DeadlockInfoProtocol


@runtime_checkable
class DiagnosticsEvents(Protocol):
    """Protocol for diagnostics and system events.

    These events track warnings, abort handling, SIGINT escalation,
    epic verification, and pipeline module state.
    """

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

    def on_epic_verification_started(
        self, epic_id: str, *, reviewer_type: str = "agent_sdk"
    ) -> None:
        """Called when epic verification begins.

        Args:
            epic_id: The epic being verified.
            reviewer_type: Type of reviewer ('agent_sdk' or 'cerberus').
        """
        ...

    def on_epic_verification_passed(
        self, epic_id: str, *, reviewer_type: str = "agent_sdk"
    ) -> None:
        """Called when epic verification passes.

        Args:
            epic_id: The epic that passed verification.
            reviewer_type: Type of reviewer ('agent_sdk' or 'cerberus').
        """
        ...

    def on_epic_verification_failed(
        self,
        epic_id: str,
        unmet_count: int,
        remediation_ids: list[str],
        *,
        reason: str | None = None,
        reviewer_type: str = "agent_sdk",
    ) -> None:
        """Called when epic verification fails with unmet criteria.

        Args:
            epic_id: The epic that failed verification.
            unmet_count: Number of unmet criteria.
            remediation_ids: IDs of created remediation issues.
            reason: Optional reason for failure (e.g., when verification couldn't run).
            reviewer_type: Type of reviewer ('agent_sdk' or 'cerberus').
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

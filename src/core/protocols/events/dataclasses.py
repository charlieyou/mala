"""Event dataclasses for orchestrator event handling.

This module defines dataclasses used by the event sink protocols.
"""

from __future__ import annotations

from dataclasses import dataclass


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

"""Run configuration builders for MalaOrchestrator.

This module contains helper functions for building run configuration
objects used during orchestrator execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.core.protocols.events import (
    EventRunConfig,
    TriggerSummary,
    ValidationTriggersSummary,
)
from src.infra.io.log_output.run_metadata import RunConfig, RunMetadata

if TYPE_CHECKING:
    from pathlib import Path

    from src.domain.validation.config import ValidationTriggersConfig


def _build_trigger_summary(
    triggers_config: ValidationTriggersConfig | None,
) -> ValidationTriggersSummary | None:
    """Build a trigger summary from ValidationTriggersConfig.

    Args:
        triggers_config: The validation triggers configuration, or None.

    Returns:
        ValidationTriggersSummary if triggers are configured, None otherwise.
    """
    if triggers_config is None:
        return None

    def make_summary(
        trigger: object | None,
    ) -> TriggerSummary | None:
        if trigger is None:
            return None
        # Access attributes dynamically to avoid importing trigger config types
        failure_mode = getattr(trigger, "failure_mode", None)
        commands = getattr(trigger, "commands", ())
        # Safely extract failure mode value - handles both Enum and string types
        mode_value = (
            getattr(failure_mode, "value", failure_mode) if failure_mode else None
        )
        # Extract command names for verbose logging
        command_names = tuple(getattr(cmd, "ref", str(cmd)) for cmd in commands)
        return TriggerSummary(
            enabled=True,
            failure_mode=mode_value,
            command_count=len(commands),
            command_names=command_names,
        )

    return ValidationTriggersSummary(
        epic_completion=make_summary(triggers_config.epic_completion),
        session_end=make_summary(triggers_config.session_end),
        periodic=make_summary(triggers_config.periodic),
        run_end=make_summary(triggers_config.run_end),
    )


def build_event_run_config(
    repo_path: Path,
    max_agents: int | None,
    timeout_seconds: int | None,
    max_issues: int | None,
    max_gate_retries: int,
    max_review_retries: int,
    epic_id: str | None,
    only_ids: list[str] | None,
    review_enabled: bool,
    review_disabled_reason: str | None,
    include_wip: bool,
    orphans_only: bool,
    cli_args: dict[str, object] | None,
    validation_triggers: ValidationTriggersConfig | None = None,
) -> EventRunConfig:
    """Build EventRunConfig for on_run_started event.

    Args:
        repo_path: Path to the repository.
        max_agents: Maximum concurrent agents.
        timeout_seconds: Timeout per agent in seconds.
        max_issues: Maximum issues to process.
        max_gate_retries: Maximum quality gate retry attempts.
        max_review_retries: Maximum code review retry attempts.
        epic_id: Epic ID filter.
        only_ids: List of issue IDs to process exclusively.
        review_enabled: Whether code review is enabled.
        review_disabled_reason: Reason review is disabled (if any).
        include_wip: Whether to include in-progress issues in scope.
        orphans_only: Whether to only process issues without parent epic.
        cli_args: CLI arguments for logging.
        validation_triggers: Validation triggers configuration from mala.yaml.

    Returns:
        EventRunConfig for the run.
    """
    return EventRunConfig(
        repo_path=str(repo_path),
        max_agents=max_agents,
        timeout_minutes=timeout_seconds // 60 if timeout_seconds else None,
        max_issues=max_issues,
        max_gate_retries=max_gate_retries,
        max_review_retries=max_review_retries,
        epic_id=epic_id,
        only_ids=only_ids,
        review_enabled=review_enabled,
        review_disabled_reason=review_disabled_reason,
        include_wip=include_wip,
        orphans_only=orphans_only,
        cli_args=cli_args,
        validation_triggers=_build_trigger_summary(validation_triggers),
    )


def build_run_metadata(
    repo_path: Path,
    max_agents: int | None,
    timeout_seconds: int | None,
    max_issues: int | None,
    epic_id: str | None,
    only_ids: list[str] | None,
    max_gate_retries: int,
    max_review_retries: int,
    review_enabled: bool,
    orphans_only: bool,
    cli_args: dict[str, object] | None,
    version: str,
    runs_dir: Path | None = None,
) -> RunMetadata:
    """Create run metadata tracker with current configuration.

    Args:
        repo_path: Path to the repository.
        max_agents: Maximum concurrent agents.
        timeout_seconds: Timeout per agent in seconds.
        max_issues: Maximum issues to process.
        epic_id: Epic ID filter.
        only_ids: List of issue IDs to process exclusively.
        max_gate_retries: Maximum quality gate retry attempts.
        max_review_retries: Maximum code review retry attempts.
        review_enabled: Whether code review is enabled.
        orphans_only: Whether to only process issues without parent epic.
        cli_args: CLI arguments for logging.
        version: Mala version string.
        runs_dir: Optional custom runs directory for test isolation.

    Returns:
        RunMetadata instance for the run.
    """
    run_config = RunConfig(
        max_agents=max_agents,
        timeout_minutes=timeout_seconds // 60 if timeout_seconds else None,
        max_issues=max_issues,
        epic_id=epic_id,
        only_ids=only_ids,
        max_gate_retries=max_gate_retries,
        max_review_retries=max_review_retries,
        review_enabled=review_enabled,
        orphans_only=orphans_only,
        cli_args=cli_args,
    )
    return RunMetadata(repo_path, run_config, version, runs_dir=runs_dir)

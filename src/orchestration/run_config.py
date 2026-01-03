"""Run configuration builders for MalaOrchestrator.

This module contains helper functions for building run configuration
objects used during orchestrator execution.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.infra.mcp import MORPH_DISALLOWED_TOOLS
from src.infra.tools.env import USER_CONFIG_DIR
from src.infra.io.event_protocol import EventRunConfig
from src.infra.io.log_output.run_metadata import RunConfig, RunMetadata

if TYPE_CHECKING:
    from pathlib import Path

    from src.infra.io.config import MalaConfig


def build_event_run_config(
    repo_path: Path,
    max_agents: int | None,
    timeout_seconds: int | None,
    max_issues: int | None,
    max_gate_retries: int,
    max_review_retries: int,
    epic_id: str | None,
    only_ids: set[str] | None,
    braintrust_enabled: bool,
    review_enabled: bool,
    review_disabled_reason: str | None,
    morph_enabled: bool,
    prioritize_wip: bool,
    orphans_only: bool,
    cli_args: dict[str, object] | None,
    mala_config: MalaConfig,
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
        only_ids: Set of issue IDs to process exclusively.
        braintrust_enabled: Whether Braintrust tracing is enabled.
        review_enabled: Whether code review is enabled.
        review_disabled_reason: Reason review is disabled (if any).
        morph_enabled: Whether MorphLLM routing is enabled.
        prioritize_wip: Whether to prioritize in-progress issues.
        orphans_only: Whether to only process issues without parent epic.
        cli_args: CLI arguments for logging.
        mala_config: MalaConfig for API key checks.

    Returns:
        EventRunConfig for the run.
    """
    # Compute morph disabled reason
    morph_disabled_reason: str | None = None
    if not morph_enabled:
        if cli_args and cli_args.get("no_morph"):
            morph_disabled_reason = "--no-morph"
        elif not mala_config.morph_api_key:
            morph_disabled_reason = "MORPH_API_KEY not set"
        else:
            morph_disabled_reason = "disabled by config"

    # Compute braintrust disabled reason
    braintrust_disabled_reason: str | None = None
    if not braintrust_enabled:
        braintrust_disabled_reason = f"add BRAINTRUST_API_KEY to {USER_CONFIG_DIR}/.env"

    return EventRunConfig(
        repo_path=str(repo_path),
        max_agents=max_agents,
        timeout_minutes=timeout_seconds // 60 if timeout_seconds else None,
        max_issues=max_issues,
        max_gate_retries=max_gate_retries,
        max_review_retries=max_review_retries,
        epic_id=epic_id,
        only_ids=list(only_ids) if only_ids else None,
        braintrust_enabled=braintrust_enabled,
        braintrust_disabled_reason=braintrust_disabled_reason,
        review_enabled=review_enabled,
        review_disabled_reason=review_disabled_reason,
        morph_enabled=morph_enabled,
        morph_disallowed_tools=list(MORPH_DISALLOWED_TOOLS) if morph_enabled else None,
        morph_disabled_reason=morph_disabled_reason,
        prioritize_wip=prioritize_wip,
        orphans_only=orphans_only,
        cli_args=cli_args,
    )


def build_run_metadata(
    repo_path: Path,
    max_agents: int | None,
    timeout_seconds: int | None,
    max_issues: int | None,
    epic_id: str | None,
    only_ids: set[str] | None,
    braintrust_enabled: bool,
    max_gate_retries: int,
    max_review_retries: int,
    review_enabled: bool,
    orphans_only: bool,
    cli_args: dict[str, object] | None,
    version: str,
) -> RunMetadata:
    """Create run metadata tracker with current configuration.

    Args:
        repo_path: Path to the repository.
        max_agents: Maximum concurrent agents.
        timeout_seconds: Timeout per agent in seconds.
        max_issues: Maximum issues to process.
        epic_id: Epic ID filter.
        only_ids: Set of issue IDs to process exclusively.
        braintrust_enabled: Whether Braintrust tracing is enabled.
        max_gate_retries: Maximum quality gate retry attempts.
        max_review_retries: Maximum code review retry attempts.
        review_enabled: Whether code review is enabled.
        orphans_only: Whether to only process issues without parent epic.
        cli_args: CLI arguments for logging.
        version: Mala version string.

    Returns:
        RunMetadata instance for the run.
    """
    run_config = RunConfig(
        max_agents=max_agents,
        timeout_minutes=timeout_seconds // 60 if timeout_seconds else None,
        max_issues=max_issues,
        epic_id=epic_id,
        only_ids=list(only_ids) if only_ids else None,
        braintrust_enabled=braintrust_enabled,
        max_gate_retries=max_gate_retries,
        max_review_retries=max_review_retries,
        review_enabled=review_enabled,
        orphans_only=orphans_only,
        cli_args=cli_args,
    )
    return RunMetadata(repo_path, run_config, version)

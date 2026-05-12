"""Run-command workflow helper.

Moves the post-validation run-command workflow out of the Typer CLI so the
CLI is left as an argument adapter plus output formatter. ``execute_run_command``
resolves the ``MalaConfig``, builds the ``OrchestratorConfig``, runs the
orchestrator, and returns a typed outcome the CLI converts into ``typer.Exit``.

The helper raises no Typer dependency and surfaces resolvable errors via
``RunCommandOutcome.error_message`` so the CLI is responsible for rendering
them (import-linter contract ``Only CLI imports typer``).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.models import WatchConfig
from src.domain.validation.config_loader import ConfigMissingError
from src.domain.validation.config_types import ConfigError
from src.orchestration import factory
from src.orchestration.cli_overrides import build_resolved_mala_config
from src.orchestration.dry_run import compute_dry_run_outcome
from src.orchestration.types import OrchestratorConfig

if TYPE_CHECKING:
    from pathlib import Path

    from src.orchestration.cli_options import OrderResolution, ScopeConfig
    from src.orchestration.cli_overrides import CLIOverrideOptions
    from src.orchestration.dry_run import DryRunOutcome


@dataclass(frozen=True)
class RunCommandOutcome:
    """Result of executing the run-command workflow.

    Attributes:
        exit_code: Exit code the CLI should propagate via ``typer.Exit``.
        error_message: Optional user-facing error message. When non-empty,
            the CLI renders it (e.g. via ``log("✗", ...)``) before exiting.
        dry_run: Populated when the CLI requested ``--dry-run``. Carries
            the resolved issue list and focus flag the CLI renders via
            ``display_dry_run_tasks`` before exiting; ``exit_code`` is 0.
    """

    exit_code: int
    error_message: str | None = None
    dry_run: DryRunOutcome | None = None


def execute_run_command(
    *,
    repo_path: Path,
    max_agents: int | None,
    timeout: int | None,
    max_issues: int | None,
    scope_config: ScopeConfig | None,
    order_resolution: OrderResolution,
    resume: bool,
    strict: bool,
    fresh: bool,
    dry_run: bool,
    watch: bool,
    override_options: CLIOverrideOptions,
    config_path: Path | None = None,
) -> RunCommandOutcome:
    """Build the orchestrator from CLI inputs and run it.

    Performs the workflow assembly previously embedded in the Typer
    command: resolves the ``MalaConfig`` via the override chain, builds
    the ``OrchestratorConfig``, creates the orchestrator, runs it under
    ``asyncio.run``, and computes the process exit code. When
    ``dry_run`` is true the helper short-circuits, computes a
    :class:`DryRunOutcome`, and returns it through
    :attr:`RunCommandOutcome.dry_run` for the CLI to render.

    Args:
        repo_path: Resolved repository path the run targets.
        max_agents: Optional concurrency cap.
        timeout: Optional per-agent timeout in minutes.
        max_issues: Optional cap on issues processed.
        scope_config: Parsed ``--scope`` value, or ``None`` for default.
        order_resolution: Resolved ``--order`` value carrying focus flag.
        resume: ``--resume`` flag (include_wip + session resume).
        strict: ``--strict`` flag.
        fresh: ``--fresh`` flag.
        dry_run: ``--dry-run`` flag; when true, returns the dry-run
            outcome without building/running the orchestrator.
        watch: ``--watch`` flag.
        override_options: Parsed CLI override options (coder, effort, etc.).
        config_path: Optional explicit project config file path. Relative paths
            are resolved relative to the current working directory by the
            config loader. Ignored by the dry-run branch, which does not load
            project configuration.

    Returns:
        ``RunCommandOutcome`` with the exit code the CLI should propagate.
        On config resolution failure, ``error_message`` carries the message
        the CLI should render before exiting. On ``--dry-run``, ``dry_run``
        is populated for the CLI's display helper.
    """
    epic_id = scope_config.epic_id if scope_config else None
    only_ids = scope_config.ids if scope_config else None
    orphans_only = scope_config.scope_type == "orphans" if scope_config else False

    if dry_run:
        outcome = compute_dry_run_outcome(
            repo_path=repo_path,
            epic_id=epic_id,
            only_ids=only_ids,
            orphans_only=orphans_only,
            include_wip=resume,
            order_preference=order_resolution.preference,
        )
        return RunCommandOutcome(exit_code=0, dry_run=outcome)

    try:
        config = build_resolved_mala_config(
            repo_path,
            override_options,
            config_path=config_path,
        )
        orch_config = OrchestratorConfig(
            repo_path=repo_path,
            config_path=config_path,
            max_agents=max_agents,
            timeout_minutes=timeout,
            max_issues=max_issues,
            epic_id=epic_id,
            only_ids=only_ids,
            include_wip=resume,
            strict_resume=strict,
            fresh_session=fresh,
            focus=order_resolution.focus,
            order_preference=order_resolution.preference,
            cli_args={"wip": resume, "max_issues": max_issues, "watch": watch},
            orphans_only=orphans_only,
        )

        orchestrator = factory.create_orchestrator(orch_config, mala_config=config)
    except (ConfigMissingError, ValueError) as exc:
        return RunCommandOutcome(exit_code=1, error_message=str(exc))
    except ConfigError as exc:
        if config_path is None:
            raise
        return RunCommandOutcome(exit_code=1, error_message=str(exc))

    watch_config = WatchConfig(enabled=watch)
    success_count, total = asyncio.run(orchestrator.run(watch_config=watch_config))

    orch_exit = orchestrator.exit_code
    if orch_exit != 0:
        return RunCommandOutcome(exit_code=orch_exit)
    return RunCommandOutcome(exit_code=0 if success_count > 0 or total == 0 else 1)


__all__ = ["RunCommandOutcome", "execute_run_command"]

#!/usr/bin/env python3
# ruff: noqa: E402
"""
mala CLI: Agent SDK orchestrator for parallel issue processing.

Usage:
    mala run [OPTIONS] [REPO_PATH]
    mala epic-verify [OPTIONS] EPIC_ID [REPO_PATH]
    mala logs [list|sessions|show] [OPTIONS]
    mala clean
    mala status
"""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import questionary

from ..orchestration.cli_support import USER_CONFIG_DIR, get_runs_dir, load_user_env

if TYPE_CHECKING:
    from src.core.models import OrderPreference
    from src.infra.io.config import MalaConfig

# Bootstrap state: tracks whether bootstrap() has been called
_bootstrapped = False

# Lazy-loaded module cache for SDK-dependent imports
# These are populated on first access via __getattr__
_lazy_modules: dict[str, Any] = {}

# Names that are lazily loaded (for __getattr__ to handle)
_LAZY_NAMES = frozenset(
    {
        "MalaConfig",
        "MalaOrchestrator",
        "OrderPreference",
        "OrchestratorConfig",
        "WatchConfig",
        "create_issue_provider",
        "create_orchestrator",
        "get_all_locks",
        "get_lock_dir",
        "get_running_instances",
        "get_running_instances_for_dir",
    }
)


def _lazy(name: str) -> Any:  # noqa: ANN401
    """Access a lazy-loaded module attribute.

    This function provides access to SDK-dependent imports that are
    lazy-loaded via __getattr__. Using this avoids runtime imports
    inside command functions while still deferring SDK loading.
    """
    return getattr(sys.modules[__name__], name)


def bootstrap() -> None:
    """Initialize environment.

    Must be called before using any CLI commands or importing claude_agent_sdk.
    This function is idempotent - calling it multiple times has no additional effect.

    Side effects:
        - Loads environment variables from ~/.config/mala/.env
    """
    global _bootstrapped

    if _bootstrapped:
        return

    # Load environment variables early (needed for config)
    load_user_env()

    _bootstrapped = True


# Import modules that do NOT depend on claude_agent_sdk at module level
import asyncio
from datetime import datetime
from typing import Annotated, Never

import click.exceptions
import typer

from ..orchestration.cli_support import Colors, ConfigError, log, set_verbose
from .logs import logs_app

# SDK-dependent imports (MalaOrchestrator, get_lock_dir, run_metadata, etc.)
# are lazy-loaded via __getattr__ to ensure bootstrap() runs before claude_agent_sdk
# is imported. Access them as module attributes: MalaOrchestrator, create_issue_provider, etc.


def display_dry_run_tasks(
    issues: list[dict[str, object]],
    focus: bool,
) -> None:
    """Display task order for dry-run preview.

    Shows tasks in order with epic grouping when focus mode is enabled.

    Args:
        issues: List of issue dicts with id, title, priority, status, parent_epic.
        focus: If True, display with epic headers for grouped output.
    """
    if not issues:
        log("○", "No ready tasks found", Colors.GRAY)
        return

    print()
    log("◐", f"Dry run: {len(issues)} task(s) would be processed", Colors.CYAN)
    print()

    if focus:
        # Group by epic for display
        current_epic: str | None = None
        epic_count = 0
        task_in_epic = 0
        for issue in issues:
            epic = issue.get("parent_epic")
            # Ensure epic is a string or None
            epic_str: str | None = str(epic) if epic is not None else None
            if epic_str != current_epic:
                if current_epic is not None or (
                    current_epic is None and task_in_epic > 0
                ):
                    print()  # Add spacing between epics
                current_epic = epic_str
                epic_count += 1
                task_in_epic = 0
                if epic_str:
                    print(f"  {Colors.MAGENTA}▸ Epic: {epic_str}{Colors.RESET}")
                else:
                    print(f"  {Colors.GRAY}▸ (Orphan tasks){Colors.RESET}")

            task_in_epic += 1
            _print_task_line(issue, indent="    ")
    else:
        # Simple flat list - show epic per task since there are no group headers
        for issue in issues:
            _print_task_line(issue, indent="  ", show_epic=True)

    print()
    # Summary: count tasks per epic
    if focus:
        epic_counts: dict[str | None, int] = {}
        for issue in issues:
            epic = issue.get("parent_epic")
            epic_key: str | None = str(epic) if epic is not None else None
            epic_counts[epic_key] = epic_counts.get(epic_key, 0) + 1

        epic_summary = ", ".join(
            f"{epic or '(orphan)'}: {count}"
            for epic, count in sorted(
                epic_counts.items(), key=lambda x: (x[0] is None, x[0] or "")
            )
        )
        log("◐", f"By epic: {epic_summary}", Colors.MUTED)


def _print_task_line(
    issue: dict[str, object], indent: str = "  ", show_epic: bool = False
) -> None:
    """Print a single task line with ID, priority, title, and optionally epic."""
    issue_id = issue.get("id", "?")
    title = issue.get("title", "")
    priority = issue.get("priority")
    status = issue.get("status", "")
    parent_epic = issue.get("parent_epic")

    # Format priority badge
    prio_str = f"P{priority}" if priority is not None else "P?"

    # Status indicator
    status_indicator = ""
    if status == "in_progress":
        status_indicator = f" {Colors.YELLOW}(WIP){Colors.RESET}"

    # Epic indicator (shown in non-focus mode)
    epic_indicator = ""
    if show_epic and parent_epic:
        epic_indicator = f" {Colors.MUTED}({parent_epic}){Colors.RESET}"

    print(
        f"{indent}{Colors.CYAN}{issue_id}{Colors.RESET} "
        f"[{Colors.MUTED}{prio_str}{Colors.RESET}] "
        f"{title}{epic_indicator}{status_indicator}"
    )


@dataclass(frozen=True)
class ScopeConfig:
    """Parsed scope configuration from --scope option.

    Attributes:
        scope_type: The type of scope ('all', 'epic', 'orphans', 'ids').
        ids: List of issue IDs to process (from 'ids:' prefix). Order-preserving.
        epic_id: Epic ID to filter by (from 'epic:' prefix).
    """

    scope_type: str = "all"
    ids: list[str] | None = None
    epic_id: str | None = None


def parse_scope(scope: str) -> ScopeConfig:
    """Parse the --scope option value into a ScopeConfig.

    Supported formats:
        - 'all' (default): Process all issues
        - 'epic:<id>': Filter to issues under the specified epic
        - 'orphans': Process only orphan issues (no parent epic)
        - 'ids:<id1>,<id2>,...': Process specific IDs in order (deduplicates with warning)

    Args:
        scope: The scope string (e.g., 'ids:T-1,T-2').

    Returns:
        ScopeConfig with parsed values.

    Raises:
        typer.Exit: If scope format is invalid.
    """
    import typer

    scope = scope.strip()

    # Handle 'all' scope
    if scope == "all":
        return ScopeConfig(scope_type="all")

    # Handle 'orphans' scope
    if scope == "orphans":
        return ScopeConfig(scope_type="orphans")

    # Handle 'epic:<id>' scope
    if scope.startswith("epic:"):
        epic_id = scope[5:].strip()
        if not epic_id:
            log(
                "✗",
                "Invalid --scope: 'epic:' requires an epic ID. "
                "Valid formats: all, epic:<id>, orphans, ids:<id,...>",
                Colors.RED,
            )
            raise typer.Exit(1)
        return ScopeConfig(scope_type="epic", epic_id=epic_id)

    # Handle 'ids:<id,...>' scope
    if scope.startswith("ids:"):
        ids_str = scope[4:].strip()
        if not ids_str:
            log(
                "✗",
                "Invalid --scope: 'ids:' requires at least one ID. "
                "Valid formats: all, epic:<id>, orphans, ids:<id,...>",
                Colors.RED,
            )
            raise typer.Exit(1)

        # Parse comma-separated IDs, preserving order
        raw_ids = [id_.strip() for id_ in ids_str.split(",") if id_.strip()]
        if not raw_ids:
            log(
                "✗",
                "Invalid --scope: 'ids:' requires at least one ID. "
                "Valid formats: all, epic:<id>, orphans, ids:<id,...>",
                Colors.RED,
            )
            raise typer.Exit(1)

        # Deduplicate while preserving order, warn if duplicates found
        seen: set[str] = set()
        unique_ids: list[str] = []
        duplicates: list[str] = []
        for id_ in raw_ids:
            if id_ in seen:
                duplicates.append(id_)
            else:
                seen.add(id_)
                unique_ids.append(id_)

        if duplicates:
            log(
                "⚠",
                f"Duplicate IDs removed from --scope: {', '.join(duplicates)}",
                Colors.YELLOW,
            )

        return ScopeConfig(scope_type="ids", ids=unique_ids)

    # Unknown scope format
    log(
        "✗",
        f"Invalid --scope value: '{scope}'. "
        "Valid formats: all, epic:<id>, orphans, ids:<id,...>",
        Colors.RED,
    )
    raise typer.Exit(1)


def _build_cli_args_metadata(
    *,
    resume: bool,
    max_issues: int | None,
    watch: bool,
) -> dict[str, object]:
    """Build the cli_args metadata dictionary for logging and OrchestratorConfig.

    Args:
        resume: Whether WIP prioritization is enabled (--resume flag).
        max_issues: Maximum issues to process.
        watch: Whether watch mode is enabled.

    Returns:
        Dictionary of CLI arguments for logging/metadata.
    """
    return {
        "wip": resume,
        "max_issues": max_issues,
        "watch": watch,
    }


def _apply_claude_settings_sources_override(
    config: MalaConfig, claude_settings_sources: str | None
) -> MalaConfig:
    """Apply --claude-settings-sources override to MalaConfig."""
    if claude_settings_sources is None:
        return config

    from src.infra.io.config import parse_claude_settings_sources

    parsed = parse_claude_settings_sources(claude_settings_sources, source="CLI")
    if parsed is None:
        return config

    init_kwargs = {
        field.name: getattr(config, field.name)
        for field in fields(config)
        if field.init and field.name != "claude_settings_sources_init"
    }
    return _lazy("MalaConfig")(**init_kwargs, claude_settings_sources_init=parsed)


def _handle_dry_run(
    repo_path: Path,
    scope_config: ScopeConfig | None,
    resume: bool,
    order_preference: OrderPreference,
) -> Never:
    """Execute dry-run mode: display task order and exit.

    Args:
        repo_path: Path to the repository.
        scope_config: Parsed scope configuration (epic, ids, orphans, or all).
        resume: Whether to include WIP issues (--resume flag).
        order_preference: Issue ordering preference (OrderPreference enum).

    Raises:
        typer.Exit: Always exits with code 0.
    """
    epic_id = scope_config.epic_id if scope_config else None
    only_ids = scope_config.ids if scope_config else None
    orphans_only = scope_config.scope_type == "orphans" if scope_config else False
    focus = order_preference in (
        _lazy("OrderPreference").FOCUS,
        _lazy("OrderPreference").EPIC_PRIORITY,
    )

    async def _dry_run() -> None:
        beads = _lazy("create_issue_provider")(repo_path)
        issues = await beads.get_ready_issues_async(
            epic_id=epic_id,
            only_ids=only_ids,
            include_wip=resume,
            focus=focus,
            orphans_only=orphans_only,
            order_preference=order_preference,
        )
        display_dry_run_tasks(issues, focus=focus)

    asyncio.run(_dry_run())
    raise typer.Exit(0)


app = typer.Typer(
    name="mala",
    help="Parallel issue processing with Claude Agent SDK",
    add_completion=False,
)


@app.command()
def run(
    repo_path: Annotated[
        Path,
        typer.Argument(
            help="Path to repository with beads issues",
        ),
    ] = Path("."),
    max_agents: Annotated[
        int | None,
        typer.Option(
            "--max-agents",
            "-n",
            help="Maximum concurrent agents (default: unlimited)",
            rich_help_panel="Execution Limits",
        ),
    ] = None,
    timeout: Annotated[
        int | None,
        typer.Option(
            "--timeout",
            "-t",
            help="Timeout per agent in minutes (default: 60)",
            rich_help_panel="Execution Limits",
        ),
    ] = None,
    max_issues: Annotated[
        int | None,
        typer.Option(
            "--max-issues",
            "-i",
            help="Maximum issues to process (default: unlimited)",
            rich_help_panel="Execution Limits",
        ),
    ] = None,
    scope: Annotated[
        str | None,
        typer.Option(
            "--scope",
            "-s",
            help="Scope filter (e.g., 'ids:T-1,T-2'). Replaces --only.",
            rich_help_panel="Scope & Ordering",
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume",
            "-r",
            help="Include in_progress issues AND attempt to resume their prior Claude sessions",
            rich_help_panel="Scope & Ordering",
        ),
    ] = False,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict/--no-strict",
            help="Fail issues if no prior session found (requires --resume)",
            rich_help_panel="Scope & Ordering",
        ),
    ] = False,
    order: Annotated[
        str | None,
        typer.Option(
            "--order",
            help="Issue ordering: 'epic-priority' (default, epic-grouped), 'issue-priority' (global priority), 'focus' (single-epic), 'input' (preserve --scope ids: order)",
            rich_help_panel="Scope & Ordering",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            "-d",
            help="Preview task order without processing; shows what would be run",
            rich_help_panel="Debugging",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output; shows full tool arguments instead of single line per tool call",
            rich_help_panel="Debugging",
        ),
    ] = False,
    watch: Annotated[
        bool,
        typer.Option(
            "--watch",
            help="Keep running and poll for new issues instead of exiting when idle",
        ),
    ] = False,
    claude_settings_sources: Annotated[
        str | None,
        typer.Option(
            "--claude-settings-sources",
            help=(
                "Comma-separated list of Claude settings sources "
                "(local, project, user). Default: local,project"
            ),
            rich_help_panel="Claude Settings",
        ),
    ] = None,
) -> Never:
    """Run parallel issue processing."""
    # Apply verbose setting
    set_verbose(verbose)

    # Validate max_issues (Typer min= doesn't work well with Optional[int])
    if max_issues is not None and max_issues < 1:
        log("✗", "Error: --max-issues must be at least 1", Colors.RED)
        raise typer.Exit(2)

    # Validate --strict requires --resume
    if strict and not resume:
        raise typer.BadParameter("--strict requires --resume flag")

    repo_path = repo_path.resolve()

    # Parse --scope option
    scope_config: ScopeConfig | None = None
    if scope is not None:
        scope_config = parse_scope(scope)

    # Parse and validate --order option
    order_preference = _lazy("OrderPreference").EPIC_PRIORITY  # default
    if order is not None:
        order_lower = order.lower()
        if order_lower == "epic-priority":
            order_preference = _lazy("OrderPreference").EPIC_PRIORITY
        elif order_lower == "issue-priority":
            order_preference = _lazy("OrderPreference").ISSUE_PRIORITY
        elif order_lower == "focus":
            order_preference = _lazy("OrderPreference").FOCUS
        elif order_lower == "input":
            order_preference = _lazy("OrderPreference").INPUT
            # --order input requires --scope ids:
            if scope_config is None or scope_config.scope_type != "ids":
                log(
                    "✗",
                    "--order input requires --scope ids:<id,...>",
                    Colors.RED,
                )
                raise typer.Exit(1)
        else:
            log(
                "✗",
                f"Invalid --order value: '{order}'. Valid values: focus, epic-priority, issue-priority, input",
                Colors.RED,
            )
            raise typer.Exit(1)

    # Derive scope values from scope_config
    epic_id = scope_config.epic_id if scope_config else None
    only_ids = scope_config.ids if scope_config else None
    orphans_only = scope_config.scope_type == "orphans" if scope_config else False
    focus = order_preference in (
        _lazy("OrderPreference").FOCUS,
        _lazy("OrderPreference").EPIC_PRIORITY,
    )

    # Validate repo_path exists
    if not repo_path.exists():
        log("✗", f"Repository not found: {repo_path}", Colors.RED)
        raise typer.Exit(1)

    # Ensure user config directory exists
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Handle dry-run mode: display task order and exit
    if dry_run:
        _handle_dry_run(
            repo_path=repo_path,
            scope_config=scope_config,
            resume=resume,
            order_preference=order_preference,
        )

    # Build and configure MalaConfig from environment
    config = _lazy("MalaConfig").from_env(validate=False)

    # Apply claude settings sources override
    try:
        config = _apply_claude_settings_sources_override(
            config, claude_settings_sources
        )
    except ValueError as exc:
        log("✗", str(exc), Colors.RED)
        raise typer.Exit(1)

    # Build cli_args metadata for logging
    cli_args = _build_cli_args_metadata(
        resume=resume,
        max_issues=max_issues,
        watch=watch,
    )

    # Build OrchestratorConfig and run
    orch_config = _lazy("OrchestratorConfig")(
        repo_path=repo_path,
        max_agents=max_agents,
        timeout_minutes=timeout,
        max_issues=max_issues,
        epic_id=epic_id,
        only_ids=only_ids,
        include_wip=resume,
        strict_resume=strict,
        focus=focus,
        order_preference=order_preference,
        cli_args=cli_args,
        orphans_only=orphans_only,
    )

    orchestrator = _lazy("create_orchestrator")(
        orch_config,
        mala_config=config,
    )

    # Build WatchConfig, then run orchestrator
    watch_config = _lazy("WatchConfig")(enabled=watch)
    success_count, total = asyncio.run(orchestrator.run(watch_config=watch_config))

    # Determine exit code:
    # - Use orchestrator.exit_code for interrupt (130), validation failure (1), abort (3)
    # - Otherwise: exit 0 if no issues to process or at least one succeeded
    # - Exit 1 only if issues were processed but all failed
    orch_exit = orchestrator.exit_code
    if orch_exit != 0:
        raise typer.Exit(orch_exit)
    raise typer.Exit(0 if success_count > 0 or total == 0 else 1)


@app.command("epic-verify")
def epic_verify(
    epic_id: Annotated[
        str,
        typer.Argument(
            help="Epic ID to verify and optionally close",
        ),
    ],
    repo_path: Annotated[
        Path,
        typer.Argument(
            help="Path to repository with beads issues",
        ),
    ] = Path("."),
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Run verification even if epic is not eligible or already closed",
        ),
    ] = False,
    close: Annotated[
        bool,
        typer.Option(
            "--close/--no-close",
            help="Close the epic after passing verification (default: close)",
        ),
    ] = True,
    human_override: Annotated[
        bool,
        typer.Option(
            "--human-override",
            help="Bypass verification and close directly (requires --close)",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output",
        ),
    ] = False,
    claude_settings_sources: Annotated[
        str | None,
        typer.Option(
            "--claude-settings-sources",
            help=(
                "Comma-separated list of Claude settings sources "
                "(local, project, user). Default: local,project"
            ),
            rich_help_panel="Claude Settings",
        ),
    ] = None,
) -> None:
    """Verify a single epic and optionally close it without running tasks."""
    set_verbose(verbose)

    if human_override and not close:
        log("✗", "--human-override requires --close", Colors.RED)
        raise typer.Exit(1)

    repo_path = repo_path.resolve()
    if not repo_path.exists():
        log("✗", f"Repository not found: {repo_path}", Colors.RED)
        raise typer.Exit(1)

    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    config = _lazy("MalaConfig").from_env(validate=False)

    try:
        config = _apply_claude_settings_sources_override(
            config, claude_settings_sources
        )
    except ValueError as exc:
        log("✗", str(exc), Colors.RED)
        raise typer.Exit(1)

    orch_config = _lazy("OrchestratorConfig")(repo_path=repo_path)
    orchestrator = _lazy("create_orchestrator")(orch_config, mala_config=config)
    verifier = getattr(orchestrator, "epic_verifier", None)

    if verifier is None:
        log("✗", "Epic verifier unavailable for this configuration", Colors.RED)
        raise typer.Exit(1)

    result = asyncio.run(
        verifier.verify_epic_with_options(
            epic_id,
            human_override=human_override,
            require_eligible=not force,
            close_epic=close,
        )
    )

    if result.verified_count == 0:
        log("○", f"No verification run for epic {epic_id}", Colors.GRAY)
        if result.ineligibility_reason:
            log("◐", f"Reason: {result.ineligibility_reason}", Colors.MUTED)
        if not force:
            log("◐", "Use --force to bypass eligibility checks", Colors.MUTED)
        raise typer.Exit(1)

    if result.remediation_issues_created:
        log(
            "◐",
            f"Remediation issues: {', '.join(result.remediation_issues_created)}",
            Colors.MUTED,
        )

    if result.failed_count > 0:
        log("✗", f"Epic {epic_id} failed verification", Colors.RED)
        raise typer.Exit(1)

    if close:
        log("✓", f"Epic {epic_id} verified and closed", Colors.GREEN)
    else:
        log("✓", f"Epic {epic_id} verified (no close)", Colors.GREEN)
    raise typer.Exit(0)


@app.command()
def clean(
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Force cleanup even if a mala instance is running",
        ),
    ] = False,
) -> None:
    """Clean up lock files.

    Removes orphaned lock files from the lock directory. Use this when
    a previous run left stale locks behind (e.g., after a crash).

    If a mala instance is currently running, the command will exit early
    unless --force is specified.
    """
    # Check for running instances
    running = _lazy("get_running_instances")()
    if running and not force:
        log(
            "⚠",
            f"A mala instance is running (PID {running[0].pid}). "
            "Use --force to clean anyway.",
            Colors.YELLOW,
        )
        raise typer.Exit(1)

    cleaned_locks = 0

    lock_dir = _lazy("get_lock_dir")()
    if lock_dir.exists():
        for lock in lock_dir.glob("*.lock"):
            lock.unlink()
            cleaned_locks += 1

    if cleaned_locks:
        log("🧹", f"Removed {cleaned_locks} lock files", Colors.GREEN)
    else:
        log("○", "No lock files to clean", Colors.GRAY)


@app.command()
def status(
    all_instances: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Show all running instances across all directories",
        ),
    ] = False,
) -> None:
    """Show status of mala instance(s).

    By default, shows status only for mala running in the current directory.
    Use --all to show all running instances across all directories.
    """
    print()
    cwd = Path.cwd().resolve()

    # Get running instances (filtered by cwd unless --all)
    if all_instances:
        instances = _lazy("get_running_instances")()
    else:
        instances = _lazy("get_running_instances_for_dir")(cwd)

    if not instances:
        if all_instances:
            log("○", "No mala instances running", Colors.GRAY)
        else:
            log("○", "No mala instance running in this directory", Colors.GRAY)
            log("◐", f"cwd: {cwd}", Colors.MUTED)
            log("◐", "Use --all to show instances in other directories", Colors.MUTED)
        print()
        return

    # Display running instances
    if all_instances:
        log("●", f"{len(instances)} running instance(s)", Colors.MAGENTA)
        print()

        # Group by repo path for display with directory headings
        instances_by_repo: dict[Path, list[object]] = {}
        for instance in instances:
            repo = getattr(instance, "repo_path", Path("."))
            if repo not in instances_by_repo:
                instances_by_repo[repo] = []
            instances_by_repo[repo].append(instance)

        for repo_path, repo_instances in instances_by_repo.items():
            # Directory heading
            print(f"{Colors.CYAN}{repo_path}:{Colors.RESET}")
            for instance in repo_instances:
                _display_instance(instance, indent=True)
            print()
    else:
        # Single instance for this directory
        log("●", "mala running", Colors.MAGENTA)
        print()
        for instance in instances:
            _display_instance(instance)
        print()

    # Show config directory
    config_env = USER_CONFIG_DIR / ".env"
    if config_env.exists():
        log("◐", f"config: {config_env}", Colors.MUTED)
    else:
        log("○", f"config: {config_env} (not found)", Colors.MUTED)
    print()

    # Check locks (show all locks grouped by agent)
    locks_by_agent = _lazy("get_all_locks")()
    if locks_by_agent:
        total_locks = sum(len(files) for files in locks_by_agent.values())
        log(
            "⚠",
            f"{total_locks} active lock(s) held by {len(locks_by_agent)} agent(s)",
            Colors.YELLOW,
        )
        for agent_id, files in sorted(locks_by_agent.items()):
            print(f"    {Colors.CYAN}{agent_id}{Colors.RESET}")
            for filepath in sorted(files):
                print(f"      {Colors.MUTED}{filepath}{Colors.RESET}")
    else:
        log("○", "No active locks", Colors.GRAY)

    # Check run metadata (completed runs)
    if get_runs_dir().exists():
        # Use rglob to find run files in both legacy flat structure
        # and new repo-segmented subdirectories
        run_files = list(get_runs_dir().rglob("*.json"))
        if run_files:
            log(
                "◐",
                f"{len(run_files)} run metadata files in {get_runs_dir()}",
                Colors.MUTED,
            )
            recent = sorted(run_files, key=lambda p: p.stat().st_mtime, reverse=True)[
                :3
            ]
            for run_file in recent:
                mtime = datetime.fromtimestamp(run_file.stat().st_mtime).strftime(
                    "%H:%M:%S"
                )
                # Show relative path from runs_dir for clarity
                rel_path = run_file.relative_to(get_runs_dir())
                print(f"    {Colors.MUTED}{mtime} {rel_path}{Colors.RESET}")
    print()


def _is_interactive() -> bool:
    """Check if stdin is a TTY (interactive mode).

    This is a separate function to allow mocking in tests.
    """
    return sys.stdin.isatty()


def _write_with_backup(path: Path, content: str) -> None:
    """Backup existing file and write new content."""
    if path.exists():
        backup_path = path.with_suffix(".yaml.bak")
        shutil.copy(path, backup_path)
    path.write_text(content)


def _get_preset_command_names(preset_name: str) -> list[str]:
    """Get command names defined in a preset.

    Args:
        preset_name: Name of the preset (e.g., 'python-uv', 'go')

    Returns:
        List of command names defined in the preset.
    """
    from ..orchestration.cli_support import get_preset_config_commands

    return get_preset_config_commands(preset_name)


def _prompt_preset_selection(presets: list[str]) -> str | None:
    """Prompt user to select a preset using questionary.

    Args:
        presets: List of available preset names.

    Returns:
        Selected preset name, or None if user chooses custom.

    Raises:
        KeyboardInterrupt: If user cancels the prompt (Esc/Ctrl-C).
    """
    choices = [*presets, "custom"]
    result = questionary.select(
        "Select a preset:",
        choices=choices,
    ).ask()
    if result is None:
        # User cancelled the prompt (Esc/Ctrl-C)
        raise KeyboardInterrupt
    if result == "custom":
        return None
    return result


def _prompt_custom_commands_questionary() -> dict[str, str] | None:
    """Prompt user for custom validation commands using questionary.

    Returns:
        Dictionary of command name to command string (empty values omitted),
        or None if user cancels (Ctrl-C).
    """
    from ..orchestration.cli_support import get_builtin_command_names

    commands: dict[str, str] = {}
    for name in sorted(get_builtin_command_names()):
        result = questionary.text(f"Command for '{name}' (leave empty to skip):").ask()
        if result is None:
            return None  # User cancelled
        result = result.strip()
        if result:
            commands[name] = result
    return commands


def _compute_evidence_defaults(commands: list[str], is_preset: bool) -> list[str]:
    """Compute default evidence check commands.

    Args:
        commands: List of command names available.
        is_preset: True if using a preset, False for custom.

    Returns:
        List of command names that should be evidence checks by default.
    """
    if not is_preset:
        return []
    # For preset path, return intersection of {test, lint} with available commands
    evidence_candidates = {"test", "lint"}
    return [cmd for cmd in commands if cmd in evidence_candidates]


def _compute_trigger_defaults(commands: list[str], is_preset: bool) -> list[str]:
    """Compute default validation trigger commands.

    Args:
        commands: List of command names available.
        is_preset: True if using a preset, False for custom.

    Returns:
        List of command names that should be triggers by default.
    """
    if not is_preset:
        return []
    # For preset path, exclude setup and e2e from defaults
    excluded = {"setup", "e2e"}
    return [cmd for cmd in commands if cmd not in excluded]


def _prompt_evidence_check(commands: list[str], is_preset: bool) -> list[str] | None:
    """Prompt user to select evidence check commands.

    Args:
        commands: List of command names available.
        is_preset: True if using a preset, False for custom.

    Returns:
        List of selected command names, or None to skip.
    """
    defaults = _compute_evidence_defaults(commands, is_preset)

    # First confirm if user wants evidence checks
    if not questionary.confirm("Configure evidence checks?", default=True).ask():
        return None

    # Show checkbox for selection
    choices = [questionary.Choice(cmd, checked=(cmd in defaults)) for cmd in commands]
    selected = questionary.checkbox(
        "Select commands that must pass for evidence verification:",
        choices=choices,
    ).ask()

    # Handle "uncheck all" scenario
    if not selected:
        skip = questionary.confirm(
            "No commands selected. Skip this section?", default=True
        ).ask()
        if skip:
            return None
        # If user doesn't want to skip, return empty list (keep section but empty)
        return []

    return selected


def _prompt_run_end_trigger(commands: list[str], is_preset: bool) -> list[str] | None:
    """Prompt user to select run-end validation trigger commands.

    Args:
        commands: List of command names available.
        is_preset: True if using a preset, False for custom.

    Returns:
        List of selected command names, or None to skip.
    """
    defaults = _compute_trigger_defaults(commands, is_preset)

    # First confirm if user wants run-end triggers
    if not questionary.confirm(
        "Configure run-end validation triggers?", default=True
    ).ask():
        return None

    # Show checkbox for selection
    choices = [questionary.Choice(cmd, checked=(cmd in defaults)) for cmd in commands]
    selected = questionary.checkbox(
        "Select commands to run at the end of mala run:",
        choices=choices,
    ).ask()

    # Handle "uncheck all" scenario
    if not selected:
        skip = questionary.confirm(
            "No commands selected. Skip this section?", default=True
        ).ask()
        if skip:
            return None
        # If user doesn't want to skip, return empty list (keep section but empty)
        return []

    return selected


def _prompt_per_issue_review() -> dict[str, Any] | None:
    """Prompt user to configure per-issue code review settings.

    Returns:
        Dict with per_issue_review config, or None to skip.
    """
    # Main enable/disable prompt
    enable = questionary.select(
        "Enable per-issue code review? (Reviews code after each issue session)",
        choices=[
            questionary.Choice("No (faster, no review overhead)", value=False),
            questionary.Choice("Yes (review every issue session)", value=True),
        ],
    ).ask()

    if enable is None:  # User cancelled
        return None
    if not enable:
        return None

    # Show note about reviewer type scope
    typer.echo(
        "\nNote: This reviewer type will be used for ALL reviews, "
        "including trigger-based reviews.\n"
    )

    # Reviewer type prompt
    reviewer_type = questionary.select(
        "Reviewer type:",
        choices=[
            questionary.Choice("Cerberus (multi-model consensus)", value="cerberus"),
            questionary.Choice("Agent SDK (single model)", value="agent_sdk"),
        ],
    ).ask()

    if reviewer_type is None:  # User cancelled
        return None

    # Max retries prompt
    max_retries_str = questionary.text(
        "Maximum review retries:",
        default="3",
    ).ask()

    if max_retries_str is None:  # User cancelled
        return None

    try:
        max_retries = int(max_retries_str)
    except ValueError:
        max_retries = 3  # Fall back to default

    # Finding threshold prompt
    threshold = questionary.select(
        "Minimum finding priority to block (P0 = critical only, none = all findings):",
        choices=[
            questionary.Choice("none (report all)", value="none"),
            questionary.Choice("P3 (low and above)", value="P3"),
            questionary.Choice("P2 (medium and above)", value="P2"),
            questionary.Choice("P1 (high and above)", value="P1"),
            questionary.Choice("P0 (critical only)", value="P0"),
        ],
    ).ask()

    if threshold is None:  # User cancelled
        return None

    return {
        "enabled": True,
        "reviewer_type": reviewer_type,
        "max_retries": max_retries,
        "finding_threshold": threshold,
    }


def _build_per_issue_review_dict(config: dict[str, Any]) -> dict[str, Any]:
    """Build per_issue_review config dict from prompt selections.

    Args:
        config: Dict with enabled, reviewer_type, max_retries, finding_threshold.

    Returns:
        Dict suitable for YAML output.
    """
    return config


def _build_evidence_check_dict(required: list[str]) -> dict[str, list[str]]:
    """Build evidence_check config dict from selected commands.

    Args:
        required: List of command names that are required evidence.

    Returns:
        Dict with 'required' key mapping to the command list.
    """
    return {"required": required}


def _build_validation_triggers_dict(commands: list[str]) -> dict[str, Any]:
    """Build validation_triggers config dict from selected commands.

    Args:
        commands: List of command names for run-end triggers.

    Returns:
        Dict with trigger configuration.
    """
    return {
        "run_end": {
            "failure_mode": "continue",
            "commands": [{"ref": cmd} for cmd in commands],
        }
    }


def _print_trigger_reference_table() -> None:
    """Print a reference table showing available trigger types."""
    import sys

    from rich.console import Console
    from rich.table import Table

    console = Console(file=sys.stderr)
    table = Table(title="Available Trigger Types")
    table.add_column("Trigger", style="cyan")
    table.add_column("Description", style="white")

    table.add_row("epic_completion", "Run verification when an epic ends")
    table.add_row("session_end", "Run verification when a session ends")
    table.add_row("periodic", "Run verification after N issues complete")
    table.add_row("run_end", "Run verification when mala run completes")

    console.print(table)


@app.command()
def init(
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Preview without writing"),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Accept defaults (requires --preset)"),
    ] = False,
    preset: Annotated[
        str | None,
        typer.Option("--preset", "-p", help="Use preset"),
    ] = None,
    skip_evidence: Annotated[
        bool,
        typer.Option("--skip-evidence", help="Omit evidence_check"),
    ] = False,
    skip_triggers: Annotated[
        bool,
        typer.Option("--skip-triggers", help="Omit validation_triggers"),
    ] = False,
) -> None:
    """Initialize mala.yaml configuration interactively."""
    try:
        from ..orchestration.cli_support import (
            dump_config_yaml,
            get_init_presets,
            validate_init_config,
        )

        # TTY detection
        is_tty = _is_interactive()

        # Flag validation
        if yes and not preset:
            typer.echo("Error: --yes requires --preset", err=True)
            raise typer.Exit(1)

        # Non-TTY validation: needs --yes or (--preset + --skip-evidence + --skip-triggers)
        prompts_needed = not (skip_evidence and skip_triggers and preset)
        if not is_tty and not yes and prompts_needed:
            typer.echo(
                "Error: Non-interactive mode requires --yes with --preset, "
                "or --preset with --skip-evidence and --skip-triggers",
                err=True,
            )
            raise typer.Exit(1)

        presets = get_init_presets()
        is_preset = False
        commands: list[str] = []

        if preset:
            # Validate preset exists
            if preset not in presets:
                typer.echo(f"Error: Unknown preset '{preset}'", err=True)
                raise typer.Exit(1)
            is_preset = True
            commands = _get_preset_command_names(preset)
            config_data: dict[str, object] = {"preset": preset}
        elif is_tty:
            # Interactive preset selection
            selected_preset = _prompt_preset_selection(presets)
            if selected_preset:
                is_preset = True
                commands = _get_preset_command_names(selected_preset)
                config_data = {"preset": selected_preset}
            else:
                # Custom flow via questionary
                custom_commands = _prompt_custom_commands_questionary()
                if custom_commands is None:
                    raise typer.Exit(1)
                if custom_commands:
                    commands = list(custom_commands.keys())
                config_data = {"commands": custom_commands}
        else:
            # Should not reach here due to validation above
            typer.echo("Error: Cannot run interactively in non-TTY mode", err=True)
            raise typer.Exit(1)

        # If no commands, print message and skip evidence/trigger prompts
        if not commands:
            typer.echo("No commands defined; skipping evidence and trigger setup.")
        else:
            # Evidence check section
            evidence_required: list[str] | None = None
            if not skip_evidence:
                if yes:
                    # Use computed defaults
                    evidence_required = _compute_evidence_defaults(commands, is_preset)
                elif is_tty:
                    evidence_required = _prompt_evidence_check(commands, is_preset)

            if evidence_required is not None:
                config_data["evidence_check"] = _build_evidence_check_dict(
                    evidence_required
                )

            # Per-issue review section (within commands block to avoid prompting
            # when init will fail for empty commands)
            # Only prompt in interactive mode; --yes uses default (disabled)
            if is_tty and not yes:
                per_issue_review_config = _prompt_per_issue_review()
                if per_issue_review_config is not None:
                    config_data["per_issue_review"] = _build_per_issue_review_dict(
                        per_issue_review_config
                    )
            # Validation triggers section
            trigger_commands: list[str] | None = None
            if not skip_triggers:
                if yes:
                    # Use computed defaults
                    trigger_commands = _compute_trigger_defaults(commands, is_preset)
                elif is_tty:
                    trigger_commands = _prompt_run_end_trigger(commands, is_preset)

            if trigger_commands is not None:
                config_data["validation_triggers"] = _build_validation_triggers_dict(
                    trigger_commands
                )

        # Validate the generated config (max 3 retry attempts)
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                validate_init_config(config_data)  # type: ignore[arg-type]
                break
            except ConfigError as e:
                if not is_tty:
                    raise
                if attempt >= max_retries:
                    typer.echo(f"Error: {e} (max retries exceeded)", err=True)
                    raise typer.Exit(1)
                # Interactive recovery
                choice = questionary.select(
                    f"Validation error: {e}",
                    choices=["Revise selections", "Skip section", "Abort init"],
                ).ask()
                if choice is None or choice == "Abort init":
                    raise typer.Exit(1)
                elif choice == "Skip section":
                    # Remove problematic sections; if commands are empty, abort
                    if not commands:
                        typer.echo("Error: Cannot skip - no commands defined", err=True)
                        raise typer.Exit(1)
                    config_data.pop("evidence_check", None)
                    config_data.pop("per_issue_review", None)
                    config_data.pop("validation_triggers", None)
                else:
                    # Revise - if no commands, re-prompt custom commands
                    if not commands:
                        custom_commands = _prompt_custom_commands_questionary()
                        if custom_commands is None:
                            raise typer.Exit(1)
                        if custom_commands:
                            commands = list(custom_commands.keys())
                            config_data["commands"] = custom_commands
                    # Re-prompt for evidence, per-issue review, and triggers
                    if not skip_evidence and commands:
                        evidence_required = _prompt_evidence_check(commands, is_preset)
                        if evidence_required is not None:
                            config_data["evidence_check"] = _build_evidence_check_dict(
                                evidence_required
                            )
                        else:
                            config_data.pop("evidence_check", None)
                    # Re-prompt per-issue review (only when commands exist)
                    if commands:
                        per_issue_review_config = _prompt_per_issue_review()
                        if per_issue_review_config is not None:
                            config_data["per_issue_review"] = (
                                _build_per_issue_review_dict(per_issue_review_config)
                            )
                        else:
                            config_data.pop("per_issue_review", None)
                    if not skip_triggers and commands:
                        trigger_commands = _prompt_run_end_trigger(commands, is_preset)
                        if trigger_commands is not None:
                            config_data["validation_triggers"] = (
                                _build_validation_triggers_dict(trigger_commands)
                            )
                        else:
                            config_data.pop("validation_triggers", None)

        # Generate YAML content
        yaml_content = dump_config_yaml(config_data)

        # File operations
        config_path = Path("mala.yaml")
        if not dry_run:
            _write_with_backup(config_path, yaml_content)

        # Output YAML to stdout (both dry-run and normal)
        typer.echo(yaml_content)

        # Success tip only when file was written
        if not dry_run:
            typer.echo()
            typer.echo("Run `mala run` to start")

        # Always print trigger reference table at end
        _print_trigger_reference_table()

    except ConfigError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except OSError as e:
        typer.echo(f"Error writing file: {e}", err=True)
        raise typer.Exit(1)
    except KeyboardInterrupt:
        raise typer.Exit(130)
    except click.exceptions.Abort:
        raise typer.Exit(1)


def _display_instance(instance: object, indent: bool = False) -> None:
    """Display a single running instance.

    Args:
        instance: RunningInstance object (typed as object to avoid import issues)
        indent: If True, indent output (for grouped display under directory heading)
    """
    # Access attributes directly (duck typing)
    repo_path = getattr(instance, "repo_path", None)
    started_at = getattr(instance, "started_at", None)
    max_agents = getattr(instance, "max_agents", None)
    pid = getattr(instance, "pid", None)

    prefix = "  " if indent else ""

    # Only show repo path when not indented (when indented, directory is already shown as heading)
    if not indent:
        log("◐", f"repo: {repo_path}", Colors.CYAN)

    if started_at:
        # Calculate duration
        from datetime import datetime, UTC

        now = datetime.now(UTC)
        duration = now - started_at
        minutes = int(duration.total_seconds() // 60)
        if minutes >= 60:
            hours = minutes // 60
            mins = minutes % 60
            duration_str = f"{hours}h {mins}m"
        elif minutes > 0:
            duration_str = f"{minutes} minute(s)"
        else:
            seconds = int(duration.total_seconds())
            duration_str = f"{seconds} second(s)"
        if indent:
            print(f"{prefix}{Colors.MUTED}started: {duration_str} ago{Colors.RESET}")
        else:
            log("◐", f"started: {duration_str} ago", Colors.MUTED)

    if max_agents is not None:
        if indent:
            print(f"{prefix}{Colors.MUTED}max-agents: {max_agents}{Colors.RESET}")
        else:
            log("◐", f"max-agents: {max_agents}", Colors.MUTED)
    else:
        if indent:
            print(f"{prefix}{Colors.MUTED}max-agents: unlimited{Colors.RESET}")
        else:
            log("◐", "max-agents: unlimited", Colors.MUTED)

    if pid:
        if indent:
            print(f"{prefix}{Colors.MUTED}pid: {pid}{Colors.RESET}")
        else:
            log("◐", f"pid: {pid}", Colors.MUTED)


# Register subcommand apps (after main commands for better --help ordering)
app.add_typer(logs_app, name="logs")


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy-load SDK-dependent modules on first access."""
    if name not in _LAZY_NAMES:
        raise AttributeError(f"module {__name__} has no attribute {name}")

    if name in _lazy_modules:
        return _lazy_modules[name]

    if name == "MalaConfig":
        from src.infra.io.config import MalaConfig

        _lazy_modules[name] = MalaConfig
    elif name == "MalaOrchestrator":
        from ..orchestration.orchestrator import MalaOrchestrator

        _lazy_modules[name] = MalaOrchestrator
    elif name == "OrderPreference":
        from ..core.models import OrderPreference

        _lazy_modules[name] = OrderPreference
    elif name == "OrchestratorConfig":
        from ..orchestration.types import OrchestratorConfig

        _lazy_modules[name] = OrchestratorConfig
    elif name == "WatchConfig":
        from ..core.models import WatchConfig

        _lazy_modules[name] = WatchConfig
    elif name == "create_issue_provider":
        from ..orchestration.factory import create_issue_provider

        _lazy_modules[name] = create_issue_provider
    elif name == "create_orchestrator":
        from ..orchestration.factory import create_orchestrator

        _lazy_modules[name] = create_orchestrator
    elif name == "get_all_locks":
        from ..infra.tools.locking import get_all_locks

        _lazy_modules[name] = get_all_locks
    elif name == "get_lock_dir":
        from ..orchestration.cli_support import get_lock_dir

        _lazy_modules[name] = get_lock_dir
    elif name == "get_running_instances":
        from ..orchestration.cli_support import get_running_instances

        _lazy_modules[name] = get_running_instances
    elif name == "get_running_instances_for_dir":
        from ..orchestration.cli_support import get_running_instances_for_dir

        _lazy_modules[name] = get_running_instances_for_dir

    return _lazy_modules[name]

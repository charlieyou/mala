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
from pathlib import Path
from typing import TYPE_CHECKING, Any

import questionary

from ..orchestration.cli_options import (
    parse_scope,
    resolve_order_option,
    validate_amp_mode_option,
    validate_codex_approval_policy_option,
    validate_codex_sandbox_option,
    validate_coder_option,
    validate_effort_option,
    validate_model_option,
)
from ..orchestration.cli_overrides import (
    CLIOverrideOptions,
    build_resolved_mala_config,
)
from ..orchestration.cli_support import (
    USER_CONFIG_DIR,
    get_preset_config_commands,
    get_runs_dir,
    load_user_env,
)
from ..orchestration.init_config import (
    RECOVERY_ABORT,
    RECOVERY_REVISE,
    RECOVERY_SKIP,
    compute_evidence_defaults,
    compute_trigger_defaults,
    execute_init_workflow,
    validate_init_invocation,
)
from ..orchestration.run_command import execute_run_command

if TYPE_CHECKING:
    from ..orchestration.cli_options import ScopeConfig

# Bootstrap state: tracks whether bootstrap() has been called
_bootstrapped = False

# Lazy-loaded module cache for SDK-dependent imports
# These are populated on first access via __getattr__
_lazy_modules: dict[str, Any] = {}

# Names that are lazily loaded (for __getattr__ to handle).
# These modules transitively import claude_agent_sdk and so must stay
# deferred behind bootstrap(); plain config helpers no longer need to
# proxy through here.
_LAZY_NAMES = frozenset(
    {
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


ConfigData = dict[str, Any]


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

# SDK-dependent imports (create_orchestrator, get_lock_dir, run_metadata, etc.)
# are lazy-loaded via __getattr__ to ensure bootstrap() runs before claude_agent_sdk
# is imported. Access them as module attributes: create_orchestrator, OrchestratorConfig, etc.


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


def _parse_scope_cli(scope: str) -> ScopeConfig:
    """Parse ``--scope`` for the CLI: delegates and prints user-facing errors.

    Converts ``ValueError`` from ``parse_scope`` into the ``log`` + ``typer.Exit(1)``
    pattern the CLI uses for invalid options, and emits a warning when the
    underlying helper had to dedupe ``ids:`` entries.
    """
    try:
        scope_config = parse_scope(scope)
    except ValueError as exc:
        log("✗", str(exc), Colors.RED)
        raise typer.Exit(1)
    if scope_config.duplicate_ids:
        log(
            "⚠",
            f"Duplicate IDs removed from --scope: {', '.join(scope_config.duplicate_ids)}",
            Colors.YELLOW,
        )
    return scope_config


def _resolve_cli_config_path(config_path: Path | None) -> Path | None:
    """Resolve an explicit ``--config`` path relative to the current directory."""
    if config_path is None:
        return None
    return config_path.expanduser().resolve()


def _validate_coder_option(value: str | None) -> str | None:
    """Typer callback for ``--coder``; delegates to orchestration helper."""
    try:
        return validate_coder_option(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))


def _validate_amp_mode_option(value: str | None) -> str | None:
    """Typer callback for ``--amp-mode``; delegates to orchestration helper."""
    try:
        return validate_amp_mode_option(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))


def _validate_effort_option(value: str | None) -> str | None:
    """Typer callback for ``--effort``; delegates to orchestration helper."""
    try:
        return validate_effort_option(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))


def _validate_model_option(value: str | None) -> str | None:
    """Typer callback for ``--model``; delegates to orchestration helper."""
    try:
        return validate_model_option(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))


def _validate_codex_approval_policy_option(value: str | None) -> str | None:
    """Typer callback for ``--codex-approval-policy``; delegates to orchestration helper."""
    try:
        return validate_codex_approval_policy_option(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))


def _validate_codex_sandbox_option(value: str | None) -> str | None:
    """Typer callback for ``--codex-sandbox``; delegates to orchestration helper."""
    try:
        return validate_codex_sandbox_option(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc))


app = typer.Typer(
    name="mala",
    help="Parallel issue processing with Claude Agent SDK",
    add_completion=False,
)

_CODER_HELP = (
    "Coder backend to drive issue execution (claude, amp, or codex). Default: claude."
)
_AMP_MODE_HELP = (
    "Amp execution mode: smart, rush, or deep. Only consulted when coder=amp. "
    "Default: deep."
)
_EFFORT_HELP = (
    "Reasoning effort forwarded to the active coder. Valid: low, medium, high, "
    "xhigh, max. Defaults: claude=xhigh, amp smart=xhigh, amp deep=medium. "
    "Codex receives the same effort when configured."
)
_RUN_EFFORT_HELP = (
    f"{_EFFORT_HELP} Forwarded to ClaudeAgentOptions.effort for coder=claude "
    "and to `amp --effort <value>` for coder=amp (smart/deep modes only; "
    "Amp smart accepts medium, high, or xhigh; Amp deep accepts low, medium, or xhigh)."
)
_MODEL_HELP = (
    "Coder model identifier for Claude and Codex coders. Defaults: "
    "claude=opus[1m], codex=gpt-5.5. Ignored by Amp."
)
_CODEX_APPROVAL_POLICY_HELP = (
    "Codex approval policy. Valid: never, on-request, on-failure, untrusted. "
    "Only consulted when coder=codex. Default: never."
)
_CODEX_SANDBOX_HELP = (
    "Codex sandbox mode. Valid: read-only, workspace-write, danger-full-access. "
    "Only consulted when coder=codex. Default: danger-full-access."
)
_CONFIG_HELP = (
    "Path to a Mala project config file. Defaults to <repo>/mala.yaml; relative "
    "paths are resolved from the current working directory."
)


@app.command()
def run(
    repo_path: Annotated[
        Path,
        typer.Argument(
            help="Path to repository with beads issues",
        ),
    ] = Path("."),
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help=_CONFIG_HELP,
            rich_help_panel="Configuration",
        ),
    ] = None,
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
            help="Timeout per agent in minutes (default: 30)",
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
    fresh: Annotated[
        bool,
        typer.Option(
            "--fresh/--no-fresh",
            help="Start new SDK session instead of resuming (requires --resume)",
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
    coder: Annotated[
        str | None,
        typer.Option(
            "--coder",
            help=_CODER_HELP,
            callback=_validate_coder_option,
            rich_help_panel="Coder",
        ),
    ] = None,
    amp_mode: Annotated[
        str | None,
        typer.Option(
            "--amp-mode",
            help=_AMP_MODE_HELP,
            callback=_validate_amp_mode_option,
            rich_help_panel="Coder",
        ),
    ] = None,
    effort: Annotated[
        str | None,
        typer.Option(
            "--effort",
            help=_RUN_EFFORT_HELP,
            callback=_validate_effort_option,
            rich_help_panel="Coder",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help=_MODEL_HELP,
            callback=_validate_model_option,
            rich_help_panel="Coder",
        ),
    ] = None,
    codex_approval_policy: Annotated[
        str | None,
        typer.Option(
            "--codex-approval-policy",
            help=_CODEX_APPROVAL_POLICY_HELP,
            callback=_validate_codex_approval_policy_option,
            rich_help_panel="Coder",
        ),
    ] = None,
    codex_sandbox: Annotated[
        str | None,
        typer.Option(
            "--codex-sandbox",
            help=_CODEX_SANDBOX_HELP,
            callback=_validate_codex_sandbox_option,
            rich_help_panel="Coder",
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

    # Validate --fresh requires --resume
    if fresh and not resume:
        raise typer.BadParameter("--fresh requires --resume flag")

    # Validate --fresh conflicts with --strict
    if fresh and strict:
        raise typer.BadParameter("--fresh and --strict are mutually exclusive")

    repo_path = repo_path.resolve()
    selected_config_path = _resolve_cli_config_path(config_path)

    # Parse --scope option
    scope_config: ScopeConfig | None = None
    if scope is not None:
        scope_config = _parse_scope_cli(scope)

    # Parse and validate --order option
    try:
        order_resolution = resolve_order_option(order, scope_config)
    except ValueError as exc:
        log("✗", str(exc), Colors.RED)
        raise typer.Exit(1)

    # Validate repo_path exists
    if not repo_path.exists():
        log("✗", f"Repository not found: {repo_path}", Colors.RED)
        raise typer.Exit(1)

    # Ensure user config directory exists
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Hand the full run workflow (including --dry-run) to the orchestration
    # helper; CLI stays as an argument adapter that renders the outcome.
    outcome = execute_run_command(
        repo_path=repo_path,
        max_agents=max_agents,
        timeout=timeout,
        max_issues=max_issues,
        scope_config=scope_config,
        order_resolution=order_resolution,
        resume=resume,
        strict=strict,
        fresh=fresh,
        dry_run=dry_run,
        watch=watch,
        override_options=CLIOverrideOptions(
            claude_settings_sources=claude_settings_sources,
            coder=coder,
            amp_mode=amp_mode,
            model=model,
            effort=effort,
            codex_approval_policy=codex_approval_policy,
            codex_sandbox=codex_sandbox,
        ),
        config_path=selected_config_path,
    )
    if outcome.dry_run is not None:
        display_dry_run_tasks(outcome.dry_run.issues, focus=outcome.dry_run.focus)
    elif outcome.error_message:
        log("✗", outcome.error_message, Colors.RED)
    raise typer.Exit(outcome.exit_code)


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
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            help=_CONFIG_HELP,
            rich_help_panel="Configuration",
        ),
    ] = None,
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
    coder: Annotated[
        str | None,
        typer.Option(
            "--coder",
            help=_CODER_HELP,
            callback=_validate_coder_option,
            rich_help_panel="Coder",
        ),
    ] = None,
    amp_mode: Annotated[
        str | None,
        typer.Option(
            "--amp-mode",
            help=_AMP_MODE_HELP,
            callback=_validate_amp_mode_option,
            rich_help_panel="Coder",
        ),
    ] = None,
    effort: Annotated[
        str | None,
        typer.Option(
            "--effort",
            help=_RUN_EFFORT_HELP,
            callback=_validate_effort_option,
            rich_help_panel="Coder",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help=_MODEL_HELP,
            callback=_validate_model_option,
            rich_help_panel="Coder",
        ),
    ] = None,
    codex_approval_policy: Annotated[
        str | None,
        typer.Option(
            "--codex-approval-policy",
            help=_CODEX_APPROVAL_POLICY_HELP,
            callback=_validate_codex_approval_policy_option,
            rich_help_panel="Coder",
        ),
    ] = None,
    codex_sandbox: Annotated[
        str | None,
        typer.Option(
            "--codex-sandbox",
            help=_CODEX_SANDBOX_HELP,
            callback=_validate_codex_sandbox_option,
            rich_help_panel="Coder",
        ),
    ] = None,
) -> None:
    """Verify a single epic and optionally close it without running tasks."""
    set_verbose(verbose)

    if human_override and not close:
        log("✗", "--human-override requires --close", Colors.RED)
        raise typer.Exit(1)

    repo_path = repo_path.resolve()
    selected_config_path = _resolve_cli_config_path(config_path)
    if not repo_path.exists():
        log("✗", f"Repository not found: {repo_path}", Colors.RED)
        raise typer.Exit(1)

    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    try:
        config = build_resolved_mala_config(
            repo_path,
            CLIOverrideOptions(
                claude_settings_sources=claude_settings_sources,
                coder=coder,
                amp_mode=amp_mode,
                model=model,
                effort=effort,
                codex_approval_policy=codex_approval_policy,
                codex_sandbox=codex_sandbox,
            ),
            config_path=selected_config_path,
        )
        orch_config = _lazy("OrchestratorConfig")(
            repo_path=repo_path,
            config_path=selected_config_path,
        )
        orchestrator = _lazy("create_orchestrator")(orch_config, mala_config=config)
    except ValueError as exc:
        log("✗", str(exc), Colors.RED)
        raise typer.Exit(1)
    except ConfigError as exc:
        if selected_config_path is None:
            raise
        log("✗", str(exc), Colors.RED)
        raise typer.Exit(1)
    verifier = getattr(orchestrator, "epic_verifier", None)

    if verifier is None:
        log("✗", "Epic verifier unavailable for this configuration", Colors.RED)
        raise typer.Exit(1)

    reviewer_type = getattr(verifier, "reviewer_type", "unknown")
    lock_timeout_display = getattr(verifier, "lock_timeout_display", "unknown")
    config_items = [
        f"reviewer={reviewer_type}",
        f"close={str(close).lower()}",
        f"force={str(force).lower()}",
        f"human_override={str(human_override).lower()}",
        f"lock_timeout={lock_timeout_display}",
    ]
    log("→", "[START] Epic verification", agent_id="epic")
    log("◦", f"Repository: {repo_path}", agent_id="epic")
    log("◦", f"Epic: {epic_id}", agent_id="epic")
    log("◐", f"Config: {' '.join(config_items)}", agent_id="epic")

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


def _prompt_evidence_check(commands: list[str], is_preset: bool) -> list[str] | None:
    """Prompt user to select evidence check commands.

    Args:
        commands: List of command names available.
        is_preset: True if using a preset, False for custom.

    Returns:
        List of selected command names, or None to skip.
    """
    defaults = compute_evidence_defaults(commands, is_preset)

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
    defaults = compute_trigger_defaults(commands, is_preset)

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

        # Flag/TTY combination validation lives in the orchestration helper;
        # CLI converts the helper's error message into typer.Exit(1).
        invocation_check = validate_init_invocation(
            yes=yes,
            preset=preset,
            skip_evidence=skip_evidence,
            skip_triggers=skip_triggers,
            is_tty=is_tty,
        )
        if invocation_check.error_message:
            typer.echo(f"Error: {invocation_check.error_message}", err=True)
            raise typer.Exit(1)

        # Drive the init state machine via the orchestration helper. The
        # CLI provides a questionary-backed prompts adapter; the helper
        # owns preset/custom branching, section construction, and the
        # validation retry/recovery loop.
        result = execute_init_workflow(
            preset=preset,
            skip_evidence=skip_evidence,
            skip_triggers=skip_triggers,
            yes=yes,
            is_tty=is_tty,
            presets=get_init_presets(),
            commands_lookup=get_preset_config_commands,
            validate_config=validate_init_config,
            prompts=_QuestionaryInitPrompts(),
        )
        if result.exit_code != 0:
            if result.error_message:
                typer.echo(f"Error: {result.error_message}", err=True)
            raise typer.Exit(result.exit_code)
        # exit_code == 0 guarantees config_data is populated.
        assert result.config_data is not None
        config_data: ConfigData = result.config_data

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


class _QuestionaryInitPrompts:
    """questionary-backed implementation of ``InitPrompts``.

    Routes the helper's prompt calls to the existing ``_prompt_*`` CLI
    helpers (and questionary directly for the recovery prompt). Keeps
    user-facing questionary calls in the CLI layer per the import-linter
    contract ``Only CLI imports typer`` and its sibling boundary that
    questionary stays out of the orchestration layer.
    """

    def prompt_preset(self, presets: list[str]) -> str | None:
        return _prompt_preset_selection(presets)

    def prompt_custom_commands(self) -> dict[str, str] | None:
        return _prompt_custom_commands_questionary()

    def prompt_evidence_check(
        self, commands: list[str], is_preset: bool
    ) -> list[str] | None:
        return _prompt_evidence_check(commands, is_preset)

    def prompt_per_issue_review(self) -> dict[str, Any] | None:
        return _prompt_per_issue_review()

    def prompt_run_end_trigger(
        self, commands: list[str], is_preset: bool
    ) -> list[str] | None:
        return _prompt_run_end_trigger(commands, is_preset)

    def prompt_recovery_choice(self, error_message: str) -> str | None:
        result = questionary.select(
            f"Validation error: {error_message}",
            choices=[RECOVERY_REVISE, RECOVERY_SKIP, RECOVERY_ABORT],
        ).ask()
        if result is None:
            return None
        return str(result)

    def notify(self, message: str) -> None:
        typer.echo(message)


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

    if name == "OrchestratorConfig":
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

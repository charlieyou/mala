#!/usr/bin/env python3
# ruff: noqa: E402
"""
mala CLI: Agent SDK orchestrator for parallel issue processing.

Usage:
    mala run [OPTIONS] [REPO_PATH]
    mala clean
    mala status
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from .tools.env import USER_CONFIG_DIR, get_runs_dir, load_user_env

# Bootstrap state: tracks whether bootstrap() has been called
# These are populated on first access via __getattr__
_bootstrapped = False
_braintrust_enabled = False

# Lazy-loaded module cache for SDK-dependent imports
# These are populated on first access via __getattr__
_lazy_modules: dict[str, Any] = {}

# Names that are lazily loaded (for __getattr__ to handle)
_LAZY_NAMES = frozenset(
    {
        "BeadsClient",
        "MalaOrchestrator",
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
    """Initialize environment and Braintrust tracing.

    Must be called before using any CLI commands or importing claude_agent_sdk.
    This function is idempotent - calling it multiple times has no additional effect.

    Side effects:
        - Loads environment variables from ~/.config/mala/.env
        - Sets up Braintrust SDK patching if BRAINTRUST_API_KEY is set
    """
    global _bootstrapped, _braintrust_enabled

    if _bootstrapped:
        return

    # Load environment variables early (needed for BRAINTRUST_API_KEY)
    load_user_env()

    # Setup Braintrust tracing BEFORE importing claude_agent_sdk anywhere
    # This patches the SDK before any imports can use the unpatched version
    braintrust_api_key = os.environ.get("BRAINTRUST_API_KEY")
    if braintrust_api_key:
        try:
            from braintrust.wrappers.claude_agent_sdk import setup_claude_agent_sdk

            setup_claude_agent_sdk(project="mala")
            _braintrust_enabled = True
        except ImportError:
            pass  # braintrust not installed

    _bootstrapped = True


def is_braintrust_enabled() -> bool:
    """Return whether Braintrust tracing was successfully enabled.

    Must be called after bootstrap().
    """
    return _braintrust_enabled


# Import modules that do NOT depend on claude_agent_sdk at module level
import asyncio
from datetime import datetime
from typing import Annotated, Never

import typer

from .logging.console import Colors, log, set_verbose

# SDK-dependent imports (BeadsClient, MalaOrchestrator, get_lock_dir, run_metadata)
# are lazy-loaded via __getattr__ to ensure bootstrap() runs before claude_agent_sdk
# is imported. Access them as module attributes: BeadsClient, MalaOrchestrator, etc.


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


# Valid values for --disable-validations flag
# Each value controls a specific validation phase:
#   post-validate: Skip all validation commands (pytest, ruff, ty) after agent commits
#   run-level-validate: Skip run-level validation at end of batch (reserved for future use)
#   slow-tests: Exclude @pytest.mark.slow tests from test runs
#   coverage: Disable code coverage enforcement (threshold check)
#   e2e: Disable end-to-end fixture repo tests (run-level only)
#   codex-review: Disable automated LLM code review after quality gate passes
#   followup-on-run-validate-fail: Disable auto-retry on run validation failures (reserved)
VALID_DISABLE_VALUES = frozenset(
    {
        "post-validate",
        "run-level-validate",
        "slow-tests",
        "coverage",
        "e2e",
        "codex-review",
        "followup-on-run-validate-fail",
    }
)


app = typer.Typer(
    name="mala",
    help="Parallel issue processing with Claude Agent SDK",
    add_completion=False,
)


def get_morph_api_key() -> str | None:
    """Get MORPH_API_KEY from environment. Returns None if missing."""
    return os.environ.get("MORPH_API_KEY") or None


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
            "--max-agents", "-n", help="Maximum concurrent agents (default: unlimited)"
        ),
    ] = None,
    timeout: Annotated[
        int | None,
        typer.Option(
            "--timeout", "-t", help="Timeout per agent in minutes (default: no timeout)"
        ),
    ] = None,
    max_issues: Annotated[
        int | None,
        typer.Option(
            "--max-issues", "-i", help="Maximum issues to process (default: unlimited)"
        ),
    ] = None,
    epic: Annotated[
        str | None,
        typer.Option(
            "--epic", "-e", help="Only process tasks that are children of this epic"
        ),
    ] = None,
    only: Annotated[
        str | None,
        typer.Option(
            "--only",
            "-o",
            help="Comma-separated list of issue IDs to process exclusively",
        ),
    ] = None,
    max_gate_retries: Annotated[
        int,
        typer.Option(
            "--max-gate-retries",
            help="Maximum quality gate retry attempts per issue (default: 3)",
        ),
    ] = 3,
    max_review_retries: Annotated[
        int,
        typer.Option(
            "--max-review-retries",
            help="Maximum codex review retry attempts per issue (default: 5)",
        ),
    ] = 5,
    disable_validations: Annotated[
        str | None,
        typer.Option(
            "--disable-validations",
            help=(
                "Comma-separated validations to skip. Options: "
                "post-validate (skip pytest/ruff/ty after commits), "
                "slow-tests (exclude @pytest.mark.slow tests), "
                "coverage (disable coverage threshold check), "
                "e2e (skip end-to-end fixture tests), "
                "codex-review (skip LLM code review)"
            ),
        ),
    ] = None,
    coverage_threshold: Annotated[
        float | None,
        typer.Option(
            "--coverage-threshold",
            help="Minimum coverage percentage (0-100). If not set, uses 'no decrease' mode which requires coverage >= previous baseline.",
        ),
    ] = None,
    wip: Annotated[
        bool,
        typer.Option(
            "--wip",
            help="Prioritize in_progress issues before open issues",
        ),
    ] = False,
    focus: Annotated[
        bool,
        typer.Option(
            "--focus/--no-focus",
            help="Group tasks by epic for focused work (default: on); --no-focus uses priority-only ordering",
        ),
    ] = True,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Preview task order without processing; shows what would be run",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output; shows full tool arguments instead of single line per tool call",
        ),
    ] = False,
    codex_thinking_mode: Annotated[
        str | None,
        typer.Option(
            "--codex-thinking-mode",
            help="Reasoning effort level for Codex reviews (e.g., 'high', 'medium', 'low', 'xhigh')",
        ),
    ] = None,
) -> Never:
    """Run parallel issue processing."""
    # Apply verbose setting
    set_verbose(verbose)

    repo_path = repo_path.resolve()

    # Parse --only flag into a set of issue IDs
    only_ids: set[str] | None = None
    if only:
        only_ids = {
            issue_id.strip() for issue_id in only.split(",") if issue_id.strip()
        }
        if not only_ids:
            log("✗", "Invalid --only value: no valid issue IDs found", Colors.RED)
            raise typer.Exit(1)

    # Parse --disable-validations flag into a set
    disable_set: set[str] | None = None
    if disable_validations:
        disable_set = {
            val.strip() for val in disable_validations.split(",") if val.strip()
        }
        if not disable_set:
            log(
                "✗",
                "Invalid --disable-validations value: no valid values found",
                Colors.RED,
            )
            raise typer.Exit(1)
        # Validate against known values
        unknown = disable_set - VALID_DISABLE_VALUES
        if unknown:
            log(
                "✗",
                f"Unknown --disable-validations value(s): {', '.join(sorted(unknown))}. "
                f"Valid values: {', '.join(sorted(VALID_DISABLE_VALUES))}",
                Colors.RED,
            )
            raise typer.Exit(1)

    # Validate coverage threshold range
    if coverage_threshold is not None and not 0 <= coverage_threshold <= 100:
        log(
            "✗",
            f"Invalid --coverage-threshold value: {coverage_threshold}. Must be between 0 and 100.",
            Colors.RED,
        )
        raise typer.Exit(1)

    # Ensure user config directory exists
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Check Morph API key (optional - enables morph features when present)
    morph_api_key = get_morph_api_key()
    morph_enabled = morph_api_key is not None
    if not morph_enabled:
        log(
            "⚠",
            "MORPH_API_KEY not set - Morph MCP features disabled",
            Colors.YELLOW,
        )

    if not repo_path.exists():
        log("✗", f"Repository not found: {repo_path}", Colors.RED)
        raise typer.Exit(1)

    # Handle dry-run mode: display task order and exit
    if dry_run:

        async def _dry_run() -> None:
            beads = _lazy("BeadsClient")(repo_path)
            issues = await beads.get_ready_issues_async(
                epic_id=epic,
                only_ids=only_ids,
                prioritize_wip=wip,
                focus=focus,
            )
            display_dry_run_tasks(issues, focus=focus)

        asyncio.run(_dry_run())
        raise typer.Exit(0)

    # Build cli_args for logging and metadata
    cli_args: dict[str, object] = {
        "disable_validations": disable_validations,
        "coverage_threshold": coverage_threshold,
        "wip": wip,
        "max_issues": max_issues,
        "max_gate_retries": max_gate_retries,
        "max_review_retries": max_review_retries,
        "braintrust": _braintrust_enabled,
        "codex_thinking_mode": codex_thinking_mode,
    }

    orchestrator = _lazy("MalaOrchestrator")(
        repo_path=repo_path,
        max_agents=max_agents,
        timeout_minutes=timeout,
        max_issues=max_issues,
        epic_id=epic,
        only_ids=only_ids,
        braintrust_enabled=_braintrust_enabled,
        max_gate_retries=max_gate_retries,
        max_review_retries=max_review_retries,
        disable_validations=disable_set,
        coverage_threshold=coverage_threshold,
        morph_enabled=morph_enabled,
        prioritize_wip=wip,
        focus=focus,
        cli_args=cli_args,
        codex_thinking_mode=codex_thinking_mode,
    )

    success_count, total = asyncio.run(orchestrator.run())
    # Exit 0 if: no issues to process (no-op) OR at least one succeeded
    # Exit 1 only if: issues were processed but all failed
    raise typer.Exit(0 if success_count > 0 or total == 0 else 1)


@app.command()
def clean() -> None:
    """Clean up locks and JSONL logs."""
    cleaned_locks = 0
    cleaned_logs = 0

    lock_dir = _lazy("get_lock_dir")()
    if lock_dir.exists():
        for lock in lock_dir.glob("*.lock"):
            lock.unlink()
            cleaned_locks += 1

    if cleaned_locks:
        log("🧹", f"Removed {cleaned_locks} lock files", Colors.GREEN)

    if get_runs_dir().exists():
        # Use rglob to find run files in both legacy flat structure
        # and new repo-segmented subdirectories
        run_files = list(get_runs_dir().rglob("*.json"))
        run_count = len(run_files)
        if run_count > 0:
            if typer.confirm(f"Remove {run_count} run metadata files?"):
                for run_file in run_files:
                    run_file.unlink()
                    cleaned_logs += 1
                # Also remove empty repo subdirectories
                for subdir in get_runs_dir().iterdir():
                    if subdir.is_dir() and not any(subdir.iterdir()):
                        subdir.rmdir()
                log("🧹", f"Removed {cleaned_logs} run metadata files", Colors.GREEN)


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

    # Check locks (show all locks, not filtered by directory)
    lock_dir = _lazy("get_lock_dir")()
    if lock_dir.exists():
        locks = list(lock_dir.glob("*.lock"))
        if locks:
            log("⚠", f"{len(locks)} active locks", Colors.YELLOW)
            for lock in locks[:5]:
                try:
                    holder = lock.read_text().strip() if lock.is_file() else "unknown"
                except OSError:
                    holder = "unknown"
                print(f"    {Colors.MUTED}{lock.stem} → {holder}{Colors.RESET}")
            if len(locks) > 5:
                print(f"    {Colors.MUTED}... and {len(locks) - 5} more{Colors.RESET}")
        else:
            log("○", "No active locks", Colors.GRAY)
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


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy-load SDK-dependent modules on first access."""
    if name not in _LAZY_NAMES:
        raise AttributeError(f"module {__name__} has no attribute {name}")

    if name in _lazy_modules:
        return _lazy_modules[name]

    if name == "BeadsClient":
        from .beads_client import BeadsClient

        _lazy_modules[name] = BeadsClient
    elif name == "MalaOrchestrator":
        from .orchestrator import MalaOrchestrator

        _lazy_modules[name] = MalaOrchestrator
    elif name == "get_lock_dir":
        from .tools.locking import get_lock_dir

        _lazy_modules[name] = get_lock_dir
    elif name == "get_running_instances":
        from .logging.run_metadata import get_running_instances

        _lazy_modules[name] = get_running_instances
    elif name == "get_running_instances_for_dir":
        from .logging.run_metadata import get_running_instances_for_dir

        _lazy_modules[name] = get_running_instances_for_dir

    return _lazy_modules[name]

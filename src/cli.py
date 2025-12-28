#!/usr/bin/env python3
# ruff: noqa: E402
"""
mala CLI: Agent SDK orchestrator for parallel issue processing.

Usage:
    mala run [OPTIONS] [REPO_PATH]
    mala clean
    mala status
"""

import os
from pathlib import Path

from .tools.env import USER_CONFIG_DIR, get_runs_dir, load_user_env

# Bootstrap state: tracks whether bootstrap() has been called
_bootstrapped = False
_braintrust_enabled = False


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

# SDK-dependent imports (MalaOrchestrator, BeadsClient, get_lock_dir) are deferred
# to function bodies to ensure bootstrap() runs before claude_agent_sdk is imported.


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
        float,
        typer.Option(
            "--coverage-threshold",
            help="Minimum coverage percentage, 0-100 (default: 85.0)",
        ),
    ] = 85.0,
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
            "--verbose/--quiet",
            "-v/-q",
            help="Verbose output shows full tool arguments; quiet mode shows single line per tool call",
        ),
    ] = True,
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
    if not 0 <= coverage_threshold <= 100:
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

    # Import SDK-dependent modules at runtime (after bootstrap has been called)
    from .beads_client import BeadsClient
    from .orchestrator import MalaOrchestrator

    # Handle dry-run mode: display task order and exit
    if dry_run:

        async def _dry_run() -> None:
            beads = BeadsClient(repo_path)
            issues = await beads.get_ready_issues_async(
                epic_id=epic,
                only_ids=only_ids,
                prioritize_wip=wip,
                focus=focus,
            )
            display_dry_run_tasks(issues, focus=focus)

        asyncio.run(_dry_run())
        raise typer.Exit(0)

    orchestrator = MalaOrchestrator(
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
    )

    success_count, total = asyncio.run(orchestrator.run())
    # Exit 0 if: no issues to process (no-op) OR at least one succeeded
    # Exit 1 only if: issues were processed but all failed
    raise typer.Exit(0 if success_count > 0 or total == 0 else 1)


@app.command()
def clean() -> None:
    """Clean up locks and JSONL logs."""
    # Import get_lock_dir at runtime to avoid SDK import at module level
    from .tools.locking import get_lock_dir

    cleaned_locks = 0
    cleaned_logs = 0

    if get_lock_dir().exists():
        for lock in get_lock_dir().glob("*.lock"):
            lock.unlink()
            cleaned_locks += 1

    if cleaned_locks:
        log("🧹", f"Removed {cleaned_locks} lock files", Colors.GREEN)

    if get_runs_dir().exists():
        run_count = len(list(get_runs_dir().glob("*.json")))
        if run_count > 0:
            if typer.confirm(f"Remove {run_count} run metadata files?"):
                for run_file in get_runs_dir().glob("*.json"):
                    run_file.unlink()
                    cleaned_logs += 1
                log("🧹", f"Removed {cleaned_logs} run metadata files", Colors.GREEN)


@app.command()
def status() -> None:
    """Show current orchestrator status."""
    # Import get_lock_dir at runtime to avoid SDK import at module level
    from .tools.locking import get_lock_dir

    print()
    log("●", "mala status", Colors.MAGENTA)
    print()

    # Show config directory
    config_env = USER_CONFIG_DIR / ".env"
    if config_env.exists():
        log("◐", f"config: {config_env}", Colors.MUTED)
    else:
        log("○", f"config: {config_env} (not found)", Colors.MUTED)
    print()

    # Check locks
    if get_lock_dir().exists():
        locks = list(get_lock_dir().glob("*.lock"))
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

    # Check run metadata
    if get_runs_dir().exists():
        run_files = list(get_runs_dir().glob("*.json"))
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
                print(f"    {Colors.MUTED}{mtime} {run_file.name}{Colors.RESET}")
    print()

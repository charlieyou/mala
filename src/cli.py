#!/usr/bin/env python3
# ruff: noqa: E402
"""
mala CLI: Agent SDK orchestrator for parallel issue processing.

Usage:
    mala run [OPTIONS] [REPO_PATH]
    mala clean
    mala status
"""

# Setup Braintrust tracing BEFORE importing claude_agent_sdk anywhere
# This patches the SDK before any imports can use the unpatched version
import os
from pathlib import Path

from .tools.env import USER_CONFIG_DIR, get_runs_dir, load_user_env

# Load environment variables early (needed for BRAINTRUST_API_KEY)
load_user_env()

_braintrust_early_setup_done = False
_braintrust_api_key = os.environ.get("BRAINTRUST_API_KEY")
if _braintrust_api_key:
    try:
        from braintrust.wrappers.claude_agent_sdk import setup_claude_agent_sdk

        setup_claude_agent_sdk(project="mala")
        _braintrust_early_setup_done = True
    except ImportError:
        pass  # braintrust not installed

# Now safe to import claude_agent_sdk (it's been patched if Braintrust is available)
import asyncio
from datetime import datetime
from typing import Annotated, Never

import typer

from .logging.console import Colors, log, set_verbose
from .tools.locking import get_lock_dir
from .orchestrator import MalaOrchestrator


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

    orchestrator = MalaOrchestrator(
        repo_path=repo_path,
        max_agents=max_agents,
        timeout_minutes=timeout,
        max_issues=max_issues,
        epic_id=epic,
        only_ids=only_ids,
        braintrust_enabled=_braintrust_early_setup_done,
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
    print()
    log("●", "mala status", Colors.MAGENTA)
    print()

    # Show config directory
    config_env = USER_CONFIG_DIR / ".env"
    if config_env.exists():
        log("◐", f"config: {config_env}", Colors.GRAY, dim=True)
    else:
        log("○", f"config: {config_env} (not found)", Colors.GRAY, dim=True)
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
                print(f"    {Colors.DIM}{lock.stem} → {holder}{Colors.RESET}")
            if len(locks) > 5:
                print(f"    {Colors.DIM}... and {len(locks) - 5} more{Colors.RESET}")
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
                Colors.GRAY,
                dim=True,
            )
            recent = sorted(run_files, key=lambda p: p.stat().st_mtime, reverse=True)[
                :3
            ]
            for run_file in recent:
                mtime = datetime.fromtimestamp(run_file.stat().st_mtime).strftime(
                    "%H:%M:%S"
                )
                print(f"    {Colors.DIM}{mtime} {run_file.name}{Colors.RESET}")
    print()

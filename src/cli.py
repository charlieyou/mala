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

from .tools.env import USER_CONFIG_DIR, RUNS_DIR, load_user_env

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
from .tools.locking import LOCK_DIR
from .orchestrator import MalaOrchestrator


app = typer.Typer(
    name="mala",
    help="Parallel issue processing with Claude Agent SDK",
    add_completion=False,
)


def validate_morph_api_key() -> str:
    """Validate MORPH_API_KEY is set. Raises SystemExit if missing."""
    api_key = os.environ.get("MORPH_API_KEY")
    if not api_key:
        raise SystemExit(
            f"Error: MORPH_API_KEY is required.\n"
            f"Add it to {USER_CONFIG_DIR}/.env or set as environment variable."
        )
    return api_key


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
    codex_review: Annotated[
        bool,
        typer.Option(
            "--codex-review/--no-codex-review",
            help="Enable/disable codex review step (default: enabled)",
        ),
    ] = True,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose output with full tool arguments and code",
        ),
    ] = False,
) -> Never:
    """Run parallel issue processing."""
    # Apply verbose setting
    if verbose:
        set_verbose(True)

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

    # Ensure user config directory exists
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Validate Morph API key (required)
    validate_morph_api_key()

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
        codex_review=codex_review,
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

    if LOCK_DIR.exists():
        for lock in LOCK_DIR.glob("*.lock"):
            lock.unlink()
            cleaned_locks += 1

    if cleaned_locks:
        log("🧹", f"Removed {cleaned_locks} lock files", Colors.GREEN)

    if RUNS_DIR.exists():
        run_count = len(list(RUNS_DIR.glob("*.json")))
        if run_count > 0:
            if typer.confirm(f"Remove {run_count} run metadata files?"):
                for run_file in RUNS_DIR.glob("*.json"):
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
    if LOCK_DIR.exists():
        locks = list(LOCK_DIR.glob("*.lock"))
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
    if RUNS_DIR.exists():
        run_files = list(RUNS_DIR.glob("*.json"))
        if run_files:
            log(
                "◐",
                f"{len(run_files)} run metadata files in {RUNS_DIR}",
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

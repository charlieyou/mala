"""Logs subcommand for mala CLI: search and inspect run logs."""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from tabulate import tabulate

from src.infra.tools.env import get_repo_runs_dir, get_runs_dir

logs_app = typer.Typer(name="logs", help="Search and inspect mala run logs")

# Required keys for valid run metadata
_REQUIRED_KEYS = {"run_id", "started_at", "issues"}

# Default limit for number of runs to display
_DEFAULT_LIMIT = 20


def _discover_run_files(all_runs: bool) -> list[Path]:
    """Discover run metadata JSON files.

    Args:
        all_runs: If True, search all repos. Otherwise, only current repo.

    Returns:
        List of JSON file paths sorted by filename descending (newest first).
    """
    if all_runs:
        runs_dir = get_runs_dir()
        if not runs_dir.exists():
            return []
        # rglob for all .json files across all repo directories
        files = list(runs_dir.rglob("*.json"))
    else:
        cwd = Path.cwd()
        repo_runs_dir = get_repo_runs_dir(cwd)
        if not repo_runs_dir.exists():
            return []
        files = list(repo_runs_dir.glob("*.json"))

    # Sort by filename descending (timestamps in filenames give newest first)
    return sorted(files, key=lambda p: p.name, reverse=True)


def _validate_run_metadata(
    data: dict[str, object] | list[object] | str | float | bool | None,
) -> bool:
    """Check if run metadata is a dict with required keys and valid values.

    Args:
        data: Parsed JSON data.

    Returns:
        True if data is a dict with all required keys and valid values.
    """
    if not isinstance(data, dict):
        return False
    if not _REQUIRED_KEYS.issubset(data.keys()):
        return False
    # Validate started_at is a non-null string
    if not isinstance(data.get("started_at"), str):
        return False
    # Validate issues is a dict (or None which we treat as empty)
    issues = data.get("issues")
    if issues is not None and not isinstance(issues, dict):
        return False
    return True


def _parse_run_file(path: Path) -> dict[str, Any] | None:
    """Parse a run metadata JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed dict if valid, None if corrupt or missing required keys.
        Prints warning to stderr for corrupt files.
    """
    try:
        with path.open() as f:
            data = json.load(f)
        if not _validate_run_metadata(data):
            return None
        # Normalize issues to empty dict if None
        if data.get("issues") is None:
            data["issues"] = {}
        return data
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: skipping corrupt file {path}: {e}", file=sys.stderr)
        return None


def _count_issue_statuses(issues: dict[str, Any]) -> tuple[int, int, int, int]:
    """Count issue statuses from issues dict.

    Args:
        issues: Dict of issue_id -> issue data.

    Returns:
        Tuple of (total, success, failed, timeout) counts.
    """
    total = len(issues)
    success = 0
    failed = 0
    timeout = 0
    for issue in issues.values():
        status = issue.get("status")
        if status == "success":
            success += 1
        elif status == "failed":
            failed += 1
        elif status == "timeout":
            timeout += 1
    return total, success, failed, timeout


def _sort_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort runs by started_at descending, then run_id ascending for ties.

    Args:
        runs: List of run metadata dicts.

    Returns:
        Sorted list.
    """
    return sorted(runs, key=lambda r: (-_parse_timestamp(r["started_at"]), r["run_id"]))


def _parse_timestamp(ts: str) -> float:
    """Parse ISO timestamp to epoch float for sorting."""
    try:
        # Handle both Z suffix and +00:00
        ts = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        return dt.timestamp()
    except (ValueError, TypeError, AttributeError):
        return 0.0


def _collect_runs(files: list[Path], limit: int) -> list[dict[str, Any]]:
    """Collect all valid run metadata, sort by started_at, and return top N.

    Args:
        files: List of JSON file paths.
        limit: Maximum runs to return after sorting.

    Returns:
        List of run metadata dicts with 'metadata_path' added, sorted by
        started_at desc then run_id asc, limited to `limit` entries.
    """
    runs: list[dict[str, Any]] = []
    for path in files:
        data = _parse_run_file(path)
        if data is not None:
            # Add metadata_path to a copy to avoid mutating parsed data
            run_entry = {**data, "metadata_path": str(path)}
            runs.append(run_entry)
    return runs


def _format_null(value: str | float | None) -> str:
    """Format value for table display, showing '-' for None."""
    return "-" if value is None else str(value)


def _extract_repo_from_path(metadata_path: str) -> str | None:
    """Extract repo path from metadata file's parent directory name.

    The parent directory is encoded as '-home-user-repo' format.
    Decodes back to '/home/user/repo'.
    """
    parent_name = Path(metadata_path).parent.name
    if parent_name.startswith("-"):
        # Decode: -home-user-repo -> /home/user/repo
        return "/" + parent_name[1:].replace("-", "/")
    return None


@logs_app.command(name="list")
def list_runs(
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output as JSON",
        ),
    ] = False,
    all_runs: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Show all runs (not just current repo)",
        ),
    ] = False,
    limit: Annotated[
        int,
        typer.Option(
            "--limit",
            "-n",
            help="Maximum number of runs to display",
        ),
    ] = _DEFAULT_LIMIT,
) -> None:
    """List recent mala runs."""
    files = _discover_run_files(all_runs)
    runs = _collect_runs(files, limit=limit)

    # Sort by started_at desc, run_id asc for determinism, then take top N
    runs = _sort_runs(runs)[:limit]

    if not runs:
        if json_output:
            print("[]")
        else:
            print("No runs found")
        return

    if json_output:
        # Build JSON output with counts
        output = []
        for run in runs:
            issues = run.get("issues", {})
            total, success, failed, timeout = _count_issue_statuses(issues)
            entry: dict[str, Any] = {
                "run_id": run["run_id"],
                "started_at": run["started_at"],
                "issue_count": total,
                "success": success,
                "fail": failed,
                "timeout": timeout,
                "metadata_path": run["metadata_path"],
            }
            if all_runs:
                # Extract repo_path from parent directory name
                entry["repo_path"] = _extract_repo_from_path(run["metadata_path"])
            output.append(entry)
        print(json.dumps(output, indent=2))
    else:
        # Build table rows (run_id truncated to 8 chars for display)
        headers = [
            "run_id",
            "started_at",
            "issues",
            "success",
            "fail",
            "timeout",
            "path",
        ]
        if all_runs:
            headers.insert(1, "repo_path")

        rows = []
        for run in runs:
            issues = run.get("issues", {})
            total, success, failed, timeout = _count_issue_statuses(issues)
            row = [
                run["run_id"][:8],  # Short ID for display
                _format_null(run.get("started_at")),
                total,
                success,
                failed,
                timeout,
                run["metadata_path"],
            ]
            if all_runs:
                repo_path = _extract_repo_from_path(run["metadata_path"])
                row.insert(1, _format_null(repo_path))
            rows.append(row)

        print(tabulate(rows, headers=headers, tablefmt="simple"))


@logs_app.command()
def sessions(
    issue: Annotated[
        str | None,
        typer.Option(
            "--issue",
            help="Filter by issue ID",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output as JSON",
        ),
    ] = False,
    all_sessions: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Show all sessions (not just recent)",
        ),
    ] = False,
) -> None:
    """List Claude sessions from mala runs."""
    raise NotImplementedError("Not implemented yet")


@logs_app.command()
def show(
    run_id: Annotated[
        str,
        typer.Argument(
            help="Run ID to show details for",
        ),
    ],
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output as JSON",
        ),
    ] = False,
) -> None:
    """Show details for a specific run."""
    raise NotImplementedError("Not implemented yet")

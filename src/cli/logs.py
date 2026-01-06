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


def _validate_run_metadata(data: object) -> bool:
    """Check if run metadata is a dict with required keys and valid values.

    Args:
        data: Parsed JSON data (any JSON value).

    Returns:
        True if data is a dict with all required keys and valid values.
    """
    if not isinstance(data, dict):
        return False
    # Cast to dict[str, Any] for type checker after isinstance check
    d = dict(data)  # type: dict[str, Any]
    if not _REQUIRED_KEYS.issubset(d.keys()):
        return False
    # Validate run_id is a non-null string
    if not isinstance(d.get("run_id"), str):
        return False
    # Validate started_at is a non-null string
    if not isinstance(d.get("started_at"), str):
        return False
    # Validate issues is a dict (or None which we treat as empty)
    issues = d.get("issues")
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
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        print(f"Warning: skipping corrupt file {path}: {e}", file=sys.stderr)
        return None


def _count_issue_statuses(issues: dict[str, Any]) -> tuple[int, int, int, int]:
    """Count issue statuses from issues dict.

    Args:
        issues: Dict of issue_id -> issue data (values should be dicts).

    Returns:
        Tuple of (total, success, failed, timeout) counts.
        Non-dict values are counted in total but not in status categories.
    """
    total = len(issues)
    success = 0
    failed = 0
    timeout = 0
    for issue in issues.values():
        # Guard against non-dict issue values
        if not isinstance(issue, dict):
            continue
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


def _extract_timestamp_prefix(filename: str) -> str:
    """Extract the timestamp prefix from a run filename.

    Filenames have format: {timestamp}_{short_id}.json
    where timestamp is %Y-%m-%dT%H-%M-%S (second resolution).

    Args:
        filename: The filename (not full path).

    Returns:
        The timestamp prefix (everything before the first underscore),
        or empty string if no underscore found.
    """
    underscore_pos = filename.find("_")
    if underscore_pos == -1:
        return ""
    return filename[:underscore_pos]


def _collect_runs(files: list[Path], limit: int | None = None) -> list[dict[str, Any]]:
    """Collect valid run metadata from files, stopping early if limit reached.

    Files are assumed to be pre-sorted by filename descending (newest first),
    allowing early termination once `limit` valid runs are collected.

    To preserve correctness when multiple runs share the same second-resolution
    timestamp, this function continues scanning files with the same timestamp
    prefix as the last-collected file, then returns all candidates for final
    sorting and slicing by the caller.

    Args:
        files: List of JSON file paths (pre-sorted newest first).
        limit: Maximum number of runs to collect. None means collect all.

    Returns:
        List of run metadata dicts with 'metadata_path' added.
        Caller must apply final sort and slice to limit for correctness.
    """
    runs: list[dict[str, Any]] = []
    boundary_prefix: str | None = None

    for path in files:
        # If we've reached limit, only continue for files with same timestamp prefix
        if limit is not None and len(runs) >= limit:
            if boundary_prefix is None:
                # Record the timestamp prefix of the last-collected file
                boundary_prefix = _extract_timestamp_prefix(runs[-1]["metadata_path"])
                # Use the filename from metadata_path (which is full path)
                boundary_prefix = _extract_timestamp_prefix(
                    Path(runs[-1]["metadata_path"]).name
                )

            current_prefix = _extract_timestamp_prefix(path.name)
            if current_prefix != boundary_prefix:
                # Different timestamp prefix, safe to stop
                break

        data = _parse_run_file(path)
        if data is not None:
            # Add metadata_path to a copy to avoid mutating parsed data
            run_entry = {**data, "metadata_path": str(path)}
            runs.append(run_entry)

    return runs


def _format_null(value: object) -> str:
    """Format value for table display, showing '-' for None."""
    return "-" if value is None else str(value)


def _get_repo_path(run: dict[str, Any]) -> str | None:
    """Get repo path from run metadata, preferring stored value.

    Falls back to the encoded directory name if repo_path is not in metadata.
    Note: The directory name encoding is lossy, so the fallback may not be exact.
    """
    # Prefer exact repo_path from metadata if available
    stored_path = run.get("repo_path")
    if isinstance(stored_path, str):
        return stored_path
    # Fallback: return encoded directory name (not decoded, since encoding is lossy)
    parent_name = Path(run["metadata_path"]).parent.name
    return parent_name if parent_name else None


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
            min=1,
        ),
    ] = _DEFAULT_LIMIT,
) -> None:
    """List recent mala runs."""
    files = _discover_run_files(all_runs)
    # Collect with limit (files are pre-sorted newest first by filename)
    # _collect_runs may return more than limit if ties exist at the boundary
    runs = _collect_runs(files, limit=limit)

    # Final sort for determinism (handles ties within the collected set)
    runs = _sort_runs(runs)

    # Slice to limit after sorting to ensure correct top-N
    runs = runs[:limit]

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
                entry["repo_path"] = _get_repo_path(run)
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
                row.insert(1, _format_null(_get_repo_path(run)))
            rows.append(row)

        print(tabulate(rows, headers=headers, tablefmt="simple"))


def _extract_sessions(
    runs: list[dict[str, Any]], issue_filter: str
) -> list[dict[str, Any]]:
    """Extract per-issue session rows from runs.

    Args:
        runs: List of run metadata dicts.
        issue_filter: Only include sessions matching this issue ID (exact match).

    Returns:
        List of session dicts with keys: run_id, session_id, issue_id, run_started_at,
        status, log_path, metadata_path, repo_path.
    """
    sessions: list[dict[str, Any]] = []
    for run in runs:
        issues = run.get("issues", {})
        if not isinstance(issues, dict):
            continue
        for issue_id, issue_data in issues.items():
            # Skip if filter doesn't match (exact, case-sensitive)
            if issue_id != issue_filter:
                continue
            if not isinstance(issue_data, dict):
                continue
            sessions.append(
                {
                    "run_id": run["run_id"],
                    "session_id": issue_data.get("session_id"),
                    "issue_id": issue_id,
                    "run_started_at": run.get("started_at"),
                    "status": issue_data.get("status"),
                    "log_path": issue_data.get("log_path"),
                    "metadata_path": run.get("metadata_path"),
                    "repo_path": _get_repo_path(run),
                }
            )
    return sessions


def _sort_sessions(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort sessions by run_started_at desc, then run_id asc, then issue_id asc.

    Args:
        sessions: List of session dicts.

    Returns:
        Sorted list.
    """
    return sorted(
        sessions,
        key=lambda s: (
            -_parse_timestamp(s.get("run_started_at") or ""),
            s.get("run_id") or "",
            s.get("issue_id") or "",
        ),
    )


@logs_app.command()
def sessions(
    issue: Annotated[
        str,
        typer.Option(
            "--issue",
            help="Filter by issue ID (exact match, case-sensitive)",
        ),
    ],
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
            help="Show sessions from all repos (not just current)",
        ),
    ] = False,
) -> None:
    """List Claude sessions from mala runs."""
    files = _discover_run_files(all_sessions)
    runs = _collect_runs(files)
    session_rows = _extract_sessions(runs, issue)
    session_rows = _sort_sessions(session_rows)

    if not session_rows:
        if json_output:
            print("[]")
        else:
            print("No sessions found")
        return

    if json_output:
        output = []
        for s in session_rows:
            entry: dict[str, Any] = {
                "run_id": s["run_id"],
                "session_id": s["session_id"],
                "issue_id": s["issue_id"],
                "run_started_at": s["run_started_at"],
                "status": s["status"],
                "log_path": s["log_path"],
            }
            if all_sessions:
                entry["repo_path"] = s["repo_path"]
            output.append(entry)
        print(json.dumps(output, indent=2))
    else:
        headers = [
            "run_id",
            "session_id",
            "issue_id",
            "run_started_at",
            "status",
            "log_path",
        ]
        if all_sessions:
            headers.insert(1, "repo_path")

        rows = []
        for s in session_rows:
            row = [
                s["run_id"][:8],  # Short ID for display
                _format_null(s["session_id"]),
                _format_null(s["issue_id"]),
                _format_null(s["run_started_at"]),
                _format_null(s["status"]),
                _format_null(s["log_path"]),
            ]
            if all_sessions:
                row.insert(1, _format_null(s["repo_path"]))
            rows.append(row)

        print(tabulate(rows, headers=headers, tablefmt="simple"))


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

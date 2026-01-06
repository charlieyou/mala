"""Logs subcommand for mala CLI: search and inspect run logs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from tabulate import tabulate

from src.infra.io.log_output.run_metadata import (
    _parse_run_file,
    discover_run_files,
    find_sessions_for_issue,
    load_runs,
    parse_timestamp,
)
from src.infra.tools.env import get_repo_runs_dir, get_runs_dir

logs_app = typer.Typer(name="logs", help="Search and inspect mala run logs")

# Default limit for number of runs to display
_DEFAULT_LIMIT = 20


def _discover_run_files_all_repos() -> list[Path]:
    """Discover run metadata JSON files across all repos.

    This is only used for --all flag in list and sessions commands.

    Returns:
        List of JSON file paths sorted by filename descending (newest first).
    """
    runs_dir = get_runs_dir()
    if not runs_dir.exists():
        return []
    # rglob for all .json files across all repo directories
    files = list(runs_dir.rglob("*.json"))
    return sorted(files, key=lambda p: p.name, reverse=True)


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
    return sorted(runs, key=lambda r: (-parse_timestamp(r["started_at"]), r["run_id"]))


def _parse_run_file_cli(path: Path) -> dict[str, Any] | None:
    """Parse a run metadata JSON file with CLI-specific error handling.

    Delegates to run_metadata._parse_run_file with _warn_corrupt_file callback
    to print warnings to stderr for CLI user visibility.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed dict if valid, None if corrupt or missing required keys.
        Prints warning to stderr for corrupt files.
    """
    return _parse_run_file(path, on_corrupt=_warn_corrupt_file)


def _warn_corrupt_file(path: Path, exc: Exception | None) -> None:
    """Print a CLI warning for corrupt or invalid run files.

    Args:
        path: Path to the corrupt file.
        exc: Exception if parse/IO error, None if validation failure.
    """
    if exc is not None:
        print(f"Warning: skipping corrupt file {path}: {exc}", file=sys.stderr)
    else:
        print(f"Warning: skipping invalid file {path}", file=sys.stderr)


def _collect_runs_with_paths(
    files: list[Path], limit: int | None = None
) -> list[dict[str, Any]]:
    """Collect runs from files using shared load_runs, adding metadata_path.

    Thin wrapper around infra's load_runs() that adds 'metadata_path' field
    for CLI display compatibility. Uses on_corrupt callback to print warnings
    to stderr for CLI user visibility.

    Args:
        files: List of JSON file paths (pre-sorted newest first).
        limit: Maximum number of runs to collect. None means collect all.

    Returns:
        List of run metadata dicts with 'metadata_path' added.
    """
    runs_with_paths = load_runs(files, limit=limit, on_corrupt=_warn_corrupt_file)
    return [{**data, "metadata_path": str(path)} for data, path in runs_with_paths]


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
    # Use all-repos discovery for --all, otherwise current repo
    files = _discover_run_files_all_repos() if all_runs else discover_run_files(None)
    # Collect with limit (files are pre-sorted newest first by filename)
    # load_runs may return more than limit if ties exist at the boundary
    runs = _collect_runs_with_paths(files, limit=limit)

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


def _find_sessions_all_repos(issue_id: str) -> list[dict[str, Any]]:
    """Find sessions across all repos for --all flag.

    Uses _discover_run_files_all_repos and the shared extract_session_from_run
    function to avoid duplicating session extraction logic. Uses on_corrupt
    callback to print warnings to stderr for CLI user visibility.

    Args:
        issue_id: The issue ID to filter by (exact match).

    Returns:
        List of session dicts sorted by run_started_at descending.
    """
    from src.infra.io.log_output.run_metadata import extract_session_from_run

    files = _discover_run_files_all_repos()
    runs_with_paths = load_runs(files, on_corrupt=_warn_corrupt_file)
    sessions: list[dict[str, Any]] = []

    for data, path in runs_with_paths:
        session_info = extract_session_from_run(data, path, issue_id)
        if session_info is None:
            continue

        # Convert SessionInfo to dict for CLI display
        sessions.append(
            {
                "run_id": session_info.run_id,
                "session_id": session_info.session_id,
                "issue_id": session_info.issue_id,
                "run_started_at": session_info.run_started_at,
                "status": session_info.status,
                "log_path": session_info.log_path,
                "metadata_path": str(session_info.metadata_path),
                "repo_path": session_info.repo_path,
            }
        )

    # Sort by started_at descending, then run_id ascending for ties
    return sorted(
        sessions,
        key=lambda s: (
            -parse_timestamp(s.get("run_started_at") or ""),
            s.get("run_id") or "",
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
    if all_sessions:
        # Search all repos
        session_rows = _find_sessions_all_repos(issue)
    else:
        # Use shared infra function for current repo
        session_infos = find_sessions_for_issue(
            None, issue, on_corrupt=_warn_corrupt_file
        )
        # Convert SessionInfo to dict for CLI display
        session_rows = [
            {
                "run_id": s.run_id,
                "session_id": s.session_id,
                "issue_id": s.issue_id,
                "run_started_at": s.run_started_at,
                "status": s.status,
                "log_path": s.log_path,
                "metadata_path": str(s.metadata_path),
                "repo_path": s.repo_path,
            }
            for s in session_infos
        ]

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
            run_id = s.get("run_id") or ""
            row = [
                run_id[:8],  # Short ID for display
                _format_null(s.get("session_id")),
                _format_null(s.get("issue_id")),
                _format_null(s.get("run_started_at")),
                _format_null(s.get("status")),
                _format_null(s.get("log_path")),
            ]
            if all_sessions:
                row.insert(1, _format_null(s["repo_path"]))
            rows.append(row)

        print(tabulate(rows, headers=headers, tablefmt="simple"))


def _extract_run_id_prefix_from_filename(filename: str) -> str | None:
    """Extract the 8-char run_id prefix from a run filename.

    Filenames have format: {timestamp}_{short_id}.json
    where short_id is the first 8 characters of the run_id.

    Args:
        filename: The filename (not full path).

    Returns:
        The 8-char run_id prefix, or None if parsing fails or length != 8.
    """
    # Remove .json extension
    if not filename.endswith(".json"):
        return None
    base = filename[:-5]  # Remove ".json"

    # Find last underscore to split timestamp from run_id prefix
    underscore_pos = base.rfind("_")
    if underscore_pos == -1:
        return None

    prefix = base[underscore_pos + 1 :]
    # Validate length is exactly 8 chars
    if len(prefix) != 8:
        return None
    return prefix


def _find_matching_runs(
    search_id: str,
) -> tuple[list[tuple[Path, dict[str, Any]]], list[Path]]:
    """Find runs matching the given run_id or prefix.

    Search strategy:
    1. Scan conforming files: use filename prefix to prune, track corrupt candidates
    2. Fallback: only scan non-conforming files if no matches found

    Args:
        search_id: Full UUID or prefix (any length >= 1) to search for.

    Returns:
        Tuple of (matches, corrupt_files) where:
        - matches: List of (path, parsed_data) tuples for matching runs
        - corrupt_files: List of paths that matched by prefix but were corrupt
    """
    cwd = Path.cwd()
    repo_runs_dir = get_repo_runs_dir(cwd)
    if not repo_runs_dir.exists():
        return [], []

    files = list(repo_runs_dir.glob("*.json"))
    search_prefix = search_id[:8] if len(search_id) >= 8 else search_id

    matches: list[tuple[Path, dict[str, Any]]] = []
    corrupt_files: list[Path] = []
    non_conforming_files: list[Path] = []

    # First pass: scan conforming files, prune by filename prefix
    for path in files:
        filename_prefix = _extract_run_id_prefix_from_filename(path.name)

        if filename_prefix is None:
            # Non-conforming filename - save for fallback
            non_conforming_files.append(path)
            continue

        # Prune: filename prefix must start with search prefix
        if not filename_prefix.startswith(search_prefix):
            continue

        # Parse and check
        data = _parse_run_file_cli(path)
        if data is None:
            corrupt_files.append(path)
            continue

        # Match if run_id starts with search_id
        run_id_field = data.get("run_id", "")
        if isinstance(run_id_field, str) and run_id_field.startswith(search_id):
            matches.append((path, data))

    # Fallback: scan non-conforming files only if no matches found
    if not matches:
        for path in non_conforming_files:
            data = _parse_run_file_cli(path)
            if data is None:
                # Track corrupt for non-conforming files too
                corrupt_files.append(path)
                continue

            run_id_field = data.get("run_id", "")
            if isinstance(run_id_field, str) and run_id_field.startswith(search_id):
                matches.append((path, data))

    return matches, corrupt_files


def _print_run_details(run: dict[str, Any]) -> None:
    """Print run details in human-readable table format.

    Args:
        run: Run metadata dict with all fields.
    """
    # Basic metadata
    print(f"Run ID:       {run.get('run_id', '-')}")
    print(f"Started:      {run.get('started_at', '-')}")
    if run.get("ended_at"):
        print(f"Ended:        {run['ended_at']}")
    if run.get("status"):
        print(f"Status:       {run['status']}")
    if run.get("repo_path"):
        print(f"Repo:         {run['repo_path']}")
    print(f"Metadata:     {run.get('metadata_path', '-')}")

    # Issues section
    issues = run.get("issues")
    if not isinstance(issues, dict) or not issues:
        print("\nIssues:       (none)")
    else:
        print(f"\nIssues ({len(issues)}):")
        for issue_id, issue_data in sorted(issues.items()):
            if not isinstance(issue_data, dict):
                print(f"  {issue_id}: (invalid data)")
                continue

            status = issue_data.get("status", "-")
            session_id = issue_data.get("session_id", "-")
            log_path = issue_data.get("log_path", "-")

            print(f"  {issue_id}:")
            print(f"    Status:     {status}")
            print(f"    Session:    {session_id}")
            print(f"    Log:        {log_path}")


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
    matches, corrupt_files = _find_matching_runs(run_id)
    # Full UUID is 36 chars (8-4-4-4-12 with hyphens); anything shorter is a prefix
    is_prefix_search = len(run_id) < 36

    # For prefix search: treat valid matches + corrupt files as potentially ambiguous
    # (corrupt file could be the intended run)
    total_candidates = len(matches) + len(corrupt_files)

    # Handle ambiguous prefix (multiple valid matches, or mix of valid + corrupt)
    if len(matches) > 1 or (
        is_prefix_search and total_candidates > 1 and corrupt_files
    ):
        match_ids = sorted([m[1].get("run_id", "unknown") for m in matches])
        corrupt_note = (
            f" ({len(corrupt_files)} additional corrupt file(s))"
            if corrupt_files
            else ""
        )
        if json_output:
            error_obj: dict[str, Any] = {
                "error": "ambiguous_prefix",
                "message": f"Ambiguous prefix '{run_id}' matches multiple runs{corrupt_note}",
                "matches": match_ids,
            }
            if corrupt_files:
                error_obj["corrupt_count"] = len(corrupt_files)
            print(json.dumps(error_obj, indent=2))
        else:
            print(
                f"Error: Ambiguous prefix '{run_id}' matches multiple runs{corrupt_note}:"
            )
            for mid in match_ids:
                print(f"  {mid}")
            if corrupt_files:
                print(f"  ({len(corrupt_files)} corrupt file(s) also matched)")
        raise typer.Exit(1)

    # Handle not found
    if len(matches) == 0:
        # For prefix search with corrupt files: report corrupt (exit 2)
        # For full UUID with corrupt files: report not-found with note (exit 1)
        # (corrupt file only matched by 8-char prefix, not confirmed as the UUID)
        if corrupt_files and is_prefix_search:
            if json_output:
                error_obj = {
                    "error": "corrupt",
                    "message": f"Run matching '{run_id}' found but file is corrupt",
                }
                print(json.dumps(error_obj, indent=2))
            else:
                print(f"Error: Run matching '{run_id}' found but file is corrupt")
            raise typer.Exit(2)

        # Not found (possibly with corrupt prefix-matches for full UUID)
        corrupt_note = (
            f" (note: {len(corrupt_files)} corrupt file(s) with matching prefix)"
            if corrupt_files
            else ""
        )
        if json_output:
            error_obj = {
                "error": "not_found",
                "message": f"No run found matching '{run_id}'{corrupt_note}",
            }
            print(json.dumps(error_obj, indent=2))
        else:
            print(f"Error: No run found matching '{run_id}'{corrupt_note}")
        raise typer.Exit(1)

    # Single match found
    path, data = matches[0]

    # Add metadata_path to output
    output_data = {**data, "metadata_path": str(path)}

    if json_output:
        print(json.dumps(output_data, indent=2))
    else:
        _print_run_details(output_data)

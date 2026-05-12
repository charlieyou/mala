"""Read Cerberus v2 per-reviewer findings from iteration state."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

from src.infra.clients.cerberus_output_parser import ReviewIssue

if TYPE_CHECKING:
    from pathlib import Path


def latest_iteration_dir(iterations_dir: Path) -> tuple[Path | None, str | None]:
    """Return the highest integer-named iteration directory."""
    if not iterations_dir.exists():
        return None, f"Missing iterations directory: {iterations_dir}"
    if not iterations_dir.is_dir():
        return None, f"Iterations path is not a directory: {iterations_dir}"

    candidates: list[tuple[int, Path]] = []
    try:
        children = list(iterations_dir.iterdir())
    except OSError as e:
        return None, f"Could not read iterations directory {iterations_dir}: {e}"

    for child in children:
        if child.is_dir():
            try:
                candidates.append((int(child.name), child))
            except ValueError:
                continue

    if not candidates:
        return None, f"No numeric iteration directories found in {iterations_dir}"

    return max(candidates, key=lambda item: item[0])[1], None


def latest_round_dir(iteration_dir: Path) -> tuple[Path | None, str | None]:
    """Return the highest round-N directory for an iteration."""
    if not iteration_dir.exists():
        return None, f"Missing iteration directory: {iteration_dir}"
    if not iteration_dir.is_dir():
        return None, f"Iteration path is not a directory: {iteration_dir}"

    candidates: list[tuple[int, Path]] = []
    try:
        children = list(iteration_dir.iterdir())
    except OSError as e:
        return None, f"Could not read iteration directory {iteration_dir}: {e}"

    for child in children:
        if not child.is_dir() or not child.name.startswith("round-"):
            continue
        round_text = child.name.removeprefix("round-")
        try:
            candidates.append((int(round_text), child))
        except ValueError:
            continue

    if not candidates:
        return None, f"No round directories found in {iteration_dir}"

    return max(candidates, key=lambda item: item[0])[1], None


def read_findings(
    state_root: Path,
    project_key: str,
    run_key: str,
    *,
    iteration: int | None = None,
    round_index: int | None = None,
) -> tuple[list[ReviewIssue], list[str]]:
    """Read findings from the latest Cerberus v2 iteration/round on disk."""
    parse_errors: list[str] = []
    iterations_dir = state_root / project_key / run_key / "iterations"

    if iteration is None:
        iteration_dir, error = latest_iteration_dir(iterations_dir)
        if error is not None:
            return [], [error]
        if iteration_dir is None:
            return [], [f"No iteration directory selected from {iterations_dir}"]
    else:
        iteration_dir = iterations_dir / str(iteration)
        if not iteration_dir.is_dir():
            return [], [f"Missing iteration directory: {iteration_dir}"]

    if round_index is None:
        round_dir, error = latest_round_dir(iteration_dir)
        if error is not None:
            return [], [error]
        if round_dir is None:
            return [], [f"No round directory selected from {iteration_dir}"]
    else:
        round_dir = iteration_dir / f"round-{round_index}"
        if not round_dir.is_dir():
            return [], [f"Missing round directory: {round_dir}"]

    reviewers_dir = round_dir / "reviewers"
    if not reviewers_dir.is_dir():
        return [], [f"Missing reviewers directory: {reviewers_dir}"]

    try:
        reviewer_dirs = sorted(
            (path for path in reviewers_dir.iterdir() if path.is_dir()),
            key=lambda path: path.name,
        )
    except OSError as e:
        return [], [f"Could not read reviewers directory {reviewers_dir}: {e}"]

    if not reviewer_dirs:
        return [], [f"No reviewer directories found in {reviewers_dir}"]

    issues: list[ReviewIssue] = []
    for reviewer_dir in reviewer_dirs:
        output_path = reviewer_dir / "output.json"
        if not output_path.is_file():
            parse_errors.append(f"Missing reviewer output file: {output_path}")
            continue

        reviewer_issues, error = _read_reviewer_output(output_path, reviewer_dir.name)
        if error is not None:
            parse_errors.append(error)
            continue
        issues.extend(reviewer_issues)

    return issues, parse_errors


def _read_reviewer_output(
    output_path: Path, reviewer: str
) -> tuple[list[ReviewIssue], str | None]:
    try:
        raw = output_path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        return [], f"Malformed reviewer output JSON at {output_path}: {e}"
    except OSError as e:
        return [], f"Could not read reviewer output file {output_path}: {e}"

    if not isinstance(data, dict):
        return [], f"Reviewer output root is not an object at {output_path}"

    findings = data.get("findings", [])
    if findings is None:
        findings = []
    if not isinstance(findings, list):
        return [], f"Reviewer output findings is not an array at {output_path}"

    issues: list[ReviewIssue] = []
    for index, finding in enumerate(findings):
        if not isinstance(finding, dict):
            return [], f"Finding {index} is not an object at {output_path}"

        issue, error = _parse_finding(
            cast("dict[str, Any]", finding), reviewer, output_path, index
        )
        if error is not None:
            return [], error
        if issue is None:
            return [], f"Finding {index} could not be parsed at {output_path}"
        issues.append(issue)

    return issues, None


def _parse_finding(
    finding: dict[str, Any],
    reviewer: str,
    output_path: Path,
    index: int,
) -> tuple[ReviewIssue | None, str | None]:
    file_path = finding.get("file_path")
    if file_path is None:
        file_value = ""
    elif isinstance(file_path, str):
        file_value = file_path
    else:
        return None, _field_error(output_path, index, "file_path", "string or null")

    line_start = finding.get("line_start")
    if line_start is None:
        line_start_value = 0
    elif isinstance(line_start, int):
        line_start_value = line_start
    else:
        return None, _field_error(output_path, index, "line_start", "integer or null")

    line_end = finding.get("line_end")
    if line_end is None:
        line_end_value = 0
    elif isinstance(line_end, int):
        line_end_value = line_end
    else:
        return None, _field_error(output_path, index, "line_end", "integer or null")

    priority = finding.get("priority")
    if priority is not None and not isinstance(priority, int):
        return None, _field_error(output_path, index, "priority", "integer or null")

    title = finding.get("title", "")
    if not isinstance(title, str):
        return None, _field_error(output_path, index, "title", "string")

    body = finding.get("body", "")
    if not isinstance(body, str):
        return None, _field_error(output_path, index, "body", "string")

    return (
        ReviewIssue(
            file=file_value,
            line_start=line_start_value,
            line_end=line_end_value,
            priority=priority,
            title=title,
            body=body,
            reviewer=reviewer,
        ),
        None,
    )


def _field_error(output_path: Path, index: int, field: str, expected: str) -> str:
    return f"Finding {index}: '{field}' must be a {expected} at {output_path}"

"""Cerberus review-gate adapter for mala orchestrator.

This module provides a thin adapter for the Cerberus review-gate CLI.
It handles:
- Exit code mapping to ReviewResult
- JSON parsing of review output
- Issue formatting for follow-up prompts

The adapter focuses on parsing and mapping; CLI execution in repo_path
is handled by the caller.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ReviewIssue:
    """A single issue found during external review.

    Matches the Cerberus JSON schema for issues.
    """

    file: str
    line_start: int
    line_end: int
    priority: int | None  # 0=P0, 1=P1, 2=P2, 3=P3, or None
    title: str
    body: str
    reviewer: str  # Which reviewer found this issue


@dataclass
class ReviewResult:
    """Result of a Cerberus review-gate review.

    Satisfies the ReviewOutcome protocol in lifecycle.py.
    """

    passed: bool
    issues: list[ReviewIssue] = field(default_factory=list)
    parse_error: str | None = None
    fatal_error: bool = False
    review_log_path: Path | None = None


def parse_cerberus_json(output: str) -> tuple[bool, list[ReviewIssue], str | None]:
    """Parse Cerberus review-gate JSON output.

    Args:
        output: JSON string from review-gate wait --json.

    Returns:
        Tuple of (passed, issues, parse_error).
        If parse_error is not None, passed will be False and issues empty.
    """
    if not output or not output.strip():
        return False, [], "Empty output from review-gate"

    try:
        data = json.loads(output)
    except json.JSONDecodeError as e:
        return False, [], f"JSON parse error: {e}"

    if not isinstance(data, dict):
        return False, [], "Root element is not an object"

    # Check consensus verdict
    consensus = data.get("consensus", {})
    if not isinstance(consensus, dict):
        return False, [], "'consensus' field must be an object"

    verdict = consensus.get("verdict")
    if verdict not in ("PASS", "FAIL", "NEEDS_WORK", "no_reviewers"):
        return False, [], f"Invalid verdict: {verdict}"

    passed = verdict == "PASS"

    # Parse issues
    raw_issues = data.get("issues", [])
    if not isinstance(raw_issues, list):
        return False, [], "'issues' field must be an array"

    issues: list[ReviewIssue] = []
    for i, item in enumerate(raw_issues):
        if not isinstance(item, dict):
            return False, [], f"Issue {i} is not an object"

        reviewer = item.get("reviewer", "")
        if not isinstance(reviewer, str):
            return False, [], f"Issue {i}: 'reviewer' must be a string"

        file_path = item.get("file", "")
        if not isinstance(file_path, str):
            return False, [], f"Issue {i}: 'file' must be a string"

        line_start = item.get("line_start", 0)
        if not isinstance(line_start, int):
            return False, [], f"Issue {i}: 'line_start' must be an integer"

        line_end = item.get("line_end", 0)
        if not isinstance(line_end, int):
            return False, [], f"Issue {i}: 'line_end' must be an integer"

        priority = item.get("priority")
        if priority is not None and not isinstance(priority, int):
            return False, [], f"Issue {i}: 'priority' must be an integer or null"

        title = item.get("title", "")
        if not isinstance(title, str):
            return False, [], f"Issue {i}: 'title' must be a string"

        body = item.get("body", "")
        if not isinstance(body, str):
            return False, [], f"Issue {i}: 'body' must be a string"

        issues.append(
            ReviewIssue(
                file=file_path,
                line_start=line_start,
                line_end=line_end,
                priority=priority,
                title=title,
                body=body,
                reviewer=reviewer,
            )
        )

    return passed, issues, None


def map_exit_code_to_result(
    exit_code: int,
    stdout: str,
    stderr: str,
    review_log_path: Path | None = None,
) -> ReviewResult:
    """Map Cerberus review-gate exit code to ReviewResult.

    Exit codes:
        0 - PASS: all reviewers agree, no issues
        1 - FAIL/NEEDS_WORK: legitimate review failure
        2 - Parse error: malformed reviewer output
        3 - Timeout: reviewers didn't respond in time
        4 - No reviewers: no reviewer CLIs available
        5 - Internal error: unexpected failure

    Args:
        exit_code: Exit code from review-gate wait command.
        stdout: Stdout from the command (JSON output).
        stderr: Stderr from the command (error messages).
        review_log_path: Optional path to review session logs.

    Returns:
        ReviewResult with appropriate fields set.
    """
    # Exit codes 4 and 5 are fatal errors
    if exit_code == 4:
        return ReviewResult(
            passed=False,
            issues=[],
            parse_error="No reviewers available",
            fatal_error=True,
            review_log_path=review_log_path,
        )

    if exit_code == 5:
        error_msg = stderr.strip() if stderr else "Internal error"
        return ReviewResult(
            passed=False,
            issues=[],
            parse_error=error_msg,
            fatal_error=True,
            review_log_path=review_log_path,
        )

    # Exit code 3 is timeout (retryable)
    if exit_code == 3:
        return ReviewResult(
            passed=False,
            issues=[],
            parse_error="timeout",
            fatal_error=False,
            review_log_path=review_log_path,
        )

    # Exit code 2 is parse error (retryable)
    if exit_code == 2:
        # Try to extract error from JSON parse_errors array
        parse_error_msg = "Parse error"
        try:
            data = json.loads(stdout)
            if isinstance(data, dict):
                parse_errors = data.get("parse_errors", [])
                if isinstance(parse_errors, list) and parse_errors:
                    parse_error_msg = "; ".join(str(e) for e in parse_errors)
        except (json.JSONDecodeError, TypeError):
            if stderr:
                parse_error_msg = stderr.strip()
        return ReviewResult(
            passed=False,
            issues=[],
            parse_error=parse_error_msg,
            fatal_error=False,
            review_log_path=review_log_path,
        )

    # Exit codes 0 and 1: parse JSON output
    json_passed, issues, parse_error = parse_cerberus_json(stdout)

    if parse_error:
        # JSON parsing failed - treat as parse error (exit code 2 equivalent)
        return ReviewResult(
            passed=False,
            issues=[],
            parse_error=parse_error,
            fatal_error=False,
            review_log_path=review_log_path,
        )

    # Derive passed status from exit code
    exit_passed = exit_code == 0

    # Warn if exit code and JSON verdict disagree
    if json_passed != exit_passed:
        from src.log_output.console import Colors, log

        log(
            "!",
            f"Exit code ({exit_code}) and JSON verdict "
            f"({'PASS' if json_passed else 'FAIL'}) disagree; "
            f"fail-closed: requiring both to pass",
            color=Colors.YELLOW,
        )

    # Security: fail-closed - BOTH exit code AND JSON verdict must pass
    # This prevents a review from passing when the consensus verdict is
    # FAIL, NEEDS_WORK, or no_reviewers even if exit code is 0
    final_passed = exit_passed and json_passed

    return ReviewResult(
        passed=final_passed,
        issues=issues,
        parse_error=None,
        fatal_error=False,
        review_log_path=review_log_path,
    )


def _to_relative_path(file_path: str, base_path: Path | None = None) -> str:
    """Convert an absolute file path to a relative path for display.

    Strips the base path prefix to show paths relative to the repository root.

    Args:
        file_path: Absolute or relative file path.
        base_path: Base path (typically repository root) for relativization.
            If None, uses Path.cwd() as fallback.

    Returns:
        Relative path suitable for display. If relativization fails,
        returns the original path to preserve directory context.
    """
    # If already relative, return as-is
    if not file_path.startswith("/"):
        return file_path

    # Use provided base_path or fall back to cwd
    base = base_path.resolve() if base_path else Path.cwd()
    try:
        abs_path = Path(file_path)
        if abs_path.is_relative_to(base):
            return str(abs_path.relative_to(base))
    except (ValueError, OSError):
        pass

    # Preserve original path if relativization fails (don't strip to filename)
    return file_path


def format_review_issues(
    issues: list[ReviewIssue], base_path: Path | None = None
) -> str:
    """Format review issues as a human-readable string for follow-up prompts.

    Args:
        issues: List of ReviewIssue objects to format.
        base_path: Base path (typically repository root) for path relativization.
            If None, uses Path.cwd() as fallback.

    Returns:
        Formatted string with issues grouped by file.
    """
    if not issues:
        return "No specific issues found."

    lines = []
    current_file = None

    # Sort by file, then by line, then by priority (lower = more important)
    sorted_issues = sorted(
        issues,
        key=lambda x: (
            x.file,
            x.line_start,
            x.priority if x.priority is not None else 4,
        ),
    )

    for issue in sorted_issues:
        # Convert absolute paths to relative for cleaner display
        display_file = _to_relative_path(issue.file, base_path)
        if display_file != current_file:
            if current_file is not None:
                lines.append("")  # Blank line between files
            current_file = display_file
            lines.append(f"File: {display_file}")

        loc = (
            f"L{issue.line_start}-{issue.line_end}"
            if issue.line_start != issue.line_end
            else f"L{issue.line_start}"
        )
        # Include reviewer attribution
        reviewer_tag = f"[{issue.reviewer}]" if issue.reviewer else ""
        lines.append(f"  {loc}: {reviewer_tag} {issue.title}")
        if issue.body:
            lines.append(f"    {issue.body}")

    return "\n".join(lines)

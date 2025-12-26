"""Codex review integration for mala orchestrator.

Runs code reviews using Codex CLI with strict JSON output parsing.
Implements retry logic for JSON parse failures (fail-closed behavior).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ReviewIssue:
    """A single issue found during code review."""

    file: str
    line: int | None
    severity: str  # "error", "warning", "info"
    message: str


@dataclass
class CodexReviewResult:
    """Result of a Codex code review."""

    passed: bool
    issues: list[ReviewIssue] = field(default_factory=list)
    raw_output: str = ""
    parse_error: str | None = None
    attempt: int = 1


# JSON schema prompt (ASCII-only, no code fences in expected output)
CODEX_REVIEW_PROMPT = """\
Review the commit and output your findings as a JSON object.
Use ASCII characters only. Do not wrap the JSON in markdown code fences.

Required JSON schema:
{
  "passed": boolean,
  "issues": [
    {
      "file": "path/to/file.py",
      "line": 42,
      "severity": "error" | "warning" | "info",
      "message": "Description of the issue"
    }
  ]
}

Rules:
- passed must be false if there are any issues with severity "error"
- passed can be true if there are only warnings or info
- line can be null if the issue is file-level
- Keep messages concise and actionable

Output only the JSON object, nothing else.
"""

# Stricter retry prompt for JSON parse failures
CODEX_REVIEW_RETRY_PROMPT = """\
Your previous response was not valid JSON. Output ONLY the JSON object below.
No explanation, no code fences, no extra text. ASCII characters only.

{
  "passed": boolean,
  "issues": [{"file": "...", "line": N, "severity": "...", "message": "..."}]
}
"""


def _extract_json(text: str) -> str | None:
    """Try to extract JSON object from text that may have extra content."""
    # Try to find JSON object in the text
    # Look for the outermost { ... } pair
    start = text.find("{")
    if start == -1:
        return None

    # Track brace depth to find matching close
    depth = 0
    in_string = False
    escape_next = False
    end = -1

    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        return None

    return text[start:end]


def _parse_review_json(output: str) -> tuple[bool, list[ReviewIssue], str | None]:
    """Parse Codex review output as JSON.

    Returns:
        Tuple of (passed, issues, parse_error).
        If parse_error is not None, passed will be False and issues empty.
    """
    # Try to extract JSON from potential surrounding text
    json_str = _extract_json(output)
    if json_str is None:
        return False, [], "No JSON object found in output"

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return False, [], f"JSON parse error: {e}"

    if not isinstance(data, dict):
        return False, [], "Root element is not an object"

    passed = data.get("passed")
    if not isinstance(passed, bool):
        return False, [], "'passed' field must be a boolean"

    raw_issues = data.get("issues", [])
    if not isinstance(raw_issues, list):
        return False, [], "'issues' field must be an array"

    issues: list[ReviewIssue] = []
    for i, item in enumerate(raw_issues):
        if not isinstance(item, dict):
            return False, [], f"Issue {i} is not an object"

        file = item.get("file")
        if not isinstance(file, str):
            return False, [], f"Issue {i}: 'file' must be a string"

        line = item.get("line")
        if line is not None and not isinstance(line, int):
            return False, [], f"Issue {i}: 'line' must be an integer or null"

        severity = item.get("severity", "info")
        if severity not in ("error", "warning", "info"):
            return False, [], f"Issue {i}: invalid severity '{severity}'"

        message = item.get("message", "")
        if not isinstance(message, str):
            return False, [], f"Issue {i}: 'message' must be a string"

        issues.append(
            ReviewIssue(
                file=file,
                line=line,
                severity=severity,
                message=message,
            )
        )

    return passed, issues, None


async def run_codex_review(
    repo_path: Path,
    commit_sha: str,
    max_retries: int = 2,
) -> CodexReviewResult:
    """Run Codex code review on a commit with JSON output and retry logic.

    Args:
        repo_path: Path to the git repository.
        commit_sha: The commit SHA to review.
        max_retries: Maximum number of attempts (default 2 = 1 initial + 1 retry).

    Returns:
        CodexReviewResult with review outcome. If JSON parsing fails after
        all retries, the result will have passed=False (fail-closed).
    """
    for attempt in range(1, max_retries + 1):
        # Choose prompt based on attempt
        prompt = CODEX_REVIEW_PROMPT if attempt == 1 else CODEX_REVIEW_RETRY_PROMPT

        # Run codex review with the commit
        proc = await asyncio.create_subprocess_exec(
            "codex",
            "review",
            "--commit",
            commit_sha,
            prompt,
            cwd=repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()
        output = stdout.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            error_msg = stderr.decode("utf-8", errors="replace").strip()
            return CodexReviewResult(
                passed=False,
                issues=[],
                raw_output=output,
                parse_error=f"codex exited with code {proc.returncode}: {error_msg}",
                attempt=attempt,
            )

        # Try to parse the JSON output
        passed, issues, parse_error = _parse_review_json(output)

        if parse_error is None:
            # Successful parse
            return CodexReviewResult(
                passed=passed,
                issues=issues,
                raw_output=output,
                parse_error=None,
                attempt=attempt,
            )

        # Parse failed - retry if we have attempts left
        if attempt < max_retries:
            continue

        # No retries left - fail closed
        return CodexReviewResult(
            passed=False,
            issues=[],
            raw_output=output,
            parse_error=parse_error,
            attempt=attempt,
        )

    # Should not reach here, but fail closed if we do
    return CodexReviewResult(
        passed=False,
        issues=[],
        raw_output="",
        parse_error="No review attempts made",
        attempt=0,
    )


def format_review_issues(issues: list[ReviewIssue]) -> str:
    """Format review issues as a human-readable string for follow-up prompts.

    Args:
        issues: List of ReviewIssue objects to format.

    Returns:
        Formatted string with issues grouped by file.
    """
    if not issues:
        return "No specific issues found."

    lines = []
    current_file = None

    # Sort by file, then by line
    sorted_issues = sorted(issues, key=lambda x: (x.file, x.line or 0))

    for issue in sorted_issues:
        if issue.file != current_file:
            if current_file is not None:
                lines.append("")  # Blank line between files
            current_file = issue.file
            lines.append(f"File: {issue.file}")

        loc = f"L{issue.line}" if issue.line else "file-level"
        lines.append(f"  [{issue.severity.upper()}] {loc}: {issue.message}")

    return "\n".join(lines)

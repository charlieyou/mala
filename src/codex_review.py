"""Codex review integration for mala orchestrator.

Runs code reviews using Codex CLI with strict JSON output parsing.
Uses codex exec with --output-schema for structured JSON responses.
"""

from __future__ import annotations

import asyncio
import json
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
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
    session_log_path: str | None = None


# JSON schema for structured output (used with codex exec --output-schema)
REVIEW_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "passed": {"type": "boolean"},
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "file": {"type": "string"},
                    "line": {"type": ["integer", "null"]},
                    "severity": {
                        "type": "string",
                        "enum": ["error", "warning", "info"],
                    },
                    "message": {"type": "string"},
                },
                "required": ["file", "line", "severity", "message"],
            },
        },
    },
    "required": ["passed", "issues"],
}

# Prompt template for code review
CODEX_REVIEW_PROMPT = """\
Review the diff {diff_description}.

## Pipeline Context

You are the **Code Review** stage in a multi-stage validation pipeline. Before this review runs, the following has ALREADY been validated:

1. **Quality Gate (already passed):**
   - `ruff check` - linting passed
   - `ruff format` - formatting passed
   - `ty check` - type checking passed
   - `pytest` - all tests passed
   - Coverage threshold met

Your job is to review **scope completeness and code correctness** - things that automated tools cannot catch.

## Issue Requirements

{issue_description}

## Your Review Focus

### 1. Scope Completeness (PRIMARY)
Compare the diff against the issue's **"In:" scope items** only:
- Flag as ERROR if any "In:" scope item is NOT addressed in the diff
- Be specific: quote the missing scope item in the error message

### 2. Code Correctness (SECONDARY)
- Bugs, logic errors, security issues, data integrity problems
- Ignore: style, formatting, test coverage, type errors (already validated)

## What NOT to Check

Do NOT flag issues for:
- Tests passing (quality gate verified this)
- Linting/formatting (ruff verified this)
- Type errors (ty verified this)
- Coverage (quality gate verified this)
- Acceptance criteria like "All existing tests pass" - these are runtime validations, not code review items

## Severity

- error: MISSING SCOPE ITEM, bug, security issue, data loss
- warning: partial implementation, risk, edge case
- info: minor suggestion

## Output

JSON with: file (repo-relative path or "N/A"), line (int or null), severity, message.

## Rules

- passed=false if ANY "In:" scope item is missing OR any severity="error" issue
- passed=true if all scope items addressed and no errors (warnings/info OK)
- Output only valid JSON matching the schema
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


def _parse_session_log_path(stderr: str) -> str | None:
    """Parse session ID from codex stderr and construct log path.

    Codex outputs "session id: <uuid>" in its header. Session logs are stored
    at ~/.codex/sessions/YYYY/MM/DD/rollout-{datetime}-{session-id}.jsonl.

    Args:
        stderr: Stderr output from codex exec.

    Returns:
        Full path to the session log file if found and exists, None otherwise.
    """
    # Match "session id: <uuid>" pattern
    match = re.search(r"session id:\s*([0-9a-f-]+)", stderr, re.IGNORECASE)
    if not match:
        return None

    session_id = match.group(1)

    # Session logs are stored in ~/.codex/sessions/YYYY/MM/DD/
    codex_sessions = Path.home() / ".codex" / "sessions"
    now = datetime.now()
    date_dir = codex_sessions / str(now.year) / f"{now.month:02d}" / f"{now.day:02d}"

    if not date_dir.exists():
        return None

    # Find the file matching the session ID
    # Format: rollout-{datetime}-{session-id}.jsonl
    for log_file in date_dir.glob(f"*{session_id}.jsonl"):
        return str(log_file)

    return None


async def run_codex_review(
    repo_path: Path,
    commit_sha: str,
    max_retries: int = 2,
    issue_description: str | None = None,
    baseline_commit: str | None = None,
    capture_session_log: bool = False,
    thinking_mode: str | None = None,
) -> CodexReviewResult:
    """Run Codex code review on a commit with JSON output and retry logic.

    Uses `codex exec` with `--output-schema` to get structured JSON responses.

    Args:
        repo_path: Path to the git repository.
        commit_sha: The commit SHA to review.
        max_retries: Maximum number of attempts (default 2 = 1 initial + 1 retry).
        issue_description: The issue description to verify scope completeness.
            If provided, Codex will verify the diff addresses all scope items.
        baseline_commit: Optional baseline commit for cumulative diff review.
            If provided, reviews the diff from baseline to commit_sha instead of
            commit vs its parent. This is used for retry scenarios where multiple
            fix commits have been made.
        capture_session_log: If True, capture and return the codex session log path.
            Only enabled in verbose mode to avoid overhead.
        thinking_mode: Optional reasoning effort level for Codex (e.g., "high",
            "medium", "low", "xhigh"). If provided, passed to codex via
            -c model_reasoning_effort=<mode>.

    Returns:
        CodexReviewResult with review outcome. If JSON parsing fails after
        all retries, the result will have passed=False (fail-closed).
    """
    for attempt in range(1, max_retries + 1):
        # Create temporary files for schema and output
        with (
            tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as schema_file,
            tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as output_file,
        ):
            schema_path = Path(schema_file.name)
            output_path = Path(output_file.name)

            # Write the schema
            json.dump(REVIEW_OUTPUT_SCHEMA, schema_file)
            schema_file.flush()

        try:
            # Build the prompt with appropriate diff description
            desc = issue_description or "(No issue description provided)"
            if baseline_commit:
                diff_description = f"from commit {baseline_commit} to {commit_sha} (cumulative changes)"
            else:
                diff_description = f"introduced by commit {commit_sha} (vs its parent)"
            prompt = CODEX_REVIEW_PROMPT.format(
                commit_sha=commit_sha,
                issue_description=desc,
                diff_description=diff_description,
            )

            # Run codex exec with output schema
            cmd = ["codex", "exec"]
            if thinking_mode is not None:
                cmd.extend(["-c", f"model_reasoning_effort={thinking_mode}"])
            cmd.extend(
                [
                    "--output-schema",
                    str(schema_path),
                    "-o",
                    str(output_path),
                    prompt,
                ]
            )
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate()
            raw_output = stdout.decode("utf-8", errors="replace")
            stderr_text = stderr.decode("utf-8", errors="replace")

            # Parse session log path from stderr if capture is enabled
            session_log_path = None
            if capture_session_log:
                session_log_path = _parse_session_log_path(stderr_text)

            if proc.returncode != 0:
                return CodexReviewResult(
                    passed=False,
                    issues=[],
                    raw_output=raw_output,
                    parse_error=f"codex exited with code {proc.returncode}: {stderr_text.strip()}",
                    attempt=attempt,
                    session_log_path=session_log_path,
                )

            # Read the structured output
            try:
                output = output_path.read_text()
            except OSError as e:
                return CodexReviewResult(
                    passed=False,
                    issues=[],
                    raw_output=raw_output,
                    parse_error=f"Failed to read output file: {e}",
                    attempt=attempt,
                    session_log_path=session_log_path,
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
                    session_log_path=session_log_path,
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
                session_log_path=session_log_path,
            )

        finally:
            # Clean up temporary files
            schema_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

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

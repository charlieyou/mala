"""Codex review integration for mala orchestrator.

Runs code reviews using Codex CLI with strict JSON output parsing.
Uses codex exec with --output-schema for structured JSON responses.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
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

## Issue Requirements

{issue_description}

## Review Focus (in priority order)

### 1. Scope Completeness (MOST IMPORTANT)
Compare the diff against the issue's scope ("In:" items) and acceptance criteria:
- Flag as ERROR if any scope item is NOT addressed in the diff
- Flag as ERROR if any acceptance criterion cannot be verified from the changes
- Be specific: quote the missing scope item in the error message

### 2. Code Correctness
- Focus on bugs, security issues, data integrity, performance problems
- Ignore style/formatting unless it affects correctness

### 3. Test Coverage
- Tests have already been executed and passed before this review
- Only flag if new functionality clearly lacks corresponding test code in the diff
- Do NOT request tests to be run - assume they passed

## Severity

- error: MISSING SCOPE ITEM, bug, security issue, data loss
- warning: partial implementation, risk, edge case, missing test coverage
- info: minor improvement

## Output Format

For each issue: use repo-relative file path, line if possible (null if file-level),
and a concise, actionable message. Use file "N/A" for scope/general issues.

## Rules

- passed MUST be false if ANY scope item from the issue description is missing
- passed MUST be false if there are any issues with severity "error"
- passed can be true if there are only warnings or info
- line can be null if the issue is file-level or scope-level
- Keep messages concise and actionable
- If no issues and all scope items addressed, return passed=true and issues=[]
- Output only JSON that conforms to the provided schema
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
    issue_description: str | None = None,
    baseline_commit: str | None = None,
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
            proc = await asyncio.create_subprocess_exec(
                "codex",
                "exec",
                "--output-schema",
                str(schema_path),
                "-o",
                str(output_path),
                prompt,
                cwd=repo_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await proc.communicate()
            raw_output = stdout.decode("utf-8", errors="replace")

            if proc.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace").strip()
                return CodexReviewResult(
                    passed=False,
                    issues=[],
                    raw_output=raw_output,
                    parse_error=f"codex exited with code {proc.returncode}: {error_msg}",
                    attempt=attempt,
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

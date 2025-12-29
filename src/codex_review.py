"""Codex review integration for mala orchestrator.

Runs code reviews using Codex CLI with strict JSON output parsing.
Uses codex exec with --output-schema for structured JSON responses.
"""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.tools.command_runner import CommandRunner


@dataclass
class ReviewIssue:
    """A single issue found during code review."""

    title: str
    body: str
    confidence_score: float
    priority: int | None  # 0=P0, 1=P1, 2=P2, 3=P3
    file: str
    line_start: int
    line_end: int


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
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "confidence_score": {"type": "number"},
                    "priority": {"type": ["integer", "null"]},
                    "code_location": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "absolute_file_path": {"type": "string"},
                            "line_range": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "start": {"type": "integer"},
                                    "end": {"type": "integer"},
                                },
                                "required": ["start", "end"],
                            },
                        },
                        "required": ["absolute_file_path", "line_range"],
                    },
                },
                "required": ["title", "body", "confidence_score", "code_location"],
            },
        },
        "overall_correctness": {
            "type": "string",
            "enum": ["patch is correct", "patch is incorrect"],
        },
        "overall_explanation": {"type": "string"},
        "overall_confidence_score": {"type": "number"},
    },
    "required": [
        "findings",
        "overall_correctness",
        "overall_explanation",
        "overall_confidence_score",
    ],
}

# Prompt template for code review
CODEX_REVIEW_PROMPT = """\
Review the diff {diff_description}.

## Issue Requirements

{issue_description}

## Review Guidelines

You are acting as a reviewer for a proposed code change made by another engineer.

Below are some default guidelines for determining whether the original author would appreciate the issue being flagged.

These are not the final word in determining whether an issue is a bug. In many cases, you will encounter other, more specific guidelines. These may be present elsewhere in a developer message, a user message, a file, or even elsewhere in this system message.
Those guidelines should be considered to override these general instructions.

Here are the general guidelines for determining whether something is a bug and should be flagged.

1. It meaningfully impacts the accuracy, performance, security, or maintainability of the code.
2. The bug is discrete and actionable (i.e. not a general issue with the codebase or a combination of multiple issues).
3. Fixing the bug does not demand a level of rigor that is not present in the rest of the codebase (e.g. one doesn't need very detailed comments and input validation in a repository of one-off scripts in personal projects)
4. The bug was introduced in the commit (pre-existing bugs should not be flagged).
5. The author of the original PR would likely fix the issue if they were made aware of it.
6. The bug does not rely on unstated assumptions about the codebase or author's intent.
7. It is not enough to speculate that a change may disrupt another part of the codebase, to be considered a bug, one must identify the other parts of the code that are provably affected.
8. The bug is clearly not just an intentional change by the original author.

When flagging a bug, you will also provide an accompanying comment. Once again, these guidelines are not the final word on how to construct a comment -- defer to any subsequent guidelines that you encounter.

1. The comment should be clear about why the issue is a bug.
2. The comment should appropriately communicate the severity of the issue. It should not claim that an issue is more severe than it actually is.
3. The comment should be brief. The body should be at most 1 paragraph. It should not introduce line breaks within the natural language flow unless it is necessary for the code fragment.
4. The comment should not include any chunks of code longer than 3 lines. Any code chunks should be wrapped in markdown inline code tags or a code block.
5. The comment should clearly and explicitly communicate the scenarios, environments, or inputs that are necessary for the bug to arise. The comment should immediately indicate that the issue's severity depends on these factors.
6. The comment's tone should be matter-of-fact and not accusatory or overly positive. It should read as a helpful AI assistant suggestion without sounding too much like a human reviewer.
7. The comment should be written such that the original author can immediately grasp the idea without close reading.
8. The comment should avoid excessive flattery and comments that are not helpful to the original author. The comment should avoid phrasing like "Great job ...", "Thanks for ...".

## How Many Findings to Return

Output all findings that the original author would fix if they knew about it. If there is no finding that a person would definitely love to see and fix, prefer outputting no findings. Do not stop at the first qualifying finding. Continue until you've listed every qualifying finding.

## Guidelines

- Ignore trivial style unless it obscures meaning or violates documented standards.
- Use one comment per distinct issue (or a multi-line range if necessary).
- Always keep the line range as short as possible for interpreting the issue. Avoid ranges longer than 5-10 lines; instead, choose the most suitable subrange that pinpoints the problem.

## Priority Tagging

At the beginning of the finding title, tag the bug with priority level. For example "[P1] Un-padding slices along wrong tensor dimensions".
- [P0] - Drop everything to fix. Blocking release, operations, or major usage. Only use for universal issues that do not depend on any assumptions about the inputs.
- [P1] - Urgent. Should be addressed in the next cycle.
- [P2] - Normal. To be fixed eventually.
- [P3] - Low. Nice to have.

Additionally, include a numeric priority field in the JSON output for each finding: set "priority" to 0 for P0, 1 for P1, 2 for P2, or 3 for P3. If a priority cannot be determined, omit the field or use null.

## Overall Correctness Verdict

At the end of your findings, output an "overall correctness" verdict of whether or not the patch should be considered "correct".
Correct implies that existing code and tests will not break, and the patch is free of bugs and other blocking issues.
Ignore non-blocking issues such as style, formatting, typos, documentation, and other nits.

## Output Format

Output valid JSON matching the schema. Do not wrap the JSON in markdown fences or extra prose.
The code_location field is required and must include absolute_file_path and line_range.
Line ranges must be as short as possible for interpreting the issue (avoid ranges over 5-10 lines; pick the most suitable subrange).
The code_location should overlap with the diff.
Do not generate a PR fix.
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

    overall_correctness = data.get("overall_correctness")
    if overall_correctness not in ("patch is correct", "patch is incorrect"):
        return (
            False,
            [],
            "'overall_correctness' must be 'patch is correct' or 'patch is incorrect'",
        )

    passed = overall_correctness == "patch is correct"

    raw_findings = data.get("findings", [])
    if not isinstance(raw_findings, list):
        return False, [], "'findings' field must be an array"

    issues: list[ReviewIssue] = []
    for i, item in enumerate(raw_findings):
        if not isinstance(item, dict):
            return False, [], f"Finding {i} is not an object"

        title = item.get("title")
        if not isinstance(title, str):
            return False, [], f"Finding {i}: 'title' must be a string"

        body = item.get("body", "")
        if not isinstance(body, str):
            return False, [], f"Finding {i}: 'body' must be a string"

        confidence_score = item.get("confidence_score", 0.0)
        if not isinstance(confidence_score, (int, float)):
            return False, [], f"Finding {i}: 'confidence_score' must be a number"

        priority = item.get("priority")
        if priority is not None and not isinstance(priority, int):
            return False, [], f"Finding {i}: 'priority' must be an integer or null"

        code_location = item.get("code_location", {})
        if not isinstance(code_location, dict):
            return False, [], f"Finding {i}: 'code_location' must be an object"

        file = code_location.get("absolute_file_path", "")
        if not isinstance(file, str):
            return False, [], f"Finding {i}: 'absolute_file_path' must be a string"

        line_range = code_location.get("line_range", {})
        if not isinstance(line_range, dict):
            return False, [], f"Finding {i}: 'line_range' must be an object"

        line_start = line_range.get("start", 0)
        line_end = line_range.get("end", 0)
        if not isinstance(line_start, int) or not isinstance(line_end, int):
            return (
                False,
                [],
                f"Finding {i}: line_range 'start' and 'end' must be integers",
            )

        issues.append(
            ReviewIssue(
                title=title,
                body=body,
                confidence_score=float(confidence_score),
                priority=priority,
                file=file,
                line_start=line_start,
                line_end=line_end,
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
    runner = CommandRunner(cwd=repo_path)

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

            result = await runner.run_async(cmd)

            # Parse session log path from stderr if capture is enabled
            session_log_path = None
            if capture_session_log:
                session_log_path = _parse_session_log_path(result.stderr)

            if result.returncode != 0:
                return CodexReviewResult(
                    passed=False,
                    issues=[],
                    raw_output=result.stdout,
                    parse_error=f"codex exited with code {result.returncode}: {result.stderr.strip()}",
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
                    raw_output=result.stdout,
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


def _to_relative_path(file_path: str) -> str:
    """Convert an absolute file path to a relative path for display.

    Strips the home directory prefix or common path prefixes to avoid
    leaking system information in logs and output.

    Args:
        file_path: Absolute or relative file path.

    Returns:
        Relative path suitable for display.
    """
    # If already relative, return as-is
    if not file_path.startswith("/"):
        return file_path

    # Try to make it relative to cwd
    cwd = Path.cwd()
    try:
        abs_path = Path(file_path)
        if abs_path.is_relative_to(cwd):
            return str(abs_path.relative_to(cwd))
    except (ValueError, OSError):
        pass

    # Fall back to just the filename if all else fails
    return Path(file_path).name


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

    # Sort by file, then by line, then by priority (lower = more important)
    # Use explicit None check since priority 0 (P0) is falsy but valid
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
        display_file = _to_relative_path(issue.file)
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
        # Don't add priority tag here - the title already contains it (e.g., "[P1] Title")
        lines.append(f"  {loc}: {issue.title}")
        if issue.body:
            lines.append(f"    {issue.body}")

    return "\n".join(lines)

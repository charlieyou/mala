"""Cerberus review-gate adapter for mala orchestrator.

This module provides a thin adapter for the Cerberus review-gate CLI.
It handles:
- Exit code mapping to ReviewResult
- JSON parsing of review output
- Issue formatting for follow-up prompts

The adapter includes DefaultReviewer for CLI execution in the repo_path.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from src.tools.command_runner import CommandRunner


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


@dataclass
class DefaultReviewer:
    """Default CodeReviewer implementation using Cerberus review-gate CLI.

    This class conforms to the CodeReviewer protocol and provides the default
    behavior for the orchestrator. Tests can inject alternative implementations.

    The reviewer spawns review-gate CLI processes and parses their output.
    Extra args/env can be provided to customize review-gate behavior.
    For initial review with an empty diff, short-circuits to PASS without spawning.
    """

    repo_path: Path
    bin_path: Path | None = None
    spawn_args: tuple[str, ...] = field(default_factory=tuple)
    wait_args: tuple[str, ...] = field(default_factory=tuple)
    env: dict[str, str] = field(default_factory=dict)

    def _review_gate_bin(self) -> str:
        if self.bin_path is not None:
            return str(self.bin_path / "review-gate")
        return "review-gate"

    def _validate_review_gate_bin(self) -> str | None:
        """Validate that the review-gate binary exists and is executable.

        Uses the merged env's PATH (respecting self.env) when checking for
        the binary to avoid false negatives when callers inject PATH via cerberus_env.

        Returns:
            None if the binary is valid, or an error message if not.
        """
        if self.bin_path is not None:
            # Explicit bin_path provided - check that the binary exists and is executable
            binary_path = self.bin_path / "review-gate"
            if not binary_path.exists():
                return f"review-gate binary not found at {binary_path}"
            if not os.access(binary_path, os.X_OK):
                return f"review-gate binary at {binary_path} is not executable"
        else:
            # Bare "review-gate" - check if it's in PATH (respecting self.env)
            # Build effective PATH by merging self.env with current os.environ
            if "PATH" in self.env:
                # Prepend self.env PATH to current PATH (matches CommandRunner._merge_env behavior)
                effective_path = (
                    self.env["PATH"] + os.pathsep + os.environ.get("PATH", "")
                )
            else:
                effective_path = os.environ.get("PATH", "")
            if shutil.which("review-gate", path=effective_path) is None:
                return "review-gate binary not found in PATH"
        return None

    def _build_env(self, claude_session_id: str | None) -> dict[str, str]:
        merged = dict(self.env)
        claude_session = (
            claude_session_id
            or merged.get("CLAUDE_SESSION_ID")
            or os.environ.get("CLAUDE_SESSION_ID")
        )
        if claude_session:
            merged["CLAUDE_SESSION_ID"] = claude_session
        return merged

    async def _is_diff_empty(self, diff_range: str, runner: CommandRunner) -> bool:
        """Check if the diff range has no changes.

        Uses git diff --stat to check if there are any changes in the range.
        If the command fails, returns False to proceed with review (fail-open).

        Args:
            diff_range: Git diff range to check (e.g., "baseline..HEAD").
            runner: CommandRunner instance to execute git command.

        Returns:
            True if the diff is empty (no changes), False otherwise.
        """
        try:
            result = await runner.run_async(
                ["git", "diff", "--stat", diff_range],
                timeout=30,
            )
            if result.returncode == 0 and not result.stdout.strip():
                return True
        except Exception:
            # If diff check fails, proceed with review anyway (fail-open)
            pass
        return False

    @staticmethod
    def _extract_wait_timeout(args: tuple[str, ...]) -> int | None:
        """Extract --timeout value from wait args if provided.

        Parses args to detect if a --timeout flag is already specified.
        Supports both '--timeout VALUE' and '--timeout=VALUE' formats.

        Args:
            args: Tuple of command-line arguments to search.

        Returns:
            The timeout value as int if found and valid, None otherwise.
        """
        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith("--timeout="):
                value = arg.split("=", 1)[1]
                if value.isdigit():
                    return int(value)
                return None
            if arg == "--timeout" and i + 1 < len(args):
                value = args[i + 1]
                if value.isdigit():
                    return int(value)
                return None
            i += 1
        return None

    async def __call__(
        self,
        diff_range: str,
        context_file: Path | None = None,
        timeout: int = 300,
        claude_session_id: str | None = None,
    ) -> ReviewResult:
        # Validate review-gate binary exists and is executable before proceeding
        validation_error = self._validate_review_gate_bin()
        if validation_error is not None:
            return ReviewResult(
                passed=False,
                issues=[],
                parse_error=validation_error,
                fatal_error=True,
                review_log_path=None,
            )

        runner = CommandRunner(cwd=self.repo_path)
        env = self._build_env(claude_session_id)
        if "CLAUDE_SESSION_ID" not in env:
            return ReviewResult(
                passed=False,
                issues=[],
                parse_error="CLAUDE_SESSION_ID missing; must be provided by agent session id",
                fatal_error=True,
                review_log_path=None,
            )

        # Check for empty diff (short-circuit without spawning review-gate)
        # This avoids parse errors or failures when there's nothing to review
        if await self._is_diff_empty(diff_range, runner):
            return ReviewResult(
                passed=True,
                issues=[],
                parse_error=None,
                fatal_error=False,
                review_log_path=None,
            )

        spawn_cmd = [self._review_gate_bin(), "spawn-code-review"]
        if context_file is not None:
            spawn_cmd.extend(["--context-file", str(context_file)])
        if self.spawn_args:
            spawn_cmd.extend(self.spawn_args)
        # Diff range is a positional argument (review-gate does not accept --diff).
        spawn_cmd.append(diff_range)

        spawn_result = await runner.run_async(spawn_cmd, env=env, timeout=timeout)
        if spawn_result.timed_out:
            return ReviewResult(
                passed=False,
                issues=[],
                parse_error="spawn timeout",
                fatal_error=False,
            )
        if spawn_result.returncode != 0:
            stderr = spawn_result.stderr_tail()
            stdout = spawn_result.stdout_tail()
            detail = stderr or stdout or "spawn failed"
            return ReviewResult(
                passed=False,
                issues=[],
                parse_error=f"spawn failed: {detail}",
                fatal_error=False,
            )

        try:
            spawn_payload = json.loads(spawn_result.stdout)
        except json.JSONDecodeError as e:
            return ReviewResult(
                passed=False,
                issues=[],
                parse_error=f"spawn parse error: {e}",
                fatal_error=False,
            )

        if not isinstance(spawn_payload, dict):
            return ReviewResult(
                passed=False,
                issues=[],
                parse_error="spawn output is not a JSON object",
                fatal_error=False,
            )

        session_key = spawn_payload.get("session_key")
        if not isinstance(session_key, str) or not session_key:
            return ReviewResult(
                passed=False,
                issues=[],
                parse_error="spawn output missing session_key",
                fatal_error=False,
            )

        # Check if wait_args already specifies --timeout to avoid duplicates
        user_timeout = self._extract_wait_timeout(self.wait_args)
        cli_timeout = user_timeout if user_timeout is not None else timeout

        wait_cmd = [
            self._review_gate_bin(),
            "wait",
            "--json",
            "--session-key",
            session_key,
        ]
        # Only add --timeout if not already specified in wait_args
        if user_timeout is None:
            wait_cmd.extend(["--timeout", str(timeout)])
        if self.wait_args:
            wait_cmd.extend(self.wait_args)

        # Use CLI timeout + grace period for subprocess timeout
        # This allows the CLI's internal timeout to trigger first with a proper error message
        subprocess_timeout = cli_timeout + 30
        wait_result = await runner.run_async(
            wait_cmd, env=env, timeout=subprocess_timeout
        )
        if wait_result.timed_out:
            return ReviewResult(
                passed=False,
                issues=[],
                parse_error="timeout",
                fatal_error=False,
            )

        # Try to extract session_dir from wait output for review_log_path
        review_log_path: Path | None = None
        try:
            wait_data = json.loads(wait_result.stdout)
            if isinstance(wait_data, dict):
                session_dir = wait_data.get("session_dir")
                if isinstance(session_dir, str) and session_dir:
                    review_log_path = Path(session_dir)
        except (json.JSONDecodeError, TypeError):
            # If we can't parse, let map_exit_code_to_result handle it
            pass

        return map_exit_code_to_result(
            wait_result.returncode,
            wait_result.stdout,
            wait_result.stderr,
            review_log_path=review_log_path,
        )


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

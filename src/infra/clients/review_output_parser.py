"""Review output parsing for Cerberus review-gate.

This module provides ReviewOutputParser for parsing JSON output from
the review-gate CLI and mapping exit codes to domain results. It handles:
- JSON decoding and validation
- Issue object mapping (aggregated_findings to ReviewIssue)
- Exit-code to ReviewResult mapping
- Parse error extraction

This is a low-level component extracted from DefaultReviewer to enable
independent testing of parsing logic.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003 (runtime import for get_type_hints compatibility)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.protocols.events import MalaEventSink


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
    interrupted: bool = False


class ReviewOutputParser:
    """Parses Cerberus review-gate JSON output and maps exit codes to results.

    This class encapsulates all JSON parsing and exit-code interpretation logic.
    It is stateless and can be used as a singleton or instantiated per-call.

    Usage:
        parser = ReviewOutputParser()

        # Parse JSON output
        passed, issues, error = parser.parse_json(stdout)

        # Map exit code to ReviewResult
        result = parser.map_exit_code_to_result(exit_code, stdout, stderr)
    """

    def parse_json(self, output: str) -> tuple[bool, list[ReviewIssue], str | None]:
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

        # Check consensus verdict (top-level consensus_verdict field)
        verdict = data.get("consensus_verdict")
        if verdict not in ("PASS", "FAIL", "NEEDS_WORK", "no_reviewers", "ERROR"):
            return False, [], f"Invalid verdict: {verdict}"

        passed = verdict == "PASS"

        # Parse issues from aggregated_findings (may be empty for PASS verdict)
        raw_issues = data.get("aggregated_findings", [])
        if not isinstance(raw_issues, list):
            return False, [], "'aggregated_findings' field must be an array"

        issues: list[ReviewIssue] = []
        for i, item in enumerate(raw_issues):
            if not isinstance(item, dict):
                return False, [], f"Issue {i} is not an object"

            reviewer = item.get("reviewer", "")
            if not isinstance(reviewer, str):
                return False, [], f"Issue {i}: 'reviewer' must be a string"

            # Cerberus uses file_path (can be null for non-file-specific findings)
            file_path = item.get("file_path")
            if file_path is None:
                file_path = ""
            elif not isinstance(file_path, str):
                return False, [], f"Issue {i}: 'file_path' must be a string or null"

            # line_start and line_end can be null
            line_start = item.get("line_start")
            if line_start is None:
                line_start = 0
            elif not isinstance(line_start, int):
                return False, [], f"Issue {i}: 'line_start' must be an integer or null"

            line_end = item.get("line_end")
            if line_end is None:
                line_end = 0
            elif not isinstance(line_end, int):
                return False, [], f"Issue {i}: 'line_end' must be an integer or null"

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
        self,
        exit_code: int,
        stdout: str,
        stderr: str,
        review_log_path: Path | None = None,
        event_sink: MalaEventSink | None = None,
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
            event_sink: Optional event sink for emitting warnings.

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
                        parse_error_msg = "; ".join(
                            str(e.get("error", e)) if isinstance(e, dict) else str(e)
                            for e in parse_errors
                        )
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
        json_passed, issues, parse_error = self.parse_json(stdout)

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
            message = (
                f"Exit code ({exit_code}) and JSON verdict "
                f"({'PASS' if json_passed else 'FAIL'}) disagree; "
                f"fail-closed: requiring both to pass"
            )
            if event_sink is not None:
                event_sink.on_review_warning(message)
            else:
                # Always log this critical diagnostic even without event_sink
                logging.warning(message)

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


# Module-level convenience functions for backward compatibility
# These delegate to a shared parser instance

_parser = ReviewOutputParser()


def parse_cerberus_json(output: str) -> tuple[bool, list[ReviewIssue], str | None]:
    """Parse Cerberus review-gate JSON output.

    This is a convenience function that delegates to ReviewOutputParser.parse_json().
    """
    return _parser.parse_json(output)


def map_exit_code_to_result(
    exit_code: int,
    stdout: str,
    stderr: str,
    review_log_path: Path | None = None,
    event_sink: MalaEventSink | None = None,
) -> ReviewResult:
    """Map Cerberus review-gate exit code to ReviewResult.

    This is a convenience function that delegates to ReviewOutputParser.map_exit_code_to_result().
    """
    return _parser.map_exit_code_to_result(
        exit_code, stdout, stderr, review_log_path, event_sink
    )

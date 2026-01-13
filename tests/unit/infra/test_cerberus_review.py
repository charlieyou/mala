"""Tests for Cerberus review-gate adapter.

Tests the cerberus_review integration including:
- JSON response parsing
- Exit code mapping (0-5)
- Issue formatting
- Parse error extraction
- Golden file tests against real Cerberus output
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from src.infra.clients.cerberus_review import (
    DefaultReviewer,
    _to_relative_path,
    format_review_issues,
)
from src.infra.clients.review_output_parser import (
    ReviewIssue,
    ReviewOutputParser,
    ReviewResult,
)
from src.infra.io.base_sink import BaseEventSink

if TYPE_CHECKING:
    from src.core.protocols.events import MalaEventSink

# Path to golden files captured from real Cerberus output
FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "cerberus"

# Shared parser instance for tests
_parser = ReviewOutputParser()


def parse_cerberus_json(output: str) -> tuple[bool, list[ReviewIssue], str | None]:
    """Test helper that delegates to parser.parse_json."""
    return _parser.parse_json(output)


def map_exit_code_to_result(
    exit_code: int,
    stdout: str,
    stderr: str,
    review_log_path: Path | None = None,
    event_sink: MalaEventSink | None = None,
) -> ReviewResult:
    """Test helper that delegates to parser.map_exit_code_to_result."""
    return _parser.map_exit_code_to_result(
        exit_code, stdout, stderr, review_log_path, event_sink
    )


class MockEventSink(BaseEventSink):
    """Mock event sink for testing event emissions.

    Inherits from BaseEventSink to get no-op implementations for all
    MalaEventSink methods, preventing AttributeError if tested code
    emits additional events.
    """

    def __init__(self) -> None:
        self.warnings: list[str] = []

    def on_review_warning(
        self,
        message: str,
        agent_id: str | None = None,
        issue_id: str | None = None,
    ) -> None:
        self.warnings.append(message)


def _make_valid_response(
    verdict: str = "PASS", issues: list[dict] | None = None
) -> str:
    """Helper to create a valid Cerberus review-gate response JSON.

    Uses the actual Cerberus output format with:
    - consensus_verdict at top level (not consensus.verdict)
    - aggregated_findings (not issues)
    """
    return json.dumps(
        {
            "status": "complete",
            "consensus_verdict": verdict,
            "reviewers": {
                "codex": {
                    "verdict": verdict,
                    "summary": "Test",
                    "findings": [],
                }
            },
            "aggregated_findings": issues or [],
            "parse_errors": [],
        }
    )


def _make_issue(
    file: str = "src/test.py",
    line_start: int = 10,
    line_end: int = 12,
    priority: int | None = 1,
    title: str = "Test finding",
    body: str = "Test body",
    reviewer: str = "codex",
) -> dict:
    """Helper to create a valid issue dict.

    Uses the actual Cerberus format with file_path (not file).
    """
    return {
        "reviewer": reviewer,
        "file_path": file,
        "line_start": line_start,
        "line_end": line_end,
        "priority": priority,
        "title": title,
        "body": body,
    }


class TestParseCerberusJson:
    """Tests for parsing Cerberus review-gate JSON output."""

    def test_parses_valid_pass_response(self) -> None:
        output = _make_valid_response(verdict="PASS")
        passed, issues, error = parse_cerberus_json(output)
        assert passed is True
        assert issues == []
        assert error is None

    def test_parses_valid_fail_response(self) -> None:
        output = _make_valid_response(verdict="FAIL")
        passed, issues, error = parse_cerberus_json(output)
        assert passed is False
        assert issues == []
        assert error is None

    def test_parses_needs_work_response(self) -> None:
        output = _make_valid_response(verdict="NEEDS_WORK")
        passed, issues, error = parse_cerberus_json(output)
        assert passed is False
        assert issues == []
        assert error is None

    def test_parses_no_reviewers_response(self) -> None:
        output = _make_valid_response(verdict="no_reviewers")
        passed, issues, error = parse_cerberus_json(output)
        assert passed is False
        assert issues == []
        assert error is None

    def test_parses_issues_correctly(self) -> None:
        issue = _make_issue(
            file="src/main.py",
            line_start=42,
            line_end=45,
            priority=1,
            title="[P1] Missing null check",
            body="Variable may be None",
            reviewer="codex",
        )
        output = _make_valid_response(verdict="FAIL", issues=[issue])
        passed, issues, error = parse_cerberus_json(output)

        assert passed is False
        assert len(issues) == 1
        assert issues[0].file == "src/main.py"
        assert issues[0].line_start == 42
        assert issues[0].line_end == 45
        assert issues[0].priority == 1
        assert issues[0].title == "[P1] Missing null check"
        assert issues[0].body == "Variable may be None"
        assert issues[0].reviewer == "codex"
        assert error is None

    def test_parses_issue_with_null_priority(self) -> None:
        issue = _make_issue(priority=None)
        output = _make_valid_response(verdict="FAIL", issues=[issue])
        _passed, issues, _error = parse_cerberus_json(output)
        assert issues[0].priority is None

    def test_parses_multiple_issues_from_different_reviewers(self) -> None:
        issues_data = [
            _make_issue(reviewer="codex", title="Codex issue"),
            _make_issue(reviewer="gemini", title="Gemini issue"),
            _make_issue(reviewer="claude", title="Claude issue"),
        ]
        output = _make_valid_response(verdict="FAIL", issues=issues_data)
        _passed, issues, _error = parse_cerberus_json(output)

        assert len(issues) == 3
        assert issues[0].reviewer == "codex"
        assert issues[1].reviewer == "gemini"
        assert issues[2].reviewer == "claude"

    def test_returns_error_for_empty_output(self) -> None:
        passed, issues, error = parse_cerberus_json("")
        assert passed is False
        assert issues == []
        assert error is not None
        assert "Empty output" in error

    def test_returns_error_for_invalid_json(self) -> None:
        passed, issues, error = parse_cerberus_json("not valid json")
        assert passed is False
        assert issues == []
        assert error is not None
        assert "JSON parse error" in error

    def test_returns_error_for_invalid_verdict(self) -> None:
        output = json.dumps(
            {
                "consensus": {"verdict": "MAYBE"},
                "issues": [],
            }
        )
        passed, _issues, error = parse_cerberus_json(output)
        assert passed is False
        assert error is not None
        assert "Invalid verdict" in error

    def test_returns_error_for_missing_consensus(self) -> None:
        output = json.dumps({"issues": []})
        passed, _issues, error = parse_cerberus_json(output)
        assert passed is False
        assert error is not None
        assert "'consensus'" in error or "Invalid verdict" in error

    def test_returns_error_for_invalid_issue_type(self) -> None:
        """Test error when aggregated_findings contains non-object items."""
        output = json.dumps(
            {
                "status": "complete",
                "consensus_verdict": "FAIL",
                "aggregated_findings": ["not an object"],
            }
        )
        passed, _issues, error = parse_cerberus_json(output)
        assert passed is False
        assert error is not None
        assert "not an object" in error

    def test_returns_error_for_non_object_root(self) -> None:
        passed, _issues, error = parse_cerberus_json("[]")
        assert passed is False
        assert error is not None
        assert "Root element is not an object" in error


class TestGoldenFiles:
    """Golden file tests using captured real Cerberus output.

    These tests ensure parse_cerberus_json correctly handles the actual
    output format from `review-gate wait --json`, not just mock data.
    """

    def test_golden_pass(self) -> None:
        """Parse real PASS output from Cerberus."""
        output = (FIXTURES_DIR / "wait_pass.json").read_text()
        passed, issues, error = parse_cerberus_json(output)

        assert error is None, f"Unexpected parse error: {error}"
        assert passed is True
        # PASS verdict but has P3 findings (non-blocking)
        assert len(issues) == 1
        assert issues[0].priority == 3
        assert issues[0].reviewer == "gemini"

    def test_golden_fail(self) -> None:
        """Parse real FAIL output from Cerberus with multiple reviewers."""
        output = (FIXTURES_DIR / "wait_fail.json").read_text()
        passed, issues, error = parse_cerberus_json(output)

        assert error is None, f"Unexpected parse error: {error}"
        assert passed is False
        assert len(issues) == 3

        # Verify all reviewers' findings are captured
        reviewers = {i.reviewer for i in issues}
        assert reviewers == {"claude", "codex", "gemini"}

        # Verify P1 issues exist
        p1_issues = [i for i in issues if i.priority == 1]
        assert len(p1_issues) == 2

        # Verify file paths are captured
        files = {i.file for i in issues}
        assert "tests/test_cli.py" in files
        assert "src/orchestrator.py" in files

    def test_golden_no_reviewers(self) -> None:
        """Parse output when no reviewers were spawned."""
        output = (FIXTURES_DIR / "wait_no_reviewers.json").read_text()
        passed, _issues, error = parse_cerberus_json(output)

        # null consensus_verdict should fail validation
        assert passed is False
        assert error is not None
        assert "Invalid verdict" in error

    def test_golden_error(self) -> None:
        """Parse output when review-gate encounters an error."""
        output = (FIXTURES_DIR / "wait_error.json").read_text()
        passed, _issues, error = parse_cerberus_json(output)

        # null consensus_verdict should fail validation
        assert passed is False
        assert error is not None
        assert "Invalid verdict" in error

    def test_golden_timeout(self) -> None:
        """Parse output when review times out."""
        output = (FIXTURES_DIR / "wait_timeout.json").read_text()
        passed, _issues, error = parse_cerberus_json(output)

        # null consensus_verdict on timeout should fail validation
        assert passed is False
        assert error is not None
        assert "Invalid verdict" in error


class TestMapExitCodeToResult:
    """Tests for exit code mapping to ReviewResult."""

    def test_exit_0_pass(self) -> None:
        """Exit code 0 = PASS: all reviewers agree, no issues."""
        output = _make_valid_response(verdict="PASS")
        result = map_exit_code_to_result(0, output, "")

        assert result.passed is True
        assert result.issues == []
        assert result.parse_error is None
        assert result.fatal_error is False

    def test_exit_0_with_issues(self) -> None:
        """Exit code 0 with issues is still a pass (low-priority issues)."""
        issue = _make_issue(priority=3)  # P3 = low priority
        output = _make_valid_response(verdict="PASS", issues=[issue])
        result = map_exit_code_to_result(0, output, "")

        assert result.passed is True
        assert len(result.issues) == 1
        assert result.parse_error is None

    def test_exit_1_fail(self) -> None:
        """Exit code 1 = FAIL/NEEDS_WORK: legitimate review failure."""
        issue = _make_issue(priority=1)
        output = _make_valid_response(verdict="FAIL", issues=[issue])
        result = map_exit_code_to_result(1, output, "")

        assert result.passed is False
        assert len(result.issues) == 1
        assert result.parse_error is None
        assert result.fatal_error is False

    def test_exit_2_parse_error(self) -> None:
        """Exit code 2 = Parse error: malformed reviewer output (retryable)."""
        output = json.dumps(
            {
                "consensus": {"verdict": "FAIL"},
                "issues": [],
                "parse_errors": ["codex: malformed JSON response", "gemini: timeout"],
            }
        )
        result = map_exit_code_to_result(2, output, "")

        assert result.passed is False
        assert result.issues == []
        assert result.parse_error is not None
        assert "codex: malformed JSON response" in result.parse_error
        assert result.fatal_error is False

    def test_exit_2_with_fallback_to_stderr(self) -> None:
        """Exit code 2 with invalid JSON falls back to stderr."""
        result = map_exit_code_to_result(2, "invalid json", "Error: connection failed")

        assert result.passed is False
        assert result.parse_error is not None
        assert "connection failed" in result.parse_error
        assert result.fatal_error is False

    def test_exit_3_timeout(self) -> None:
        """Exit code 3 = Timeout: reviewers didn't respond (retryable)."""
        result = map_exit_code_to_result(3, "", "")

        assert result.passed is False
        assert result.issues == []
        assert result.parse_error == "timeout"
        assert result.fatal_error is False

    def test_exit_4_no_reviewers(self) -> None:
        """Exit code 4 = No reviewers: no reviewer CLIs available (fatal)."""
        result = map_exit_code_to_result(4, "", "")

        assert result.passed is False
        assert result.issues == []
        assert result.parse_error == "No reviewers available"
        assert result.fatal_error is True

    def test_exit_5_internal_error(self) -> None:
        """Exit code 5 = Internal error: unexpected failure (fatal)."""
        result = map_exit_code_to_result(5, "", "Unexpected error occurred")

        assert result.passed is False
        assert result.issues == []
        assert "Unexpected error occurred" in (result.parse_error or "")
        assert result.fatal_error is True

    def test_exit_5_with_empty_stderr(self) -> None:
        """Exit code 5 with empty stderr uses default message."""
        result = map_exit_code_to_result(5, "", "")

        assert result.passed is False
        assert result.fatal_error is True
        assert result.parse_error == "Internal error"

    def test_malformed_json_on_exit_0(self) -> None:
        """Malformed JSON on exit 0 is treated as parse error."""
        result = map_exit_code_to_result(0, "not json", "")

        assert result.passed is False
        assert result.parse_error is not None
        assert "JSON parse error" in result.parse_error
        assert result.fatal_error is False

    def test_malformed_json_on_exit_1(self) -> None:
        """Malformed JSON on exit 1 is treated as parse error."""
        result = map_exit_code_to_result(1, "broken", "")

        assert result.passed is False
        assert result.parse_error is not None
        assert result.fatal_error is False

    def test_review_log_path_preserved(self) -> None:
        """Review log path is preserved in result."""
        log_path = Path("/tmp/review-session")
        output = _make_valid_response(verdict="PASS")
        result = map_exit_code_to_result(0, output, "", review_log_path=log_path)

        assert result.review_log_path == log_path

    def test_exit_0_with_json_fail_fails_closed(self) -> None:
        """Exit code 0 but JSON verdict FAIL should fail (fail-closed security)."""
        # Exit code 0 but JSON says FAIL - should FAIL (fail-closed)
        sink = MockEventSink()
        output = _make_valid_response(verdict="FAIL")
        result = map_exit_code_to_result(
            0, output, "", event_sink=cast("MalaEventSink", sink)
        )

        # Fail-closed: review must fail because JSON verdict is not PASS
        assert result.passed is False
        assert result.parse_error is None
        assert result.fatal_error is False

        # Check that warning was emitted to event sink
        assert len(sink.warnings) == 1
        assert "disagree" in sink.warnings[0]
        assert "FAIL" in sink.warnings[0]
        assert "fail-closed" in sink.warnings[0]

    def test_exit_1_with_json_pass_fails_closed(self) -> None:
        """Exit code 1 but JSON verdict PASS should fail (fail-closed security)."""
        # Exit code 1 but JSON says PASS - should FAIL (fail-closed)
        sink = MockEventSink()
        output = _make_valid_response(verdict="PASS")
        result = map_exit_code_to_result(
            1, output, "", event_sink=cast("MalaEventSink", sink)
        )

        # Fail-closed: review must fail because exit code is not 0
        assert result.passed is False
        assert result.parse_error is None
        assert result.fatal_error is False

        # Check that warning was emitted to event sink
        assert len(sink.warnings) == 1
        assert "disagree" in sink.warnings[0]
        assert "PASS" in sink.warnings[0]
        assert "fail-closed" in sink.warnings[0]

    def test_no_warning_when_exit_code_and_verdict_agree(self) -> None:
        """No warning when exit code and JSON verdict agree."""
        # Exit code 0 and JSON says PASS - should not warn
        sink = MockEventSink()
        output = _make_valid_response(verdict="PASS")
        result = map_exit_code_to_result(
            0, output, "", event_sink=cast("MalaEventSink", sink)
        )

        assert result.passed is True
        assert len(sink.warnings) == 0

    def test_exit_0_with_needs_work_fails_closed(self) -> None:
        """Exit code 0 with NEEDS_WORK verdict should fail (fail-closed security)."""
        output = _make_valid_response(verdict="NEEDS_WORK")
        result = map_exit_code_to_result(0, output, "")

        # Fail-closed: NEEDS_WORK is not PASS, so must fail
        assert result.passed is False
        assert result.parse_error is None
        assert result.fatal_error is False

    def test_exit_0_with_no_reviewers_fails_closed(self) -> None:
        """Exit code 0 with no_reviewers verdict should fail (fail-closed security)."""
        output = _make_valid_response(verdict="no_reviewers")
        result = map_exit_code_to_result(0, output, "")

        # Fail-closed: no_reviewers is not PASS, so must fail
        assert result.passed is False
        assert result.parse_error is None
        assert result.fatal_error is False


class TestFormatReviewIssues:
    """Tests for formatting review issues for follow-up prompts."""

    def test_formats_empty_issues(self) -> None:
        result = format_review_issues([])
        assert result == "No specific issues found."

    def test_formats_single_issue(self) -> None:
        issues = [
            ReviewIssue(
                file="src/main.py",
                line_start=10,
                line_end=10,
                priority=1,
                title="[P1] Missing import",
                body="The os module is not imported",
                reviewer="codex",
            )
        ]
        result = format_review_issues(issues)
        assert "File: src/main.py" in result
        assert "L10:" in result
        assert "[codex]" in result
        assert "[P1] Missing import" in result
        assert "The os module is not imported" in result

    def test_formats_multiline_issue(self) -> None:
        issues = [
            ReviewIssue(
                file="src/utils.py",
                line_start=5,
                line_end=15,
                priority=2,
                title="Complex function",
                body="Consider refactoring",
                reviewer="gemini",
            )
        ]
        result = format_review_issues(issues)
        assert "L5-15:" in result
        assert "[gemini]" in result

    def test_groups_by_file(self) -> None:
        issues = [
            ReviewIssue(
                file="b.py",
                line_start=1,
                line_end=1,
                priority=1,
                title="Issue 1",
                body="",
                reviewer="codex",
            ),
            ReviewIssue(
                file="a.py",
                line_start=5,
                line_end=5,
                priority=1,
                title="Issue 2",
                body="",
                reviewer="codex",
            ),
            ReviewIssue(
                file="a.py",
                line_start=10,
                line_end=10,
                priority=2,
                title="Issue 3",
                body="",
                reviewer="codex",
            ),
        ]
        result = format_review_issues(issues)
        # Should be sorted by file, then by line
        lines = result.split("\n")
        file_lines = [line for line in lines if line.startswith("File:")]
        assert file_lines[0] == "File: a.py"
        assert file_lines[1] == "File: b.py"

    def test_includes_reviewer_attribution(self) -> None:
        issues = [
            ReviewIssue(
                file="test.py",
                line_start=1,
                line_end=1,
                priority=1,
                title="Issue",
                body="",
                reviewer="claude",
            )
        ]
        result = format_review_issues(issues)
        assert "[claude]" in result

    def test_handles_empty_reviewer(self) -> None:
        issues = [
            ReviewIssue(
                file="test.py",
                line_start=1,
                line_end=1,
                priority=1,
                title="Issue",
                body="",
                reviewer="",
            )
        ]
        result = format_review_issues(issues)
        # Should not have empty brackets
        assert "[]" not in result
        assert "Issue" in result


class TestReviewResultProtocol:
    """Tests verifying ReviewResult satisfies ReviewOutcome protocol."""

    def test_result_has_required_fields(self) -> None:
        """ReviewResult must have passed, parse_error, fatal_error, issues."""
        result = ReviewResult(
            passed=True,
            issues=[],
            parse_error=None,
            fatal_error=False,
        )

        # These are the fields required by ReviewOutcome protocol
        assert hasattr(result, "passed")
        assert hasattr(result, "parse_error")
        assert hasattr(result, "fatal_error")
        assert hasattr(result, "issues")

    def test_parse_error_is_str_or_none(self) -> None:
        """parse_error must be str | None, not bool."""
        result_none = ReviewResult(passed=True, parse_error=None)
        result_str = ReviewResult(passed=False, parse_error="error message")

        assert result_none.parse_error is None
        assert isinstance(result_str.parse_error, str)


class TestReviewIssueProtocol:
    """Tests verifying ReviewIssue satisfies lifecycle.ReviewIssue protocol."""

    def test_issue_has_required_fields(self) -> None:
        """ReviewIssue must have all fields required by lifecycle protocol."""
        issue = ReviewIssue(
            file="test.py",
            line_start=1,
            line_end=2,
            priority=1,
            title="Title",
            body="Body",
            reviewer="codex",
        )

        # These are the fields required by lifecycle.ReviewIssue protocol
        assert hasattr(issue, "file")
        assert hasattr(issue, "line_start")
        assert hasattr(issue, "line_end")
        assert hasattr(issue, "priority")
        assert hasattr(issue, "title")
        assert hasattr(issue, "body")
        assert hasattr(issue, "reviewer")


class TestToRelativePath:
    """Tests for path relativization helper."""

    def test_relative_path_unchanged(self) -> None:
        """Relative paths are returned unchanged."""
        assert _to_relative_path("src/main.py") == "src/main.py"
        assert _to_relative_path("test.py") == "test.py"
        assert _to_relative_path("./foo/bar.py") == "./foo/bar.py"

    def test_absolute_path_with_base_path(self) -> None:
        """Absolute paths are relativized against base_path."""
        base = Path("/home/user/project")
        assert (
            _to_relative_path("/home/user/project/src/main.py", base) == "src/main.py"
        )
        assert _to_relative_path("/home/user/project/test.py", base) == "test.py"

    def test_absolute_path_outside_base_preserved(self) -> None:
        """Absolute paths outside base_path are preserved (not stripped to filename)."""
        base = Path("/home/user/project")
        # Path outside base_path should be preserved fully
        result = _to_relative_path("/other/path/to/file.py", base)
        assert result == "/other/path/to/file.py"

    def test_absolute_path_no_base_uses_cwd(self) -> None:
        """Without base_path, falls back to cwd."""
        # This test checks that relative paths in cwd work
        cwd = Path.cwd()
        # Create a path that is inside cwd
        test_path = str(cwd / "src" / "test.py")
        result = _to_relative_path(test_path)
        assert result == "src/test.py"

    def test_preserves_directory_context_on_failure(self) -> None:
        """When relativization fails, full path is preserved (not just filename)."""
        base = Path("/home/user/project")
        # A path that cannot be relativized should keep full directory context
        result = _to_relative_path("/completely/different/path/important/file.py", base)
        # Should NOT be stripped to just "file.py"
        assert "important" in result
        assert result == "/completely/different/path/important/file.py"


class TestExtractWaitTimeout:
    """Tests for DefaultReviewer._extract_wait_timeout method."""

    def test_returns_none_for_empty_args(self) -> None:
        """Returns None when args is empty."""
        assert DefaultReviewer._extract_wait_timeout(()) is None

    def test_returns_none_when_no_timeout_flag(self) -> None:
        """Returns None when --timeout is not present."""
        args = ("--json", "--session-key", "abc123")
        assert DefaultReviewer._extract_wait_timeout(args) is None

    def test_extracts_timeout_with_equals_format(self) -> None:
        """Extracts timeout from --timeout=VALUE format."""
        args = ("--json", "--timeout=600", "--session-key", "abc123")
        assert DefaultReviewer._extract_wait_timeout(args) == 600

    def test_extracts_timeout_with_space_format(self) -> None:
        """Extracts timeout from --timeout VALUE format."""
        args = ("--json", "--timeout", "300", "--session-key", "abc123")
        assert DefaultReviewer._extract_wait_timeout(args) == 300

    def test_returns_none_for_non_numeric_equals_value(self) -> None:
        """Returns None when --timeout=VALUE has non-numeric value."""
        args = ("--timeout=abc",)
        assert DefaultReviewer._extract_wait_timeout(args) is None

    def test_returns_none_for_non_numeric_space_value(self) -> None:
        """Returns None when --timeout VALUE has non-numeric value."""
        args = ("--timeout", "abc")
        assert DefaultReviewer._extract_wait_timeout(args) is None

    def test_returns_none_for_timeout_at_end_without_value(self) -> None:
        """Returns None when --timeout is at end without value."""
        args = ("--json", "--timeout")
        assert DefaultReviewer._extract_wait_timeout(args) is None

    def test_timeout_at_beginning_of_args(self) -> None:
        """Extracts timeout when it's the first argument."""
        args = ("--timeout", "120", "--json")
        assert DefaultReviewer._extract_wait_timeout(args) == 120

    def test_timeout_at_end_of_args_with_value(self) -> None:
        """Extracts timeout when it's the last argument pair."""
        args = ("--json", "--timeout", "450")
        assert DefaultReviewer._extract_wait_timeout(args) == 450


class TestAlreadyActiveGateError:
    """Tests for 'already active' gate error handling.

    When spawn fails with 'already active', the adapter now auto-resolves
    the stale gate and retries spawn once. This handles the case where a
    prior review attempt hit a parse error (e.g., invalid_verdict from one
    model) and left a gate pending. Since we use the same CLAUDE_SESSION_ID,
    we're resolving our own session's gate, not interfering with other runs.

    If spawn still fails with 'already active' after resolve, that means
    another session owns the gate, so we return a fatal error.
    """

    async def test_already_active_auto_resolves_and_retries(self) -> None:
        """Auto-resolves stale gate and retries spawn successfully."""
        from unittest.mock import AsyncMock, MagicMock, patch

        reviewer = DefaultReviewer(repo_path=Path("/tmp"))

        with patch(
            "src.infra.clients.cerberus_gate_cli.CerberusGateCLI.validate_binary",
            return_value=None,
        ):
            # First spawn fails with "already active"
            spawn_fail_result = MagicMock()
            spawn_fail_result.returncode = 1
            spawn_fail_result.timed_out = False
            spawn_fail_result.stderr_tail.return_value = (
                "Error: review gate already active"
            )
            spawn_fail_result.stdout_tail.return_value = ""

            # Resolve succeeds
            resolve_result = MagicMock()
            resolve_result.returncode = 0
            resolve_result.stderr = ""
            resolve_result.stdout = ""

            # Retry spawn succeeds
            spawn_ok_result = MagicMock()
            spawn_ok_result.returncode = 0
            spawn_ok_result.timed_out = False

            # Wait returns PASS
            wait_result = MagicMock()
            wait_result.returncode = 0
            wait_result.timed_out = False
            wait_result.stdout = '{"status":"complete","consensus_verdict":"PASS","reviewers":{},"aggregated_findings":[],"parse_errors":[]}'

            with patch(
                "src.infra.clients.cerberus_review.CommandRunner"
            ) as mock_runner_class:
                mock_runner = AsyncMock()
                # Sequence: spawn (fail), resolve, spawn (ok), wait
                mock_runner.run_async.side_effect = [
                    spawn_fail_result,
                    resolve_result,
                    spawn_ok_result,
                    wait_result,
                ]
                mock_runner_class.return_value = mock_runner

                with patch.dict("os.environ", {"CLAUDE_SESSION_ID": "test-session"}):
                    result = await reviewer(commit_shas=["abc123"])

            assert result.passed is True
            assert result.fatal_error is False
            assert result.parse_error is None

            # Verify call sequence
            calls = [call[0][0] for call in mock_runner.run_async.call_args_list]
            assert any("spawn-code-review" in str(c) for c in calls)
            assert any("resolve" in str(c) for c in calls)
            assert any("wait" in str(c) for c in calls)

    async def test_already_active_after_resolve_is_fatal(self) -> None:
        """Returns fatal error if still 'already active' after resolve (another session)."""
        from unittest.mock import AsyncMock, MagicMock, patch

        reviewer = DefaultReviewer(repo_path=Path("/tmp"))

        with patch(
            "src.infra.clients.cerberus_gate_cli.CerberusGateCLI.validate_binary",
            return_value=None,
        ):
            # First spawn fails with "already active"
            spawn_fail_result = MagicMock()
            spawn_fail_result.returncode = 1
            spawn_fail_result.timed_out = False
            spawn_fail_result.stderr_tail.return_value = (
                "Error: review gate already active"
            )
            spawn_fail_result.stdout_tail.return_value = ""

            # Resolve succeeds
            resolve_result = MagicMock()
            resolve_result.returncode = 0
            resolve_result.stderr = ""
            resolve_result.stdout = ""

            # Retry spawn STILL fails with "already active" (another session)
            spawn_still_active = MagicMock()
            spawn_still_active.returncode = 1
            spawn_still_active.timed_out = False
            spawn_still_active.stderr_tail.return_value = (
                "Error: review gate already active"
            )
            spawn_still_active.stdout_tail.return_value = ""

            with patch(
                "src.infra.clients.cerberus_review.CommandRunner"
            ) as mock_runner_class:
                mock_runner = AsyncMock()
                mock_runner.run_async.side_effect = [
                    spawn_fail_result,
                    resolve_result,
                    spawn_still_active,
                ]
                mock_runner_class.return_value = mock_runner

                with patch.dict("os.environ", {"CLAUDE_SESSION_ID": "test-session"}):
                    result = await reviewer(commit_shas=["abc123"])

            # Should be fatal error (another session owns the gate)
            assert result.passed is False
            assert result.fatal_error is True
            assert result.parse_error is not None
            assert "not from this session" in result.parse_error

    async def test_resolve_failure_is_retryable(self) -> None:
        """If resolve fails, returns retryable error (not fatal)."""
        from unittest.mock import AsyncMock, MagicMock, patch

        reviewer = DefaultReviewer(repo_path=Path("/tmp"))

        with patch(
            "src.infra.clients.cerberus_gate_cli.CerberusGateCLI.validate_binary",
            return_value=None,
        ):
            spawn_fail_result = MagicMock()
            spawn_fail_result.returncode = 1
            spawn_fail_result.timed_out = False
            spawn_fail_result.stderr_tail.return_value = (
                "Error: review gate already active"
            )
            spawn_fail_result.stdout_tail.return_value = ""

            # Resolve fails
            resolve_result = MagicMock()
            resolve_result.returncode = 1
            resolve_result.stderr = "Permission denied"
            resolve_result.stdout = ""

            with patch(
                "src.infra.clients.cerberus_review.CommandRunner"
            ) as mock_runner_class:
                mock_runner = AsyncMock()
                mock_runner.run_async.side_effect = [
                    spawn_fail_result,
                    resolve_result,
                ]
                mock_runner_class.return_value = mock_runner

                with patch.dict("os.environ", {"CLAUDE_SESSION_ID": "test-session"}):
                    result = await reviewer(commit_shas=["abc123"])

            # Retryable error (not fatal)
            assert result.passed is False
            assert result.fatal_error is False
            assert result.parse_error is not None
            assert "auto-resolve failed" in result.parse_error

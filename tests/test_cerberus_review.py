"""Tests for Cerberus review-gate adapter.

Tests the cerberus_review integration including:
- JSON response parsing
- Exit code mapping (0-5)
- Issue formatting
- Parse error extraction
"""

from __future__ import annotations

import json
from pathlib import Path

from src.cerberus_review import (
    ReviewIssue,
    ReviewResult,
    _to_relative_path,
    format_review_issues,
    map_exit_code_to_result,
    parse_cerberus_json,
)


def _make_valid_response(
    verdict: str = "PASS", issues: list[dict] | None = None
) -> str:
    """Helper to create a valid Cerberus review-gate response JSON."""
    return json.dumps(
        {
            "status": "resolved",
            "consensus": {"verdict": verdict, "iteration": 1},
            "reviewers": {
                "codex": {
                    "verdict": verdict,
                    "summary": "Test",
                    "issues": [],
                    "error": None,
                }
            },
            "issues": issues or [],
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
    """Helper to create a valid issue dict."""
    return {
        "reviewer": reviewer,
        "file": file,
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
        output = json.dumps(
            {
                "consensus": {"verdict": "FAIL"},
                "issues": ["not an object"],
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

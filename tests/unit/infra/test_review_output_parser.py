"""Tests for ReviewOutputParser component.

Tests the review_output_parser module including:
- JSON response parsing (via ReviewOutputParser.parse_json)
- Exit code mapping (0-5) (via ReviewOutputParser.map_exit_code_to_result)
- Golden file tests against real Cerberus output

This module tests the parsing logic independently from the CLI adapter.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from src.infra.clients.review_output_parser import (
    ReviewOutputParser,
)
from src.infra.io.base_sink import BaseEventSink

if TYPE_CHECKING:
    from src.core.protocols.events import MalaEventSink

# Path to golden files captured from real Cerberus output
FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "cerberus"


class MockEventSink(BaseEventSink):
    """Mock event sink for testing event emissions."""

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
    """Helper to create a valid Cerberus review-gate response JSON."""
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
    """Helper to create a valid issue dict."""
    return {
        "reviewer": reviewer,
        "file_path": file,
        "line_start": line_start,
        "line_end": line_end,
        "priority": priority,
        "title": title,
        "body": body,
    }


class TestReviewOutputParserClass:
    """Tests for ReviewOutputParser class instantiation and methods."""

    def test_parser_is_stateless(self) -> None:
        """Parser can be instantiated and reused without state issues."""
        parser = ReviewOutputParser()
        output = _make_valid_response(verdict="PASS")

        # Call multiple times
        result1 = parser.parse_json(output)
        result2 = parser.parse_json(output)

        assert result1 == result2


class TestParseJson:
    """Tests for parsing Cerberus review-gate JSON output."""

    @pytest.fixture
    def parser(self) -> ReviewOutputParser:
        return ReviewOutputParser()

    def test_parses_valid_pass_response(self, parser: ReviewOutputParser) -> None:
        output = _make_valid_response(verdict="PASS")
        passed, issues, error = parser.parse_json(output)
        assert passed is True
        assert issues == []
        assert error is None

    def test_parses_valid_fail_response(self, parser: ReviewOutputParser) -> None:
        output = _make_valid_response(verdict="FAIL")
        passed, issues, error = parser.parse_json(output)
        assert passed is False
        assert issues == []
        assert error is None

    def test_parses_needs_work_response(self, parser: ReviewOutputParser) -> None:
        output = _make_valid_response(verdict="NEEDS_WORK")
        passed, issues, error = parser.parse_json(output)
        assert passed is False
        assert issues == []
        assert error is None

    def test_parses_no_reviewers_response(self, parser: ReviewOutputParser) -> None:
        output = _make_valid_response(verdict="no_reviewers")
        passed, issues, error = parser.parse_json(output)
        assert passed is False
        assert issues == []
        assert error is None

    def test_parses_error_verdict_response(self, parser: ReviewOutputParser) -> None:
        output = _make_valid_response(verdict="ERROR")
        passed, issues, error = parser.parse_json(output)
        assert passed is False
        assert issues == []
        assert error is None

    def test_parses_issues_correctly(self, parser: ReviewOutputParser) -> None:
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
        passed, issues, error = parser.parse_json(output)

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

    def test_parses_issue_with_null_priority(self, parser: ReviewOutputParser) -> None:
        issue = _make_issue(priority=None)
        output = _make_valid_response(verdict="FAIL", issues=[issue])
        _passed, issues, _error = parser.parse_json(output)
        assert issues[0].priority is None

    def test_parses_issue_with_null_file_path(self, parser: ReviewOutputParser) -> None:
        """Handles null file_path for non-file-specific findings."""
        issue = {
            "reviewer": "codex",
            "file_path": None,
            "line_start": None,
            "line_end": None,
            "priority": 2,
            "title": "General observation",
            "body": "No specific location",
        }
        output = _make_valid_response(verdict="FAIL", issues=[issue])
        _passed, issues, error = parser.parse_json(output)
        assert error is None
        assert issues[0].file == ""
        assert issues[0].line_start == 0
        assert issues[0].line_end == 0

    def test_parses_multiple_issues_from_different_reviewers(
        self, parser: ReviewOutputParser
    ) -> None:
        issues_data = [
            _make_issue(reviewer="codex", title="Codex issue"),
            _make_issue(reviewer="gemini", title="Gemini issue"),
            _make_issue(reviewer="claude", title="Claude issue"),
        ]
        output = _make_valid_response(verdict="FAIL", issues=issues_data)
        _passed, issues, _error = parser.parse_json(output)

        assert len(issues) == 3
        assert issues[0].reviewer == "codex"
        assert issues[1].reviewer == "gemini"
        assert issues[2].reviewer == "claude"

    def test_returns_error_for_empty_output(self, parser: ReviewOutputParser) -> None:
        passed, issues, error = parser.parse_json("")
        assert passed is False
        assert issues == []
        assert error is not None
        assert "Empty output" in error

    def test_returns_error_for_whitespace_only(
        self, parser: ReviewOutputParser
    ) -> None:
        passed, issues, error = parser.parse_json("   \n\t  ")
        assert passed is False
        assert issues == []
        assert error is not None
        assert "Empty output" in error

    def test_returns_error_for_invalid_json(self, parser: ReviewOutputParser) -> None:
        passed, issues, error = parser.parse_json("not valid json")
        assert passed is False
        assert issues == []
        assert error is not None
        assert "JSON parse error" in error

    def test_returns_error_for_invalid_verdict(
        self, parser: ReviewOutputParser
    ) -> None:
        output = json.dumps(
            {
                "consensus_verdict": "MAYBE",
                "aggregated_findings": [],
            }
        )
        passed, _issues, error = parser.parse_json(output)
        assert passed is False
        assert error is not None
        assert "Invalid verdict" in error

    def test_returns_error_for_missing_verdict(
        self, parser: ReviewOutputParser
    ) -> None:
        output = json.dumps({"aggregated_findings": []})
        passed, _issues, error = parser.parse_json(output)
        assert passed is False
        assert error is not None
        assert "Invalid verdict" in error

    def test_returns_error_for_invalid_issue_type(
        self, parser: ReviewOutputParser
    ) -> None:
        output = json.dumps(
            {
                "consensus_verdict": "FAIL",
                "aggregated_findings": ["not an object"],
            }
        )
        passed, _issues, error = parser.parse_json(output)
        assert passed is False
        assert error is not None
        assert "not an object" in error

    def test_returns_error_for_non_object_root(
        self, parser: ReviewOutputParser
    ) -> None:
        passed, _issues, error = parser.parse_json("[]")
        assert passed is False
        assert error is not None
        assert "Root element is not an object" in error

    def test_returns_error_for_invalid_reviewer_type(
        self, parser: ReviewOutputParser
    ) -> None:
        output = json.dumps(
            {
                "consensus_verdict": "FAIL",
                "aggregated_findings": [{"reviewer": 123}],
            }
        )
        passed, _issues, error = parser.parse_json(output)
        assert passed is False
        assert error is not None
        assert "'reviewer' must be a string" in error

    def test_returns_error_for_invalid_file_path_type(
        self, parser: ReviewOutputParser
    ) -> None:
        output = json.dumps(
            {
                "consensus_verdict": "FAIL",
                "aggregated_findings": [{"reviewer": "codex", "file_path": 123}],
            }
        )
        passed, _issues, error = parser.parse_json(output)
        assert passed is False
        assert error is not None
        assert "'file_path' must be a string or null" in error

    def test_returns_error_for_non_list_findings(
        self, parser: ReviewOutputParser
    ) -> None:
        output = json.dumps(
            {
                "consensus_verdict": "FAIL",
                "aggregated_findings": "not a list",
            }
        )
        passed, _issues, error = parser.parse_json(output)
        assert passed is False
        assert error is not None
        assert "'aggregated_findings' field must be an array" in error


class TestMapExitCodeToResult:
    """Tests for exit code mapping to ReviewResult."""

    @pytest.fixture
    def parser(self) -> ReviewOutputParser:
        return ReviewOutputParser()

    def test_exit_0_pass(self, parser: ReviewOutputParser) -> None:
        output = _make_valid_response(verdict="PASS")
        result = parser.map_exit_code_to_result(0, output, "")

        assert result.passed is True
        assert result.issues == []
        assert result.parse_error is None
        assert result.fatal_error is False

    def test_exit_0_with_issues(self, parser: ReviewOutputParser) -> None:
        issue = _make_issue(priority=3)
        output = _make_valid_response(verdict="PASS", issues=[issue])
        result = parser.map_exit_code_to_result(0, output, "")

        assert result.passed is True
        assert len(result.issues) == 1
        assert result.parse_error is None

    def test_exit_1_fail(self, parser: ReviewOutputParser) -> None:
        issue = _make_issue(priority=1)
        output = _make_valid_response(verdict="FAIL", issues=[issue])
        result = parser.map_exit_code_to_result(1, output, "")

        assert result.passed is False
        assert len(result.issues) == 1
        assert result.parse_error is None
        assert result.fatal_error is False

    def test_exit_2_parse_error(self, parser: ReviewOutputParser) -> None:
        output = json.dumps(
            {
                "consensus_verdict": "FAIL",
                "aggregated_findings": [],
                "parse_errors": ["codex: malformed JSON response", "gemini: timeout"],
            }
        )
        result = parser.map_exit_code_to_result(2, output, "")

        assert result.passed is False
        assert result.issues == []
        assert result.parse_error is not None
        assert "codex: malformed JSON response" in result.parse_error
        assert result.fatal_error is False

    def test_exit_2_with_fallback_to_stderr(self, parser: ReviewOutputParser) -> None:
        result = parser.map_exit_code_to_result(
            2, "invalid json", "Error: connection failed"
        )

        assert result.passed is False
        assert result.parse_error is not None
        assert "connection failed" in result.parse_error
        assert result.fatal_error is False

    def test_exit_2_with_dict_parse_errors(self, parser: ReviewOutputParser) -> None:
        """Handles parse_errors that are dict objects with 'error' key."""
        output = json.dumps(
            {
                "consensus_verdict": "FAIL",
                "aggregated_findings": [],
                "parse_errors": [{"error": "Invalid response", "reviewer": "codex"}],
            }
        )
        result = parser.map_exit_code_to_result(2, output, "")

        assert result.passed is False
        assert result.parse_error is not None
        assert "Invalid response" in result.parse_error

    def test_exit_3_timeout(self, parser: ReviewOutputParser) -> None:
        result = parser.map_exit_code_to_result(3, "", "")

        assert result.passed is False
        assert result.issues == []
        assert result.parse_error == "timeout"
        assert result.fatal_error is False

    def test_exit_4_no_reviewers(self, parser: ReviewOutputParser) -> None:
        result = parser.map_exit_code_to_result(4, "", "")

        assert result.passed is False
        assert result.issues == []
        assert result.parse_error == "No reviewers available"
        assert result.fatal_error is True

    def test_exit_5_internal_error(self, parser: ReviewOutputParser) -> None:
        result = parser.map_exit_code_to_result(5, "", "Unexpected error occurred")

        assert result.passed is False
        assert result.issues == []
        assert "Unexpected error occurred" in (result.parse_error or "")
        assert result.fatal_error is True

    def test_exit_5_with_empty_stderr(self, parser: ReviewOutputParser) -> None:
        result = parser.map_exit_code_to_result(5, "", "")

        assert result.passed is False
        assert result.fatal_error is True
        assert result.parse_error == "Internal error"

    def test_malformed_json_on_exit_0(self, parser: ReviewOutputParser) -> None:
        result = parser.map_exit_code_to_result(0, "not json", "")

        assert result.passed is False
        assert result.parse_error is not None
        assert "JSON parse error" in result.parse_error
        assert result.fatal_error is False

    def test_malformed_json_on_exit_1(self, parser: ReviewOutputParser) -> None:
        result = parser.map_exit_code_to_result(1, "broken", "")

        assert result.passed is False
        assert result.parse_error is not None
        assert result.fatal_error is False

    def test_review_log_path_preserved(self, parser: ReviewOutputParser) -> None:
        log_path = Path("/tmp/review-session")
        output = _make_valid_response(verdict="PASS")
        result = parser.map_exit_code_to_result(0, output, "", review_log_path=log_path)

        assert result.review_log_path == log_path


class TestFailClosedBehavior:
    """Tests for fail-closed security behavior."""

    @pytest.fixture
    def parser(self) -> ReviewOutputParser:
        return ReviewOutputParser()

    def test_exit_0_with_json_fail_fails_closed(
        self, parser: ReviewOutputParser
    ) -> None:
        """Exit code 0 but JSON verdict FAIL should fail (fail-closed security)."""
        sink = MockEventSink()
        output = _make_valid_response(verdict="FAIL")
        result = parser.map_exit_code_to_result(
            0, output, "", event_sink=cast("MalaEventSink", sink)
        )

        assert result.passed is False
        assert result.parse_error is None
        assert result.fatal_error is False

        assert len(sink.warnings) == 1
        assert "disagree" in sink.warnings[0]
        assert "FAIL" in sink.warnings[0]
        assert "fail-closed" in sink.warnings[0]

    def test_exit_1_with_json_pass_fails_closed(
        self, parser: ReviewOutputParser
    ) -> None:
        """Exit code 1 but JSON verdict PASS should fail (fail-closed security)."""
        sink = MockEventSink()
        output = _make_valid_response(verdict="PASS")
        result = parser.map_exit_code_to_result(
            1, output, "", event_sink=cast("MalaEventSink", sink)
        )

        assert result.passed is False
        assert result.parse_error is None
        assert result.fatal_error is False

        assert len(sink.warnings) == 1
        assert "disagree" in sink.warnings[0]
        assert "PASS" in sink.warnings[0]
        assert "fail-closed" in sink.warnings[0]

    def test_no_warning_when_exit_code_and_verdict_agree(
        self, parser: ReviewOutputParser
    ) -> None:
        """No warning when exit code and JSON verdict agree."""
        sink = MockEventSink()
        output = _make_valid_response(verdict="PASS")
        result = parser.map_exit_code_to_result(
            0, output, "", event_sink=cast("MalaEventSink", sink)
        )

        assert result.passed is True
        assert len(sink.warnings) == 0

    def test_exit_0_with_needs_work_fails_closed(
        self, parser: ReviewOutputParser
    ) -> None:
        """Exit code 0 with NEEDS_WORK verdict should fail (fail-closed security)."""
        output = _make_valid_response(verdict="NEEDS_WORK")
        result = parser.map_exit_code_to_result(0, output, "")

        assert result.passed is False
        assert result.parse_error is None
        assert result.fatal_error is False

    def test_exit_0_with_no_reviewers_fails_closed(
        self, parser: ReviewOutputParser
    ) -> None:
        """Exit code 0 with no_reviewers verdict should fail (fail-closed security)."""
        output = _make_valid_response(verdict="no_reviewers")
        result = parser.map_exit_code_to_result(0, output, "")

        assert result.passed is False
        assert result.parse_error is None
        assert result.fatal_error is False


class TestGoldenFiles:
    """Golden file tests using captured real Cerberus output."""

    @pytest.fixture
    def parser(self) -> ReviewOutputParser:
        return ReviewOutputParser()

    def test_golden_pass(self, parser: ReviewOutputParser) -> None:
        """Parse real PASS output from Cerberus."""
        output = (FIXTURES_DIR / "wait_pass.json").read_text()
        passed, issues, error = parser.parse_json(output)

        assert error is None, f"Unexpected parse error: {error}"
        assert passed is True
        # PASS verdict but has P3 findings (non-blocking)
        assert len(issues) == 1
        assert issues[0].priority == 3
        assert issues[0].reviewer == "gemini"

    def test_golden_fail(self, parser: ReviewOutputParser) -> None:
        """Parse real FAIL output from Cerberus with multiple reviewers."""
        output = (FIXTURES_DIR / "wait_fail.json").read_text()
        passed, issues, error = parser.parse_json(output)

        assert error is None, f"Unexpected parse error: {error}"
        assert passed is False
        assert len(issues) == 3

        reviewers = {i.reviewer for i in issues}
        assert reviewers == {"claude", "codex", "gemini"}

        p1_issues = [i for i in issues if i.priority == 1]
        assert len(p1_issues) == 2

        files = {i.file for i in issues}
        assert "tests/test_cli.py" in files
        assert "src/orchestrator.py" in files

    def test_golden_no_reviewers(self, parser: ReviewOutputParser) -> None:
        """Parse output when no reviewers were spawned."""
        output = (FIXTURES_DIR / "wait_no_reviewers.json").read_text()
        passed, _issues, error = parser.parse_json(output)

        assert passed is False
        assert error is not None
        assert "Invalid verdict" in error

    def test_golden_error(self, parser: ReviewOutputParser) -> None:
        """Parse output when review-gate encounters an error."""
        output = (FIXTURES_DIR / "wait_error.json").read_text()
        passed, _issues, error = parser.parse_json(output)

        assert passed is False
        assert error is not None
        assert "Invalid verdict" in error

    def test_golden_timeout(self, parser: ReviewOutputParser) -> None:
        """Parse output when review times out."""
        output = (FIXTURES_DIR / "wait_timeout.json").read_text()
        passed, _issues, error = parser.parse_json(output)

        assert passed is False
        assert error is not None
        assert "Invalid verdict" in error

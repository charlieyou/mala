"""Integration tests for Codex review functionality.

Tests the codex review integration including:
- CLI invocation with correct arguments
- JSON response parsing
- Retry logic for parse failures
- Error handling for CLI failures
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from src.codex_review import (
    ReviewIssue,
    _extract_json,
    _parse_review_json,
    format_review_issues,
    run_codex_review,
)
from src.tools.command_runner import CommandResult


class TestExtractJson:
    """Tests for JSON extraction from mixed text."""

    def test_extracts_clean_json(self) -> None:
        text = '{"passed": true, "issues": []}'
        result = _extract_json(text)
        assert result == '{"passed": true, "issues": []}'

    def test_extracts_json_with_surrounding_text(self) -> None:
        text = 'Here is the review:\n{"passed": false, "issues": []}\nDone.'
        result = _extract_json(text)
        assert result == '{"passed": false, "issues": []}'

    def test_extracts_nested_json(self) -> None:
        text = '{"passed": true, "issues": [{"file": "a.py", "line": 1}]}'
        result = _extract_json(text)
        assert result == text

    def test_returns_none_for_no_json(self) -> None:
        text = "No JSON here"
        result = _extract_json(text)
        assert result is None

    def test_handles_strings_with_braces(self) -> None:
        text = '{"passed": true, "message": "use {foo} syntax"}'
        result = _extract_json(text)
        assert result is not None
        parsed = json.loads(result)
        assert parsed["message"] == "use {foo} syntax"


class TestParseReviewJson:
    """Tests for parsing Codex review JSON output."""

    def test_parses_valid_passed_review(self) -> None:
        output = '{"passed": true, "issues": []}'
        passed, issues, error = _parse_review_json(output)
        assert passed is True
        assert issues == []
        assert error is None

    def test_parses_valid_failed_review_with_issues(self) -> None:
        output = json.dumps(
            {
                "passed": False,
                "issues": [
                    {
                        "file": "src/main.py",
                        "line": 42,
                        "severity": "error",
                        "message": "Missing type annotation",
                    }
                ],
            }
        )
        passed, issues, error = _parse_review_json(output)
        assert passed is False
        assert len(issues) == 1
        assert issues[0].file == "src/main.py"
        assert issues[0].line == 42
        assert issues[0].severity == "error"
        assert issues[0].message == "Missing type annotation"
        assert error is None

    def test_parses_issue_with_null_line(self) -> None:
        output = json.dumps(
            {
                "passed": False,
                "issues": [
                    {
                        "file": "README.md",
                        "line": None,
                        "severity": "warning",
                        "message": "File-level issue",
                    }
                ],
            }
        )
        _passed, issues, _error = _parse_review_json(output)
        assert issues[0].line is None

    def test_returns_error_for_invalid_json(self) -> None:
        output = "not valid json"
        passed, issues, error = _parse_review_json(output)
        assert passed is False
        assert issues == []
        assert error is not None
        assert "No JSON object found" in error

    def test_returns_error_for_missing_passed_field(self) -> None:
        output = '{"issues": []}'
        passed, _issues, error = _parse_review_json(output)
        assert passed is False
        assert error is not None
        assert "'passed' field must be a boolean" in error

    def test_returns_error_for_invalid_severity(self) -> None:
        output = json.dumps(
            {
                "passed": False,
                "issues": [
                    {"file": "a.py", "line": 1, "severity": "critical", "message": "x"}
                ],
            }
        )
        _passed, _issues, error = _parse_review_json(output)
        assert error is not None
        assert "invalid severity" in error


class TestFormatReviewIssues:
    """Tests for formatting review issues."""

    def test_formats_empty_issues(self) -> None:
        result = format_review_issues([])
        assert result == "No specific issues found."

    def test_formats_single_issue(self) -> None:
        issues = [
            ReviewIssue(
                file="src/main.py", line=10, severity="error", message="Missing import"
            )
        ]
        result = format_review_issues(issues)
        assert "File: src/main.py" in result
        assert "[ERROR] L10: Missing import" in result

    def test_formats_file_level_issue(self) -> None:
        issues = [
            ReviewIssue(
                file="README.md", line=None, severity="warning", message="Outdated docs"
            )
        ]
        result = format_review_issues(issues)
        assert "[WARNING] file-level: Outdated docs" in result

    def test_groups_by_file(self) -> None:
        issues = [
            ReviewIssue(file="b.py", line=1, severity="error", message="Issue 1"),
            ReviewIssue(file="a.py", line=5, severity="error", message="Issue 2"),
            ReviewIssue(file="a.py", line=10, severity="warning", message="Issue 3"),
        ]
        result = format_review_issues(issues)
        # Should be sorted by file, then by line
        lines = result.split("\n")
        file_lines = [line for line in lines if line.startswith("File:")]
        assert file_lines[0] == "File: a.py"
        assert file_lines[1] == "File: b.py"


@pytest.fixture
def mock_codex_script(tmp_path: Path) -> tuple[Path, Path]:
    """Create a mock codex script that captures invocations and writes to output file."""
    script_path = tmp_path / "codex"
    invocation_log = tmp_path / "codex_invocations.jsonl"

    # Create a script that:
    # 1. Logs invocations to a file
    # 2. Writes the mock response to the -o output file (if provided)
    script_content = f"""\
#!/usr/bin/env python3
import json
import os
import sys

# Log the invocation
invocation = {{"args": sys.argv[1:], "cwd": os.getcwd()}}
with open("{invocation_log}", "a") as f:
    f.write(json.dumps(invocation) + "\\n")

# Check for test-controlled behavior
exit_code = int(os.environ.get("MOCK_CODEX_EXIT_CODE", "0"))
response = os.environ.get("MOCK_CODEX_STDOUT", '{{"passed": true, "issues": []}}')
stderr = os.environ.get("MOCK_CODEX_STDERR", "")

# Find -o flag and write response to that file
args = sys.argv[1:]
for i, arg in enumerate(args):
    if arg == "-o" and i + 1 < len(args):
        output_path = args[i + 1]
        with open(output_path, "w") as f:
            f.write(response)
        break

if stderr:
    print(stderr, file=sys.stderr)
sys.exit(exit_code)
"""
    script_path.write_text(script_content)
    script_path.chmod(script_path.stat().st_mode | stat.S_IEXEC)

    return script_path, invocation_log


class TestRunCodexReview:
    """Integration tests for run_codex_review function."""

    @pytest.mark.asyncio
    async def test_invokes_codex_with_correct_arguments(
        self, tmp_path: Path, mock_codex_script: tuple[Path, Path]
    ) -> None:
        """Codex should be invoked with correct arguments for commit review."""
        script_path, invocation_log = mock_codex_script
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # Add mock script to PATH
        env = {
            **os.environ,
            "PATH": f"{script_path.parent}:{os.environ.get('PATH', '')}",
            "MOCK_CODEX_STDOUT": '{"passed": true, "issues": []}',
        }

        with patch.dict(os.environ, env, clear=True):
            await run_codex_review(repo_path, "abc1234")

        # Check invocation
        invocations = [
            json.loads(line) for line in invocation_log.read_text().strip().split("\n")
        ]
        assert len(invocations) == 1

        args = invocations[0]["args"]
        # Should use codex exec with --output-schema and -o
        assert args[0] == "exec"
        assert "--output-schema" in args
        assert "-o" in args
        # Should include commit SHA in the prompt
        assert any("abc1234" in arg for arg in args)

    @pytest.mark.asyncio
    async def test_returns_success_for_passing_review(
        self, tmp_path: Path, mock_codex_script: tuple[Path, Path]
    ) -> None:
        """Should return passed=True when codex reports no errors."""
        script_path, _ = mock_codex_script
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        env = {
            **os.environ,
            "PATH": f"{script_path.parent}:{os.environ.get('PATH', '')}",
            "MOCK_CODEX_STDOUT": '{"passed": true, "issues": []}',
        }

        with patch.dict(os.environ, env, clear=True):
            result = await run_codex_review(repo_path, "abc1234")

        assert result.passed is True
        assert result.issues == []
        assert result.parse_error is None

    @pytest.mark.asyncio
    async def test_returns_failure_with_issues(
        self, tmp_path: Path, mock_codex_script: tuple[Path, Path]
    ) -> None:
        """Should return passed=False with issues when codex finds errors."""
        script_path, _ = mock_codex_script
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        review_output = json.dumps(
            {
                "passed": False,
                "issues": [
                    {
                        "file": "src/bug.py",
                        "line": 5,
                        "severity": "error",
                        "message": "Potential null pointer",
                    }
                ],
            }
        )

        env = {
            **os.environ,
            "PATH": f"{script_path.parent}:{os.environ.get('PATH', '')}",
            "MOCK_CODEX_STDOUT": review_output,
        }

        with patch.dict(os.environ, env, clear=True):
            result = await run_codex_review(repo_path, "abc1234")

        assert result.passed is False
        assert len(result.issues) == 1
        assert result.issues[0].message == "Potential null pointer"

    @pytest.mark.asyncio
    async def test_handles_codex_nonzero_exit(
        self, tmp_path: Path, mock_codex_script: tuple[Path, Path]
    ) -> None:
        """Should return parse_error when codex exits non-zero."""
        script_path, _ = mock_codex_script
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        env = {
            **os.environ,
            "PATH": f"{script_path.parent}:{os.environ.get('PATH', '')}",
            "MOCK_CODEX_EXIT_CODE": "2",
            "MOCK_CODEX_STDERR": "error: some problem occurred",
        }

        with patch.dict(os.environ, env, clear=True):
            result = await run_codex_review(repo_path, "abc1234")

        assert result.passed is False
        assert result.parse_error is not None
        assert "exited with code 2" in result.parse_error

    @pytest.mark.asyncio
    async def test_retries_on_parse_failure(self, tmp_path: Path) -> None:
        """Should retry when JSON parsing fails."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        # First return invalid JSON, second returns valid
        call_count = 0

        async def mock_run_async(
            self: object, cmd: list[str], **kwargs: object
        ) -> CommandResult:
            nonlocal call_count
            call_count += 1

            # Find -o flag and write response to that file
            for i, arg in enumerate(cmd):
                if arg == "-o" and i + 1 < len(cmd):
                    output_path = cmd[i + 1]
                    if call_count == 1:
                        Path(output_path).write_text("Invalid JSON response")
                    else:
                        Path(output_path).write_text('{"passed": true, "issues": []}')
                    break

            return CommandResult(
                command=cmd,
                returncode=0,
                stdout="",
                stderr="",
            )

        with patch(
            "src.codex_review.CommandRunner.run_async",
            mock_run_async,
        ):
            result = await run_codex_review(repo_path, "abc1234", max_retries=2)

        assert call_count == 2
        assert result.passed is True
        assert result.attempt == 2

    @pytest.mark.asyncio
    async def test_fails_closed_after_max_retries(self, tmp_path: Path) -> None:
        """Should return passed=False after all retries exhausted."""
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        async def mock_run_async(
            self: object, cmd: list[str], **kwargs: object
        ) -> CommandResult:
            # Find -o flag and write invalid response
            for i, arg in enumerate(cmd):
                if arg == "-o" and i + 1 < len(cmd):
                    output_path = cmd[i + 1]
                    Path(output_path).write_text("Not JSON at all")
                    break

            return CommandResult(
                command=cmd,
                returncode=0,
                stdout="",
                stderr="",
            )

        with patch(
            "src.codex_review.CommandRunner.run_async",
            mock_run_async,
        ):
            result = await run_codex_review(repo_path, "abc1234", max_retries=2)

        assert result.passed is False
        assert result.parse_error is not None
        assert result.attempt == 2


class TestCodexExecApproach:
    """Test that we use codex exec with --output-schema instead of codex review.

    The codex review --commit CLI does not allow combining with a positional PROMPT,
    so we use codex exec with --output-schema to get structured JSON responses.
    """

    @pytest.mark.asyncio
    async def test_uses_codex_exec_not_review(
        self, tmp_path: Path, mock_codex_script: tuple[Path, Path]
    ) -> None:
        """
        Verify we use 'codex exec' with output schema, not 'codex review'.

        The codex review --commit CLI cannot accept custom prompts, so we use
        codex exec with --output-schema for structured JSON output.
        """
        script_path, invocation_log = mock_codex_script
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        env = {
            **os.environ,
            "PATH": f"{script_path.parent}:{os.environ.get('PATH', '')}",
            "MOCK_CODEX_STDOUT": '{"passed": true, "issues": []}',
        }

        with patch.dict(os.environ, env, clear=True):
            await run_codex_review(repo_path, "abc1234")

        invocations = [
            json.loads(line) for line in invocation_log.read_text().strip().split("\n")
        ]
        assert len(invocations) == 1

        args = invocations[0]["args"]

        # Should use 'exec' subcommand, not 'review'
        assert args[0] == "exec", f"Should use 'codex exec', got args: {args}"
        # Should NOT use 'review' at all
        assert "review" not in args, f"Should not use 'review' subcommand: {args}"
        # Should have --output-schema for structured JSON
        assert "--output-schema" in args, f"Should use --output-schema: {args}"
        # Should have -o for output file
        assert "-o" in args, f"Should use -o for output file: {args}"


class TestBaselineCommitParameter:
    """Tests for the baseline_commit parameter in run_codex_review.

    When a baseline_commit is provided, the review should look at the cumulative
    diff from baseline to the current commit, rather than just the current commit
    vs its parent. This is important for retry scenarios.
    """

    @pytest.mark.asyncio
    async def test_uses_baseline_in_prompt_when_provided(
        self, tmp_path: Path, mock_codex_script: tuple[Path, Path]
    ) -> None:
        """When baseline_commit is provided, the prompt should describe cumulative diff."""
        script_path, invocation_log = mock_codex_script
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        env = {
            **os.environ,
            "PATH": f"{script_path.parent}:{os.environ.get('PATH', '')}",
            "MOCK_CODEX_STDOUT": '{"passed": true, "issues": []}',
        }

        with patch.dict(os.environ, env, clear=True):
            await run_codex_review(
                repo_path,
                "currentsha",
                baseline_commit="baselinesha",
            )

        invocations = [
            json.loads(line) for line in invocation_log.read_text().strip().split("\n")
        ]
        assert len(invocations) == 1

        # The prompt (last argument) should mention both baseline and current commit
        prompt_arg = invocations[0]["args"][-1]
        assert "baselinesha" in prompt_arg, "Prompt should include baseline commit"
        assert "currentsha" in prompt_arg, "Prompt should include current commit"
        assert "cumulative" in prompt_arg, "Prompt should mention cumulative changes"

    @pytest.mark.asyncio
    async def test_uses_parent_diff_when_no_baseline(
        self, tmp_path: Path, mock_codex_script: tuple[Path, Path]
    ) -> None:
        """Without baseline_commit, the prompt should describe commit vs parent."""
        script_path, invocation_log = mock_codex_script
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        env = {
            **os.environ,
            "PATH": f"{script_path.parent}:{os.environ.get('PATH', '')}",
            "MOCK_CODEX_STDOUT": '{"passed": true, "issues": []}',
        }

        with patch.dict(os.environ, env, clear=True):
            await run_codex_review(repo_path, "abc1234")

        invocations = [
            json.loads(line) for line in invocation_log.read_text().strip().split("\n")
        ]
        assert len(invocations) == 1

        # The prompt (last argument) should mention "vs its parent"
        prompt_arg = invocations[0]["args"][-1]
        assert "abc1234" in prompt_arg, "Prompt should include commit SHA"
        assert "parent" in prompt_arg, "Prompt should mention parent comparison"
        assert "cumulative" not in prompt_arg, (
            "Should not mention cumulative without baseline"
        )

    @pytest.mark.asyncio
    async def test_baseline_none_uses_parent_diff(
        self, tmp_path: Path, mock_codex_script: tuple[Path, Path]
    ) -> None:
        """Explicitly passing None for baseline_commit should behave same as omitting it."""
        script_path, invocation_log = mock_codex_script
        repo_path = tmp_path / "repo"
        repo_path.mkdir()

        env = {
            **os.environ,
            "PATH": f"{script_path.parent}:{os.environ.get('PATH', '')}",
            "MOCK_CODEX_STDOUT": '{"passed": true, "issues": []}',
        }

        with patch.dict(os.environ, env, clear=True):
            await run_codex_review(
                repo_path,
                "def5678",
                baseline_commit=None,
            )

        invocations = [
            json.loads(line) for line in invocation_log.read_text().strip().split("\n")
        ]
        assert len(invocations) == 1

        prompt_arg = invocations[0]["args"][-1]
        assert "parent" in prompt_arg, "Should use parent diff when baseline is None"

"""Unit tests for validation helper functions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from src.domain.validation.helpers import (
    _generate_fixture_programmatically,
    annotate_issue,
    check_e2e_prereqs,
    decode_timeout_output,
    format_step_output,
    get_ready_issue_id,
    init_fixture_repo,
    tail,
    write_fixture_repo,
)
from src.infra.tools.command_runner import CommandResult


@pytest.mark.unit
class TestTail:
    """Test the tail function."""

    def test_empty_string(self) -> None:
        """Returns empty string for empty input."""
        assert tail("") == ""

    def test_short_text_unchanged(self) -> None:
        """Short text within limits is unchanged."""
        text = "line1\nline2\nline3"
        assert tail(text, max_chars=100, max_lines=10) == text

    def test_truncates_to_max_lines(self) -> None:
        """Truncates to last N lines."""
        lines = [f"line{i}" for i in range(10)]
        text = "\n".join(lines)
        result = tail(text, max_chars=1000, max_lines=3)
        assert result == "line7\nline8\nline9"

    def test_truncates_to_max_chars(self) -> None:
        """Truncates to last N characters after line limiting."""
        text = "a" * 100
        result = tail(text, max_chars=10, max_lines=100)
        assert result == "a" * 10

    def test_applies_both_limits(self) -> None:
        """Applies both line and character limits."""
        lines = [f"line{i:03d}" for i in range(100)]
        text = "\n".join(lines)
        result = tail(text, max_chars=50, max_lines=5)
        # After line truncation: "line095\nline096\nline097\nline098\nline099" (47 chars)
        assert len(result) <= 50
        assert result.endswith("line099")


@pytest.mark.unit
class TestDecodeTimeoutOutput:
    """Test the decode_timeout_output function."""

    def test_none_returns_empty(self) -> None:
        """None input returns empty string."""
        assert decode_timeout_output(None) == ""

    def test_string_input(self) -> None:
        """String input is passed through tail."""
        assert decode_timeout_output("hello") == "hello"

    def test_bytes_input(self) -> None:
        """Bytes input is decoded and tailed."""
        assert decode_timeout_output(b"hello bytes") == "hello bytes"

    def test_bytes_long_input_tailed(self) -> None:
        """Long bytes input is tailed."""
        long_bytes = b"x" * 10000
        result = decode_timeout_output(long_bytes)
        assert len(result) <= 800  # Default max_chars in tail


@pytest.mark.unit
class TestFormatStepOutput:
    """Test the format_step_output function."""

    def test_prefers_stderr(self) -> None:
        """Prefers stderr when both are available."""
        result = format_step_output("stdout", "stderr")
        assert "stderr:" in result
        assert "stdout:" not in result

    def test_uses_stdout_when_no_stderr(self) -> None:
        """Uses stdout when stderr is empty."""
        result = format_step_output("stdout", "")
        assert "stdout: stdout" in result

    def test_empty_when_both_empty(self) -> None:
        """Returns empty string when both are empty."""
        result = format_step_output("", "")
        assert result == ""


@pytest.mark.unit
class TestCheckE2EPrereqs:
    """Test the check_e2e_prereqs function."""

    def test_mala_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns error when mala CLI is not found."""
        monkeypatch.setattr("shutil.which", lambda cmd: None)
        result = check_e2e_prereqs({})
        assert result == "E2E prereq missing: mala CLI not found in PATH"

    def test_bd_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns error when bd CLI is not found."""

        def which(cmd: str) -> str | None:
            return "/usr/bin/mala" if cmd == "mala" else None

        monkeypatch.setattr("shutil.which", which)
        result = check_e2e_prereqs({})
        assert result == "E2E prereq missing: br CLI not found in PATH"

    def test_both_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Returns None when both CLIs are present."""

        def which(cmd: str) -> str:
            return f"/usr/bin/{cmd}"

        monkeypatch.setattr("shutil.which", which)
        result = check_e2e_prereqs({})
        assert result is None


@pytest.mark.unit
class TestGenerateFixtureProgrammatically:
    """Test the _generate_fixture_programmatically function."""

    def test_creates_expected_files(self, tmp_path: Path) -> None:
        """Creates the expected fixture files."""
        _generate_fixture_programmatically(tmp_path)

        # Check app.py exists with the bug
        app_py = tmp_path / "src" / "app.py"
        assert app_py.exists()
        content = app_py.read_text()
        assert "def add(a: int, b: int)" in content
        assert "return a - b" in content  # The bug

        # Check test file exists
        test_py = tmp_path / "tests" / "test_app.py"
        assert test_py.exists()
        assert "def test_add" in test_py.read_text()

        # Check pyproject.toml
        pyproject = tmp_path / "pyproject.toml"
        assert pyproject.exists()
        assert "mala-e2e-fixture" in pyproject.read_text()

        # Check mala.yaml
        mala_yaml = tmp_path / "mala.yaml"
        assert mala_yaml.exists()
        assert "preset: python-uv" in mala_yaml.read_text()


@pytest.mark.unit
class TestWriteFixtureRepo:
    """Test the write_fixture_repo function."""

    def test_uses_programmatic_when_no_template(self, tmp_path: Path) -> None:
        """Falls back to programmatic generation when fixture template missing."""
        # Use a fake repo_root that doesn't have the fixture
        fake_root = tmp_path / "fake_root"
        fake_root.mkdir()

        repo_path = tmp_path / "fixture"
        repo_path.mkdir()

        write_fixture_repo(repo_path, repo_root=fake_root)

        # Should have created files programmatically
        assert (repo_path / "src" / "app.py").exists()
        assert (repo_path / "mala.yaml").exists()

    def test_uses_template_when_available(self, tmp_path: Path) -> None:
        """Uses fixture template when available."""
        # Create a fake fixture template
        fake_root = tmp_path / "fake_root"
        fixture_dir = fake_root / "tests" / "fixtures" / "e2e-fixture"
        fixture_dir.mkdir(parents=True)
        (fixture_dir / "custom.txt").write_text("from template")

        repo_path = tmp_path / "fixture"
        repo_path.mkdir()

        write_fixture_repo(repo_path, repo_root=fake_root)

        # Should have copied the template
        assert (repo_path / "custom.txt").exists()
        assert (repo_path / "custom.txt").read_text() == "from template"


@pytest.mark.unit
class TestInitFixtureRepo:
    """Test the init_fixture_repo function."""

    def test_runs_all_commands_on_success(self, tmp_path: Path) -> None:
        """Runs git init, config, add, commit, bd init, bd create."""
        commands_run: list[list[str]] = []

        mock_runner = MagicMock()

        def fake_run(cmd: list[str], **kwargs: object) -> CommandResult:
            commands_run.append(cmd)
            return CommandResult(command=cmd, returncode=0, stdout="", stderr="")

        mock_runner.run = fake_run

        result = init_fixture_repo(tmp_path, mock_runner)

        assert result is None  # Success
        assert len(commands_run) == 7
        assert commands_run[0] == ["git", "init"]
        assert commands_run[5] == ["br", "init"]
        assert commands_run[6][0:2] == ["br", "create"]

    def test_returns_error_on_failure(self, tmp_path: Path) -> None:
        """Returns error message when a command fails."""
        mock_runner = MagicMock()
        mock_runner.run.return_value = CommandResult(
            command=["git", "init"],
            returncode=1,
            stdout="",
            stderr="fatal: already a git repo",
        )

        result = init_fixture_repo(tmp_path, mock_runner)

        assert result is not None
        assert "git init" in result
        assert "exit 1" in result


@pytest.mark.unit
class TestGetReadyIssueId:
    """Test the get_ready_issue_id function."""

    def test_returns_first_ready_issue_id(self, tmp_path: Path) -> None:
        """Returns the ID of the first ready issue."""
        mock_runner = MagicMock()
        issues = [{"id": "issue-1"}, {"id": "issue-2"}]
        mock_runner.run.return_value = CommandResult(
            command=["br", "ready", "--json"],
            returncode=0,
            stdout=json.dumps(issues),
            stderr="",
        )

        result = get_ready_issue_id(tmp_path, mock_runner)

        assert result == "issue-1"

    def test_returns_none_on_failure(self, tmp_path: Path) -> None:
        """Returns None when bd ready fails."""
        mock_runner = MagicMock()
        mock_runner.run.return_value = CommandResult(
            command=["br", "ready", "--json"],
            returncode=1,
            stdout="",
            stderr="error",
        )

        result = get_ready_issue_id(tmp_path, mock_runner)

        assert result is None

    def test_returns_none_on_invalid_json(self, tmp_path: Path) -> None:
        """Returns None when bd returns invalid JSON."""
        mock_runner = MagicMock()
        mock_runner.run.return_value = CommandResult(
            command=["br", "ready", "--json"],
            returncode=0,
            stdout="not json",
            stderr="",
        )

        result = get_ready_issue_id(tmp_path, mock_runner)

        assert result is None

    def test_returns_none_when_no_issues(self, tmp_path: Path) -> None:
        """Returns None when there are no ready issues."""
        mock_runner = MagicMock()
        mock_runner.run.return_value = CommandResult(
            command=["br", "ready", "--json"],
            returncode=0,
            stdout="[]",
            stderr="",
        )

        result = get_ready_issue_id(tmp_path, mock_runner)

        assert result is None

    def test_skips_non_string_ids(self, tmp_path: Path) -> None:
        """Skips issues with non-string IDs."""
        mock_runner = MagicMock()
        issues = [{"id": 123}, {"id": "valid-id"}]
        mock_runner.run.return_value = CommandResult(
            command=["br", "ready", "--json"],
            returncode=0,
            stdout=json.dumps(issues),
            stderr="",
        )

        result = get_ready_issue_id(tmp_path, mock_runner)

        assert result == "valid-id"


@pytest.mark.unit
class TestAnnotateIssue:
    """Test the annotate_issue function."""

    def test_calls_bd_update_with_notes(self, tmp_path: Path) -> None:
        """Calls bd update with test plan notes."""
        mock_runner = MagicMock()
        mock_runner.run.return_value = CommandResult(
            command=[], returncode=0, stdout="", stderr=""
        )

        annotate_issue(tmp_path, "issue-1", mock_runner)

        mock_runner.run.assert_called_once()
        call_args = mock_runner.run.call_args[0][0]
        assert call_args[0:3] == ["br", "update", "issue-1"]
        assert "--notes" in call_args

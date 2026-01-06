"""Integration tests for mala logs subcommand.

These tests exercise the end-to-end CLI path via CliRunner, verifying that:
1. The logs subcommand is registered correctly via app.add_typer()
2. All subcommands (list, sessions, show) are accessible
3. Help text displays correct options/arguments

Stub command tests use dynamic pytest.xfail() so they:
- Xfail when stubs raise NotImplementedError (expected behavior)
- Fail loudly if a different exception occurs (regression detection)
- Pass when implementation lands (prompting test updates)
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from src.cli.cli import app

pytestmark = pytest.mark.integration

runner = CliRunner()


class TestLogsSubcommandRegistration:
    """Tests verifying logs subcommand is properly registered."""

    def test_logs_help_shows_subcommands(self) -> None:
        """Verify 'mala logs --help' shows available subcommands."""
        result = runner.invoke(app, ["logs", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "sessions" in result.output
        assert "show" in result.output

    def test_logs_list_help_shows_options(self) -> None:
        """Verify 'mala logs list --help' shows --json and --all options."""
        result = runner.invoke(app, ["logs", "list", "--help"])
        assert result.exit_code == 0
        assert "--json" in result.output
        assert "--all" in result.output

    def test_logs_sessions_help_shows_options(self) -> None:
        """Verify 'mala logs sessions --help' shows --issue, --json, --all."""
        result = runner.invoke(app, ["logs", "sessions", "--help"])
        assert result.exit_code == 0
        assert "--issue" in result.output
        assert "--json" in result.output
        assert "--all" in result.output

    def test_logs_show_help_shows_run_id_and_json(self) -> None:
        """Verify 'mala logs show --help' shows run_id argument and --json."""
        result = runner.invoke(app, ["logs", "show", "--help"])
        assert result.exit_code == 0
        assert "RUN_ID" in result.output or "run_id" in result.output.lower()
        assert "--json" in result.output


class TestLogsListCommand:
    """Tests for 'mala logs list' command (expected to fail until T003)."""

    def test_list_basic_invocation(self) -> None:
        """Test basic 'mala logs list' invocation."""
        result = runner.invoke(app, ["logs", "list"])
        if isinstance(result.exception, NotImplementedError):
            pytest.xfail("Stub raises NotImplementedError - T003")
        assert result.exception is None
        assert result.exit_code == 0

    def test_list_with_json_flag(self) -> None:
        """Test 'mala logs list --json' produces valid output."""
        result = runner.invoke(app, ["logs", "list", "--json"])
        if isinstance(result.exception, NotImplementedError):
            pytest.xfail("Stub raises NotImplementedError - T003")
        assert result.exception is None
        assert result.exit_code == 0

    def test_list_with_all_flag(self) -> None:
        """Test 'mala logs list --all' shows all runs."""
        result = runner.invoke(app, ["logs", "list", "--all"])
        if isinstance(result.exception, NotImplementedError):
            pytest.xfail("Stub raises NotImplementedError - T003")
        assert result.exception is None
        assert result.exit_code == 0


class TestLogsSessionsCommand:
    """Tests for 'mala logs sessions' command (expected to fail until T004)."""

    def test_sessions_basic_invocation(self) -> None:
        """Test basic 'mala logs sessions' invocation."""
        result = runner.invoke(app, ["logs", "sessions"])
        if isinstance(result.exception, NotImplementedError):
            pytest.xfail("Stub raises NotImplementedError - T004")
        assert result.exception is None
        assert result.exit_code == 0

    def test_sessions_with_issue_filter(self) -> None:
        """Test 'mala logs sessions --issue test-123' filters by issue."""
        result = runner.invoke(app, ["logs", "sessions", "--issue", "test-123"])
        if isinstance(result.exception, NotImplementedError):
            pytest.xfail("Stub raises NotImplementedError - T004")
        assert result.exception is None
        assert result.exit_code == 0

    def test_sessions_with_json_flag(self) -> None:
        """Test 'mala logs sessions --json' produces valid output."""
        result = runner.invoke(app, ["logs", "sessions", "--json"])
        if isinstance(result.exception, NotImplementedError):
            pytest.xfail("Stub raises NotImplementedError - T004")
        assert result.exception is None
        assert result.exit_code == 0

    def test_sessions_with_all_flag(self) -> None:
        """Test 'mala logs sessions --all' shows all sessions."""
        result = runner.invoke(app, ["logs", "sessions", "--all"])
        if isinstance(result.exception, NotImplementedError):
            pytest.xfail("Stub raises NotImplementedError - T004")
        assert result.exception is None
        assert result.exit_code == 0


class TestLogsShowCommand:
    """Tests for 'mala logs show' command (expected to fail until T005)."""

    def test_show_basic_invocation(self) -> None:
        """Test 'mala logs show <run_id>' invocation."""
        result = runner.invoke(app, ["logs", "show", "abc12345"])
        if isinstance(result.exception, NotImplementedError):
            pytest.xfail("Stub raises NotImplementedError - T005")
        assert result.exception is None
        assert result.exit_code == 0

    def test_show_with_json_flag(self) -> None:
        """Test 'mala logs show <run_id> --json' produces valid output."""
        result = runner.invoke(app, ["logs", "show", "abc12345", "--json"])
        if isinstance(result.exception, NotImplementedError):
            pytest.xfail("Stub raises NotImplementedError - T005")
        assert result.exception is None
        assert result.exit_code == 0

    def test_show_missing_run_id_fails(self) -> None:
        """Test 'mala logs show' without run_id argument fails."""
        result = runner.invoke(app, ["logs", "show"])
        # Missing required argument should cause non-zero exit
        assert result.exit_code != 0

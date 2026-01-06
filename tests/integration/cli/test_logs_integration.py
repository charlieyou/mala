"""Integration tests for mala logs subcommand.

These tests exercise the end-to-end CLI path via CliRunner, verifying that:
1. The logs subcommand is registered correctly via app.add_typer()
2. All subcommands (list, sessions, show) are accessible
3. Help text displays correct options/arguments

Stub command tests use dynamic pytest.xfail() so they:
- Xfail when stubs raise NotImplementedError (expected behavior)
- Fail loudly if a different exception occurs (regression detection)
- Fail explicitly when implementation lands (prompting test updates via XPASS-like signal)
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
    """Tests for 'mala logs list' command."""

    def test_list_basic_invocation(self) -> None:
        """Test basic 'mala logs list' invocation."""
        result = runner.invoke(app, ["logs", "list"])
        assert result.exit_code == 0
        # Empty runs is valid - shows "No runs found" message
        assert result.exception is None

    def test_list_with_json_flag(self) -> None:
        """Test 'mala logs list --json' produces valid JSON output."""
        result = runner.invoke(app, ["logs", "list", "--json"])
        assert result.exit_code == 0
        assert result.exception is None
        # Output should be valid JSON (either [] or list of runs)
        import json

        parsed = json.loads(result.output)
        assert isinstance(parsed, list)

    def test_list_with_all_flag(self) -> None:
        """Test 'mala logs list --all' shows all runs."""
        result = runner.invoke(app, ["logs", "list", "--all"])
        assert result.exit_code == 0
        assert result.exception is None


class TestLogsSessionsCommand:
    """Tests for 'mala logs sessions' command."""

    def test_sessions_requires_issue(self) -> None:
        """Test 'mala logs sessions' without --issue fails."""
        result = runner.invoke(app, ["logs", "sessions"])
        assert result.exit_code != 0
        # Should indicate missing required option
        assert "--issue" in result.output

    def test_sessions_with_issue_filter(self) -> None:
        """Test 'mala logs sessions --issue test-123' filters by issue."""
        result = runner.invoke(app, ["logs", "sessions", "--issue", "test-123"])
        assert result.exit_code == 0
        # Empty result expected when no matching sessions
        assert "No sessions found" in result.output

    def test_sessions_with_json_flag(self) -> None:
        """Test 'mala logs sessions --json --issue' produces valid JSON output."""
        result = runner.invoke(
            app, ["logs", "sessions", "--issue", "test-123", "--json"]
        )
        assert result.exit_code == 0
        # Should produce valid JSON (empty array when no sessions)
        import json

        data = json.loads(result.output)
        assert isinstance(data, list)
        assert data == []

    def test_sessions_with_all_flag(self) -> None:
        """Test 'mala logs sessions --all --issue' runs without error."""
        result = runner.invoke(
            app, ["logs", "sessions", "--issue", "test-123", "--all"]
        )
        assert result.exit_code == 0
        # When no sessions match, should show "No sessions found"
        assert "No sessions found" in result.output


class TestLogsShowCommand:
    """Tests for 'mala logs show' command."""

    def test_show_basic_invocation(self) -> None:
        """Test 'mala logs show <run_id>' returns not-found for nonexistent run."""
        result = runner.invoke(app, ["logs", "show", "abc12345"])
        # Exit code 1 for not found
        assert result.exit_code == 1
        assert "No run found" in result.output

    def test_show_with_json_flag(self) -> None:
        """Test 'mala logs show <run_id> --json' produces valid JSON error for not found."""
        import json

        result = runner.invoke(app, ["logs", "show", "abc12345", "--json"])
        # Exit code 1 for not found
        assert result.exit_code == 1
        output = json.loads(result.output)
        assert output["error"] == "not_found"

    def test_show_missing_run_id_fails(self) -> None:
        """Test 'mala logs show' without run_id argument fails."""
        result = runner.invoke(app, ["logs", "show"])
        # Missing required argument should cause non-zero exit
        assert result.exit_code != 0

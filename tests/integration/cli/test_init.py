"""Integration tests for mala init command.

These tests exercise the end-to-end CLI path via CliRunner, verifying that:
1. The init command is registered correctly
2. --help works and shows --dry-run option
3. Interactive flows produce valid YAML output

Most tests will fail initially until T002 implements the full command.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
import yaml
from typer.testing import CliRunner

from src.cli.cli import app

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.integration

runner = CliRunner()


def _extract_yaml_from_output(output: str) -> str:
    """Extract YAML content from CLI output by filtering prompt lines.

    CliRunner mixes stderr into stdout by default. This filters out CLI prompt
    patterns using precise regexes that won't match YAML values.
    """
    import re

    # Precise patterns that match exact CLI prompts, not YAML content
    prompt_patterns = [
        r"^Select configuration:$",  # Menu header
        r"^\s+\d+\)\s+\S+$",  # Menu items: "  1) go"
        r"^Enter choice \[.*\]: ",  # Choice prompt with brackets
        r"^Command for '\w+' \(Enter to skip\): ",  # Command prompts
        r"^Invalid choice, please enter \d+-\d+:$",  # Validation error
        r"^Run `mala run` to start$",  # Success tip
        r"^$",  # Blank lines
        r"^\s*[┏┗┡━]",  # Rich table borders (top/bottom/divider)
        r"^\s*[┃│]",  # Rich table row content (header/data)
        r"^\s*[└─┴]",  # Rich table bottom corners
        r"^.*Available Trigger Types.*$",  # Table title
    ]
    combined = re.compile("|".join(prompt_patterns))

    lines = output.strip().split("\n")
    return "\n".join(line for line in lines if not combined.match(line))


class TestInitHelp:
    """Tests for init --help (should pass immediately)."""

    def test_help_shows_dry_run_option(self) -> None:
        """Verify 'mala init --help' shows --dry-run option."""
        result = runner.invoke(app, ["init", "--help"])
        assert result.exit_code == 0
        assert "--dry-run" in result.output


class TestInitDryRun:
    """Tests for init --dry-run mode."""

    def test_dry_run_preset(self) -> None:
        """--dry-run with --preset outputs valid YAML to stdout."""
        result = runner.invoke(
            app,
            ["init", "--dry-run", "--preset", "go", "--yes"],
        )
        assert result.exit_code == 0
        yaml_output = _extract_yaml_from_output(result.output)
        config = yaml.safe_load(yaml_output)
        assert config.get("preset") == "go"
        # --yes uses computed defaults (evidence_check and validation_triggers)
        assert "evidence_check" in config
        assert "validation_triggers" in config

    def test_dry_run_no_backup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Existing file + --dry-run leaves file unchanged."""
        monkeypatch.chdir(tmp_path)
        mala_yaml = tmp_path / "mala.yaml"
        original_content = "preset: existing\n"
        mala_yaml.write_text(original_content)

        result = runner.invoke(
            app,
            ["init", "--dry-run", "--preset", "go", "--yes"],
        )
        # Command outputs to stdout, doesn't touch file
        assert result.exit_code == 0
        assert mala_yaml.read_text() == original_content

    def test_dry_run_skip_evidence_triggers(self) -> None:
        """--dry-run with --skip-evidence --skip-triggers omits those sections."""
        result = runner.invoke(
            app,
            [
                "init",
                "--dry-run",
                "--preset",
                "go",
                "--skip-evidence",
                "--skip-triggers",
            ],
        )
        assert result.exit_code == 0
        yaml_output = _extract_yaml_from_output(result.output)
        config = yaml.safe_load(yaml_output)
        assert config == {"preset": "go"}


class TestInitCustomFlow:
    """Tests for init custom commands flow via questionary (mocked)."""

    def test_custom_flow_via_mocked_questionary(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_questionary: MagicMock,
    ) -> None:
        """Custom flow via mocked questionary produces valid config."""
        monkeypatch.chdir(tmp_path)
        # Mock _is_interactive to enable interactive mode
        monkeypatch.setattr("src.cli.cli._is_interactive", lambda: True)

        # Mock questionary returns: select "custom", then per_issue_review disabled
        # First select is preset selection, second is per_issue_review enable
        mock_questionary.select_returns = ["custom", False]
        mock_questionary.confirm_return = False  # Skip evidence and trigger prompts

        # Mock text for custom commands
        # Order: build, e2e, format, lint, setup, test, typecheck
        text_responses = iter(["uv run build", "", "", "", "uv sync", "", ""])

        def make_text(*args: object, **kwargs: object) -> MagicMock:
            result = MagicMock()
            result.ask.return_value = next(text_responses, "")
            return result

        monkeypatch.setattr("src.cli.cli.questionary.text", make_text)

        result = runner.invoke(app, ["init"])

        assert result.exit_code == 0
        mala_yaml = tmp_path / "mala.yaml"
        assert mala_yaml.exists()
        config = yaml.safe_load(mala_yaml.read_text())
        assert config.get("commands", {}).get("build") == "uv run build"
        assert config.get("commands", {}).get("setup") == "uv sync"


class TestInitBackup:
    """Tests for backup creation when mala.yaml exists."""

    def test_backup_created(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Existing mala.yaml creates .bak with old content."""
        monkeypatch.chdir(tmp_path)
        mala_yaml = tmp_path / "mala.yaml"
        old_content = "preset: old\n"
        mala_yaml.write_text(old_content)

        result = runner.invoke(
            app,
            ["init", "--preset", "go", "--yes"],
        )

        assert result.exit_code == 0
        backup = tmp_path / "mala.yaml.bak"
        assert backup.exists()
        assert backup.read_text() == old_content


class TestInitValidation:
    """Tests for validation failure scenarios."""

    def test_validation_fail_empty_commands_via_questionary(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mock_questionary: MagicMock,
    ) -> None:
        """Custom flow with empty commands -> exit 1, error in output."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("src.cli.cli._is_interactive", lambda: True)

        # First select returns "custom" for preset selection
        # Second select returns "Abort init" for validation error recovery
        select_responses = iter(["custom", "Abort init"])

        def make_select(*args: object, **kwargs: object) -> MagicMock:
            result = MagicMock()
            result.ask.return_value = next(select_responses, "Abort init")
            return result

        monkeypatch.setattr("src.cli.cli.questionary.select", make_select)
        mock_questionary.confirm_return = False

        # All text prompts return empty
        def make_text(*args: object, **kwargs: object) -> MagicMock:
            result = MagicMock()
            result.ask.return_value = ""
            return result

        monkeypatch.setattr("src.cli.cli.questionary.text", make_text)

        result = runner.invoke(app, ["init"])

        assert result.exit_code == 1


class TestInitInputValidation:
    """Tests for input validation and CLI flag errors."""

    def test_yes_without_preset_fails(self) -> None:
        """--yes without --preset shows error."""
        result = runner.invoke(app, ["init", "--yes"])
        assert result.exit_code == 1
        assert "--yes requires --preset" in result.output

    def test_non_tty_without_flags_fails(self) -> None:
        """Non-TTY mode without proper flags shows error."""
        result = runner.invoke(app, ["init"])
        assert result.exit_code == 1
        assert "non-interactive" in result.output.lower()

    def test_invalid_preset_fails(self) -> None:
        """Unknown preset shows error."""
        result = runner.invoke(app, ["init", "--preset", "nonexistent", "--yes"])
        assert result.exit_code == 1
        assert "unknown preset" in result.output.lower()


# ============================================================================
# New test classes for T001: Evidence checks and triggers
# These tests specify behavior that will be implemented in T002/T003
# ============================================================================


@pytest.fixture
def mock_questionary(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Fixture that mocks questionary.confirm/checkbox/select/text.

    Returns a mock object with methods to set return values for prompts.
    Set select_returns as a list for sequenced returns across multiple prompts.
    Set text_return for text prompt return value.
    """
    mock = MagicMock()

    # Default return values
    mock.confirm_return = True
    mock.checkbox_return = []
    mock.select_return = None
    mock.select_returns: list[object] = []  # For sequenced select returns
    mock.text_return = ""

    # Track call index for sequenced returns
    mock._select_call_index = 0

    def make_confirm(*args: object, **kwargs: object) -> MagicMock:
        result = MagicMock()
        result.ask.return_value = mock.confirm_return
        return result

    def make_checkbox(*args: object, **kwargs: object) -> MagicMock:
        result = MagicMock()
        result.ask.return_value = mock.checkbox_return
        return result

    def make_select(*args: object, **kwargs: object) -> MagicMock:
        result = MagicMock()
        # Use sequenced returns if available, otherwise use single value
        if mock.select_returns:
            if mock._select_call_index < len(mock.select_returns):
                result.ask.return_value = mock.select_returns[mock._select_call_index]
                mock._select_call_index += 1
            else:
                result.ask.return_value = None
        else:
            result.ask.return_value = mock.select_return
        return result

    def make_text(*args: object, **kwargs: object) -> MagicMock:
        result = MagicMock()
        result.ask.return_value = mock.text_return
        return result

    monkeypatch.setattr("src.cli.cli.questionary.confirm", make_confirm)
    monkeypatch.setattr("src.cli.cli.questionary.checkbox", make_checkbox)
    monkeypatch.setattr("src.cli.cli.questionary.select", make_select)
    monkeypatch.setattr("src.cli.cli.questionary.text", make_text)

    return mock


class TestInitEvidencePrompts:
    """Tests for evidence check prompting behavior."""

    def test_prompt_evidence_check_preset_returns_selection(
        self,
        mock_questionary: MagicMock,
    ) -> None:
        """With preset, confirm Y and selection returns that list."""
        from src.cli.cli import _prompt_evidence_check

        mock_questionary.confirm_return = True
        mock_questionary.checkbox_return = ["test", "lint"]
        result = _prompt_evidence_check(["test", "lint", "typecheck"], is_preset=True)
        assert result == ["test", "lint"]

    def test_prompt_evidence_check_skip_returns_none(
        self,
        mock_questionary: MagicMock,
    ) -> None:
        """User declines configure evidence checks -> returns None."""
        from src.cli.cli import _prompt_evidence_check

        mock_questionary.confirm_return = False
        result = _prompt_evidence_check(["build", "test"], is_preset=False)
        assert result is None

    def test_compute_evidence_defaults_preset_path(self) -> None:
        """Preset path returns intersection with {test, lint}."""
        from src.cli.cli import _compute_evidence_defaults

        result = _compute_evidence_defaults(["test", "lint", "format"], is_preset=True)
        assert result == ["test", "lint"]

    def test_compute_evidence_defaults_custom_path(self) -> None:
        """Custom path returns empty list."""
        from src.cli.cli import _compute_evidence_defaults

        result = _compute_evidence_defaults(["build", "test", "lint"], is_preset=False)
        assert result == []


class TestInitTriggerPrompts:
    """Tests for validation trigger prompting behavior."""

    def test_prompt_run_end_trigger_returns_selection(
        self,
        mock_questionary: MagicMock,
    ) -> None:
        """With preset, confirm Y and selection returns that list."""
        from src.cli.cli import _prompt_run_end_trigger

        mock_questionary.confirm_return = True
        mock_questionary.checkbox_return = ["test", "lint"]
        result = _prompt_run_end_trigger(["test", "lint"], is_preset=True)
        assert result == ["test", "lint"]

    def test_compute_trigger_defaults_excludes_setup_e2e(self) -> None:
        """Preset path excludes setup and e2e from defaults."""
        from src.cli.cli import _compute_trigger_defaults

        result = _compute_trigger_defaults(
            ["setup", "test", "lint", "e2e"], is_preset=True
        )
        assert result == ["test", "lint"]

    def test_compute_trigger_defaults_custom_path(self) -> None:
        """Custom path returns empty list."""
        from src.cli.cli import _compute_trigger_defaults

        result = _compute_trigger_defaults(
            ["setup", "build", "test", "lint"], is_preset=False
        )
        assert result == []

    def test_print_trigger_reference_table_outputs(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Table output contains key trigger types and phrases (printed to stderr)."""
        from src.cli.cli import _print_trigger_reference_table

        _print_trigger_reference_table()
        captured = capsys.readouterr()
        output = captured.err  # Table is printed to stderr
        # Verify semantic content (key phrases), not exact formatting
        assert "epic_completion" in output
        assert "session_end" in output
        assert "periodic" in output
        assert "run_end" in output
        assert "ends" in output.lower() or "verification" in output.lower()


class TestInitPerIssueReviewPrompts:
    """Tests for per-issue review prompting behavior."""

    def test_prompt_per_issue_review_disabled_returns_none(
        self,
        mock_questionary: MagicMock,
    ) -> None:
        """User selecting 'No' returns None (no config section)."""
        from src.cli.cli import _prompt_per_issue_review

        mock_questionary.select_return = False  # User selects "No"
        result = _prompt_per_issue_review()
        assert result is None

    def test_prompt_per_issue_review_enabled_returns_config(
        self,
        mock_questionary: MagicMock,
    ) -> None:
        """User enabling review with all options returns complete config."""
        from src.cli.cli import _prompt_per_issue_review

        # Sequence: enable=True, reviewer_type="cerberus", threshold="P2"
        mock_questionary.select_returns = [True, "cerberus", "P2"]
        mock_questionary.text_return = "5"  # max_retries

        result = _prompt_per_issue_review()

        assert result is not None
        assert result["enabled"] is True
        assert result["reviewer_type"] == "cerberus"
        assert result["max_retries"] == 5
        assert result["finding_threshold"] == "P2"

    def test_prompt_per_issue_review_agent_sdk_reviewer(
        self,
        mock_questionary: MagicMock,
    ) -> None:
        """User selecting agent_sdk reviewer type."""
        from src.cli.cli import _prompt_per_issue_review

        mock_questionary.select_returns = [True, "agent_sdk", "none"]
        mock_questionary.text_return = "3"

        result = _prompt_per_issue_review()

        assert result is not None
        assert result["reviewer_type"] == "agent_sdk"

    def test_prompt_per_issue_review_invalid_retries_defaults(
        self,
        mock_questionary: MagicMock,
    ) -> None:
        """Invalid max_retries input falls back to default of 3."""
        from src.cli.cli import _prompt_per_issue_review

        mock_questionary.select_returns = [True, "cerberus", "none"]
        mock_questionary.text_return = "invalid"  # Non-numeric

        result = _prompt_per_issue_review()

        assert result is not None
        assert result["max_retries"] == 3  # Default

    def test_build_per_issue_review_dict_passthrough(self) -> None:
        """Build function returns config as-is."""
        from src.cli.cli import _build_per_issue_review_dict

        config = {
            "enabled": True,
            "reviewer_type": "cerberus",
            "max_retries": 3,
            "finding_threshold": "none",
        }
        result = _build_per_issue_review_dict(config)
        assert result == config


class TestInitNonInteractive:
    """Tests for non-interactive/defaults behavior."""

    def test_get_preset_command_names_returns_commands(self) -> None:
        """Returns list of command names from preset."""
        from src.cli.cli import _get_preset_command_names

        result = _get_preset_command_names("python-uv")
        # python-uv preset has these commands defined
        assert "test" in result
        assert "lint" in result
        assert "setup" in result

    def test_get_preset_command_names_invalid_raises(self) -> None:
        """Invalid preset raises PresetNotFoundError."""
        from src.cli.cli import _get_preset_command_names
        from src.domain.validation.preset_registry import PresetNotFoundError

        with pytest.raises(PresetNotFoundError):
            _get_preset_command_names("nonexistent-preset")

    def test_build_evidence_check_dict(self) -> None:
        """Returns dict with 'required' key."""
        from src.cli.cli import _build_evidence_check_dict

        result = _build_evidence_check_dict(["test", "lint"])
        assert result == {"required": ["test", "lint"]}

    def test_build_evidence_check_dict_empty(self) -> None:
        """Empty list returns dict with empty 'required'."""
        from src.cli.cli import _build_evidence_check_dict

        result = _build_evidence_check_dict([])
        assert result == {"required": []}

    def test_build_validation_triggers_dict_ref_syntax(self) -> None:
        """Returns dict with run_end.commands using ref syntax and failure_mode."""
        from src.cli.cli import _build_validation_triggers_dict

        result = _build_validation_triggers_dict(["test", "lint"])
        assert result == {
            "run_end": {
                "failure_mode": "continue",
                "commands": [{"ref": "test"}, {"ref": "lint"}],
            }
        }

    def test_prompt_preset_selection_returns_preset(
        self, mock_questionary: MagicMock
    ) -> None:
        """User selecting a preset returns that preset name."""
        from src.cli.cli import _prompt_preset_selection

        mock_questionary.select_return = "python-uv"
        result = _prompt_preset_selection(["go", "python-uv", "rust"])
        assert result == "python-uv"

    def test_prompt_preset_selection_custom_returns_none(
        self, mock_questionary: MagicMock
    ) -> None:
        """User selecting custom returns None."""
        from src.cli.cli import _prompt_preset_selection

        mock_questionary.select_return = "custom"
        result = _prompt_preset_selection(["go", "python-uv", "rust"])
        assert result is None

    def test_prompt_preset_selection_cancelled_raises_keyboard_interrupt(
        self, mock_questionary: MagicMock
    ) -> None:
        """User cancelling (Esc/Ctrl-C) raises KeyboardInterrupt."""
        from src.cli.cli import _prompt_preset_selection

        mock_questionary.select_return = None  # Cancellation returns None from ask()
        with pytest.raises(KeyboardInterrupt):
            _prompt_preset_selection(["go", "python-uv", "rust"])

    def test_prompt_custom_commands_questionary_returns_commands(
        self, mock_questionary: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Returns dict of command name to command string."""
        from src.cli.cli import _prompt_custom_commands_questionary

        # Commands are sorted: build, e2e, format, lint, setup, test, typecheck
        # Provide values for build and setup, skip others
        responses = iter(["cargo build", "", "", "", "cargo install", "", ""])

        def make_text(*args: object, **kwargs: object) -> MagicMock:
            result = MagicMock()
            result.ask.return_value = next(responses, "")
            return result

        monkeypatch.setattr("src.cli.cli.questionary.text", make_text)
        result = _prompt_custom_commands_questionary()
        # Only non-empty commands are returned
        assert result is not None
        assert "build" in result
        assert result["build"] == "cargo build"
        assert "setup" in result
        assert result["setup"] == "cargo install"


class TestInitTriggerTable:
    """Tests for trigger reference table display."""

    def test_trigger_table_contains_all_types(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Table output contains all four trigger types (printed to stderr)."""
        from src.cli.cli import _print_trigger_reference_table

        _print_trigger_reference_table()
        captured = capsys.readouterr()
        output = captured.err  # Table is printed to stderr
        assert "epic_completion" in output
        assert "session_end" in output
        assert "periodic" in output
        assert "run_end" in output

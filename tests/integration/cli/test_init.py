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
        """Input '1' (preset), --dry-run outputs valid YAML to stdout."""
        result = runner.invoke(app, ["init", "--dry-run"], input="1\n")
        assert result.exit_code == 0
        yaml_output = _extract_yaml_from_output(result.output)
        config = yaml.safe_load(yaml_output)
        # Presets are sorted alphabetically: ['go', 'node-npm', 'python-uv', 'rust']
        assert config == {"preset": "go"}

    def test_dry_run_no_backup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Existing file + --dry-run leaves file unchanged."""
        monkeypatch.chdir(tmp_path)
        mala_yaml = tmp_path / "mala.yaml"
        original_content = "preset: existing\n"
        mala_yaml.write_text(original_content)

        result = runner.invoke(app, ["init", "--dry-run"], input="1\n")
        # Command outputs to stdout, doesn't touch file
        assert result.exit_code == 0
        assert mala_yaml.read_text() == original_content

    def test_dry_run_custom(self) -> None:
        """Input '5' (custom), --dry-run outputs valid YAML with indented commands."""
        # 5 = custom, provide setup and build, skip remaining 5 commands
        input_text = "5\nuv sync\nuv run build\n\n\n\n\n\n"
        result = runner.invoke(app, ["init", "--dry-run"], input=input_text)
        assert result.exit_code == 0
        yaml_output = _extract_yaml_from_output(result.output)
        config = yaml.safe_load(yaml_output)
        assert config == {"commands": {"setup": "uv sync", "build": "uv run build"}}


class TestInitCustomFlow:
    """Tests for init custom commands flow."""

    def test_custom_flow(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Input '5' (custom), provide commands, verify mala.yaml content."""
        monkeypatch.chdir(tmp_path)

        # Input: 5 = custom, then provide setup and build commands, skip remaining 5
        # Command order: setup, build, test, lint, format, typecheck, e2e
        input_text = "5\nuv sync\nuv run build\n\n\n\n\n\n"
        result = runner.invoke(app, ["init"], input=input_text)

        assert result.exit_code == 0
        mala_yaml = tmp_path / "mala.yaml"
        assert mala_yaml.exists()
        config = yaml.safe_load(mala_yaml.read_text())
        assert config.get("commands", {}).get("setup") == "uv sync"
        assert config.get("commands", {}).get("build") == "uv run build"


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

        result = runner.invoke(app, ["init"], input="1\n")

        assert result.exit_code == 0
        backup = tmp_path / "mala.yaml.bak"
        assert backup.exists()
        assert backup.read_text() == old_content


class TestInitValidation:
    """Tests for validation failure scenarios."""

    def test_validation_fail_empty_commands(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Input '5' (custom), skip all commands -> exit 1, error in stderr."""
        monkeypatch.chdir(tmp_path)

        # Input: 5 = custom, then skip all 7 commands (just newlines)
        input_text = "5\n\n\n\n\n\n\n\n"
        result = runner.invoke(app, ["init"], input=input_text)

        assert result.exit_code == 1
        assert "error" in result.stderr.lower() or "command" in result.stderr.lower()


class TestInitInputValidation:
    """Tests for input validation and retry loops."""

    def test_invalid_input_loop(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid inputs ('0', '6', 'abc') are rejected, then '1' succeeds."""
        monkeypatch.chdir(tmp_path)

        # Input: invalid choices, then valid choice
        input_text = "0\n6\nabc\n1\n"
        result = runner.invoke(app, ["init", "--dry-run"], input=input_text)

        assert result.exit_code == 0
        yaml_output = _extract_yaml_from_output(result.output)
        config = yaml.safe_load(yaml_output)
        # Presets are sorted alphabetically: ['go', 'node-npm', 'python-uv', 'rust']
        assert config == {"preset": "go"}


# ============================================================================
# New test classes for T001: Evidence checks and triggers
# These tests specify behavior that will be implemented in T002/T003
# ============================================================================


@pytest.fixture
def mock_questionary(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Fixture that mocks questionary.confirm/checkbox/select.

    Returns a mock object with methods to set return values for prompts.
    """
    mock = MagicMock()

    # Default return values
    mock.confirm_return = True
    mock.checkbox_return = []
    mock.select_return = None

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
        result.ask.return_value = mock.select_return
        return result

    monkeypatch.setattr("src.cli.cli.questionary.confirm", make_confirm)
    monkeypatch.setattr("src.cli.cli.questionary.checkbox", make_checkbox)
    monkeypatch.setattr("src.cli.cli.questionary.select", make_select)

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
        """Table output contains key trigger types and phrases."""
        from src.cli.cli import _print_trigger_reference_table

        _print_trigger_reference_table()
        captured = capsys.readouterr()
        output = captured.out
        # Verify semantic content (key phrases), not exact formatting
        assert "epic_completion" in output
        assert "session_end" in output
        assert "periodic" in output
        assert "run_end" in output
        assert "ends" in output.lower() or "verification" in output.lower()


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
        """Returns dict with run_end.commands using ref syntax."""
        from src.cli.cli import _build_validation_triggers_dict

        result = _build_validation_triggers_dict(["test", "lint"])
        assert result == {
            "run_end": {
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
        assert "build" in result
        assert result["build"] == "cargo build"
        assert "setup" in result
        assert result["setup"] == "cargo install"


class TestInitTriggerTable:
    """Tests for trigger reference table display."""

    def test_trigger_table_contains_all_types(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Table output contains all four trigger types."""
        from src.cli.cli import _print_trigger_reference_table

        _print_trigger_reference_table()
        captured = capsys.readouterr()
        output = captured.out
        assert "epic_completion" in output
        assert "session_end" in output
        assert "periodic" in output
        assert "run_end" in output

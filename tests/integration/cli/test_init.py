"""Integration tests for mala init command.

These tests exercise the end-to-end CLI path via CliRunner, verifying that:
1. The init command is registered correctly
2. --help works and shows --dry-run option
3. Interactive flows produce valid YAML output

Most tests will fail initially until T002 implements the full command.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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

    @pytest.mark.xfail(reason="T003 will implement file operations")
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

    @pytest.mark.xfail(reason="T003 will implement file operations")
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

    @pytest.mark.xfail(reason="T003 will implement exception handling")
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

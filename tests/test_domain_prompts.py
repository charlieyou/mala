"""Tests for domain/prompts.py module.

Tests the prompt loading utilities and build_prompt_validation_commands function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest  # noqa: F401  # pytest imported for fixtures

from src.domain.prompts import (
    build_prompt_validation_commands,
    get_gate_followup_prompt,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestGetGateFollowupPrompt:
    """Tests for get_gate_followup_prompt function."""

    def test_returns_string(self) -> None:
        """get_gate_followup_prompt returns a non-empty string."""
        result = get_gate_followup_prompt()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_template_placeholders(self) -> None:
        """Gate followup prompt contains expected placeholders."""
        result = get_gate_followup_prompt()
        # Check for placeholders that should be in the template
        assert "{attempt}" in result
        assert "{max_attempts}" in result
        assert "{failure_reasons}" in result
        assert "{issue_id}" in result
        # Check for validation command placeholders
        assert "{lint_command}" in result
        assert "{format_command}" in result
        assert "{typecheck_command}" in result
        assert "{test_command}" in result


class TestBuildPromptValidationCommands:
    """Tests for build_prompt_validation_commands function."""

    def test_with_python_uv_preset(self, tmp_path: Path) -> None:
        """Python-uv preset returns correct commands."""
        # Create mala.yaml with python-uv preset
        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text("preset: python-uv\n")

        result = build_prompt_validation_commands(tmp_path)

        assert "ruff check" in result.lint
        assert "ruff format" in result.format
        assert "ty check" in result.typecheck
        assert "pytest" in result.test

    def test_with_go_preset(self, tmp_path: Path) -> None:
        """Go preset returns correct commands."""
        # Create mala.yaml with go preset
        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text("preset: go\n")

        result = build_prompt_validation_commands(tmp_path)

        assert result.lint == "golangci-lint run"
        assert "gofmt" in result.format
        assert result.test == "go test ./..."

    def test_with_custom_commands(self, tmp_path: Path) -> None:
        """Custom commands override preset defaults."""
        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text(
            """
preset: python-uv
commands:
  lint: "custom-lint"
  test: "custom-test"
"""
        )

        result = build_prompt_validation_commands(tmp_path)

        assert result.lint == "custom-lint"
        assert result.test == "custom-test"
        # format and typecheck should still come from preset
        assert "ruff format" in result.format
        assert "ty check" in result.typecheck

    def test_missing_config_returns_defaults(self, tmp_path: Path) -> None:
        """Missing mala.yaml returns default Python/uv commands with isolation flags."""
        # No mala.yaml file exists
        result = build_prompt_validation_commands(tmp_path)

        # Default commands include isolation flags for parallel agent runs
        assert (
            result.lint == "uvx ruff check . --cache-dir=/tmp/ruff-${AGENT_ID:-default}"
        )
        assert (
            result.format
            == "uvx ruff format . --cache-dir=/tmp/ruff-${AGENT_ID:-default}"
        )
        assert result.typecheck == "uvx ty check"
        assert (
            result.test
            == "uv run pytest --cache-dir=/tmp/pytest-${AGENT_ID:-default} --cov-fail-under=0"
        )

    def test_partial_commands_use_fallbacks(self, tmp_path: Path) -> None:
        """Partial commands config uses fallbacks for missing ones."""
        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text(
            """
commands:
  test: "pytest"
  lint: "ruff check ."
"""
        )

        result = build_prompt_validation_commands(tmp_path)

        assert result.test == "pytest"
        assert result.lint == "ruff check ."
        # format and typecheck should use fallbacks
        assert "No format command configured" in result.format
        assert "No typecheck command configured" in result.typecheck


class TestPromptTemplateIntegration:
    """Integration tests for prompt template rendering."""

    def test_implementer_prompt_renders_with_python_commands(
        self, tmp_path: Path
    ) -> None:
        """Implementer prompt renders correctly with Python validation commands."""
        from src.orchestration.prompts import get_implementer_prompt

        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text("preset: python-uv\n")

        cmds = build_prompt_validation_commands(tmp_path)
        prompt = get_implementer_prompt().format(
            issue_id="test-123",
            repo_path=tmp_path,
            lock_dir="/tmp/locks",
            scripts_dir="/scripts",
            agent_id="test-agent",
            lint_command=cmds.lint,
            format_command=cmds.format,
            typecheck_command=cmds.typecheck,
            test_command=cmds.test,
        )

        # Verify Python commands appear in rendered prompt
        assert "ruff check" in prompt
        assert "ruff format" in prompt
        assert "ty check" in prompt
        assert "pytest" in prompt
        # Verify no unsubstituted placeholders remain
        assert "{lint_command}" not in prompt
        assert "{format_command}" not in prompt
        assert "{typecheck_command}" not in prompt
        assert "{test_command}" not in prompt

    def test_implementer_prompt_renders_with_go_commands(self, tmp_path: Path) -> None:
        """Implementer prompt renders correctly with Go validation commands."""
        from src.orchestration.prompts import get_implementer_prompt

        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text("preset: go\n")

        cmds = build_prompt_validation_commands(tmp_path)
        prompt = get_implementer_prompt().format(
            issue_id="test-123",
            repo_path=tmp_path,
            lock_dir="/tmp/locks",
            scripts_dir="/scripts",
            agent_id="test-agent",
            lint_command=cmds.lint,
            format_command=cmds.format,
            typecheck_command=cmds.typecheck,
            test_command=cmds.test,
        )

        # Verify Go commands appear in rendered prompt
        assert "golangci-lint" in prompt
        assert "gofmt" in prompt
        assert "go test" in prompt
        # Verify no Python commands
        assert "ruff" not in prompt
        assert "uvx" not in prompt

    def test_gate_followup_renders_with_commands(self, tmp_path: Path) -> None:
        """Gate followup prompt renders correctly with validation commands."""
        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text("preset: go\n")

        cmds = build_prompt_validation_commands(tmp_path)
        prompt = get_gate_followup_prompt().format(
            attempt=1,
            max_attempts=3,
            failure_reasons="- lint failed\n- tests failed",
            issue_id="test-123",
            lint_command=cmds.lint,
            format_command=cmds.format,
            typecheck_command=cmds.typecheck,
            test_command=cmds.test,
        )

        # Verify Go commands appear
        assert "golangci-lint" in prompt
        assert "go test" in prompt
        # Verify no unsubstituted placeholders
        assert "{" not in prompt or "}" not in prompt.split("{")[-1]

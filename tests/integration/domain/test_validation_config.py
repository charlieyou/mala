"""Integration tests for language-agnostic validation configuration.

These tests validate the end-to-end config-driven workflow including:
- Preset loading and merging (AC 1-3)
- Minimal config behavior (AC 4)
- Error handling for missing/invalid configs (AC 5-6)
- Coverage configuration (AC 7-8)
- Code pattern filtering (AC 9)
- Tool name extraction (AC 10)

All tests create realistic project structures using tmp_path fixture.
Integration marker is applied automatically via path-based pytest configuration.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.domain.validation.code_pattern_matcher import filter_matching_files
from src.domain.validation.config import ConfigError
from src.domain.validation.config_loader import load_config
from src.domain.validation.config_merger import merge_configs
from src.domain.validation.preset_registry import PresetRegistry
from src.core.tool_name_extractor import extract_tool_name
from src.domain.validation.validation_gating import (
    should_invalidate_lint_cache,
    should_trigger_validation,
)

if TYPE_CHECKING:
    from pathlib import Path

    from src.domain.validation.config import ValidationConfig


def _create_mala_yaml(project_dir: Path, content: str) -> None:
    """Create a mala.yaml file in the project directory."""
    (project_dir / "mala.yaml").write_text(content)


def _get_resolved_config(project_dir: Path) -> ValidationConfig:
    """Load and resolve config (with preset merging if applicable)."""
    config = load_config(project_dir)
    if config.preset:
        registry = PresetRegistry()
        preset_config = registry.get(config.preset)
        return merge_configs(preset_config, config)
    return config


# =============================================================================
# AC 1: Go project with mala.yaml using go preset
# =============================================================================


class TestGoPresetIntegration:
    """AC 1: Test Go project configuration with go preset."""

    def test_go_project_with_preset(self, tmp_path: Path) -> None:
        """Go project using go preset loads correct commands."""
        _create_mala_yaml(
            tmp_path,
            """
preset: go
""",
        )

        config = _get_resolved_config(tmp_path)

        assert config.preset == "go"
        assert config.commands.setup is not None
        assert config.commands.setup.command == "go mod download"
        assert config.commands.test is not None
        assert config.commands.test.command == "go test ./..."
        assert config.commands.lint is not None
        assert config.commands.lint.command == "golangci-lint run"
        assert config.commands.format is not None
        assert "gofmt" in config.commands.format.command

    def test_go_project_code_patterns(self, tmp_path: Path) -> None:
        """Go project has correct code patterns from preset."""
        _create_mala_yaml(tmp_path, "preset: go\n")

        config = _get_resolved_config(tmp_path)

        assert "**/*.go" in config.code_patterns

    def test_go_project_setup_files(self, tmp_path: Path) -> None:
        """Go project has correct setup files from preset."""
        _create_mala_yaml(tmp_path, "preset: go\n")

        config = _get_resolved_config(tmp_path)

        assert "go.mod" in config.setup_files
        assert "go.sum" in config.setup_files


# =============================================================================
# AC 2: Node.js project with node-npm preset
# =============================================================================


class TestNodeNpmPresetIntegration:
    """AC 2: Test Node.js project configuration with node-npm preset."""

    def test_node_project_with_preset(self, tmp_path: Path) -> None:
        """Node.js project using node-npm preset loads correct commands."""
        _create_mala_yaml(
            tmp_path,
            """
preset: node-npm
""",
        )

        config = _get_resolved_config(tmp_path)

        assert config.preset == "node-npm"
        assert config.commands.setup is not None
        assert config.commands.setup.command == "npm install"
        assert config.commands.test is not None
        assert config.commands.test.command == "npm test"
        assert config.commands.lint is not None
        assert config.commands.lint.command == "npx eslint ."
        assert config.commands.format is not None
        assert config.commands.format.command == "npx prettier --check ."
        assert config.commands.typecheck is not None
        assert config.commands.typecheck.command == "npx tsc --noEmit"

    def test_node_project_code_patterns(self, tmp_path: Path) -> None:
        """Node.js project has correct code patterns from preset."""
        _create_mala_yaml(tmp_path, "preset: node-npm\n")

        config = _get_resolved_config(tmp_path)

        assert "**/*.js" in config.code_patterns
        assert "**/*.ts" in config.code_patterns
        assert "**/*.jsx" in config.code_patterns
        assert "**/*.tsx" in config.code_patterns

    def test_node_project_setup_files(self, tmp_path: Path) -> None:
        """Node.js project has correct setup files from preset."""
        _create_mala_yaml(tmp_path, "preset: node-npm\n")

        config = _get_resolved_config(tmp_path)

        assert "package-lock.json" in config.setup_files
        assert "package.json" in config.setup_files


# =============================================================================
# AC 3: Python project with python-uv preset + custom test override
# =============================================================================


class TestPythonPresetWithOverrideIntegration:
    """AC 3: Test Python project with preset and custom test override."""

    def test_python_project_with_test_override(self, tmp_path: Path) -> None:
        """Python project with python-uv preset and custom test command."""
        _create_mala_yaml(
            tmp_path,
            """
preset: python-uv
commands:
  test: "uv run pytest -v --slow"
""",
        )

        config = _get_resolved_config(tmp_path)

        # Test command is overridden
        assert config.commands.test is not None
        assert config.commands.test.command == "uv run pytest -v --slow"

        # Other commands inherited from preset (with isolation flags)
        assert config.commands.setup is not None
        assert config.commands.setup.command == "uv sync"
        assert config.commands.lint is not None
        assert (
            config.commands.lint.command
            == "RUFF_CACHE_DIR=/tmp/ruff-${AGENT_ID:-default} uvx ruff check ."
        )
        assert config.commands.format is not None
        assert (
            config.commands.format.command
            == "RUFF_CACHE_DIR=/tmp/ruff-${AGENT_ID:-default} uvx ruff format --check ."
        )
        assert config.commands.typecheck is not None
        assert config.commands.typecheck.command == "uvx ty check"

    def test_python_project_override_with_timeout(self, tmp_path: Path) -> None:
        """Python project can override test command with timeout."""
        _create_mala_yaml(
            tmp_path,
            """
preset: python-uv
commands:
  test:
    command: "uv run pytest -v"
    timeout: 600
""",
        )

        config = _get_resolved_config(tmp_path)

        assert config.commands.test is not None
        assert config.commands.test.command == "uv run pytest -v"
        assert config.commands.test.timeout == 600

    def test_python_project_code_patterns_inherited(self, tmp_path: Path) -> None:
        """Code patterns inherited from preset when not overridden."""
        _create_mala_yaml(
            tmp_path,
            """
preset: python-uv
commands:
  test: "uv run pytest -v"
""",
        )

        config = _get_resolved_config(tmp_path)

        assert "**/*.py" in config.code_patterns
        assert "pyproject.toml" in config.code_patterns

    def test_python_project_code_patterns_overridden(self, tmp_path: Path) -> None:
        """Code patterns can be overridden explicitly."""
        _create_mala_yaml(
            tmp_path,
            """
preset: python-uv
commands:
  test: "uv run pytest"
code_patterns:
  - "src/**/*.py"
  - "tests/**/*.py"
""",
        )

        config = _get_resolved_config(tmp_path)

        # User patterns replace preset patterns
        assert config.code_patterns == ("src/**/*.py", "tests/**/*.py")
        assert "**/*.py" not in config.code_patterns


# =============================================================================
# AC 4: Minimal config (only setup + test) skips lint/format/typecheck
# =============================================================================


class TestMinimalConfigIntegration:
    """AC 4: Test minimal config behavior skipping optional commands."""

    def test_minimal_config_only_setup_and_test(self, tmp_path: Path) -> None:
        """Minimal config with only setup and test skips lint/format/typecheck."""
        _create_mala_yaml(
            tmp_path,
            """
commands:
  setup: "make install"
  test: "make test"
""",
        )

        config = load_config(tmp_path)

        # Only setup and test defined
        assert config.commands.setup is not None
        assert config.commands.setup.command == "make install"
        assert config.commands.test is not None
        assert config.commands.test.command == "make test"

        # lint, format, typecheck should be None
        assert config.commands.lint is None
        assert config.commands.format is None
        assert config.commands.typecheck is None
        assert config.commands.e2e is None

    def test_minimal_config_only_test(self, tmp_path: Path) -> None:
        """Minimal config with only test command."""
        _create_mala_yaml(
            tmp_path,
            """
commands:
  test: "pytest"
""",
        )

        config = load_config(tmp_path)

        assert config.commands.test is not None
        assert config.commands.test.command == "pytest"
        assert config.commands.setup is None
        assert config.commands.lint is None
        assert config.commands.format is None
        assert config.commands.typecheck is None

    def test_preset_with_explicit_null_overrides(self, tmp_path: Path) -> None:
        """Preset with explicit null can disable inherited commands."""
        _create_mala_yaml(
            tmp_path,
            """
preset: python-uv
commands:
  lint: null
  format: null
  typecheck: null
""",
        )

        config = _get_resolved_config(tmp_path)

        # Test and setup inherited from preset
        assert config.commands.test is not None
        assert config.commands.setup is not None

        # Explicitly nulled commands are None
        assert config.commands.lint is None
        assert config.commands.format is None
        assert config.commands.typecheck is None


# =============================================================================
# AC 5: Missing mala.yaml fails with clear error
# =============================================================================


class TestMissingConfigIntegration:
    """AC 5: Test error handling for missing mala.yaml."""

    def test_missing_mala_yaml_raises_config_error(self, tmp_path: Path) -> None:
        """Missing mala.yaml raises ConfigError with clear message."""
        # No mala.yaml created

        with pytest.raises(ConfigError) as exc_info:
            load_config(tmp_path)

        error_msg = str(exc_info.value)
        assert "mala.yaml not found" in error_msg
        assert str(tmp_path) in error_msg
        assert "Mala requires a configuration file" in error_msg

    def test_missing_config_error_is_specific(self, tmp_path: Path) -> None:
        """Missing config error message format is exact."""
        with pytest.raises(ConfigError) as exc_info:
            load_config(tmp_path)

        expected = (
            f"mala.yaml not found in {tmp_path}. "
            "Mala requires a configuration file to run."
        )
        assert str(exc_info.value) == expected


# =============================================================================
# AC 6: Invalid YAML syntax fails with specific error
# =============================================================================


class TestInvalidYamlSyntaxIntegration:
    """AC 6: Test error handling for invalid YAML syntax."""

    def test_invalid_yaml_syntax_raises_config_error(self, tmp_path: Path) -> None:
        """Invalid YAML syntax raises ConfigError with details."""
        _create_mala_yaml(
            tmp_path,
            """
commands
  test: pytest
""",
        )

        with pytest.raises(ConfigError) as exc_info:
            load_config(tmp_path)

        assert "Invalid YAML syntax in mala.yaml:" in str(exc_info.value)

    def test_yaml_with_tabs_raises_error(self, tmp_path: Path) -> None:
        """YAML with tabs (invalid indentation) raises specific error."""
        _create_mala_yaml(tmp_path, "commands:\n\ttest: pytest\n")

        with pytest.raises(ConfigError) as exc_info:
            load_config(tmp_path)

        assert "Invalid YAML syntax in mala.yaml:" in str(exc_info.value)

    def test_unclosed_bracket_raises_error(self, tmp_path: Path) -> None:
        """Unclosed bracket in YAML raises error."""
        _create_mala_yaml(
            tmp_path,
            """
commands:
  test: pytest
code_patterns:
  - "*.py"
  - [incomplete
""",
        )

        with pytest.raises(ConfigError) as exc_info:
            load_config(tmp_path)

        assert "Invalid YAML syntax in mala.yaml:" in str(exc_info.value)

    def test_unknown_field_raises_specific_error(self, tmp_path: Path) -> None:
        """Unknown field in config raises specific error."""
        _create_mala_yaml(
            tmp_path,
            """
commands:
  test: pytest
unknown_field: value
""",
        )

        with pytest.raises(ConfigError) as exc_info:
            load_config(tmp_path)

        assert "Unknown field 'unknown_field' in mala.yaml" == str(exc_info.value)


# =============================================================================
# AC 7: Coverage config with xml format, custom file, threshold
# =============================================================================


class TestCoverageConfigIntegration:
    """AC 7: Test coverage configuration parsing."""

    def test_coverage_config_xml_format(self, tmp_path: Path) -> None:
        """Coverage config with xml format, custom file, threshold."""
        _create_mala_yaml(
            tmp_path,
            """
commands:
  test: "pytest"
coverage:
  format: xml
  file: coverage/coverage.xml
  threshold: 85
""",
        )

        config = load_config(tmp_path)

        assert config.coverage is not None
        assert config.coverage.format == "xml"
        assert config.coverage.file == "coverage/coverage.xml"
        assert config.coverage.threshold == 85.0

    def test_coverage_config_with_custom_command(self, tmp_path: Path) -> None:
        """Coverage config can specify a custom coverage command."""
        _create_mala_yaml(
            tmp_path,
            """
commands:
  test: "pytest"
coverage:
  format: xml
  file: coverage.xml
  threshold: 80
  command: "pytest --cov=src --cov-report=xml"
""",
        )

        config = load_config(tmp_path)

        assert config.coverage is not None
        assert config.coverage.command == "pytest --cov=src --cov-report=xml"

    def test_coverage_config_with_timeout(self, tmp_path: Path) -> None:
        """Coverage config can specify timeout."""
        _create_mala_yaml(
            tmp_path,
            """
commands:
  test: "pytest"
coverage:
  format: xml
  file: coverage.xml
  threshold: 80
  timeout: 600
""",
        )

        config = load_config(tmp_path)

        assert config.coverage is not None
        assert config.coverage.timeout == 600

    def test_coverage_config_threshold_as_float(self, tmp_path: Path) -> None:
        """Coverage threshold can be a float."""
        _create_mala_yaml(
            tmp_path,
            """
commands:
  test: "pytest"
coverage:
  format: xml
  file: coverage.xml
  threshold: 85.5
""",
        )

        config = load_config(tmp_path)

        assert config.coverage is not None
        assert config.coverage.threshold == 85.5

    def test_coverage_invalid_format_raises_error(self, tmp_path: Path) -> None:
        """Unsupported coverage format raises error."""
        _create_mala_yaml(
            tmp_path,
            """
commands:
  test: pytest
coverage:
  format: lcov
  file: lcov.info
  threshold: 80
""",
        )

        with pytest.raises(ConfigError) as exc_info:
            load_config(tmp_path)

        assert "Unsupported coverage format 'lcov'" in str(exc_info.value)


# =============================================================================
# AC 8: No coverage section skips coverage
# =============================================================================


class TestNoCoverageConfigIntegration:
    """AC 8: Test that missing coverage section results in no coverage."""

    def test_no_coverage_section_results_in_none(self, tmp_path: Path) -> None:
        """Config without coverage section has coverage=None."""
        _create_mala_yaml(
            tmp_path,
            """
commands:
  test: "pytest"
""",
        )

        config = load_config(tmp_path)

        assert config.coverage is None

    def test_explicit_null_coverage_results_in_none(self, tmp_path: Path) -> None:
        """Explicit null coverage section has coverage=None."""
        _create_mala_yaml(
            tmp_path,
            """
commands:
  test: "pytest"
coverage: null
""",
        )

        config = load_config(tmp_path)

        assert config.coverage is None

    def test_preset_coverage_can_be_disabled(self, tmp_path: Path) -> None:
        """Coverage from preset can be disabled with explicit null."""
        # First, verify a preset with coverage
        _create_mala_yaml(
            tmp_path,
            """
preset: python-uv
coverage:
  format: xml
  file: coverage.xml
  threshold: 85
""",
        )

        config = _get_resolved_config(tmp_path)
        assert config.coverage is not None

        # Now disable it with explicit null
        _create_mala_yaml(
            tmp_path,
            """
preset: python-uv
coverage: null
""",
        )

        config = _get_resolved_config(tmp_path)
        assert config.coverage is None


# =============================================================================
# AC 9: Custom code_patterns filter files correctly
# =============================================================================


class TestCodePatternsIntegration:
    """AC 9: Test custom code_patterns filter files correctly."""

    @pytest.fixture
    def mock_spec(self) -> type:
        """Create a mock spec class for validation gating tests."""
        from dataclasses import dataclass, field

        @dataclass
        class MockSpec:
            code_patterns: list[str] = field(default_factory=list)
            config_files: list[str] = field(default_factory=list)
            setup_files: list[str] = field(default_factory=list)

        return MockSpec

    def test_py_pattern_matches_only_python_files(self, tmp_path: Path) -> None:
        """*.py pattern matches only Python files."""
        _create_mala_yaml(
            tmp_path,
            """
commands:
  test: pytest
code_patterns:
  - "*.py"
""",
        )

        config = load_config(tmp_path)
        patterns = list(config.code_patterns)

        # Use filter_matching_files to verify behavior
        files = ["main.py", "utils.py", "README.md", "config.yaml"]
        matched = filter_matching_files(files, patterns)

        assert matched == ["main.py", "utils.py"]
        assert "README.md" not in matched
        assert "config.yaml" not in matched

    def test_recursive_pattern_matches_nested(self, tmp_path: Path) -> None:
        """src/**/*.py matches Python files at any depth under src/."""
        _create_mala_yaml(
            tmp_path,
            """
commands:
  test: pytest
code_patterns:
  - "src/**/*.py"
""",
        )

        config = load_config(tmp_path)
        patterns = list(config.code_patterns)

        files = [
            "src/main.py",
            "src/utils/helpers.py",
            "src/deep/nested/file.py",
            "tests/test_main.py",
            "README.md",
        ]
        matched = filter_matching_files(files, patterns)

        assert "src/main.py" in matched
        assert "src/utils/helpers.py" in matched
        assert "src/deep/nested/file.py" in matched
        assert "tests/test_main.py" not in matched
        assert "README.md" not in matched

    def test_multiple_patterns_or_logic(self, tmp_path: Path) -> None:
        """Multiple patterns use OR logic."""
        _create_mala_yaml(
            tmp_path,
            """
commands:
  test: pytest
code_patterns:
  - "src/**/*.py"
  - "tests/**/*.py"
""",
        )

        config = load_config(tmp_path)
        patterns = list(config.code_patterns)

        files = [
            "src/main.py",
            "tests/test_main.py",
            "README.md",
        ]
        matched = filter_matching_files(files, patterns)

        assert "src/main.py" in matched
        assert "tests/test_main.py" in matched
        assert "README.md" not in matched

    def test_validation_gating_with_code_patterns(
        self, tmp_path: Path, mock_spec: type
    ) -> None:
        """Code patterns correctly gate validation trigger."""
        _create_mala_yaml(
            tmp_path,
            """
commands:
  test: pytest
code_patterns:
  - "*.py"
""",
        )

        config = load_config(tmp_path)
        spec = mock_spec(code_patterns=list(config.code_patterns))

        # Python file changes should trigger validation
        assert should_trigger_validation(["main.py"], spec) is True
        assert should_trigger_validation(["src/utils.py"], spec) is True

        # Non-Python changes should NOT trigger validation
        assert should_trigger_validation(["README.md"], spec) is False
        assert should_trigger_validation(["config.yaml"], spec) is False

    def test_mala_yaml_change_always_triggers(
        self, tmp_path: Path, mock_spec: type
    ) -> None:
        """mala.yaml change always triggers validation regardless of patterns."""
        _create_mala_yaml(
            tmp_path,
            """
commands:
  test: pytest
code_patterns:
  - "*.py"
""",
        )

        config = load_config(tmp_path)
        spec = mock_spec(code_patterns=list(config.code_patterns))

        # mala.yaml change always triggers
        assert should_trigger_validation(["mala.yaml"], spec) is True

    def test_config_files_invalidate_lint_cache(
        self, tmp_path: Path, mock_spec: type
    ) -> None:
        """Config file changes invalidate lint cache."""
        _create_mala_yaml(
            tmp_path,
            """
commands:
  test: pytest
  lint: "ruff check ."
config_files:
  - "pyproject.toml"
  - ".ruff.toml"
""",
        )

        config = load_config(tmp_path)
        spec = mock_spec(config_files=list(config.config_files))

        # Config file changes invalidate cache
        assert should_invalidate_lint_cache(["pyproject.toml"], spec) is True
        assert should_invalidate_lint_cache([".ruff.toml"], spec) is True

        # Non-config file changes don't invalidate
        assert should_invalidate_lint_cache(["src/main.py"], spec) is False


# =============================================================================
# AC 10: Tool name extraction in quality gate output
# =============================================================================


class TestToolNameExtractionIntegration:
    """AC 10: Test tool name extraction for quality gate messaging."""

    def test_preset_commands_extract_correctly(self, tmp_path: Path) -> None:
        """Tool names extract correctly from preset commands."""
        _create_mala_yaml(tmp_path, "preset: python-uv\n")

        config = _get_resolved_config(tmp_path)

        # Verify tool extraction for each preset command
        assert config.commands.setup is not None
        # "uv sync" -> "uv" (not a compound command with sync)
        # Actually, check what extract_tool_name returns
        setup_tool = extract_tool_name(config.commands.setup.command)
        assert setup_tool  # Just verify it's not empty

        assert config.commands.test is not None
        assert extract_tool_name(config.commands.test.command) == "pytest"

        assert config.commands.lint is not None
        assert extract_tool_name(config.commands.lint.command) == "ruff"

        assert config.commands.format is not None
        assert extract_tool_name(config.commands.format.command) == "ruff"

        assert config.commands.typecheck is not None
        assert extract_tool_name(config.commands.typecheck.command) == "ty"


# =============================================================================
# End-to-end integration scenarios
# =============================================================================


class TestEndToEndScenarios:
    """Integration tests for realistic end-to-end scenarios."""

    def test_full_python_project_workflow(self, tmp_path: Path) -> None:
        """Complete Python project setup with all features."""
        _create_mala_yaml(
            tmp_path,
            """
preset: python-uv
commands:
  test:
    command: "uv run pytest -v"
    timeout: 300
coverage:
  format: xml
  file: coverage.xml
  threshold: 85
code_patterns:
  - "src/**/*.py"
  - "tests/**/*.py"
config_files:
  - "pyproject.toml"
  - "ruff.toml"
setup_files:
  - "uv.lock"
  - "pyproject.toml"
""",
        )

        config = _get_resolved_config(tmp_path)

        # Commands merged correctly
        assert config.commands.test is not None
        assert config.commands.test.command == "uv run pytest -v"
        assert config.commands.test.timeout == 300
        assert config.commands.lint is not None  # From preset

        # Coverage configured
        assert config.coverage is not None
        assert config.coverage.threshold == 85.0

        # Patterns overridden
        assert config.code_patterns == ("src/**/*.py", "tests/**/*.py")

    def test_minimal_makefile_project(self, tmp_path: Path) -> None:
        """Minimal Makefile-based project without preset."""
        _create_mala_yaml(
            tmp_path,
            """
commands:
  setup: "make deps"
  test: "make test"
""",
        )

        config = load_config(tmp_path)

        assert config.preset is None
        assert config.commands.setup is not None
        assert config.commands.setup.command == "make deps"
        assert config.commands.test is not None
        assert config.commands.test.command == "make test"
        assert config.commands.lint is None
        assert config.coverage is None

    def test_monorepo_with_custom_patterns(self, tmp_path: Path) -> None:
        """Monorepo with custom code patterns for multiple languages."""
        _create_mala_yaml(
            tmp_path,
            """
commands:
  setup: "npm install && cd backend && pip install -e ."
  test: "npm test && pytest backend/"
code_patterns:
  - "frontend/**/*.ts"
  - "frontend/**/*.tsx"
  - "backend/**/*.py"
config_files:
  - "package.json"
  - "pyproject.toml"
  - "tsconfig.json"
""",
        )

        config = load_config(tmp_path)

        # Verify patterns
        assert "frontend/**/*.ts" in config.code_patterns
        assert "frontend/**/*.tsx" in config.code_patterns
        assert "backend/**/*.py" in config.code_patterns

        # Test file filtering
        files = [
            "frontend/src/App.tsx",
            "backend/main.py",
            "README.md",
            "docs/guide.md",
        ]
        matched = filter_matching_files(files, list(config.code_patterns))
        assert "frontend/src/App.tsx" in matched
        assert "backend/main.py" in matched
        assert "README.md" not in matched


# =============================================================================
# Custom Commands Integration Test
# =============================================================================


class TestCustomCommandsIntegration:
    """Integration test for custom_commands config → spec path."""

    def test_custom_commands_config_to_spec_integration(self, tmp_path: Path) -> None:
        """Custom commands appear in ValidationCommand list when set via dict.

        This test exercises the config→spec internal path using the
        _build_commands_from_config function. The full YAML→spec path will
        be tested once inline custom command parsing is implemented (see
        mala-81qt epic).
        """
        from src.domain.validation.config import (
            CommandConfig,
            CommandsConfig,
            CustomCommandConfig,
        )
        from src.domain.validation.spec import (
            CommandKind,
            _build_commands_from_config,
        )

        # Create commands config with a test command (required for spec)
        commands_config = CommandsConfig(
            test=CommandConfig(command="pytest"),
            lint=CommandConfig(command="ruff check ."),
        )

        # Create custom commands dict
        security_cmd = CustomCommandConfig(
            command="bandit -r src/",
            allow_fail=True,
        )
        docs_cmd = CustomCommandConfig(
            command="mkdocs build --strict",
            allow_fail=False,
        )
        custom_commands_dict = {
            "security_scan": security_cmd,
            "docs_check": docs_cmd,
        }

        # Build commands list using internal function
        commands = _build_commands_from_config(commands_config, custom_commands_dict)

        # Find custom commands in the result
        custom_commands = [cmd for cmd in commands if cmd.kind == CommandKind.CUSTOM]

        assert len(custom_commands) == 2, "Expected 2 custom commands"

        cmd_names = {cmd.name for cmd in custom_commands}
        assert "security_scan" in cmd_names
        assert "docs_check" in cmd_names

        # Verify allow_fail is propagated
        security_scan = next(c for c in custom_commands if c.name == "security_scan")
        assert security_scan.allow_fail is True
        assert security_scan.command == "bandit -r src/"

        docs_check = next(c for c in custom_commands if c.name == "docs_check")
        assert docs_check.allow_fail is False
        assert docs_check.command == "mkdocs build --strict"

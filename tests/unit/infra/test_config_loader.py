"""Tests for config_loader module.

Tests the YAML configuration loading functionality including:
- Successful loading with valid configurations
- Missing file error handling
- Invalid YAML syntax error handling
- Unknown field rejection
- Type validation
- Semantic validation (no commands defined, empty command, unsupported coverage format)
"""

from pathlib import Path
from typing import Any

import pytest

from src.domain.validation.config import ConfigError
from src.domain.validation.config_loader import (
    ConfigMissingError,
    _build_config,
    _parse_yaml,
    _validate_config,
    _validate_schema,
    load_config,
)


# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures" / "mala-configs"


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_minimal(self, tmp_path: Path) -> None:
        """Load minimal valid config with just a test command."""
        config_file = tmp_path / "mala.yaml"
        config_file.write_text("commands:\n  test: pytest\n")

        config = load_config(tmp_path)

        assert config.commands.test is not None
        assert config.commands.test.command == "pytest"
        assert config.preset is None

    def test_load_valid_full(self, tmp_path: Path) -> None:
        """Load full valid config with all options."""
        config_content = """
preset: python-uv
commands:
  setup: uv sync
  test:
    command: uv run pytest
    timeout: 300
  lint: uvx ruff check .
coverage:
  format: xml
  file: coverage.xml
  threshold: 85
code_patterns:
  - "**/*.py"
config_files:
  - pyproject.toml
setup_files:
  - uv.lock
"""
        config_file = tmp_path / "mala.yaml"
        config_file.write_text(config_content)

        config = load_config(tmp_path)

        assert config.preset == "python-uv"
        assert config.commands.setup is not None
        assert config.commands.setup.command == "uv sync"
        assert config.commands.test is not None
        assert config.commands.test.command == "uv run pytest"
        assert config.commands.test.timeout == 300
        assert config.commands.lint is not None
        assert config.commands.lint.command == "uvx ruff check ."
        assert config.coverage is not None
        assert config.coverage.format == "xml"
        assert config.coverage.file == "coverage.xml"
        assert config.coverage.threshold == 85.0
        assert config.code_patterns == ("**/*.py",)
        assert config.config_files == ("pyproject.toml",)
        assert config.setup_files == ("uv.lock",)

    def test_load_preset_only(self, tmp_path: Path) -> None:
        """Load config with only a preset (valid because preset may define commands)."""
        config_file = tmp_path / "mala.yaml"
        config_file.write_text("preset: python-uv\n")

        config = load_config(tmp_path)

        assert config.preset == "python-uv"
        assert config.commands.test is None

    def test_missing_file_raises_config_missing_error(self, tmp_path: Path) -> None:
        """Missing mala.yaml raises ConfigMissingError with exact message."""
        with pytest.raises(ConfigMissingError) as exc_info:
            load_config(tmp_path)

        expected = f"mala.yaml not found in {tmp_path}. Mala requires a configuration file to run."
        assert str(exc_info.value) == expected
        # Also verify it's a subclass of ConfigError for callers catching broader errors
        assert isinstance(exc_info.value, ConfigError)
        assert exc_info.value.repo_path == tmp_path

    def test_missing_file_catchable_as_config_error(self, tmp_path: Path) -> None:
        """ConfigMissingError is catchable as ConfigError for backward compatibility."""
        # This test ensures that callers catching ConfigError will still work
        with pytest.raises(ConfigError):
            load_config(tmp_path)

    def test_invalid_yaml_syntax_error(self, tmp_path: Path) -> None:
        """Invalid YAML syntax raises ConfigError with details."""
        config_file = tmp_path / "mala.yaml"
        config_file.write_text("commands\n  test: pytest\n")

        with pytest.raises(ConfigError) as exc_info:
            load_config(tmp_path)

        assert "Invalid YAML syntax in mala.yaml:" in str(exc_info.value)

    def test_unknown_field_error(self, tmp_path: Path) -> None:
        """Unknown top-level field raises ConfigError."""
        config_file = tmp_path / "mala.yaml"
        config_file.write_text("commands:\n  test: pytest\nunknown_field: value\n")

        with pytest.raises(ConfigError) as exc_info:
            load_config(tmp_path)

        assert "Unknown field 'unknown_field' in mala.yaml" == str(exc_info.value)

    def test_invalid_preset_type_error(self, tmp_path: Path) -> None:
        """Non-string preset raises ConfigError."""
        config_file = tmp_path / "mala.yaml"
        config_file.write_text("preset: 123\ncommands:\n  test: pytest\n")

        with pytest.raises(ConfigError) as exc_info:
            load_config(tmp_path)

        assert "preset must be a string, got int" in str(exc_info.value)

    def test_no_commands_defined_error(self, tmp_path: Path) -> None:
        """No commands and no preset raises ConfigError."""
        config_file = tmp_path / "mala.yaml"
        config_file.write_text("code_patterns:\n  - '*.py'\n")

        with pytest.raises(ConfigError) as exc_info:
            load_config(tmp_path)

        expected = "At least one command must be defined. Specify a preset or define commands directly."
        assert str(exc_info.value) == expected

    def test_global_validation_commands_unknown_field(self, tmp_path: Path) -> None:
        """global_validation_commands is rejected with migration message."""
        config_file = tmp_path / "mala.yaml"
        config_file.write_text("global_validation_commands:\n  test: pytest\n")

        with pytest.raises(ConfigError) as exc_info:
            load_config(tmp_path)

        assert "global_validation_commands is not supported" in str(exc_info.value)

    def test_empty_command_string_error(self, tmp_path: Path) -> None:
        """Empty command string raises ConfigError."""
        config_file = tmp_path / "mala.yaml"
        config_file.write_text('commands:\n  test: ""\n')

        with pytest.raises(ConfigError) as exc_info:
            load_config(tmp_path)

        assert "cannot be empty string" in str(exc_info.value)

    def test_unsupported_coverage_format_error(self, tmp_path: Path) -> None:
        """Unsupported coverage format raises ConfigError."""
        config_content = """
commands:
  test: pytest
coverage:
  format: lcov
  file: lcov.info
  threshold: 80
"""
        config_file = tmp_path / "mala.yaml"
        config_file.write_text(config_content)

        with pytest.raises(ConfigError) as exc_info:
            load_config(tmp_path)

        assert "Unsupported coverage format 'lcov'. Supported formats: xml" in str(
            exc_info.value
        )

    def test_empty_file_no_preset_error(self, tmp_path: Path) -> None:
        """Empty file with no commands or preset raises ConfigError."""
        config_file = tmp_path / "mala.yaml"
        config_file.write_text("")

        with pytest.raises(ConfigError) as exc_info:
            load_config(tmp_path)

        assert "At least one command must be defined" in str(exc_info.value)

    def test_file_with_only_comments_error(self, tmp_path: Path) -> None:
        """File with only comments (effectively empty) raises ConfigError."""
        config_file = tmp_path / "mala.yaml"
        config_file.write_text("# Just a comment\n# Another comment\n")

        with pytest.raises(ConfigError) as exc_info:
            load_config(tmp_path)

        assert "At least one command must be defined" in str(exc_info.value)

    def test_oserror_wrapped_as_config_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OSError during file read is wrapped as ConfigError."""
        config_file = tmp_path / "mala.yaml"
        config_file.write_text("commands:\n  test: pytest\n")

        def raise_permission_error(*args: object, **kwargs: object) -> str:
            raise PermissionError("Permission denied")

        monkeypatch.setattr(Path, "read_text", raise_permission_error)

        with pytest.raises(ConfigError) as exc_info:
            load_config(tmp_path)

        assert "Failed to read" in str(exc_info.value)
        assert "Permission denied" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, PermissionError)

    def test_unicode_decode_error_wrapped_as_config_error(self, tmp_path: Path) -> None:
        """UnicodeDecodeError during file read is wrapped as ConfigError."""
        config_file = tmp_path / "mala.yaml"
        # Write invalid UTF-8 bytes directly
        config_file.write_bytes(b"\xff\xfe invalid utf-8")

        with pytest.raises(ConfigError) as exc_info:
            load_config(tmp_path)

        assert "Failed to decode" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, UnicodeDecodeError)


class TestParseYaml:
    """Tests for _parse_yaml function."""

    def test_parse_valid_yaml(self) -> None:
        """Parse valid YAML content."""
        content = "key: value\nnested:\n  inner: 42\n"
        result = _parse_yaml(content)

        assert result == {"key": "value", "nested": {"inner": 42}}

    def test_parse_empty_content(self) -> None:
        """Parse empty content returns empty dict."""
        result = _parse_yaml("")
        assert result == {}

    def test_parse_null_content(self) -> None:
        """Parse 'null' or '~' returns empty dict."""
        result = _parse_yaml("null")
        assert result == {}

        result = _parse_yaml("~")
        assert result == {}

    def test_parse_comments_only(self) -> None:
        """Parse content with only comments returns empty dict."""
        result = _parse_yaml("# Just a comment\n")
        assert result == {}

    def test_parse_invalid_syntax(self) -> None:
        """Parse invalid YAML raises ConfigError."""
        with pytest.raises(ConfigError) as exc_info:
            _parse_yaml("key: [unbalanced")

        assert "Invalid YAML syntax in mala.yaml:" in str(exc_info.value)

    def test_parse_non_dict_raises_error(self) -> None:
        """Parse YAML that's not a mapping raises ConfigError."""
        with pytest.raises(ConfigError) as exc_info:
            _parse_yaml("- item1\n- item2\n")

        assert "mala.yaml must be a YAML mapping, got list" in str(exc_info.value)

    def test_parse_scalar_raises_error(self) -> None:
        """Parse YAML that's a scalar raises ConfigError."""
        with pytest.raises(ConfigError) as exc_info:
            _parse_yaml("just a string")

        assert "mala.yaml must be a YAML mapping, got str" in str(exc_info.value)


class TestValidateSchema:
    """Tests for _validate_schema function."""

    def test_valid_schema_all_fields(self) -> None:
        """All known top-level fields are accepted."""
        data = {
            "preset": "python-uv",
            "commands": {"test": "pytest"},
            "coverage": {"format": "xml", "file": "cov.xml", "threshold": 80},
            "code_patterns": ["*.py"],
            "config_files": ["pyproject.toml"],
            "setup_files": ["uv.lock"],
            "validation_triggers": {
                "session_end": {
                    "failure_mode": "continue",
                    "commands": [{"ref": "test"}],
                }
            },
            "claude_settings_sources": ["local", "project"],
        }
        # Should not raise
        _validate_schema(data)

    def test_valid_schema_subset(self) -> None:
        """Subset of fields is valid."""
        data = {"commands": {"test": "pytest"}}
        _validate_schema(data)

    def test_valid_schema_empty(self) -> None:
        """Empty dict is valid at schema level."""
        _validate_schema({})

    def test_unknown_field_raises_error(self) -> None:
        """Unknown field raises ConfigError."""
        data = {"commands": {"test": "pytest"}, "unknown": "value"}

        with pytest.raises(ConfigError) as exc_info:
            _validate_schema(data)

        assert "Unknown field 'unknown' in mala.yaml" == str(exc_info.value)

    def test_multiple_unknown_fields_reports_first(self) -> None:
        """Multiple unknown fields report the first one (sorted)."""
        data = {"zebra": 1, "alpha": 2, "commands": {"test": "pytest"}}

        with pytest.raises(ConfigError) as exc_info:
            _validate_schema(data)

        # Should report 'alpha' first (sorted alphabetically)
        assert "Unknown field 'alpha' in mala.yaml" == str(exc_info.value)

    def test_non_string_keys_do_not_cause_type_error(self) -> None:
        """Non-string top-level keys (e.g., null, integers) raise ConfigError, not TypeError."""
        # YAML allows non-string keys like null and integers
        # Cast to dict[str, Any] to match function signature - this simulates
        # what yaml.safe_load returns for malformed YAML with non-string keys
        data: dict[str, Any] = {None: "foo", 1: "bar", "commands": {"test": "pytest"}}  # type: ignore[dict-item]

        with pytest.raises(ConfigError) as exc_info:
            _validate_schema(data)

        # Should raise ConfigError (not TypeError) with consistent message
        assert "Unknown field" in str(exc_info.value)
        assert "in mala.yaml" in str(exc_info.value)


class TestRemovedConfigFieldsRejected:
    """Regression tests for removed config fields."""

    def test_context_restart_threshold_rejected(self) -> None:
        """Removed context_restart_threshold raises ConfigError."""
        data = {"commands": {"test": "pytest"}, "context_restart_threshold": 0.85}

        with pytest.raises(ConfigError) as exc_info:
            _validate_schema(data)

        assert "Unknown field 'context_restart_threshold' in mala.yaml" == str(
            exc_info.value
        )

    def test_context_limit_rejected(self) -> None:
        """Removed context_limit raises ConfigError."""
        data = {"commands": {"test": "pytest"}, "context_limit": 200000}

        with pytest.raises(ConfigError) as exc_info:
            _validate_schema(data)

        assert "Unknown field 'context_limit' in mala.yaml" == str(exc_info.value)


class TestBuildConfig:
    """Tests for _build_config function."""

    def test_build_minimal_config(self) -> None:
        """Build config from minimal valid data."""
        data = {"commands": {"test": "pytest"}}
        config = _build_config(data)

        assert config.commands.test is not None
        assert config.commands.test.command == "pytest"

    def test_build_full_config(self) -> None:
        """Build config from full data."""
        data = {
            "preset": "go",
            "commands": {
                "setup": "go mod download",
                "test": {"command": "go test ./...", "timeout": 120},
            },
            "code_patterns": ["*.go", "go.mod"],
        }
        config = _build_config(data)

        assert config.preset == "go"
        assert config.commands.setup is not None
        assert config.commands.setup.command == "go mod download"
        assert config.commands.test is not None
        assert config.commands.test.command == "go test ./..."
        assert config.commands.test.timeout == 120
        assert config.code_patterns == ("*.go", "go.mod")

    def test_build_empty_config(self) -> None:
        """Build config from empty dict."""
        config = _build_config({})

        assert config.preset is None
        assert config.commands.test is None
        assert config.coverage is None
        assert config.code_patterns == ()


class TestValidateConfig:
    """Tests for _validate_config function."""

    def test_valid_with_command(self) -> None:
        """Config with at least one command is valid."""
        data = {"commands": {"test": "pytest"}}
        config = _build_config(data)

        # Should not raise
        _validate_config(config)

    def test_valid_with_preset(self) -> None:
        """Config with preset (no inline commands) is valid."""
        data = {"preset": "python-uv"}
        config = _build_config(data)

        # Should not raise - preset may define commands
        _validate_config(config)

    def test_invalid_no_commands_no_preset(self) -> None:
        """Config with no commands and no preset raises ConfigError."""
        data = {"code_patterns": ["*.py"]}
        config = _build_config(data)

        with pytest.raises(ConfigError) as exc_info:
            _validate_config(config)

        expected = "At least one command must be defined. Specify a preset or define commands directly."
        assert str(exc_info.value) == expected

    def test_invalid_empty_config(self) -> None:
        """Empty config raises ConfigError."""
        config = _build_config({})

        with pytest.raises(ConfigError) as exc_info:
            _validate_config(config)

        assert "At least one command must be defined" in str(exc_info.value)


class TestLoadConfigWithFixtures:
    """Tests using fixture files."""

    def test_fixture_valid_minimal(self) -> None:
        """Load valid-minimal.yaml fixture."""
        # Copy fixture to temp location as mala.yaml
        import shutil

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            shutil.copy(FIXTURES_DIR / "valid-minimal.yaml", tmp_path / "mala.yaml")

            config = load_config(tmp_path)

            assert config.commands.test is not None
            assert config.commands.test.command == "pytest"

    def test_fixture_valid_full(self) -> None:
        """Load valid-full.yaml fixture."""
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            shutil.copy(FIXTURES_DIR / "valid-full.yaml", tmp_path / "mala.yaml")

            config = load_config(tmp_path)

            assert config.preset == "python-uv"
            assert config.commands.test is not None
            assert config.commands.test.command == "uv run pytest"
            assert config.commands.test.timeout == 300
            assert config.coverage is not None
            assert config.coverage.threshold == 85.0
            assert len(config.code_patterns) == 2

    def test_fixture_valid_preset_only(self) -> None:
        """Load valid-preset-only.yaml fixture."""
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            shutil.copy(FIXTURES_DIR / "valid-preset-only.yaml", tmp_path / "mala.yaml")

            config = load_config(tmp_path)

            assert config.preset == "python-uv"

    def test_fixture_invalid_syntax(self) -> None:
        """Load invalid-syntax.yaml fixture fails with syntax error."""
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            shutil.copy(FIXTURES_DIR / "invalid-syntax.yaml", tmp_path / "mala.yaml")

            with pytest.raises(ConfigError) as exc_info:
                load_config(tmp_path)

            assert "Invalid YAML syntax in mala.yaml:" in str(exc_info.value)

    def test_fixture_invalid_unknown_field(self) -> None:
        """Load invalid-unknown-field.yaml fixture fails with unknown field error."""
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            shutil.copy(
                FIXTURES_DIR / "invalid-unknown-field.yaml", tmp_path / "mala.yaml"
            )

            with pytest.raises(ConfigError) as exc_info:
                load_config(tmp_path)

            assert "Unknown field 'unknown_field' in mala.yaml" == str(exc_info.value)

    def test_fixture_invalid_no_commands(self) -> None:
        """Load invalid-no-commands.yaml fixture fails with no commands error."""
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            shutil.copy(
                FIXTURES_DIR / "invalid-no-commands.yaml", tmp_path / "mala.yaml"
            )

            with pytest.raises(ConfigError) as exc_info:
                load_config(tmp_path)

            assert "At least one command must be defined" in str(exc_info.value)

    def test_fixture_invalid_empty_command(self) -> None:
        """Load invalid-empty-command.yaml fixture fails with empty command error."""
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            shutil.copy(
                FIXTURES_DIR / "invalid-empty-command.yaml", tmp_path / "mala.yaml"
            )

            with pytest.raises(ConfigError) as exc_info:
                load_config(tmp_path)

            assert "cannot be empty string" in str(exc_info.value)

    def test_fixture_invalid_coverage_format(self) -> None:
        """Load invalid-coverage-format.yaml fixture fails with format error."""
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            shutil.copy(
                FIXTURES_DIR / "invalid-coverage-format.yaml", tmp_path / "mala.yaml"
            )

            with pytest.raises(ConfigError) as exc_info:
                load_config(tmp_path)

            assert "Unsupported coverage format 'lcov'" in str(exc_info.value)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_yaml_with_tabs(self, tmp_path: Path) -> None:
        """YAML with tabs instead of spaces raises syntax error."""
        config_file = tmp_path / "mala.yaml"
        config_file.write_text("commands:\n\ttest: pytest\n")

        with pytest.raises(ConfigError) as exc_info:
            load_config(tmp_path)

        assert "Invalid YAML syntax in mala.yaml:" in str(exc_info.value)

    def test_unicode_in_config(self, tmp_path: Path) -> None:
        """Unicode characters in config are handled correctly."""
        config_file = tmp_path / "mala.yaml"
        config_file.write_text("commands:\n  test: pytest # 日本語コメント\n")

        config = load_config(tmp_path)

        assert config.commands.test is not None

    def test_very_long_command(self, tmp_path: Path) -> None:
        """Very long command string is accepted."""
        long_command = "pytest " + " ".join([f"--option{i}" for i in range(100)])
        config_file = tmp_path / "mala.yaml"
        config_file.write_text(f"commands:\n  test: {long_command}\n")

        config = load_config(tmp_path)

        assert config.commands.test is not None
        assert config.commands.test.command == long_command

    def test_command_with_special_yaml_chars(self, tmp_path: Path) -> None:
        """Command with special YAML characters is handled."""
        config_file = tmp_path / "mala.yaml"
        config_file.write_text('commands:\n  test: "pytest: with colon"\n')

        config = load_config(tmp_path)

        assert config.commands.test is not None
        assert config.commands.test.command == "pytest: with colon"

    def test_null_command_is_none(self, tmp_path: Path) -> None:
        """Explicit null command results in None."""
        config_file = tmp_path / "mala.yaml"
        config_file.write_text("commands:\n  test: pytest\n  lint: null\n")

        config = load_config(tmp_path)

        assert config.commands.test is not None
        assert config.commands.lint is None

    def test_invalid_commands_type(self, tmp_path: Path) -> None:
        """Non-object commands field raises ConfigError."""
        config_file = tmp_path / "mala.yaml"
        config_file.write_text("commands: invalid\n")

        with pytest.raises(ConfigError) as exc_info:
            load_config(tmp_path)

        assert "commands must be an object" in str(exc_info.value)

    def test_invalid_coverage_type(self, tmp_path: Path) -> None:
        """Non-object coverage field raises ConfigError."""
        config_file = tmp_path / "mala.yaml"
        config_file.write_text("commands:\n  test: pytest\ncoverage: invalid\n")

        with pytest.raises(ConfigError) as exc_info:
            load_config(tmp_path)

        assert "coverage must be an object" in str(exc_info.value)

    def test_invalid_code_patterns_type(self, tmp_path: Path) -> None:
        """Non-list code_patterns field raises ConfigError."""
        config_file = tmp_path / "mala.yaml"
        config_file.write_text("commands:\n  test: pytest\ncode_patterns: '*.py'\n")

        with pytest.raises(ConfigError) as exc_info:
            load_config(tmp_path)

        assert "code_patterns must be a list" in str(exc_info.value)

    def test_non_string_top_level_keys(self, tmp_path: Path) -> None:
        """Non-string top-level keys (null, integers) raise ConfigError, not TypeError."""
        # YAML allows non-string keys; this tests the fix for TypeError in sorted()
        config_file = tmp_path / "mala.yaml"
        # null: and 1: are valid YAML keys that become None and int in Python
        config_file.write_text("null: foo\n1: bar\ncommands:\n  test: pytest\n")

        with pytest.raises(ConfigError) as exc_info:
            load_config(tmp_path)

        # Should raise ConfigError about unknown field, not TypeError
        assert "Unknown field" in str(exc_info.value)
        assert "in mala.yaml" in str(exc_info.value)

"""Tests for validation configuration dataclasses.

Tests the configuration schema for mala.yaml including:
- CommandConfig: Single command with optional timeout
- YamlCoverageConfig: Coverage settings
- CommandsConfig: All validation commands
- ValidationConfig: Top-level configuration
- ConfigError and PresetNotFoundError exceptions
"""

import pytest

from src.domain.validation.config import (
    CommandConfig,
    CommandsConfig,
    ConfigError,
    CustomCommandConfig,
    PresetNotFoundError,
    PromptValidationCommands,
    ValidationConfig,
    YamlCoverageConfig,
)


class TestCommandConfig:
    """Tests for CommandConfig dataclass."""

    def test_from_string(self) -> None:
        """String value creates CommandConfig with no timeout."""
        config = CommandConfig.from_value("uv run pytest")
        assert config.command == "uv run pytest"
        assert config.timeout is None

    def test_from_dict_with_timeout(self) -> None:
        """Dict with timeout creates CommandConfig with timeout."""
        config = CommandConfig.from_value({"command": "pytest", "timeout": 300})
        assert config.command == "pytest"
        assert config.timeout == 300

    def test_from_dict_without_timeout(self) -> None:
        """Dict without timeout creates CommandConfig with None timeout."""
        config = CommandConfig.from_value({"command": "go test ./..."})
        assert config.command == "go test ./..."
        assert config.timeout is None

    def test_from_dict_missing_command(self) -> None:
        """Dict without 'command' key raises ConfigError."""
        with pytest.raises(ConfigError, match="must have a 'command' string field"):
            CommandConfig.from_value({"timeout": 60})

    def test_from_dict_empty_command(self) -> None:
        """Dict with empty command string raises ConfigError."""
        with pytest.raises(ConfigError, match="cannot be empty string"):
            CommandConfig.from_value({"command": ""})

    def test_from_string_empty(self) -> None:
        """Empty string shorthand raises ConfigError."""
        with pytest.raises(ConfigError, match="cannot be empty string"):
            CommandConfig.from_value("")

    def test_from_dict_invalid_timeout_type(self) -> None:
        """Dict with non-integer timeout raises ConfigError."""
        with pytest.raises(ConfigError, match="timeout must be an integer"):
            CommandConfig.from_value({"command": "pytest", "timeout": "60"})

    def test_from_dict_boolean_timeout_rejected(self) -> None:
        """Boolean timeout is rejected (bool is subclass of int)."""
        with pytest.raises(ConfigError, match="timeout must be an integer"):
            CommandConfig.from_value({"command": "pytest", "timeout": True})
        with pytest.raises(ConfigError, match="timeout must be an integer"):
            CommandConfig.from_value({"command": "pytest", "timeout": False})

    def test_from_invalid_type(self) -> None:
        """Non-string, non-dict value raises ConfigError."""
        with pytest.raises(ConfigError, match="must be a string or object"):
            CommandConfig.from_value(123)  # type: ignore[arg-type]


class TestYamlCoverageConfig:
    """Tests for YamlCoverageConfig dataclass."""

    def test_from_dict_all_fields(self) -> None:
        """Dict with all fields creates valid config."""
        config = YamlCoverageConfig.from_dict(
            {
                "format": "xml",
                "file": "coverage.xml",
                "threshold": 80,
                "command": "pytest --cov",
                "timeout": 600,
            }
        )
        assert config.format == "xml"
        assert config.file == "coverage.xml"
        assert config.threshold == 80.0
        assert config.command == "pytest --cov"
        assert config.timeout == 600

    def test_from_dict_required_fields_only(self) -> None:
        """Dict with only required fields creates config with None optionals."""
        config = YamlCoverageConfig.from_dict(
            {"format": "xml", "file": "coverage.xml", "threshold": 85}
        )
        assert config.format == "xml"
        assert config.file == "coverage.xml"
        assert config.threshold == 85.0
        assert config.command is None
        assert config.timeout is None

    def test_from_dict_missing_format(self) -> None:
        """Dict missing 'format' raises ConfigError."""
        with pytest.raises(ConfigError, match=r"missing required field.*format"):
            YamlCoverageConfig.from_dict({"file": "coverage.xml", "threshold": 80})

    def test_from_dict_missing_file(self) -> None:
        """Dict missing 'file' raises ConfigError."""
        with pytest.raises(ConfigError, match=r"missing required field.*file"):
            YamlCoverageConfig.from_dict({"format": "xml", "threshold": 80})

    def test_from_dict_missing_threshold(self) -> None:
        """Dict missing 'threshold' raises ConfigError."""
        with pytest.raises(ConfigError, match=r"missing required field.*threshold"):
            YamlCoverageConfig.from_dict({"format": "xml", "file": "coverage.xml"})

    def test_unsupported_format(self) -> None:
        """Unsupported format raises ConfigError."""
        with pytest.raises(ConfigError, match="Unsupported coverage format 'lcov'"):
            YamlCoverageConfig.from_dict(
                {"format": "lcov", "file": "lcov.info", "threshold": 80}
            )

    def test_threshold_below_zero(self) -> None:
        """Threshold below 0 raises ConfigError."""
        with pytest.raises(ConfigError, match="threshold must be between 0 and 100"):
            YamlCoverageConfig.from_dict(
                {"format": "xml", "file": "coverage.xml", "threshold": -5}
            )

    def test_threshold_above_100(self) -> None:
        """Threshold above 100 raises ConfigError."""
        with pytest.raises(ConfigError, match="threshold must be between 0 and 100"):
            YamlCoverageConfig.from_dict(
                {"format": "xml", "file": "coverage.xml", "threshold": 105}
            )

    def test_threshold_as_float(self) -> None:
        """Float threshold is accepted and stored."""
        config = YamlCoverageConfig.from_dict(
            {"format": "xml", "file": "coverage.xml", "threshold": 85.5}
        )
        assert config.threshold == 85.5

    def test_boolean_threshold_rejected(self) -> None:
        """Boolean threshold is rejected (bool is subclass of int)."""
        with pytest.raises(ConfigError, match="threshold must be a number"):
            YamlCoverageConfig.from_dict(
                {"format": "xml", "file": "coverage.xml", "threshold": True}
            )
        with pytest.raises(ConfigError, match="threshold must be a number"):
            YamlCoverageConfig.from_dict(
                {"format": "xml", "file": "coverage.xml", "threshold": False}
            )

    def test_empty_command_string(self) -> None:
        """Empty command string raises ConfigError."""
        with pytest.raises(ConfigError, match="command cannot be empty string"):
            YamlCoverageConfig.from_dict(
                {
                    "format": "xml",
                    "file": "coverage.xml",
                    "threshold": 80,
                    "command": "",
                }
            )

    def test_empty_file_string(self) -> None:
        """Empty file string raises ConfigError."""
        with pytest.raises(ConfigError, match="file path cannot be empty"):
            YamlCoverageConfig.from_dict({"format": "xml", "file": "", "threshold": 80})

    def test_invalid_timeout_type(self) -> None:
        """Non-integer timeout raises ConfigError."""
        with pytest.raises(ConfigError, match="timeout must be an integer"):
            YamlCoverageConfig.from_dict(
                {
                    "format": "xml",
                    "file": "coverage.xml",
                    "threshold": 80,
                    "timeout": "300",
                }
            )

    def test_boolean_timeout_rejected(self) -> None:
        """Boolean timeout is rejected (bool is subclass of int)."""
        with pytest.raises(ConfigError, match="timeout must be an integer"):
            YamlCoverageConfig.from_dict(
                {
                    "format": "xml",
                    "file": "coverage.xml",
                    "threshold": 80,
                    "timeout": True,
                }
            )
        with pytest.raises(ConfigError, match="timeout must be an integer"):
            YamlCoverageConfig.from_dict(
                {
                    "format": "xml",
                    "file": "coverage.xml",
                    "threshold": 80,
                    "timeout": False,
                }
            )


class TestCommandsConfig:
    """Tests for CommandsConfig dataclass."""

    def test_from_dict_all_commands(self) -> None:
        """Dict with all command types creates full config."""
        config = CommandsConfig.from_dict(
            {
                "setup": "npm install",
                "test": "npm test",
                "lint": "npx eslint .",
                "format": "npx prettier --check .",
                "typecheck": "npx tsc --noEmit",
                "e2e": "npm run e2e",
            }
        )
        assert config.setup is not None
        assert config.setup.command == "npm install"
        assert config.test is not None
        assert config.test.command == "npm test"
        assert config.lint is not None
        assert config.lint.command == "npx eslint ."
        assert config.format is not None
        assert config.format.command == "npx prettier --check ."
        assert config.typecheck is not None
        assert config.typecheck.command == "npx tsc --noEmit"
        assert config.e2e is not None
        assert config.e2e.command == "npm run e2e"

    def test_from_dict_partial_commands(self) -> None:
        """Dict with some commands leaves others as None."""
        config = CommandsConfig.from_dict(
            {"setup": "go mod download", "test": "go test ./..."}
        )
        assert config.setup is not None
        assert config.setup.command == "go mod download"
        assert config.test is not None
        assert config.test.command == "go test ./..."
        assert config.lint is None
        assert config.format is None
        assert config.typecheck is None
        assert config.e2e is None

    def test_from_none(self) -> None:
        """None input creates empty CommandsConfig."""
        config = CommandsConfig.from_dict(None)
        assert config.setup is None
        assert config.test is None
        assert config.lint is None
        assert config.format is None
        assert config.typecheck is None
        assert config.e2e is None

    def test_from_dict_with_null_command(self) -> None:
        """Null command value is stored as None."""
        config = CommandsConfig.from_dict({"setup": "npm install", "lint": None})
        assert config.setup is not None
        assert config.lint is None

    def test_from_dict_with_timeout(self) -> None:
        """Command with timeout object is parsed correctly."""
        config = CommandsConfig.from_dict(
            {"test": {"command": "pytest", "timeout": 300}}
        )
        assert config.test is not None
        assert config.test.command == "pytest"
        assert config.test.timeout == 300

    def test_from_dict_unknown_kind(self) -> None:
        """Unknown command kind raises ConfigError."""
        with pytest.raises(ConfigError, match=r"Unknown command kind.*unknown"):
            CommandsConfig.from_dict({"unknown": "some command"})

    def test_from_dict_empty_command_string(self) -> None:
        """Empty command string raises ConfigError."""
        with pytest.raises(ConfigError, match="cannot be empty string"):
            CommandsConfig.from_dict({"test": ""})


class TestValidationConfig:
    """Tests for ValidationConfig dataclass."""

    def test_from_dict_minimal(self) -> None:
        """Empty dict creates config with defaults."""
        config = ValidationConfig.from_dict({})
        assert config.preset is None
        assert config.commands.setup is None
        assert config.coverage is None
        assert config.code_patterns == ()
        assert config.config_files == ()
        assert config.setup_files == ()

    def test_from_dict_preset_only(self) -> None:
        """Dict with only preset creates config with preset set."""
        config = ValidationConfig.from_dict({"preset": "python-uv"})
        assert config.preset == "python-uv"
        assert config.commands.setup is None

    def test_from_dict_full_config(self) -> None:
        """Dict with all fields creates complete config."""
        config = ValidationConfig.from_dict(
            {
                "preset": "go",
                "commands": {
                    "setup": "go mod download",
                    "test": "go test ./...",
                    "lint": "golangci-lint run",
                },
                "coverage": {
                    "format": "xml",
                    "file": "coverage.xml",
                    "threshold": 80,
                },
                "code_patterns": ["*.go", "go.mod"],
                "config_files": [".golangci.yml"],
                "setup_files": ["go.mod", "go.sum"],
            }
        )
        assert config.preset == "go"
        assert config.commands.setup is not None
        assert config.commands.setup.command == "go mod download"
        assert config.coverage is not None
        assert config.coverage.threshold == 80.0
        assert config.code_patterns == ("*.go", "go.mod")
        assert config.config_files == (".golangci.yml",)
        assert config.setup_files == ("go.mod", "go.sum")

    def test_from_dict_invalid_preset_type(self) -> None:
        """Non-string preset raises ConfigError."""
        with pytest.raises(ConfigError, match="preset must be a string"):
            ValidationConfig.from_dict({"preset": 123})

    def test_from_dict_invalid_commands_type(self) -> None:
        """Non-object commands raises ConfigError."""
        with pytest.raises(ConfigError, match="commands must be an object"):
            ValidationConfig.from_dict({"commands": "invalid"})

    def test_from_dict_invalid_coverage_type(self) -> None:
        """Non-object coverage raises ConfigError."""
        with pytest.raises(ConfigError, match="coverage must be an object"):
            ValidationConfig.from_dict({"coverage": "invalid"})

    def test_from_dict_invalid_patterns_type(self) -> None:
        """Non-list code_patterns raises ConfigError."""
        with pytest.raises(ConfigError, match="code_patterns must be a list"):
            ValidationConfig.from_dict({"code_patterns": "*.py"})

    def test_from_dict_invalid_pattern_item_type(self) -> None:
        """Non-string pattern item raises ConfigError."""
        with pytest.raises(ConfigError, match=r"code_patterns\[0\] must be a string"):
            ValidationConfig.from_dict({"code_patterns": [123]})

    def test_deprecated_custom_commands_key_raises_error(self) -> None:
        """Top-level 'custom_commands' key raises ConfigError with migration hint."""
        with pytest.raises(
            ConfigError,
            match=r"'custom_commands' is no longer supported.*'commands' section",
        ):
            ValidationConfig.from_dict({"custom_commands": {"my_check": "echo test"}})

    def test_deprecated_custom_commands_error_references_spec(self) -> None:
        """custom_commands error message references the spec document."""
        with pytest.raises(
            ConfigError,
            match=r"plans/2026-01-06-custom-validation-commands-spec\.md",
        ):
            ValidationConfig.from_dict({"custom_commands": {}})

    def test_deprecated_run_level_custom_commands_key_raises_error(self) -> None:
        """Top-level 'run_level_custom_commands' key raises ConfigError with migration hint."""
        with pytest.raises(
            ConfigError,
            match=r"'run_level_custom_commands' is no longer supported.*'run_level_commands' section",
        ):
            ValidationConfig.from_dict(
                {"run_level_custom_commands": {"my_check": "echo test"}}
            )

    def test_deprecated_run_level_custom_commands_error_references_spec(self) -> None:
        """run_level_custom_commands error message references the spec document."""
        with pytest.raises(
            ConfigError,
            match=r"plans/2026-01-06-custom-validation-commands-spec\.md",
        ):
            ValidationConfig.from_dict({"run_level_custom_commands": {}})

    def test_deprecated_both_custom_commands_keys_raises_first_error(self) -> None:
        """Both deprecated keys present raises error for custom_commands first."""
        # custom_commands is checked first, so its error is raised
        with pytest.raises(ConfigError, match=r"'custom_commands' is no longer"):
            ValidationConfig.from_dict(
                {
                    "custom_commands": {"a": "cmd a"},
                    "run_level_custom_commands": {"b": "cmd b"},
                }
            )

    def test_has_any_command_true(self) -> None:
        """has_any_command returns True when at least one command defined."""
        config = ValidationConfig.from_dict({"commands": {"test": "pytest"}})
        assert config.has_any_command() is True

    def test_has_any_command_false(self) -> None:
        """has_any_command returns False when no commands defined."""
        config = ValidationConfig.from_dict({})
        assert config.has_any_command() is False

    def test_has_any_command_with_preset_only(self) -> None:
        """has_any_command returns False with only preset (no inline commands)."""
        config = ValidationConfig.from_dict({"preset": "python-uv"})
        assert config.has_any_command() is False

    def test_patterns_converted_to_tuples(self) -> None:
        """List patterns are converted to tuples for immutability."""
        config = ValidationConfig.from_dict(
            {"code_patterns": ["*.py"], "config_files": ["ruff.toml"]}
        )
        assert isinstance(config.code_patterns, tuple)
        assert isinstance(config.config_files, tuple)
        assert isinstance(config.setup_files, tuple)

    def test_custom_commands_field_populated_directly(self) -> None:
        """Custom commands field can be populated directly on ValidationConfig."""
        # Note: from_dict no longer supports top-level custom_commands (deprecated).
        # This test verifies the field works when set directly via constructor.
        security_cmd = CustomCommandConfig.from_value(
            "security", {"command": "bandit -r src/", "allow_fail": True}
        )
        docs_cmd = CustomCommandConfig.from_value("docs", "mkdocs build --strict")
        config = ValidationConfig(
            custom_commands={"security": security_cmd, "docs": docs_cmd}
        )
        assert len(config.custom_commands) == 2
        assert "security" in config.custom_commands
        assert "docs" in config.custom_commands

        security = config.custom_commands["security"]
        assert isinstance(security, CustomCommandConfig)
        assert security.command == "bandit -r src/"
        assert security.allow_fail is True

        docs = config.custom_commands["docs"]
        assert isinstance(docs, CustomCommandConfig)
        assert docs.command == "mkdocs build --strict"
        assert docs.allow_fail is False

    def test_custom_commands_default_empty(self) -> None:
        """Default custom_commands is empty dict."""
        config = ValidationConfig.from_dict({})
        assert config.custom_commands == {}

    def test_custom_commands_not_tracked_in_fields_set(self) -> None:
        """custom_commands is not tracked in _fields_set when parsed via from_dict.

        Note: The deprecated top-level 'custom_commands' key now raises an error.
        The field is only populated when using inline +prefix commands (future)
        or setting directly on the constructor.
        """
        config = ValidationConfig.from_dict({})
        assert "custom_commands" not in config._fields_set


class TestPresetNotFoundError:
    """Tests for PresetNotFoundError exception."""

    def test_with_available_presets(self) -> None:
        """Error message includes available presets."""
        error = PresetNotFoundError("unknown", ["go", "python-uv", "rust"])
        assert error.preset_name == "unknown"
        assert "unknown" in str(error)
        assert "go" in str(error)
        assert "python-uv" in str(error)
        assert "rust" in str(error)
        # Available presets should be sorted
        assert "go, python-uv, rust" in str(error)

    def test_without_available_presets(self) -> None:
        """Error message works without available presets list."""
        error = PresetNotFoundError("unknown")
        assert error.preset_name == "unknown"
        assert error.available == []
        assert "Unknown preset 'unknown'" in str(error)


class TestPromptValidationCommands:
    """Tests for PromptValidationCommands dataclass."""

    def test_from_validation_config_full_commands(self) -> None:
        """Full config creates PromptValidationCommands with all commands."""
        config = ValidationConfig.from_dict(
            {
                "commands": {
                    "lint": "golangci-lint run",
                    "format": 'test -z "$(gofmt -l .)"',
                    "typecheck": "go vet ./...",
                    "test": "go test ./...",
                }
            }
        )
        prompt_cmds = PromptValidationCommands.from_validation_config(config)
        assert prompt_cmds.lint == "golangci-lint run"
        assert prompt_cmds.format == 'test -z "$(gofmt -l .)"'
        assert prompt_cmds.typecheck == "go vet ./..."
        assert prompt_cmds.test == "go test ./..."

    def test_from_validation_config_missing_commands_use_fallbacks(self) -> None:
        """Missing commands use fallback messages."""
        config = ValidationConfig.from_dict(
            {
                "commands": {
                    "test": "pytest",
                }
            }
        )
        prompt_cmds = PromptValidationCommands.from_validation_config(config)
        assert prompt_cmds.test == "pytest"
        assert "No lint command configured" in prompt_cmds.lint
        assert "No format command configured" in prompt_cmds.format
        assert "No typecheck command configured" in prompt_cmds.typecheck

    def test_from_validation_config_no_commands(self) -> None:
        """Empty commands config uses all fallbacks."""
        config = ValidationConfig.from_dict({})
        prompt_cmds = PromptValidationCommands.from_validation_config(config)
        assert "No lint command configured" in prompt_cmds.lint
        assert "No format command configured" in prompt_cmds.format
        assert "No typecheck command configured" in prompt_cmds.typecheck
        assert "No test command configured" in prompt_cmds.test
        assert prompt_cmds.custom_commands == ()

    def test_prompt_validation_commands_includes_custom_commands(self) -> None:
        """Custom commands are populated as (name, command, timeout, allow_fail) tuples."""
        # Build ValidationConfig with custom_commands directly (from_dict deprecated)
        check_types_cmd = CustomCommandConfig.from_value("check_types", "mypy .")
        slow_check_cmd = CustomCommandConfig.from_value(
            "slow_check",
            {"command": "slow-cmd", "timeout": 300, "allow_fail": True},
        )
        config = ValidationConfig(
            commands=CommandsConfig(test=CommandConfig(command="pytest")),
            custom_commands={
                "check_types": check_types_cmd,
                "slow_check": slow_check_cmd,
            },
        )
        prompt_cmds = PromptValidationCommands.from_validation_config(config)

        # Verify custom_commands contains tuples with correct values
        assert len(prompt_cmds.custom_commands) == 2

        # Convert to dict for easier verification (order not guaranteed)
        custom_dict = {
            name: (cmd, timeout, allow_fail)
            for name, cmd, timeout, allow_fail in prompt_cmds.custom_commands
        }

        # check_types: string shorthand uses default timeout 120, allow_fail=False
        assert custom_dict["check_types"] == ("mypy .", 120, False)

        # slow_check: object form with explicit timeout and allow_fail
        assert custom_dict["slow_check"] == ("slow-cmd", 300, True)


class TestCustomCommandConfig:
    """Tests for CustomCommandConfig dataclass."""

    def test_from_string_shorthand(self) -> None:
        """String shorthand creates config with defaults."""
        config = CustomCommandConfig.from_value("my_cmd", "uvx cmd")
        assert config.command == "uvx cmd"
        assert config.timeout == 120  # default per spec
        assert config.allow_fail is False

    def test_from_object_form(self) -> None:
        """Object form with all fields creates correct config."""
        config = CustomCommandConfig.from_value(
            "my_cmd", {"command": "uvx cmd", "timeout": 60, "allow_fail": True}
        )
        assert config.command == "uvx cmd"
        assert config.timeout == 60
        assert config.allow_fail is True

    def test_defaults_from_object_form(self) -> None:
        """Object form with only command uses defaults for other fields."""
        config = CustomCommandConfig.from_value("my_cmd", {"command": "go test ./..."})
        assert config.command == "go test ./..."
        assert config.timeout == 120  # default
        assert config.allow_fail is False  # default

    def test_explicit_null_timeout_uses_default(self) -> None:
        """Explicit timeout: null uses default (120), not None."""
        config = CustomCommandConfig.from_value(
            "my_cmd", {"command": "uvx cmd", "timeout": None}
        )
        assert config.timeout == 120

    def test_invalid_name_starts_with_digit(self) -> None:
        """Name starting with digit raises ConfigError."""
        with pytest.raises(ConfigError, match="Invalid custom command name '123abc'"):
            CustomCommandConfig.from_value("123abc", "some cmd")

    def test_valid_name_contains_hyphen(self) -> None:
        """Name containing hyphen is valid."""
        config = CustomCommandConfig.from_value("cmd-name", "some cmd")
        assert config.command == "some cmd"

    def test_invalid_name_contains_dot(self) -> None:
        """Name containing dot raises ConfigError."""
        with pytest.raises(
            ConfigError, match=r"Invalid custom command name 'cmd\.name'"
        ):
            CustomCommandConfig.from_value("cmd.name", "some cmd")

    def test_valid_name_with_underscore(self) -> None:
        """Name with underscore is valid."""
        config = CustomCommandConfig.from_value("my_cmd_2", "some cmd")
        assert config.command == "some cmd"

    def test_valid_name_starts_with_underscore(self) -> None:
        """Name starting with underscore is valid."""
        config = CustomCommandConfig.from_value("_private", "some cmd")
        assert config.command == "some cmd"

    def test_null_value_error(self) -> None:
        """Null value raises ConfigError with guidance."""
        with pytest.raises(ConfigError, match="use run-level override to disable"):
            CustomCommandConfig.from_value("my_cmd", None)  # type: ignore[arg-type]

    def test_empty_command_string_error(self) -> None:
        """Empty command string raises ConfigError."""
        with pytest.raises(ConfigError, match="cannot be empty"):
            CustomCommandConfig.from_value("my_cmd", "")

    def test_whitespace_only_command_error(self) -> None:
        """Whitespace-only command raises ConfigError."""
        with pytest.raises(ConfigError, match="cannot be empty"):
            CustomCommandConfig.from_value("my_cmd", "   ")

    def test_empty_command_in_object_form_error(self) -> None:
        """Empty command in object form raises ConfigError."""
        with pytest.raises(ConfigError, match="cannot be empty"):
            CustomCommandConfig.from_value("my_cmd", {"command": ""})

    def test_unknown_keys_error(self) -> None:
        """Unknown keys in object form raises ConfigError."""
        with pytest.raises(ConfigError, match="Unknown key 'foo'"):
            CustomCommandConfig.from_value(
                "my_cmd", {"command": "some cmd", "foo": "bar"}
            )

    def test_multiple_unknown_keys_error(self) -> None:
        """Multiple unknown keys mentions first unknown key."""
        with pytest.raises(ConfigError, match="Unknown key"):
            CustomCommandConfig.from_value(
                "my_cmd", {"command": "some cmd", "foo": "bar", "baz": 123}
            )

    def test_missing_command_in_object_form_error(self) -> None:
        """Object form without command key raises ConfigError."""
        with pytest.raises(ConfigError, match="must have a 'command' string field"):
            CustomCommandConfig.from_value("my_cmd", {"timeout": 60})

    def test_invalid_timeout_type(self) -> None:
        """Non-integer timeout raises ConfigError."""
        with pytest.raises(ConfigError, match="timeout must be an integer"):
            CustomCommandConfig.from_value(
                "my_cmd", {"command": "cmd", "timeout": "60"}
            )

    def test_boolean_timeout_rejected(self) -> None:
        """Boolean timeout is rejected."""
        with pytest.raises(ConfigError, match="timeout must be an integer"):
            CustomCommandConfig.from_value(
                "my_cmd", {"command": "cmd", "timeout": True}
            )

    def test_invalid_allow_fail_type(self) -> None:
        """Non-boolean allow_fail raises ConfigError."""
        with pytest.raises(ConfigError, match="allow_fail must be a boolean"):
            CustomCommandConfig.from_value(
                "my_cmd", {"command": "cmd", "allow_fail": "yes"}
            )

    def test_invalid_value_type(self) -> None:
        """Non-string, non-dict value raises ConfigError."""
        with pytest.raises(ConfigError, match="must be a string or object"):
            CustomCommandConfig.from_value("my_cmd", 123)  # type: ignore[arg-type]

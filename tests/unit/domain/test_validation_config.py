"""Tests for validation configuration dataclasses.

Tests the configuration schema for mala.yaml including:
- CommandConfig: Single command with optional timeout
- YamlCoverageConfig: Coverage settings
- CommandsConfig: All validation commands
- ValidationConfig: Top-level configuration
- ConfigError and PresetNotFoundError exceptions
"""

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import pathlib

from src.domain.validation.config import (
    CerberusConfig,
    CodeReviewConfig,
    CommandConfig,
    CommandsConfig,
    ConfigError,
    CustomCommandConfig,
    CustomOverrideMode,
    EpicCompletionTriggerConfig,
    EpicDepth,
    FailureMode,
    FireOn,
    PeriodicTriggerConfig,
    PresetNotFoundError,
    PromptValidationCommands,
    RunEndTriggerConfig,
    SessionEndTriggerConfig,
    TriggerCommandRef,
    TriggerType,
    ValidationConfig,
    ValidationTriggersConfig,
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

    def test_from_dict_unknown_kind_parsed_as_custom(self) -> None:
        """Unknown command kind is parsed as custom command."""
        config = CommandsConfig.from_dict({"unknown": "some command"})
        assert "unknown" in config.custom_commands
        assert config.custom_commands["unknown"].command == "some command"

    def test_from_dict_empty_command_string(self) -> None:
        """Empty command string raises ConfigError."""
        with pytest.raises(ConfigError, match="cannot be empty string"):
            CommandsConfig.from_dict({"test": ""})

    def test_from_dict_non_string_key_raises_error(self) -> None:
        """Non-string command key raises ConfigError."""
        with pytest.raises(ConfigError, match="Command key must be a string"):
            CommandsConfig.from_dict({1: "some command"})  # type: ignore[dict-item]

    def test_default_custom_commands_and_override_mode(self) -> None:
        """CommandsConfig defaults: empty custom_commands and INHERIT mode."""
        config = CommandsConfig.from_dict(None)
        assert config.custom_commands == {}
        assert config.custom_override_mode == CustomOverrideMode.INHERIT

    def test_from_dict_accepts_is_global_param(self) -> None:
        """from_dict accepts is_global kwarg (stub for future parsing)."""
        # is_global doesn't change behavior yet, but signature should accept it
        config = CommandsConfig.from_dict({"lint": "ruff check ."}, is_global=True)
        assert config.lint is not None
        assert config.lint.command == "ruff check ."
        # custom_commands still empty until inline parsing implemented
        assert config.custom_commands == {}


class TestClearCustoms:
    """Tests for _clear_customs reserved key handling."""

    def test_clear_customs_true_at_global_sets_clear_mode(self) -> None:
        """_clear_customs: true at global sets mode=CLEAR with empty customs."""
        config = CommandsConfig.from_dict({"_clear_customs": True}, is_global=True)
        assert config.custom_override_mode == CustomOverrideMode.CLEAR
        assert config.custom_commands == {}

    def test_clear_customs_false_raises_error(self) -> None:
        """_clear_customs: false raises ConfigError."""
        with pytest.raises(ConfigError, match="_clear_customs must be true"):
            CommandsConfig.from_dict({"_clear_customs": False}, is_global=True)

    def test_clear_customs_string_yes_raises_error(self) -> None:
        """_clear_customs: 'yes' raises ConfigError (must be boolean true)."""
        with pytest.raises(ConfigError, match="_clear_customs must be true"):
            CommandsConfig.from_dict({"_clear_customs": "yes"}, is_global=True)

    def test_clear_customs_at_repo_level_raises_error(self) -> None:
        """_clear_customs at repo-level (is_global=False) raises ConfigError."""
        with pytest.raises(
            ConfigError,
            match="_clear_customs is only valid in global_validation_commands",
        ):
            CommandsConfig.from_dict({"_clear_customs": True}, is_global=False)

    def test_clear_customs_with_custom_keys_raises_error(self) -> None:
        """_clear_customs combined with custom command keys raises ConfigError."""
        with pytest.raises(
            ConfigError, match="_clear_customs cannot be combined with custom commands"
        ):
            CommandsConfig.from_dict(
                {"_clear_customs": True, "my_custom": "some cmd"}, is_global=True
            )

    def test_clear_customs_with_plus_prefixed_custom_keys_raises_error(self) -> None:
        """_clear_customs combined with +prefixed custom keys raises ConfigError."""
        with pytest.raises(
            ConfigError, match="_clear_customs cannot be combined with custom commands"
        ):
            CommandsConfig.from_dict(
                {"_clear_customs": True, "+my_custom": "some cmd"}, is_global=True
            )

    def test_plus_clear_customs_raises_error(self) -> None:
        """+_clear_customs raises ConfigError (reserved key cannot be prefixed)."""
        with pytest.raises(ConfigError, match=r"\+_clear_customs is not allowed"):
            CommandsConfig.from_dict({"+_clear_customs": True}, is_global=True)

    def test_clear_customs_with_builtin_commands_allowed(self) -> None:
        """_clear_customs can be combined with built-in commands like lint."""
        config = CommandsConfig.from_dict(
            {"_clear_customs": True, "lint": "ruff check ."}, is_global=True
        )
        assert config.custom_override_mode == CustomOverrideMode.CLEAR
        assert config.custom_commands == {}
        assert config.lint is not None
        assert config.lint.command == "ruff check ."


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

    def test_deprecated_custom_commands_error_references_plan(self) -> None:
        """custom_commands error message references the plan document."""
        with pytest.raises(
            ConfigError,
            match=r"plans/2026-01-06-inline-custom-commands-plan\.md",
        ):
            ValidationConfig.from_dict({"custom_commands": {}})

    def test_deprecated_global_custom_commands_key_raises_error(self) -> None:
        """Top-level 'global_custom_commands' key raises ConfigError with migration hint."""
        with pytest.raises(
            ConfigError,
            match=r"'global_custom_commands' is no longer supported.*'global_validation_commands' section",
        ):
            ValidationConfig.from_dict(
                {"global_custom_commands": {"my_check": "echo test"}}
            )

    def test_deprecated_global_custom_commands_error_references_plan(self) -> None:
        """global_custom_commands error message references the plan document."""
        with pytest.raises(
            ConfigError,
            match=r"plans/2026-01-06-inline-custom-commands-plan\.md",
        ):
            ValidationConfig.from_dict({"global_custom_commands": {}})

    def test_deprecated_both_custom_commands_keys_raises_first_error(self) -> None:
        """Both deprecated keys present raises error for custom_commands first."""
        # custom_commands is checked first, so its error is raised
        with pytest.raises(ConfigError, match=r"'custom_commands' is no longer"):
            ValidationConfig.from_dict(
                {
                    "custom_commands": {"a": "cmd a"},
                    "global_custom_commands": {"b": "cmd b"},
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
        Custom commands are now defined inline in the 'commands' section and parsed
        into CommandsConfig.custom_commands (not ValidationConfig.custom_commands).
        """
        config = ValidationConfig.from_dict({})
        assert "custom_commands" not in config._fields_set

    def test_inline_custom_commands_integration_path(self) -> None:
        """Integration test: inline custom commands in commands section.

        This test exercises the full parsing path from YAML-like dict through
        ValidationConfig.from_dict -> CommandsConfig.from_dict. Custom commands
        defined alongside standard commands are parsed into CommandsConfig.custom_commands.
        """
        yaml_data = {
            "preset": "python-uv",
            "commands": {
                "lint": "uvx ruff check .",
                "test": "uv run pytest",
                "security": "bandit -r src/",
            },
        }

        config = ValidationConfig.from_dict(yaml_data)
        assert "security" in config.commands.custom_commands
        assert config.commands.custom_commands["security"].command == "bandit -r src/"
        # At repo-level, mode stays INHERIT (mode detection only matters at global)
        assert config.commands.custom_override_mode == CustomOverrideMode.INHERIT

    def test_reviewer_type_defaults_to_agent_sdk(self) -> None:
        """Default reviewer_type is 'agent_sdk'."""
        config = ValidationConfig.from_dict({})
        assert config.reviewer_type == "agent_sdk"

    def test_reviewer_type_cerberus(self) -> None:
        """reviewer_type can be set to 'cerberus'."""
        config = ValidationConfig.from_dict({"reviewer_type": "cerberus"})
        assert config.reviewer_type == "cerberus"

    def test_reviewer_type_invalid_value(self) -> None:
        """Invalid reviewer_type raises ConfigError."""
        with pytest.raises(
            ConfigError, match="reviewer_type must be 'agent_sdk' or 'cerberus'"
        ):
            ValidationConfig.from_dict({"reviewer_type": "invalid"})

    def test_agent_sdk_review_timeout_defaults(self) -> None:
        """Default agent_sdk_review_timeout is 600."""
        config = ValidationConfig.from_dict({})
        assert config.agent_sdk_review_timeout == 600

    def test_agent_sdk_review_timeout_custom(self) -> None:
        """agent_sdk_review_timeout can be customized."""
        config = ValidationConfig.from_dict({"agent_sdk_review_timeout": 300})
        assert config.agent_sdk_review_timeout == 300

    def test_agent_sdk_review_timeout_invalid_type(self) -> None:
        """Non-integer agent_sdk_review_timeout raises ConfigError."""
        with pytest.raises(
            ConfigError, match="agent_sdk_review_timeout must be an integer"
        ):
            ValidationConfig.from_dict({"agent_sdk_review_timeout": "300"})

    def test_agent_sdk_review_timeout_boolean_rejected(self) -> None:
        """Boolean agent_sdk_review_timeout is rejected (bool is subclass of int)."""
        with pytest.raises(
            ConfigError, match="agent_sdk_review_timeout must be an integer"
        ):
            ValidationConfig.from_dict({"agent_sdk_review_timeout": True})

    def test_agent_sdk_review_timeout_zero_rejected(self) -> None:
        """Zero agent_sdk_review_timeout is rejected."""
        with pytest.raises(
            ConfigError, match="agent_sdk_review_timeout must be positive"
        ):
            ValidationConfig.from_dict({"agent_sdk_review_timeout": 0})

    def test_agent_sdk_review_timeout_negative_rejected(self) -> None:
        """Negative agent_sdk_review_timeout is rejected."""
        with pytest.raises(
            ConfigError, match="agent_sdk_review_timeout must be positive"
        ):
            ValidationConfig.from_dict({"agent_sdk_review_timeout": -100})

    def test_agent_sdk_reviewer_model_defaults(self) -> None:
        """Default agent_sdk_reviewer_model is 'sonnet'."""
        config = ValidationConfig.from_dict({})
        assert config.agent_sdk_reviewer_model == "sonnet"

    def test_agent_sdk_reviewer_model_opus(self) -> None:
        """agent_sdk_reviewer_model can be set to 'opus'."""
        config = ValidationConfig.from_dict({"agent_sdk_reviewer_model": "opus"})
        assert config.agent_sdk_reviewer_model == "opus"

    def test_agent_sdk_reviewer_model_haiku(self) -> None:
        """agent_sdk_reviewer_model can be set to 'haiku'."""
        config = ValidationConfig.from_dict({"agent_sdk_reviewer_model": "haiku"})
        assert config.agent_sdk_reviewer_model == "haiku"

    def test_agent_sdk_reviewer_model_invalid(self) -> None:
        """Invalid agent_sdk_reviewer_model raises ConfigError."""
        with pytest.raises(
            ConfigError,
            match="agent_sdk_reviewer_model must be 'sonnet', 'opus', or 'haiku'",
        ):
            ValidationConfig.from_dict({"agent_sdk_reviewer_model": "gpt-4"})

    def test_reviewer_config_in_full_config(self) -> None:
        """Reviewer config fields work together in a full configuration."""
        config = ValidationConfig.from_dict(
            {
                "preset": "python-uv",
                "reviewer_type": "cerberus",
                "agent_sdk_review_timeout": 120,
                "agent_sdk_reviewer_model": "haiku",
                "commands": {"test": "uv run pytest"},
            }
        )
        assert config.reviewer_type == "cerberus"
        assert config.agent_sdk_review_timeout == 120
        assert config.agent_sdk_reviewer_model == "haiku"
        assert config.commands.test is not None

    def test_reviewer_fields_tracked_in_fields_set(self) -> None:
        """Reviewer fields are tracked in _fields_set when explicitly set."""
        config = ValidationConfig.from_dict(
            {
                "reviewer_type": "agent_sdk",
                "agent_sdk_review_timeout": 600,
                "agent_sdk_reviewer_model": "sonnet",
            }
        )
        assert "reviewer_type" in config._fields_set
        assert "agent_sdk_review_timeout" in config._fields_set
        assert "agent_sdk_reviewer_model" in config._fields_set

    def test_reviewer_fields_not_in_fields_set_when_omitted(self) -> None:
        """Reviewer fields are not in _fields_set when omitted."""
        config = ValidationConfig.from_dict({})
        assert "reviewer_type" not in config._fields_set
        assert "agent_sdk_review_timeout" not in config._fields_set
        assert "agent_sdk_reviewer_model" not in config._fields_set

    def test_from_dict_parses_validation_triggers(self) -> None:
        """from_dict correctly parses validation_triggers field."""
        data = {
            "preset": "python-uv",
            "validation_triggers": {
                "periodic": {
                    "interval": 3600,
                    "failure_mode": "continue",
                    "commands": [{"ref": "test"}],
                }
            },
        }
        config = ValidationConfig.from_dict(data)
        assert config.validation_triggers is not None
        assert config.validation_triggers.periodic is not None
        assert config.validation_triggers.periodic.interval == 3600
        assert "validation_triggers" in config._fields_set

    def test_from_dict_validation_triggers_none_when_omitted(self) -> None:
        """validation_triggers is None when not specified in dict."""
        config = ValidationConfig.from_dict({"preset": "python-uv"})
        assert config.validation_triggers is None
        assert "validation_triggers" not in config._fields_set

    def test_from_dict_validation_triggers_explicit_null(self) -> None:
        """validation_triggers: null is tracked in _fields_set but value is None."""
        data = {"preset": "python-uv", "validation_triggers": None}
        config = ValidationConfig.from_dict(data)
        assert config.validation_triggers is None
        assert "validation_triggers" in config._fields_set

    def test_from_dict_validation_triggers_empty_dict(self) -> None:
        """validation_triggers: {} creates empty config with all triggers None."""
        data = {"preset": "python-uv", "validation_triggers": {}}
        config = ValidationConfig.from_dict(data)
        assert config.validation_triggers is not None
        assert config.validation_triggers.epic_completion is None
        assert config.validation_triggers.session_end is None
        assert config.validation_triggers.periodic is None
        assert "validation_triggers" in config._fields_set

    def test_from_dict_validation_triggers_invalid_type(self) -> None:
        """validation_triggers with non-object type raises ConfigError."""
        with pytest.raises(ConfigError, match="validation_triggers must be an object"):
            ValidationConfig.from_dict({"validation_triggers": "invalid"})


class TestClaudeSettingsSources:
    """Tests for claude_settings_sources field in ValidationConfig."""

    def test_claude_settings_sources_defaults_to_none(self) -> None:
        """Default claude_settings_sources is None (use default behavior)."""
        config = ValidationConfig.from_dict({})
        assert config.claude_settings_sources is None

    def test_claude_settings_sources_local_only(self) -> None:
        """Single 'local' source is accepted."""
        config = ValidationConfig.from_dict({"claude_settings_sources": ["local"]})
        assert config.claude_settings_sources == ("local",)

    def test_claude_settings_sources_project_only(self) -> None:
        """Single 'project' source is accepted."""
        config = ValidationConfig.from_dict({"claude_settings_sources": ["project"]})
        assert config.claude_settings_sources == ("project",)

    def test_claude_settings_sources_user_only(self) -> None:
        """Single 'user' source is accepted."""
        config = ValidationConfig.from_dict({"claude_settings_sources": ["user"]})
        assert config.claude_settings_sources == ("user",)

    def test_claude_settings_sources_multiple(self) -> None:
        """Multiple valid sources are accepted."""
        config = ValidationConfig.from_dict(
            {"claude_settings_sources": ["local", "project"]}
        )
        assert config.claude_settings_sources == ("local", "project")

    def test_claude_settings_sources_all_three(self) -> None:
        """All three valid sources are accepted."""
        config = ValidationConfig.from_dict(
            {"claude_settings_sources": ["local", "project", "user"]}
        )
        assert config.claude_settings_sources == ("local", "project", "user")

    def test_claude_settings_sources_empty_list(self) -> None:
        """Empty list is valid (means no sources)."""
        config = ValidationConfig.from_dict({"claude_settings_sources": []})
        assert config.claude_settings_sources == ()

    def test_claude_settings_sources_invalid_source(self) -> None:
        """Invalid source raises ConfigError."""
        with pytest.raises(
            ConfigError,
            match=r"Invalid Claude settings source 'foo'\. Valid sources: local, project, user",
        ):
            ValidationConfig.from_dict({"claude_settings_sources": ["foo"]})

    def test_claude_settings_sources_invalid_mixed(self) -> None:
        """Invalid source in list with valid sources raises ConfigError."""
        with pytest.raises(
            ConfigError,
            match=r"Invalid Claude settings source 'invalid'\. Valid sources: local, project, user",
        ):
            ValidationConfig.from_dict(
                {"claude_settings_sources": ["local", "invalid", "project"]}
            )

    def test_claude_settings_sources_not_list(self) -> None:
        """Non-list value raises ConfigError."""
        with pytest.raises(ConfigError, match="claude_settings_sources must be a list"):
            ValidationConfig.from_dict({"claude_settings_sources": "local"})

    def test_claude_settings_sources_non_string_item(self) -> None:
        """Non-string item in list raises ConfigError."""
        with pytest.raises(
            ConfigError, match=r"claude_settings_sources\[0\] must be a string"
        ):
            ValidationConfig.from_dict({"claude_settings_sources": [123]})

    def test_claude_settings_sources_strips_whitespace(self) -> None:
        """Whitespace is stripped from source names (forgiving parsing)."""
        config = ValidationConfig.from_dict(
            {"claude_settings_sources": ["  local  ", " project"]}
        )
        assert config.claude_settings_sources == ("local", "project")

    def test_claude_settings_sources_tracked_in_fields_set(self) -> None:
        """claude_settings_sources is tracked in _fields_set when explicitly set."""
        config = ValidationConfig.from_dict({"claude_settings_sources": ["local"]})
        assert "claude_settings_sources" in config._fields_set

    def test_claude_settings_sources_empty_list_tracked_in_fields_set(self) -> None:
        """Empty list is tracked in _fields_set (explicit empty differs from None)."""
        config = ValidationConfig.from_dict({"claude_settings_sources": []})
        assert "claude_settings_sources" in config._fields_set
        assert config.claude_settings_sources == ()

    def test_claude_settings_sources_not_in_fields_set_when_omitted(self) -> None:
        """claude_settings_sources is not in _fields_set when omitted."""
        config = ValidationConfig.from_dict({})
        assert "claude_settings_sources" not in config._fields_set


class TestClaudeSettingsSourcesIntegration:
    """Integration test for claude_settings_sources full config path.

    This test verifies the path: mala.yaml → ValidationConfig → orchestrator → MalaConfig.
    It is expected to FAIL until T004 wires the orchestrator.
    """

    def test_claude_settings_sources_mala_yaml_to_validation_config(
        self, tmp_path: "pathlib.Path"
    ) -> None:
        """Test that claude_settings_sources flows from mala.yaml to ValidationConfig.

        This test creates a real mala.yaml file and uses load_config to parse it,
        exercising the full mala.yaml → ValidationConfig path.
        """
        from src.domain.validation.config_loader import load_config

        # Step 1: Create temp mala.yaml with claude_settings_sources
        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text(
            """\
preset: python-uv
claude_settings_sources:
  - local
  - project
"""
        )

        # Step 2: Load via config_loader (exercises full parsing path)
        validation_config = load_config(tmp_path)

        # Step 3: Verify ValidationConfig received the sources
        assert validation_config.claude_settings_sources == ("local", "project")
        assert "claude_settings_sources" in validation_config._fields_set

    def test_claude_settings_sources_full_path_integration(
        self, tmp_path: "pathlib.Path"
    ) -> None:
        """Test that claude_settings_sources flows through the full config path.

        This test creates a real mala.yaml file and verifies that the factory reads
        ValidationConfig from repo_path and wires claude_settings_sources to MalaConfig.

        T004 implemented the orchestrator wiring so this test now passes.
        """
        from src.domain.validation.config_loader import load_config
        from src.orchestration.factory import create_orchestrator
        from src.orchestration.types import OrchestratorConfig

        # Step 1: Create temp mala.yaml with non-default claude_settings_sources
        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text(
            """\
preset: python-uv
claude_settings_sources:
  - user
"""
        )

        # Step 2: Verify ValidationConfig parses correctly (mala.yaml → ValidationConfig)
        validation_config = load_config(tmp_path)
        assert validation_config.claude_settings_sources == ("user",)

        # Step 3: Create orchestrator via factory WITHOUT explicit mala_config
        # The factory should read ValidationConfig from repo_path and wire
        # claude_settings_sources to MalaConfig (this is what T004 will implement)
        orchestrator_config = OrchestratorConfig(repo_path=tmp_path)
        orchestrator = create_orchestrator(orchestrator_config)

        # Step 4: Verify MalaConfig received sources from ValidationConfig
        # T004 wired the factory to pass ValidationConfig.claude_settings_sources to MalaConfig
        assert orchestrator._mala_config.claude_settings_sources == ("user",)


class TestGlobalCustomCommandsMode:
    """Tests for global custom command mode detection."""

    def test_global_all_plus_prefixed_sets_additive_mode(self) -> None:
        """All +prefixed custom keys at global sets mode=ADDITIVE."""
        config = CommandsConfig.from_dict(
            {"+my_check": "cmd1", "+other_check": "cmd2"}, is_global=True
        )
        assert config.custom_override_mode == CustomOverrideMode.ADDITIVE
        assert "my_check" in config.custom_commands
        assert "other_check" in config.custom_commands
        # Names stored without + prefix
        assert config.custom_commands["my_check"].command == "cmd1"
        assert config.custom_commands["other_check"].command == "cmd2"

    def test_global_all_unprefixed_sets_replace_mode(self) -> None:
        """All unprefixed custom keys at global sets mode=REPLACE."""
        config = CommandsConfig.from_dict(
            {"my_check": "cmd1", "other_check": "cmd2"}, is_global=True
        )
        assert config.custom_override_mode == CustomOverrideMode.REPLACE
        assert "my_check" in config.custom_commands
        assert "other_check" in config.custom_commands

    def test_global_no_custom_keys_sets_inherit_mode(self) -> None:
        """No custom keys at global keeps mode=INHERIT."""
        config = CommandsConfig.from_dict({"lint": "ruff check ."}, is_global=True)
        assert config.custom_override_mode == CustomOverrideMode.INHERIT
        assert config.custom_commands == {}

    def test_global_mixed_prefixes_raises_error(self) -> None:
        """Mixed +prefixed and unprefixed custom keys raises ConfigError."""
        with pytest.raises(ConfigError, match=r"Cannot mix.*prefixed and unprefixed"):
            CommandsConfig.from_dict(
                {"+additive_cmd": "cmd1", "replace_cmd": "cmd2"}, is_global=True
            )

    def test_global_plus_builtin_collision_raises_error(self) -> None:
        """Plus-prefixed key colliding with built-in after stripping raises error."""
        with pytest.raises(ConfigError, match=r"\+lint.*conflicts.*built-in"):
            CommandsConfig.from_dict({"+lint": "some cmd"}, is_global=True)

    def test_global_builtin_key_parsed_as_builtin_not_custom(self) -> None:
        """Built-in key names are parsed as built-ins, not custom commands."""
        # "lint" matches a valid_kind so it's parsed as built-in, not custom
        config = CommandsConfig.from_dict({"lint": "some cmd"}, is_global=True)
        assert config.lint is not None
        assert config.lint.command == "some cmd"
        assert "lint" not in config.custom_commands

    def test_repo_level_plus_prefix_raises_error(self) -> None:
        """Plus-prefixed key at repo-level raises ConfigError."""
        with pytest.raises(ConfigError, match=r"Plus-prefixed.*only allowed"):
            CommandsConfig.from_dict({"+my_check": "cmd"}, is_global=False)

    def test_repo_level_custom_commands_stored_directly(self) -> None:
        """Custom commands at repo-level stored without mode change."""
        config = CommandsConfig.from_dict(
            {"my_check": "cmd1", "lint": "ruff check ."}, is_global=False
        )
        assert config.custom_override_mode == CustomOverrideMode.INHERIT
        assert "my_check" in config.custom_commands
        assert config.custom_commands["my_check"].command == "cmd1"

    def test_global_custom_with_builtin_override(self) -> None:
        """Global can have both built-in overrides and custom commands."""
        config = CommandsConfig.from_dict(
            {"lint": "new lint cmd", "security": "bandit -r src/"},
            is_global=True,
        )
        assert config.lint is not None
        assert config.lint.command == "new lint cmd"
        assert config.custom_override_mode == CustomOverrideMode.REPLACE
        assert "security" in config.custom_commands

    def test_global_plus_custom_with_builtin_override(self) -> None:
        """Global can have built-in overrides and +prefixed customs."""
        config = CommandsConfig.from_dict(
            {"lint": "new lint cmd", "+security": "bandit -r src/"},
            is_global=True,
        )
        assert config.lint is not None
        assert config.lint.command == "new lint cmd"
        assert config.custom_override_mode == CustomOverrideMode.ADDITIVE
        assert "security" in config.custom_commands

    def test_custom_command_with_object_form(self) -> None:
        """Custom commands can use object form with timeout and allow_fail."""
        config = CommandsConfig.from_dict(
            {
                "+slow_check": {
                    "command": "slow cmd",
                    "timeout": 300,
                    "allow_fail": True,
                }
            },
            is_global=True,
        )
        assert "slow_check" in config.custom_commands
        custom = config.custom_commands["slow_check"]
        assert custom.command == "slow cmd"
        assert custom.timeout == 300
        assert custom.allow_fail is True

    def test_hyphenated_custom_command_name_works(self) -> None:
        """Hyphenated custom command names are valid."""
        config = CommandsConfig.from_dict(
            {"arch-check": "some cmd", "my-other-cmd": "other cmd"},
            is_global=False,
        )
        assert "arch-check" in config.custom_commands
        assert "my-other-cmd" in config.custom_commands


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
        # Build ValidationConfig with inline custom_commands in commands
        check_types_cmd = CustomCommandConfig.from_value("check_types", "mypy .")
        slow_check_cmd = CustomCommandConfig.from_value(
            "slow_check",
            {"command": "slow-cmd", "timeout": 300, "allow_fail": True},
        )
        config = ValidationConfig(
            commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
                custom_commands={
                    "check_types": check_types_cmd,
                    "slow_check": slow_check_cmd,
                },
            ),
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
        with pytest.raises(ConfigError, match="use global override to disable"):
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


class TestBaseTriggerConfig:
    """Tests for BaseTriggerConfig dataclass."""

    def test_commands_tuple_preserved(self) -> None:
        """Tuple commands are preserved as-is."""
        cmd_ref = TriggerCommandRef(ref="test")
        # Use a concrete subclass since BaseTriggerConfig is abstract-ish (kw_only)
        config = SessionEndTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(cmd_ref,),
        )
        assert config.commands == (cmd_ref,)
        assert isinstance(config.commands, tuple)

    def test_commands_list_coerced_to_tuple(self) -> None:
        """List commands are coerced to tuple for immutability."""
        cmd_ref = TriggerCommandRef(ref="lint")
        # Pass a list instead of tuple - should be coerced
        config = SessionEndTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=[cmd_ref],  # type: ignore[arg-type]
        )
        assert config.commands == (cmd_ref,)
        assert isinstance(config.commands, tuple)

    def test_empty_commands_list_coerced(self) -> None:
        """Empty list is coerced to empty tuple."""
        config = SessionEndTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=[],  # type: ignore[arg-type]
        )
        assert config.commands == ()
        assert isinstance(config.commands, tuple)

    def test_max_retries_default_none(self) -> None:
        """max_retries defaults to None."""
        config = SessionEndTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(),
        )
        assert config.max_retries is None

    def test_max_retries_can_be_set(self) -> None:
        """max_retries can be explicitly set."""
        config = SessionEndTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(),
            max_retries=3,
        )
        assert config.max_retries == 3


class TestEpicCompletionTriggerConfig:
    """Tests for EpicCompletionTriggerConfig dataclass."""

    def test_construction_with_all_fields(self) -> None:
        """All fields can be set on construction."""
        cmd_ref = TriggerCommandRef(ref="test")
        config = EpicCompletionTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(cmd_ref,),
            max_retries=2,
            epic_depth=EpicDepth.TOP_LEVEL,
            fire_on=FireOn.SUCCESS,
        )
        assert config.failure_mode == FailureMode.ABORT
        assert config.commands == (cmd_ref,)
        assert config.max_retries == 2
        assert config.epic_depth == EpicDepth.TOP_LEVEL
        assert config.fire_on == FireOn.SUCCESS

    def test_epic_depth_all(self) -> None:
        """epic_depth can be ALL."""
        config = EpicCompletionTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(),
            epic_depth=EpicDepth.ALL,
            fire_on=FireOn.BOTH,
        )
        assert config.epic_depth == EpicDepth.ALL

    def test_fire_on_failure(self) -> None:
        """fire_on can be FAILURE."""
        config = EpicCompletionTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(),
            epic_depth=EpicDepth.TOP_LEVEL,
            fire_on=FireOn.FAILURE,
        )
        assert config.fire_on == FireOn.FAILURE

    def test_commands_list_coerced_to_tuple(self) -> None:
        """List commands are coerced to tuple via inherited __post_init__."""
        cmd_ref = TriggerCommandRef(ref="lint")
        config = EpicCompletionTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=[cmd_ref],  # type: ignore[arg-type]
            epic_depth=EpicDepth.ALL,
            fire_on=FireOn.BOTH,
        )
        assert isinstance(config.commands, tuple)
        assert config.commands == (cmd_ref,)


class TestSessionEndTriggerConfig:
    """Tests for SessionEndTriggerConfig dataclass."""

    def test_construction(self) -> None:
        """SessionEndTriggerConfig can be constructed with base fields only."""
        cmd_ref = TriggerCommandRef(ref="format")
        config = SessionEndTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(cmd_ref,),
        )
        assert config.failure_mode == FailureMode.CONTINUE
        assert config.commands == (cmd_ref,)
        assert config.max_retries is None

    def test_is_frozen(self) -> None:
        """SessionEndTriggerConfig is immutable."""
        config = SessionEndTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(),
        )
        with pytest.raises(AttributeError):
            config.failure_mode = FailureMode.CONTINUE  # type: ignore[misc]


class TestPeriodicTriggerConfig:
    """Tests for PeriodicTriggerConfig dataclass."""

    def test_construction_with_interval(self) -> None:
        """interval field is required and set correctly."""
        cmd_ref = TriggerCommandRef(ref="test")
        config = PeriodicTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(cmd_ref,),
            interval=3600,
        )
        assert config.interval == 3600
        assert config.failure_mode == FailureMode.ABORT
        assert config.commands == (cmd_ref,)

    def test_commands_list_coerced_to_tuple(self) -> None:
        """List commands are coerced to tuple via inherited __post_init__."""
        cmd_ref = TriggerCommandRef(ref="typecheck")
        config = PeriodicTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=[cmd_ref],  # type: ignore[arg-type]
            interval=1800,
        )
        assert isinstance(config.commands, tuple)
        assert config.commands == (cmd_ref,)


class TestRunEndTriggerConfig:
    """Tests for RunEndTriggerConfig dataclass."""

    def test_construction_with_defaults(self) -> None:
        """fire_on defaults to SUCCESS."""
        config = RunEndTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(),
        )
        assert config.fire_on == FireOn.SUCCESS
        assert config.failure_mode == FailureMode.ABORT
        assert config.commands == ()
        assert config.max_retries is None

    def test_construction_with_fire_on(self) -> None:
        """fire_on can be explicitly set."""
        config = RunEndTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(),
            fire_on=FireOn.BOTH,
        )
        assert config.fire_on == FireOn.BOTH

    def test_fire_on_failure(self) -> None:
        """fire_on can be FAILURE."""
        config = RunEndTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(),
            fire_on=FireOn.FAILURE,
        )
        assert config.fire_on == FireOn.FAILURE

    def test_commands_list_coerced_to_tuple(self) -> None:
        """List commands are coerced to tuple via inherited __post_init__."""
        cmd_ref = TriggerCommandRef(ref="test")
        config = RunEndTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=[cmd_ref],  # type: ignore[arg-type]
        )
        assert isinstance(config.commands, tuple)
        assert config.commands == (cmd_ref,)

    def test_is_frozen(self) -> None:
        """RunEndTriggerConfig is immutable."""
        config = RunEndTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(),
        )
        with pytest.raises(AttributeError):
            config.fire_on = FireOn.FAILURE  # type: ignore[misc]


class TestCerberusConfig:
    """Tests for CerberusConfig dataclass."""

    def test_defaults(self) -> None:
        """CerberusConfig has sensible defaults."""
        config = CerberusConfig()
        assert config.timeout == 300
        assert config.spawn_args == ()
        assert config.wait_args == ()
        assert config.env == ()

    def test_with_all_fields(self) -> None:
        """All fields can be set on construction."""
        config = CerberusConfig(
            timeout=600,
            spawn_args=("--verbose", "--debug"),
            wait_args=("--timeout=300",),
            env=(("API_KEY", "secret"), ("DEBUG", "1")),
        )
        assert config.timeout == 600
        assert config.spawn_args == ("--verbose", "--debug")
        assert config.wait_args == ("--timeout=300",)
        assert config.env == (("API_KEY", "secret"), ("DEBUG", "1"))

    def test_is_frozen(self) -> None:
        """CerberusConfig is immutable."""
        config = CerberusConfig()
        with pytest.raises(AttributeError):
            config.timeout = 600  # type: ignore[misc]


class TestCodeReviewConfig:
    """Tests for CodeReviewConfig dataclass."""

    def test_defaults(self) -> None:
        """CodeReviewConfig has sensible defaults."""
        config = CodeReviewConfig()
        assert config.enabled is False
        assert config.reviewer_type == "cerberus"
        assert config.failure_mode == FailureMode.CONTINUE
        assert config.max_retries == 3
        assert config.finding_threshold == "none"
        assert config.baseline is None
        assert config.cerberus is None

    def test_with_all_fields(self) -> None:
        """All fields can be set on construction."""
        cerberus = CerberusConfig(timeout=600)
        config = CodeReviewConfig(
            enabled=True,
            reviewer_type="agent_sdk",
            failure_mode=FailureMode.ABORT,
            max_retries=5,
            finding_threshold="P1",
            baseline="since_run_start",
            cerberus=cerberus,
        )
        assert config.enabled is True
        assert config.reviewer_type == "agent_sdk"
        assert config.failure_mode == FailureMode.ABORT
        assert config.max_retries == 5
        assert config.finding_threshold == "P1"
        assert config.baseline == "since_run_start"
        assert config.cerberus == cerberus

    def test_reviewer_type_cerberus(self) -> None:
        """reviewer_type can be cerberus."""
        config = CodeReviewConfig(reviewer_type="cerberus")
        assert config.reviewer_type == "cerberus"

    def test_finding_threshold_values(self) -> None:
        """finding_threshold accepts all valid values."""
        for threshold in ("P0", "P1", "P2", "P3", "none"):
            config = CodeReviewConfig(finding_threshold=threshold)  # type: ignore[arg-type]
            assert config.finding_threshold == threshold

    def test_baseline_since_last_review(self) -> None:
        """baseline can be since_last_review."""
        config = CodeReviewConfig(baseline="since_last_review")
        assert config.baseline == "since_last_review"

    def test_is_frozen(self) -> None:
        """CodeReviewConfig is immutable."""
        config = CodeReviewConfig()
        with pytest.raises(AttributeError):
            config.enabled = True  # type: ignore[misc]


class TestTriggerType:
    """Tests for TriggerType enum."""

    def test_has_run_end(self) -> None:
        """TriggerType has RUN_END value."""
        assert TriggerType.RUN_END.value == "run_end"

    def test_all_values(self) -> None:
        """TriggerType has all expected values."""
        values = {t.value for t in TriggerType}
        assert values == {"epic_completion", "session_end", "periodic", "run_end"}


class TestBaseTriggerConfigCodeReview:
    """Tests for code_review field in BaseTriggerConfig."""

    def test_code_review_default_none(self) -> None:
        """code_review defaults to None."""
        config = SessionEndTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(),
        )
        assert config.code_review is None

    def test_code_review_can_be_set(self) -> None:
        """code_review can be explicitly set."""
        review_config = CodeReviewConfig(enabled=True)
        config = SessionEndTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(),
            code_review=review_config,
        )
        assert config.code_review == review_config
        assert review_config.enabled is True


class TestValidationTriggersConfig:
    """Tests for ValidationTriggersConfig dataclass."""

    def test_is_empty_all_none(self) -> None:
        """is_empty returns True when all triggers are None."""
        config = ValidationTriggersConfig()
        assert config.is_empty() is True

    def test_is_empty_with_epic_completion(self) -> None:
        """is_empty returns False when epic_completion is set."""
        epic_config = EpicCompletionTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(),
            epic_depth=EpicDepth.TOP_LEVEL,
            fire_on=FireOn.SUCCESS,
        )
        config = ValidationTriggersConfig(epic_completion=epic_config)
        assert config.is_empty() is False

    def test_is_empty_with_session_end(self) -> None:
        """is_empty returns False when session_end is set."""
        session_config = SessionEndTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(),
        )
        config = ValidationTriggersConfig(session_end=session_config)
        assert config.is_empty() is False

    def test_is_empty_with_periodic(self) -> None:
        """is_empty returns False when periodic is set."""
        periodic_config = PeriodicTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(),
            interval=600,
        )
        config = ValidationTriggersConfig(periodic=periodic_config)
        assert config.is_empty() is False

    def test_is_empty_with_run_end(self) -> None:
        """is_empty returns False when run_end is set."""
        run_end_config = RunEndTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(),
        )
        config = ValidationTriggersConfig(run_end=run_end_config)
        assert config.is_empty() is False

    def test_is_empty_with_all_triggers(self) -> None:
        """is_empty returns False when all triggers are set."""
        epic_config = EpicCompletionTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(),
            epic_depth=EpicDepth.ALL,
            fire_on=FireOn.BOTH,
        )
        session_config = SessionEndTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(),
        )
        periodic_config = PeriodicTriggerConfig(
            failure_mode=FailureMode.ABORT,
            commands=(),
            interval=300,
        )
        run_end_config = RunEndTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(),
            fire_on=FireOn.BOTH,
        )
        config = ValidationTriggersConfig(
            epic_completion=epic_config,
            session_end=session_config,
            periodic=periodic_config,
            run_end=run_end_config,
        )
        assert config.is_empty() is False

    def test_defaults_to_all_none(self) -> None:
        """Default construction has all triggers as None."""
        config = ValidationTriggersConfig()
        assert config.epic_completion is None
        assert config.session_end is None
        assert config.periodic is None
        assert config.run_end is None

    def test_is_frozen(self) -> None:
        """ValidationTriggersConfig is immutable."""
        config = ValidationTriggersConfig()
        with pytest.raises(AttributeError):
            config.epic_completion = None  # type: ignore[misc]


class TestTriggerCommandRef:
    """Tests for TriggerCommandRef dataclass."""

    def test_construction_ref_only(self) -> None:
        """TriggerCommandRef can be created with just ref."""
        cmd_ref = TriggerCommandRef(ref="test")
        assert cmd_ref.ref == "test"
        assert cmd_ref.command is None
        assert cmd_ref.timeout is None

    def test_construction_with_overrides(self) -> None:
        """TriggerCommandRef can have command and timeout overrides."""
        cmd_ref = TriggerCommandRef(ref="lint", command="ruff check .", timeout=120)
        assert cmd_ref.ref == "lint"
        assert cmd_ref.command == "ruff check ."
        assert cmd_ref.timeout == 120

    def test_is_frozen(self) -> None:
        """TriggerCommandRef is immutable."""
        cmd_ref = TriggerCommandRef(ref="test")
        with pytest.raises(AttributeError):
            cmd_ref.ref = "other"  # type: ignore[misc]


class TestGlobalValidationCommandsFieldLevelMerge:
    """Tests for field-level deep merge in global_validation_commands.

    Per T005: When project sets global_validation_commands, fields should merge
    at the command field level, not just at the command level.

    Example:
    - Preset: {test: {command: 'pytest', timeout: 300}}
    - Project: {test: {timeout: 120}}
    - Effective: {test: {command: 'pytest', timeout: 120}}

    The project's timeout overrides the preset's, but the command is inherited.
    """

    def test_preset_only_command_inherited(self) -> None:
        """Command from preset-only entry is inherited."""
        from src.domain.validation.config_merger import merge_configs

        preset = ValidationConfig(
            global_validation_commands=CommandsConfig(
                test=CommandConfig(command="pytest", timeout=300),
            ),
        )
        user = ValidationConfig()  # No global_validation_commands set
        result = merge_configs(preset, user)

        assert result.global_validation_commands.test is not None
        assert result.global_validation_commands.test.command == "pytest"
        assert result.global_validation_commands.test.timeout == 300

    def test_project_only_command_added(self) -> None:
        """Command defined only in project is added to pool."""
        from src.domain.validation.config_merger import merge_configs

        preset = ValidationConfig(
            global_validation_commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
            ),
        )
        user = ValidationConfig.from_dict(
            {
                "global_validation_commands": {
                    "lint": "ruff check .",
                }
            }
        )
        result = merge_configs(preset, user)

        # test inherited from preset
        assert result.global_validation_commands.test is not None
        assert result.global_validation_commands.test.command == "pytest"
        # lint added from project
        assert result.global_validation_commands.lint is not None
        assert result.global_validation_commands.lint.command == "ruff check ."

    def test_field_level_merge_timeout_override(self) -> None:
        """Project timeout overrides preset timeout, command inherited from preset."""
        from src.domain.validation.config_merger import merge_configs

        preset = ValidationConfig(
            global_validation_commands=CommandsConfig(
                test=CommandConfig(command="pytest", timeout=300),
            ),
        )
        # User sets only timeout for test command
        user = ValidationConfig.from_dict(
            {
                "global_validation_commands": {
                    "test": {"timeout": 120},
                }
            }
        )
        result = merge_configs(preset, user)

        # command should be inherited from preset, timeout overridden
        assert result.global_validation_commands.test is not None
        assert result.global_validation_commands.test.command == "pytest"
        assert result.global_validation_commands.test.timeout == 120

    def test_field_level_merge_command_override(self) -> None:
        """Project command overrides preset command."""
        from src.domain.validation.config_merger import merge_configs

        preset = ValidationConfig(
            global_validation_commands=CommandsConfig(
                test=CommandConfig(command="pytest", timeout=300),
            ),
        )
        # User sets only command string (shorthand)
        user = ValidationConfig.from_dict(
            {
                "global_validation_commands": {
                    "test": "pytest -v",
                }
            }
        )
        result = merge_configs(preset, user)

        # command overridden, timeout inherited from preset
        assert result.global_validation_commands.test is not None
        assert result.global_validation_commands.test.command == "pytest -v"
        assert result.global_validation_commands.test.timeout == 300

    def test_field_level_merge_both_fields(self) -> None:
        """Both fields explicitly set override both preset fields."""
        from src.domain.validation.config_merger import merge_configs

        preset = ValidationConfig(
            global_validation_commands=CommandsConfig(
                test=CommandConfig(command="pytest", timeout=300),
            ),
        )
        user = ValidationConfig.from_dict(
            {
                "global_validation_commands": {
                    "test": {"command": "pytest -v", "timeout": 120},
                }
            }
        )
        result = merge_configs(preset, user)

        # Both overridden
        assert result.global_validation_commands.test is not None
        assert result.global_validation_commands.test.command == "pytest -v"
        assert result.global_validation_commands.test.timeout == 120

    def test_empty_global_validation_commands_uses_preset_pool(self) -> None:
        """Empty global_validation_commands: {} uses preset pool."""
        from src.domain.validation.config_merger import merge_configs

        preset = ValidationConfig(
            global_validation_commands=CommandsConfig(
                test=CommandConfig(command="pytest", timeout=300),
                lint=CommandConfig(command="ruff check ."),
            ),
        )
        # Empty {} should inherit preset pool (not clear it)
        user = ValidationConfig.from_dict({"global_validation_commands": {}})
        result = merge_configs(preset, user)

        # Empty {} inherits preset commands per T005 acceptance criteria
        assert result.global_validation_commands.test is not None
        assert result.global_validation_commands.test.command == "pytest"
        assert result.global_validation_commands.test.timeout == 300
        assert result.global_validation_commands.lint is not None
        assert result.global_validation_commands.lint.command == "ruff check ."

    def test_project_adds_new_command_to_preset_pool(self) -> None:
        """Project can add new commands while inheriting preset commands."""
        from src.domain.validation.config_merger import merge_configs

        preset = ValidationConfig(
            global_validation_commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
            ),
        )
        # User adds format command, doesn't touch test
        user = ValidationConfig.from_dict(
            {
                "global_validation_commands": {
                    "format": "ruff format --check .",
                }
            }
        )
        result = merge_configs(preset, user)

        # test inherited from preset
        assert result.global_validation_commands.test is not None
        assert result.global_validation_commands.test.command == "pytest"
        # format added from project
        assert result.global_validation_commands.format is not None
        assert (
            result.global_validation_commands.format.command == "ruff format --check ."
        )

    def test_field_level_merge_no_preset_timeout(self) -> None:
        """When preset has no timeout, project timeout is used."""
        from src.domain.validation.config_merger import merge_configs

        preset = ValidationConfig(
            global_validation_commands=CommandsConfig(
                test=CommandConfig(command="pytest"),  # No timeout
            ),
        )
        user = ValidationConfig.from_dict(
            {
                "global_validation_commands": {
                    "test": {"timeout": 120},
                }
            }
        )
        result = merge_configs(preset, user)

        # command inherited, timeout from project
        assert result.global_validation_commands.test is not None
        assert result.global_validation_commands.test.command == "pytest"
        assert result.global_validation_commands.test.timeout == 120

    def test_field_level_merge_null_timeout_clears_preset_timeout(self) -> None:
        """Explicit null timeout clears preset timeout."""
        from src.domain.validation.config_merger import merge_configs

        preset = ValidationConfig(
            global_validation_commands=CommandsConfig(
                test=CommandConfig(command="pytest", timeout=300),
            ),
        )
        user = ValidationConfig.from_dict(
            {
                "global_validation_commands": {
                    "test": {"timeout": None},
                }
            }
        )
        result = merge_configs(preset, user)

        # command inherited, timeout explicitly cleared to None
        assert result.global_validation_commands.test is not None
        assert result.global_validation_commands.test.command == "pytest"
        assert result.global_validation_commands.test.timeout is None

    def test_partial_override_without_preset_raises_error(self) -> None:
        """Partial override for command not in preset raises ConfigError."""
        from src.domain.validation.config_merger import merge_configs

        preset = ValidationConfig(
            global_validation_commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
            ),
        )
        # lint not defined in preset, but user tries to set only timeout
        user = ValidationConfig.from_dict(
            {
                "global_validation_commands": {
                    "lint": {"timeout": 120},
                }
            }
        )
        with pytest.raises(
            ConfigError,
            match=r"Cannot specify partial override for 'lint'.*no preset command",
        ):
            merge_configs(preset, user)

    def test_custom_commands_inherited_from_preset(self) -> None:
        """Custom commands from preset are inherited in merged result."""
        from src.domain.validation.config_merger import merge_configs

        preset = ValidationConfig(
            global_validation_commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
                custom_commands={
                    "security": CustomCommandConfig(command="bandit -r src/"),
                },
            ),
        )
        # User overrides test timeout, shouldn't lose preset custom commands
        user = ValidationConfig.from_dict(
            {
                "global_validation_commands": {
                    "test": {"timeout": 120},
                }
            }
        )
        result = merge_configs(preset, user)

        # Built-in command merged
        assert result.global_validation_commands.test is not None
        assert result.global_validation_commands.test.command == "pytest"
        assert result.global_validation_commands.test.timeout == 120
        # Custom command inherited from preset
        assert "security" in result.global_validation_commands.custom_commands
        assert (
            result.global_validation_commands.custom_commands["security"].command
            == "bandit -r src/"
        )

    def test_custom_commands_user_overrides_preset(self) -> None:
        """User custom commands override preset custom commands by name."""
        from src.domain.validation.config_merger import merge_configs

        preset = ValidationConfig(
            global_validation_commands=CommandsConfig(
                custom_commands={
                    "security": CustomCommandConfig(command="bandit -r src/"),
                    "docs": CustomCommandConfig(command="mkdocs build"),
                },
            ),
        )
        user = ValidationConfig.from_dict(
            {
                "global_validation_commands": {
                    "security": "bandit -r src/ --skip B101",  # override
                }
            }
        )
        result = merge_configs(preset, user)

        # security overridden by user
        assert "security" in result.global_validation_commands.custom_commands
        assert (
            result.global_validation_commands.custom_commands["security"].command
            == "bandit -r src/ --skip B101"
        )
        # docs inherited from preset
        assert "docs" in result.global_validation_commands.custom_commands
        assert (
            result.global_validation_commands.custom_commands["docs"].command
            == "mkdocs build"
        )

    def test_partial_config_without_preset_raises_error(self) -> None:
        """Partial config (timeout-only) without preset raises ConfigError."""
        from src.domain.validation.config_merger import merge_configs

        # User has partial config but no preset to inherit from
        user = ValidationConfig.from_dict(
            {
                "global_validation_commands": {
                    "test": {"timeout": 120},  # No command - partial config
                }
            }
        )
        with pytest.raises(
            ConfigError,
            match=r"Command 'test' in global_validation_commands has no 'command' field",
        ):
            merge_configs(None, user)

    def test_partial_custom_command_without_preset_raises_error(self) -> None:
        """Partial custom command (timeout-only) without preset raises ConfigError."""
        from src.domain.validation.config_merger import merge_configs

        # User has partial custom command config but no preset to inherit from
        # Create CustomCommandConfig directly with empty command to bypass from_value validation
        user = ValidationConfig(
            global_validation_commands=CommandsConfig(
                custom_commands={
                    "my_check": CustomCommandConfig(command="", timeout=120),
                },
            ),
        )
        with pytest.raises(
            ConfigError,
            match=r"Custom command 'my_check' in global_validation_commands has no 'command' field",
        ):
            merge_configs(None, user)

    def test_programmatic_config_overrides_preset(self) -> None:
        """Programmatic (dataclass) configs override preset values."""
        from src.domain.validation.config_merger import merge_configs

        preset = ValidationConfig(
            global_validation_commands=CommandsConfig(
                test=CommandConfig(command="pytest", timeout=300),
            ),
        )
        # Programmatic config - no _fields_set tracking
        user = ValidationConfig(
            global_validation_commands=CommandsConfig(
                test=CommandConfig(command="pytest -v"),  # Override command
            ),
        )
        result = merge_configs(preset, user)

        # Programmatic command value should override preset
        assert result.global_validation_commands.test is not None
        assert result.global_validation_commands.test.command == "pytest -v"
        # Timeout inherited from preset since user's is None
        assert result.global_validation_commands.test.timeout == 300

    def test_clear_customs_clears_preset_custom_commands(self) -> None:
        """_clear_customs: true clears all custom commands including preset's."""
        from src.domain.validation.config_merger import merge_configs

        preset = ValidationConfig(
            global_validation_commands=CommandsConfig(
                test=CommandConfig(command="pytest"),
                custom_commands={
                    "security": CustomCommandConfig(command="bandit -r src/"),
                },
            ),
        )
        # User sets _clear_customs: true
        user = ValidationConfig.from_dict(
            {
                "global_validation_commands": {
                    "_clear_customs": True,
                }
            }
        )
        result = merge_configs(preset, user)

        # Built-in command inherited
        assert result.global_validation_commands.test is not None
        assert result.global_validation_commands.test.command == "pytest"
        # Custom commands cleared
        assert len(result.global_validation_commands.custom_commands) == 0


class TestValidationTriggersConfigParsing:
    """Tests for parsing validation_triggers from YAML via config_loader."""

    def test_parse_none_returns_none(self) -> None:
        """None input returns None."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        result = _parse_validation_triggers(None)
        assert result is None

    def test_parse_empty_dict_returns_empty_config(self) -> None:
        """Empty dict returns ValidationTriggersConfig with all None."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        result = _parse_validation_triggers({})
        assert result is not None
        assert result.epic_completion is None
        assert result.session_end is None
        assert result.periodic is None

    def test_parse_all_triggers_with_all_fields(self) -> None:
        """All trigger types parse correctly with all fields."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        data = {
            "epic_completion": {
                "epic_depth": "all",
                "fire_on": "both",
                "failure_mode": "remediate",
                "max_retries": 2,
                "commands": [{"ref": "test", "timeout": 120}],
            },
            "session_end": {
                "failure_mode": "continue",
                "commands": [{"ref": "lint"}],
            },
            "periodic": {
                "interval": 10,
                "failure_mode": "abort",
                "commands": [],
            },
        }

        result = _parse_validation_triggers(data)
        assert result is not None

        # epic_completion
        assert result.epic_completion is not None
        assert result.epic_completion.epic_depth == EpicDepth.ALL
        assert result.epic_completion.fire_on == FireOn.BOTH
        assert result.epic_completion.failure_mode == FailureMode.REMEDIATE
        assert result.epic_completion.max_retries == 2
        assert len(result.epic_completion.commands) == 1
        assert result.epic_completion.commands[0].ref == "test"
        assert result.epic_completion.commands[0].timeout == 120

        # session_end
        assert result.session_end is not None
        assert result.session_end.failure_mode == FailureMode.CONTINUE
        assert result.session_end.commands[0].ref == "lint"

        # periodic
        assert result.periodic is not None
        assert result.periodic.interval == 10
        assert result.periodic.failure_mode == FailureMode.ABORT
        assert result.periodic.commands == ()

    def test_parse_enum_failure_mode_values(self) -> None:
        """All FailureMode enum values parse correctly."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        for mode in ["abort", "continue", "remediate"]:
            data = {
                "session_end": {
                    "failure_mode": mode,
                    "commands": [],
                    **({"max_retries": 1} if mode == "remediate" else {}),
                }
            }
            result = _parse_validation_triggers(data)
            assert result is not None
            assert result.session_end is not None
            assert result.session_end.failure_mode == FailureMode(mode)

    def test_parse_enum_epic_depth_values(self) -> None:
        """All EpicDepth enum values parse correctly."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        for depth in ["top_level", "all"]:
            data = {
                "epic_completion": {
                    "epic_depth": depth,
                    "fire_on": "success",
                    "failure_mode": "abort",
                    "commands": [],
                }
            }
            result = _parse_validation_triggers(data)
            assert result is not None
            assert result.epic_completion is not None
            assert result.epic_completion.epic_depth == EpicDepth(depth)

    def test_parse_enum_fire_on_values(self) -> None:
        """All FireOn enum values parse correctly."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        for fire_on in ["success", "failure", "both"]:
            data = {
                "epic_completion": {
                    "epic_depth": "top_level",
                    "fire_on": fire_on,
                    "failure_mode": "abort",
                    "commands": [],
                }
            }
            result = _parse_validation_triggers(data)
            assert result is not None
            assert result.epic_completion is not None
            assert result.epic_completion.fire_on == FireOn(fire_on)

    def test_parse_trigger_command_ref_with_overrides(self) -> None:
        """TriggerCommandRef parses with command and timeout overrides."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        data = {
            "session_end": {
                "failure_mode": "abort",
                "commands": [
                    {"ref": "test", "command": "pytest -x", "timeout": 300},
                ],
            }
        }

        result = _parse_validation_triggers(data)
        assert result is not None
        assert result.session_end is not None
        cmd = result.session_end.commands[0]
        assert cmd.ref == "test"
        assert cmd.command == "pytest -x"
        assert cmd.timeout == 300

    def test_parse_trigger_command_ref_ref_only(self) -> None:
        """TriggerCommandRef parses with just ref."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        data = {
            "session_end": {
                "failure_mode": "abort",
                "commands": [{"ref": "lint"}],
            }
        }

        result = _parse_validation_triggers(data)
        assert result is not None
        assert result.session_end is not None
        cmd = result.session_end.commands[0]
        assert cmd.ref == "lint"
        assert cmd.command is None
        assert cmd.timeout is None

    def test_parse_trigger_command_ref_string_shorthand(self) -> None:
        """Command can be a plain string (ref shorthand)."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        data = {
            "session_end": {
                "failure_mode": "abort",
                "commands": ["test", "lint"],
            }
        }

        result = _parse_validation_triggers(data)
        assert result is not None
        assert result.session_end is not None
        assert len(result.session_end.commands) == 2
        assert result.session_end.commands[0].ref == "test"
        assert result.session_end.commands[1].ref == "lint"

    def test_parse_empty_commands_list_is_valid(self) -> None:
        """Empty commands list is valid (parsed as empty tuple)."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        data = {
            "session_end": {
                "failure_mode": "abort",
                "commands": [],
            }
        }

        result = _parse_validation_triggers(data)
        assert result is not None
        assert result.session_end is not None
        assert result.session_end.commands == ()

    def test_parse_failure_mode_missing_raises_error(self) -> None:
        """Missing failure_mode raises ConfigError."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        data = {
            "session_end": {
                "commands": [],
            }
        }

        with pytest.raises(
            ConfigError, match="failure_mode required for trigger session_end"
        ):
            _parse_validation_triggers(data)

    def test_parse_max_retries_required_when_remediate(self) -> None:
        """max_retries required when failure_mode=remediate."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        data = {
            "session_end": {
                "failure_mode": "remediate",
                "commands": [],
            }
        }

        with pytest.raises(
            ConfigError,
            match="max_retries required when failure_mode=remediate for trigger session_end",
        ):
            _parse_validation_triggers(data)

    def test_parse_epic_completion_missing_epic_depth_raises_error(self) -> None:
        """epic_completion without epic_depth raises ConfigError."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        data = {
            "epic_completion": {
                "fire_on": "success",
                "failure_mode": "abort",
                "commands": [],
            }
        }

        with pytest.raises(
            ConfigError, match="epic_depth required for trigger epic_completion"
        ):
            _parse_validation_triggers(data)

    def test_parse_epic_completion_missing_fire_on_raises_error(self) -> None:
        """epic_completion without fire_on raises ConfigError."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        data = {
            "epic_completion": {
                "epic_depth": "top_level",
                "failure_mode": "abort",
                "commands": [],
            }
        }

        with pytest.raises(
            ConfigError, match="fire_on required for trigger epic_completion"
        ):
            _parse_validation_triggers(data)

    def test_parse_periodic_missing_interval_raises_error(self) -> None:
        """periodic without interval raises ConfigError."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        data = {
            "periodic": {
                "failure_mode": "abort",
                "commands": [],
            }
        }

        with pytest.raises(ConfigError, match="interval required for trigger periodic"):
            _parse_validation_triggers(data)

    def test_parse_invalid_failure_mode_raises_error(self) -> None:
        """Invalid failure_mode string raises ConfigError."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        data = {
            "session_end": {
                "failure_mode": "invalid_mode",
                "commands": [],
            }
        }

        with pytest.raises(ConfigError, match="Invalid failure_mode 'invalid_mode'"):
            _parse_validation_triggers(data)

    def test_parse_invalid_epic_depth_raises_error(self) -> None:
        """Invalid epic_depth string raises ConfigError."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        data = {
            "epic_completion": {
                "epic_depth": "invalid",
                "fire_on": "success",
                "failure_mode": "abort",
                "commands": [],
            }
        }

        with pytest.raises(ConfigError, match="Invalid epic_depth 'invalid'"):
            _parse_validation_triggers(data)

    def test_parse_invalid_fire_on_raises_error(self) -> None:
        """Invalid fire_on string raises ConfigError."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        data = {
            "epic_completion": {
                "epic_depth": "top_level",
                "fire_on": "invalid",
                "failure_mode": "abort",
                "commands": [],
            }
        }

        with pytest.raises(ConfigError, match="Invalid fire_on 'invalid'"):
            _parse_validation_triggers(data)

    def test_parse_trigger_not_dict_raises_error(self) -> None:
        """Trigger value that's not a dict raises ConfigError."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        data = {"session_end": "not_a_dict"}

        with pytest.raises(ConfigError, match="session_end must be an object"):
            _parse_validation_triggers(data)

    def test_parse_commands_not_list_raises_error(self) -> None:
        """commands that's not a list raises ConfigError."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        data = {
            "session_end": {
                "failure_mode": "abort",
                "commands": "not_a_list",
            }
        }

        with pytest.raises(ConfigError, match="'commands' must be a list"):
            _parse_validation_triggers(data)

    def test_parse_command_ref_missing_ref_raises_error(self) -> None:
        """Command dict without 'ref' raises ConfigError."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        data = {
            "session_end": {
                "failure_mode": "abort",
                "commands": [{"command": "pytest"}],
            }
        }

        with pytest.raises(
            ConfigError, match="'ref' is required for command 0 in trigger session_end"
        ):
            _parse_validation_triggers(data)

    def test_parse_command_invalid_type_raises_error(self) -> None:
        """Command that's not string or dict raises ConfigError."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        data = {
            "session_end": {
                "failure_mode": "abort",
                "commands": [123],
            }
        }

        with pytest.raises(
            ConfigError, match="Command 0 in trigger session_end must be"
        ):
            _parse_validation_triggers(data)

    def test_parse_validation_triggers_not_dict_raises_error(self) -> None:
        """validation_triggers must be a dict, not a list or other type."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        with pytest.raises(ConfigError, match="validation_triggers must be an object"):
            _parse_validation_triggers([])  # type: ignore[arg-type]

        with pytest.raises(ConfigError, match="validation_triggers must be an object"):
            _parse_validation_triggers("not a dict")  # type: ignore[arg-type]

    def test_parse_unknown_trigger_key_raises_error(self) -> None:
        """Unknown keys under validation_triggers raise ConfigError."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        data = {
            "session_endd": {  # typo
                "failure_mode": "abort",
            }
        }

        with pytest.raises(
            ConfigError, match="Unknown trigger 'session_endd' in validation_triggers"
        ):
            _parse_validation_triggers(data)

    def test_parse_unknown_field_in_trigger_raises_error(self) -> None:
        """Unknown fields within a trigger config raise ConfigError."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        data = {
            "session_end": {
                "failure_mode": "abort",
                "unknown_field": "value",
            }
        }

        with pytest.raises(
            ConfigError, match="Unknown field 'unknown_field' in trigger session_end"
        ):
            _parse_validation_triggers(data)

    def test_parse_unknown_field_in_command_ref_raises_error(self) -> None:
        """Unknown fields in command ref raise ConfigError with context."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        data = {
            "session_end": {
                "failure_mode": "abort",
                "commands": [{"ref": "lint", "unknown": "value"}],
            }
        }

        with pytest.raises(
            ConfigError,
            match="Unknown field 'unknown' in command 0 of trigger session_end",
        ):
            _parse_validation_triggers(data)

    def test_parse_boolean_timeout_raises_error(self) -> None:
        """Boolean values for timeout are rejected (bool is subclass of int)."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        data = {
            "session_end": {
                "failure_mode": "abort",
                "commands": [{"ref": "lint", "timeout": True}],
            }
        }

        with pytest.raises(ConfigError, match="'timeout' must be an integer"):
            _parse_validation_triggers(data)

    def test_parse_boolean_max_retries_raises_error(self) -> None:
        """Boolean values for max_retries are rejected."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        data = {
            "session_end": {
                "failure_mode": "remediate",
                "max_retries": True,
            }
        }

        with pytest.raises(ConfigError, match="max_retries must be an integer"):
            _parse_validation_triggers(data)

    def test_parse_boolean_interval_raises_error(self) -> None:
        """Boolean values for interval are rejected."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        data = {
            "periodic": {
                "failure_mode": "abort",
                "interval": False,
            }
        }

        with pytest.raises(ConfigError, match="interval must be an integer"):
            _parse_validation_triggers(data)

    def test_parse_empty_ref_string_raises_error(self) -> None:
        """Empty ref string in command raises ConfigError."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        data = {
            "session_end": {
                "failure_mode": "abort",
                "commands": [{"ref": "  "}],
            }
        }

        with pytest.raises(
            ConfigError,
            match="'ref' cannot be empty in command 0 of trigger session_end",
        ):
            _parse_validation_triggers(data)

    def test_parse_empty_command_string_shorthand_raises_error(self) -> None:
        """Empty string shorthand for command raises ConfigError."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        data = {
            "session_end": {
                "failure_mode": "abort",
                "commands": [""],
            }
        }

        with pytest.raises(
            ConfigError, match="Command 0 in trigger session_end cannot be an empty"
        ):
            _parse_validation_triggers(data)

    def test_parse_empty_command_override_raises_error(self) -> None:
        """Empty command override string raises ConfigError."""
        from src.domain.validation.config_loader import _parse_validation_triggers

        data = {
            "session_end": {
                "failure_mode": "abort",
                "commands": [{"ref": "lint", "command": "  "}],
            }
        }

        with pytest.raises(
            ConfigError,
            match="'command' cannot be empty in command 0 of trigger session_end",
        ):
            _parse_validation_triggers(data)


class TestMigrationValidation:
    """Tests for migration validation that catches deprecated config patterns."""

    def test_validate_every_present_raises_error(
        self, tmp_path: "pathlib.Path"
    ) -> None:
        """validate_every field raises ConfigError with deprecation message."""
        from src.domain.validation.config_loader import load_config

        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text(
            """
preset: python-uv
validate_every: 10
"""
        )

        with pytest.raises(ConfigError, match=r"validate_every is deprecated"):
            load_config(tmp_path)

    def test_validate_every_error_mentions_periodic_trigger(
        self, tmp_path: "pathlib.Path"
    ) -> None:
        """validate_every error suggests using validation_triggers.periodic."""
        from src.domain.validation.config_loader import load_config

        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text(
            """
preset: python-uv
validate_every: 10
"""
        )

        with pytest.raises(
            ConfigError, match=r"validation_triggers\.periodic.*interval"
        ):
            load_config(tmp_path)

    def test_validate_every_error_includes_migration_url(
        self, tmp_path: "pathlib.Path"
    ) -> None:
        """validate_every error includes migration guide URL."""
        from src.domain.validation.config_loader import load_config

        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text(
            """
preset: python-uv
validate_every: 10
"""
        )

        with pytest.raises(
            ConfigError, match=r"https://docs\.mala\.ai/migration/validation-triggers"
        ):
            load_config(tmp_path)

    def test_global_validation_commands_without_triggers_raises_error(
        self, tmp_path: "pathlib.Path"
    ) -> None:
        """Non-empty global_validation_commands without validation_triggers raises error."""
        from src.domain.validation.spec import build_validation_spec

        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text(
            """
preset: python-uv
global_validation_commands:
  test: "pytest -x"
"""
        )

        with pytest.raises(
            ConfigError, match=r"validation_triggers required when global_validation"
        ):
            build_validation_spec(tmp_path)

    def test_global_validation_commands_error_contains_example_snippet(
        self, tmp_path: "pathlib.Path"
    ) -> None:
        """global_validation_commands error includes concrete example snippet."""
        from src.domain.validation.spec import build_validation_spec

        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text(
            """
preset: python-uv
global_validation_commands:
  test: "pytest"
"""
        )

        with pytest.raises(ConfigError, match=r"Example:") as exc_info:
            build_validation_spec(tmp_path)

        # Verify the example includes key elements
        error_msg = str(exc_info.value)
        assert "validation_triggers:" in error_msg
        assert "session_end:" in error_msg
        assert "failure_mode:" in error_msg
        assert "commands:" in error_msg
        assert "ref:" in error_msg

    def test_global_validation_commands_error_includes_migration_url(
        self, tmp_path: "pathlib.Path"
    ) -> None:
        """global_validation_commands error includes migration guide URL."""
        from src.domain.validation.spec import build_validation_spec

        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text(
            """
preset: python-uv
global_validation_commands:
  test: "pytest"
"""
        )

        with pytest.raises(
            ConfigError, match=r"https://docs\.mala\.ai/migration/validation-triggers"
        ):
            build_validation_spec(tmp_path)

    def test_empty_global_validation_commands_without_triggers_ok(
        self, tmp_path: "pathlib.Path"
    ) -> None:
        """Empty global_validation_commands without validation_triggers is valid."""
        from src.domain.validation.spec import build_validation_spec

        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text(
            """
preset: python-uv
global_validation_commands: {}
"""
        )

        # Should not raise - empty global_validation_commands is valid
        spec = build_validation_spec(tmp_path)
        assert spec is not None

    def test_global_validation_commands_with_triggers_ok(
        self, tmp_path: "pathlib.Path"
    ) -> None:
        """global_validation_commands WITH validation_triggers is valid."""
        from src.domain.validation.spec import build_validation_spec

        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text(
            """
preset: python-uv
global_validation_commands:
  test: "pytest -x"
validation_triggers:
  session_end:
    failure_mode: abort
    commands:
      - ref: test
"""
        )

        # Should not raise
        spec = build_validation_spec(tmp_path)
        assert spec is not None

    def test_empty_validation_triggers_dict_ok(self, tmp_path: "pathlib.Path") -> None:
        """Empty validation_triggers {} is valid (explicit opt-out)."""
        from src.domain.validation.spec import build_validation_spec

        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text(
            """
preset: python-uv
global_validation_commands:
  test: "pytest -x"
validation_triggers: {}
"""
        )

        # Should not raise - empty triggers is explicit opt-out
        spec = build_validation_spec(tmp_path)
        assert spec is not None

    def test_global_custom_commands_without_triggers_raises_error(
        self, tmp_path: "pathlib.Path"
    ) -> None:
        """Custom commands in global_validation_commands also require triggers."""
        from src.domain.validation.spec import build_validation_spec

        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text(
            """
preset: python-uv
global_validation_commands:
  my_check: "echo custom"
"""
        )

        with pytest.raises(
            ConfigError, match=r"validation_triggers required when global_validation"
        ):
            build_validation_spec(tmp_path)

    def test_preset_global_commands_without_triggers_raises_error(
        self, tmp_path: "pathlib.Path"
    ) -> None:
        """Preset-provided global_validation_commands without triggers raises error.

        This tests that migration validation runs on the merged config, catching
        cases where a preset contributes global_validation_commands but the user
        config omits validation_triggers.
        """
        from src.domain.validation.config import CommandConfig, CommandsConfig
        from src.domain.validation.config_loader import _validate_migration
        from src.domain.validation.config_merger import merge_configs

        # Create a preset config with global_validation_commands
        preset_config = ValidationConfig(
            preset=None,
            commands=CommandsConfig(test=CommandConfig(command="pytest")),
            global_validation_commands=CommandsConfig(
                test=CommandConfig(command="pytest --strict")
            ),
        )

        # Create a user config without validation_triggers
        user_config = ValidationConfig(
            preset="fake-preset",  # Would be set in real scenario
            commands=CommandsConfig(),
        )

        # Merge configs (simulating what build_validation_spec does)
        merged = merge_configs(preset_config, user_config)

        # Migration validation should fail on merged config
        with pytest.raises(
            ConfigError, match=r"validation_triggers required when global_validation"
        ):
            _validate_migration(merged)

        # Verify error includes example
        try:
            _validate_migration(merged)
        except ConfigError as e:
            error_msg = str(e)
            assert "Example:" in error_msg

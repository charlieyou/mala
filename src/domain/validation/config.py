"""Configuration dataclasses for mala.yaml validation configuration.

This module provides the data structures for the language-agnostic configuration
system. Users define their validation commands in mala.yaml, which is parsed into
these frozen dataclasses.

These dataclasses represent the deserialized configuration. They are immutable
(frozen) to ensure configuration cannot be accidentally modified after loading.

Key types:
- CommandConfig: A single command with optional timeout
- YamlCoverageConfig: Coverage settings (named to avoid collision with spec.CoverageConfig)
- CommandsConfig: All validation commands (setup, test, lint, format, typecheck, e2e)
- ValidationConfig: Top-level configuration with preset, commands, coverage, patterns
- PromptValidationCommands: Validation commands formatted for prompt templates
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import cast

# Regex for valid custom command names: starts with letter or underscore,
# followed by letters, digits, underscores, or hyphens
CUSTOM_COMMAND_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


class CustomOverrideMode(Enum):
    """Mode for how custom commands in run_level_commands override per_issue_commands.

    When run_level_commands defines custom commands, this mode determines how they
    combine with custom commands from per_issue_commands.
    """

    INHERIT = "inherit"  # Keep per_issue customs unchanged (no run_level customs)
    CLEAR = "clear"  # Remove all per_issue customs (no customs at run-level)
    REPLACE = "replace"  # Replace per_issue customs with run_level customs
    ADDITIVE = "additive"  # Merge run_level customs into per_issue customs


class ConfigError(Exception):
    """Base exception for configuration errors.

    Raised when mala.yaml has invalid content, missing required fields,
    or other configuration problems.
    """

    pass


class PresetNotFoundError(ConfigError):
    """Raised when a referenced preset does not exist.

    Example:
        >>> raise PresetNotFoundError("unknown-preset", ["python-uv", "go", "rust"])
        PresetNotFoundError: Unknown preset 'unknown-preset'. Available presets: python-uv, go, rust
    """

    def __init__(self, preset_name: str, available: list[str] | None = None) -> None:
        self.preset_name = preset_name
        self.available = available or []
        if self.available:
            available_str = ", ".join(sorted(self.available))
            message = (
                f"Unknown preset '{preset_name}'. Available presets: {available_str}"
            )
        else:
            message = f"Unknown preset '{preset_name}'"
        super().__init__(message)


@dataclass(frozen=True)
class CommandConfig:
    """Configuration for a single validation command.

    Commands can be specified in two forms in mala.yaml:
    - String shorthand: "uv run pytest"
    - Object form: {command: "uv run pytest", timeout: 300}

    The factory method `from_value` handles both forms.

    Attributes:
        command: The shell command string to execute.
        timeout: Optional timeout in seconds. None means use system default.
    """

    command: str
    timeout: int | None = None

    @classmethod
    def from_value(cls, value: str | dict[str, object]) -> CommandConfig:
        """Create CommandConfig from YAML value (string or dict).

        Args:
            value: Either a command string or a dict with 'command' and
                optional 'timeout' keys.

        Returns:
            CommandConfig instance.

        Raises:
            ConfigError: If value is neither string nor valid dict.

        Examples:
            >>> CommandConfig.from_value("uv run pytest")
            CommandConfig(command='uv run pytest', timeout=None)

            >>> CommandConfig.from_value({"command": "pytest", "timeout": 60})
            CommandConfig(command='pytest', timeout=60)
        """
        if isinstance(value, str):
            if not value:
                raise ConfigError(
                    "Command cannot be empty string. Use null to disable."
                )
            return cls(command=value)

        if isinstance(value, dict):
            command = value.get("command")
            if not isinstance(command, str):
                raise ConfigError("Command object must have a 'command' string field")
            if not command:
                raise ConfigError(
                    "Command cannot be empty string. Use null to disable."
                )

            timeout = value.get("timeout")
            if timeout is not None:
                # Reject booleans explicitly (bool is subclass of int)
                if isinstance(timeout, bool) or not isinstance(timeout, int):
                    raise ConfigError(
                        f"Command timeout must be an integer, got {type(timeout).__name__}"
                    )

            return cls(command=command, timeout=cast("int | None", timeout))

        raise ConfigError(
            f"Command must be a string or object, got {type(value).__name__}"
        )


@dataclass(frozen=True)
class CustomCommandConfig:
    """Configuration for a custom validation command.

    Custom commands allow users to define additional validation steps
    beyond the standard commands (lint, format, test, etc.).

    Attributes:
        command: The shell command string to execute.
        timeout: Optional timeout in seconds. None means use system default.
        allow_fail: If True, command failure won't fail the validation.
    """

    command: str
    timeout: int | None = None
    allow_fail: bool = False

    @classmethod
    def from_value(
        cls, name: str, value: str | dict[str, object] | None
    ) -> CustomCommandConfig:
        """Create CustomCommandConfig from YAML value (string or dict).

        Args:
            name: The custom command name (used as key in custom_commands dict).
            value: Either a command string or a dict with 'command' and
                optional 'timeout', 'allow_fail' keys.

        Returns:
            CustomCommandConfig instance.

        Raises:
            ConfigError: If name is invalid, value is null/invalid, or
                object has unknown keys.

        Examples:
            >>> CustomCommandConfig.from_value("my_check", "uvx cmd")
            CustomCommandConfig(command='uvx cmd', timeout=120, allow_fail=False)

            >>> CustomCommandConfig.from_value("slow_check", {"command": "cmd", "timeout": 300})
            CustomCommandConfig(command='cmd', timeout=300, allow_fail=False)
        """
        # Validate command name
        if not CUSTOM_COMMAND_NAME_PATTERN.match(name):
            raise ConfigError(
                f"Invalid custom command name '{name}'. "
                "Names must start with a letter or underscore, "
                "followed by letters, digits, underscores, or hyphens."
            )

        # Reject null values
        if value is None:
            raise ConfigError(
                f"Custom command '{name}' cannot be null. "
                "To disable, use run-level override to disable."
            )

        # String shorthand
        if isinstance(value, str):
            if not value or not value.strip():
                raise ConfigError(
                    f"Custom command '{name}' cannot be empty. "
                    "Provide a command string."
                )
            return cls(command=value, timeout=120, allow_fail=False)

        # Object form
        if isinstance(value, dict):
            known_keys = {"command", "timeout", "allow_fail"}
            unknown_keys = set(value.keys()) - known_keys
            if unknown_keys:
                # Use str() to handle mixed-type keys (e.g., int keys in YAML)
                first_unknown = sorted(str(k) for k in unknown_keys)[0]
                raise ConfigError(
                    f"Unknown key '{first_unknown}' in custom command '{name}'. "
                    f"Allowed keys: {', '.join(sorted(known_keys))}"
                )

            command = value.get("command")
            if not isinstance(command, str):
                raise ConfigError(
                    f"Custom command '{name}' object must have a 'command' string field"
                )
            if not command or not command.strip():
                raise ConfigError(
                    f"Custom command '{name}' cannot be empty. "
                    "Provide a command string."
                )

            timeout = value.get("timeout")
            if timeout is None:
                timeout = 120
            else:
                # Reject booleans explicitly (bool is subclass of int)
                if isinstance(timeout, bool) or not isinstance(timeout, int):
                    raise ConfigError(
                        f"Custom command '{name}' timeout must be an integer, "
                        f"got {type(timeout).__name__}"
                    )

            allow_fail = value.get("allow_fail", False)
            if not isinstance(allow_fail, bool):
                raise ConfigError(
                    f"Custom command '{name}' allow_fail must be a boolean, "
                    f"got {type(allow_fail).__name__}"
                )

            return cls(
                command=command,
                timeout=cast("int | None", timeout),
                allow_fail=allow_fail,
            )

        raise ConfigError(
            f"Custom command '{name}' must be a string or object, "
            f"got {type(value).__name__}"
        )


@dataclass(frozen=True)
class YamlCoverageConfig:
    """Coverage configuration from mala.yaml.

    Named YamlCoverageConfig to avoid collision with the existing CoverageConfig
    in spec.py which is used by the validation runner.

    When the coverage section is present in mala.yaml, all required fields
    (format, file, threshold) must be specified. The coverage section can be
    omitted entirely to disable coverage, or set to null.

    Attributes:
        command: Optional separate command to run tests with coverage.
            If omitted, uses the test command from commands section.
        format: Coverage report format. MVP supports only "xml" (Cobertura).
        file: Path to coverage report file, relative to repo root.
        threshold: Minimum coverage percentage (0-100).
        timeout: Optional timeout in seconds for the coverage command.
    """

    format: str
    file: str
    threshold: float
    command: str | None = None
    timeout: int | None = None

    def __post_init__(self) -> None:
        """Validate coverage configuration after initialization."""
        # Validate format
        supported_formats = ("xml",)
        if self.format not in supported_formats:
            raise ConfigError(
                f"Unsupported coverage format '{self.format}'. "
                f"Supported formats: {', '.join(supported_formats)}"
            )

        # Validate threshold range
        if not 0 <= self.threshold <= 100:
            raise ConfigError(
                f"Coverage threshold must be between 0 and 100, got {self.threshold}"
            )

        # Validate file is not empty
        if not self.file:
            raise ConfigError("Coverage file path cannot be empty")

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> YamlCoverageConfig:
        """Create YamlCoverageConfig from a YAML dict.

        Args:
            data: Dict with 'format', 'file', 'threshold', and optionally
                'command' and 'timeout' keys.

        Returns:
            YamlCoverageConfig instance.

        Raises:
            ConfigError: If required fields are missing or invalid.
        """
        # Validate required fields
        required = ("format", "file", "threshold")
        missing = [f for f in required if f not in data or data[f] is None]
        if missing:
            raise ConfigError(
                f"Coverage enabled but missing required field(s): {', '.join(missing)}"
            )

        format_val = data["format"]
        if not isinstance(format_val, str):
            raise ConfigError(
                f"Coverage format must be a string, got {type(format_val).__name__}"
            )

        file_val = data["file"]
        if not isinstance(file_val, str):
            raise ConfigError(
                f"Coverage file must be a string, got {type(file_val).__name__}"
            )
        if not file_val:
            raise ConfigError("Coverage file path cannot be empty")

        threshold_val = data["threshold"]
        # Reject booleans explicitly (bool is subclass of int)
        if isinstance(threshold_val, bool) or not isinstance(
            threshold_val, int | float
        ):
            raise ConfigError(
                f"Coverage threshold must be a number, got {type(threshold_val).__name__}"
            )

        command_val = data.get("command")
        if command_val is not None and not isinstance(command_val, str):
            raise ConfigError(
                f"Coverage command must be a string, got {type(command_val).__name__}"
            )
        if command_val == "":
            raise ConfigError(
                "Coverage command cannot be empty string. "
                "Omit the field to use test command."
            )

        timeout_val = data.get("timeout")
        if timeout_val is not None:
            # Reject booleans explicitly (bool is subclass of int)
            if isinstance(timeout_val, bool) or not isinstance(timeout_val, int):
                raise ConfigError(
                    f"Coverage timeout must be an integer, got {type(timeout_val).__name__}"
                )

        return cls(
            format=format_val,
            file=file_val,
            threshold=float(threshold_val),
            command=command_val,
            timeout=cast("int | None", timeout_val),
        )


@dataclass(frozen=True)
class CommandsConfig:
    """Configuration for all validation commands.

    All fields are optional. When a field is None, it means the command
    is not defined (may inherit from preset or be skipped). Commands
    can be explicitly disabled by setting them to None even if a preset
    defines them.

    Attributes:
        setup: Environment setup command (e.g., "uv sync", "npm install").
        test: Test runner command (e.g., "uv run pytest", "go test ./...").
        lint: Linter command (e.g., "uvx ruff check .", "golangci-lint run").
        format: Formatter check command (e.g., "uvx ruff format --check .").
        typecheck: Type checker command (e.g., "uvx ty check", "tsc --noEmit").
        e2e: End-to-end test command (e.g., "uv run pytest -m e2e").
        custom_commands: Dictionary of custom validation commands (name -> config).
        custom_override_mode: How run_level custom commands combine with per_issue.
        _fields_set: Set of field names that were explicitly provided in source.
            Used by the merger to distinguish "not set" from "explicitly null".
    """

    setup: CommandConfig | None = None
    test: CommandConfig | None = None
    lint: CommandConfig | None = None
    format: CommandConfig | None = None
    typecheck: CommandConfig | None = None
    e2e: CommandConfig | None = None
    custom_commands: dict[str, CustomCommandConfig] = field(default_factory=dict)
    custom_override_mode: CustomOverrideMode = CustomOverrideMode.INHERIT
    _fields_set: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_dict(
        cls, data: dict[str, object] | None, *, is_run_level: bool = False
    ) -> CommandsConfig:
        """Create CommandsConfig from a YAML dict.

        Args:
            data: Dict with optional command fields. Each can be a string,
                command object, or null.
            is_run_level: If True, this is for run_level_commands section.
                Enables _clear_customs validation and +prefix custom command parsing.

        Returns:
            CommandsConfig instance.

        Raises:
            ConfigError: If a command value is invalid.
        """
        if data is None:
            return cls()

        valid_kinds = ("setup", "test", "lint", "format", "typecheck", "e2e")

        # Check for +_clear_customs (error: reserved key cannot be prefixed)
        if "+_clear_customs" in data:
            raise ConfigError("+_clear_customs is not allowed")

        # Handle _clear_customs reserved key
        custom_override_mode = CustomOverrideMode.INHERIT
        if "_clear_customs" in data:
            clear_value = data["_clear_customs"]
            # Value must be exactly boolean true
            if clear_value is not True:
                raise ConfigError(
                    "_clear_customs must be true (got "
                    f"{type(clear_value).__name__}: {clear_value!r})"
                )
            # Only allowed at run-level
            if not is_run_level:
                raise ConfigError("_clear_customs is only valid in run_level_commands")
            # Check for custom command keys (non-builtin, non-reserved)
            custom_keys = set(data.keys()) - set(valid_kinds) - {"_clear_customs"}
            if custom_keys:
                raise ConfigError(
                    "_clear_customs cannot be combined with custom commands"
                )
            custom_override_mode = CustomOverrideMode.CLEAR

        # Identify custom command keys (preserving YAML order via data iteration)
        reserved_keys = set(valid_kinds) | {"_clear_customs"}
        custom_keys_ordered: list[str] = []
        for k in data:
            if k in reserved_keys:
                continue
            if not isinstance(k, str):
                raise ConfigError(
                    f"Command key must be a string, got {type(k).__name__}: {k!r}"
                )
            custom_keys_ordered.append(k)

        # Parse custom commands from unknown keys
        custom_commands: dict[str, CustomCommandConfig] = {}
        if custom_keys_ordered:
            # Count +prefixed vs unprefixed for mode detection
            plus_prefixed = [k for k in custom_keys_ordered if k.startswith("+")]
            unprefixed = [k for k in custom_keys_ordered if not k.startswith("+")]

            if is_run_level:
                # At run-level: detect mode based on prefix pattern
                if plus_prefixed and unprefixed:
                    # Mixed prefixes not allowed
                    raise ConfigError(
                        "Cannot mix +prefixed and unprefixed custom commands "
                        "in run_level_commands. Use all +prefix for additive mode "
                        "or all unprefixed for replace mode."
                    )
                if plus_prefixed:
                    # All +prefixed: ADDITIVE mode
                    custom_override_mode = CustomOverrideMode.ADDITIVE
                    for key in plus_prefixed:
                        # Strip + prefix for the stored name
                        name = key[1:]
                        # Validate stripped name against built-in collision
                        if name in valid_kinds:
                            raise ConfigError(
                                f"Custom command '+{name}' conflicts with built-in "
                                f"command '{name}'. Use a different name."
                            )
                        value = data[key]
                        custom_commands[name] = CustomCommandConfig.from_value(
                            name, cast("str | dict[str, object] | None", value)
                        )
                else:
                    # All unprefixed: REPLACE mode
                    custom_override_mode = CustomOverrideMode.REPLACE
                    for key in unprefixed:
                        value = data[key]
                        custom_commands[key] = CustomCommandConfig.from_value(
                            key, cast("str | dict[str, object] | None", value)
                        )
            else:
                # At repo-level: +prefix not allowed
                if plus_prefixed:
                    first_prefixed = plus_prefixed[0]
                    raise ConfigError(
                        f"Plus-prefixed custom command '{first_prefixed}' is only "
                        "allowed in run_level_commands. Remove the '+' prefix or "
                        "move to run_level_commands section."
                    )
                # Parse unprefixed as custom commands (order preserved)
                for key in unprefixed:
                    value = data[key]
                    custom_commands[key] = CustomCommandConfig.from_value(
                        key, cast("str | dict[str, object] | None", value)
                    )

        # Track which fields were explicitly present in the source dict
        fields_set: set[str] = set()

        def parse_command(key: str) -> CommandConfig | None:
            if key in data:
                fields_set.add(key)
            value = data.get(key)
            if value is None:
                return None
            if value == "":
                raise ConfigError(
                    f"Command '{key}' cannot be empty string. Use null to disable."
                )
            # After the above checks, value is str or dict (from YAML)
            return CommandConfig.from_value(cast("str | dict[str, object]", value))

        return cls(
            setup=parse_command("setup"),
            test=parse_command("test"),
            lint=parse_command("lint"),
            format=parse_command("format"),
            typecheck=parse_command("typecheck"),
            e2e=parse_command("e2e"),
            custom_commands=custom_commands,
            custom_override_mode=custom_override_mode,
            _fields_set=frozenset(fields_set),
        )


@dataclass(frozen=True)
class ValidationConfig:
    """Top-level configuration from mala.yaml.

    This dataclass represents the fully parsed mala.yaml configuration.
    It is frozen (immutable) after creation.

    Attributes:
        preset: Optional preset name to extend (e.g., "python-uv", "go").
        commands: Command definitions. May be partially filled if extending preset.
        run_level_commands: Optional overrides for run-level validation commands.
        coverage: Coverage configuration. None means coverage is disabled.
        custom_commands: User-defined custom validation commands. Execution order
            follows YAML key order (Python 3.7+ dict insertion order is preserved).
        run_level_custom_commands: Optional run-level override for custom commands.
            None means use repo-level custom_commands. Empty dict {} disables all
            custom commands. Non-empty dict fully replaces (not merges) repo-level.
        code_patterns: Glob patterns for code files that trigger validation.
        config_files: Tool config files that invalidate lint/format cache.
        setup_files: Lock/dependency files that invalidate setup cache.
        _fields_set: Set of field names that were explicitly provided in source.
            Used by the merger to distinguish "not set" from "explicitly set".
    """

    commands: CommandsConfig = field(default_factory=CommandsConfig)
    run_level_commands: CommandsConfig = field(default_factory=CommandsConfig)
    preset: str | None = None
    coverage: YamlCoverageConfig | None = None
    custom_commands: dict[str, CustomCommandConfig] = field(default_factory=dict)
    run_level_custom_commands: dict[str, CustomCommandConfig] | None = None
    code_patterns: tuple[str, ...] = field(default_factory=tuple)
    config_files: tuple[str, ...] = field(default_factory=tuple)
    setup_files: tuple[str, ...] = field(default_factory=tuple)
    _fields_set: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Normalize list fields to tuples for immutability."""
        # Convert any list fields to tuples
        if isinstance(self.code_patterns, list):
            object.__setattr__(self, "code_patterns", tuple(self.code_patterns))
        if isinstance(self.config_files, list):
            object.__setattr__(self, "config_files", tuple(self.config_files))
        if isinstance(self.setup_files, list):
            object.__setattr__(self, "setup_files", tuple(self.setup_files))

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ValidationConfig:
        """Create ValidationConfig from a parsed YAML dict.

        Args:
            data: Dict representing the parsed mala.yaml content.

        Returns:
            ValidationConfig instance.

        Raises:
            ConfigError: If any field is invalid.
        """
        # Check for deprecated top-level keys and provide migration hints
        if "custom_commands" in data:
            raise ConfigError(
                "'custom_commands' is no longer supported. Move custom commands "
                "into the 'commands' section as additional keys (e.g., "
                "'import_lint: \"uvx import-linter\"'). "
                "See plans/2026-01-06-inline-custom-commands-plan.md"
            )
        if "run_level_custom_commands" in data:
            raise ConfigError(
                "'run_level_custom_commands' is no longer supported. Move custom "
                "commands into the 'run_level_commands' section. Use unprefixed "
                "keys to replace repo-level customs, or '+' prefix for additive "
                "merge. See plans/2026-01-06-inline-custom-commands-plan.md"
            )

        # Track which fields were explicitly present in the source dict
        fields_set: set[str] = set()

        # Parse preset
        preset = data.get("preset")
        if "preset" in data:
            fields_set.add("preset")
        if preset is not None and not isinstance(preset, str):
            raise ConfigError(f"preset must be a string, got {type(preset).__name__}")

        # Parse commands
        commands_data = data.get("commands")
        if "commands" in data:
            fields_set.add("commands")
        if commands_data is not None and not isinstance(commands_data, dict):
            raise ConfigError(
                f"commands must be an object, got {type(commands_data).__name__}"
            )
        # commands_data is either None or dict at this point
        commands = CommandsConfig.from_dict(
            cast("dict[str, object] | None", commands_data)
        )

        # Parse run-level commands
        run_level_commands_data = data.get("run_level_commands")
        if "run_level_commands" in data:
            fields_set.add("run_level_commands")
        if run_level_commands_data is not None and not isinstance(
            run_level_commands_data, dict
        ):
            raise ConfigError(
                "run_level_commands must be an object, got "
                f"{type(run_level_commands_data).__name__}"
            )
        run_level_commands = CommandsConfig.from_dict(
            cast("dict[str, object] | None", run_level_commands_data),
            is_run_level=True,
        )

        # Parse coverage - track if explicitly present (even if null)
        if "coverage" in data:
            fields_set.add("coverage")
        coverage_data = data.get("coverage")
        coverage: YamlCoverageConfig | None = None
        if coverage_data is not None:
            if not isinstance(coverage_data, dict):
                raise ConfigError(
                    f"coverage must be an object, got {type(coverage_data).__name__}"
                )
            # coverage_data is confirmed to be a dict here
            coverage = YamlCoverageConfig.from_dict(
                cast("dict[str, object]", coverage_data)
            )

        # Parse list fields - track if explicitly present (even if empty list)
        def parse_string_list(key: str) -> tuple[str, ...]:
            if key in data:
                fields_set.add(key)
            value = data.get(key)
            if value is None:
                return ()
            if not isinstance(value, list):
                raise ConfigError(f"{key} must be a list, got {type(value).__name__}")
            result: list[str] = []
            for i, item in enumerate(value):
                if not isinstance(item, str):
                    raise ConfigError(
                        f"{key}[{i}] must be a string, got {type(item).__name__}"
                    )
                result.append(item)
            return tuple(result)

        code_patterns = parse_string_list("code_patterns")
        config_files = parse_string_list("config_files")
        setup_files = parse_string_list("setup_files")

        # Parse custom_commands - track if explicitly present (even if empty)
        custom_commands: dict[str, CustomCommandConfig] = {}
        if "custom_commands" in data:
            fields_set.add("custom_commands")
            custom_commands_data = data.get("custom_commands")
            if custom_commands_data is not None:
                if not isinstance(custom_commands_data, dict):
                    raise ConfigError(
                        "custom_commands must be an object, "
                        f"got {type(custom_commands_data).__name__}"
                    )
                for name, value in custom_commands_data.items():
                    if not isinstance(name, str):
                        raise ConfigError(
                            f"custom_commands key must be a string, got {type(name).__name__}"
                        )
                    custom_commands[name] = CustomCommandConfig.from_value(
                        name, cast("str | dict[str, object] | None", value)
                    )

        # Parse run_level_custom_commands - track in _fields_set only when dict
        # None/null = not set (use repo-level), {} = explicitly empty (disable all)
        # Only add to fields_set when value is a dict (including {}), so null
        # truly behaves like omission at the metadata level
        run_level_custom_commands: dict[str, CustomCommandConfig] | None = None
        if "run_level_custom_commands" in data:
            rlcc_data = data.get("run_level_custom_commands")
            if rlcc_data is not None:
                if not isinstance(rlcc_data, dict):
                    raise ConfigError(
                        "run_level_custom_commands must be an object, "
                        f"got {type(rlcc_data).__name__}"
                    )
                # Only track in fields_set when value is a dict (not null)
                fields_set.add("run_level_custom_commands")
                run_level_custom_commands = {}
                for name, value in rlcc_data.items():
                    if not isinstance(name, str):
                        raise ConfigError(
                            f"run_level_custom_commands key must be a string, "
                            f"got {type(name).__name__}"
                        )
                    run_level_custom_commands[name] = CustomCommandConfig.from_value(
                        name, cast("str | dict[str, object] | None", value)
                    )

        return cls(
            preset=preset,
            commands=commands,
            run_level_commands=run_level_commands,
            coverage=coverage,
            custom_commands=custom_commands,
            run_level_custom_commands=run_level_custom_commands,
            code_patterns=code_patterns,
            config_files=config_files,
            setup_files=setup_files,
            _fields_set=frozenset(fields_set),
        )

    def has_any_command(self) -> bool:
        """Check if at least one command is defined.

        Returns:
            True if at least one command is defined, False otherwise.
        """
        return any(
            [
                self.commands.setup,
                self.commands.test,
                self.commands.lint,
                self.commands.format,
                self.commands.typecheck,
                self.commands.e2e,
                self.run_level_commands.setup,
                self.run_level_commands.test,
                self.run_level_commands.lint,
                self.run_level_commands.format,
                self.run_level_commands.typecheck,
                self.run_level_commands.e2e,
                self.custom_commands,  # Non-empty dict is truthy
                self.run_level_custom_commands,  # Non-empty dict is truthy
            ]
        )


@dataclass(frozen=True)
class PromptValidationCommands:
    """Validation commands formatted for use in prompt templates.

    This dataclass holds the actual command strings to be substituted into
    prompt templates like implementer_prompt.md and gate_followup.md.
    Commands that are not configured will use fallback messages that exit
    with code 0 to indicate the step was skipped (not falsely passing).

    Attributes:
        lint: Lint command string (e.g., "uvx ruff check ." or "golangci-lint run")
        format: Format command string (e.g., "uvx ruff format ." or "gofmt -l .")
        typecheck: Type check command string (e.g., "uvx ty check" or "go vet ./...")
        test: Test command string (e.g., "uv run pytest" or "go test ./...")
        custom_commands: Tuple of custom commands as (name, command, timeout, allow_fail) tuples.
            These are run after lint/format/typecheck but before test. Immutable to match
            frozen dataclass contract.
    """

    lint: str
    format: str
    typecheck: str
    test: str
    custom_commands: tuple[tuple[str, str, int, bool], ...]

    # Default fallback message for unconfigured commands - exits with code 0
    # since missing optional tooling is not a validation failure
    _NOT_CONFIGURED = "echo 'No {kind} command configured - skipping' >&2 && exit 0"

    @classmethod
    def from_validation_config(
        cls, config: ValidationConfig
    ) -> PromptValidationCommands:
        """Build PromptValidationCommands from a merged ValidationConfig.

        Args:
            config: The merged ValidationConfig (after preset merging).

        Returns:
            PromptValidationCommands with command strings for prompt templates.
        """
        cmds = config.commands

        # Build custom_commands tuple from config (immutable for frozen dataclass)
        # Each entry: (name, command, timeout, allow_fail)
        custom_cmds_list: list[tuple[str, str, int, bool]] = []
        for name, custom_cmd in config.custom_commands.items():
            # Use default timeout of 120 if not specified
            timeout = custom_cmd.timeout if custom_cmd.timeout is not None else 120
            custom_cmds_list.append(
                (name, custom_cmd.command, timeout, custom_cmd.allow_fail)
            )

        return cls(
            lint=cmds.lint.command
            if cmds.lint
            else cls._NOT_CONFIGURED.format(kind="lint"),
            format=cmds.format.command
            if cmds.format
            else cls._NOT_CONFIGURED.format(kind="format"),
            typecheck=cmds.typecheck.command
            if cmds.typecheck
            else cls._NOT_CONFIGURED.format(kind="typecheck"),
            test=cmds.test.command
            if cmds.test
            else cls._NOT_CONFIGURED.format(kind="test"),
            custom_commands=tuple(custom_cmds_list),
        )

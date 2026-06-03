"""Configuration dataclasses for mala.yaml validation configuration.

This module provides the data structures for the language-agnostic configuration
system. Users define their validation commands in mala.yaml, which is parsed into
these frozen dataclasses.

These dataclasses represent the deserialized configuration. They are immutable
(frozen) to ensure configuration cannot be accidentally modified after loading.

The pure type definitions live here so that ``config_parser`` (the YAML/dict
parser) can import them without creating a circular dependency. All parsing
logic — including helpers for the heavyweight sections — lives in
``config_parser``.

Key types:
- CommandConfig: A single command with optional timeout
- YamlCoverageConfig: Coverage settings (named to avoid collision with spec.CoverageConfig)
- CommandsConfig: All validation commands (setup, test, lint, format, typecheck, e2e)
- ValidationConfig: Top-level configuration with preset, commands, coverage, patterns
- PromptValidationCommands: Validation commands formatted for prompt templates
- TriggerType, FailureMode, EpicDepth, FireOn: Enums for trigger configuration
- ValidationTriggersConfig: Configuration for validation triggers
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, cast

from src.core.constants import (
    DEFAULT_AGENT_SDK_REVIEW_TIMEOUT_SECONDS,
    DEFAULT_CERBERUS_REVIEW_TIMEOUT_SECONDS,
)

# Regex for valid custom command names: starts with letter or underscore,
# followed by letters, digits, underscores, or hyphens
CUSTOM_COMMAND_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")

# Built-in command names supported by CommandsConfig.
# Used for validation in multiple places (config parsing, preset validation,
# evidence_check reference validation).
BUILTIN_COMMAND_NAMES: frozenset[str] = frozenset(
    {"setup", "build", "test", "lint", "format", "typecheck", "e2e"}
)


class TriggerType(Enum):
    """Type of validation trigger.

    Determines when validation commands are executed.
    """

    EPIC_COMPLETION = "epic_completion"  # When an epic (story/milestone) completes
    SESSION_END = "session_end"  # When a session ends
    PERIODIC = "periodic"  # At regular time intervals
    RUN_END = "run_end"  # When mala run completes


class FailureMode(Enum):
    """How to handle validation failures.

    Determines the behavior when a triggered validation fails.
    """

    ABORT = "abort"  # Stop immediately on failure
    CONTINUE = "continue"  # Continue despite failure
    REMEDIATE = "remediate"  # Attempt to fix the failure


class EpicDepth(Enum):
    """Which epics trigger validation.

    Controls whether only top-level epics or all nested epics trigger validation.
    """

    TOP_LEVEL = "top_level"  # Only top-level (root) epics
    ALL = "all"  # All epics including nested


class FireOn(Enum):
    """When to fire the trigger based on completion status.

    Determines whether to trigger on success, failure, or both.
    """

    SUCCESS = "success"  # Fire only on successful completion
    FAILURE = "failure"  # Fire only on failed completion
    BOTH = "both"  # Fire on both success and failure


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
        _fields_set: Fields explicitly set by user (for merge tracking).
    """

    command: str
    timeout: int | None = None
    _fields_set: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_value(
        cls,
        value: str | dict[str, object],
        *,
        requires_command: bool = True,
    ) -> CommandConfig:
        """Create CommandConfig from YAML value (string or dict).

        Args:
            value: Either a command string or a dict with 'command' and
                optional 'timeout' keys.
            requires_command: If True (default), 'command' field is required.
                Set to False to allow timeout-only configs in programmatic use.

        Returns:
            CommandConfig instance.

        Raises:
            ConfigError: If value is neither string nor valid dict, or if
                requires_command=True and command is missing.

        Examples:
            >>> CommandConfig.from_value("uv run pytest")
            CommandConfig(command='uv run pytest', timeout=None, _fields_set=frozenset({'command'}))

            >>> CommandConfig.from_value({"command": "pytest", "timeout": 60})
            CommandConfig(command='pytest', timeout=60, _fields_set=frozenset({'command', 'timeout'}))

            >>> CommandConfig.from_value({"timeout": 120}, requires_command=False)
            CommandConfig(command='', timeout=120, _fields_set=frozenset({'timeout'}))
        """
        if isinstance(value, str):
            if not value:
                raise ConfigError(
                    "Command cannot be empty string. Use null to disable."
                )
            return cls(command=value, _fields_set=frozenset({"command"}))

        if isinstance(value, dict):
            fields_set: set[str] = set()

            command = value.get("command")
            if command is not None:
                if not isinstance(command, str):
                    raise ConfigError(
                        "Command object must have a 'command' string field"
                    )
                if not command:
                    raise ConfigError(
                        "Command cannot be empty string. Use null to disable."
                    )
                fields_set.add("command")
            elif requires_command:
                raise ConfigError("Command object must have a 'command' string field")
            else:
                command = ""  # Sentinel for partial config

            timeout = value.get("timeout")
            # Track timeout even if explicitly set to None
            if "timeout" in value:
                fields_set.add("timeout")
            if timeout is not None:
                # Reject booleans explicitly (bool is subclass of int)
                if isinstance(timeout, bool) or not isinstance(timeout, int):
                    raise ConfigError(
                        f"Command timeout must be an integer, got {type(timeout).__name__}"
                    )

            return cls(
                command=cast("str", command),
                timeout=cast("int | None", timeout),
                _fields_set=frozenset(fields_set),
            )

        raise ConfigError(
            f"Command must be a string or object, got {type(value).__name__}"
        )


@dataclass(frozen=True)
class TriggerCommandRef:
    """Reference to a validation command for use in triggers.

    Allows triggers to reference existing commands (by name) with optional
    overrides for command string and timeout.

    Attributes:
        ref: Name of the command to reference (e.g., "test", "lint", "my_custom").
        command: Optional override for the command string.
        timeout: Optional override for the timeout in seconds.
    """

    ref: str
    command: str | None = None
    timeout: int | None = None


@dataclass(frozen=True)
class CerberusConfig:
    """Cerberus-specific settings.

    Attributes:
        timeout: Timeout in seconds for cerberus operations.
        spawn_args: Additional arguments for spawn command.
        wait_args: Additional arguments for wait command.
        env: Environment variables as key-value pairs.
    """

    timeout: int = DEFAULT_CERBERUS_REVIEW_TIMEOUT_SECONDS
    spawn_args: tuple[str, ...] = field(default_factory=tuple)
    wait_args: tuple[str, ...] = field(default_factory=tuple)
    env: tuple[tuple[str, str], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CodeReviewConfig:
    """Unified code review configuration.

    Attributes:
        enabled: Whether code review is enabled.
        reviewer_type: Type of reviewer to use.
        failure_mode: How to handle review failures.
        max_retries: Maximum number of retry attempts.
        finding_threshold: Minimum severity to report findings.
        baseline: What code to include in review.
        cerberus: Cerberus-specific settings if reviewer_type is "cerberus".
        agent_sdk_timeout: Timeout in seconds for Agent SDK reviews (default: 300).
        agent_sdk_model: Model for Agent SDK reviewer ('sonnet', 'opus', 'haiku').
        track_review_issues: Whether to create beads issues for P2/P3 review findings.
    """

    enabled: bool = False
    reviewer_type: Literal["cerberus", "agent_sdk"] = "cerberus"
    failure_mode: FailureMode = FailureMode.CONTINUE
    max_retries: int = 3
    finding_threshold: Literal["P0", "P1", "P2", "P3", "none"] = "none"
    baseline: Literal["since_run_start", "since_last_review"] | None = None
    cerberus: CerberusConfig | None = None
    agent_sdk_timeout: int = DEFAULT_AGENT_SDK_REVIEW_TIMEOUT_SECONDS
    agent_sdk_model: Literal["sonnet", "opus", "haiku"] = "opus"
    track_review_issues: bool = True


@dataclass(frozen=True)
class VerificationRetryPolicy:
    """Per-category retry limits for epic verification failures.

    Different failure categories have different characteristics:
    - Timeout errors are often transient and worth aggressive retrying
    - Execution errors may be environmental issues worth moderate retrying
    - Parse errors are often deterministic and unlikely to succeed on retry

    Attributes:
        timeout_retries: Max retries for timeout errors (default: 3).
        execution_retries: Max retries for execution errors (default: 2).
        parse_retries: Max retries for parse errors (default: 1).
    """

    timeout_retries: int = 3
    execution_retries: int = 2
    parse_retries: int = 1


@dataclass(frozen=True)
class EpicVerifierConfig:
    """Configuration for epic verification reviewer choice.

    This configuration controls which backend is used for epic verification
    during epic completion triggers. Similar to CodeReviewConfig's structure,
    this allows selecting between different verification implementations.

    Note: Named EpicVerifierConfig (not EpicVerificationConfig) to avoid collision
    with src.pipeline.epic_verification_coordinator.EpicVerificationConfig which
    handles retry configuration.

    Attributes:
        enabled: Whether epic verification is enabled.
        reviewer_type: Type of reviewer to use for epic verification.
            - "agent_sdk": Use Claude Agent SDK for verification (default).
            - "cerberus": Use Cerberus-based verification.
        timeout: Top-level timeout in seconds (default: 600).
        max_retries: Maximum retry attempts on failure (default: 3).
            This is the global fallback; per-category limits in retry_policy take precedence.
        failure_mode: How to handle verification failures.
        cerberus: Cerberus-specific settings (timeout, spawn_args, wait_args, env).
        agent_sdk_timeout: Timeout in seconds for Agent SDK verification (default: 300).
        agent_sdk_model: Model for Agent SDK verifier ('sonnet', 'opus', 'haiku').
        retry_policy: Per-category retry limits for different failure types.
    """

    enabled: bool = True
    reviewer_type: Literal["cerberus", "agent_sdk"] = "agent_sdk"
    timeout: int = 600
    max_retries: int = 3
    failure_mode: FailureMode = FailureMode.CONTINUE
    cerberus: CerberusConfig | None = None
    agent_sdk_timeout: int = DEFAULT_AGENT_SDK_REVIEW_TIMEOUT_SECONDS
    agent_sdk_model: Literal["sonnet", "opus", "haiku"] = "opus"
    retry_policy: VerificationRetryPolicy = field(
        default_factory=VerificationRetryPolicy
    )


@dataclass(frozen=True, kw_only=True)
class BaseTriggerConfig:
    """Base configuration for validation triggers.

    Common fields shared by all trigger types.

    Attributes:
        failure_mode: How to handle validation failures.
        commands: Commands to run when the trigger fires.
        max_retries: Maximum number of retry attempts on failure. None means no retries.
        code_review: Optional code review configuration for this trigger.
    """

    failure_mode: FailureMode
    commands: tuple[TriggerCommandRef, ...]
    max_retries: int | None = None
    code_review: CodeReviewConfig | None = None

    def __post_init__(self) -> None:
        """Normalize commands to tuple for immutability."""
        if isinstance(self.commands, list):
            object.__setattr__(self, "commands", tuple(self.commands))


@dataclass(frozen=True, kw_only=True)
class EpicCompletionTriggerConfig(BaseTriggerConfig):
    """Configuration for epic completion triggers.

    Triggers validation when an epic (story/milestone) completes.

    Attributes:
        epic_depth: Which epics trigger validation (top-level only or all).
        fire_on: When to fire based on completion status.
        max_epic_verification_retries: Maximum retries for epic verification loop.
            This controls how many times the orchestrator will retry epic verification
            after the first attempt fails. None means use the default (from env var or 3).
        epic_verify_lock_timeout_seconds: Timeout in seconds for acquiring epic verification
            lock. None means use the default (300 seconds).
    """

    epic_depth: EpicDepth
    fire_on: FireOn
    max_epic_verification_retries: int | None = None
    epic_verify_lock_timeout_seconds: int | None = None


@dataclass(frozen=True, kw_only=True)
class SessionEndTriggerConfig(BaseTriggerConfig):
    """Configuration for session end triggers.

    Triggers validation when a session ends.
    No additional fields beyond BaseTriggerConfig.
    """

    pass


@dataclass(frozen=True, kw_only=True)
class PeriodicTriggerConfig(BaseTriggerConfig):
    """Configuration for periodic triggers.

    Triggers validation at regular time intervals.

    Attributes:
        interval: Time between trigger activations in seconds.
    """

    interval: int


@dataclass(frozen=True, kw_only=True)
class RunEndTriggerConfig(BaseTriggerConfig):
    """Configuration for run end triggers.

    Triggers validation when mala run completes.

    Attributes:
        fire_on: When to fire based on completion status.
    """

    fire_on: FireOn = FireOn.SUCCESS


@dataclass(frozen=True)
class ValidationTriggersConfig:
    """Configuration for all validation triggers.

    Container for the different types of validation triggers.

    Attributes:
        epic_completion: Configuration for epic completion triggers.
        session_end: Configuration for session end triggers.
        periodic: Configuration for periodic triggers.
        run_end: Configuration for run end triggers.
    """

    epic_completion: EpicCompletionTriggerConfig | None = None
    session_end: SessionEndTriggerConfig | None = None
    periodic: PeriodicTriggerConfig | None = None
    run_end: RunEndTriggerConfig | None = None

    def is_empty(self) -> bool:
        """Return True if no triggers are configured."""
        return (
            self.epic_completion is None
            and self.session_end is None
            and self.periodic is None
            and self.run_end is None
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
        _fields_set: Fields explicitly set by user (for merge tracking).
    """

    command: str
    timeout: int | None = None
    allow_fail: bool = False
    _fields_set: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_value(
        cls,
        name: str,
        value: str | dict[str, object] | None,
        *,
        requires_command: bool = True,
    ) -> CustomCommandConfig:
        """Create CustomCommandConfig from YAML value (string or dict).

        Args:
            name: The custom command name (used as key in custom_commands dict).
            value: Either a command string or a dict with 'command' and
                optional 'timeout', 'allow_fail' keys.
            requires_command: If True (default), command field is required.
                If False, allows partial configs in programmatic use.

        Returns:
            CustomCommandConfig instance.

        Raises:
            ConfigError: If name is invalid, value is null/invalid, or
                object has unknown keys.

        Examples:
            >>> CustomCommandConfig.from_value("my_check", "uvx cmd")
            CustomCommandConfig(command='uvx cmd', timeout=120, allow_fail=False, ...)

            >>> CustomCommandConfig.from_value("slow_check", {"command": "cmd", "timeout": 300})
            CustomCommandConfig(command='cmd', timeout=300, allow_fail=False, ...)

            >>> CustomCommandConfig.from_value("override", {"timeout": 60}, requires_command=False)
            CustomCommandConfig(command='', timeout=60, allow_fail=False, ...)
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
                "Remove the command to disable it."
            )

        # String shorthand
        if isinstance(value, str):
            if not value or not value.strip():
                raise ConfigError(
                    f"Custom command '{name}' cannot be empty. "
                    "Provide a command string."
                )
            return cls(
                command=value,
                timeout=120,
                allow_fail=False,
                _fields_set=frozenset({"command"}),
            )

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

            fields_set: set[str] = set()

            command = value.get("command")
            if command is not None:
                if not isinstance(command, str):
                    raise ConfigError(
                        f"Custom command '{name}' object must have a 'command' string field"
                    )
                if not command or not command.strip():
                    raise ConfigError(
                        f"Custom command '{name}' cannot be empty. "
                        "Provide a command string."
                    )
                fields_set.add("command")
            elif requires_command:
                raise ConfigError(
                    f"Custom command '{name}' object must have a 'command' string field"
                )
            else:
                command = ""  # Sentinel for partial config

            timeout = value.get("timeout")
            # Track timeout when present in source dict
            if "timeout" in value:
                fields_set.add("timeout")
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
            # Track allow_fail when present in source dict
            if "allow_fail" in value:
                fields_set.add("allow_fail")
            if not isinstance(allow_fail, bool):
                raise ConfigError(
                    f"Custom command '{name}' allow_fail must be a boolean, "
                    f"got {type(allow_fail).__name__}"
                )

            return cls(
                command=cast("str", command),
                timeout=cast("int | None", timeout),
                allow_fail=allow_fail,
                _fields_set=frozenset(fields_set),
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
        _fields_set: Set of field names that were explicitly provided in source.
            Used by the merger to distinguish "not set" from "explicitly null".
    """

    setup: CommandConfig | None = None
    build: CommandConfig | None = None
    test: CommandConfig | None = None
    lint: CommandConfig | None = None
    format: CommandConfig | None = None
    typecheck: CommandConfig | None = None
    e2e: CommandConfig | None = None
    custom_commands: dict[str, CustomCommandConfig] = field(default_factory=dict)
    _fields_set: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_dict(cls, data: dict[str, object] | None) -> CommandsConfig:
        """Create CommandsConfig from a YAML dict.

        Args:
            data: Dict with optional command fields. Each can be a string,
                command object, or null.
        Returns:
            CommandsConfig instance.

        Raises:
            ConfigError: If a command value is invalid.
        """
        if data is None:
            return cls()

        # Identify custom command keys (preserving YAML order via data iteration)
        reserved_keys = BUILTIN_COMMAND_NAMES
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
            if any(key.startswith("+") for key in custom_keys_ordered):
                first_prefixed = next(
                    key for key in custom_keys_ordered if key.startswith("+")
                )
                raise ConfigError(
                    f"Plus-prefixed custom command '{first_prefixed}' is not supported. "
                    "Remove the '+' prefix."
                )
            # Parse unprefixed as custom commands (order preserved)
            for key in custom_keys_ordered:
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
            build=parse_command("build"),
            test=parse_command("test"),
            lint=parse_command("lint"),
            format=parse_command("format"),
            typecheck=parse_command("typecheck"),
            e2e=parse_command("e2e"),
            custom_commands=custom_commands,
            _fields_set=frozenset(fields_set),
        )


@dataclass(frozen=True, slots=True)
class EvidenceCheckConfig:
    """Configuration for evidence_check in mala.yaml.

    Controls which evidence types are required for validation gating.

    Attributes:
        required: Tuple of evidence type names that must be present.
    """

    required: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CodexOptionsConfig:
    """Yaml-side container for ``coder_options.codex.*``.

    Mirrors :class:`src.infra.io.config.CodexOptions` but holds optional
    fields so the orchestrator can detect "user supplied X" vs. "use the
    default" when threading values into :meth:`MalaConfig.from_env`.

    Attributes:
        approval_policy: Codex turn approval policy.
        sandbox: Codex sandbox mode.
        mcp_servers: Optional pre-merged MCP server map (Phase G3 merges).
    """

    approval_policy: (
        Literal["never", "on-request", "on-failure", "untrusted"] | None
    ) = None
    sandbox: Literal["read-only", "workspace-write", "danger-full-access"] | None = None
    mcp_servers: tuple[tuple[str, object], ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LongRunningConfig:
    """Configuration for the long-running background wait/resume mechanism.

    When a Claude agent backgrounds work via ``Bash(run_in_background=true)`` and
    yields, mala keeps the SDK connection open, waits for that task's completion
    notification on the continuous stream, and resumes the agent on the same
    client to finalize before gating once at the true end. Completion is
    event-driven (a structured task notification), so there is no polling
    interval to configure.

    Attributes:
        enabled: Whether the wait/resume mechanism is active. When False, mala
            falls back to the immediate-gate behavior.
        max_wait_seconds: Maximum time to wait for a single backgrounded task to
            complete before giving up (default: 14400 = 4 hours).
        max_resume_cycles: Maximum number of background launch/resume cycles to
            follow within one issue before stopping (default: 3).
    """

    enabled: bool = True
    max_wait_seconds: int = 14400
    max_resume_cycles: int = 3


@dataclass(frozen=True)
class LockWaitConfig:
    """Configuration for the lock park-and-resume (wait-until-free) mechanism.

    When a Claude agent is blocked on a peer-held file lock, it yields via the
    ``lock_wait`` tool. mala parks the session between turns (outside the hard
    timeout), waits until each contended path has no holder, then resumes the
    agent on the same client to re-acquire and finalize before gating once at
    the true end. mala only waits until a path is free; it does not acquire on
    the agent's behalf, so the agent re-runs ``lock_acquire`` itself on resume.

    Unlike :class:`LongRunningConfig`, freeing is detected by polling
    ``get_lock_holder`` (there is no completion notification), so a
    ``poll_interval_ms`` is configurable.

    Attributes:
        enabled: Whether the park/resume mechanism is active. When False, mala
            falls back to the in-foreground timeout-escalation behavior.
        max_wait_seconds: Maximum time to wait for the contended locks to free
            before resuming the agent with status ``unavailable`` (default:
            1800 = 30 minutes).
        max_resume_cycles: Maximum number of park/resume cycles to follow within
            one issue before stopping (default: 3).
        poll_interval_ms: How often to re-check lock holders while parked, in
            milliseconds (default: 500). Must be positive.
    """

    enabled: bool = True
    max_wait_seconds: int = 1800
    max_resume_cycles: int = 3
    poll_interval_ms: int = 500


@dataclass(frozen=True)
class ValidationConfig:
    """Top-level configuration from mala.yaml.

    This dataclass represents the fully parsed mala.yaml configuration.
    It is frozen (immutable) after creation.

    Attributes:
        preset: Optional preset name to extend (e.g., "python-uv", "go").
        commands: Command definitions. May be partially filled if extending preset.
        coverage: Coverage configuration. None means coverage is disabled.
        code_patterns: Glob patterns for code files that trigger validation.
        config_files: Tool config files that invalidate lint/format cache.
        setup_files: Lock/dependency files that invalidate setup cache.
        validation_triggers: Configuration for validation triggers. None means no triggers.
        claude_settings_sources: Sources to load Claude settings from. Valid sources are
            'local', 'project', 'user'. None means use default. Empty tuple means no sources.
        timeout_minutes: Timeout per agent in minutes. None means use default (30).
        max_idle_retries: Maximum number of idle timeout retries. None means use default (2).
        idle_timeout_seconds: Idle timeout for SDK stream in seconds.
            None means derive from agent timeout; 0 disables idle timeout.
        max_diff_size_kb: Maximum diff size in KB for epic verification.
            None means use default (no limit).
        evidence_check: Evidence check configuration. None means no evidence filtering.
        epic_verification: Epic verification configuration. Controls which backend
            (agent_sdk or cerberus) is used for epic verification.
        long_running: Configuration for the background wait/resume mechanism that
            keeps the SDK connection open while a backgrounded task finishes.
        lock_wait: Configuration for the lock park-and-resume mechanism that
            parks an agent blocked on a peer-held lock until it frees.
        _fields_set: Set of field names that were explicitly provided in source.
            Used by the merger to distinguish "not set" from "explicitly set".
    """

    commands: CommandsConfig = field(default_factory=CommandsConfig)
    preset: str | None = None
    coverage: YamlCoverageConfig | None = None
    code_patterns: tuple[str, ...] = field(default_factory=tuple)
    config_files: tuple[str, ...] = field(default_factory=tuple)
    setup_files: tuple[str, ...] = field(default_factory=tuple)
    validation_triggers: ValidationTriggersConfig | None = None
    claude_settings_sources: tuple[str, ...] | None = None
    timeout_minutes: int | None = None
    max_idle_retries: int | None = None
    idle_timeout_seconds: float | None = None
    max_diff_size_kb: int | None = None
    evidence_check: EvidenceCheckConfig | None = None
    epic_verification: EpicVerifierConfig = field(default_factory=EpicVerifierConfig)
    per_issue_review: CodeReviewConfig = field(
        default_factory=lambda: CodeReviewConfig(enabled=False)
    )
    long_running: LongRunningConfig = field(default_factory=LongRunningConfig)
    lock_wait: LockWaitConfig = field(default_factory=LockWaitConfig)
    # Coder selection (validated above as strict enum). Stored here so the
    # orchestrator can flow yaml values through to MalaConfig.from_env's
    # yaml_coder / yaml_amp_mode parameters (CLI > env > yaml > default).
    coder: Literal["claude", "amp", "codex"] | None = None
    amp_mode: Literal["smart", "rush", "deep"] | None = None
    # Mala-level model. Forwarded by the orchestrator to
    # ``MalaConfig.from_env(yaml_model=...)``; the resolver layer in
    # src/infra/io/config.py applies CLI > env > yaml > default precedence.
    model: str | None = None
    # Mala-level reasoning effort. Strict-enum validated below as
    # ``low | medium | high | xhigh | max``. Forwarded by the orchestrator
    # to ``MalaConfig.from_env(yaml_effort=...)``; the resolver layer in
    # src/infra/io/config.py applies CLI > env > yaml > default precedence.
    effort: str | None = None
    # Codex coder options (Phase B). Strict-validated at YAML parse time.
    # Flowed by the orchestrator into MalaConfig.from_env(yaml_codex_options=...).
    codex_options: CodexOptionsConfig | None = None
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
        if isinstance(self.claude_settings_sources, list):
            object.__setattr__(
                self, "claude_settings_sources", tuple(self.claude_settings_sources)
            )

    def has_any_command(self) -> bool:
        """Check if at least one command is defined.

        Returns:
            True if at least one command is defined, False otherwise.
        """
        return any(
            [
                self.commands.setup,
                self.commands.build,
                self.commands.test,
                self.commands.lint,
                self.commands.format,
                self.commands.typecheck,
                self.commands.e2e,
                self.commands.custom_commands,  # Non-empty dict is truthy
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
        lint_timeout: Timeout in seconds enforced by the canonical wrapper for
            the lint command. Falls back to 120 when not configured.
        format_timeout: Same as lint_timeout for the format command.
        typecheck_timeout: Same as lint_timeout for the typecheck command.
        test_timeout: Same as lint_timeout for the test command.
        custom_commands: Tuple of custom commands as (name, command, timeout, allow_fail) tuples.
            These are run after lint/format/typecheck but before test. Immutable to match
            frozen dataclass contract.
    """

    lint: str
    format: str
    typecheck: str
    test: str
    custom_commands: tuple[tuple[str, str, int, bool], ...]
    lint_timeout: int = 120
    format_timeout: int = 120
    typecheck_timeout: int = 120
    test_timeout: int = 120

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

        # Build custom_commands tuple from inline commands.custom_commands
        # (immutable for frozen dataclass). Each entry: (name, command, timeout, allow_fail)
        custom_cmds_list: list[tuple[str, str, int, bool]] = []
        for name, custom_cmd in cmds.custom_commands.items():
            # Use default timeout of 120 if not specified
            timeout = custom_cmd.timeout if custom_cmd.timeout is not None else 120
            custom_cmds_list.append(
                (name, custom_cmd.command, timeout, custom_cmd.allow_fail)
            )

        def _timeout(cmd: CommandConfig | None) -> int:
            if cmd is None or cmd.timeout is None:
                return 120
            return cmd.timeout

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
            lint_timeout=_timeout(cmds.lint),
            format_timeout=_timeout(cmds.format),
            typecheck_timeout=_timeout(cmds.typecheck),
            test_timeout=_timeout(cmds.test),
        )

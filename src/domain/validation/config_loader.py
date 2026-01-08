"""YAML configuration loader for mala.yaml.

This module provides functionality to load, parse, and validate the mala.yaml
configuration file. It enforces strict schema validation and provides clear
error messages for common misconfigurations.

Key functions:
- load_config: Load and validate mala.yaml from a repository path
- parse_yaml: Parse YAML content with error handling
- validate_schema: Validate against expected schema
- build_config: Convert parsed dict to ValidationConfig dataclass
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import yaml

from src.domain.validation.config import (
    ConfigError,
    EpicCompletionTriggerConfig,
    EpicDepth,
    FailureMode,
    FireOn,
    PeriodicTriggerConfig,
    SessionEndTriggerConfig,
    TriggerCommandRef,
    ValidationConfig,
    ValidationTriggersConfig,
)

if TYPE_CHECKING:
    from pathlib import Path


class ConfigMissingError(ConfigError):
    """Raised when the mala.yaml configuration file is not found.

    This is a subclass of ConfigError to allow callers to catch either:
    - ConfigMissingError: Only handle missing file case
    - ConfigError: Handle all config errors (missing, invalid syntax, etc.)

    Example:
        >>> raise ConfigMissingError(Path("/path/to/repo"))
        ConfigMissingError: mala.yaml not found in /path/to/repo.
    """

    repo_path: Path  # Explicit class-level annotation for type checkers

    def __init__(self, repo_path: Path) -> None:
        self.repo_path = repo_path
        message = f"mala.yaml not found in {repo_path}. Mala requires a configuration file to run."
        super().__init__(message)


# Fields allowed at the top level of mala.yaml
_ALLOWED_TOP_LEVEL_FIELDS = frozenset(
    {
        "preset",
        "commands",
        "global_validation_commands",
        "coverage",
        "code_patterns",
        "config_files",
        "setup_files",
        "custom_commands",
        "global_custom_commands",
        "reviewer_type",
        "agent_sdk_review_timeout",
        "agent_sdk_reviewer_model",
        "validation_triggers",
        "claude_settings_sources",
    }
)


def load_config(repo_path: Path) -> ValidationConfig:
    """Load and validate mala.yaml from the repository root.

    This is the main entry point for loading configuration. It reads the file,
    parses YAML, validates the schema, builds the config dataclass, and runs
    post-build validation.

    Args:
        repo_path: Path to the repository root directory.

    Returns:
        ValidationConfig instance with all configuration loaded.

    Raises:
        ConfigError: If the file is missing, has invalid YAML syntax,
            contains unknown fields, has invalid types, or fails validation.

    Example:
        >>> config = load_config(Path("/path/to/repo"))
        >>> print(config.preset)
        'python-uv'
    """
    config_file = repo_path / "mala.yaml"

    if not config_file.exists():
        raise ConfigMissingError(repo_path)

    try:
        content = config_file.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"Failed to read {config_file}: {e}") from e
    except UnicodeDecodeError as e:
        raise ConfigError(f"Failed to decode {config_file}: {e}") from e
    data = _parse_yaml(content)
    _validate_schema(data)
    config = _build_config(data)
    _validate_config(config)
    # Note: _validate_migration is NOT called here because it must run on the
    # effective merged config (after preset merge). Call sites that merge configs
    # (build_validation_spec, build_prompt_validation_commands) call it after merge.

    return config


def _parse_yaml(content: str) -> dict[str, Any]:
    """Parse YAML content into a dictionary.

    Args:
        content: Raw YAML string content.

    Returns:
        Parsed dictionary. Returns empty dict for empty/null YAML.

    Raises:
        ConfigError: If YAML syntax is invalid.
    """
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as e:
        # Extract useful error details from the exception
        details = str(e)
        raise ConfigError(f"Invalid YAML syntax in mala.yaml: {details}") from e

    # Handle empty file or file with only comments
    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ConfigError(
            f"mala.yaml must be a YAML mapping, got {type(data).__name__}"
        )

    return data


def _validate_schema(data: dict[str, Any]) -> None:
    """Validate the parsed YAML against the expected schema.

    This function checks for unknown fields at the top level. Field type
    validation is handled by the dataclass constructors.

    Args:
        data: Parsed YAML dictionary.

    Raises:
        ConfigError: If unknown fields are present or deprecated fields are used.
    """
    # Check for deprecated validate_every field with helpful migration message
    if "validate_every" in data:
        raise ConfigError(
            "validate_every is deprecated. Use validation_triggers.periodic with "
            "interval field. See migration guide at "
            "https://docs.mala.ai/migration/validation-triggers"
        )

    unknown_fields = set(data.keys()) - _ALLOWED_TOP_LEVEL_FIELDS
    if unknown_fields:
        # Sort for consistent error messages; convert to str to handle
        # non-string YAML keys (e.g., null, integers) without TypeError
        unknown_as_strs = sorted(str(k) for k in unknown_fields)
        first_unknown = unknown_as_strs[0]
        raise ConfigError(f"Unknown field '{first_unknown}' in mala.yaml")


_TRIGGER_COMMAND_REF_FIELDS = frozenset({"ref", "command", "timeout"})


def _parse_trigger_command_ref(
    data: dict[str, Any], trigger_name: str, cmd_index: int
) -> TriggerCommandRef:
    """Parse a single command reference from a trigger's commands list.

    Args:
        data: Dict with 'ref' (required), optional 'command' and 'timeout' overrides.
        trigger_name: Name of the trigger for error messages.
        cmd_index: Index of the command in the list for error messages.

    Returns:
        TriggerCommandRef instance.

    Raises:
        ConfigError: If 'ref' is missing, has wrong type, or unknown fields present.
    """
    # Validate unknown fields
    unknown = set(data.keys()) - _TRIGGER_COMMAND_REF_FIELDS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(
            f"Unknown field '{first}' in command {cmd_index} of trigger {trigger_name}"
        )

    if "ref" not in data:
        raise ConfigError(
            f"'ref' is required for command {cmd_index} in trigger {trigger_name}"
        )
    ref = data["ref"]
    if not isinstance(ref, str):
        raise ConfigError(
            f"'ref' must be a string in command {cmd_index} of trigger {trigger_name}, "
            f"got {type(ref).__name__}"
        )
    if not ref.strip():
        raise ConfigError(
            f"'ref' cannot be empty in command {cmd_index} of trigger {trigger_name}"
        )

    command = data.get("command")
    if command is not None:
        if not isinstance(command, str):
            raise ConfigError(
                f"'command' must be a string in command {cmd_index} of trigger {trigger_name}, "
                f"got {type(command).__name__}"
            )
        if not command.strip():
            raise ConfigError(
                f"'command' cannot be empty in command {cmd_index} of trigger {trigger_name}"
            )

    timeout = data.get("timeout")
    if timeout is not None:
        if isinstance(timeout, bool) or not isinstance(timeout, int):
            raise ConfigError(
                f"'timeout' must be an integer in command {cmd_index} of trigger {trigger_name}, "
                f"got {type(timeout).__name__}"
            )

    return TriggerCommandRef(ref=ref, command=command, timeout=timeout)


def _parse_commands_list(
    data: dict[str, Any], trigger_name: str
) -> tuple[TriggerCommandRef, ...]:
    """Parse the commands list for a trigger.

    Args:
        data: The trigger config dict.
        trigger_name: Name of the trigger for error messages.

    Returns:
        Tuple of TriggerCommandRef instances.

    Raises:
        ConfigError: If commands has wrong type.
    """
    commands_data = data.get("commands", [])
    if not isinstance(commands_data, list):
        raise ConfigError(
            f"'commands' must be a list for trigger {trigger_name}, "
            f"got {type(commands_data).__name__}"
        )

    commands: list[TriggerCommandRef] = []
    for i, cmd_data in enumerate(commands_data):
        if isinstance(cmd_data, str):
            # Allow shorthand: just the ref string
            if not cmd_data.strip():
                raise ConfigError(
                    f"Command {i} in trigger {trigger_name} cannot be an empty string"
                )
            commands.append(TriggerCommandRef(ref=cmd_data))
        elif isinstance(cmd_data, dict):
            commands.append(_parse_trigger_command_ref(cmd_data, trigger_name, i))
        else:
            raise ConfigError(
                f"Command {i} in trigger {trigger_name} must be a string or object, "
                f"got {type(cmd_data).__name__}"
            )

    return tuple(commands)


def _parse_failure_mode(data: dict[str, Any], trigger_name: str) -> FailureMode:
    """Parse and validate failure_mode for a trigger.

    Args:
        data: The trigger config dict.
        trigger_name: Name of the trigger for error messages.

    Returns:
        FailureMode enum value.

    Raises:
        ConfigError: If failure_mode is missing or invalid.
    """
    if "failure_mode" not in data:
        raise ConfigError(f"failure_mode required for trigger {trigger_name}")

    mode_str = data["failure_mode"]
    if not isinstance(mode_str, str):
        raise ConfigError(
            f"failure_mode must be a string for trigger {trigger_name}, "
            f"got {type(mode_str).__name__}"
        )

    try:
        return FailureMode(mode_str)
    except ValueError:
        valid = ", ".join(m.value for m in FailureMode)
        raise ConfigError(
            f"Invalid failure_mode '{mode_str}' for trigger {trigger_name}. "
            f"Valid values: {valid}"
        ) from None


def _parse_max_retries(
    data: dict[str, Any], trigger_name: str, failure_mode: FailureMode
) -> int | None:
    """Parse and validate max_retries for a trigger.

    Args:
        data: The trigger config dict.
        trigger_name: Name of the trigger for error messages.
        failure_mode: The parsed failure mode.

    Returns:
        max_retries value or None.

    Raises:
        ConfigError: If max_retries is required but missing, or has wrong type.
    """
    max_retries = data.get("max_retries")

    if failure_mode == FailureMode.REMEDIATE and max_retries is None:
        raise ConfigError(
            f"max_retries required when failure_mode=remediate for trigger {trigger_name}"
        )

    if max_retries is not None:
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise ConfigError(
                f"max_retries must be an integer for trigger {trigger_name}, "
                f"got {type(max_retries).__name__}"
            )

    return max_retries


_EPIC_COMPLETION_FIELDS = frozenset(
    {"failure_mode", "max_retries", "commands", "epic_depth", "fire_on"}
)


def _parse_epic_completion_trigger(
    data: dict[str, Any],
) -> EpicCompletionTriggerConfig:
    """Parse epic_completion trigger config.

    Args:
        data: The epic_completion config dict.

    Returns:
        EpicCompletionTriggerConfig instance.

    Raises:
        ConfigError: If required fields missing, invalid, or unknown fields present.
    """
    trigger_name = "epic_completion"

    # Validate unknown fields
    unknown = set(data.keys()) - _EPIC_COMPLETION_FIELDS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(f"Unknown field '{first}' in trigger {trigger_name}")

    failure_mode = _parse_failure_mode(data, trigger_name)
    max_retries = _parse_max_retries(data, trigger_name, failure_mode)
    commands = _parse_commands_list(data, trigger_name)

    # Parse epic_depth (required for epic_completion)
    if "epic_depth" not in data:
        raise ConfigError(f"epic_depth required for trigger {trigger_name}")
    depth_str = data["epic_depth"]
    if not isinstance(depth_str, str):
        raise ConfigError(
            f"epic_depth must be a string for trigger {trigger_name}, "
            f"got {type(depth_str).__name__}"
        )
    try:
        epic_depth = EpicDepth(depth_str)
    except ValueError:
        valid = ", ".join(d.value for d in EpicDepth)
        raise ConfigError(
            f"Invalid epic_depth '{depth_str}' for trigger {trigger_name}. "
            f"Valid values: {valid}"
        ) from None

    # Parse fire_on (required for epic_completion)
    if "fire_on" not in data:
        raise ConfigError(f"fire_on required for trigger {trigger_name}")
    fire_on_str = data["fire_on"]
    if not isinstance(fire_on_str, str):
        raise ConfigError(
            f"fire_on must be a string for trigger {trigger_name}, "
            f"got {type(fire_on_str).__name__}"
        )
    try:
        fire_on = FireOn(fire_on_str)
    except ValueError:
        valid = ", ".join(f.value for f in FireOn)
        raise ConfigError(
            f"Invalid fire_on '{fire_on_str}' for trigger {trigger_name}. "
            f"Valid values: {valid}"
        ) from None

    return EpicCompletionTriggerConfig(
        failure_mode=failure_mode,
        commands=commands,
        max_retries=max_retries,
        epic_depth=epic_depth,
        fire_on=fire_on,
    )


_SESSION_END_FIELDS = frozenset({"failure_mode", "max_retries", "commands"})


def _parse_session_end_trigger(data: dict[str, Any]) -> SessionEndTriggerConfig:
    """Parse session_end trigger config.

    Args:
        data: The session_end config dict.

    Returns:
        SessionEndTriggerConfig instance.

    Raises:
        ConfigError: If required fields missing, invalid, or unknown fields present.
    """
    trigger_name = "session_end"

    # Validate unknown fields
    unknown = set(data.keys()) - _SESSION_END_FIELDS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(f"Unknown field '{first}' in trigger {trigger_name}")

    failure_mode = _parse_failure_mode(data, trigger_name)
    max_retries = _parse_max_retries(data, trigger_name, failure_mode)
    commands = _parse_commands_list(data, trigger_name)

    return SessionEndTriggerConfig(
        failure_mode=failure_mode,
        commands=commands,
        max_retries=max_retries,
    )


_PERIODIC_FIELDS = frozenset({"failure_mode", "max_retries", "commands", "interval"})


def _parse_periodic_trigger(data: dict[str, Any]) -> PeriodicTriggerConfig:
    """Parse periodic trigger config.

    Args:
        data: The periodic config dict.

    Returns:
        PeriodicTriggerConfig instance.

    Raises:
        ConfigError: If required fields missing, invalid, or unknown fields present.
    """
    trigger_name = "periodic"

    # Validate unknown fields
    unknown = set(data.keys()) - _PERIODIC_FIELDS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(f"Unknown field '{first}' in trigger {trigger_name}")

    failure_mode = _parse_failure_mode(data, trigger_name)
    max_retries = _parse_max_retries(data, trigger_name, failure_mode)
    commands = _parse_commands_list(data, trigger_name)

    # Parse interval (required for periodic)
    if "interval" not in data:
        raise ConfigError(f"interval required for trigger {trigger_name}")
    interval = data["interval"]
    if isinstance(interval, bool) or not isinstance(interval, int):
        raise ConfigError(
            f"interval must be an integer for trigger {trigger_name}, "
            f"got {type(interval).__name__}"
        )

    return PeriodicTriggerConfig(
        failure_mode=failure_mode,
        commands=commands,
        max_retries=max_retries,
        interval=interval,
    )


_VALIDATION_TRIGGERS_FIELDS = frozenset({"epic_completion", "session_end", "periodic"})


def _parse_validation_triggers(
    data: dict[str, Any] | None,
) -> ValidationTriggersConfig | None:
    """Parse the validation_triggers section from mala.yaml.

    Args:
        data: The validation_triggers dict from the YAML.

    Returns:
        ValidationTriggersConfig if data is provided, None otherwise.

    Raises:
        ConfigError: If data is not a dict, has unknown keys, or any trigger is invalid.
    """
    if data is None:
        return None

    if not isinstance(data, dict):
        raise ConfigError(
            f"validation_triggers must be an object, got {type(data).__name__}"
        )

    # Validate unknown fields
    unknown = set(data.keys()) - _VALIDATION_TRIGGERS_FIELDS
    if unknown:
        first = sorted(str(k) for k in unknown)[0]
        raise ConfigError(f"Unknown trigger '{first}' in validation_triggers")

    epic_completion = None
    session_end = None
    periodic = None

    if "epic_completion" in data:
        epic_data = data["epic_completion"]
        if epic_data is not None:
            if not isinstance(epic_data, dict):
                raise ConfigError(
                    f"epic_completion must be an object, got {type(epic_data).__name__}"
                )
            epic_completion = _parse_epic_completion_trigger(epic_data)

    if "session_end" in data:
        session_data = data["session_end"]
        if session_data is not None:
            if not isinstance(session_data, dict):
                raise ConfigError(
                    f"session_end must be an object, got {type(session_data).__name__}"
                )
            session_end = _parse_session_end_trigger(session_data)

    if "periodic" in data:
        periodic_data = data["periodic"]
        if periodic_data is not None:
            if not isinstance(periodic_data, dict):
                raise ConfigError(
                    f"periodic must be an object, got {type(periodic_data).__name__}"
                )
            periodic = _parse_periodic_trigger(periodic_data)

    return ValidationTriggersConfig(
        epic_completion=epic_completion,
        session_end=session_end,
        periodic=periodic,
    )


def _build_config(data: dict[str, Any]) -> ValidationConfig:
    """Convert a validated YAML dict to a ValidationConfig dataclass.

    This function delegates to ValidationConfig.from_dict which handles
    parsing of nested structures (commands, coverage, validation_triggers, etc.).

    Args:
        data: Validated YAML dictionary.

    Returns:
        ValidationConfig instance.

    Raises:
        ConfigError: If any field has an invalid type or value.
    """
    return ValidationConfig.from_dict(data)


def _validate_config(config: ValidationConfig) -> None:
    """Perform post-build validation on the configuration.

    This validates semantic constraints that can't be checked during
    parsing, such as ensuring at least one command is defined (when
    no preset is specified).

    Args:
        config: Built ValidationConfig instance.

    Raises:
        ConfigError: If configuration is semantically invalid.
    """
    # If no preset is specified, at least one command must be defined
    if config.preset is None and not config.has_any_command():
        raise ConfigError(
            "At least one command must be defined. "
            "Specify a preset or define commands directly."
        )


def _validate_migration(config: ValidationConfig) -> None:
    """Validate that deprecated config patterns are not used.

    This function fails fast on deprecated config patterns that require
    migration. It runs on the effective merged config (after preset merge).

    Args:
        config: Built ValidationConfig instance.

    Raises:
        ConfigError: If deprecated config patterns are detected.
    """
    # Check if global_validation_commands has any commands defined.
    # Note: Top-level custom_commands is rejected in ValidationConfig.from_dict,
    # so we only need to check global_validation_commands here.
    gvc = config.global_validation_commands
    has_global_commands = any(
        (
            gvc.setup,
            gvc.test,
            gvc.lint,
            gvc.format,
            gvc.typecheck,
            gvc.e2e,
            gvc.custom_commands,
        )
    )

    # If global_validation_commands is non-empty but validation_triggers is not set,
    # require explicit validation_triggers configuration
    if has_global_commands and config.validation_triggers is None:
        raise ConfigError(
            "validation_triggers required when global_validation_commands is defined. "
            "See migration guide at https://docs.mala.ai/migration/validation-triggers\n\n"
            "Example:\n"
            "validation_triggers:\n"
            "  session_end:\n"
            "    failure_mode: remediate\n"
            "    max_retries: 3\n"
            "    commands:\n"
            "      - ref: test"
        )

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

from src.domain.validation.config_parser import parse_validation_config
from src.domain.validation.config_types import (
    BUILTIN_COMMAND_NAMES,
    ConfigError,
    TriggerType,
)

if TYPE_CHECKING:
    from pathlib import Path

    from src.domain.validation.config_types import BaseTriggerConfig, ValidationConfig


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
        "coverage",
        "code_patterns",
        "config_files",
        "setup_files",
        "validation_triggers",
        "claude_settings_sources",
        "timeout_minutes",
        "max_idle_retries",
        "idle_timeout_seconds",
        "max_diff_size_kb",
        "evidence_check",
        "epic_verification",
        "per_issue_review",
        "coder",
        "amp_mode",
        "model",
        "effort",
        "coder_options",
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
        ConfigError: If unknown fields are present.
    """
    # Check for removed global_validation_commands with helpful migration message
    if "global_validation_commands" in data:
        raise ConfigError(
            "global_validation_commands is not supported. "
            "Migrate to validation_triggers with commands:\n\n"
            "validation_triggers:\n"
            "  run_end:\n"
            "    failure_mode: continue\n"
            "    commands:\n"
            "      - ref: test\n"
            "      - ref: lint\n\n"
            "See migration guide at "
            "https://docs.mala.ai/migration/validation-triggers"
        )

    # Check for removed custom_commands top-level field
    if "custom_commands" in data:
        raise ConfigError(
            "custom_commands (top-level) is not supported. "
            "Define custom commands under the 'commands' key:\n\n"
            "commands:\n"
            "  my_custom_cmd:\n"
            "    command: 'my-tool --check'\n"
            "    timeout: 120\n"
        )

    # Check for removed validate_every field with helpful migration message
    if "validate_every" in data:
        raise ConfigError(
            "validate_every is not supported. Use validation_triggers.periodic with "
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


def _build_config(data: dict[str, Any]) -> ValidationConfig:
    """Convert a validated YAML dict to a ValidationConfig dataclass.

    Delegates to :func:`config_parser.parse_validation_config`, the dedicated
    parser entry point. ``ValidationConfig`` is a pure dataclass — all
    section-level parsing lives in ``config_parser``.

    Args:
        data: Validated YAML dictionary.

    Returns:
        ValidationConfig instance.

    Raises:
        ConfigError: If any field has an invalid type or value.
    """
    return parse_validation_config(data)


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


def _validate_trigger_command_refs(config: ValidationConfig) -> None:
    """Validate that all trigger command refs exist in the effective base pool.

    The base pool is constructed from the merged config commands,
    following the same logic as run_coordinator._build_base_pool. This validation ensures
    invalid refs are caught at startup rather than at runtime when triggers fire.

    Args:
        config: The effective merged ValidationConfig (after preset merge).

    Raises:
        ConfigError: If any trigger references a command that doesn't exist in the base pool.
    """
    triggers = config.validation_triggers
    if triggers is None:
        return  # No triggers configured

    # Build base pool following run_coordinator._build_base_pool logic:
    # - Built-in commands: from commands
    # - Custom commands: from commands.custom_commands
    base_pool: set[str] = set()
    base_cmds = config.commands

    # Add built-in commands
    for cmd_name in BUILTIN_COMMAND_NAMES:
        base_cmd = getattr(base_cmds, cmd_name, None)
        if base_cmd is not None:
            base_pool.add(cmd_name)

    # Add custom commands from commands section
    for name in base_cmds.custom_commands:
        base_pool.add(name)

    # Validate each configured trigger's command refs
    trigger_configs: list[tuple[TriggerType, BaseTriggerConfig | None]] = [
        (TriggerType.EPIC_COMPLETION, triggers.epic_completion),
        (TriggerType.SESSION_END, triggers.session_end),
        (TriggerType.PERIODIC, triggers.periodic),
        (TriggerType.RUN_END, triggers.run_end),
    ]

    for trigger_type, maybe_config in trigger_configs:
        if maybe_config is None:
            continue
        trigger_config: BaseTriggerConfig = maybe_config
        for cmd_ref in trigger_config.commands:
            if cmd_ref.ref not in base_pool:
                available = ", ".join(sorted(base_pool)) if base_pool else "(none)"
                raise ConfigError(
                    f"trigger {trigger_type.value} references unknown command "
                    f"'{cmd_ref.ref}'. Available: {available}"
                )


def validate_generated_config(data: dict[str, Any]) -> None:
    """Validate a dictionary against the config schema and semantic rules.

    This is a convenience wrapper for validating programmatically-generated
    config data (e.g., from `mala init`).

    Args:
        data: Dictionary containing mala.yaml configuration.

    Raises:
        ConfigError: If schema validation or semantic validation fails.
    """
    _validate_schema(data)
    config = parse_validation_config(data)  # raises ConfigError
    _validate_config(config)  # raises ConfigError


def dump_config_yaml(data: dict[str, Any]) -> str:
    """Dump config data to YAML string.

    Args:
        data: Dictionary containing mala.yaml configuration.

    Returns:
        YAML-formatted string.
    """
    return yaml.dump(data, default_flow_style=False, sort_keys=False)

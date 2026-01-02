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

from src.domain.validation.config import ConfigError, ValidationConfig

if TYPE_CHECKING:
    from pathlib import Path


# Fields allowed at the top level of mala.yaml
_ALLOWED_TOP_LEVEL_FIELDS = frozenset(
    {
        "preset",
        "commands",
        "coverage",
        "code_patterns",
        "config_files",
        "setup_files",
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
        raise ConfigError(
            f"mala.yaml not found in {repo_path}. "
            "Mala requires a configuration file to run."
        )

    content = config_file.read_text(encoding="utf-8")
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
    unknown_fields = set(data.keys()) - _ALLOWED_TOP_LEVEL_FIELDS
    if unknown_fields:
        # Sort for consistent error messages; convert to str to handle
        # non-string YAML keys (e.g., null, integers) without TypeError
        unknown_as_strs = sorted(str(k) for k in unknown_fields)
        first_unknown = unknown_as_strs[0]
        raise ConfigError(f"Unknown field '{first_unknown}' in mala.yaml")


def _build_config(data: dict[str, Any]) -> ValidationConfig:
    """Convert a validated YAML dict to a ValidationConfig dataclass.

    This function delegates to ValidationConfig.from_dict which handles
    parsing of nested structures (commands, coverage, etc.).

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

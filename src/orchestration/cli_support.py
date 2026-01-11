"""CLI support module for re-exporting infra utilities.

This module is part of the orchestration layer and provides CLI-safe access
to infrastructure utilities. CLI imports from this module instead of directly
from infra layer modules.

This maintains the architectural boundary: CLI -> orchestration -> infra.
"""

from __future__ import annotations

from typing import Any

# Environment configuration (from src.infra.tools.env)
from src.infra.tools.env import (
    USER_CONFIG_DIR,
    get_runs_dir,
    load_user_env,
)

# Locking utilities (from src.infra.tools.locking)
from src.infra.tools.locking import get_lock_dir

# Run metadata
from src.infra.io.log_output.run_metadata import (
    get_running_instances,
    get_running_instances_for_dir,
)

# Console utilities
from src.infra.io.log_output.console import (
    Colors,
    log,
    set_verbose,
)

# Config error for init command
from src.domain.validation.config import ConfigError


def get_init_presets() -> list[str]:
    """Get available preset names for mala init.

    Returns:
        Sorted list of preset names (e.g., ['go', 'node-npm', 'python-uv', 'rust']).
    """
    from src.domain.validation.preset_registry import PresetRegistry

    return PresetRegistry().list_presets()


def validate_init_config(data: dict[str, Any]) -> None:
    """Validate a programmatically-generated config dict.

    Args:
        data: Dictionary containing mala.yaml configuration.

    Raises:
        ConfigError: If validation fails.
    """
    from src.domain.validation.config_loader import validate_generated_config

    validate_generated_config(data)


def dump_config_yaml(data: dict[str, Any]) -> str:
    """Dump config data to YAML string.

    Args:
        data: Dictionary containing mala.yaml configuration.

    Returns:
        YAML-formatted string.
    """
    from src.domain.validation.config_loader import (
        dump_config_yaml as _dump_config_yaml,
    )

    return _dump_config_yaml(data)


def get_builtin_command_names() -> frozenset[str]:
    """Get the set of built-in command names.

    Returns:
        Frozenset of built-in command names (e.g., 'setup', 'test', 'lint', etc.).
    """
    from src.domain.validation.config import BUILTIN_COMMAND_NAMES

    return BUILTIN_COMMAND_NAMES


def get_preset_config_commands(preset_name: str) -> list[str]:
    """Get the command names defined in a preset.

    Args:
        preset_name: Name of the preset (e.g., 'python-uv', 'go').

    Returns:
        Sorted list of command names defined in the preset.

    Raises:
        PresetNotFoundError: If the preset name is not recognized.
    """
    from src.domain.validation.config import BUILTIN_COMMAND_NAMES
    from src.domain.validation.preset_registry import PresetRegistry

    registry = PresetRegistry()
    config = registry.get(preset_name)

    # Get command names that are defined (not None and have non-empty command) in the preset
    commands_config = config.commands
    result: list[str] = []
    for name in BUILTIN_COMMAND_NAMES:
        cmd = getattr(commands_config, name, None)
        # Exclude None and empty-string sentinels (from requires_command=False)
        if cmd is not None and cmd.command:
            result.append(name)
    # Sort for consistent ordering
    return sorted(result)


__all__ = [
    "USER_CONFIG_DIR",
    "Colors",
    "ConfigError",
    "dump_config_yaml",
    "get_builtin_command_names",
    "get_init_presets",
    "get_lock_dir",
    "get_preset_config_commands",
    "get_running_instances",
    "get_running_instances_for_dir",
    "get_runs_dir",
    "load_user_env",
    "log",
    "set_verbose",
    "validate_init_config",
]

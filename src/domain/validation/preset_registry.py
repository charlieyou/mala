"""Preset registry for built-in validation configuration presets.

This module provides the PresetRegistry class for loading and managing built-in
validation configuration presets. Presets are YAML files bundled with the package
and discovered via importlib.resources.

Key types:
- PresetRegistry: Class with get(), list_presets() methods for preset access
"""

from __future__ import annotations

from importlib import resources
from typing import TYPE_CHECKING, Any, ClassVar

import yaml

from src.domain.validation.config import ConfigError, PresetNotFoundError
from src.domain.validation.config_loader import _build_config

if TYPE_CHECKING:
    from src.domain.validation.config import ValidationConfig


class PresetRegistry:
    """Registry for built-in validation configuration presets.

    Presets are YAML files stored in the src/domain/validation/presets/ package
    and discovered via importlib.resources for wheel compatibility.

    Example:
        >>> registry = PresetRegistry()
        >>> config = registry.get("python-uv")
        >>> print(config.commands.test.command)
        'uv run pytest'
    """

    # Package containing preset YAML files
    _PRESETS_PACKAGE: ClassVar[str] = "src.domain.validation.presets"

    # Built-in command keys that presets are allowed to define
    _BUILTIN_COMMAND_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"setup", "format", "lint", "typecheck", "test", "e2e"}
    )

    # Mapping of preset names to YAML filenames
    _PRESET_FILES: ClassVar[dict[str, str]] = {
        "go": "go.yaml",
        "node-npm": "node-npm.yaml",
        "python-uv": "python-uv.yaml",
        "rust": "rust.yaml",
    }

    def get(self, name: str) -> ValidationConfig:
        """Load and return a preset configuration by name.

        Args:
            name: Name of the preset (e.g., "python-uv", "go").

        Returns:
            ValidationConfig instance with preset values.

        Raises:
            PresetNotFoundError: If the preset name is not recognized.

        Example:
            >>> registry = PresetRegistry()
            >>> config = registry.get("go")
            >>> print(config.commands.test.command)
            'go test ./...'
        """
        if name not in self._PRESET_FILES:
            raise PresetNotFoundError(name, self.list_presets())

        data = self._load_preset_yaml(name)

        # Presets MUST NOT define custom_commands (per spec R1)
        if "custom_commands" in data:
            raise ConfigError("presets cannot define custom_commands")

        # Validate that commands only contain built-in keys (no custom commands)
        self._validate_preset_commands(data.get("commands", {}))

        return _build_config(data)

    def list_presets(self) -> list[str]:
        """Return a sorted list of available preset names.

        Returns:
            List of preset names in alphabetical order.

        Example:
            >>> registry = PresetRegistry()
            >>> registry.list_presets()
            ['go', 'node-npm', 'python-uv', 'rust']
        """
        return sorted(self._PRESET_FILES.keys())

    def _load_preset_yaml(self, name: str) -> dict[str, Any]:
        """Load and parse a preset YAML file.

        Args:
            name: Name of the preset.

        Returns:
            Parsed YAML dictionary.

        Raises:
            PresetNotFoundError: If the preset file cannot be loaded.
        """
        filename = self._PRESET_FILES[name]
        try:
            package_files = resources.files(self._PRESETS_PACKAGE)
            preset_file = package_files.joinpath(filename)
            content = preset_file.read_text(encoding="utf-8")
            data = yaml.safe_load(content)
            return data if data is not None else {}
        except (ModuleNotFoundError, FileNotFoundError, TypeError) as e:
            raise PresetNotFoundError(name, self.list_presets()) from e

    def _validate_preset_commands(self, commands_data: dict[str, Any]) -> None:
        """Validate that preset commands only contain built-in keys.

        Presets should only define built-in commands. Any unknown key would
        become a custom command with inline custom commands, which must be
        blocked for presets.

        Args:
            commands_data: The 'commands' section from a preset YAML file.

        Raises:
            ConfigError: If commands_data contains keys not in _BUILTIN_COMMAND_KEYS.
        """
        if not commands_data:
            return

        unknown_keys = set(commands_data.keys()) - self._BUILTIN_COMMAND_KEYS
        if unknown_keys:
            sorted_keys = sorted(unknown_keys)
            raise ConfigError(
                f"Preset commands contain unknown keys: {sorted_keys}. "
                "Presets can only define built-in commands."
            )

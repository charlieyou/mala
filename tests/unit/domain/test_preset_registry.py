"""Unit tests for the preset registry module."""

from __future__ import annotations

import pytest

from src.domain.validation.config import ConfigError, PresetNotFoundError
from src.domain.validation.preset_registry import PresetRegistry


class TestPresetRegistryUnit:
    """Unit tests for PresetRegistry."""

    @pytest.fixture
    def registry(self) -> PresetRegistry:
        """Create a PresetRegistry instance."""
        return PresetRegistry()

    @pytest.mark.unit
    def test_get_unknown_preset_raises_error(self, registry: PresetRegistry) -> None:
        """get() raises PresetNotFoundError for unknown preset."""
        with pytest.raises(PresetNotFoundError) as exc_info:
            registry.get("unknown-preset")

        error = exc_info.value
        assert error.preset_name == "unknown-preset"
        assert error.available == ["go", "node-npm", "python-uv", "rust"]
        assert "Unknown preset 'unknown-preset'" in str(error)
        assert "go, node-npm, python-uv, rust" in str(error)


class TestPresetRegistryProhibitions:
    """Unit tests for PresetRegistry prohibition rules."""

    @pytest.fixture
    def registry(self) -> PresetRegistry:
        """Create a PresetRegistry instance."""
        return PresetRegistry()

    @pytest.mark.unit
    def test_preset_with_custom_commands_raises_config_error(
        self, registry: PresetRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Presets cannot define custom_commands (per spec R1)."""

        # Mock _load_preset_yaml to return a preset with custom_commands
        def mock_load_preset_yaml(self: PresetRegistry, name: str) -> dict:
            return {
                "commands": {"test": "echo test"},
                "custom_commands": {},  # Even empty dict is forbidden
            }

        monkeypatch.setattr(PresetRegistry, "_load_preset_yaml", mock_load_preset_yaml)

        with pytest.raises(ConfigError) as exc_info:
            registry.get("python-uv")

        assert "presets cannot define custom_commands" in str(exc_info.value)

    @pytest.mark.unit
    def test_preset_with_unknown_command_keys_raises_config_error(
        self, registry: PresetRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Presets cannot define unknown keys in commands section."""

        def mock_load_preset_yaml(self: PresetRegistry, name: str) -> dict:
            return {
                "commands": {
                    "test": "echo test",
                    "my_custom": "echo custom",  # Unknown key
                    "another_custom": "echo another",  # Another unknown key
                },
            }

        monkeypatch.setattr(PresetRegistry, "_load_preset_yaml", mock_load_preset_yaml)

        with pytest.raises(ConfigError) as exc_info:
            registry.get("python-uv")

        error_msg = str(exc_info.value)
        assert "Preset commands contain unknown keys:" in error_msg
        assert "another_custom" in error_msg
        assert "my_custom" in error_msg
        assert "Presets can only define built-in commands" in error_msg

    @pytest.mark.unit
    def test_preset_with_only_builtin_keys_loads_successfully(
        self, registry: PresetRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Presets with only built-in command keys load without error."""

        def mock_load_preset_yaml(self: PresetRegistry, name: str) -> dict:
            return {
                "commands": {
                    "setup": "uv sync",
                    "test": "uv run pytest",
                    "lint": "uvx ruff check .",
                    "format": "uvx ruff format --check .",
                    "typecheck": "uvx ty check",
                    "e2e": "uv run pytest -m e2e",
                },
            }

        monkeypatch.setattr(PresetRegistry, "_load_preset_yaml", mock_load_preset_yaml)

        # Should not raise
        config = registry.get("python-uv")
        assert config.commands.test is not None
        assert config.commands.test.command == "uv run pytest"

    @pytest.mark.unit
    def test_preset_with_empty_commands_loads_successfully(
        self, registry: PresetRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Presets with empty commands section load without error."""

        def mock_load_preset_yaml(self: PresetRegistry, name: str) -> dict:
            return {"commands": {}}

        monkeypatch.setattr(PresetRegistry, "_load_preset_yaml", mock_load_preset_yaml)

        # Should not raise
        config = registry.get("python-uv")
        assert config.commands.test is None

    @pytest.mark.unit
    def test_preset_without_commands_loads_successfully(
        self, registry: PresetRegistry, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Presets without commands section load without error."""

        def mock_load_preset_yaml(self: PresetRegistry, name: str) -> dict:
            return {}

        monkeypatch.setattr(PresetRegistry, "_load_preset_yaml", mock_load_preset_yaml)

        # Should not raise
        config = registry.get("python-uv")
        assert config.commands.test is None

"""Tests for the preset registry module."""

from __future__ import annotations

import pytest

from src.domain.validation.config import (
    ConfigError,
    PresetNotFoundError,
    ValidationConfig,
)
from src.domain.validation.preset_registry import PresetRegistry


# =============================================================================
# Unit Tests (mocked importlib.resources)
# =============================================================================


class TestPresetRegistryUnit:
    """Unit tests for PresetRegistry with mocked file loading."""

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


# =============================================================================
# Integration Tests (real package files)
# =============================================================================


class TestPresetRegistryIntegration:
    """Integration tests for PresetRegistry with real preset files."""

    @pytest.fixture
    def registry(self) -> PresetRegistry:
        """Create a PresetRegistry instance."""
        return PresetRegistry()

    def test_all_presets_have_valid_yaml(self, registry: PresetRegistry) -> None:
        """All preset files have valid YAML syntax."""
        for preset_name in registry.list_presets():
            config = registry.get(preset_name)
            assert isinstance(config, ValidationConfig)
            # All presets should have at least test command
            assert config.commands.test is not None

    def test_all_presets_have_code_patterns(self, registry: PresetRegistry) -> None:
        """All presets define at least one code pattern."""
        for preset_name in registry.list_presets():
            config = registry.get(preset_name)
            assert len(config.code_patterns) > 0, f"{preset_name} missing code_patterns"

    def test_all_presets_have_setup_files(self, registry: PresetRegistry) -> None:
        """All presets define at least one setup file."""
        for preset_name in registry.list_presets():
            config = registry.get(preset_name)
            assert len(config.setup_files) > 0, f"{preset_name} missing setup_files"

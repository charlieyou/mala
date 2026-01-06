"""Integration tests for the preset registry module."""

from __future__ import annotations

import pytest

from src.domain.validation.config import ValidationConfig
from src.domain.validation.preset_registry import PresetRegistry


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

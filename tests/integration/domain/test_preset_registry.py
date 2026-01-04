"""Tests for the preset registry module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.domain.validation.config import PresetNotFoundError, ValidationConfig
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
    def test_list_presets_returns_sorted_list(self, registry: PresetRegistry) -> None:
        """list_presets() returns all 4 presets in alphabetical order."""
        result = registry.list_presets()
        assert result == ["go", "node-npm", "python-uv", "rust"]

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
    def test_get_with_mocked_yaml_returns_validation_config(
        self, registry: PresetRegistry
    ) -> None:
        """get() returns ValidationConfig when YAML is mocked."""
        mock_yaml_content = """
commands:
  setup: "mock setup"
  test: "mock test"
code_patterns:
  - "*.mock"
"""
        mock_file = MagicMock()
        mock_file.read_text.return_value = mock_yaml_content

        mock_files = MagicMock()
        mock_files.joinpath.return_value = mock_file

        with patch("src.domain.validation.preset_registry.resources.files") as mock:
            mock.return_value = mock_files
            result = registry.get("python-uv")

        assert isinstance(result, ValidationConfig)
        assert result.commands.setup is not None
        assert result.commands.setup.command == "mock setup"
        assert result.commands.test is not None
        assert result.commands.test.command == "mock test"
        assert result.code_patterns == ("*.mock",)

    @pytest.mark.unit
    def test_get_handles_file_not_found(self, registry: PresetRegistry) -> None:
        """get() raises PresetNotFoundError when file loading fails."""
        with patch("src.domain.validation.preset_registry.resources.files") as mock:
            mock.side_effect = FileNotFoundError("Preset file not found")

            with pytest.raises(PresetNotFoundError) as exc_info:
                registry.get("go")

            assert exc_info.value.preset_name == "go"

    @pytest.mark.unit
    def test_get_handles_module_not_found(self, registry: PresetRegistry) -> None:
        """get() raises PresetNotFoundError when module loading fails."""
        with patch("src.domain.validation.preset_registry.resources.files") as mock:
            mock.side_effect = ModuleNotFoundError("Module not found")

            with pytest.raises(PresetNotFoundError) as exc_info:
                registry.get("rust")

            assert exc_info.value.preset_name == "rust"


# =============================================================================
# Integration Tests (real package files)
# =============================================================================


class TestPresetRegistryIntegration:
    """Integration tests for PresetRegistry with real preset files."""

    @pytest.fixture
    def registry(self) -> PresetRegistry:
        """Create a PresetRegistry instance."""
        return PresetRegistry()

    @pytest.mark.integration
    def test_get_python_uv_preset(self, registry: PresetRegistry) -> None:
        """get('python-uv') returns valid ValidationConfig with isolation flags."""
        config = registry.get("python-uv")

        assert isinstance(config, ValidationConfig)
        assert config.commands.setup is not None
        assert config.commands.setup.command == "uv sync"
        assert config.commands.test is not None
        assert (
            config.commands.test.command
            == "uv run pytest -o cache_dir=/tmp/pytest-${AGENT_ID:-default}"
        )
        assert config.commands.lint is not None
        assert (
            config.commands.lint.command
            == "RUFF_CACHE_DIR=/tmp/ruff-${AGENT_ID:-default} uvx ruff check ."
        )
        assert config.commands.format is not None
        assert (
            config.commands.format.command
            == "RUFF_CACHE_DIR=/tmp/ruff-${AGENT_ID:-default} uvx ruff format ."
        )
        assert config.commands.typecheck is not None
        assert config.commands.typecheck.command == "uvx ty check"
        assert config.commands.e2e is not None
        assert (
            config.commands.e2e.command
            == "uv run pytest -m e2e -o cache_dir=/tmp/pytest-${AGENT_ID:-default}"
        )
        assert "**/*.py" in config.code_patterns
        assert "pyproject.toml" in config.code_patterns
        assert "uv.lock" in config.setup_files

    @pytest.mark.integration
    def test_get_node_npm_preset(self, registry: PresetRegistry) -> None:
        """get('node-npm') returns valid ValidationConfig."""
        config = registry.get("node-npm")

        assert isinstance(config, ValidationConfig)
        assert config.commands.setup is not None
        assert config.commands.setup.command == "npm install"
        assert config.commands.test is not None
        assert config.commands.test.command == "npm test"
        assert config.commands.lint is not None
        assert config.commands.lint.command == "npx eslint ."
        assert config.commands.format is not None
        assert config.commands.format.command == "npx prettier --check ."
        assert config.commands.typecheck is not None
        assert config.commands.typecheck.command == "npx tsc --noEmit"
        assert "**/*.js" in config.code_patterns
        assert "**/*.ts" in config.code_patterns
        assert "package-lock.json" in config.setup_files

    @pytest.mark.integration
    def test_get_go_preset(self, registry: PresetRegistry) -> None:
        """get('go') returns valid ValidationConfig."""
        config = registry.get("go")

        assert isinstance(config, ValidationConfig)
        assert config.commands.setup is not None
        assert config.commands.setup.command == "go mod download"
        assert config.commands.test is not None
        assert config.commands.test.command == "go test ./..."
        assert config.commands.lint is not None
        assert config.commands.lint.command == "golangci-lint run"
        assert config.commands.format is not None
        assert 'test -z "$(gofmt -l .)"' in config.commands.format.command
        assert config.commands.typecheck is None  # Go doesn't have separate typecheck
        assert "**/*.go" in config.code_patterns
        assert "go.mod" in config.setup_files

    @pytest.mark.integration
    def test_get_rust_preset(self, registry: PresetRegistry) -> None:
        """get('rust') returns valid ValidationConfig."""
        config = registry.get("rust")

        assert isinstance(config, ValidationConfig)
        assert config.commands.setup is not None
        assert config.commands.setup.command == "cargo fetch"
        assert config.commands.test is not None
        assert config.commands.test.command == "cargo test"
        assert config.commands.lint is not None
        assert config.commands.lint.command == "cargo clippy -- -D warnings"
        assert config.commands.format is not None
        assert config.commands.format.command == "cargo fmt --check"
        assert config.commands.typecheck is None  # Rust doesn't have separate typecheck
        assert "**/*.rs" in config.code_patterns
        assert "Cargo.toml" in config.setup_files

    @pytest.mark.integration
    def test_all_presets_have_valid_yaml(self, registry: PresetRegistry) -> None:
        """All preset files have valid YAML syntax."""
        for preset_name in registry.list_presets():
            config = registry.get(preset_name)
            assert isinstance(config, ValidationConfig)
            # All presets should have at least test command
            assert config.commands.test is not None

    @pytest.mark.integration
    def test_all_presets_have_code_patterns(self, registry: PresetRegistry) -> None:
        """All presets define at least one code pattern."""
        for preset_name in registry.list_presets():
            config = registry.get(preset_name)
            assert len(config.code_patterns) > 0, f"{preset_name} missing code_patterns"

    @pytest.mark.integration
    def test_all_presets_have_setup_files(self, registry: PresetRegistry) -> None:
        """All presets define at least one setup file."""
        for preset_name in registry.list_presets():
            config = registry.get(preset_name)
            assert len(config.setup_files) > 0, f"{preset_name} missing setup_files"

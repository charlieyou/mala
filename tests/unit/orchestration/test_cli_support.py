"""Unit tests for ``src.orchestration.cli_support`` helpers.

These cover CLI-support helpers that touch packaged resources (preset
registry loading). They keep this resource I/O out of the pure
``init_config`` helpers per the layering contract.
"""

from __future__ import annotations

import pytest

from src.orchestration.cli_support import get_preset_config_commands

pytestmark = pytest.mark.unit


class TestGetPresetConfigCommands:
    def test_known_preset_returns_commands(self) -> None:
        result = get_preset_config_commands("python-uv")
        assert "test" in result
        assert "lint" in result
        assert "setup" in result

    def test_unknown_preset_raises(self) -> None:
        from src.domain.validation.preset_registry import PresetNotFoundError

        with pytest.raises(PresetNotFoundError):
            get_preset_config_commands("nonexistent-preset")

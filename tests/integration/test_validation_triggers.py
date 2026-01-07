"""Integration tests for validation_triggers config loading."""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def test_config_loads_validation_triggers_via_normal_path(tmp_path: Path) -> None:
    """Test that validation_triggers are loaded through the normal config path.

    This test exercises the full config loading path:
    load_config() → _build_config() → _parse_validation_triggers()

    Expected: FAILS with NotImplementedError (skeleton implementation)
    After T003: Test will pass with populated ValidationTriggersConfig
    """
    from src.domain.validation.config_loader import load_config

    # Create a minimal mala.yaml with validation_triggers section
    config_content = dedent("""\
        preset: python-uv

        validation_triggers:
          epic_completion:
            when: always
          session_end:
            when: on_changes
          periodic:
            when: always
            interval: 300
    """)

    config_file = tmp_path / "mala.yaml"
    config_file.write_text(config_content)

    # This should raise NotImplementedError from the skeleton
    with pytest.raises(
        NotImplementedError, match="validation_triggers parsing not yet implemented"
    ):
        load_config(tmp_path)

"""Integration tests for config loading path.

These tests verify the end-to-end config loading flow, from YAML file
to populated config dataclasses.
"""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from src.domain.validation.config_loader import load_config


class TestCodeReviewParsing:
    """Tests for code_review block parsing in triggers."""

    def test_session_end_code_review_raises_not_implemented(
        self, tmp_path: Path
    ) -> None:
        """Integration test: config_loader.load() with code_review raises NotImplementedError.

        This test verifies the wiring is correct - it should fail with
        NotImplementedError (parsing not implemented), NOT with ImportError
        or other wiring issues.
        """
        yaml_content = dedent("""\
            preset: python-uv
            validation_triggers:
              session_end:
                commands: []
                failure_mode: continue
                code_review:
                  enabled: true
                  reviewer_type: cerberus
        """)
        config_file = tmp_path / "mala.yaml"
        config_file.write_text(yaml_content)

        with pytest.raises(
            NotImplementedError, match="code_review parsing not yet implemented"
        ):
            load_config(tmp_path)

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

    def test_run_end_code_review_raises_not_implemented(self, tmp_path: Path) -> None:
        """Integration test: run_end trigger with code_review raises NotImplementedError."""
        yaml_content = dedent("""\
            preset: python-uv
            validation_triggers:
              run_end:
                commands: []
                failure_mode: continue
                code_review:
                  enabled: true
        """)
        config_file = tmp_path / "mala.yaml"
        config_file.write_text(yaml_content)

        with pytest.raises(
            NotImplementedError, match="code_review parsing not yet implemented"
        ):
            load_config(tmp_path)

    def test_epic_completion_code_review_raises_not_implemented(
        self, tmp_path: Path
    ) -> None:
        """Integration test: epic_completion trigger with code_review raises NotImplementedError."""
        yaml_content = dedent("""\
            preset: python-uv
            validation_triggers:
              epic_completion:
                commands: []
                failure_mode: continue
                epic_depth: top_level
                fire_on: success
                code_review:
                  enabled: true
        """)
        config_file = tmp_path / "mala.yaml"
        config_file.write_text(yaml_content)

        with pytest.raises(
            NotImplementedError, match="code_review parsing not yet implemented"
        ):
            load_config(tmp_path)

    def test_periodic_code_review_raises_not_implemented(self, tmp_path: Path) -> None:
        """Integration test: periodic trigger with code_review raises NotImplementedError."""
        yaml_content = dedent("""\
            preset: python-uv
            validation_triggers:
              periodic:
                commands: []
                failure_mode: continue
                interval: 60
                code_review:
                  enabled: true
        """)
        config_file = tmp_path / "mala.yaml"
        config_file.write_text(yaml_content)

        with pytest.raises(
            NotImplementedError, match="code_review parsing not yet implemented"
        ):
            load_config(tmp_path)


class TestRunEndTriggerParsing:
    """Tests for run_end trigger parsing."""

    def test_run_end_without_code_review_parses_successfully(
        self, tmp_path: Path
    ) -> None:
        """run_end trigger without code_review should parse successfully."""
        yaml_content = dedent("""\
            preset: python-uv
            validation_triggers:
              run_end:
                commands:
                  - ref: test
                failure_mode: continue
        """)
        config_file = tmp_path / "mala.yaml"
        config_file.write_text(yaml_content)

        config = load_config(tmp_path)

        assert config.validation_triggers is not None
        assert config.validation_triggers.run_end is not None
        assert config.validation_triggers.run_end.failure_mode.value == "continue"
        assert config.validation_triggers.run_end.code_review is None

    def test_run_end_with_fire_on(self, tmp_path: Path) -> None:
        """run_end trigger with fire_on should parse successfully."""
        yaml_content = dedent("""\
            preset: python-uv
            validation_triggers:
              run_end:
                commands: []
                failure_mode: continue
                fire_on: both
        """)
        config_file = tmp_path / "mala.yaml"
        config_file.write_text(yaml_content)

        config = load_config(tmp_path)

        assert config.validation_triggers is not None
        assert config.validation_triggers.run_end is not None
        assert config.validation_triggers.run_end.fire_on.value == "both"

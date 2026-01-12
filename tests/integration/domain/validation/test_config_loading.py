"""Integration tests for config loading path.

These tests verify the end-to-end config loading flow, from YAML file
to populated config dataclasses.
"""

from __future__ import annotations

from textwrap import dedent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from src.domain.validation.config import FailureMode
from src.domain.validation.config_loader import load_config


class TestCodeReviewParsing:
    """Tests for code_review block parsing in triggers."""

    def test_session_end_code_review_parsed(self, tmp_path: Path) -> None:
        """Integration test: config_loader.load() parses code_review block.

        This test verifies the end-to-end config loading path works, from
        YAML file to populated CodeReviewConfig dataclass.
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

        config = load_config(tmp_path)

        assert config.validation_triggers is not None
        assert config.validation_triggers.session_end is not None
        code_review = config.validation_triggers.session_end.code_review
        assert code_review is not None
        assert code_review.enabled is True
        assert code_review.reviewer_type == "cerberus"

    def test_epic_completion_code_review_with_cerberus(self, tmp_path: Path) -> None:
        """Integration test: epic_completion trigger parses code_review with cerberus."""
        yaml_content = dedent("""\
            preset: python-uv
            validation_triggers:
              epic_completion:
                commands: []
                failure_mode: continue
                epic_depth: all
                fire_on: success
                code_review:
                  enabled: true
                  reviewer_type: cerberus
                  failure_mode: remediate
                  max_retries: 5
                  finding_threshold: P1
                  baseline: since_last_review
                  cerberus:
                    timeout: 600
                    spawn_args:
                      - "--flag"
                    env:
                      MY_VAR: value
        """)
        config_file = tmp_path / "mala.yaml"
        config_file.write_text(yaml_content)

        config = load_config(tmp_path)

        assert config.validation_triggers is not None
        assert config.validation_triggers.epic_completion is not None
        code_review = config.validation_triggers.epic_completion.code_review
        assert code_review is not None
        assert code_review.enabled is True
        assert code_review.reviewer_type == "cerberus"
        assert code_review.failure_mode == FailureMode.REMEDIATE
        assert code_review.max_retries == 5
        assert code_review.finding_threshold == "P1"
        assert code_review.baseline == "since_last_review"
        assert code_review.cerberus is not None
        assert code_review.cerberus.timeout == 600
        assert code_review.cerberus.spawn_args == ("--flag",)
        assert code_review.cerberus.env == (("MY_VAR", "value"),)

    def test_run_end_code_review_parsed(self, tmp_path: Path) -> None:
        """Integration test: run_end trigger parses code_review block."""
        yaml_content = dedent("""\
            preset: python-uv
            validation_triggers:
              run_end:
                commands: []
                failure_mode: continue
                fire_on: both
                code_review:
                  enabled: true
                  reviewer_type: agent_sdk
                  baseline: since_run_start
        """)
        config_file = tmp_path / "mala.yaml"
        config_file.write_text(yaml_content)

        config = load_config(tmp_path)

        assert config.validation_triggers is not None
        assert config.validation_triggers.run_end is not None
        code_review = config.validation_triggers.run_end.code_review
        assert code_review is not None
        assert code_review.enabled is True
        assert code_review.reviewer_type == "agent_sdk"
        assert code_review.baseline == "since_run_start"

    def test_run_end_code_review_finding_threshold_parsed(self, tmp_path: Path) -> None:
        """Integration test: run_end code_review parses finding_threshold."""
        yaml_content = dedent("""\
            preset: python-uv
            validation_triggers:
              run_end:
                commands: []
                failure_mode: continue
                fire_on: success
                code_review:
                  enabled: true
                  reviewer_type: cerberus
                  finding_threshold: P1
                  baseline: since_run_start
        """)
        config_file = tmp_path / "mala.yaml"
        config_file.write_text(yaml_content)

        config = load_config(tmp_path)

        assert config.validation_triggers is not None
        assert config.validation_triggers.run_end is not None
        code_review = config.validation_triggers.run_end.code_review
        assert code_review is not None
        assert code_review.finding_threshold == "P1"

    def test_run_end_code_review_remediate_parsed(self, tmp_path: Path) -> None:
        """Integration test: run_end code_review parses failure_mode remediate."""
        yaml_content = dedent("""\
            preset: python-uv
            validation_triggers:
              run_end:
                commands: []
                failure_mode: continue
                fire_on: both
                code_review:
                  enabled: true
                  reviewer_type: cerberus
                  failure_mode: remediate
                  max_retries: 3
                  baseline: since_run_start
        """)
        config_file = tmp_path / "mala.yaml"
        config_file.write_text(yaml_content)

        config = load_config(tmp_path)

        assert config.validation_triggers is not None
        assert config.validation_triggers.run_end is not None
        code_review = config.validation_triggers.run_end.code_review
        assert code_review is not None
        assert code_review.failure_mode == FailureMode.REMEDIATE
        assert code_review.max_retries == 3

    def test_run_end_full_code_review_config(self, tmp_path: Path) -> None:
        """Integration test: run_end parses complete code_review configuration."""
        yaml_content = dedent("""\
            preset: python-uv
            validation_triggers:
              run_end:
                commands: []
                failure_mode: continue
                fire_on: success
                code_review:
                  enabled: true
                  reviewer_type: cerberus
                  failure_mode: remediate
                  max_retries: 5
                  finding_threshold: P2
                  baseline: since_run_start
                  cerberus:
                    timeout: 300
                    spawn_args:
                      - "--verbose"
                    env:
                      DEBUG: "1"
        """)
        config_file = tmp_path / "mala.yaml"
        config_file.write_text(yaml_content)

        config = load_config(tmp_path)

        assert config.validation_triggers is not None
        assert config.validation_triggers.run_end is not None
        code_review = config.validation_triggers.run_end.code_review
        assert code_review is not None
        assert code_review.enabled is True
        assert code_review.reviewer_type == "cerberus"
        assert code_review.failure_mode == FailureMode.REMEDIATE
        assert code_review.max_retries == 5
        assert code_review.finding_threshold == "P2"
        assert code_review.baseline == "since_run_start"
        assert code_review.cerberus is not None
        assert code_review.cerberus.timeout == 300
        assert code_review.cerberus.spawn_args == ("--verbose",)
        assert code_review.cerberus.env == (("DEBUG", "1"),)

    def test_code_review_track_review_issues_default(self, tmp_path: Path) -> None:
        """Integration test: track_review_issues defaults to True."""
        yaml_content = dedent("""\
            preset: python-uv
            validation_triggers:
              session_end:
                commands: []
                failure_mode: continue
                code_review:
                  enabled: true
        """)
        config_file = tmp_path / "mala.yaml"
        config_file.write_text(yaml_content)

        config = load_config(tmp_path)

        assert config.validation_triggers is not None
        assert config.validation_triggers.session_end is not None
        code_review = config.validation_triggers.session_end.code_review
        assert code_review is not None
        assert code_review.track_review_issues is True

    def test_code_review_track_review_issues_explicit_false(
        self, tmp_path: Path
    ) -> None:
        """Integration test: track_review_issues can be set to false."""
        yaml_content = dedent("""\
            preset: python-uv
            validation_triggers:
              session_end:
                commands: []
                failure_mode: continue
                code_review:
                  enabled: true
                  track_review_issues: false
        """)
        config_file = tmp_path / "mala.yaml"
        config_file.write_text(yaml_content)

        config = load_config(tmp_path)

        assert config.validation_triggers is not None
        assert config.validation_triggers.session_end is not None
        code_review = config.validation_triggers.session_end.code_review
        assert code_review is not None
        assert code_review.track_review_issues is False


class TestPerIssueReviewConfig:
    """Tests for per_issue_review root-level config parsing."""

    def test_per_issue_review_config_loads_from_yaml(self, tmp_path: Path) -> None:
        """Integration test: per_issue_review is parsed from mala.yaml.

        This test exercises the full config loading path to verify that
        per_issue_review flows from YAML to ValidationConfig. Currently
        expected to fail because parsing is not yet implemented (TDD).
        """
        yaml_content = dedent("""\
            preset: python-uv
            per_issue_review:
              enabled: true
              reviewer_type: cerberus
        """)
        config_file = tmp_path / "mala.yaml"
        config_file.write_text(yaml_content)

        config = load_config(tmp_path)

        # This should pass once T002 implements parsing
        assert config.per_issue_review.enabled is True
        assert config.per_issue_review.reviewer_type == "cerberus"


class TestEpicCompletionTriggerFields:
    """Tests for epic_completion trigger-specific fields."""

    def test_epic_verify_lock_timeout_seconds_default(self, tmp_path: Path) -> None:
        """epic_verify_lock_timeout_seconds defaults to None when not set."""
        yaml_content = dedent("""\
            preset: python-uv
            validation_triggers:
              epic_completion:
                epic_depth: all
                fire_on: success
                failure_mode: continue
                commands: []
        """)
        config_file = tmp_path / "mala.yaml"
        config_file.write_text(yaml_content)

        config = load_config(tmp_path)

        assert config.validation_triggers is not None
        epic_completion = config.validation_triggers.epic_completion
        assert epic_completion is not None
        assert epic_completion.epic_verify_lock_timeout_seconds is None

    def test_epic_verify_lock_timeout_seconds_explicit(self, tmp_path: Path) -> None:
        """epic_verify_lock_timeout_seconds can be set explicitly."""
        yaml_content = dedent("""\
            preset: python-uv
            validation_triggers:
              epic_completion:
                epic_depth: all
                fire_on: success
                failure_mode: continue
                commands: []
                epic_verify_lock_timeout_seconds: 120
        """)
        config_file = tmp_path / "mala.yaml"
        config_file.write_text(yaml_content)

        config = load_config(tmp_path)

        assert config.validation_triggers is not None
        epic_completion = config.validation_triggers.epic_completion
        assert epic_completion is not None
        assert epic_completion.epic_verify_lock_timeout_seconds == 120

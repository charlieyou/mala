"""Unit tests for CLI deprecation warnings.

Tests for:
- --reviewer-type CLI flag deprecation warning
- Legacy top-level reviewer_type config deprecation warning (in factory)
- Config takes precedence over CLI
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import pytest

from src.cli.cli import _check_cli_reviewer_type_deprecation
from src.orchestration.factory import (
    _check_legacy_review_config,
    _extract_reviewer_config,
    _has_new_code_review_config,
)


class TestCliReviewerTypeDeprecation:
    """Test --reviewer-type CLI flag deprecation warning."""

    def test_emits_warning_when_reviewer_type_provided(self) -> None:
        """--reviewer-type flag emits DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_cli_reviewer_type_deprecation("cerberus")

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "--reviewer-type is deprecated" in str(w[0].message)
            assert "validation_triggers.session_end.code_review" in str(w[0].message)

    def test_no_warning_when_reviewer_type_none(self) -> None:
        """No warning when --reviewer-type is not provided."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_cli_reviewer_type_deprecation(None)

            assert len(w) == 0

    def test_warning_for_agent_sdk_type(self) -> None:
        """Warning emitted for agent_sdk reviewer type."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_cli_reviewer_type_deprecation("agent_sdk")

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)


# Minimal mock classes to simulate ValidationConfig structure
@dataclass(frozen=True)
class MockCodeReviewConfig:
    """Mock code review config for testing."""

    enabled: bool = False
    reviewer_type: Literal["cerberus", "agent_sdk"] = "cerberus"


@dataclass(frozen=True)
class MockTriggerConfig:
    """Mock trigger config for testing."""

    code_review: MockCodeReviewConfig | None = None


@dataclass(frozen=True)
class MockTriggersConfig:
    """Mock triggers config for testing."""

    session_end: MockTriggerConfig | None = None
    epic_completion: MockTriggerConfig | None = None
    run_end: MockTriggerConfig | None = None
    periodic: MockTriggerConfig | None = None


@dataclass(frozen=True)
class MockValidationConfig:
    """Mock validation config for testing."""

    reviewer_type: Literal["agent_sdk", "cerberus"] = "agent_sdk"
    agent_sdk_review_timeout: int = 600
    agent_sdk_reviewer_model: Literal["sonnet", "opus", "haiku"] = "sonnet"
    validation_triggers: MockTriggersConfig | None = None
    _fields_set: frozenset[str] = field(default_factory=frozenset)


class TestHasNewCodeReviewConfig:
    """Test _has_new_code_review_config helper."""

    def test_returns_false_for_none(self) -> None:
        """Returns False when config is None."""
        assert _has_new_code_review_config(None) is False

    def test_returns_false_when_no_triggers(self) -> None:
        """Returns False when validation_triggers is None."""
        config = MockValidationConfig(validation_triggers=None)
        assert _has_new_code_review_config(config) is False

    def test_returns_false_when_code_review_disabled(self) -> None:
        """Returns False when code_review exists but is disabled."""
        config = MockValidationConfig(
            validation_triggers=MockTriggersConfig(
                session_end=MockTriggerConfig(
                    code_review=MockCodeReviewConfig(enabled=False)
                )
            )
        )
        assert _has_new_code_review_config(config) is False

    def test_returns_true_when_code_review_enabled(self) -> None:
        """Returns True when code_review is enabled in any trigger."""
        config = MockValidationConfig(
            validation_triggers=MockTriggersConfig(
                session_end=MockTriggerConfig(
                    code_review=MockCodeReviewConfig(enabled=True)
                )
            )
        )
        assert _has_new_code_review_config(config) is True

    def test_returns_true_for_epic_completion_trigger(self) -> None:
        """Returns True when code_review enabled in epic_completion."""
        config = MockValidationConfig(
            validation_triggers=MockTriggersConfig(
                epic_completion=MockTriggerConfig(
                    code_review=MockCodeReviewConfig(enabled=True)
                )
            )
        )
        assert _has_new_code_review_config(config) is True


class TestLegacyReviewConfigDeprecation:
    """Test legacy top-level reviewer_type config deprecation warning."""

    def test_no_warning_when_config_none(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No warning when config is None."""
        with caplog.at_level(logging.WARNING):
            _check_legacy_review_config(None)

        assert "DEPRECATION" not in caplog.text

    def test_no_warning_when_reviewer_type_not_in_fields_set(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No warning when reviewer_type is not explicitly set."""
        config = MockValidationConfig(
            reviewer_type="agent_sdk",
            _fields_set=frozenset(),  # reviewer_type not in fields_set
        )
        with caplog.at_level(logging.WARNING):
            _check_legacy_review_config(config)

        assert "DEPRECATION" not in caplog.text

    def test_warning_when_legacy_reviewer_type_set_without_new_config(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warning when reviewer_type is explicitly set but no code_review config."""
        config = MockValidationConfig(
            reviewer_type="cerberus",
            _fields_set=frozenset({"reviewer_type"}),
            validation_triggers=None,
        )
        with caplog.at_level(logging.WARNING):
            _check_legacy_review_config(config)

        assert "DEPRECATION" in caplog.text
        assert "reviewer_type" in caplog.text
        assert "validation_triggers.session_end.code_review" in caplog.text

    def test_no_warning_when_new_code_review_enabled(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No warning when new code_review config is present and enabled."""
        config = MockValidationConfig(
            reviewer_type="cerberus",
            _fields_set=frozenset({"reviewer_type"}),
            validation_triggers=MockTriggersConfig(
                session_end=MockTriggerConfig(
                    code_review=MockCodeReviewConfig(enabled=True)
                )
            ),
        )
        with caplog.at_level(logging.WARNING):
            _check_legacy_review_config(config)

        assert "DEPRECATION" not in caplog.text

    def test_warning_when_code_review_disabled(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warning when code_review config exists but is disabled."""
        config = MockValidationConfig(
            reviewer_type="cerberus",
            _fields_set=frozenset({"reviewer_type"}),
            validation_triggers=MockTriggersConfig(
                session_end=MockTriggerConfig(
                    code_review=MockCodeReviewConfig(enabled=False)
                )
            ),
        )
        with caplog.at_level(logging.WARNING):
            _check_legacy_review_config(config)

        assert "DEPRECATION" in caplog.text

    def test_no_warning_when_epic_completion_code_review_enabled(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """No warning when any trigger has enabled code_review."""
        config = MockValidationConfig(
            reviewer_type="cerberus",
            _fields_set=frozenset({"reviewer_type"}),
            validation_triggers=MockTriggersConfig(
                epic_completion=MockTriggerConfig(
                    code_review=MockCodeReviewConfig(enabled=True)
                )
            ),
        )
        with caplog.at_level(logging.WARNING):
            _check_legacy_review_config(config)

        assert "DEPRECATION" not in caplog.text

    def test_warning_when_triggers_exist_but_no_code_review(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Warning when triggers exist but have no code_review config."""
        config = MockValidationConfig(
            reviewer_type="agent_sdk",
            _fields_set=frozenset({"reviewer_type"}),
            validation_triggers=MockTriggersConfig(
                session_end=MockTriggerConfig(code_review=None)
            ),
        )
        with caplog.at_level(logging.WARNING):
            _check_legacy_review_config(config)

        assert "DEPRECATION" in caplog.text


class TestExtractReviewerConfigPrecedence:
    """Test reviewer_type precedence in _extract_reviewer_config."""

    def test_cli_override_used_when_no_new_config(self) -> None:
        """CLI --reviewer-type is used when no new code_review config exists."""
        config = MockValidationConfig(
            reviewer_type="agent_sdk",  # Legacy config value
            validation_triggers=None,
        )
        result = _extract_reviewer_config(config, cli_reviewer_type="cerberus")
        assert result.reviewer_type == "cerberus"

    def test_config_used_when_new_code_review_present(self) -> None:
        """Config reviewer_type used when new code_review config exists."""
        config = MockValidationConfig(
            reviewer_type="agent_sdk",
            validation_triggers=MockTriggersConfig(
                session_end=MockTriggerConfig(
                    code_review=MockCodeReviewConfig(enabled=True)
                )
            ),
        )
        # CLI override should be ignored when new config exists
        result = _extract_reviewer_config(config, cli_reviewer_type="cerberus")
        assert result.reviewer_type == "agent_sdk"

    def test_legacy_config_used_when_no_cli_override(self) -> None:
        """Legacy config reviewer_type used when no CLI override."""
        config = MockValidationConfig(
            reviewer_type="cerberus",
            validation_triggers=None,
        )
        result = _extract_reviewer_config(config, cli_reviewer_type=None)
        assert result.reviewer_type == "cerberus"

    def test_default_used_when_nothing_set(self) -> None:
        """Default reviewer_type used when nothing is set."""
        config = MockValidationConfig(
            reviewer_type="agent_sdk",  # Default
            validation_triggers=None,
        )
        result = _extract_reviewer_config(config, cli_reviewer_type=None)
        assert result.reviewer_type == "agent_sdk"


class TestConfigPrecedence:
    """Test that config takes precedence over CLI for reviewer settings."""

    def test_cli_flag_is_hidden_and_deprecated(self) -> None:
        """CLI --reviewer-type flag should be hidden and emit deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _check_cli_reviewer_type_deprecation("cerberus")

            assert len(w) == 1
            assert "deprecated" in str(w[0].message).lower()

    def test_new_config_takes_precedence_over_cli(self) -> None:
        """New code_review config takes precedence over CLI --reviewer-type."""
        config = MockValidationConfig(
            reviewer_type="agent_sdk",
            validation_triggers=MockTriggersConfig(
                session_end=MockTriggerConfig(
                    code_review=MockCodeReviewConfig(enabled=True)
                )
            ),
        )
        # Even with CLI override, new config wins
        result = _extract_reviewer_config(config, cli_reviewer_type="cerberus")
        assert result.reviewer_type == "agent_sdk"

    def test_cli_takes_precedence_over_legacy_config(self) -> None:
        """CLI --reviewer-type takes precedence over legacy top-level config."""
        config = MockValidationConfig(
            reviewer_type="agent_sdk",  # Legacy value
            validation_triggers=None,  # No new config
        )
        result = _extract_reviewer_config(config, cli_reviewer_type="cerberus")
        assert result.reviewer_type == "cerberus"

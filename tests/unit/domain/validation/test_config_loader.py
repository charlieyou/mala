"""Unit tests for config_loader code_review parsing.

Tests the _parse_code_review_config and _parse_cerberus_config functions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest

from src.domain.validation.config import ConfigError, FailureMode
from src.domain.validation.config_loader import (
    _parse_cerberus_config,
    _parse_code_review_config,
    _parse_periodic_trigger,
)

if TYPE_CHECKING:
    from _pytest.logging import LogCaptureFixture


class TestParseCerberusConfig:
    """Tests for _parse_cerberus_config function."""

    def test_defaults(self) -> None:
        """Empty dict returns defaults."""
        result = _parse_cerberus_config({})
        assert result.timeout == 300
        assert result.spawn_args == ()
        assert result.wait_args == ()
        assert result.env == ()

    def test_custom_timeout(self) -> None:
        """Custom timeout is parsed."""
        result = _parse_cerberus_config({"timeout": 600})
        assert result.timeout == 600

    def test_timeout_bool_rejected(self) -> None:
        """Boolean timeout raises ConfigError."""
        with pytest.raises(ConfigError, match=r"cerberus\.timeout must be an integer"):
            _parse_cerberus_config({"timeout": True})

    def test_timeout_string_rejected(self) -> None:
        """String timeout raises ConfigError."""
        with pytest.raises(ConfigError, match=r"cerberus\.timeout must be an integer"):
            _parse_cerberus_config({"timeout": "600"})

    def test_spawn_args(self) -> None:
        """spawn_args list is parsed as tuple."""
        result = _parse_cerberus_config({"spawn_args": ["--foo", "--bar"]})
        assert result.spawn_args == ("--foo", "--bar")

    def test_spawn_args_not_list_rejected(self) -> None:
        """Non-list spawn_args raises ConfigError."""
        with pytest.raises(ConfigError, match=r"cerberus\.spawn_args must be a list"):
            _parse_cerberus_config({"spawn_args": "--foo"})

    def test_spawn_args_non_string_element_rejected(self) -> None:
        """Non-string spawn_args element raises ConfigError."""
        with pytest.raises(
            ConfigError, match=r"cerberus.spawn_args\[1\] must be a string"
        ):
            _parse_cerberus_config({"spawn_args": ["--foo", 123]})

    def test_wait_args(self) -> None:
        """wait_args list is parsed as tuple."""
        result = _parse_cerberus_config({"wait_args": ["--timeout", "60"]})
        assert result.wait_args == ("--timeout", "60")

    def test_wait_args_not_list_rejected(self) -> None:
        """Non-list wait_args raises ConfigError."""
        with pytest.raises(ConfigError, match=r"cerberus\.wait_args must be a list"):
            _parse_cerberus_config({"wait_args": {"key": "value"}})

    def test_wait_args_non_string_element_rejected(self) -> None:
        """Non-string wait_args element raises ConfigError."""
        with pytest.raises(
            ConfigError, match=r"cerberus.wait_args\[0\] must be a string"
        ):
            _parse_cerberus_config({"wait_args": [None]})

    def test_env_dict(self) -> None:
        """env dict is parsed as sorted tuple of tuples."""
        result = _parse_cerberus_config({"env": {"Z_VAR": "z", "A_VAR": "a"}})
        assert result.env == (("A_VAR", "a"), ("Z_VAR", "z"))

    def test_env_not_dict_rejected(self) -> None:
        """Non-dict env raises ConfigError."""
        with pytest.raises(ConfigError, match=r"cerberus\.env must be an object"):
            _parse_cerberus_config({"env": [("KEY", "value")]})

    def test_env_non_string_value_rejected(self) -> None:
        """Non-string env value raises ConfigError."""
        with pytest.raises(
            ConfigError, match=r"cerberus.env\['MY_VAR'\] must be a string"
        ):
            _parse_cerberus_config({"env": {"MY_VAR": 123}})

    def test_unknown_field_rejected(self) -> None:
        """Unknown field in cerberus raises ConfigError."""
        with pytest.raises(ConfigError, match="Unknown field 'bad_field'"):
            _parse_cerberus_config({"bad_field": "value"})

    def test_all_fields_together(self) -> None:
        """All cerberus fields can be used together."""
        result = _parse_cerberus_config(
            {
                "timeout": 120,
                "spawn_args": ["--flag"],
                "wait_args": ["--wait-flag"],
                "env": {"KEY": "value"},
            }
        )
        assert result.timeout == 120
        assert result.spawn_args == ("--flag",)
        assert result.wait_args == ("--wait-flag",)
        assert result.env == (("KEY", "value"),)


class TestParseCodeReviewConfig:
    """Tests for _parse_code_review_config function."""

    def test_none_returns_none(self) -> None:
        """None data returns None."""
        result = _parse_code_review_config(None, "session_end")  # type: ignore[arg-type]
        assert result is None

    def test_not_dict_rejected(self) -> None:
        """Non-dict code_review raises ConfigError."""
        with pytest.raises(ConfigError, match="code_review must be an object"):
            _parse_code_review_config("invalid", "session_end")  # type: ignore[arg-type]

    def test_defaults(self) -> None:
        """Empty dict returns defaults for session_end."""
        result = _parse_code_review_config({}, "session_end")
        assert result is not None
        assert result.enabled is False
        assert result.reviewer_type == "cerberus"
        assert result.failure_mode == FailureMode.CONTINUE
        assert result.max_retries == 3
        assert result.finding_threshold == "none"
        assert result.baseline is None
        assert result.cerberus is None

    def test_enabled_true(self) -> None:
        """enabled: true is parsed."""
        result = _parse_code_review_config({"enabled": True}, "session_end")
        assert result is not None
        assert result.enabled is True

    def test_enabled_non_bool_rejected(self) -> None:
        """Non-boolean enabled raises ConfigError."""
        with pytest.raises(
            ConfigError, match=r"code_review\.enabled must be a boolean"
        ):
            _parse_code_review_config({"enabled": "true"}, "session_end")

    def test_reviewer_type_cerberus(self) -> None:
        """reviewer_type cerberus is parsed."""
        result = _parse_code_review_config({"reviewer_type": "cerberus"}, "session_end")
        assert result is not None
        assert result.reviewer_type == "cerberus"

    def test_reviewer_type_agent_sdk(self) -> None:
        """reviewer_type agent_sdk is parsed."""
        result = _parse_code_review_config(
            {"reviewer_type": "agent_sdk"}, "session_end"
        )
        assert result is not None
        assert result.reviewer_type == "agent_sdk"

    def test_reviewer_type_non_string_rejected(self) -> None:
        """Non-string reviewer_type raises ConfigError."""
        with pytest.raises(
            ConfigError, match=r"code_review\.reviewer_type must be a string"
        ):
            _parse_code_review_config({"reviewer_type": 123}, "session_end")

    def test_reviewer_type_invalid_rejected_when_enabled(self) -> None:
        """Invalid reviewer_type with enabled: true raises ConfigError."""
        with pytest.raises(
            ConfigError, match=r"code_review\.reviewer_type must be 'cerberus' or"
        ):
            _parse_code_review_config(
                {"enabled": True, "reviewer_type": "invalid"}, "session_end"
            )

    def test_reviewer_type_invalid_allowed_when_disabled(self) -> None:
        """Invalid reviewer_type is allowed when enabled: false.

        Per spec, only error on invalid reviewer_type when enabled: true.
        """
        result = _parse_code_review_config(
            {"enabled": False, "reviewer_type": "invalid"}, "session_end"
        )
        assert result is not None
        assert result.enabled is False
        # Value is preserved even though invalid (won't be used since disabled)
        assert result.reviewer_type == "invalid"

    def test_failure_mode_continue(self) -> None:
        """failure_mode continue is parsed."""
        result = _parse_code_review_config({"failure_mode": "continue"}, "session_end")
        assert result is not None
        assert result.failure_mode == FailureMode.CONTINUE

    def test_failure_mode_abort(self) -> None:
        """failure_mode abort is parsed."""
        result = _parse_code_review_config({"failure_mode": "abort"}, "session_end")
        assert result is not None
        assert result.failure_mode == FailureMode.ABORT

    def test_failure_mode_remediate(self) -> None:
        """failure_mode remediate is parsed."""
        result = _parse_code_review_config({"failure_mode": "remediate"}, "session_end")
        assert result is not None
        assert result.failure_mode == FailureMode.REMEDIATE

    def test_failure_mode_non_string_rejected(self) -> None:
        """Non-string failure_mode raises ConfigError."""
        with pytest.raises(
            ConfigError, match=r"code_review\.failure_mode must be a string"
        ):
            _parse_code_review_config({"failure_mode": True}, "session_end")

    def test_failure_mode_invalid_rejected(self) -> None:
        """Invalid failure_mode raises ConfigError."""
        with pytest.raises(
            ConfigError, match=r"Invalid code_review\.failure_mode 'invalid'"
        ):
            _parse_code_review_config({"failure_mode": "invalid"}, "session_end")

    def test_max_retries_custom(self) -> None:
        """Custom max_retries is parsed."""
        result = _parse_code_review_config({"max_retries": 5}, "session_end")
        assert result is not None
        assert result.max_retries == 5

    def test_max_retries_zero(self) -> None:
        """max_retries: 0 is valid."""
        result = _parse_code_review_config({"max_retries": 0}, "session_end")
        assert result is not None
        assert result.max_retries == 0

    def test_max_retries_negative_rejected(self) -> None:
        """Negative max_retries raises ConfigError."""
        with pytest.raises(ConfigError, match=r"code_review\.max_retries must be >= 0"):
            _parse_code_review_config({"max_retries": -1}, "session_end")

    def test_max_retries_bool_rejected(self) -> None:
        """Boolean max_retries raises ConfigError."""
        with pytest.raises(
            ConfigError, match=r"code_review\.max_retries must be an integer"
        ):
            _parse_code_review_config({"max_retries": True}, "session_end")

    def test_max_retries_string_rejected(self) -> None:
        """String max_retries raises ConfigError."""
        with pytest.raises(
            ConfigError, match=r"code_review\.max_retries must be an integer"
        ):
            _parse_code_review_config({"max_retries": "5"}, "session_end")

    def test_finding_threshold_p0(self) -> None:
        """finding_threshold P0 is parsed."""
        result = _parse_code_review_config({"finding_threshold": "P0"}, "session_end")
        assert result is not None
        assert result.finding_threshold == "P0"

    def test_finding_threshold_p3(self) -> None:
        """finding_threshold P3 is parsed."""
        result = _parse_code_review_config({"finding_threshold": "P3"}, "session_end")
        assert result is not None
        assert result.finding_threshold == "P3"

    def test_finding_threshold_none(self) -> None:
        """finding_threshold none is parsed."""
        result = _parse_code_review_config({"finding_threshold": "none"}, "session_end")
        assert result is not None
        assert result.finding_threshold == "none"

    def test_finding_threshold_non_string_rejected(self) -> None:
        """Non-string finding_threshold raises ConfigError."""
        with pytest.raises(
            ConfigError, match=r"code_review\.finding_threshold must be a string"
        ):
            _parse_code_review_config({"finding_threshold": 0}, "session_end")

    def test_finding_threshold_invalid_rejected(self) -> None:
        """Invalid finding_threshold raises ConfigError."""
        with pytest.raises(
            ConfigError, match=r"Invalid code_review\.finding_threshold 'P4'"
        ):
            _parse_code_review_config({"finding_threshold": "P4"}, "session_end")

    def test_unknown_field_rejected(self) -> None:
        """Unknown field in code_review raises ConfigError."""
        with pytest.raises(ConfigError, match="Unknown field 'bad_field'"):
            _parse_code_review_config({"bad_field": "value"}, "session_end")

    def test_cerberus_nested(self) -> None:
        """Nested cerberus config is parsed."""
        result = _parse_code_review_config(
            {"cerberus": {"timeout": 120, "spawn_args": ["--flag"]}}, "session_end"
        )
        assert result is not None
        assert result.cerberus is not None
        assert result.cerberus.timeout == 120
        assert result.cerberus.spawn_args == ("--flag",)

    def test_cerberus_null_ignored(self) -> None:
        """cerberus: null is ignored."""
        result = _parse_code_review_config({"cerberus": None}, "session_end")
        assert result is not None
        assert result.cerberus is None

    def test_cerberus_non_dict_rejected(self) -> None:
        """Non-dict cerberus raises ConfigError."""
        with pytest.raises(
            ConfigError, match=r"code_review\.cerberus must be an object"
        ):
            _parse_code_review_config({"cerberus": "invalid"}, "session_end")


class TestCodeReviewBaselineValidation:
    """Tests for baseline validation in different trigger contexts."""

    def test_baseline_since_run_start_for_epic_completion(self) -> None:
        """baseline since_run_start is parsed for epic_completion."""
        result = _parse_code_review_config(
            {"baseline": "since_run_start"}, "epic_completion"
        )
        assert result is not None
        assert result.baseline == "since_run_start"

    def test_baseline_since_last_review_for_run_end(self) -> None:
        """baseline since_last_review is parsed for run_end."""
        result = _parse_code_review_config({"baseline": "since_last_review"}, "run_end")
        assert result is not None
        assert result.baseline == "since_last_review"

    def test_baseline_non_string_rejected(self) -> None:
        """Non-string baseline raises ConfigError."""
        with pytest.raises(
            ConfigError, match=r"code_review\.baseline must be a string"
        ):
            _parse_code_review_config({"baseline": 123}, "epic_completion")

    def test_baseline_invalid_rejected(self) -> None:
        """Invalid baseline value raises ConfigError."""
        with pytest.raises(
            ConfigError, match=r"Invalid code_review\.baseline 'invalid'"
        ):
            _parse_code_review_config({"baseline": "invalid"}, "epic_completion")

    def test_baseline_set_for_session_end_warns_and_ignores(
        self, caplog: LogCaptureFixture
    ) -> None:
        """baseline set for session_end logs warning and ignores field."""
        with caplog.at_level(logging.WARNING):
            result = _parse_code_review_config(
                {"baseline": "since_run_start"}, "session_end"
            )

        assert result is not None
        assert result.baseline is None
        assert "not applicable for session_end" in caplog.text
        assert "ignoring baseline='since_run_start'" in caplog.text

    def test_baseline_missing_for_epic_completion_warns_and_defaults(
        self, caplog: LogCaptureFixture
    ) -> None:
        """baseline missing for epic_completion logs warning and defaults."""
        with caplog.at_level(logging.WARNING):
            result = _parse_code_review_config({}, "epic_completion")

        assert result is not None
        assert result.baseline == "since_run_start"
        assert "baseline not specified for epic_completion trigger" in caplog.text
        assert "defaulting to 'since_run_start'" in caplog.text

    def test_baseline_missing_for_run_end_warns_and_defaults(
        self, caplog: LogCaptureFixture
    ) -> None:
        """baseline missing for run_end logs warning and defaults."""
        with caplog.at_level(logging.WARNING):
            result = _parse_code_review_config({}, "run_end")

        assert result is not None
        assert result.baseline == "since_run_start"
        assert "baseline not specified for run_end trigger" in caplog.text
        assert "defaulting to 'since_run_start'" in caplog.text

    def test_baseline_explicit_null_for_run_end_warns_and_defaults(
        self, caplog: LogCaptureFixture
    ) -> None:
        """baseline: null for run_end triggers warning and default."""
        with caplog.at_level(logging.WARNING):
            result = _parse_code_review_config({"baseline": None}, "run_end")

        assert result is not None
        assert result.baseline == "since_run_start"
        assert "baseline not specified for run_end trigger" in caplog.text

    def test_baseline_missing_for_session_end_no_warning(
        self, caplog: LogCaptureFixture
    ) -> None:
        """baseline missing for session_end does not warn or default."""
        with caplog.at_level(logging.WARNING):
            result = _parse_code_review_config({}, "session_end")

        assert result is not None
        assert result.baseline is None
        assert "baseline" not in caplog.text

    def test_baseline_explicit_null_for_epic_completion_warns_and_defaults(
        self, caplog: LogCaptureFixture
    ) -> None:
        """baseline: null for epic_completion triggers warning and default.

        Explicit null is treated like missing because baseline is required
        for epic_completion and run_end triggers.
        """
        with caplog.at_level(logging.WARNING):
            result = _parse_code_review_config({"baseline": None}, "epic_completion")

        assert result is not None
        # Explicit null counts as missing since baseline is required
        assert result.baseline == "since_run_start"
        assert "baseline not specified for epic_completion trigger" in caplog.text


class TestCodeReviewFullConfig:
    """Tests for complete code_review configurations."""

    def test_full_config_session_end(self) -> None:
        """Full code_review config for session_end."""
        result = _parse_code_review_config(
            {
                "enabled": True,
                "reviewer_type": "cerberus",
                "failure_mode": "remediate",
                "max_retries": 5,
                "finding_threshold": "P1",
                "cerberus": {
                    "timeout": 600,
                    "spawn_args": ["--spawn-flag"],
                    "wait_args": ["--wait-flag"],
                    "env": {"MY_VAR": "value"},
                },
            },
            "session_end",
        )

        assert result is not None
        assert result.enabled is True
        assert result.reviewer_type == "cerberus"
        assert result.failure_mode == FailureMode.REMEDIATE
        assert result.max_retries == 5
        assert result.finding_threshold == "P1"
        assert result.baseline is None
        assert result.cerberus is not None
        assert result.cerberus.timeout == 600
        assert result.cerberus.spawn_args == ("--spawn-flag",)
        assert result.cerberus.wait_args == ("--wait-flag",)
        assert result.cerberus.env == (("MY_VAR", "value"),)

    def test_full_config_epic_completion(self) -> None:
        """Full code_review config for epic_completion."""
        result = _parse_code_review_config(
            {
                "enabled": True,
                "reviewer_type": "agent_sdk",
                "failure_mode": "abort",
                "max_retries": 2,
                "finding_threshold": "P0",
                "baseline": "since_last_review",
            },
            "epic_completion",
        )

        assert result is not None
        assert result.enabled is True
        assert result.reviewer_type == "agent_sdk"
        assert result.failure_mode == FailureMode.ABORT
        assert result.max_retries == 2
        assert result.finding_threshold == "P0"
        assert result.baseline == "since_last_review"
        assert result.cerberus is None

    def test_full_config_run_end(self) -> None:
        """Full code_review config for run_end."""
        result = _parse_code_review_config(
            {
                "enabled": True,
                "reviewer_type": "cerberus",
                "failure_mode": "continue",
                "max_retries": 0,
                "finding_threshold": "P2",
                "baseline": "since_run_start",
                "cerberus": {"timeout": 300},
            },
            "run_end",
        )

        assert result is not None
        assert result.enabled is True
        assert result.reviewer_type == "cerberus"
        assert result.failure_mode == FailureMode.CONTINUE
        assert result.max_retries == 0
        assert result.finding_threshold == "P2"
        assert result.baseline == "since_run_start"
        assert result.cerberus is not None
        assert result.cerberus.timeout == 300


class TestParsePeriodicTrigger:
    """Tests for _parse_periodic_trigger function."""

    def test_minimal_config(self) -> None:
        """Minimal periodic config with required fields."""
        result = _parse_periodic_trigger(
            {
                "interval": 60,
                "failure_mode": "abort",
            }
        )
        assert result.interval == 60
        assert result.failure_mode == FailureMode.ABORT
        assert result.commands == ()
        assert result.max_retries is None
        assert result.code_review is None

    def test_with_code_review(self) -> None:
        """Periodic trigger parses code_review configuration."""
        result = _parse_periodic_trigger(
            {
                "interval": 300,
                "failure_mode": "continue",
                "code_review": {
                    "enabled": True,
                    "reviewer_type": "cerberus",
                },
            }
        )
        assert result.interval == 300
        assert result.failure_mode == FailureMode.CONTINUE
        assert result.code_review is not None
        assert result.code_review.enabled is True
        assert result.code_review.reviewer_type == "cerberus"

    def test_code_review_null_ignored(self) -> None:
        """code_review: null is treated as no code review."""
        result = _parse_periodic_trigger(
            {
                "interval": 120,
                "failure_mode": "abort",
                "code_review": None,
            }
        )
        assert result.code_review is None

    def test_unknown_field_rejected(self) -> None:
        """Unknown field raises ConfigError."""
        with pytest.raises(
            ConfigError, match=r"Unknown field 'bogus' in trigger periodic"
        ):
            _parse_periodic_trigger(
                {
                    "interval": 60,
                    "failure_mode": "abort",
                    "bogus": True,
                }
            )

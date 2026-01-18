"""Unit tests for _parse_epic_verification_config function.

Tests parsing of the epic_verification section in mala.yaml including:
- Valid config with all fields
- Default values when section/fields omitted
- Rejection of invalid reviewer_type values
- Rejection of unknown fields
- Type validation for all fields
"""

from __future__ import annotations

import textwrap

import pytest
import yaml

from src.domain.validation.config import (
    CerberusConfig,
    ConfigError,
    EpicVerifierConfig,
    FailureMode,
)
from src.domain.validation.config_loader import _parse_epic_verification_config


class TestParseEpicVerificationConfigDefaults:
    """Tests for default values when fields are omitted."""

    def test_returns_defaults_when_data_is_none(self) -> None:
        """Parser returns default config when data is None."""
        config = _parse_epic_verification_config(None)
        assert config.enabled is True
        assert config.reviewer_type == "agent_sdk"
        assert config.timeout == 600
        assert config.max_retries == 3
        assert config.failure_mode == FailureMode.CONTINUE
        assert config.cerberus is None
        assert config.agent_sdk_timeout == 600
        assert config.agent_sdk_model == "sonnet"

    def test_returns_defaults_when_data_is_empty_dict(self) -> None:
        """Parser returns defaults for empty dict."""
        config = _parse_epic_verification_config({})
        assert config.enabled is True
        assert config.reviewer_type == "agent_sdk"
        assert config.timeout == 600
        assert config.max_retries == 3
        assert config.failure_mode == FailureMode.CONTINUE
        assert config.cerberus is None
        assert config.agent_sdk_timeout == 600
        assert config.agent_sdk_model == "sonnet"

    def test_defaults_when_individual_fields_omitted(self) -> None:
        """Parser applies defaults for omitted individual fields."""
        config = _parse_epic_verification_config({"reviewer_type": "cerberus"})
        assert config.enabled is True
        assert config.reviewer_type == "cerberus"
        assert config.timeout == 600
        assert config.max_retries == 3
        assert config.failure_mode == FailureMode.CONTINUE
        assert config.cerberus is None
        assert config.agent_sdk_timeout == 600
        assert config.agent_sdk_model == "sonnet"


class TestParseEpicVerificationConfigValidInput:
    """Tests for valid configuration input."""

    def test_parses_all_fields(self) -> None:
        """Parser correctly parses all fields."""
        config = _parse_epic_verification_config(
            {
                "enabled": False,
                "reviewer_type": "cerberus",
                "timeout": 300,
                "max_retries": 5,
                "failure_mode": "abort",
                "cerberus": {"timeout": 120},
                "agent_sdk_timeout": 900,
                "agent_sdk_model": "opus",
            }
        )
        assert config.enabled is False
        assert config.reviewer_type == "cerberus"
        assert config.timeout == 300
        assert config.max_retries == 5
        assert config.failure_mode == FailureMode.ABORT
        assert config.cerberus is not None
        assert config.cerberus.timeout == 120
        assert config.agent_sdk_timeout == 900
        assert config.agent_sdk_model == "opus"

    def test_parses_reviewer_type_agent_sdk(self) -> None:
        """Parser correctly parses reviewer_type: agent_sdk."""
        config = _parse_epic_verification_config({"reviewer_type": "agent_sdk"})
        assert config.reviewer_type == "agent_sdk"

    def test_parses_reviewer_type_cerberus(self) -> None:
        """Parser correctly parses reviewer_type: cerberus."""
        config = _parse_epic_verification_config({"reviewer_type": "cerberus"})
        assert config.reviewer_type == "cerberus"

    def test_parses_enabled_true(self) -> None:
        """Parser correctly parses enabled: true."""
        config = _parse_epic_verification_config({"enabled": True})
        assert config.enabled is True

    def test_parses_enabled_false(self) -> None:
        """Parser correctly parses enabled: false."""
        config = _parse_epic_verification_config({"enabled": False})
        assert config.enabled is False

    def test_parses_failure_mode_continue(self) -> None:
        """Parser correctly parses failure_mode: continue."""
        config = _parse_epic_verification_config({"failure_mode": "continue"})
        assert config.failure_mode == FailureMode.CONTINUE

    def test_parses_failure_mode_abort(self) -> None:
        """Parser correctly parses failure_mode: abort."""
        config = _parse_epic_verification_config({"failure_mode": "abort"})
        assert config.failure_mode == FailureMode.ABORT

    def test_parses_agent_sdk_model_sonnet(self) -> None:
        """Parser correctly parses agent_sdk_model: sonnet."""
        config = _parse_epic_verification_config({"agent_sdk_model": "sonnet"})
        assert config.agent_sdk_model == "sonnet"

    def test_parses_agent_sdk_model_opus(self) -> None:
        """Parser correctly parses agent_sdk_model: opus."""
        config = _parse_epic_verification_config({"agent_sdk_model": "opus"})
        assert config.agent_sdk_model == "opus"

    def test_parses_agent_sdk_model_haiku(self) -> None:
        """Parser correctly parses agent_sdk_model: haiku."""
        config = _parse_epic_verification_config({"agent_sdk_model": "haiku"})
        assert config.agent_sdk_model == "haiku"

    def test_parses_cerberus_block_with_all_fields(self) -> None:
        """Parser correctly parses full cerberus block."""
        config = _parse_epic_verification_config(
            {
                "cerberus": {
                    "timeout": 180,
                    "spawn_args": ["--flag1", "--flag2"],
                    "wait_args": ["--timeout=60"],
                    "env": {"MY_VAR": "value"},
                }
            }
        )
        assert config.cerberus is not None
        assert config.cerberus.timeout == 180
        assert config.cerberus.spawn_args == ("--flag1", "--flag2")
        assert config.cerberus.wait_args == ("--timeout=60",)
        assert config.cerberus.env == (("MY_VAR", "value"),)

    def test_parses_cerberus_null_returns_none(self) -> None:
        """Parser returns None cerberus when set to null."""
        config = _parse_epic_verification_config({"cerberus": None})
        assert config.cerberus is None

    def test_parses_zero_timeout(self) -> None:
        """Parser accepts zero timeout."""
        config = _parse_epic_verification_config({"timeout": 0})
        assert config.timeout == 0

    def test_parses_zero_max_retries(self) -> None:
        """Parser accepts zero max_retries."""
        config = _parse_epic_verification_config({"max_retries": 0})
        assert config.max_retries == 0


class TestParseEpicVerificationConfigInvalidInput:
    """Tests for invalid configuration input."""

    def test_rejects_non_dict_data(self) -> None:
        """Parser raises ConfigError when data is not a dict."""
        with pytest.raises(ConfigError, match="must be an object"):
            _parse_epic_verification_config("not a dict")  # type: ignore[arg-type]

    def test_rejects_unknown_fields(self) -> None:
        """Parser raises ConfigError for unknown fields."""
        with pytest.raises(ConfigError, match="Unknown field 'unknown'"):
            _parse_epic_verification_config({"unknown": "value"})

    def test_rejects_unknown_fields_sorted_alphabetically(self) -> None:
        """Parser reports first unknown field alphabetically."""
        with pytest.raises(ConfigError, match="Unknown field 'aaa'"):
            _parse_epic_verification_config({"zzz": 1, "aaa": 2})

    def test_rejects_invalid_reviewer_type(self) -> None:
        """Parser raises ConfigError for invalid reviewer_type."""
        with pytest.raises(ConfigError, match="must be 'cerberus' or 'agent_sdk'"):
            _parse_epic_verification_config({"reviewer_type": "invalid"})

    def test_rejects_enabled_non_boolean(self) -> None:
        """Parser raises ConfigError when enabled is not boolean."""
        with pytest.raises(ConfigError, match="enabled must be a boolean"):
            _parse_epic_verification_config({"enabled": "yes"})

    def test_rejects_enabled_integer(self) -> None:
        """Parser raises ConfigError when enabled is integer."""
        with pytest.raises(ConfigError, match="enabled must be a boolean"):
            _parse_epic_verification_config({"enabled": 1})

    def test_rejects_timeout_non_integer(self) -> None:
        """Parser raises ConfigError when timeout is not integer."""
        with pytest.raises(ConfigError, match="timeout must be an integer"):
            _parse_epic_verification_config({"timeout": "300"})

    def test_rejects_timeout_boolean(self) -> None:
        """Parser raises ConfigError when timeout is boolean."""
        with pytest.raises(ConfigError, match="timeout must be an integer"):
            _parse_epic_verification_config({"timeout": True})

    def test_rejects_timeout_negative(self) -> None:
        """Parser raises ConfigError when timeout is negative."""
        with pytest.raises(ConfigError, match="timeout must be non-negative"):
            _parse_epic_verification_config({"timeout": -1})

    def test_rejects_max_retries_non_integer(self) -> None:
        """Parser raises ConfigError when max_retries is not integer."""
        with pytest.raises(ConfigError, match="max_retries must be an integer"):
            _parse_epic_verification_config({"max_retries": "3"})

    def test_rejects_max_retries_boolean(self) -> None:
        """Parser raises ConfigError when max_retries is boolean."""
        with pytest.raises(ConfigError, match="max_retries must be an integer"):
            _parse_epic_verification_config({"max_retries": True})

    def test_rejects_max_retries_negative(self) -> None:
        """Parser raises ConfigError when max_retries is negative."""
        with pytest.raises(ConfigError, match="max_retries must be non-negative"):
            _parse_epic_verification_config({"max_retries": -1})

    def test_rejects_failure_mode_non_string(self) -> None:
        """Parser raises ConfigError when failure_mode is not string."""
        with pytest.raises(ConfigError, match="failure_mode must be a string"):
            _parse_epic_verification_config({"failure_mode": 123})

    def test_rejects_failure_mode_invalid_value(self) -> None:
        """Parser raises ConfigError for invalid failure_mode."""
        with pytest.raises(ConfigError, match="failure_mode must be one of"):
            _parse_epic_verification_config({"failure_mode": "invalid"})

    def test_rejects_cerberus_non_object(self) -> None:
        """Parser raises ConfigError when cerberus is not object."""
        with pytest.raises(ConfigError, match="cerberus must be an object"):
            _parse_epic_verification_config({"cerberus": "not-a-dict"})

    def test_rejects_agent_sdk_timeout_non_integer(self) -> None:
        """Parser raises ConfigError when agent_sdk_timeout is not integer."""
        with pytest.raises(ConfigError, match="agent_sdk_timeout must be an integer"):
            _parse_epic_verification_config({"agent_sdk_timeout": "600"})

    def test_rejects_agent_sdk_timeout_boolean(self) -> None:
        """Parser raises ConfigError when agent_sdk_timeout is boolean."""
        with pytest.raises(ConfigError, match="agent_sdk_timeout must be an integer"):
            _parse_epic_verification_config({"agent_sdk_timeout": False})

    def test_rejects_agent_sdk_timeout_negative(self) -> None:
        """Parser raises ConfigError when agent_sdk_timeout is negative."""
        with pytest.raises(ConfigError, match="agent_sdk_timeout must be non-negative"):
            _parse_epic_verification_config({"agent_sdk_timeout": -10})

    def test_rejects_invalid_agent_sdk_model(self) -> None:
        """Parser raises ConfigError for invalid agent_sdk_model."""
        with pytest.raises(
            ConfigError, match="agent_sdk_model must be 'sonnet', 'opus', or 'haiku'"
        ):
            _parse_epic_verification_config({"agent_sdk_model": "gpt-4"})


class TestParseEpicVerificationConfigFromYaml:
    """Tests loading config from YAML string to verify non-default values."""

    def test_yaml_with_all_non_default_values(self) -> None:
        """Load from YAML string with all non-default values."""
        yaml_content = textwrap.dedent("""
            enabled: false
            reviewer_type: cerberus
            timeout: 1200
            max_retries: 10
            failure_mode: abort
            cerberus:
              timeout: 300
              spawn_args:
                - "--verbose"
              wait_args:
                - "--timeout=120"
              env:
                DEBUG: "true"
            agent_sdk_timeout: 1800
            agent_sdk_model: opus
        """)
        data = yaml.safe_load(yaml_content)
        config = _parse_epic_verification_config(data)

        assert config.enabled is False
        assert config.reviewer_type == "cerberus"
        assert config.timeout == 1200
        assert config.max_retries == 10
        assert config.failure_mode == FailureMode.ABORT
        assert config.cerberus is not None
        assert config.cerberus.timeout == 300
        assert config.cerberus.spawn_args == ("--verbose",)
        assert config.cerberus.wait_args == ("--timeout=120",)
        assert config.cerberus.env == (("DEBUG", "true"),)
        assert config.agent_sdk_timeout == 1800
        assert config.agent_sdk_model == "opus"

    def test_yaml_with_minimal_config(self) -> None:
        """Load from YAML with only reviewer_type set."""
        yaml_content = "reviewer_type: cerberus"
        data = yaml.safe_load(yaml_content)
        config = _parse_epic_verification_config(data)

        assert config.reviewer_type == "cerberus"
        # All other fields should be defaults
        assert config.enabled is True
        assert config.timeout == 600
        assert config.max_retries == 3

    def test_yaml_empty_produces_defaults(self) -> None:
        """Empty YAML produces default config."""
        yaml_content = "{}"
        data = yaml.safe_load(yaml_content)
        config = _parse_epic_verification_config(data)

        assert config.enabled is True
        assert config.reviewer_type == "agent_sdk"

    def test_yaml_null_produces_defaults(self) -> None:
        """YAML null produces default config."""
        yaml_content = "null"
        data = yaml.safe_load(yaml_content)
        config = _parse_epic_verification_config(data)

        assert config.enabled is True
        assert config.reviewer_type == "agent_sdk"


class TestEpicVerifierConfigDataclass:
    """Tests for the EpicVerifierConfig dataclass itself."""

    def test_dataclass_has_all_expected_fields(self) -> None:
        """EpicVerifierConfig has all expected fields."""
        config = EpicVerifierConfig()
        assert hasattr(config, "enabled")
        assert hasattr(config, "reviewer_type")
        assert hasattr(config, "timeout")
        assert hasattr(config, "max_retries")
        assert hasattr(config, "failure_mode")
        assert hasattr(config, "cerberus")
        assert hasattr(config, "agent_sdk_timeout")
        assert hasattr(config, "agent_sdk_model")

    def test_dataclass_defaults(self) -> None:
        """EpicVerifierConfig has correct defaults."""
        config = EpicVerifierConfig()
        assert config.enabled is True
        assert config.reviewer_type == "agent_sdk"
        assert config.timeout == 600
        assert config.max_retries == 3
        assert config.failure_mode == FailureMode.CONTINUE
        assert config.cerberus is None
        assert config.agent_sdk_timeout == 600
        assert config.agent_sdk_model == "sonnet"

    def test_dataclass_is_frozen(self) -> None:
        """EpicVerifierConfig is immutable (frozen)."""
        config = EpicVerifierConfig()
        with pytest.raises(AttributeError):
            config.enabled = False  # type: ignore[misc]

    def test_dataclass_can_be_created_with_all_fields(self) -> None:
        """EpicVerifierConfig can be created with all field values."""
        cerberus = CerberusConfig(timeout=120)
        config = EpicVerifierConfig(
            enabled=False,
            reviewer_type="cerberus",
            timeout=300,
            max_retries=5,
            failure_mode=FailureMode.ABORT,
            cerberus=cerberus,
            agent_sdk_timeout=900,
            agent_sdk_model="haiku",
        )
        assert config.enabled is False
        assert config.reviewer_type == "cerberus"
        assert config.timeout == 300
        assert config.max_retries == 5
        assert config.failure_mode == FailureMode.ABORT
        assert config.cerberus is cerberus
        assert config.agent_sdk_timeout == 900
        assert config.agent_sdk_model == "haiku"

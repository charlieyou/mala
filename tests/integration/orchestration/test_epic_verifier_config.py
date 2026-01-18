"""Integration tests for EpicVerifierConfig and factory wiring.

This test verifies:
1. EpicVerifierConfig dataclass is properly defined
2. _parse_epic_verification_config parses config correctly
3. _create_epic_verification_model returns ClaudeEpicVerificationModel for agent_sdk
4. _check_epic_verifier_availability returns correct availability status
5. ValidationConfig.from_dict correctly parses epic_verification block
6. epic_verification field is in _ALLOWED_TOP_LEVEL_FIELDS

The test exercises: mala.yaml (epic_verification) → _parse_epic_verification_config →
EpicVerifierConfig → _create_epic_verification_model → EpicVerificationModel
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


class TestEpicVerifierConfigDataclass:
    """Test EpicVerifierConfig dataclass structure."""

    def test_default_reviewer_type_is_agent_sdk(self) -> None:
        """EpicVerifierConfig defaults to agent_sdk reviewer."""
        from src.domain.validation.config import EpicVerifierConfig

        config = EpicVerifierConfig()
        assert config.reviewer_type == "agent_sdk"

    def test_can_create_with_cerberus_reviewer_type(self) -> None:
        """EpicVerifierConfig accepts cerberus reviewer_type."""
        from src.domain.validation.config import EpicVerifierConfig

        config = EpicVerifierConfig(reviewer_type="cerberus")
        assert config.reviewer_type == "cerberus"

    def test_is_frozen(self) -> None:
        """EpicVerifierConfig is immutable (frozen)."""
        from src.domain.validation.config import EpicVerifierConfig

        config = EpicVerifierConfig()
        with pytest.raises(AttributeError):
            config.reviewer_type = "cerberus"  # type: ignore[misc]


class TestParseEpicVerificationConfig:
    """Test _parse_epic_verification_config function."""

    def test_returns_defaults_when_data_is_none(self) -> None:
        """Parser returns default config when data is None."""
        from src.domain.validation.config_loader import _parse_epic_verification_config

        config = _parse_epic_verification_config(None)
        assert config.reviewer_type == "agent_sdk"

    def test_parses_reviewer_type_agent_sdk(self) -> None:
        """Parser correctly parses reviewer_type: agent_sdk."""
        from src.domain.validation.config_loader import _parse_epic_verification_config

        config = _parse_epic_verification_config({"reviewer_type": "agent_sdk"})
        assert config.reviewer_type == "agent_sdk"

    def test_parses_reviewer_type_cerberus(self) -> None:
        """Parser correctly parses reviewer_type: cerberus."""
        from src.domain.validation.config_loader import _parse_epic_verification_config

        config = _parse_epic_verification_config({"reviewer_type": "cerberus"})
        assert config.reviewer_type == "cerberus"

    def test_rejects_invalid_reviewer_type(self) -> None:
        """Parser raises ConfigError for invalid reviewer_type."""
        from src.domain.validation.config import ConfigError
        from src.domain.validation.config_loader import _parse_epic_verification_config

        with pytest.raises(ConfigError, match="must be 'cerberus' or 'agent_sdk'"):
            _parse_epic_verification_config({"reviewer_type": "invalid"})

    def test_rejects_unknown_fields(self) -> None:
        """Parser raises ConfigError for unknown fields."""
        from src.domain.validation.config import ConfigError
        from src.domain.validation.config_loader import _parse_epic_verification_config

        with pytest.raises(ConfigError, match="Unknown field 'unknown'"):
            _parse_epic_verification_config({"unknown": "value"})

    def test_rejects_non_dict_data(self) -> None:
        """Parser raises ConfigError when data is not a dict."""
        from src.domain.validation.config import ConfigError
        from src.domain.validation.config_loader import _parse_epic_verification_config

        with pytest.raises(ConfigError, match="must be an object"):
            _parse_epic_verification_config("not a dict")  # type: ignore[arg-type]


class TestCheckEpicVerifierAvailability:
    """Test _check_epic_verifier_availability function."""

    def test_agent_sdk_always_available(self) -> None:
        """agent_sdk verifier is always available."""
        from src.orchestration.factory import _check_epic_verifier_availability

        result = _check_epic_verifier_availability("agent_sdk")
        assert result is None

    def test_cerberus_not_yet_implemented(self) -> None:
        """cerberus verifier returns 'not yet implemented' reason."""
        from src.orchestration.factory import _check_epic_verifier_availability

        result = _check_epic_verifier_availability("cerberus")
        assert result is not None
        assert "not yet implemented" in result

    def test_unknown_reviewer_type_returns_error(self) -> None:
        """Unknown reviewer_type returns error reason."""
        from src.orchestration.factory import _check_epic_verifier_availability

        result = _check_epic_verifier_availability("unknown")
        assert result is not None
        assert "unknown" in result


class TestCreateEpicVerificationModel:
    """Test _create_epic_verification_model function."""

    def test_agent_sdk_returns_claude_model(self, tmp_path: Path) -> None:
        """agent_sdk creates ClaudeEpicVerificationModel."""
        from src.infra.epic_verifier import ClaudeEpicVerificationModel
        from src.orchestration.factory import _create_epic_verification_model

        model = _create_epic_verification_model(
            reviewer_type="agent_sdk",
            repo_path=tmp_path,
            timeout_ms=60000,
        )
        assert isinstance(model, ClaudeEpicVerificationModel)

    def test_cerberus_raises_not_implemented(self, tmp_path: Path) -> None:
        """cerberus raises NotImplementedError (stub behavior)."""
        from src.orchestration.factory import _create_epic_verification_model

        with pytest.raises(
            NotImplementedError, match="Cerberus-based epic verification"
        ):
            _create_epic_verification_model(
                reviewer_type="cerberus",
                repo_path=tmp_path,
                timeout_ms=60000,
            )

    def test_unknown_reviewer_type_raises_value_error(self, tmp_path: Path) -> None:
        """Unknown reviewer_type raises ValueError."""
        from src.orchestration.factory import _create_epic_verification_model

        with pytest.raises(ValueError, match="Unknown epic verification reviewer_type"):
            _create_epic_verification_model(
                reviewer_type="invalid",
                repo_path=tmp_path,
                timeout_ms=60000,
            )


class TestValidationConfigEpicVerification:
    """Test epic_verification field in ValidationConfig."""

    def test_epic_verification_in_allowed_fields(self) -> None:
        """epic_verification is in _ALLOWED_TOP_LEVEL_FIELDS."""
        from src.domain.validation.config_loader import _ALLOWED_TOP_LEVEL_FIELDS

        assert "epic_verification" in _ALLOWED_TOP_LEVEL_FIELDS

    def test_validation_config_has_epic_verification_field(self) -> None:
        """ValidationConfig has epic_verification field with default."""
        from src.domain.validation.config import EpicVerifierConfig, ValidationConfig

        config = ValidationConfig()
        assert hasattr(config, "epic_verification")
        assert isinstance(config.epic_verification, EpicVerifierConfig)
        assert config.epic_verification.reviewer_type == "agent_sdk"

    def test_validation_config_from_dict_parses_epic_verification(self) -> None:
        """ValidationConfig.from_dict correctly parses epic_verification block."""
        from src.domain.validation.config import ValidationConfig

        data = {
            "epic_verification": {"reviewer_type": "agent_sdk"},
            "commands": {"test": {"command": "echo test"}},
        }
        config = ValidationConfig.from_dict(data)
        assert config.epic_verification.reviewer_type == "agent_sdk"

    def test_validation_config_from_dict_parses_cerberus_reviewer(self) -> None:
        """ValidationConfig.from_dict parses cerberus reviewer_type."""
        from src.domain.validation.config import ValidationConfig

        data = {
            "epic_verification": {"reviewer_type": "cerberus"},
            "commands": {"test": {"command": "echo test"}},
        }
        config = ValidationConfig.from_dict(data)
        assert config.epic_verification.reviewer_type == "cerberus"

    def test_validation_config_from_dict_defaults_without_epic_verification(
        self,
    ) -> None:
        """ValidationConfig.from_dict defaults epic_verification when absent."""
        from src.domain.validation.config import ValidationConfig

        data = {"commands": {"test": {"command": "echo test"}}}
        config = ValidationConfig.from_dict(data)
        assert config.epic_verification.reviewer_type == "agent_sdk"


@pytest.mark.integration
class TestEpicVerifierConfigIntegration:
    """Integration test for the full config → factory → model path.

    This test exercises the wiring from config parsing through to model creation,
    verifying the complete integration path is functional.
    """

    def test_full_path_agent_sdk(self, tmp_path: Path) -> None:
        """Integration: parse config → check availability → create model for agent_sdk."""
        from src.domain.validation.config_loader import _parse_epic_verification_config
        from src.infra.epic_verifier import ClaudeEpicVerificationModel
        from src.orchestration.factory import (
            _check_epic_verifier_availability,
            _create_epic_verification_model,
        )

        # Step 1: Parse config
        config = _parse_epic_verification_config({"reviewer_type": "agent_sdk"})
        assert config.reviewer_type == "agent_sdk"

        # Step 2: Check availability
        unavailable_reason = _check_epic_verifier_availability(config.reviewer_type)
        assert unavailable_reason is None

        # Step 3: Create model
        model = _create_epic_verification_model(
            reviewer_type=config.reviewer_type,
            repo_path=tmp_path,
            timeout_ms=60000,
        )
        assert isinstance(model, ClaudeEpicVerificationModel)

    def test_full_path_cerberus_fails_expected(self, tmp_path: Path) -> None:
        """Integration: cerberus path fails with expected NotImplementedError.

        This test documents the expected behavior until T003 completes
        CerberusEpicVerifier implementation.
        """
        from src.domain.validation.config_loader import _parse_epic_verification_config
        from src.orchestration.factory import (
            _check_epic_verifier_availability,
            _create_epic_verification_model,
        )

        # Step 1: Parse config
        config = _parse_epic_verification_config({"reviewer_type": "cerberus"})
        assert config.reviewer_type == "cerberus"

        # Step 2: Check availability - returns reason (not available yet)
        unavailable_reason = _check_epic_verifier_availability(config.reviewer_type)
        assert unavailable_reason is not None
        assert "not yet implemented" in unavailable_reason

        # Step 3: Create model - raises NotImplementedError
        with pytest.raises(
            NotImplementedError, match="Cerberus-based epic verification"
        ):
            _create_epic_verification_model(
                reviewer_type=config.reviewer_type,
                repo_path=tmp_path,
                timeout_ms=60000,
            )

    def test_validation_config_to_factory_path(self, tmp_path: Path) -> None:
        """Integration: ValidationConfig.from_dict → factory functions."""
        from src.domain.validation.config import ValidationConfig
        from src.infra.epic_verifier import ClaudeEpicVerificationModel
        from src.orchestration.factory import (
            _check_epic_verifier_availability,
            _create_epic_verification_model,
        )

        # Step 1: Parse ValidationConfig from dict (as if from mala.yaml)
        data = {
            "epic_verification": {"reviewer_type": "agent_sdk"},
            "commands": {"test": {"command": "echo test"}},
        }
        validation_config = ValidationConfig.from_dict(data)

        # Step 2: Extract reviewer_type (mimics create_orchestrator behavior)
        reviewer_type = validation_config.epic_verification.reviewer_type

        # Step 3: Check availability and create model
        unavailable_reason = _check_epic_verifier_availability(reviewer_type)
        assert unavailable_reason is None

        model = _create_epic_verification_model(
            reviewer_type=reviewer_type,
            repo_path=tmp_path,
            timeout_ms=60000,
        )
        assert isinstance(model, ClaudeEpicVerificationModel)

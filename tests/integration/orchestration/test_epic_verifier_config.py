"""Integration tests for EpicVerificationConfig and factory wiring.

This test verifies:
1. EpicVerificationConfig dataclass is properly defined
2. _parse_epic_verification_config parses config correctly
3. _create_epic_verification_model returns ClaudeEpicVerificationModel for agent_sdk
4. _check_epic_verifier_availability returns correct availability status

The test exercises: mala.yaml (epic_verification) → _parse_epic_verification_config →
EpicVerificationConfig → _create_epic_verification_model → EpicVerificationModel
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


class TestEpicVerificationConfigDataclass:
    """Test EpicVerificationConfig dataclass structure."""

    def test_default_reviewer_type_is_agent_sdk(self) -> None:
        """EpicVerificationConfig defaults to agent_sdk reviewer."""
        from src.domain.validation.config import EpicVerificationConfig

        config = EpicVerificationConfig()
        assert config.reviewer_type == "agent_sdk"

    def test_can_create_with_cerberus_reviewer_type(self) -> None:
        """EpicVerificationConfig accepts cerberus reviewer_type."""
        from src.domain.validation.config import EpicVerificationConfig

        config = EpicVerificationConfig(reviewer_type="cerberus")
        assert config.reviewer_type == "cerberus"

    def test_is_frozen(self) -> None:
        """EpicVerificationConfig is immutable (frozen)."""
        from src.domain.validation.config import EpicVerificationConfig

        config = EpicVerificationConfig()
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

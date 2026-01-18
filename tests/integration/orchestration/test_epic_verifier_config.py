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

    def test_cerberus_unavailable_when_binary_missing(self) -> None:
        """cerberus verifier returns unavailable when review-gate binary missing."""
        from src.orchestration.factory import _check_epic_verifier_availability

        # Without mala_config, binary won't be found
        result = _check_epic_verifier_availability("cerberus")
        assert result is not None
        assert "review-gate unavailable" in result or "not detected" in result

    def test_cerberus_unavailable_when_spawn_epic_review_not_supported(
        self, tmp_path: Path
    ) -> None:
        """cerberus returns unavailable when spawn-epic-verify subcommand fails."""
        import os
        import stat

        from src.infra.io.config import MalaConfig
        from src.orchestration.factory import _check_epic_verifier_availability

        # Create a fake review-gate binary that fails spawn-epic-verify --help
        bin_path = tmp_path / "bin"
        bin_path.mkdir()
        review_gate = bin_path / "review-gate"
        review_gate.write_text(
            '#!/usr/bin/env sh\necho "unknown subcommand" >&2; exit 1\n'
        )
        os.chmod(review_gate, stat.S_IRWXU)

        mala_config = MalaConfig(cerberus_bin_path=bin_path)
        result = _check_epic_verifier_availability("cerberus", mala_config=mala_config)
        assert result is not None
        assert "does not support epic verification" in result

    def test_cerberus_available_when_spawn_epic_review_works(
        self, tmp_path: Path
    ) -> None:
        """cerberus verifier is available when spawn-epic-verify --help succeeds."""
        import os
        import stat

        from src.infra.io.config import MalaConfig
        from src.orchestration.factory import _check_epic_verifier_availability

        # Create a fake review-gate binary that supports spawn-epic-verify --help
        bin_path = tmp_path / "bin"
        bin_path.mkdir()
        review_gate = bin_path / "review-gate"
        review_gate.write_text(
            '#!/usr/bin/env sh\ncase "$1" in spawn-epic-verify) echo "Usage: spawn-epic-verify"; exit 0;; *) exit 1;; esac\n'
        )
        os.chmod(review_gate, stat.S_IRWXU)

        mala_config = MalaConfig(cerberus_bin_path=bin_path)
        result = _check_epic_verifier_availability("cerberus", mala_config=mala_config)
        assert result is None

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

    def test_cerberus_returns_cerberus_verifier(self, tmp_path: Path) -> None:
        """cerberus creates CerberusEpicVerifier."""
        from src.infra.clients.cerberus_epic_verifier import CerberusEpicVerifier
        from src.orchestration.factory import _create_epic_verification_model

        model = _create_epic_verification_model(
            reviewer_type="cerberus",
            repo_path=tmp_path,
            timeout_ms=60000,
        )
        assert isinstance(model, CerberusEpicVerifier)

    def test_cerberus_uses_mala_config(self, tmp_path: Path) -> None:
        """cerberus uses mala_config for bin_path and env."""
        from src.infra.clients.cerberus_epic_verifier import CerberusEpicVerifier
        from src.infra.io.config import MalaConfig
        from src.orchestration.factory import _create_epic_verification_model

        bin_path = tmp_path / "bin"
        mala_config = MalaConfig(
            cerberus_bin_path=bin_path,
            cerberus_env=(("TEST_VAR", "test_value"),),
        )

        model = _create_epic_verification_model(
            reviewer_type="cerberus",
            repo_path=tmp_path,
            timeout_ms=60000,
            mala_config=mala_config,
        )
        assert isinstance(model, CerberusEpicVerifier)
        assert model.bin_path == bin_path
        assert model.env == {"TEST_VAR": "test_value"}

    def test_cerberus_prefers_cerberus_config(self, tmp_path: Path) -> None:
        """cerberus prefers cerberus_config over mala_config for timeout/env/args."""
        from src.domain.validation.config import CerberusConfig
        from src.infra.clients.cerberus_epic_verifier import CerberusEpicVerifier
        from src.infra.io.config import MalaConfig
        from src.orchestration.factory import _create_epic_verification_model

        bin_path = tmp_path / "bin"
        mala_config = MalaConfig(
            cerberus_bin_path=bin_path,
            cerberus_env=(("OLD_VAR", "old_value"),),
        )
        cerberus_config = CerberusConfig(
            timeout=120,
            spawn_args=("--mode", "fast"),
            wait_args=("--timeout", "99"),
            env=(("NEW_VAR", "new_value"),),
        )

        model = _create_epic_verification_model(
            reviewer_type="cerberus",
            repo_path=tmp_path,
            timeout_ms=60000,
            mala_config=mala_config,
            cerberus_config=cerberus_config,
        )
        assert isinstance(model, CerberusEpicVerifier)
        # bin_path comes from mala_config (cerberus_config doesn't have it)
        assert model.bin_path == bin_path
        # timeout and env come from cerberus_config
        assert model.timeout == 120
        assert model.env == {"NEW_VAR": "new_value"}
        assert model.spawn_args == ("--mode", "fast")
        assert model.wait_args == ("--timeout", "99")

    def test_unknown_reviewer_type_raises_value_error(self, tmp_path: Path) -> None:
        """Unknown reviewer_type raises ValueError."""
        from src.orchestration.factory import _create_epic_verification_model

        with pytest.raises(ValueError, match="Unknown epic verification reviewer_type"):
            _create_epic_verification_model(
                reviewer_type="invalid",
                repo_path=tmp_path,
                timeout_ms=60000,
            )

    def test_agent_sdk_uses_timeout_ms_parameter(self, tmp_path: Path) -> None:
        """agent_sdk model uses the timeout_ms parameter correctly."""
        from src.infra.epic_verifier import ClaudeEpicVerificationModel
        from src.orchestration.factory import _create_epic_verification_model

        # Pass specific timeout to verify it's used (not some global default)
        model = _create_epic_verification_model(
            reviewer_type="agent_sdk",
            repo_path=tmp_path,
            timeout_ms=120000,  # 120 seconds
        )
        assert isinstance(model, ClaudeEpicVerificationModel)
        assert model.timeout_ms == 120000


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

    def test_full_path_cerberus_unavailable_without_binary(
        self, tmp_path: Path
    ) -> None:
        """Integration: cerberus path fails when binary unavailable."""
        from src.domain.validation.config_loader import _parse_epic_verification_config
        from src.orchestration.factory import _check_epic_verifier_availability

        # Step 1: Parse config
        config = _parse_epic_verification_config({"reviewer_type": "cerberus"})
        assert config.reviewer_type == "cerberus"

        # Step 2: Check availability - returns reason (binary not found)
        unavailable_reason = _check_epic_verifier_availability(config.reviewer_type)
        assert unavailable_reason is not None
        assert (
            "review-gate unavailable" in unavailable_reason
            or "not detected" in unavailable_reason
        )

    def test_full_path_cerberus_available_with_binary(self, tmp_path: Path) -> None:
        """Integration: cerberus path succeeds when binary available."""
        import os
        import stat

        from src.domain.validation.config_loader import _parse_epic_verification_config
        from src.infra.clients.cerberus_epic_verifier import CerberusEpicVerifier
        from src.infra.io.config import MalaConfig
        from src.orchestration.factory import (
            _check_epic_verifier_availability,
            _create_epic_verification_model,
        )

        # Create a fake review-gate binary
        bin_path = tmp_path / "bin"
        bin_path.mkdir()
        review_gate = bin_path / "review-gate"
        review_gate.write_text(
            '#!/usr/bin/env sh\ncase "$1" in spawn-epic-verify) echo "Usage: spawn-epic-verify"; exit 0;; *) exit 1;; esac\n'
        )
        os.chmod(review_gate, stat.S_IRWXU)

        mala_config = MalaConfig(cerberus_bin_path=bin_path)

        # Step 1: Parse config
        config = _parse_epic_verification_config({"reviewer_type": "cerberus"})
        assert config.reviewer_type == "cerberus"

        # Step 2: Check availability - should be available
        unavailable_reason = _check_epic_verifier_availability(
            config.reviewer_type, mala_config=mala_config
        )
        assert unavailable_reason is None

        # Step 3: Create model - should return CerberusEpicVerifier
        model = _create_epic_verification_model(
            reviewer_type=config.reviewer_type,
            repo_path=tmp_path,
            timeout_ms=60000,
            mala_config=mala_config,
        )
        assert isinstance(model, CerberusEpicVerifier)

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


class TestEpicVerifierTimeoutFallback:
    """Test timeout fallback logic for epic verifier creation in create_orchestrator.

    Tests verify fix for issue where EpicVerifierConfig.timeout was ignored.
    """

    def test_cerberus_without_config_uses_generic_timeout(self, tmp_path: Path) -> None:
        """When cerberus has no cerberus config, use generic timeout field."""
        import os
        import stat

        import yaml

        from src.infra.clients.cerberus_epic_verifier import CerberusEpicVerifier
        from src.infra.epic_verifier import EpicVerifier
        from src.infra.io.config import MalaConfig
        from src.orchestration.factory import create_orchestrator
        from src.orchestration.types import OrchestratorConfig

        # Create a fake review-gate binary that supports spawn-epic-verify --help
        # This allows the real _check_epic_verifier_availability to pass
        bin_path = tmp_path / "bin"
        bin_path.mkdir()
        review_gate = bin_path / "review-gate"
        review_gate.write_text(
            "#!/usr/bin/env sh\n"
            'case "$1" in spawn-epic-verify) echo "Usage: spawn-epic-verify"; exit 0;; *) exit 1;; esac\n'
        )
        os.chmod(review_gate, stat.S_IRWXU)

        # Create mala.yaml with cerberus type but no cerberus config
        # and a custom generic timeout
        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text(
            yaml.dump(
                {
                    "epic_verification": {
                        "reviewer_type": "cerberus",
                        "timeout": 120,  # Generic timeout - should be used
                        # No cerberus config
                    },
                    "commands": {"test": {"command": "echo test"}},
                }
            )
        )

        config = OrchestratorConfig(repo_path=tmp_path)
        mala_config = MalaConfig(cerberus_bin_path=bin_path)
        orchestrator = create_orchestrator(config, mala_config=mala_config)

        # Verify the epic verifier uses generic timeout (120 seconds)
        assert orchestrator.epic_verifier is not None
        assert isinstance(orchestrator.epic_verifier, EpicVerifier)
        assert isinstance(orchestrator.epic_verifier.model, CerberusEpicVerifier)
        assert orchestrator.epic_verifier.model.timeout == 120

    def test_agent_sdk_uses_agent_sdk_timeout(self, tmp_path: Path) -> None:
        """agent_sdk reviewer uses agent_sdk_timeout, not generic timeout."""
        import yaml

        from src.infra.epic_verifier import ClaudeEpicVerificationModel
        from src.orchestration.factory import create_orchestrator
        from src.orchestration.types import OrchestratorConfig

        # Create mala.yaml with agent_sdk_timeout different from generic timeout
        mala_yaml = tmp_path / "mala.yaml"
        mala_yaml.write_text(
            yaml.dump(
                {
                    "epic_verification": {
                        "reviewer_type": "agent_sdk",
                        "timeout": 100,  # Generic timeout - ignored for agent_sdk
                        "agent_sdk_timeout": 200,  # Should be used
                    },
                    "commands": {"test": {"command": "echo test"}},
                }
            )
        )

        config = OrchestratorConfig(repo_path=tmp_path)
        orchestrator = create_orchestrator(config)

        # Verify the epic verifier uses agent_sdk_timeout (200 seconds = 200000 ms)
        # EpicVerifier wraps the actual model, so access via .model attribute
        from src.infra.epic_verifier import EpicVerifier

        assert orchestrator.epic_verifier is not None
        assert isinstance(orchestrator.epic_verifier, EpicVerifier)
        assert isinstance(orchestrator.epic_verifier.model, ClaudeEpicVerificationModel)
        assert orchestrator.epic_verifier.model.timeout_ms == 200000

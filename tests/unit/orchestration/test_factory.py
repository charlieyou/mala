"""Unit tests for orchestration factory functions.

Tests for:
- _check_review_availability: Review availability checking by reviewer_type
- _derive_config: Derived configuration extraction
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.infra.io.config import MalaConfig
from src.orchestration.factory import (
    _check_review_availability,
    _derive_config,
    _extract_reviewer_config,
)
from src.orchestration.types import OrchestratorConfig


class TestCheckReviewAvailability:
    """Tests for _check_review_availability function."""

    @pytest.fixture
    def mala_config(self) -> MalaConfig:
        """Create a minimal MalaConfig for testing."""
        return MalaConfig(
            runs_dir=Path("/tmp/runs"),
            lock_dir=Path("/tmp/locks"),
        )

    def test_agent_sdk_reviewer_is_available(self, mala_config: MalaConfig) -> None:
        """agent_sdk reviewer should always be available (no external deps)."""
        result = _check_review_availability(
            mala_config=mala_config,
            disabled_validations=set(),
            reviewer_type="agent_sdk",
        )
        assert result is None

    def test_explicitly_disabled_review_returns_none(
        self, mala_config: MalaConfig
    ) -> None:
        """Explicitly disabled review should return None (no warning needed)."""
        result = _check_review_availability(
            mala_config=mala_config,
            disabled_validations={"review"},
            reviewer_type="agent_sdk",
        )
        assert result is None

    def test_unknown_reviewer_type_returns_reason(
        self, mala_config: MalaConfig
    ) -> None:
        """Unknown reviewer_type should disable review with warning."""
        result = _check_review_availability(
            mala_config=mala_config,
            disabled_validations=set(),
            reviewer_type="unknown_type",
        )
        assert result is not None
        assert "unknown reviewer_type" in result
        assert "unknown_type" in result

    def test_cerberus_without_binary_returns_reason(
        self, mala_config: MalaConfig
    ) -> None:
        """cerberus reviewer without binary should disable review."""
        # Patch shutil.which to return None (no binary found)
        with patch("shutil.which", return_value=None):
            result = _check_review_availability(
                mala_config=mala_config,
                disabled_validations=set(),
                reviewer_type="cerberus",
            )
        assert result is not None
        assert "review-gate" in result

    def test_cerberus_with_binary_is_available(self, mala_config: MalaConfig) -> None:
        """cerberus reviewer with binary available should return None."""
        # Patch shutil.which to return a path (binary found)
        with patch("shutil.which", return_value="/usr/bin/review-gate"):
            result = _check_review_availability(
                mala_config=mala_config,
                disabled_validations=set(),
                reviewer_type="cerberus",
            )
        assert result is None

    def test_cerberus_with_explicit_bin_path_existing(self) -> None:
        """cerberus reviewer with explicit bin_path to existing binary is available."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            bin_path = Path(tmpdir)
            # Create the review-gate binary
            review_gate = bin_path / "review-gate"
            review_gate.touch()
            review_gate.chmod(0o755)

            config = MalaConfig(
                runs_dir=Path("/tmp/runs"),
                lock_dir=Path("/tmp/locks"),
                cerberus_bin_path=bin_path,
            )
            result = _check_review_availability(
                mala_config=config,
                disabled_validations=set(),
                reviewer_type="cerberus",
            )
        assert result is None

    def test_cerberus_with_explicit_bin_path_missing_binary(self) -> None:
        """cerberus reviewer with explicit bin_path but missing binary is disabled."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            bin_path = Path(tmpdir)
            # Do NOT create the review-gate binary

            config = MalaConfig(
                runs_dir=Path("/tmp/runs"),
                lock_dir=Path("/tmp/locks"),
                cerberus_bin_path=bin_path,
            )
            result = _check_review_availability(
                mala_config=config,
                disabled_validations=set(),
                reviewer_type="cerberus",
            )
        assert result is not None
        assert "review-gate" in result


class TestDeriveConfig:
    """Tests for _derive_config function."""

    def test_max_gate_retries_from_session_end_config(self) -> None:
        """max_gate_retries is extracted from session_end.max_retries."""
        from src.domain.validation.config import (
            FailureMode,
            SessionEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        validation_config = ValidationConfig(
            validation_triggers=ValidationTriggersConfig(
                session_end=SessionEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(),
                    max_retries=7,
                ),
            ),
        )
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        assert derived.max_gate_retries == 7

    def test_max_gate_retries_none_when_no_session_end(self) -> None:
        """max_gate_retries is None when no session_end trigger is configured."""
        from src.domain.validation.config import ValidationConfig

        validation_config = ValidationConfig()
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        assert derived.max_gate_retries is None

    def test_max_gate_retries_none_when_no_validation_config(self) -> None:
        """max_gate_retries is None when validation_config is None."""
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=None,
            validation_config_missing=True,
        )

        assert derived.max_gate_retries is None

    def test_max_gate_retries_none_when_session_end_max_retries_unset(self) -> None:
        """max_gate_retries is None when session_end.max_retries is not set."""
        from src.domain.validation.config import (
            FailureMode,
            SessionEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        # max_retries defaults to None in SessionEndTriggerConfig
        validation_config = ValidationConfig(
            validation_triggers=ValidationTriggersConfig(
                session_end=SessionEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(),
                    # max_retries not set
                ),
            ),
        )
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        assert derived.max_gate_retries is None

    def test_max_epic_verification_retries_from_epic_completion_config(self) -> None:
        """max_epic_verification_retries is extracted from epic_completion config."""
        from src.domain.validation.config import (
            EpicCompletionTriggerConfig,
            EpicDepth,
            FailureMode,
            FireOn,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        validation_config = ValidationConfig(
            validation_triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(),
                    epic_depth=EpicDepth.ALL,
                    fire_on=FireOn.SUCCESS,
                    max_epic_verification_retries=5,
                ),
            ),
        )
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        assert derived.max_epic_verification_retries == 5

    def test_max_epic_verification_retries_none_when_no_epic_completion(self) -> None:
        """max_epic_verification_retries is None when no epic_completion trigger."""
        from src.domain.validation.config import ValidationConfig

        validation_config = ValidationConfig()
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        assert derived.max_epic_verification_retries is None

    def test_max_epic_verification_retries_none_when_no_validation_config(self) -> None:
        """max_epic_verification_retries is None when validation_config is None."""
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=None,
            validation_config_missing=True,
        )

        assert derived.max_epic_verification_retries is None

    def test_max_epic_verification_retries_none_when_field_unset(self) -> None:
        """max_epic_verification_retries is None when field not set in config."""
        from src.domain.validation.config import (
            EpicCompletionTriggerConfig,
            EpicDepth,
            FailureMode,
            FireOn,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        # max_epic_verification_retries defaults to None
        validation_config = ValidationConfig(
            validation_triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(),
                    epic_depth=EpicDepth.ALL,
                    fire_on=FireOn.SUCCESS,
                    # max_epic_verification_retries not set
                ),
            ),
        )
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        assert derived.max_epic_verification_retries is None

    def test_epic_verify_lock_timeout_seconds_from_epic_completion_config(self) -> None:
        """epic_verify_lock_timeout_seconds is extracted from epic_completion config."""
        from src.domain.validation.config import (
            EpicCompletionTriggerConfig,
            EpicDepth,
            FailureMode,
            FireOn,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        validation_config = ValidationConfig(
            validation_triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(),
                    epic_depth=EpicDepth.ALL,
                    fire_on=FireOn.SUCCESS,
                    epic_verify_lock_timeout_seconds=120,
                ),
            ),
        )
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        assert derived.epic_verify_lock_timeout_seconds == 120

    def test_epic_verify_lock_timeout_seconds_none_when_no_epic_completion(
        self,
    ) -> None:
        """epic_verify_lock_timeout_seconds is None when no epic_completion trigger."""
        from src.domain.validation.config import ValidationConfig

        validation_config = ValidationConfig()
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        assert derived.epic_verify_lock_timeout_seconds is None

    def test_epic_verify_lock_timeout_seconds_none_when_field_unset(self) -> None:
        """epic_verify_lock_timeout_seconds is None when field not set in config."""
        from src.domain.validation.config import (
            EpicCompletionTriggerConfig,
            EpicDepth,
            FailureMode,
            FireOn,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        validation_config = ValidationConfig(
            validation_triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(),
                    epic_depth=EpicDepth.ALL,
                    fire_on=FireOn.SUCCESS,
                    # epic_verify_lock_timeout_seconds not set
                ),
            ),
        )
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        assert derived.epic_verify_lock_timeout_seconds is None

    def test_timeout_from_validation_config(self) -> None:
        """timeout_minutes is read from validation_config when CLI is not set."""
        from src.domain.validation.config import ValidationConfig

        validation_config = ValidationConfig(timeout_minutes=45)
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        # Should use mala.yaml timeout: 45 minutes = 2700 seconds
        assert derived.timeout_seconds == 45 * 60

    def test_cli_timeout_overrides_validation_config(self) -> None:
        """CLI timeout_minutes overrides mala.yaml timeout_minutes."""
        from src.domain.validation.config import ValidationConfig

        validation_config = ValidationConfig(timeout_minutes=45)
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"), timeout_minutes=90)

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        # CLI should override: 90 minutes = 5400 seconds
        assert derived.timeout_seconds == 90 * 60

    def test_timeout_defaults_when_no_validation_config(self) -> None:
        """timeout_seconds uses default when validation_config is None."""
        from src.orchestration.types import DEFAULT_AGENT_TIMEOUT_MINUTES

        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=None,
            validation_config_missing=True,
        )

        assert derived.timeout_seconds == DEFAULT_AGENT_TIMEOUT_MINUTES * 60

    def test_timeout_defaults_when_validation_config_has_none(self) -> None:
        """timeout_seconds uses default when validation_config.timeout_minutes is None."""
        from src.domain.validation.config import ValidationConfig
        from src.orchestration.types import DEFAULT_AGENT_TIMEOUT_MINUTES

        validation_config = ValidationConfig()  # timeout_minutes=None
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        assert derived.timeout_seconds == DEFAULT_AGENT_TIMEOUT_MINUTES * 60

    def test_cli_timeout_zero_bypasses_yaml_timeout(self) -> None:
        """CLI timeout=0 explicitly uses default, bypassing mala.yaml timeout."""
        from src.domain.validation.config import ValidationConfig
        from src.orchestration.types import DEFAULT_AGENT_TIMEOUT_MINUTES

        # mala.yaml has timeout_minutes=45
        validation_config = ValidationConfig(timeout_minutes=45)
        # CLI passes 0 explicitly
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"), timeout_minutes=0)

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        # CLI 0 should bypass yaml and use default (legacy behavior)
        assert derived.timeout_seconds == DEFAULT_AGENT_TIMEOUT_MINUTES * 60

    def test_max_idle_retries_from_validation_config(self) -> None:
        """max_idle_retries is read from validation_config."""
        from src.domain.validation.config import ValidationConfig

        validation_config = ValidationConfig(max_idle_retries=5)
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        assert derived.max_idle_retries == 5

    def test_max_idle_retries_default_when_no_config(self) -> None:
        """max_idle_retries uses default when validation_config doesn't set it."""
        from src.orchestration.types import DEFAULT_MAX_IDLE_RETRIES

        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=None,
            validation_config_missing=True,
        )

        assert derived.max_idle_retries == DEFAULT_MAX_IDLE_RETRIES

    def test_max_idle_retries_default_when_null_in_config(self) -> None:
        """max_idle_retries uses default when validation_config has it as None."""
        from src.domain.validation.config import ValidationConfig
        from src.orchestration.types import DEFAULT_MAX_IDLE_RETRIES

        validation_config = ValidationConfig(max_idle_retries=None)
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        assert derived.max_idle_retries == DEFAULT_MAX_IDLE_RETRIES

    def test_idle_timeout_seconds_from_validation_config(self) -> None:
        """idle_timeout_seconds is read from validation_config."""
        from src.domain.validation.config import ValidationConfig

        validation_config = ValidationConfig(idle_timeout_seconds=600.0)
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        assert derived.idle_timeout_seconds == 600.0

    def test_idle_timeout_seconds_none_when_not_set(self) -> None:
        """idle_timeout_seconds is None when not set in validation_config."""
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=None,
            validation_config_missing=True,
        )

        assert derived.idle_timeout_seconds is None

    def test_idle_timeout_seconds_zero_disables(self) -> None:
        """idle_timeout_seconds=0 is passed through (disables idle timeout)."""
        from src.domain.validation.config import ValidationConfig

        validation_config = ValidationConfig(idle_timeout_seconds=0.0)
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        assert derived.idle_timeout_seconds == 0.0

    def test_derive_config_passes_per_issue_review(self) -> None:
        """per_issue_review is passed through from ValidationConfig."""
        from src.domain.validation.config import CodeReviewConfig, ValidationConfig

        per_issue_review = CodeReviewConfig(enabled=True, max_retries=5)
        validation_config = ValidationConfig(per_issue_review=per_issue_review)
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        assert derived.per_issue_review is per_issue_review

    def test_derive_config_per_issue_review_none_when_no_validation_config(
        self,
    ) -> None:
        """per_issue_review is None when validation_config is None."""
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=None,
            validation_config_missing=True,
        )

        assert derived.per_issue_review is None

    def test_derive_config_max_review_retries_from_per_issue_review(self) -> None:
        """max_review_retries uses per_issue_review.max_retries when enabled."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        # Set up per_issue_review with max_retries=7
        per_issue_review = CodeReviewConfig(enabled=True, max_retries=7)
        # Also set up a trigger with different max_retries=3
        validation_config = ValidationConfig(
            per_issue_review=per_issue_review,
            validation_triggers=ValidationTriggersConfig(
                run_end=RunEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(),
                    fire_on=FireOn.SUCCESS,
                    code_review=CodeReviewConfig(enabled=True, max_retries=3),
                ),
            ),
        )
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        # per_issue_review takes precedence when enabled
        assert derived.max_review_retries == 7

    def test_derive_config_max_review_retries_falls_back_to_triggers(self) -> None:
        """max_review_retries falls back to trigger config when per_issue_review disabled."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        # Set up per_issue_review disabled
        per_issue_review = CodeReviewConfig(enabled=False, max_retries=7)
        # Set up a trigger with max_retries=3
        validation_config = ValidationConfig(
            per_issue_review=per_issue_review,
            validation_triggers=ValidationTriggersConfig(
                run_end=RunEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(),
                    fire_on=FireOn.SUCCESS,
                    code_review=CodeReviewConfig(enabled=True, max_retries=3),
                ),
            ),
        )
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        # Falls back to trigger config when per_issue_review disabled
        assert derived.max_review_retries == 3

    def test_derive_config_max_review_retries_per_issue_review_uses_default(
        self,
    ) -> None:
        """per_issue_review uses its default max_retries (3) when enabled."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        # per_issue_review enabled, max_retries defaults to 3
        per_issue_review = CodeReviewConfig(enabled=True)
        validation_config = ValidationConfig(
            per_issue_review=per_issue_review,
            validation_triggers=ValidationTriggersConfig(
                run_end=RunEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(),
                    fire_on=FireOn.SUCCESS,
                    code_review=CodeReviewConfig(enabled=True, max_retries=4),
                ),
            ),
        )
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        # per_issue_review.max_retries defaults to 3, which takes precedence
        assert derived.max_review_retries == 3


class TestExtractReviewerConfig:
    """Tests for _extract_reviewer_config priority order."""

    def test_per_issue_review_takes_priority_when_enabled(self) -> None:
        """per_issue_review settings win over triggers when enabled=True."""
        from src.domain.validation.config import (
            CerberusConfig,
            CodeReviewConfig,
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        # per_issue_review: cerberus with timeout=300, model=opus
        per_issue_review = CodeReviewConfig(
            enabled=True,
            reviewer_type="cerberus",
            agent_sdk_timeout=300,
            agent_sdk_model="opus",
            cerberus=CerberusConfig(timeout=400),
        )
        # trigger: agent_sdk with timeout=900, model=haiku
        trigger_review = CodeReviewConfig(
            enabled=True,
            reviewer_type="agent_sdk",
            agent_sdk_timeout=900,
            agent_sdk_model="haiku",
        )
        validation_config = ValidationConfig(
            per_issue_review=per_issue_review,
            validation_triggers=ValidationTriggersConfig(
                run_end=RunEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(),
                    fire_on=FireOn.SUCCESS,
                    code_review=trigger_review,
                ),
            ),
        )

        result = _extract_reviewer_config(validation_config)

        # per_issue_review wins
        assert result.reviewer_type == "cerberus"
        assert result.agent_sdk_review_timeout == 300
        assert result.agent_sdk_reviewer_model == "opus"
        assert result.cerberus_config is not None

    def test_disabled_per_issue_review_ignored(self) -> None:
        """per_issue_review with enabled=False is completely ignored."""
        from src.domain.validation.config import (
            CerberusConfig,
            CodeReviewConfig,
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        # per_issue_review disabled but has other settings
        per_issue_review = CodeReviewConfig(
            enabled=False,
            reviewer_type="cerberus",
            agent_sdk_timeout=300,
            agent_sdk_model="opus",
            cerberus=CerberusConfig(timeout=400),
        )
        # trigger enabled with agent_sdk
        trigger_review = CodeReviewConfig(
            enabled=True,
            reviewer_type="agent_sdk",
            agent_sdk_timeout=900,
            agent_sdk_model="haiku",
        )
        validation_config = ValidationConfig(
            per_issue_review=per_issue_review,
            validation_triggers=ValidationTriggersConfig(
                run_end=RunEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(),
                    fire_on=FireOn.SUCCESS,
                    code_review=trigger_review,
                ),
            ),
        )

        result = _extract_reviewer_config(validation_config)

        # Falls back to trigger config since per_issue_review disabled
        assert result.reviewer_type == "agent_sdk"
        assert result.agent_sdk_review_timeout == 900
        assert result.agent_sdk_reviewer_model == "haiku"
        assert result.cerberus_config is None

    def test_triggers_used_when_no_per_issue_review(self) -> None:
        """Trigger config used when per_issue_review not set."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        trigger_review = CodeReviewConfig(
            enabled=True,
            reviewer_type="agent_sdk",
            agent_sdk_timeout=500,
            agent_sdk_model="haiku",
        )
        validation_config = ValidationConfig(
            validation_triggers=ValidationTriggersConfig(
                run_end=RunEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(),
                    fire_on=FireOn.SUCCESS,
                    code_review=trigger_review,
                ),
            ),
        )

        result = _extract_reviewer_config(validation_config)

        assert result.reviewer_type == "agent_sdk"
        assert result.agent_sdk_review_timeout == 500
        assert result.agent_sdk_reviewer_model == "haiku"

    def test_defaults_when_both_disabled(self) -> None:
        """Returns defaults when both per_issue_review and triggers disabled."""
        from src.domain.validation.config import (
            CodeReviewConfig,
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        per_issue_review = CodeReviewConfig(
            enabled=False,
            reviewer_type="cerberus",
        )
        trigger_review = CodeReviewConfig(
            enabled=False,
            reviewer_type="cerberus",
        )
        validation_config = ValidationConfig(
            per_issue_review=per_issue_review,
            validation_triggers=ValidationTriggersConfig(
                run_end=RunEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(),
                    fire_on=FireOn.SUCCESS,
                    code_review=trigger_review,
                ),
            ),
        )

        result = _extract_reviewer_config(validation_config)

        # Falls back to defaults
        assert result.reviewer_type == "agent_sdk"
        assert result.agent_sdk_review_timeout == 600
        assert result.agent_sdk_reviewer_model == "sonnet"

    def test_defaults_when_no_validation_config(self) -> None:
        """Returns defaults when validation_config is None."""
        result = _extract_reviewer_config(None)

        assert result.reviewer_type == "agent_sdk"
        assert result.agent_sdk_review_timeout == 600
        assert result.agent_sdk_reviewer_model == "sonnet"


class TestBuildDependenciesRuntimeDeps:
    """Tests for RuntimeDeps construction in _build_dependencies."""

    def test_constructs_defaults_when_deps_none(self, tmp_path: Path) -> None:
        """Factory constructs CommandRunner, EnvConfig, LockManager when deps=None."""
        from src.infra.tools.command_runner import CommandRunner
        from src.infra.tools.env import EnvConfig
        from src.infra.tools.locking import LockManager
        from src.orchestration.factory import _build_dependencies, _ReviewerConfig
        from src.orchestration.types import _DerivedConfig

        config = OrchestratorConfig(repo_path=tmp_path)
        mala_config = MalaConfig.from_env(validate=False)
        derived = _DerivedConfig(
            timeout_seconds=600,
            disabled_validations=set(),
            max_idle_retries=3,
            idle_timeout_seconds=None,
        )
        reviewer_config = _ReviewerConfig()

        result = _build_dependencies(
            config, mala_config, derived, None, reviewer_config
        )

        # Unpack result (10 elements)
        (
            _issue_provider,
            _code_reviewer,
            _gate_checker,
            _log_provider,
            _telemetry_provider,
            _event_sink,
            _epic_verifier,
            command_runner,
            env_config,
            lock_manager,
        ) = result

        # Verify types are concrete implementations
        assert isinstance(command_runner, CommandRunner)
        assert isinstance(env_config, EnvConfig)
        assert isinstance(lock_manager, LockManager)

    def test_uses_provided_command_runner(self, tmp_path: Path) -> None:
        """Factory uses provided command_runner instead of creating default."""
        from tests.fakes.command_runner import FakeCommandRunner
        from src.orchestration.factory import _build_dependencies, _ReviewerConfig
        from src.orchestration.types import OrchestratorDependencies, _DerivedConfig

        fake_runner = FakeCommandRunner(allow_unregistered=True)
        deps = OrchestratorDependencies(command_runner=fake_runner)
        config = OrchestratorConfig(repo_path=tmp_path)
        mala_config = MalaConfig.from_env(validate=False)
        derived = _DerivedConfig(
            timeout_seconds=600,
            disabled_validations=set(),
            max_idle_retries=3,
            idle_timeout_seconds=None,
        )
        reviewer_config = _ReviewerConfig()

        result = _build_dependencies(
            config, mala_config, derived, deps, reviewer_config
        )
        command_runner = result[7]

        assert command_runner is fake_runner

    def test_uses_provided_env_config(self, tmp_path: Path) -> None:
        """Factory uses provided env_config instead of creating default."""
        from tests.fakes.env_config import FakeEnvConfig
        from src.orchestration.factory import _build_dependencies, _ReviewerConfig
        from src.orchestration.types import OrchestratorDependencies, _DerivedConfig

        fake_env = FakeEnvConfig()
        deps = OrchestratorDependencies(env_config=fake_env)
        config = OrchestratorConfig(repo_path=tmp_path)
        mala_config = MalaConfig.from_env(validate=False)
        derived = _DerivedConfig(
            timeout_seconds=600,
            disabled_validations=set(),
            max_idle_retries=3,
            idle_timeout_seconds=None,
        )
        reviewer_config = _ReviewerConfig()

        result = _build_dependencies(
            config, mala_config, derived, deps, reviewer_config
        )
        env_config = result[8]

        assert env_config is fake_env

    def test_uses_provided_lock_manager(self, tmp_path: Path) -> None:
        """Factory uses provided lock_manager instead of creating default."""
        from tests.fakes.lock_manager import FakeLockManager
        from src.orchestration.factory import _build_dependencies, _ReviewerConfig
        from src.orchestration.types import OrchestratorDependencies, _DerivedConfig

        fake_manager = FakeLockManager()
        deps = OrchestratorDependencies(lock_manager=fake_manager)
        config = OrchestratorConfig(repo_path=tmp_path)
        mala_config = MalaConfig.from_env(validate=False)
        derived = _DerivedConfig(
            timeout_seconds=600,
            disabled_validations=set(),
            max_idle_retries=3,
            idle_timeout_seconds=None,
        )
        reviewer_config = _ReviewerConfig()

        result = _build_dependencies(
            config, mala_config, derived, deps, reviewer_config
        )
        lock_manager = result[9]

        assert lock_manager is fake_manager

    def test_fills_gaps_with_defaults(self, tmp_path: Path) -> None:
        """Factory fills None fields with defaults while respecting provided ones."""
        from tests.fakes.command_runner import FakeCommandRunner
        from tests.fakes.env_config import FakeEnvConfig
        from src.infra.tools.locking import LockManager
        from src.orchestration.factory import _build_dependencies, _ReviewerConfig
        from src.orchestration.types import OrchestratorDependencies, _DerivedConfig

        # Provide only command_runner and env_config, leave lock_manager as None
        fake_runner = FakeCommandRunner(allow_unregistered=True)
        fake_env = FakeEnvConfig()
        deps = OrchestratorDependencies(
            command_runner=fake_runner,
            env_config=fake_env,
            lock_manager=None,  # Should be filled with default
        )
        config = OrchestratorConfig(repo_path=tmp_path)
        mala_config = MalaConfig.from_env(validate=False)
        derived = _DerivedConfig(
            timeout_seconds=600,
            disabled_validations=set(),
            max_idle_retries=3,
            idle_timeout_seconds=None,
        )
        reviewer_config = _ReviewerConfig()

        result = _build_dependencies(
            config, mala_config, derived, deps, reviewer_config
        )
        command_runner = result[7]
        env_config = result[8]
        lock_manager = result[9]

        # Provided values are used
        assert command_runner is fake_runner
        assert env_config is fake_env
        # Missing value is filled with default
        assert isinstance(lock_manager, LockManager)

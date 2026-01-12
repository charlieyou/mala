"""Unit tests for orchestration factory functions.

Tests for:
- _check_review_availability: Review availability checking by reviewer_type
- _derive_config: Derived configuration extraction
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.infra.io.config import MalaConfig
from src.orchestration.factory import _check_review_availability, _derive_config
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

    def test_context_restart_threshold_from_validation_config(self) -> None:
        """context_restart_threshold is read from validation_config when CLI is default."""
        from src.domain.validation.config import ValidationConfig

        validation_config = ValidationConfig(context_restart_threshold=0.85)
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        assert derived.context_restart_threshold == 0.85

    def test_cli_context_restart_threshold_overrides_validation_config(self) -> None:
        """CLI context_restart_threshold overrides mala.yaml value."""
        from src.domain.validation.config import ValidationConfig

        validation_config = ValidationConfig(context_restart_threshold=0.85)
        orch_config = OrchestratorConfig(
            repo_path=Path("/tmp"), context_restart_threshold=0.95
        )

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        # CLI value should take precedence
        assert derived.context_restart_threshold == 0.95

    def test_context_restart_threshold_default_when_no_config(self) -> None:
        """context_restart_threshold uses default when neither CLI nor yaml set."""
        from src.orchestration.types import DEFAULT_CONTEXT_RESTART_THRESHOLD

        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=None,
            validation_config_missing=True,
        )

        assert derived.context_restart_threshold == DEFAULT_CONTEXT_RESTART_THRESHOLD

    def test_context_limit_from_validation_config(self) -> None:
        """context_limit is read from validation_config when CLI is default."""
        from src.domain.validation.config import ValidationConfig

        validation_config = ValidationConfig(context_limit=150_000)
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        assert derived.context_limit == 150_000

    def test_cli_context_limit_overrides_validation_config(self) -> None:
        """CLI context_limit overrides mala.yaml value."""
        from src.domain.validation.config import ValidationConfig

        validation_config = ValidationConfig(context_limit=150_000)
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"), context_limit=180_000)

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        # CLI value should take precedence
        assert derived.context_limit == 180_000

    def test_context_limit_default_when_no_config(self) -> None:
        """context_limit uses default when neither CLI nor yaml set."""
        from src.orchestration.types import DEFAULT_CONTEXT_LIMIT

        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=None,
            validation_config_missing=True,
        )

        assert derived.context_limit == DEFAULT_CONTEXT_LIMIT

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

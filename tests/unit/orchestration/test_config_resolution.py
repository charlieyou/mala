"""Unit tests for pure config-resolution helpers.

Covers the CLI > yaml > default precedence chain owned by
``_derive_config`` (the env tier is handled by ``MalaConfig.from_env``
and exercised by its own tests), plus reviewer-type extraction and
reviewer-timeout resolution.
"""

from pathlib import Path

import pytest

from src.infra.io.config import MalaConfig
from src.orchestration.config_resolution import (
    _ReviewerConfig,
    _derive_config,
    _extract_reviewer_config,
    _resolve_review_timeout_seconds,
)
from src.orchestration.types import OrchestratorConfig


class TestDeriveConfig:
    """Tests for _derive_config function."""

    def test_max_gate_retries_from_session_end_config(self) -> None:
        """max_gate_retries is extracted from session_end.max_retries."""
        from src.domain.validation.config_types import (
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
        from src.domain.validation.config_types import ValidationConfig

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
        from src.domain.validation.config_types import (
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
        from src.domain.validation.config_types import (
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
        from src.domain.validation.config_types import ValidationConfig

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
        from src.domain.validation.config_types import (
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
        from src.domain.validation.config_types import (
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
        from src.domain.validation.config_types import ValidationConfig

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
        from src.domain.validation.config_types import (
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
        from src.domain.validation.config_types import ValidationConfig

        validation_config = ValidationConfig(timeout_minutes=45)
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"))

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        assert derived.timeout_seconds == 45 * 60

    def test_cli_timeout_overrides_validation_config(self) -> None:
        """CLI timeout_minutes overrides mala.yaml timeout_minutes."""
        from src.domain.validation.config_types import ValidationConfig

        validation_config = ValidationConfig(timeout_minutes=45)
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"), timeout_minutes=90)

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

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
        from src.domain.validation.config_types import ValidationConfig
        from src.orchestration.types import DEFAULT_AGENT_TIMEOUT_MINUTES

        validation_config = ValidationConfig()
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
        from src.domain.validation.config_types import ValidationConfig
        from src.orchestration.types import DEFAULT_AGENT_TIMEOUT_MINUTES

        validation_config = ValidationConfig(timeout_minutes=45)
        orch_config = OrchestratorConfig(repo_path=Path("/tmp"), timeout_minutes=0)

        derived = _derive_config(
            orch_config,
            MalaConfig.from_env(validate=False),
            validation_config=validation_config,
            validation_config_missing=False,
        )

        assert derived.timeout_seconds == DEFAULT_AGENT_TIMEOUT_MINUTES * 60

    def test_max_idle_retries_from_validation_config(self) -> None:
        """max_idle_retries is read from validation_config."""
        from src.domain.validation.config_types import ValidationConfig

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
        from src.domain.validation.config_types import ValidationConfig
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
        from src.domain.validation.config_types import ValidationConfig

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
        from src.domain.validation.config_types import ValidationConfig

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
        from src.domain.validation.config_types import (
            CodeReviewConfig,
            ValidationConfig,
        )

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
        from src.domain.validation.config_types import (
            CodeReviewConfig,
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        per_issue_review = CodeReviewConfig(enabled=True, max_retries=7)
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

        assert derived.max_review_retries == 7

    def test_derive_config_max_review_retries_falls_back_to_triggers(self) -> None:
        """max_review_retries falls back to trigger config when per_issue_review disabled."""
        from src.domain.validation.config_types import (
            CodeReviewConfig,
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        per_issue_review = CodeReviewConfig(enabled=False, max_retries=7)
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

        assert derived.max_review_retries == 3

    def test_derive_config_max_review_retries_per_issue_review_uses_default(
        self,
    ) -> None:
        """per_issue_review uses its default max_retries (3) when enabled."""
        from src.domain.validation.config_types import (
            CodeReviewConfig,
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )

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

        assert derived.max_review_retries == 3


class TestResolveReviewTimeoutSeconds:
    """Tests for reviewer-specific timeout resolution."""

    @pytest.mark.parametrize(
        ("reviewer_config", "expected"),
        [
            (_ReviewerConfig(reviewer_type="agent_sdk"), 300),
            (_ReviewerConfig(reviewer_type="cerberus"), 600),
        ],
    )
    def test_uses_reviewer_defaults(
        self,
        reviewer_config: _ReviewerConfig,
        expected: int,
    ) -> None:
        config = MalaConfig(
            runs_dir=Path("/tmp/runs"),
            lock_dir=Path("/tmp/locks"),
        )

        assert _resolve_review_timeout_seconds(config, reviewer_config) == expected

    def test_legacy_mala_config_override_wins(self) -> None:
        config = MalaConfig(
            runs_dir=Path("/tmp/runs"),
            lock_dir=Path("/tmp/locks"),
            review_timeout=900,
        )

        assert _resolve_review_timeout_seconds(config, _ReviewerConfig()) == 900


class TestExtractReviewerConfig:
    """Tests for _extract_reviewer_config priority order."""

    def test_per_issue_review_takes_priority_when_enabled(self) -> None:
        """per_issue_review settings win over triggers when enabled=True."""
        from src.domain.validation.config_types import (
            CerberusConfig,
            CodeReviewConfig,
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        per_issue_review = CodeReviewConfig(
            enabled=True,
            reviewer_type="cerberus",
            agent_sdk_timeout=300,
            agent_sdk_model="opus",
            cerberus=CerberusConfig(timeout=400),
        )
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

        assert result.reviewer_type == "cerberus"
        assert result.agent_sdk_review_timeout == 300
        assert result.agent_sdk_reviewer_model == "opus"
        assert result.cerberus_config is not None

    def test_disabled_per_issue_review_ignored(self) -> None:
        """per_issue_review with enabled=False is completely ignored."""
        from src.domain.validation.config_types import (
            CerberusConfig,
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
            agent_sdk_timeout=300,
            agent_sdk_model="opus",
            cerberus=CerberusConfig(timeout=400),
        )
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

        assert result.reviewer_type == "agent_sdk"
        assert result.agent_sdk_review_timeout == 900
        assert result.agent_sdk_reviewer_model == "haiku"
        assert result.cerberus_config is None

    def test_triggers_used_when_no_per_issue_review(self) -> None:
        """Trigger config used when per_issue_review not set."""
        from src.domain.validation.config_types import (
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
        from src.domain.validation.config_types import (
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

        assert result.reviewer_type == "agent_sdk"
        assert result.agent_sdk_review_timeout == 300
        assert result.agent_sdk_reviewer_model == "opus"

    def test_defaults_when_no_validation_config(self) -> None:
        """Returns defaults when validation_config is None."""
        result = _extract_reviewer_config(None)

        assert result.reviewer_type == "agent_sdk"
        assert result.agent_sdk_review_timeout == 300
        assert result.agent_sdk_reviewer_model == "opus"

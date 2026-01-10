"""Unit tests for lifecycle state machine session_end transitions.

Tests verify:
- RUNNING_SESSION_END state and Effect.RUN_SESSION_END transitions
- Gate passed → RUNNING_SESSION_END transition (when session_end enabled)
- Session_end → RUNNING_REVIEW transition
- Session_end → SUCCESS transition (when review disabled)
- SESSION_END_REMEDIATING state for retry loops
- Invalid transition rejection
"""

import pytest

from src.domain.lifecycle import (
    Effect,
    ImplementerLifecycle,
    LifecycleConfig,
    LifecycleContext,
    LifecycleState,
)
from src.domain.evidence_check import GateResult
from src.core.session_end_result import SessionEndResult


class TestSessionEndTransitions:
    """Tests for session_end lifecycle transitions."""

    def _setup_for_gate(
        self, session_end_enabled: bool = True, review_enabled: bool = True
    ) -> tuple[ImplementerLifecycle, LifecycleContext]:
        """Helper to set up lifecycle ready for gate result."""
        config = LifecycleConfig(
            session_end_enabled=session_end_enabled,
            review_enabled=review_enabled,
        )
        lifecycle = ImplementerLifecycle(config)
        lifecycle.start()
        ctx = LifecycleContext()
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)
        return lifecycle, ctx

    def test_gate_passed_transitions_to_running_session_end(self) -> None:
        """Gate passed with session_end enabled → RUNNING_SESSION_END."""
        lifecycle, ctx = self._setup_for_gate(session_end_enabled=True)
        gate_result = GateResult(passed=True, commit_hash="abc123")

        result = lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)

        assert lifecycle.state == LifecycleState.RUNNING_SESSION_END
        assert result.effect == Effect.RUN_SESSION_END
        assert ctx.retry_state.session_end_attempt == 1

    def test_gate_passed_skips_session_end_when_disabled(self) -> None:
        """Gate passed with session_end disabled → RUNNING_REVIEW."""
        lifecycle, ctx = self._setup_for_gate(
            session_end_enabled=False, review_enabled=True
        )
        gate_result = GateResult(passed=True, commit_hash="abc123")

        result = lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)

        assert lifecycle.state == LifecycleState.RUNNING_REVIEW
        assert result.effect == Effect.RUN_REVIEW

    def test_gate_passed_skips_session_end_without_commit(self) -> None:
        """Gate passed without commit hash → SUCCESS (no session_end needed)."""
        lifecycle, ctx = self._setup_for_gate(session_end_enabled=True)
        gate_result = GateResult(passed=True, commit_hash=None)

        result = lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)

        assert lifecycle.state == LifecycleState.SUCCESS
        assert result.effect == Effect.COMPLETE_SUCCESS

    def test_session_end_passed_transitions_to_running_review(self) -> None:
        """Session_end passed → RUNNING_REVIEW (when review enabled)."""
        lifecycle, ctx = self._setup_for_gate(session_end_enabled=True)
        gate_result = GateResult(passed=True, commit_hash="abc123")
        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)

        session_end_result = SessionEndResult(status="pass")

        result = lifecycle.on_session_end_result(
            ctx, session_end_result, new_log_offset=200
        )

        assert lifecycle.state == LifecycleState.RUNNING_REVIEW
        assert result.effect == Effect.RUN_REVIEW
        assert ctx.retry_state.review_attempt == 1

    def test_session_end_passed_transitions_to_success_when_review_disabled(
        self,
    ) -> None:
        """Session_end passed → SUCCESS (when review disabled)."""
        lifecycle, ctx = self._setup_for_gate(
            session_end_enabled=True, review_enabled=False
        )
        gate_result = GateResult(passed=True, commit_hash="abc123")
        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)

        session_end_result = SessionEndResult(status="pass")

        result = lifecycle.on_session_end_result(
            ctx, session_end_result, new_log_offset=200
        )

        assert lifecycle.state == LifecycleState.SUCCESS
        assert result.effect == Effect.COMPLETE_SUCCESS
        assert ctx.success

    def test_session_end_skipped_transitions_to_review(self) -> None:
        """Session_end skipped → RUNNING_REVIEW."""
        lifecycle, ctx = self._setup_for_gate(session_end_enabled=True)
        gate_result = GateResult(passed=True, commit_hash="abc123")
        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)

        session_end_result = SessionEndResult(status="skipped")

        result = lifecycle.on_session_end_result(
            ctx, session_end_result, new_log_offset=200
        )

        assert lifecycle.state == LifecycleState.RUNNING_REVIEW
        assert result.effect == Effect.RUN_REVIEW

    def test_session_end_interrupted_transitions_to_review(self) -> None:
        """Session_end interrupted → RUNNING_REVIEW."""
        lifecycle, ctx = self._setup_for_gate(session_end_enabled=True)
        gate_result = GateResult(passed=True, commit_hash="abc123")
        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)

        session_end_result = SessionEndResult(status="interrupted")

        result = lifecycle.on_session_end_result(
            ctx, session_end_result, new_log_offset=200
        )

        assert lifecycle.state == LifecycleState.RUNNING_REVIEW
        assert result.effect == Effect.RUN_REVIEW

    def test_session_end_failed_with_can_remediate_triggers_retry(self) -> None:
        """Session_end failed with can_remediate=True → SESSION_END_REMEDIATING."""
        config = LifecycleConfig(
            session_end_enabled=True,
            max_session_end_retries=3,
        )
        lifecycle = ImplementerLifecycle(config)
        lifecycle.start()
        ctx = LifecycleContext()
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)
        gate_result = GateResult(passed=True, commit_hash="abc123")
        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)

        session_end_result = SessionEndResult(status="fail")

        result = lifecycle.on_session_end_result(
            ctx, session_end_result, new_log_offset=200, can_remediate=True
        )

        assert lifecycle.state == LifecycleState.SESSION_END_REMEDIATING
        assert result.effect == Effect.SEND_SESSION_END_RETRY
        assert ctx.retry_state.session_end_attempt == 2

    def test_session_end_failed_without_can_remediate_proceeds_to_review(self) -> None:
        """Session_end failed without can_remediate → RUNNING_REVIEW."""
        lifecycle, ctx = self._setup_for_gate(session_end_enabled=True)
        gate_result = GateResult(passed=True, commit_hash="abc123")
        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)

        session_end_result = SessionEndResult(status="fail")

        result = lifecycle.on_session_end_result(
            ctx, session_end_result, new_log_offset=200, can_remediate=False
        )

        assert lifecycle.state == LifecycleState.RUNNING_REVIEW
        assert result.effect == Effect.RUN_REVIEW

    def test_session_end_timeout_without_can_remediate_proceeds_to_review(self) -> None:
        """Session_end timeout without can_remediate → RUNNING_REVIEW."""
        lifecycle, ctx = self._setup_for_gate(session_end_enabled=True)
        gate_result = GateResult(passed=True, commit_hash="abc123")
        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)

        session_end_result = SessionEndResult(status="timeout")

        result = lifecycle.on_session_end_result(
            ctx, session_end_result, new_log_offset=200, can_remediate=False
        )

        assert lifecycle.state == LifecycleState.RUNNING_REVIEW
        assert result.effect == Effect.RUN_REVIEW

    def test_session_end_failed_exhausted_retries_proceeds_to_review(self) -> None:
        """Session_end failed with no retries left → RUNNING_REVIEW."""
        config = LifecycleConfig(
            session_end_enabled=True,
            max_session_end_retries=2,
        )
        lifecycle = ImplementerLifecycle(config)
        lifecycle.start()
        ctx = LifecycleContext()
        ctx.retry_state.session_end_attempt = 2  # Already at max
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)
        gate_result = GateResult(passed=True, commit_hash="abc123")
        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)

        session_end_result = SessionEndResult(status="fail")

        result = lifecycle.on_session_end_result(
            ctx, session_end_result, new_log_offset=200, can_remediate=True
        )

        assert lifecycle.state == LifecycleState.RUNNING_REVIEW
        assert result.effect == Effect.RUN_REVIEW

    def test_session_end_failed_no_progress_proceeds_to_review(self) -> None:
        """Session_end failed with no_progress=True → RUNNING_REVIEW."""
        config = LifecycleConfig(
            session_end_enabled=True,
            max_session_end_retries=3,
        )
        lifecycle = ImplementerLifecycle(config)
        lifecycle.start()
        ctx = LifecycleContext()
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)
        gate_result = GateResult(passed=True, commit_hash="abc123")
        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)

        session_end_result = SessionEndResult(status="fail")

        result = lifecycle.on_session_end_result(
            ctx,
            session_end_result,
            new_log_offset=200,
            no_progress=True,
            can_remediate=True,
        )

        assert lifecycle.state == LifecycleState.RUNNING_REVIEW
        assert result.effect == Effect.RUN_REVIEW


class TestSessionEndRemediationTransitions:
    """Tests for session_end remediation state transitions."""

    def _setup_for_remediation(
        self,
    ) -> tuple[ImplementerLifecycle, LifecycleContext]:
        """Helper to set up lifecycle in SESSION_END_REMEDIATING state."""
        config = LifecycleConfig(
            session_end_enabled=True,
            max_session_end_retries=3,
        )
        lifecycle = ImplementerLifecycle(config)
        lifecycle.start()
        ctx = LifecycleContext()
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)
        gate_result = GateResult(passed=True, commit_hash="abc123")
        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)
        # Fail session_end with can_remediate=True to enter remediation state
        session_end_result = SessionEndResult(status="fail")
        lifecycle.on_session_end_result(
            ctx, session_end_result, new_log_offset=200, can_remediate=True
        )
        return lifecycle, ctx

    def test_remediation_complete_transitions_to_running_session_end(self) -> None:
        """Remediation complete → RUNNING_SESSION_END."""
        lifecycle, ctx = self._setup_for_remediation()

        result = lifecycle.on_session_end_remediation_complete(ctx)

        assert lifecycle.state == LifecycleState.RUNNING_SESSION_END
        assert result.effect == Effect.RUN_SESSION_END

    def test_session_end_result_after_remediation_complete(self) -> None:
        """Session_end result after remediation complete → RUNNING_REVIEW."""
        lifecycle, ctx = self._setup_for_remediation()
        # After remediation, transition back to RUNNING_SESSION_END
        lifecycle.on_session_end_remediation_complete(ctx)
        assert lifecycle.state == LifecycleState.RUNNING_SESSION_END

        # Now simulate a passing result from the re-run
        session_end_result = SessionEndResult(status="pass")
        result = lifecycle.on_session_end_result(
            ctx, session_end_result, new_log_offset=300
        )

        assert lifecycle.state == LifecycleState.RUNNING_REVIEW
        assert result.effect == Effect.RUN_REVIEW


class TestSessionEndInvalidTransitions:
    """Tests for invalid session_end state transitions."""

    def test_session_end_result_from_initial_raises(self) -> None:
        """session_end_result from INITIAL state raises error."""
        lifecycle = ImplementerLifecycle(LifecycleConfig())
        ctx = LifecycleContext()
        session_end_result = SessionEndResult(status="pass")

        with pytest.raises(ValueError, match="Unexpected state"):
            lifecycle.on_session_end_result(ctx, session_end_result, new_log_offset=0)

    def test_session_end_result_from_processing_raises(self) -> None:
        """session_end_result from PROCESSING state raises error."""
        lifecycle = ImplementerLifecycle(LifecycleConfig())
        lifecycle.start()
        ctx = LifecycleContext()
        session_end_result = SessionEndResult(status="pass")

        with pytest.raises(ValueError, match="Unexpected state"):
            lifecycle.on_session_end_result(ctx, session_end_result, new_log_offset=0)

    def test_session_end_result_from_running_gate_raises(self) -> None:
        """session_end_result from RUNNING_GATE state raises error."""
        lifecycle = ImplementerLifecycle(LifecycleConfig())
        lifecycle.start()
        ctx = LifecycleContext()
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)
        session_end_result = SessionEndResult(status="pass")

        with pytest.raises(ValueError, match="Unexpected state"):
            lifecycle.on_session_end_result(ctx, session_end_result, new_log_offset=0)

    def test_session_end_result_from_running_review_raises(self) -> None:
        """session_end_result from RUNNING_REVIEW state raises error."""
        config = LifecycleConfig(session_end_enabled=False)
        lifecycle = ImplementerLifecycle(config)
        lifecycle.start()
        ctx = LifecycleContext()
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)
        gate_result = GateResult(passed=True, commit_hash="abc123")
        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)
        # Now in RUNNING_REVIEW
        session_end_result = SessionEndResult(status="pass")

        with pytest.raises(ValueError, match="Unexpected state"):
            lifecycle.on_session_end_result(ctx, session_end_result, new_log_offset=0)

    def test_session_end_result_from_remediating_raises(self) -> None:
        """session_end_result from SESSION_END_REMEDIATING state raises error.

        Must call on_session_end_remediation_complete first to transition back
        to RUNNING_SESSION_END before reporting session_end result.
        """
        config = LifecycleConfig(session_end_enabled=True, max_session_end_retries=3)
        lifecycle = ImplementerLifecycle(config)
        lifecycle.start()
        ctx = LifecycleContext()
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)
        gate_result = GateResult(passed=True, commit_hash="abc123")
        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)
        # Fail session_end with can_remediate=True to enter remediation state
        session_end_result = SessionEndResult(status="fail")
        lifecycle.on_session_end_result(
            ctx, session_end_result, new_log_offset=200, can_remediate=True
        )
        # Now in SESSION_END_REMEDIATING
        assert lifecycle.state == LifecycleState.SESSION_END_REMEDIATING

        # Attempting to report session_end result without completing remediation
        new_result = SessionEndResult(status="pass")
        with pytest.raises(ValueError, match="Unexpected state"):
            lifecycle.on_session_end_result(ctx, new_result, new_log_offset=300)

    def test_session_end_remediation_complete_from_running_session_end_raises(
        self,
    ) -> None:
        """remediation_complete from RUNNING_SESSION_END raises error."""
        config = LifecycleConfig(session_end_enabled=True)
        lifecycle = ImplementerLifecycle(config)
        lifecycle.start()
        ctx = LifecycleContext()
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)
        gate_result = GateResult(passed=True, commit_hash="abc123")
        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)
        # Now in RUNNING_SESSION_END

        with pytest.raises(ValueError, match="Unexpected state"):
            lifecycle.on_session_end_remediation_complete(ctx)


class TestSessionEndStateAndEffectEnums:
    """Tests for session_end state and effect enum values."""

    def test_running_session_end_state_exists(self) -> None:
        """RUNNING_SESSION_END state exists in LifecycleState enum."""
        assert hasattr(LifecycleState, "RUNNING_SESSION_END")
        assert LifecycleState.RUNNING_SESSION_END is not None

    def test_session_end_remediating_state_exists(self) -> None:
        """SESSION_END_REMEDIATING state exists in LifecycleState enum."""
        assert hasattr(LifecycleState, "SESSION_END_REMEDIATING")
        assert LifecycleState.SESSION_END_REMEDIATING is not None

    def test_run_session_end_effect_exists(self) -> None:
        """RUN_SESSION_END effect exists in Effect enum."""
        assert hasattr(Effect, "RUN_SESSION_END")
        assert Effect.RUN_SESSION_END is not None

    def test_send_session_end_retry_effect_exists(self) -> None:
        """SEND_SESSION_END_RETRY effect exists in Effect enum."""
        assert hasattr(Effect, "SEND_SESSION_END_RETRY")
        assert Effect.SEND_SESSION_END_RETRY is not None


class TestSessionEndConfigFields:
    """Tests for session_end configuration fields."""

    def test_lifecycle_config_has_session_end_enabled(self) -> None:
        """LifecycleConfig has session_end_enabled field."""
        config = LifecycleConfig()
        assert hasattr(config, "session_end_enabled")
        assert config.session_end_enabled is True  # Default enabled

    def test_lifecycle_config_has_max_session_end_retries(self) -> None:
        """LifecycleConfig has max_session_end_retries field."""
        config = LifecycleConfig()
        assert hasattr(config, "max_session_end_retries")
        assert config.max_session_end_retries == 3  # Default

    def test_retry_state_has_session_end_attempt(self) -> None:
        """RetryState has session_end_attempt field."""
        ctx = LifecycleContext()
        assert hasattr(ctx.retry_state, "session_end_attempt")
        assert ctx.retry_state.session_end_attempt == 0  # Default

    def test_lifecycle_context_has_last_session_end_result(self) -> None:
        """LifecycleContext has last_session_end_result field."""
        ctx = LifecycleContext()
        assert hasattr(ctx, "last_session_end_result")
        assert ctx.last_session_end_result is None  # Default

"""Unit tests for ImplementerLifecycle state machine.

These tests verify retry/gate/review policy transitions without requiring
Claude SDK, subprocess mocking, or any I/O. The lifecycle is a pure state
machine that can be tested with simple data inputs.
"""

import pytest

from src.infra.clients.review_output_parser import ReviewIssue, ReviewResult
from src.domain.lifecycle import (
    Effect,
    ImplementerLifecycle,
    LifecycleConfig,
    LifecycleContext,
    LifecycleState,
)
from src.domain.evidence_check import GateResult


class TestLifecycleStart:
    """Tests for lifecycle initialization and start transition."""

    def test_initial_state(self) -> None:
        """Lifecycle starts in INITIAL state."""
        lifecycle = ImplementerLifecycle(LifecycleConfig())
        assert lifecycle.state == LifecycleState.INITIAL
        assert not lifecycle.is_terminal

    def test_start_transitions_to_processing(self) -> None:
        """start() transitions from INITIAL to PROCESSING."""
        lifecycle = ImplementerLifecycle(LifecycleConfig())
        result = lifecycle.start()

        assert lifecycle.state == LifecycleState.PROCESSING
        assert result.state == LifecycleState.PROCESSING
        assert result.effect == Effect.CONTINUE

    def test_start_fails_from_non_initial_state(self) -> None:
        """start() raises error if not in INITIAL state."""
        lifecycle = ImplementerLifecycle(LifecycleConfig())
        lifecycle.start()  # Now in PROCESSING

        with pytest.raises(ValueError, match="Cannot start from state"):
            lifecycle.start()


class TestMessagesComplete:
    """Tests for on_messages_complete transition."""

    def test_with_session_id_transitions_to_awaiting_log(self) -> None:
        """With session ID, transitions to AWAITING_LOG."""
        lifecycle = ImplementerLifecycle(LifecycleConfig())
        lifecycle.start()
        ctx = LifecycleContext()

        result = lifecycle.on_messages_complete(ctx, has_session_id=True)

        assert lifecycle.state == LifecycleState.AWAITING_LOG
        assert result.effect == Effect.WAIT_FOR_LOG

    def test_without_session_id_fails(self) -> None:
        """Without session ID, transitions to FAILED."""
        lifecycle = ImplementerLifecycle(LifecycleConfig())
        lifecycle.start()
        ctx = LifecycleContext()

        result = lifecycle.on_messages_complete(ctx, has_session_id=False)

        assert lifecycle.state == LifecycleState.FAILED
        assert result.effect == Effect.COMPLETE_FAILURE
        assert "No session ID" in ctx.final_result
        assert not ctx.success

    def test_fails_from_wrong_state(self) -> None:
        """on_messages_complete raises error if not in PROCESSING."""
        lifecycle = ImplementerLifecycle(LifecycleConfig())
        # Still in INITIAL
        ctx = LifecycleContext()

        with pytest.raises(ValueError, match="Unexpected state"):
            lifecycle.on_messages_complete(ctx, has_session_id=True)


class TestLogReady:
    """Tests for on_log_ready and on_log_timeout transitions."""

    def test_log_ready_transitions_to_running_gate(self) -> None:
        """on_log_ready transitions to RUNNING_GATE."""
        lifecycle = ImplementerLifecycle(LifecycleConfig())
        lifecycle.start()
        ctx = LifecycleContext()
        lifecycle.on_messages_complete(ctx, has_session_id=True)

        result = lifecycle.on_log_ready(ctx)

        assert lifecycle.state == LifecycleState.RUNNING_GATE
        assert result.effect == Effect.RUN_GATE

    def test_log_timeout_fails(self) -> None:
        """on_log_timeout transitions to FAILED."""
        lifecycle = ImplementerLifecycle(LifecycleConfig())
        lifecycle.start()
        ctx = LifecycleContext()
        lifecycle.on_messages_complete(ctx, has_session_id=True)

        result = lifecycle.on_log_timeout(ctx, "/path/to/log.jsonl")

        assert lifecycle.state == LifecycleState.FAILED
        assert result.effect == Effect.COMPLETE_FAILURE
        assert "/path/to/log.jsonl" in ctx.final_result


class TestGateResult:
    """Tests for on_gate_result transition."""

    def _setup_for_gate(self) -> tuple[ImplementerLifecycle, LifecycleContext]:
        """Helper to set up lifecycle ready for gate result.

        Note: Uses session_end_enabled=False to test gate→review flow directly.
        """
        lifecycle = ImplementerLifecycle(LifecycleConfig(session_end_enabled=False))
        lifecycle.start()
        ctx = LifecycleContext()
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)
        return lifecycle, ctx

    def test_gate_passed_with_review_enabled_runs_review(self) -> None:
        """Gate passed with review enabled transitions to RUNNING_REVIEW.

        Note: Uses session_end_enabled=False to test gate→review flow directly.
        """
        lifecycle, ctx = self._setup_for_gate()
        gate_result = GateResult(passed=True, commit_hash="abc123")

        result = lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)

        assert lifecycle.state == LifecycleState.RUNNING_REVIEW
        assert result.effect == Effect.RUN_REVIEW
        assert ctx.retry_state.review_attempt == 1

    def test_gate_passed_without_commit_hash_succeeds(self) -> None:
        """Gate passed without commit hash completes successfully."""
        config = LifecycleConfig(review_enabled=True, session_end_enabled=False)
        lifecycle = ImplementerLifecycle(config)
        lifecycle.start()
        ctx = LifecycleContext()
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)
        gate_result = GateResult(passed=True, commit_hash=None)  # No commit

        result = lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)

        assert lifecycle.state == LifecycleState.SUCCESS
        assert result.effect == Effect.COMPLETE_SUCCESS
        assert ctx.success

    def test_gate_passed_with_review_disabled_succeeds(self) -> None:
        """Gate passed with review disabled completes successfully."""
        config = LifecycleConfig(review_enabled=False, session_end_enabled=False)
        lifecycle = ImplementerLifecycle(config)
        lifecycle.start()
        ctx = LifecycleContext()
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)
        gate_result = GateResult(passed=True, commit_hash="abc123")

        result = lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)

        assert lifecycle.state == LifecycleState.SUCCESS
        assert result.effect == Effect.COMPLETE_SUCCESS
        assert ctx.success

    def test_gate_failed_with_retries_remaining_retries(self) -> None:
        """Gate failed with retries remaining sends retry prompt."""
        config = LifecycleConfig(max_gate_retries=3, session_end_enabled=False)
        lifecycle = ImplementerLifecycle(config)
        lifecycle.start()
        ctx = LifecycleContext()
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)
        gate_result = GateResult(
            passed=False,
            failure_reasons=["Missing pytest evidence"],
            commit_hash="abc123",
        )

        result = lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)

        assert lifecycle.state == LifecycleState.PROCESSING
        assert result.effect == Effect.SEND_GATE_RETRY
        assert ctx.retry_state.gate_attempt == 2
        assert ctx.retry_state.log_offset == 100
        assert ctx.retry_state.previous_commit_hash == "abc123"

    def test_gate_failed_no_retries_left_fails(self) -> None:
        """Gate failed with no retries left transitions to FAILED."""
        config = LifecycleConfig(max_gate_retries=2, session_end_enabled=False)
        lifecycle = ImplementerLifecycle(config)
        lifecycle.start()
        ctx = LifecycleContext()
        ctx.retry_state.gate_attempt = 2  # Already at max
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)
        gate_result = GateResult(
            passed=False,
            failure_reasons=["Missing pytest evidence"],
        )

        result = lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)

        assert lifecycle.state == LifecycleState.FAILED
        assert result.effect == Effect.COMPLETE_FAILURE
        assert "Missing pytest evidence" in ctx.final_result
        assert not ctx.success

    def test_gate_failed_no_progress_fails_immediately(self) -> None:
        """Gate failed with no_progress=True fails without retry."""
        config = LifecycleConfig(max_gate_retries=3, session_end_enabled=False)
        lifecycle = ImplementerLifecycle(config)
        lifecycle.start()
        ctx = LifecycleContext()
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)
        gate_result = GateResult(
            passed=False,
            failure_reasons=["No progress"],
            no_progress=True,
        )

        result = lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)

        assert lifecycle.state == LifecycleState.FAILED
        assert result.effect == Effect.COMPLETE_FAILURE
        # Should fail even though retries remain because no_progress is set

    def test_gate_stores_resolution(self) -> None:
        """Gate result resolution is stored in context."""
        from src.domain.validation.spec import IssueResolution, ResolutionOutcome

        lifecycle, ctx = self._setup_for_gate()
        resolution = IssueResolution(
            outcome=ResolutionOutcome.NO_CHANGE, rationale="No changes needed"
        )
        gate_result = GateResult(passed=True, resolution=resolution)

        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)

        assert ctx.resolution == resolution

    @pytest.mark.parametrize(
        "outcome",
        [
            pytest.param("NO_CHANGE", id="no_change"),
            pytest.param("OBSOLETE", id="obsolete"),
            pytest.param("ALREADY_COMPLETE", id="already_complete"),
        ],
    )
    def test_gate_passed_skips_review_for_no_work_resolutions(
        self, outcome: str
    ) -> None:
        """Gate passed with no-work resolutions skips review even if enabled."""
        from src.domain.validation.spec import IssueResolution, ResolutionOutcome

        lifecycle, ctx = self._setup_for_gate()
        resolution = IssueResolution(
            outcome=ResolutionOutcome[outcome], rationale="Work was already done"
        )
        # Gate passes with commit_hash but resolution indicates no new work
        gate_result = GateResult(
            passed=True, commit_hash="abc123", resolution=resolution
        )

        result = lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)

        # Should skip review and go straight to success
        assert lifecycle.state == LifecycleState.SUCCESS
        assert result.effect == Effect.COMPLETE_SUCCESS
        assert ctx.success


class TestReviewResult:
    """Tests for on_review_result transition."""

    def _setup_for_review(self) -> tuple[ImplementerLifecycle, LifecycleContext]:
        """Helper to set up lifecycle ready for review result.

        Note: Uses session_end_enabled=False to test gate→review flow directly.
        """
        lifecycle = ImplementerLifecycle(LifecycleConfig(session_end_enabled=False))
        lifecycle.start()
        ctx = LifecycleContext()
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)
        gate_result = GateResult(passed=True, commit_hash="abc123")
        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)
        return lifecycle, ctx

    def test_review_passed_succeeds(self) -> None:
        """Review passed transitions to SUCCESS."""
        lifecycle, ctx = self._setup_for_review()
        review_result = ReviewResult(passed=True)

        result = lifecycle.on_review_result(ctx, review_result, new_log_offset=200)

        assert lifecycle.state == LifecycleState.SUCCESS
        assert result.effect == Effect.COMPLETE_SUCCESS
        assert ctx.success

    def test_review_failed_with_retries_remaining_retries(self) -> None:
        """Review failed with retries remaining sends retry prompt."""
        config = LifecycleConfig(max_review_retries=2, session_end_enabled=False)
        lifecycle = ImplementerLifecycle(config)
        lifecycle.start()
        ctx = LifecycleContext()
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)
        gate_result = GateResult(passed=True, commit_hash="abc123")
        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)
        # review_attempt is 1 after gate_result, max is 2

        review_result = ReviewResult(
            passed=False,
            issues=[
                ReviewIssue(
                    file="test.py",
                    line_start=10,
                    line_end=10,
                    priority=1,
                    title="Bug",
                    body="",
                    reviewer="cerberus",
                )
            ],
        )

        result = lifecycle.on_review_result(ctx, review_result, new_log_offset=200)

        assert lifecycle.state == LifecycleState.PROCESSING
        assert result.effect == Effect.SEND_REVIEW_RETRY
        assert ctx.retry_state.review_attempt == 2
        assert ctx.retry_state.log_offset == 200

    def test_review_failed_no_retries_left_fails(self) -> None:
        """Review failed with no retries left transitions to FAILED."""
        config = LifecycleConfig(max_review_retries=1, session_end_enabled=False)
        lifecycle = ImplementerLifecycle(config)
        lifecycle.start()
        ctx = LifecycleContext()
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)
        gate_result = GateResult(passed=True, commit_hash="abc123")
        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)
        # review_attempt is 1 after gate_result, max is also 1

        review_result = ReviewResult(
            passed=False,
            issues=[
                ReviewIssue(
                    file="test.py",
                    line_start=10,
                    line_end=10,
                    priority=1,
                    title="Bug found",
                    body="",
                    reviewer="cerberus",
                )
            ],
        )

        result = lifecycle.on_review_result(ctx, review_result, new_log_offset=200)

        assert lifecycle.state == LifecycleState.FAILED
        assert result.effect == Effect.COMPLETE_FAILURE
        assert "Bug found" in ctx.final_result

    def test_review_parse_error_triggers_rerun_when_retries_remain(self) -> None:
        """Parse error triggers RUN_REVIEW (not SEND_REVIEW_RETRY) when retries remain."""
        config = LifecycleConfig(max_review_retries=2, session_end_enabled=False)
        lifecycle = ImplementerLifecycle(config)
        lifecycle.start()
        ctx = LifecycleContext()
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)
        gate_result = GateResult(passed=True, commit_hash="abc123")
        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)
        # review_attempt is 1 after gate_result

        review_result = ReviewResult(
            passed=False,
            parse_error="Invalid JSON response",
        )

        result = lifecycle.on_review_result(ctx, review_result, new_log_offset=200)

        # Parse error should trigger RUN_REVIEW to re-run the external review tool
        assert lifecycle.state == LifecycleState.RUNNING_REVIEW
        assert result.effect == Effect.RUN_REVIEW
        assert ctx.retry_state.review_attempt == 2
        assert "parse error" in (result.message or "").lower()

    def test_review_parse_error_fails_when_no_retries_left(self) -> None:
        """Parse error fails when no retries remain."""
        config = LifecycleConfig(max_review_retries=1, session_end_enabled=False)
        lifecycle = ImplementerLifecycle(config)
        lifecycle.start()
        ctx = LifecycleContext()
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)
        gate_result = GateResult(passed=True, commit_hash="abc123")
        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)
        # review_attempt is 1 after gate_result, max is also 1

        review_result = ReviewResult(
            passed=False,
            parse_error="Invalid JSON response",
        )

        result = lifecycle.on_review_result(ctx, review_result, new_log_offset=200)

        assert lifecycle.state == LifecycleState.FAILED
        assert result.effect == Effect.COMPLETE_FAILURE
        assert "Invalid JSON response" in ctx.final_result

    def test_review_failed_with_fatal_error_fails_immediately(self) -> None:
        """Fatal review errors should fail immediately without retries."""
        lifecycle, ctx = self._setup_for_review()

        review_result = ReviewResult(
            passed=False,
            parse_error="Invalid schema for response_format 'codex_output_schema'",
            fatal_error=True,
        )

        result = lifecycle.on_review_result(ctx, review_result, new_log_offset=200)

        assert lifecycle.state == LifecycleState.FAILED
        assert result.effect == Effect.COMPLETE_FAILURE
        assert "Invalid schema" in ctx.final_result

    def test_review_failed_no_progress_fails_immediately(self) -> None:
        """Review failed with no_progress=True fails without retry.

        This tests the progress gate: if the agent made no progress since
        the last review attempt (same commit, no new validation evidence),
        fail fast instead of burning more retries.
        """
        config = LifecycleConfig(max_review_retries=3, session_end_enabled=False)
        lifecycle = ImplementerLifecycle(config)
        lifecycle.start()
        ctx = LifecycleContext()
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)
        gate_result = GateResult(passed=True, commit_hash="abc123")
        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)

        # First review fails (attempt 1)
        review_result = ReviewResult(
            passed=False,
            issues=[
                ReviewIssue(
                    file="test.py",
                    line_start=10,
                    line_end=10,
                    priority=1,
                    title="Bug",
                    body="",
                    reviewer="cerberus",
                )
            ],
        )
        result = lifecycle.on_review_result(ctx, review_result, new_log_offset=200)
        assert result.effect == Effect.SEND_REVIEW_RETRY
        assert ctx.retry_state.review_attempt == 2

        # Agent "works" but makes no progress, messages complete
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)

        # Gate passes again
        gate_result = GateResult(passed=True, commit_hash="abc123")  # Same commit!
        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=300)

        # Second review also fails, but no_progress=True
        # (orchestrator detected same commit, no new validation evidence)
        review_result = ReviewResult(
            passed=False,
            issues=[
                ReviewIssue(
                    file="test.py",
                    line_start=10,
                    line_end=10,
                    priority=1,
                    title="Bug",
                    body="",
                    reviewer="cerberus",
                )
            ],
        )
        result = lifecycle.on_review_result(
            ctx, review_result, new_log_offset=400, no_progress=True
        )

        # Should fail immediately even though retries remain (2 < 3)
        assert result.effect == Effect.COMPLETE_FAILURE
        assert lifecycle.state == LifecycleState.FAILED
        assert "No progress" in ctx.final_result
        assert result.message is not None
        assert "no progress detected" in result.message.lower()


class TestTimeoutAndError:
    """Tests for timeout and error handling."""

    def test_timeout_fails_from_any_state(self) -> None:
        """Timeout transitions to FAILED from any non-terminal state."""
        lifecycle = ImplementerLifecycle(LifecycleConfig())
        lifecycle.start()
        ctx = LifecycleContext()

        result = lifecycle.on_timeout(ctx, timeout_minutes=30)

        assert lifecycle.state == LifecycleState.FAILED
        assert result.effect == Effect.COMPLETE_FAILURE
        assert "30 minutes" in ctx.final_result

    def test_error_fails_from_any_state(self) -> None:
        """Error transitions to FAILED from any non-terminal state."""
        lifecycle = ImplementerLifecycle(LifecycleConfig())
        lifecycle.start()
        ctx = LifecycleContext()

        result = lifecycle.on_error(ctx, Exception("Connection lost"))

        assert lifecycle.state == LifecycleState.FAILED
        assert result.effect == Effect.COMPLETE_FAILURE
        assert "Connection lost" in ctx.final_result

    def test_timeout_from_terminal_state_is_noop(self) -> None:
        """Timeout from terminal state doesn't change state."""
        lifecycle = ImplementerLifecycle(
            LifecycleConfig(review_enabled=False, session_end_enabled=False)
        )
        lifecycle.start()
        ctx = LifecycleContext()
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)
        gate_result = GateResult(passed=True)
        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)
        assert lifecycle.state == LifecycleState.SUCCESS

        result = lifecycle.on_timeout(ctx, timeout_minutes=30)

        # State unchanged, still SUCCESS
        assert lifecycle.state == LifecycleState.SUCCESS
        # But effect is still COMPLETE_FAILURE
        assert result.effect == Effect.COMPLETE_FAILURE


class TestTerminalStates:
    """Tests for is_terminal property."""

    def test_success_is_terminal(self) -> None:
        """SUCCESS state is terminal."""
        lifecycle = ImplementerLifecycle(
            LifecycleConfig(review_enabled=False, session_end_enabled=False)
        )
        lifecycle.start()
        ctx = LifecycleContext()
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)
        gate_result = GateResult(passed=True)
        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)

        assert lifecycle.is_terminal
        assert lifecycle.state == LifecycleState.SUCCESS

    def test_failed_is_terminal(self) -> None:
        """FAILED state is terminal."""
        lifecycle = ImplementerLifecycle(LifecycleConfig())
        lifecycle.start()
        ctx = LifecycleContext()
        lifecycle.on_messages_complete(ctx, has_session_id=False)

        assert lifecycle.is_terminal
        assert lifecycle.state == LifecycleState.FAILED


class TestFullLifecycleScenarios:
    """Integration tests for complete lifecycle flows."""

    def test_happy_path_gate_only(self) -> None:
        """Full lifecycle: agent completes, gate passes, no review."""
        config = LifecycleConfig(review_enabled=False, session_end_enabled=False)
        lifecycle = ImplementerLifecycle(config)
        ctx = LifecycleContext()

        # 1. Start
        result = lifecycle.start()
        assert result.effect == Effect.CONTINUE

        # 2. Messages complete
        result = lifecycle.on_messages_complete(ctx, has_session_id=True)
        assert result.effect == Effect.WAIT_FOR_LOG

        # 3. Log ready
        result = lifecycle.on_log_ready(ctx)
        assert result.effect == Effect.RUN_GATE

        # 4. Gate passes
        gate_result = GateResult(passed=True, commit_hash="abc123")
        result = lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)

        assert result.effect == Effect.COMPLETE_SUCCESS
        assert lifecycle.is_terminal
        assert ctx.success

    def test_happy_path_with_review(self) -> None:
        """Full lifecycle: agent completes, gate passes, review passes."""
        lifecycle = ImplementerLifecycle(LifecycleConfig(session_end_enabled=False))
        ctx = LifecycleContext()

        lifecycle.start()
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)
        gate_result = GateResult(passed=True, commit_hash="abc123")
        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)

        # Now in RUNNING_REVIEW
        assert lifecycle.state == LifecycleState.RUNNING_REVIEW

        review_result = ReviewResult(passed=True)
        result = lifecycle.on_review_result(ctx, review_result, new_log_offset=200)

        assert result.effect == Effect.COMPLETE_SUCCESS
        assert ctx.success

    def test_gate_retry_then_success(self) -> None:
        """Gate fails once, retry succeeds."""
        config = LifecycleConfig(
            max_gate_retries=3, review_enabled=False, session_end_enabled=False
        )
        lifecycle = ImplementerLifecycle(config)
        ctx = LifecycleContext()

        lifecycle.start()
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)

        # First gate fails
        gate_result = GateResult(passed=False, failure_reasons=["Missing pytest"])
        result = lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)
        assert result.effect == Effect.SEND_GATE_RETRY
        assert ctx.retry_state.gate_attempt == 2

        # Agent fixes issues, messages complete again
        result = lifecycle.on_messages_complete(ctx, has_session_id=True)
        assert result.effect == Effect.WAIT_FOR_LOG

        # Log ready
        lifecycle.on_log_ready(ctx)

        # Second gate passes
        gate_result = GateResult(passed=True, commit_hash="fixed123")
        result = lifecycle.on_gate_result(ctx, gate_result, new_log_offset=200)
        assert result.effect == Effect.COMPLETE_SUCCESS
        assert ctx.success

    def test_review_retry_then_success(self) -> None:
        """Review fails once, retry succeeds."""
        config = LifecycleConfig(max_review_retries=2, session_end_enabled=False)
        lifecycle = ImplementerLifecycle(config)
        ctx = LifecycleContext()

        lifecycle.start()
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)
        gate_result = GateResult(passed=True, commit_hash="abc123")
        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)

        # First review fails
        review_result = ReviewResult(
            passed=False,
            issues=[
                ReviewIssue(
                    file="a.py",
                    line_start=1,
                    line_end=1,
                    priority=1,
                    title="Bug",
                    body="",
                    reviewer="cerberus",
                )
            ],
        )
        result = lifecycle.on_review_result(ctx, review_result, new_log_offset=200)
        assert result.effect == Effect.SEND_REVIEW_RETRY
        assert ctx.retry_state.review_attempt == 2

        # Agent fixes, messages complete
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)

        # Gate passes again
        gate_result = GateResult(passed=True, commit_hash="fixed123")
        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=300)

        # Review passes
        review_result = ReviewResult(passed=True)
        result = lifecycle.on_review_result(ctx, review_result, new_log_offset=400)
        assert result.effect == Effect.COMPLETE_SUCCESS
        assert ctx.success

    def test_review_attempt_persists_across_gate_rerun(self) -> None:
        """Review attempt counter persists when gate runs again after review retry.

        This tests the scenario:
        1. Gate passes, review runs (attempt 1)
        2. Review fails, retry sent
        3. Agent fixes, gate runs again
        4. Gate passes - review_attempt should still be 2 (not reset to 1)
        5. Review fails again - should hit max_review_retries and fail
        """
        config = LifecycleConfig(max_review_retries=2, session_end_enabled=False)
        lifecycle = ImplementerLifecycle(config)
        ctx = LifecycleContext()

        # Initial flow to first review
        lifecycle.start()
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)
        gate_result = GateResult(passed=True, commit_hash="abc123")
        lifecycle.on_gate_result(ctx, gate_result, new_log_offset=100)

        # First review fails (attempt 1)
        assert ctx.retry_state.review_attempt == 1
        review_result = ReviewResult(
            passed=False,
            issues=[
                ReviewIssue(
                    file="a.py",
                    line_start=1,
                    line_end=1,
                    priority=1,
                    title="Bug",
                    body="",
                    reviewer="cerberus",
                )
            ],
        )
        result = lifecycle.on_review_result(ctx, review_result, new_log_offset=200)
        assert result.effect == Effect.SEND_REVIEW_RETRY
        assert ctx.retry_state.review_attempt == 2

        # Agent fixes, messages complete, gate runs again
        lifecycle.on_messages_complete(ctx, has_session_id=True)
        lifecycle.on_log_ready(ctx)

        # Gate passes again - review_attempt should NOT reset
        gate_result = GateResult(passed=True, commit_hash="fixed123")
        result = lifecycle.on_gate_result(ctx, gate_result, new_log_offset=300)
        assert result.effect == Effect.RUN_REVIEW
        # Critical check: review_attempt should still be 2, not reset to 1
        assert ctx.retry_state.review_attempt == 2

        # Second review also fails - should hit max and fail
        review_result = ReviewResult(
            passed=False,
            issues=[
                ReviewIssue(
                    file="a.py",
                    line_start=5,
                    line_end=5,
                    priority=1,
                    title="Another bug",
                    body="",
                    reviewer="cerberus",
                )
            ],
        )
        result = lifecycle.on_review_result(ctx, review_result, new_log_offset=400)
        # Should fail because we're at max_review_retries (2)
        assert result.effect == Effect.COMPLETE_FAILURE
        assert lifecycle.state == LifecycleState.FAILED
        assert "Another bug" in ctx.final_result

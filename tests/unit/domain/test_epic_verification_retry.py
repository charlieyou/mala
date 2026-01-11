"""Unit tests for epic verification retry loop in EpicVerificationCoordinator.

Tests the retry logic in EpicVerificationCoordinator.check_epic_closure which:
1. Runs epic verification
2. If verification fails and creates remediation issues, executes them
3. Re-verifies the epic
4. Repeats until verification passes OR max retries reached
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from src.infra.io.config import MalaConfig
from src.pipeline.issue_result import IssueResult
from src.pipeline.epic_verification_coordinator import (
    EpicVerificationCallbacks,
    EpicVerificationConfig,
    EpicVerificationCoordinator,
)
from tests.fakes import (
    FakeVerificationResults,
    make_failing_verification_result as make_failing_result,
    make_not_eligible_verification_result as make_not_eligible_result,
    make_passing_verification_result as make_passing_result,
)

if TYPE_CHECKING:
    from src.core.models import EpicVerificationResult
    from src.domain.validation.config import EpicCompletionTriggerConfig, TriggerType
    from src.infra.io.log_output.run_metadata import RunMetadata


# ---------------------------------------------------------------------------
# Test helpers: Fake callbacks
# ---------------------------------------------------------------------------


@dataclass
class FakeCallbacks:
    """Collection of fake callbacks for EpicVerificationCoordinator.

    Uses FakeVerificationResults to provide coordinator-level verification
    results (EpicVerificationResult). For test assertions, use
    fake_verifier.attempts[i] (CoordinatorVerificationAttempt) to inspect
    each verification call.

    Observable state:
        spawned_issues: issues passed to spawn_remediation
        finalized_results: (issue_id, result) pairs passed to finalize
        completed_issues: issues passed to mark_completed
        warnings: warning messages emitted
        closed_epics: epic IDs emitted via on_epic_closed
    """

    parent_epic: str | None = "epic-1"
    fake_verifier: FakeVerificationResults = field(
        default_factory=FakeVerificationResults
    )
    has_epic_verifier: bool = True
    spawn_returns_task: bool = True
    spawn_raises: Exception | None = None
    task_success: bool = True
    task_error: Exception | None = None

    # Observable state
    spawned_issues: list[str] = field(default_factory=list)
    finalized_results: list[tuple[str, IssueResult]] = field(default_factory=list)
    completed_issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    closed_epics: list[str] = field(default_factory=list)
    eligible_epics_closed: bool = field(default=False)

    async def get_parent_epic(self, issue_id: str) -> str | None:
        return self.parent_epic

    async def verify_epic(self, epic_id: str, force: bool) -> EpicVerificationResult:
        return await self.fake_verifier.verify(epic_id, force)

    async def spawn_remediation(
        self, issue_id: str, flow: str = "implementer"
    ) -> asyncio.Task[IssueResult] | None:
        self.spawned_issues.append(issue_id)
        if self.spawn_raises:
            raise self.spawn_raises
        if not self.spawn_returns_task:
            return None

        async def run_task() -> IssueResult:
            if self.task_error:
                raise self.task_error
            return IssueResult(
                issue_id=issue_id,
                agent_id="test-agent",
                success=self.task_success,
                summary="done" if self.task_success else "failed",
            )

        return asyncio.create_task(run_task())

    async def finalize_remediation(
        self, issue_id: str, result: IssueResult, run_metadata: object
    ) -> None:
        self.finalized_results.append((issue_id, result))

    def mark_completed(self, issue_id: str) -> None:
        self.completed_issues.append(issue_id)

    def is_issue_failed(self, issue_id: str) -> bool:
        return False

    async def close_eligible_epics(self) -> bool:
        self.eligible_epics_closed = True
        return True

    def on_epic_closed(self, issue_id: str) -> None:
        self.closed_epics.append(issue_id)

    def on_warning(self, message: str) -> None:
        self.warnings.append(message)

    def get_agent_id(self, issue_id: str) -> str:
        return "test-agent"

    def queue_trigger_validation(
        self, trigger_type: TriggerType, context: dict[str, Any]
    ) -> None:
        pass

    def get_epic_completion_trigger(self) -> EpicCompletionTriggerConfig | None:
        return None

    def to_callbacks(self) -> EpicVerificationCallbacks:
        """Convert to EpicVerificationCallbacks for coordinator injection."""
        return EpicVerificationCallbacks(
            get_parent_epic=self.get_parent_epic,
            verify_epic=self.verify_epic,
            spawn_remediation=self.spawn_remediation,
            finalize_remediation=self.finalize_remediation,
            mark_completed=self.mark_completed,
            is_issue_failed=self.is_issue_failed,
            close_eligible_epics=self.close_eligible_epics,
            on_epic_closed=self.on_epic_closed,
            on_warning=self.on_warning,
            has_epic_verifier=lambda: self.has_epic_verifier,
            get_agent_id=self.get_agent_id,
            queue_trigger_validation=self.queue_trigger_validation,
            get_epic_completion_trigger=self.get_epic_completion_trigger,
        )


def stub_run_metadata() -> RunMetadata:
    """Create a stub RunMetadata for tests.

    The coordinator only passes run_metadata through to callbacks,
    so we can use an object() cast to the expected type.
    """
    return cast("RunMetadata", object())


# ---------------------------------------------------------------------------
# Tests: Retry loop behavior
# ---------------------------------------------------------------------------


class TestEpicVerificationRetryLoop:
    """Tests for the epic verification retry loop."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "verdict_sequence,expected_attempts,should_pass",
        [
            # Pass on first attempt
            pytest.param(
                [make_passing_result()],
                1,
                True,
                id="pass_first_attempt",
            ),
            # Fail then pass (with remediation)
            pytest.param(
                [
                    make_failing_result(remediation_issues=["rem-1"]),
                    make_passing_result(),
                ],
                2,
                True,
                id="fail_then_pass",
            ),
            # Fail twice then pass
            pytest.param(
                [
                    make_failing_result(remediation_issues=["rem-1"]),
                    make_failing_result(remediation_issues=["rem-2"]),
                    make_passing_result(),
                ],
                3,
                True,
                id="fail_twice_then_pass",
            ),
        ],
    )
    async def test_retry_scenarios(
        self,
        verdict_sequence: list[EpicVerificationResult],
        expected_attempts: int,
        should_pass: bool,
    ) -> None:
        """Verify retry behavior with different verdict sequences."""
        fake_verifier = FakeVerificationResults(results=verdict_sequence)
        fake_callbacks = FakeCallbacks(fake_verifier=fake_verifier)

        coordinator = EpicVerificationCoordinator(
            config=EpicVerificationConfig(max_retries=3),
            callbacks=fake_callbacks.to_callbacks(),
        )

        await coordinator.check_epic_closure("child-1", stub_run_metadata())

        # Assert on observable attempts
        assert len(fake_verifier.attempts) == expected_attempts
        # Final attempt should match expected pass/fail
        final_attempt = fake_verifier.attempts[-1]
        assert final_attempt.result.passed_count == (1 if should_pass else 0)
        # Epic should be marked as verified
        assert "epic-1" in coordinator.verified_epics

    @pytest.mark.asyncio
    async def test_stops_at_max_retries(self) -> None:
        """Should stop retrying after max_retries, even if still failing."""
        # All attempts fail with remediation issues
        fake_verifier = FakeVerificationResults(
            results=[make_failing_result(remediation_issues=["rem-1"])] * 10
        )
        fake_callbacks = FakeCallbacks(fake_verifier=fake_verifier)

        coordinator = EpicVerificationCoordinator(
            config=EpicVerificationConfig(max_retries=3),
            callbacks=fake_callbacks.to_callbacks(),
        )

        await coordinator.check_epic_closure("child-1", stub_run_metadata())

        # Should have attempted 1 initial + 3 retries = 4 times
        assert len(fake_verifier.attempts) == 4
        # All attempts should have failed
        for attempt in fake_verifier.attempts:
            assert attempt.result.passed_count == 0
        # Should still mark as verified (to prevent infinite loops)
        assert "epic-1" in coordinator.verified_epics
        # Should have logged warning about max retries
        assert any("failed after" in w for w in fake_callbacks.warnings)

    @pytest.mark.asyncio
    async def test_stops_when_no_remediation_issues(self) -> None:
        """Should stop if verification fails but no remediation issues created."""
        # Fail without remediation issues - no retry possible
        fake_verifier = FakeVerificationResults(
            results=[make_failing_result(remediation_issues=[])]
        )
        fake_callbacks = FakeCallbacks(fake_verifier=fake_verifier)

        coordinator = EpicVerificationCoordinator(
            config=EpicVerificationConfig(max_retries=3),
            callbacks=fake_callbacks.to_callbacks(),
        )

        await coordinator.check_epic_closure("child-1", stub_run_metadata())

        # Should only call verify once (no retry since no remediation issues)
        assert len(fake_verifier.attempts) == 1
        # No remediation issues spawned
        assert fake_callbacks.spawned_issues == []


class TestEpicVerificationSkipConditions:
    """Tests for conditions that skip verification entirely."""

    @pytest.mark.asyncio
    async def test_skips_already_verified_epic(self) -> None:
        """Should skip verification for already verified epics."""
        fake_verifier = FakeVerificationResults(results=[make_passing_result()])
        fake_callbacks = FakeCallbacks(fake_verifier=fake_verifier)

        coordinator = EpicVerificationCoordinator(
            config=EpicVerificationConfig(max_retries=3),
            callbacks=fake_callbacks.to_callbacks(),
        )
        coordinator.verified_epics.add("epic-1")  # Already verified

        await coordinator.check_epic_closure("child-1", stub_run_metadata())

        # Should not call verify
        assert len(fake_verifier.attempts) == 0

    @pytest.mark.asyncio
    async def test_skips_when_no_parent_epic(self) -> None:
        """Should skip verification when issue has no parent epic."""
        fake_verifier = FakeVerificationResults(results=[make_passing_result()])
        fake_callbacks = FakeCallbacks(fake_verifier=fake_verifier, parent_epic=None)

        coordinator = EpicVerificationCoordinator(
            config=EpicVerificationConfig(max_retries=3),
            callbacks=fake_callbacks.to_callbacks(),
        )

        await coordinator.check_epic_closure("child-1", stub_run_metadata())

        # Should not call verify
        assert len(fake_verifier.attempts) == 0

    @pytest.mark.asyncio
    async def test_does_not_mark_verified_when_not_eligible(self) -> None:
        """Should NOT mark epic as verified when not eligible (children still open)."""
        fake_verifier = FakeVerificationResults(results=[make_not_eligible_result()])
        fake_callbacks = FakeCallbacks(fake_verifier=fake_verifier)

        coordinator = EpicVerificationCoordinator(
            config=EpicVerificationConfig(max_retries=3),
            callbacks=fake_callbacks.to_callbacks(),
        )

        await coordinator.check_epic_closure("child-1", stub_run_metadata())

        # Should have called verify once
        assert len(fake_verifier.attempts) == 1
        # Should NOT mark as verified (so it can be re-checked later)
        assert "epic-1" not in coordinator.verified_epics


class TestReentryGuard:
    """Tests for re-entrant verification guard."""

    @pytest.mark.asyncio
    async def test_skips_epic_being_verified(self) -> None:
        """Should skip verification for epics already being verified."""
        fake_verifier = FakeVerificationResults(results=[make_passing_result()])
        fake_callbacks = FakeCallbacks(fake_verifier=fake_verifier)

        coordinator = EpicVerificationCoordinator(
            config=EpicVerificationConfig(max_retries=3),
            callbacks=fake_callbacks.to_callbacks(),
        )
        coordinator.epics_being_verified.add("epic-1")  # Already being verified

        await coordinator.check_epic_closure("child-1", stub_run_metadata())

        # Should NOT call verify due to re-entry guard
        assert len(fake_verifier.attempts) == 0

    @pytest.mark.asyncio
    async def test_removes_from_being_verified_on_completion(self) -> None:
        """Should remove epic from epics_being_verified when done."""
        fake_verifier = FakeVerificationResults(results=[make_passing_result()])
        fake_callbacks = FakeCallbacks(fake_verifier=fake_verifier)

        coordinator = EpicVerificationCoordinator(
            config=EpicVerificationConfig(max_retries=3),
            callbacks=fake_callbacks.to_callbacks(),
        )

        await coordinator.check_epic_closure("child-1", stub_run_metadata())

        # Should have removed from epics_being_verified
        assert "epic-1" not in coordinator.epics_being_verified

    @pytest.mark.asyncio
    async def test_removes_from_being_verified_on_error(self) -> None:
        """Should remove epic from epics_being_verified even on error."""
        fake_verifier = FakeVerificationResults(results=[])
        fake_callbacks = FakeCallbacks(fake_verifier=fake_verifier)

        # Make verify raise an exception
        async def raise_error(epic_id: str, force: bool) -> EpicVerificationResult:
            raise RuntimeError("Test error")

        fake_callbacks.verify_epic = raise_error  # type: ignore[method-assign]

        coordinator = EpicVerificationCoordinator(
            config=EpicVerificationConfig(max_retries=3),
            callbacks=fake_callbacks.to_callbacks(),
        )

        with pytest.raises(RuntimeError, match="Test error"):
            await coordinator.check_epic_closure("child-1", stub_run_metadata())

        # Should have removed from epics_being_verified even on error
        assert "epic-1" not in coordinator.epics_being_verified


class TestRemediationExecution:
    """Tests for _execute_remediation_issues method."""

    @pytest.mark.asyncio
    async def test_spawns_and_finalizes_remediation_issues(self) -> None:
        """Should spawn tasks and finalize results for remediation issues."""
        fake_verifier = FakeVerificationResults(
            results=[
                make_failing_result(remediation_issues=["rem-1", "rem-2"]),
                make_passing_result(),
            ]
        )
        fake_callbacks = FakeCallbacks(fake_verifier=fake_verifier)

        coordinator = EpicVerificationCoordinator(
            config=EpicVerificationConfig(max_retries=3),
            callbacks=fake_callbacks.to_callbacks(),
        )

        await coordinator.check_epic_closure("child-1", stub_run_metadata())

        # Should have spawned both remediation issues
        assert fake_callbacks.spawned_issues == ["rem-1", "rem-2"]
        # Should have finalized both
        assert len(fake_callbacks.finalized_results) == 2
        finalized_ids = [r[0] for r in fake_callbacks.finalized_results]
        assert "rem-1" in finalized_ids
        assert "rem-2" in finalized_ids
        # Should have marked both as completed
        assert "rem-1" in fake_callbacks.completed_issues
        assert "rem-2" in fake_callbacks.completed_issues

    @pytest.mark.asyncio
    async def test_handles_task_exceptions(self) -> None:
        """Should handle task exceptions and still finalize with error result."""
        fake_verifier = FakeVerificationResults(
            results=[
                make_failing_result(remediation_issues=["rem-1"]),
                make_passing_result(),
            ]
        )
        fake_callbacks = FakeCallbacks(
            fake_verifier=fake_verifier, task_error=RuntimeError("Task failed!")
        )

        coordinator = EpicVerificationCoordinator(
            config=EpicVerificationConfig(max_retries=3),
            callbacks=fake_callbacks.to_callbacks(),
        )

        await coordinator.check_epic_closure("child-1", stub_run_metadata())

        # Should have finalized with failure result
        assert len(fake_callbacks.finalized_results) == 1
        issue_id, result = fake_callbacks.finalized_results[0]
        assert issue_id == "rem-1"
        assert not result.success
        assert "Task failed!" in result.summary


class TestFallbackBehavior:
    """Tests for fallback behavior when EpicVerifier is not available."""

    @pytest.mark.asyncio
    async def test_uses_fallback_when_no_epic_verifier(self) -> None:
        """Should use close_eligible_epics fallback when no EpicVerifier."""
        fake_verifier = FakeVerificationResults(results=[make_passing_result()])
        fake_callbacks = FakeCallbacks(
            fake_verifier=fake_verifier, has_epic_verifier=False
        )

        coordinator = EpicVerificationCoordinator(
            config=EpicVerificationConfig(max_retries=3),
            callbacks=fake_callbacks.to_callbacks(),
        )

        await coordinator.check_epic_closure("child-1", stub_run_metadata())

        # Should NOT call verify
        assert len(fake_verifier.attempts) == 0
        # Should call close_eligible_epics fallback
        assert fake_callbacks.eligible_epics_closed is True
        # Should emit on_epic_closed
        assert "child-1" in fake_callbacks.closed_epics


class TestMalaConfigEpicVerificationRetries:
    """Tests for max_epic_verification_retries config."""

    def test_default_value(self) -> None:
        """Should default to 3 retries."""
        config = MalaConfig(
            runs_dir=Path("/tmp/runs"),
            lock_dir=Path("/tmp/locks"),
            claude_config_dir=Path("/tmp/claude"),
        )
        assert config.max_epic_verification_retries == 3

    def test_custom_value(self) -> None:
        """Should accept custom retry count."""
        config = MalaConfig(
            runs_dir=Path("/tmp/runs"),
            lock_dir=Path("/tmp/locks"),
            claude_config_dir=Path("/tmp/claude"),
            max_epic_verification_retries=5,
        )
        assert config.max_epic_verification_retries == 5

    def test_from_env_ignores_deprecated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should ignore MALA_MAX_EPIC_VERIFICATION_RETRIES and emit deprecation warning."""
        monkeypatch.setenv("MALA_MAX_EPIC_VERIFICATION_RETRIES", "7")
        with pytest.warns(
            DeprecationWarning, match="MALA_MAX_EPIC_VERIFICATION_RETRIES is deprecated"
        ):
            config = MalaConfig.from_env(validate=False)
        # Env var is ignored; default is used
        assert config.max_epic_verification_retries == 3

"""Unit tests for epic verification retry loop in EpicVerificationCoordinator.

Tests the retry logic in EpicVerificationCoordinator.check_epic_closure which:
1. Runs epic verification
2. If verification fails and creates remediation issues, executes them
3. Re-verifies the epic
4. Repeats until verification passes OR max retries reached
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.models import EpicVerdict, EpicVerificationResult
from src.infra.io.config import MalaConfig
from src.pipeline.issue_result import IssueResult
from src.pipeline.epic_verification_coordinator import (
    EpicVerificationCallbacks,
    EpicVerificationConfig,
    EpicVerificationCoordinator,
)


@pytest.fixture
def mock_beads() -> MagicMock:
    """Create a mock BeadsClient."""
    beads = MagicMock()
    beads.get_parent_epic_async = AsyncMock(return_value="epic-1")
    beads.claim_async = AsyncMock(return_value=True)
    beads.close_async = AsyncMock(return_value=True)
    beads.close_eligible_epics_async = AsyncMock(return_value=True)
    beads.get_issue_description_async = AsyncMock(return_value="Issue description")
    return beads


@pytest.fixture
def mock_event_sink() -> MagicMock:
    """Create a mock event sink."""
    sink = MagicMock()
    sink.on_warning = MagicMock()
    sink.on_epic_closed = MagicMock()
    sink.on_agent_started = MagicMock()
    return sink


@pytest.fixture
def mock_epic_verifier() -> MagicMock:
    """Create a mock EpicVerifier."""
    verifier = MagicMock()
    verifier.verify_and_close_epic = AsyncMock(
        return_value=EpicVerificationResult(
            verified_count=1,
            passed_count=1,
            failed_count=0,
            verdicts={
                "epic-1": EpicVerdict(
                    passed=True,
                    unmet_criteria=[],
                    confidence=0.9,
                    reasoning="All criteria met",
                )
            },
            remediation_issues_created=[],
        )
    )
    return verifier


@pytest.fixture
def mock_mala_config() -> MalaConfig:
    """Create a MalaConfig with test values."""
    return MalaConfig(
        runs_dir=Path("/tmp/test-runs"),
        lock_dir=Path("/tmp/test-locks"),
        claude_config_dir=Path("/tmp/test-claude"),
        max_epic_verification_retries=3,
    )


@pytest.fixture
def mock_run_metadata() -> MagicMock:
    """Create a mock RunMetadata."""
    return MagicMock()


def create_coordinator(
    mock_beads: MagicMock,
    mock_event_sink: MagicMock,
    mock_epic_verifier: MagicMock,
    max_retries: int = 3,
    epic_override_ids: set[str] | None = None,
    has_epic_verifier: bool = True,
    spawn_remediation: AsyncMock | None = None,
    finalize_remediation: AsyncMock | None = None,
    mark_completed: MagicMock | None = None,
    is_issue_failed: MagicMock | None = None,
    get_agent_id: MagicMock | None = None,
) -> EpicVerificationCoordinator:
    """Create a coordinator with mock callbacks."""
    callbacks = EpicVerificationCallbacks(
        get_parent_epic=mock_beads.get_parent_epic_async,
        verify_epic=mock_epic_verifier.verify_and_close_epic,
        spawn_remediation=spawn_remediation or AsyncMock(return_value=None),
        finalize_remediation=finalize_remediation or AsyncMock(),
        mark_completed=mark_completed or MagicMock(),
        is_issue_failed=is_issue_failed or MagicMock(return_value=False),
        close_eligible_epics=mock_beads.close_eligible_epics_async,
        on_epic_closed=mock_event_sink.on_epic_closed,
        on_warning=mock_event_sink.on_warning,
        has_epic_verifier=lambda: has_epic_verifier,
        get_agent_id=get_agent_id or MagicMock(return_value="unknown"),
    )
    config = EpicVerificationConfig(max_retries=max_retries)
    return EpicVerificationCoordinator(
        config=config,
        callbacks=callbacks,
        epic_override_ids=epic_override_ids or set(),
    )


class TestEpicVerificationRetryLoop:
    """Tests for the epic verification retry loop."""

    @pytest.mark.asyncio
    async def test_passes_on_first_attempt(
        self,
        mock_beads: MagicMock,
        mock_event_sink: MagicMock,
        mock_epic_verifier: MagicMock,
        mock_run_metadata: MagicMock,
    ) -> None:
        """Should close epic without retry when verification passes first time."""
        coordinator = create_coordinator(
            mock_beads, mock_event_sink, mock_epic_verifier
        )

        await coordinator.check_epic_closure("child-1", mock_run_metadata)

        # Should have called verify_and_close_epic once
        mock_epic_verifier.verify_and_close_epic.assert_called_once_with(
            "epic-1", False
        )
        # Should have marked epic as verified
        assert "epic-1" in coordinator.verified_epics

    @pytest.mark.asyncio
    async def test_retries_when_remediation_issues_created(
        self,
        mock_beads: MagicMock,
        mock_event_sink: MagicMock,
        mock_epic_verifier: MagicMock,
        mock_run_metadata: MagicMock,
    ) -> None:
        """Should retry verification after executing remediation issues."""
        # First call fails with remediation issues, second call passes
        mock_epic_verifier.verify_and_close_epic = AsyncMock(
            side_effect=[
                EpicVerificationResult(
                    verified_count=1,
                    passed_count=0,
                    failed_count=1,
                    verdicts={},
                    remediation_issues_created=["rem-1", "rem-2"],
                ),
                EpicVerificationResult(
                    verified_count=1,
                    passed_count=1,
                    failed_count=0,
                    verdicts={},
                    remediation_issues_created=[],
                ),
            ]
        )

        # Mock spawn_remediation to return tasks
        async def mock_spawn(issue_id: str) -> asyncio.Task[IssueResult]:
            async def dummy() -> IssueResult:
                return IssueResult(
                    issue_id=issue_id, agent_id="test", success=True, summary="done"
                )

            return asyncio.create_task(dummy())

        spawn_mock = AsyncMock(side_effect=mock_spawn)
        coordinator = create_coordinator(
            mock_beads,
            mock_event_sink,
            mock_epic_verifier,
            spawn_remediation=spawn_mock,
        )

        await coordinator.check_epic_closure("child-1", mock_run_metadata)

        # Should have called verify_and_close_epic twice
        assert mock_epic_verifier.verify_and_close_epic.call_count == 2
        # Should have logged retry attempt
        mock_event_sink.on_warning.assert_called()
        # Should have marked epic as verified after passing
        assert "epic-1" in coordinator.verified_epics

    @pytest.mark.asyncio
    async def test_stops_at_max_retries(
        self,
        mock_beads: MagicMock,
        mock_event_sink: MagicMock,
        mock_epic_verifier: MagicMock,
        mock_run_metadata: MagicMock,
    ) -> None:
        """Should stop retrying after max_epic_verification_retries."""
        # All calls fail with remediation issues
        mock_epic_verifier.verify_and_close_epic = AsyncMock(
            return_value=EpicVerificationResult(
                verified_count=1,
                passed_count=0,
                failed_count=1,
                verdicts={},
                remediation_issues_created=["rem-1"],
            )
        )

        # Mock spawn_remediation
        async def mock_spawn(issue_id: str) -> asyncio.Task[IssueResult]:
            async def dummy() -> IssueResult:
                return IssueResult(
                    issue_id=issue_id, agent_id="test", success=True, summary="done"
                )

            return asyncio.create_task(dummy())

        spawn_mock = AsyncMock(side_effect=mock_spawn)
        coordinator = create_coordinator(
            mock_beads,
            mock_event_sink,
            mock_epic_verifier,
            spawn_remediation=spawn_mock,
        )

        await coordinator.check_epic_closure("child-1", mock_run_metadata)

        # Should have called verify exactly max_attempts times (1 + 3 retries = 4)
        assert mock_epic_verifier.verify_and_close_epic.call_count == 4
        # Should have logged max retries warning
        assert any(
            "failed after" in str(call)
            for call in mock_event_sink.on_warning.call_args_list
        )
        # Should still mark as verified (to prevent infinite loops)
        assert "epic-1" in coordinator.verified_epics

    @pytest.mark.asyncio
    async def test_stops_when_no_remediation_issues(
        self,
        mock_beads: MagicMock,
        mock_event_sink: MagicMock,
        mock_epic_verifier: MagicMock,
        mock_run_metadata: MagicMock,
    ) -> None:
        """Should stop if verification fails but no remediation issues created."""
        # Fail without creating remediation issues
        mock_epic_verifier.verify_and_close_epic = AsyncMock(
            return_value=EpicVerificationResult(
                verified_count=1,
                passed_count=0,
                failed_count=1,
                verdicts={},
                remediation_issues_created=[],  # No remediation issues
            )
        )

        coordinator = create_coordinator(
            mock_beads, mock_event_sink, mock_epic_verifier
        )

        await coordinator.check_epic_closure("child-1", mock_run_metadata)

        # Should only call verify once (no retry since no remediation issues)
        assert mock_epic_verifier.verify_and_close_epic.call_count == 1
        # Should mark as verified
        assert "epic-1" in coordinator.verified_epics

    @pytest.mark.asyncio
    async def test_skips_already_verified_epic(
        self,
        mock_beads: MagicMock,
        mock_event_sink: MagicMock,
        mock_epic_verifier: MagicMock,
        mock_run_metadata: MagicMock,
    ) -> None:
        """Should skip verification for already verified epics."""
        coordinator = create_coordinator(
            mock_beads, mock_event_sink, mock_epic_verifier
        )
        coordinator.verified_epics.add("epic-1")  # Already verified

        await coordinator.check_epic_closure("child-1", mock_run_metadata)

        # Should not call verify_and_close_epic
        mock_epic_verifier.verify_and_close_epic.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_no_parent_epic(
        self,
        mock_beads: MagicMock,
        mock_event_sink: MagicMock,
        mock_epic_verifier: MagicMock,
        mock_run_metadata: MagicMock,
    ) -> None:
        """Should skip verification when issue has no parent epic."""
        mock_beads.get_parent_epic_async = AsyncMock(return_value=None)

        coordinator = create_coordinator(
            mock_beads, mock_event_sink, mock_epic_verifier
        )

        await coordinator.check_epic_closure("child-1", mock_run_metadata)

        # Should not call verify_and_close_epic
        mock_epic_verifier.verify_and_close_epic.assert_not_called()


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

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should load from MALA_MAX_EPIC_VERIFICATION_RETRIES env var."""
        monkeypatch.setenv("MALA_MAX_EPIC_VERIFICATION_RETRIES", "7")
        config = MalaConfig.from_env(validate=False)
        assert config.max_epic_verification_retries == 7

    def test_from_env_invalid_uses_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Should use default when env var is invalid."""
        monkeypatch.setenv("MALA_MAX_EPIC_VERIFICATION_RETRIES", "not-a-number")
        config = MalaConfig.from_env(validate=False)
        assert config.max_epic_verification_retries == 3


class TestExecuteRemediationIssues:
    """Tests for _execute_remediation_issues method in coordinator."""

    @pytest.mark.asyncio
    async def test_uses_task_returned_by_spawn_agent(
        self,
        mock_beads: MagicMock,
        mock_event_sink: MagicMock,
        mock_epic_verifier: MagicMock,
    ) -> None:
        """Should use task returned by spawn_remediation directly."""
        tasks_awaited: set[str] = set()

        async def mock_spawn(issue_id: str) -> asyncio.Task[IssueResult]:
            """Return a task directly."""

            async def dummy_result() -> IssueResult:
                tasks_awaited.add(issue_id)
                return IssueResult(
                    issue_id=issue_id, agent_id="test", success=True, summary="done"
                )

            return asyncio.create_task(dummy_result())

        spawn_mock = AsyncMock(side_effect=mock_spawn)
        mark_completed = MagicMock()

        coordinator = create_coordinator(
            mock_beads,
            mock_event_sink,
            mock_epic_verifier,
            spawn_remediation=spawn_mock,
            mark_completed=mark_completed,
        )

        mock_run_metadata = MagicMock()
        await coordinator._execute_remediation_issues(
            ["rem-1", "rem-2"], mock_run_metadata
        )

        # Verify all tasks were actually awaited
        assert tasks_awaited == {
            "rem-1",
            "rem-2",
        }, f"All tasks should be awaited, but only got: {tasks_awaited}"

        # Verify mark_completed was called for each issue
        assert mark_completed.call_count == 2

    @pytest.mark.asyncio
    async def test_finalizes_results_with_run_metadata(
        self,
        mock_beads: MagicMock,
        mock_event_sink: MagicMock,
        mock_epic_verifier: MagicMock,
    ) -> None:
        """Should finalize results when run_metadata is provided."""

        async def mock_spawn(issue_id: str) -> asyncio.Task[IssueResult]:
            async def dummy_result() -> IssueResult:
                return IssueResult(
                    issue_id=issue_id, agent_id="test", success=True, summary="done"
                )

            return asyncio.create_task(dummy_result())

        spawn_mock = AsyncMock(side_effect=mock_spawn)
        finalize_mock = AsyncMock()

        coordinator = create_coordinator(
            mock_beads,
            mock_event_sink,
            mock_epic_verifier,
            spawn_remediation=spawn_mock,
            finalize_remediation=finalize_mock,
        )

        mock_run_metadata = MagicMock()

        await coordinator._execute_remediation_issues(
            ["rem-1", "rem-2"], mock_run_metadata
        )

        # Verify finalize_remediation was called for each issue
        assert finalize_mock.call_count == 2
        # Verify run_metadata was passed
        for call in finalize_mock.call_args_list:
            assert call[0][2] == mock_run_metadata

    @pytest.mark.asyncio
    async def test_handles_task_exceptions(
        self,
        mock_beads: MagicMock,
        mock_event_sink: MagicMock,
        mock_epic_verifier: MagicMock,
    ) -> None:
        """Should handle task exceptions and still finalize with error result."""

        async def mock_spawn(issue_id: str) -> asyncio.Task[IssueResult]:
            async def failing_result() -> IssueResult:
                raise RuntimeError("Task failed!")

            return asyncio.create_task(failing_result())

        spawn_mock = AsyncMock(side_effect=mock_spawn)
        finalize_mock = AsyncMock()

        coordinator = create_coordinator(
            mock_beads,
            mock_event_sink,
            mock_epic_verifier,
            spawn_remediation=spawn_mock,
            finalize_remediation=finalize_mock,
        )

        mock_run_metadata = MagicMock()

        await coordinator._execute_remediation_issues(["rem-1"], mock_run_metadata)

        # Verify finalize_remediation was still called
        assert finalize_mock.call_count == 1
        # Verify the result indicates failure
        call_args = finalize_mock.call_args[0]
        result = call_args[1]
        assert not result.success
        assert "Task failed!" in result.summary


class TestEpicNotEligible:
    """Tests for handling epics that aren't eligible yet (children still open)."""

    @pytest.mark.asyncio
    async def test_does_not_mark_verified_when_not_eligible(
        self,
        mock_beads: MagicMock,
        mock_event_sink: MagicMock,
        mock_epic_verifier: MagicMock,
        mock_run_metadata: MagicMock,
    ) -> None:
        """Should NOT mark epic as verified when it isn't eligible (verified_count=0)."""
        # Return result indicating epic wasn't eligible (children still open)
        mock_epic_verifier.verify_and_close_epic = AsyncMock(
            return_value=EpicVerificationResult(
                verified_count=0,  # Epic not eligible - children still open
                passed_count=0,
                failed_count=0,
                verdicts={},
                remediation_issues_created=[],
            )
        )

        coordinator = create_coordinator(
            mock_beads, mock_event_sink, mock_epic_verifier
        )

        await coordinator.check_epic_closure("child-1", mock_run_metadata)

        # Should NOT mark epic as verified (so it can be re-checked later)
        assert "epic-1" not in coordinator.verified_epics
        # Should have called verify_and_close_epic exactly once
        assert mock_epic_verifier.verify_and_close_epic.call_count == 1


class TestReentryGuard:
    """Tests for re-entrant verification guard."""

    @pytest.mark.asyncio
    async def test_skips_epic_being_verified(
        self,
        mock_beads: MagicMock,
        mock_event_sink: MagicMock,
        mock_epic_verifier: MagicMock,
        mock_run_metadata: MagicMock,
    ) -> None:
        """Should skip verification for epics already being verified."""
        coordinator = create_coordinator(
            mock_beads, mock_event_sink, mock_epic_verifier
        )
        coordinator.epics_being_verified.add("epic-1")  # Already being verified

        await coordinator.check_epic_closure("child-1", mock_run_metadata)

        # Should NOT call verify_and_close_epic due to re-entry guard
        mock_epic_verifier.verify_and_close_epic.assert_not_called()

    @pytest.mark.asyncio
    async def test_removes_from_being_verified_on_completion(
        self,
        mock_beads: MagicMock,
        mock_event_sink: MagicMock,
        mock_epic_verifier: MagicMock,
        mock_run_metadata: MagicMock,
    ) -> None:
        """Should remove epic from epics_being_verified when done."""
        coordinator = create_coordinator(
            mock_beads, mock_event_sink, mock_epic_verifier
        )

        await coordinator.check_epic_closure("child-1", mock_run_metadata)

        # Should have removed from epics_being_verified
        assert "epic-1" not in coordinator.epics_being_verified

    @pytest.mark.asyncio
    async def test_removes_from_being_verified_on_error(
        self,
        mock_beads: MagicMock,
        mock_event_sink: MagicMock,
        mock_epic_verifier: MagicMock,
        mock_run_metadata: MagicMock,
    ) -> None:
        """Should remove epic from epics_being_verified even on error."""
        # Make verify_and_close_epic raise an exception
        mock_epic_verifier.verify_and_close_epic = AsyncMock(
            side_effect=RuntimeError("Test error")
        )

        coordinator = create_coordinator(
            mock_beads, mock_event_sink, mock_epic_verifier
        )

        with pytest.raises(RuntimeError, match="Test error"):
            await coordinator.check_epic_closure("child-1", mock_run_metadata)

        # Should have removed from epics_being_verified even on error
        assert "epic-1" not in coordinator.epics_being_verified


class TestFallbackForMockProviders:
    """Tests for fallback behavior when EpicVerifier is not available."""

    @pytest.mark.asyncio
    async def test_uses_fallback_when_no_epic_verifier(
        self,
        mock_beads: MagicMock,
        mock_event_sink: MagicMock,
        mock_epic_verifier: MagicMock,
        mock_run_metadata: MagicMock,
    ) -> None:
        """Should use close_eligible_epics fallback when no EpicVerifier."""
        coordinator = create_coordinator(
            mock_beads,
            mock_event_sink,
            mock_epic_verifier,
            has_epic_verifier=False,
        )

        await coordinator.check_epic_closure("child-1", mock_run_metadata)

        # Should NOT call verify_and_close_epic
        mock_epic_verifier.verify_and_close_epic.assert_not_called()
        # Should call close_eligible_epics fallback
        mock_beads.close_eligible_epics_async.assert_called_once()
        # Should emit on_epic_closed
        mock_event_sink.on_epic_closed.assert_called_once_with("child-1")

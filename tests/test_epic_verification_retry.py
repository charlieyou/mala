"""Unit tests for epic verification retry loop in orchestrator.

Tests the retry logic in MalaOrchestrator._check_epic_closure which:
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
from src.orchestration.issue_result import IssueResult


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
            human_review_count=0,
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


class TestEpicVerificationRetryLoop:
    """Tests for the epic verification retry loop."""

    @pytest.mark.asyncio
    async def test_passes_on_first_attempt(
        self,
        mock_beads: MagicMock,
        mock_event_sink: MagicMock,
        mock_epic_verifier: MagicMock,
        mock_mala_config: MalaConfig,
    ) -> None:
        """Should close epic without retry when verification passes first time."""
        # Create a minimal orchestrator mock with required attributes
        orchestrator = MagicMock()
        orchestrator._mala_config = mock_mala_config
        orchestrator.beads = mock_beads
        orchestrator.event_sink = mock_event_sink
        orchestrator.epic_verifier = mock_epic_verifier
        orchestrator.epic_override_ids = set()
        orchestrator.verified_epics = set()
        orchestrator.epics_being_verified = set()
        orchestrator.failed_issues = set()
        orchestrator.active_tasks = {}

        # Import and patch the method
        from src.orchestration.orchestrator import MalaOrchestrator

        # Call the method directly
        await MalaOrchestrator._check_epic_closure(orchestrator, "child-1")

        # Should have called verify_and_close_epic once
        mock_epic_verifier.verify_and_close_epic.assert_called_once_with(
            "epic-1", human_override=False
        )
        # Should have marked epic as verified
        assert "epic-1" in orchestrator.verified_epics

    @pytest.mark.asyncio
    async def test_retries_when_remediation_issues_created(
        self,
        mock_beads: MagicMock,
        mock_event_sink: MagicMock,
        mock_epic_verifier: MagicMock,
        mock_mala_config: MalaConfig,
    ) -> None:
        """Should retry verification after executing remediation issues."""
        # First call fails with remediation issues, second call passes
        mock_epic_verifier.verify_and_close_epic = AsyncMock(
            side_effect=[
                EpicVerificationResult(
                    verified_count=1,
                    passed_count=0,
                    failed_count=1,
                    human_review_count=0,
                    verdicts={},
                    remediation_issues_created=["rem-1", "rem-2"],
                ),
                EpicVerificationResult(
                    verified_count=1,
                    passed_count=1,
                    failed_count=0,
                    human_review_count=0,
                    verdicts={},
                    remediation_issues_created=[],
                ),
            ]
        )

        orchestrator = MagicMock()
        orchestrator._mala_config = mock_mala_config
        orchestrator.beads = mock_beads
        orchestrator.event_sink = mock_event_sink
        orchestrator.epic_verifier = mock_epic_verifier
        orchestrator.epic_override_ids = set()
        orchestrator.verified_epics = set()
        orchestrator.epics_being_verified = set()
        orchestrator.failed_issues = set()
        orchestrator.active_tasks = {}

        # Mock _execute_remediation_issues as an async function
        orchestrator._execute_remediation_issues = AsyncMock()

        from src.orchestration.orchestrator import MalaOrchestrator

        await MalaOrchestrator._check_epic_closure(orchestrator, "child-1")

        # Should have called verify_and_close_epic twice
        assert mock_epic_verifier.verify_and_close_epic.call_count == 2
        # Should have logged retry attempt
        mock_event_sink.on_warning.assert_called()
        # Should have marked epic as verified after passing
        assert "epic-1" in orchestrator.verified_epics

    @pytest.mark.asyncio
    async def test_stops_at_max_retries(
        self,
        mock_beads: MagicMock,
        mock_event_sink: MagicMock,
        mock_epic_verifier: MagicMock,
        mock_mala_config: MalaConfig,
    ) -> None:
        """Should stop retrying after max_epic_verification_retries."""
        # All calls fail with remediation issues
        mock_epic_verifier.verify_and_close_epic = AsyncMock(
            return_value=EpicVerificationResult(
                verified_count=1,
                passed_count=0,
                failed_count=1,
                human_review_count=0,
                verdicts={},
                remediation_issues_created=["rem-1"],
            )
        )

        orchestrator = MagicMock()
        orchestrator._mala_config = mock_mala_config
        orchestrator.beads = mock_beads
        orchestrator.event_sink = mock_event_sink
        orchestrator.epic_verifier = mock_epic_verifier
        orchestrator.epic_override_ids = set()
        orchestrator.verified_epics = set()
        orchestrator.epics_being_verified = set()
        orchestrator.failed_issues = set()
        orchestrator.active_tasks = {}

        # Mock _execute_remediation_issues as an async function
        orchestrator._execute_remediation_issues = AsyncMock()

        from src.orchestration.orchestrator import MalaOrchestrator

        await MalaOrchestrator._check_epic_closure(orchestrator, "child-1")

        # Should have called verify exactly max_attempts times (1 + 3 retries = 4)
        assert mock_epic_verifier.verify_and_close_epic.call_count == 4
        # Should have logged max retries warning
        assert any(
            "failed after" in str(call)
            for call in mock_event_sink.on_warning.call_args_list
        )
        # Should still mark as verified (to prevent infinite loops)
        assert "epic-1" in orchestrator.verified_epics

    @pytest.mark.asyncio
    async def test_stops_when_no_remediation_issues(
        self,
        mock_beads: MagicMock,
        mock_event_sink: MagicMock,
        mock_epic_verifier: MagicMock,
        mock_mala_config: MalaConfig,
    ) -> None:
        """Should stop if verification fails but no remediation issues created."""
        # Fail without creating remediation issues
        mock_epic_verifier.verify_and_close_epic = AsyncMock(
            return_value=EpicVerificationResult(
                verified_count=1,
                passed_count=0,
                failed_count=1,
                human_review_count=0,
                verdicts={},
                remediation_issues_created=[],  # No remediation issues
            )
        )

        orchestrator = MagicMock()
        orchestrator._mala_config = mock_mala_config
        orchestrator.beads = mock_beads
        orchestrator.event_sink = mock_event_sink
        orchestrator.epic_verifier = mock_epic_verifier
        orchestrator.epic_override_ids = set()
        orchestrator.verified_epics = set()
        orchestrator.epics_being_verified = set()
        orchestrator.failed_issues = set()
        orchestrator.active_tasks = {}

        from src.orchestration.orchestrator import MalaOrchestrator

        await MalaOrchestrator._check_epic_closure(orchestrator, "child-1")

        # Should only call verify once (no retry since no remediation issues)
        assert mock_epic_verifier.verify_and_close_epic.call_count == 1
        # Should mark as verified
        assert "epic-1" in orchestrator.verified_epics

    @pytest.mark.asyncio
    async def test_skips_already_verified_epic(
        self,
        mock_beads: MagicMock,
        mock_event_sink: MagicMock,
        mock_epic_verifier: MagicMock,
        mock_mala_config: MalaConfig,
    ) -> None:
        """Should skip verification for already verified epics."""
        orchestrator = MagicMock()
        orchestrator._mala_config = mock_mala_config
        orchestrator.beads = mock_beads
        orchestrator.event_sink = mock_event_sink
        orchestrator.epic_verifier = mock_epic_verifier
        orchestrator.epic_override_ids = set()
        orchestrator.verified_epics = {"epic-1"}  # Already verified
        orchestrator.epics_being_verified = set()
        orchestrator.failed_issues = set()
        orchestrator.active_tasks = {}

        from src.orchestration.orchestrator import MalaOrchestrator

        await MalaOrchestrator._check_epic_closure(orchestrator, "child-1")

        # Should not call verify_and_close_epic
        mock_epic_verifier.verify_and_close_epic.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_no_parent_epic(
        self,
        mock_beads: MagicMock,
        mock_event_sink: MagicMock,
        mock_epic_verifier: MagicMock,
        mock_mala_config: MalaConfig,
    ) -> None:
        """Should skip verification when issue has no parent epic."""
        mock_beads.get_parent_epic_async = AsyncMock(return_value=None)

        orchestrator = MagicMock()
        orchestrator._mala_config = mock_mala_config
        orchestrator.beads = mock_beads
        orchestrator.event_sink = mock_event_sink
        orchestrator.epic_verifier = mock_epic_verifier
        orchestrator.epic_override_ids = set()
        orchestrator.verified_epics = set()
        orchestrator.epics_being_verified = set()
        orchestrator.failed_issues = set()
        orchestrator.active_tasks = {}

        from src.orchestration.orchestrator import MalaOrchestrator

        await MalaOrchestrator._check_epic_closure(orchestrator, "child-1")

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
    """Tests for _execute_remediation_issues method."""

    @pytest.mark.asyncio
    async def test_uses_task_returned_by_spawn_agent(
        self,
        mock_beads: MagicMock,
        mock_event_sink: MagicMock,
    ) -> None:
        """Should use task returned by spawn_agent directly, not from active_tasks."""
        orchestrator = MagicMock()
        orchestrator.beads = mock_beads
        orchestrator.event_sink = mock_event_sink
        orchestrator.failed_issues = set()
        # active_tasks intentionally empty to verify we don't rely on it
        orchestrator.active_tasks = {}

        task_was_awaited = False

        async def mock_spawn(issue_id: str) -> asyncio.Task[IssueResult]:
            """Return a task directly without registering to active_tasks."""
            nonlocal task_was_awaited

            async def dummy_result() -> IssueResult:
                nonlocal task_was_awaited
                task_was_awaited = True
                return IssueResult(
                    issue_id=issue_id, agent_id="test", success=True, summary="done"
                )

            return asyncio.create_task(dummy_result())

        orchestrator.spawn_agent = mock_spawn

        from src.orchestration.orchestrator import MalaOrchestrator

        await MalaOrchestrator._execute_remediation_issues(
            orchestrator, ["rem-1", "rem-2"]
        )

        # Verify task was actually awaited
        assert task_was_awaited, "Task returned by spawn_agent should be awaited"


class TestEpicNotEligible:
    """Tests for handling epics that aren't eligible yet (children still open)."""

    @pytest.mark.asyncio
    async def test_does_not_mark_verified_when_not_eligible(
        self,
        mock_beads: MagicMock,
        mock_event_sink: MagicMock,
        mock_epic_verifier: MagicMock,
        mock_mala_config: MalaConfig,
    ) -> None:
        """Should NOT mark epic as verified when it isn't eligible (verified_count=0)."""
        # Return result indicating epic wasn't eligible (children still open)
        mock_epic_verifier.verify_and_close_epic = AsyncMock(
            return_value=EpicVerificationResult(
                verified_count=0,  # Epic not eligible - children still open
                passed_count=0,
                failed_count=0,
                human_review_count=0,
                verdicts={},
                remediation_issues_created=[],
            )
        )

        orchestrator = MagicMock()
        orchestrator._mala_config = mock_mala_config
        orchestrator.beads = mock_beads
        orchestrator.event_sink = mock_event_sink
        orchestrator.epic_verifier = mock_epic_verifier
        orchestrator.epic_override_ids = set()
        orchestrator.verified_epics = set()
        orchestrator.epics_being_verified = set()
        orchestrator.failed_issues = set()
        orchestrator.active_tasks = {}

        from src.orchestration.orchestrator import MalaOrchestrator

        await MalaOrchestrator._check_epic_closure(orchestrator, "child-1")

        # Should NOT mark epic as verified (so it can be re-checked later)
        assert "epic-1" not in orchestrator.verified_epics
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
        mock_mala_config: MalaConfig,
    ) -> None:
        """Should skip verification for epics already being verified."""
        orchestrator = MagicMock()
        orchestrator._mala_config = mock_mala_config
        orchestrator.beads = mock_beads
        orchestrator.event_sink = mock_event_sink
        orchestrator.epic_verifier = mock_epic_verifier
        orchestrator.epic_override_ids = set()
        orchestrator.verified_epics = set()
        orchestrator.epics_being_verified = {"epic-1"}  # Already being verified
        orchestrator.failed_issues = set()
        orchestrator.active_tasks = {}

        from src.orchestration.orchestrator import MalaOrchestrator

        await MalaOrchestrator._check_epic_closure(orchestrator, "child-1")

        # Should NOT call verify_and_close_epic due to re-entry guard
        mock_epic_verifier.verify_and_close_epic.assert_not_called()

    @pytest.mark.asyncio
    async def test_removes_from_being_verified_on_completion(
        self,
        mock_beads: MagicMock,
        mock_event_sink: MagicMock,
        mock_epic_verifier: MagicMock,
        mock_mala_config: MalaConfig,
    ) -> None:
        """Should remove epic from epics_being_verified when done."""
        orchestrator = MagicMock()
        orchestrator._mala_config = mock_mala_config
        orchestrator.beads = mock_beads
        orchestrator.event_sink = mock_event_sink
        orchestrator.epic_verifier = mock_epic_verifier
        orchestrator.epic_override_ids = set()
        orchestrator.verified_epics = set()
        orchestrator.epics_being_verified = set()
        orchestrator.failed_issues = set()
        orchestrator.active_tasks = {}

        from src.orchestration.orchestrator import MalaOrchestrator

        await MalaOrchestrator._check_epic_closure(orchestrator, "child-1")

        # Should have removed from epics_being_verified
        assert "epic-1" not in orchestrator.epics_being_verified

    @pytest.mark.asyncio
    async def test_removes_from_being_verified_on_error(
        self,
        mock_beads: MagicMock,
        mock_event_sink: MagicMock,
        mock_epic_verifier: MagicMock,
        mock_mala_config: MalaConfig,
    ) -> None:
        """Should remove epic from epics_being_verified even on error."""
        # Make verify_and_close_epic raise an exception
        mock_epic_verifier.verify_and_close_epic = AsyncMock(
            side_effect=RuntimeError("Test error")
        )

        orchestrator = MagicMock()
        orchestrator._mala_config = mock_mala_config
        orchestrator.beads = mock_beads
        orchestrator.event_sink = mock_event_sink
        orchestrator.epic_verifier = mock_epic_verifier
        orchestrator.epic_override_ids = set()
        orchestrator.verified_epics = set()
        orchestrator.epics_being_verified = set()
        orchestrator.failed_issues = set()
        orchestrator.active_tasks = {}

        from src.orchestration.orchestrator import MalaOrchestrator

        with pytest.raises(RuntimeError, match="Test error"):
            await MalaOrchestrator._check_epic_closure(orchestrator, "child-1")

        # Should have removed from epics_being_verified even on error
        assert "epic-1" not in orchestrator.epics_being_verified

"""Integration tests for CumulativeReviewRunner skeleton.

This test verifies:
1. RunCoordinator wiring triggers CumulativeReviewRunner via real path
2. CumulativeReviewRunner.run_review raises NotImplementedError (skeleton behavior)
3. CumulativeReviewResult dataclass is properly defined

The test exercises: trigger firing → RunCoordinator → CumulativeReviewRunner
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.asyncio
@pytest.mark.integration
async def test_epic_completion_trigger_invokes_cumulative_review(
    tmp_path: Path,
) -> None:
    """Integration: epic_completion trigger with code_review fires CumulativeReviewRunner.

    This test exercises the real wiring path:
    1. Create RunCoordinator with CumulativeReviewRunner wired
    2. Queue epic_completion trigger with code_review enabled
    3. Fire trigger via run_trigger_validation
    4. Assert CumulativeReviewRunner.run_review was called (raises NotImplementedError)
    """
    from src.domain.validation.config import (
        CodeReviewConfig,
        CommandConfig,
        CommandsConfig,
        EpicCompletionTriggerConfig,
        EpicDepth,
        FailureMode,
        FireOn,
        TriggerType,
        ValidationConfig,
        ValidationTriggersConfig,
    )
    from src.infra.clients.beads_client import BeadsClient
    from src.infra.git_utils import DiffStat
    from src.pipeline.cumulative_review_runner import (
        CumulativeReviewRunner,
        GitUtilsProtocol,
    )
    from src.pipeline.review_runner import ReviewRunner, ReviewRunnerConfig
    from src.pipeline.run_coordinator import RunCoordinator, RunCoordinatorConfig
    from tests.fakes import FakeEnvConfig
    from tests.fakes.command_runner import FakeCommandRunner
    from tests.fakes.lock_manager import FakeLockManager

    # Create minimal dependencies
    test_logger = logging.getLogger("test_cumulative_review")

    # Create fake git_utils that satisfies the protocol
    class FakeGitUtils(GitUtilsProtocol):
        async def get_diff_stat(
            self,
            repo_path: Path,
            from_commit: str,
            to_commit: str = "HEAD",
        ) -> DiffStat:
            return DiffStat(total_lines=100, files_changed=["test.py"])

        async def get_diff_content(
            self,
            repo_path: Path,
            from_commit: str,
            to_commit: str = "HEAD",
        ) -> str:
            return "diff content"

    # ReviewRunner with mock CodeReviewer
    mock_code_reviewer = MagicMock()
    review_runner = ReviewRunner(
        code_reviewer=mock_code_reviewer,
        config=ReviewRunnerConfig(),
    )

    # BeadsClient
    beads_client = BeadsClient(repo_path=tmp_path)

    # Create CumulativeReviewRunner
    cumulative_runner = CumulativeReviewRunner(
        review_runner=review_runner,
        git_utils=FakeGitUtils(),
        beads_client=beads_client,
        logger=test_logger,
    )

    # Create mock run_metadata
    mock_run_metadata = MagicMock()
    mock_run_metadata.run_start_commit = "abc123"
    mock_run_metadata.last_cumulative_review_commits = {}

    # Create code_review config for trigger
    code_review_config = CodeReviewConfig(
        enabled=True,
        reviewer_type="cerberus",
        failure_mode=FailureMode.CONTINUE,
        baseline="since_run_start",
    )

    # Create validation config with epic_completion trigger that has code_review
    validation_config = ValidationConfig(
        commands=CommandsConfig(
            test=CommandConfig(command="echo test"),
        ),
        validation_triggers=ValidationTriggersConfig(
            epic_completion=EpicCompletionTriggerConfig(
                failure_mode=FailureMode.CONTINUE,
                commands=(),  # No commands - just code_review
                epic_depth=EpicDepth.TOP_LEVEL,
                fire_on=FireOn.SUCCESS,
                code_review=code_review_config,
            )
        ),
    )

    # Create command runner
    command_runner = FakeCommandRunner(allow_unregistered=True)

    config = RunCoordinatorConfig(
        repo_path=tmp_path,
        timeout_seconds=60,
        validation_config=validation_config,
    )

    # Create RunCoordinator with CumulativeReviewRunner wired
    coordinator = RunCoordinator(
        config=config,
        gate_checker=MagicMock(),
        command_runner=command_runner,
        env_config=FakeEnvConfig(),
        lock_manager=FakeLockManager(),
        sdk_client_factory=MagicMock(),
        cumulative_review_runner=cumulative_runner,
        run_metadata=mock_run_metadata,
    )

    # Queue epic_completion trigger
    coordinator.queue_trigger_validation(
        TriggerType.EPIC_COMPLETION,
        {"issue_id": "test-123", "epic_id": "epic-1"},
    )

    # Act: Run trigger validation - should raise NotImplementedError from CumulativeReviewRunner
    with pytest.raises(NotImplementedError) as exc_info:
        await coordinator.run_trigger_validation()

    # Assert: Error message indicates CumulativeReviewRunner skeleton
    assert "CumulativeReviewRunner.run_review not yet implemented" in str(
        exc_info.value
    )


@pytest.mark.integration
def test_cumulative_review_result_dataclass_is_valid() -> None:
    """Integration: CumulativeReviewResult dataclass is properly defined.

    Verifies the result type exists and has expected fields:
    - status: Literal["success", "skipped", "failed"]
    - findings: tuple of ReviewFinding
    - new_baseline_commit: str | None
    - skip_reason: str | None
    """
    from src.pipeline.cumulative_review_runner import (
        CumulativeReviewResult,
        ReviewFinding,
    )

    # Create a ReviewFinding
    finding = ReviewFinding(
        file="src/test.py",
        line_start=10,
        line_end=15,
        priority=2,
        title="Test finding",
        body="This is a test finding",
        reviewer="cerberus",
    )

    # Create success result
    success_result = CumulativeReviewResult(
        status="success",
        findings=(finding,),
        new_baseline_commit="abc123",
    )
    assert success_result.status == "success"
    assert len(success_result.findings) == 1
    assert success_result.new_baseline_commit == "abc123"
    assert success_result.skip_reason is None

    # Create skipped result
    skipped_result = CumulativeReviewResult(
        status="skipped",
        findings=(),
        new_baseline_commit=None,
        skip_reason="No changes since last review",
    )
    assert skipped_result.status == "skipped"
    assert skipped_result.findings == ()
    assert skipped_result.new_baseline_commit is None
    assert skipped_result.skip_reason == "No changes since last review"


@pytest.mark.integration
def test_cumulative_review_runner_has_expected_interface() -> None:
    """Integration: CumulativeReviewRunner has expected class attributes and methods.

    Verifies the skeleton class is complete:
    - LARGE_DIFF_THRESHOLD class attribute
    - run_review async method
    - _get_baseline_commit method
    - _generate_diff method
    """
    from src.pipeline.cumulative_review_runner import CumulativeReviewRunner

    # Check class attribute
    assert hasattr(CumulativeReviewRunner, "LARGE_DIFF_THRESHOLD")
    assert CumulativeReviewRunner.LARGE_DIFF_THRESHOLD == 5000

    # Check methods exist (inspection only, not calling)
    assert hasattr(CumulativeReviewRunner, "run_review")
    assert hasattr(CumulativeReviewRunner, "_get_baseline_commit")
    assert hasattr(CumulativeReviewRunner, "_generate_diff")
    assert asyncio.iscoroutinefunction(CumulativeReviewRunner.run_review)

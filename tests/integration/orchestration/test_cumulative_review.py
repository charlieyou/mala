"""Integration tests for CumulativeReviewRunner skeleton.

This test verifies:
1. CumulativeReviewRunner can be instantiated with real dependencies
2. The run_review method raises NotImplementedError (skeleton behavior)
3. CumulativeReviewResult dataclass is properly defined

The skeleton is ready for T007-T008 implementation and T012 integration.
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
async def test_cumulative_review_runner_raises_not_implemented(
    tmp_path: Path,
) -> None:
    """Integration: CumulativeReviewRunner.run_review raises NotImplementedError.

    This test verifies the skeleton is properly wired:
    1. Class can be instantiated with protocol dependencies
    2. run_review raises NotImplementedError (not import/wiring errors)
    3. Error message indicates this is expected skeleton behavior
    """
    from src.domain.validation.config import (
        CodeReviewConfig,
        FailureMode,
        TriggerType,
    )
    from src.infra.clients.beads_client import BeadsClient
    from src.pipeline.cumulative_review_runner import CumulativeReviewRunner
    from src.pipeline.review_runner import ReviewRunner, ReviewRunnerConfig

    # Create minimal dependencies
    logger = logging.getLogger("test_cumulative_review")

    # ReviewRunner requires a CodeReviewer - use a mock for skeleton test
    mock_code_reviewer = MagicMock()
    review_runner = ReviewRunner(
        code_reviewer=mock_code_reviewer,
        config=ReviewRunnerConfig(),
    )

    # BeadsClient requires repo_path
    beads_client = BeadsClient(repo_path=tmp_path)

    # Create CumulativeReviewRunner
    runner = CumulativeReviewRunner(
        review_runner=review_runner,
        beads_client=beads_client,
        logger=logger,
    )

    # Create minimal config
    code_review_config = CodeReviewConfig(
        enabled=True,
        reviewer_type="cerberus",
        failure_mode=FailureMode.CONTINUE,
        baseline="since_run_start",
    )

    # Create mock run_metadata
    mock_run_metadata = MagicMock()
    mock_run_metadata.run_start_commit = "abc123"
    mock_run_metadata.last_cumulative_review_commits = {}

    interrupt_event = asyncio.Event()

    # Act: Call run_review and expect NotImplementedError
    with pytest.raises(NotImplementedError) as exc_info:
        await runner.run_review(
            trigger_type=TriggerType.EPIC_COMPLETION,
            config=code_review_config,
            run_metadata=mock_run_metadata,
            repo_path=tmp_path,
            interrupt_event=interrupt_event,
            epic_id="epic-123",
        )

    # Assert: Error message indicates skeleton behavior
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

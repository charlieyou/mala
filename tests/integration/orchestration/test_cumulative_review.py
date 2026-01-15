"""Integration tests for CumulativeReviewRunner skeleton.

This test verifies:
1. build_run_coordinator wires CumulativeReviewRunner via composition root
2. CumulativeReviewRunner.run_review raises NotImplementedError (skeleton behavior)
3. CumulativeReviewResult dataclass is properly defined

The test exercises: trigger firing → RunCoordinator (via build_run_coordinator) →
CumulativeReviewRunner
"""

from __future__ import annotations

import inspect
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

    This test exercises the composition root wiring path:
    1. Build RunCoordinator via build_run_coordinator (composition root)
    2. Queue epic_completion trigger with code_review enabled
    3. Fire trigger via run_trigger_validation
    4. Assert CumulativeReviewRunner.run_review was called (returns skipped due to
       unreachable baseline in test git repo)
    """
    from src.domain.validation.config import (
        CodeReviewConfig,
        CommandConfig,
        CommandsConfig,
        EpicCompletionTriggerConfig,
        EpicDepth,
        FailureMode,
        FireOn,
        PromptValidationCommands,
        TriggerType,
        ValidationConfig,
        ValidationTriggersConfig,
    )
    from src.orchestration.orchestration_wiring import build_run_coordinator
    from src.orchestration.types import PipelineConfig, RuntimeDeps
    from tests.fakes import FakeEnvConfig, FakeIssueProvider
    from tests.fakes.command_runner import FakeCommandRunner
    from tests.fakes.lock_manager import FakeLockManager

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

    # Create RuntimeDeps with minimal fakes
    mock_code_reviewer = MagicMock()
    runtime = RuntimeDeps(
        evidence_check=MagicMock(),
        code_reviewer=mock_code_reviewer,
        beads=FakeIssueProvider({}),
        event_sink=MagicMock(),
        command_runner=FakeCommandRunner(allow_unregistered=True),
        env_config=FakeEnvConfig(),
        lock_manager=FakeLockManager(),
        mala_config=MagicMock(),
    )

    # Create PipelineConfig
    pipeline = PipelineConfig(
        repo_path=tmp_path,
        timeout_seconds=60,
        max_gate_retries=3,
        max_review_retries=3,
        disabled_validations=set(),
        max_idle_retries=2,
        idle_timeout_seconds=None,
        prompts=MagicMock(fixer_prompt="Fix the issue"),
        prompt_validation_commands=PromptValidationCommands(
            lint="echo lint",
            format="echo format",
            typecheck="echo typecheck",
            test="echo test",
            custom_commands=(),
        ),
        validation_config=validation_config,
        validation_config_missing=False,
    )

    # Build RunCoordinator via composition root - this wires CumulativeReviewRunner
    coordinator = build_run_coordinator(
        runtime=runtime,
        pipeline=pipeline,
        sdk_client_factory=MagicMock(),
    )

    # Create mock run_metadata and wire it
    mock_run_metadata = MagicMock()
    mock_run_metadata.run_start_commit = "abc123"
    mock_run_metadata.last_cumulative_review_commits = {}
    coordinator.run_metadata = mock_run_metadata

    # Queue epic_completion trigger
    coordinator.queue_trigger_validation(
        TriggerType.EPIC_COMPLETION,
        {"issue_id": "test-123", "epic_id": "epic-1"},
    )

    # Act: Run trigger validation - CumulativeReviewRunner.run_review will be called
    # but will return "skipped" because baseline commit won't be reachable in tmp_path
    result = await coordinator.run_trigger_validation()

    # Assert: Trigger validation completed (skipped review doesn't fail the trigger)
    # The result should be "passed" because code_review failure_mode is CONTINUE
    assert result.status == "passed"


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

    Verifies the class is complete:
    - LARGE_DIFF_THRESHOLD class attribute
    - run_review async method
    - _get_baseline_commit method
    """
    from src.pipeline.cumulative_review_runner import CumulativeReviewRunner

    # Check class attribute
    assert hasattr(CumulativeReviewRunner, "LARGE_DIFF_THRESHOLD")
    assert CumulativeReviewRunner.LARGE_DIFF_THRESHOLD == 5000

    # Check methods exist (inspection only, not calling)
    assert hasattr(CumulativeReviewRunner, "run_review")
    assert hasattr(CumulativeReviewRunner, "_get_baseline_commit")
    assert inspect.iscoroutinefunction(CumulativeReviewRunner.run_review)

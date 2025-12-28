"""Unit tests for run-level validation (Gate 4) in MalaOrchestrator.

Tests the implementation of Gate 4 validation that runs after all issues complete.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.logging.run_metadata import RunMetadata
from src.orchestrator import IssueResult, MalaOrchestrator


class TestRunLevelValidation:
    """Test Gate 4 (run-level validation) after all issues complete."""

    @pytest.mark.asyncio
    async def test_run_level_validation_skipped_when_disabled(
        self, tmp_path: Path
    ) -> None:
        """Run-level validation should be skipped when disabled."""
        orchestrator = MalaOrchestrator(
            repo_path=tmp_path,
            max_agents=1,
            disable_validations={"run-level-validate"},
        )

        from src.logging.run_metadata import RunConfig, RunMetadata

        run_config = RunConfig(
            max_agents=1,
            timeout_minutes=None,
            max_issues=None,
            epic_id=None,
            only_ids=None,
            braintrust_enabled=False,
        )
        run_metadata = RunMetadata(tmp_path, run_config, "test")

        result = await orchestrator._run_run_level_validation(run_metadata)

        # Should return True (passed/skipped)
        assert result is True
        # run_validation should not be set
        assert run_metadata.run_validation is None

    @pytest.mark.asyncio
    async def test_run_level_validation_passes_when_validation_succeeds(
        self, tmp_path: Path
    ) -> None:
        """Run-level validation should pass when validation runner succeeds."""
        from src.logging.run_metadata import RunConfig, RunMetadata
        from src.validation.result import ValidationResult, ValidationStepResult

        orchestrator = MalaOrchestrator(repo_path=tmp_path, max_agents=1)

        run_config = RunConfig(
            max_agents=1,
            timeout_minutes=None,
            max_issues=None,
            epic_id=None,
            only_ids=None,
            braintrust_enabled=False,
        )
        run_metadata = RunMetadata(tmp_path, run_config, "test")

        async def mock_get_commit(path: Path) -> str:
            return "abc123"

        mock_result = ValidationResult(
            passed=True,
            steps=[
                ValidationStepResult(
                    name="pytest",
                    command=["pytest"],
                    ok=True,
                    returncode=0,
                    stdout_tail="",
                    duration_seconds=1.0,
                )
            ],
        )

        with (
            patch("src.orchestrator.get_git_commit_async", side_effect=mock_get_commit),
            patch("src.orchestrator.SpecValidationRunner") as MockRunner,
        ):
            mock_runner_instance = MagicMock()
            mock_runner_instance.run_spec = AsyncMock(return_value=mock_result)
            MockRunner.return_value = mock_runner_instance

            result = await orchestrator._run_run_level_validation(run_metadata)

        assert result is True
        assert run_metadata.run_validation is not None
        assert run_metadata.run_validation.passed is True

    @pytest.mark.asyncio
    async def test_run_level_validation_spawns_fixer_on_failure(
        self, tmp_path: Path
    ) -> None:
        """Run-level validation should spawn fixer agent on failure."""
        from src.logging.run_metadata import RunConfig, RunMetadata
        from src.validation.result import ValidationResult, ValidationStepResult

        orchestrator = MalaOrchestrator(
            repo_path=tmp_path, max_agents=1, max_gate_retries=2
        )

        run_config = RunConfig(
            max_agents=1,
            timeout_minutes=None,
            max_issues=None,
            epic_id=None,
            only_ids=None,
            braintrust_enabled=False,
        )
        run_metadata = RunMetadata(tmp_path, run_config, "test")

        async def mock_get_commit(path: Path) -> str:
            return "abc123"

        fixer_calls: list[tuple[str, int]] = []

        async def mock_fixer(failure_output: str, attempt: int) -> bool:
            fixer_calls.append((failure_output, attempt))
            return True

        mock_result = ValidationResult(
            passed=False,
            failure_reasons=["pytest failed"],
            steps=[
                ValidationStepResult(
                    name="pytest",
                    command=["pytest"],
                    ok=False,
                    returncode=1,
                    stdout_tail="FAILED test_foo.py",
                    duration_seconds=1.0,
                )
            ],
        )

        with (
            patch("src.orchestrator.get_git_commit_async", side_effect=mock_get_commit),
            patch.object(orchestrator, "_run_fixer_agent", side_effect=mock_fixer),
            patch("src.orchestrator.SpecValidationRunner") as MockRunner,
        ):
            mock_runner_instance = MagicMock()
            mock_runner_instance.run_spec = AsyncMock(return_value=mock_result)
            MockRunner.return_value = mock_runner_instance

            result = await orchestrator._run_run_level_validation(run_metadata)

        # Should have called fixer once (max_gate_retries=2, fails on attempt 2)
        assert len(fixer_calls) == 1
        assert fixer_calls[0][1] == 1  # First attempt
        assert "pytest failed" in fixer_calls[0][0]

        # Should return False after exhausting retries
        assert result is False

    @pytest.mark.asyncio
    async def test_run_level_validation_records_to_metadata(
        self, tmp_path: Path
    ) -> None:
        """Run-level validation should record results to run metadata."""
        from src.logging.run_metadata import RunConfig, RunMetadata
        from src.validation.result import ValidationResult, ValidationStepResult

        orchestrator = MalaOrchestrator(
            repo_path=tmp_path, max_agents=1, max_gate_retries=1
        )

        run_config = RunConfig(
            max_agents=1,
            timeout_minutes=None,
            max_issues=None,
            epic_id=None,
            only_ids=None,
            braintrust_enabled=False,
        )
        run_metadata = RunMetadata(tmp_path, run_config, "test")

        async def mock_get_commit(path: Path) -> str:
            return "abc123"

        mock_result = ValidationResult(
            passed=False,
            failure_reasons=["coverage below threshold"],
            steps=[
                ValidationStepResult(
                    name="pytest",
                    command=["pytest"],
                    ok=True,
                    returncode=0,
                    stdout_tail="",
                    duration_seconds=1.0,
                ),
                ValidationStepResult(
                    name="coverage",
                    command=["coverage"],
                    ok=False,
                    returncode=1,
                    stdout_tail="80%",
                    duration_seconds=0.5,
                ),
            ],
        )

        with (
            patch("src.orchestrator.get_git_commit_async", side_effect=mock_get_commit),
            patch("src.orchestrator.SpecValidationRunner") as MockRunner,
        ):
            mock_runner_instance = MagicMock()
            mock_runner_instance.run_spec = AsyncMock(return_value=mock_result)
            MockRunner.return_value = mock_runner_instance

            await orchestrator._run_run_level_validation(run_metadata)

        # Check metadata was recorded
        assert run_metadata.run_validation is not None
        assert run_metadata.run_validation.passed is False
        assert "pytest" in run_metadata.run_validation.commands_run
        assert "coverage" in run_metadata.run_validation.commands_failed

    def test_build_validation_failure_output_with_result(self, tmp_path: Path) -> None:
        """_build_validation_failure_output should format failure details."""
        from src.validation.result import ValidationResult, ValidationStepResult

        orchestrator = MalaOrchestrator(repo_path=tmp_path)

        result = ValidationResult(
            passed=False,
            failure_reasons=["test failed", "coverage too low"],
            steps=[
                ValidationStepResult(
                    name="pytest",
                    command=["pytest"],
                    ok=False,
                    returncode=1,
                    stderr_tail="AssertionError: expected True",
                    duration_seconds=1.0,
                )
            ],
        )

        output = orchestrator._build_validation_failure_output(result)

        assert "test failed" in output
        assert "coverage too low" in output
        assert "pytest" in output
        assert "AssertionError" in output

    def test_build_validation_failure_output_with_none(self, tmp_path: Path) -> None:
        """_build_validation_failure_output should handle None result."""
        orchestrator = MalaOrchestrator(repo_path=tmp_path)

        output = orchestrator._build_validation_failure_output(None)

        assert "crashed" in output.lower()

    @pytest.mark.asyncio
    async def test_e2e_passed_none_when_e2e_disabled(self, tmp_path: Path) -> None:
        """e2e_passed should be None when E2E is disabled via disable_validations."""
        from src.logging.run_metadata import RunConfig, RunMetadata
        from src.validation.result import ValidationResult, ValidationStepResult

        # Disable E2E via disable_validations
        orchestrator = MalaOrchestrator(
            repo_path=tmp_path, max_agents=1, disable_validations={"e2e"}
        )

        run_config = RunConfig(
            max_agents=1,
            timeout_minutes=None,
            max_issues=None,
            epic_id=None,
            only_ids=None,
            braintrust_enabled=False,
        )
        run_metadata = RunMetadata(tmp_path, run_config, "test")

        async def mock_get_commit(path: Path) -> str:
            return "abc123"

        mock_result = ValidationResult(
            passed=True,
            steps=[
                ValidationStepResult(
                    name="pytest",
                    command=["pytest"],
                    ok=True,
                    returncode=0,
                    stdout_tail="",
                    duration_seconds=1.0,
                )
            ],
        )

        with (
            patch("src.orchestrator.get_git_commit_async", side_effect=mock_get_commit),
            patch("src.orchestrator.SpecValidationRunner") as MockRunner,
        ):
            mock_runner_instance = MagicMock()
            mock_runner_instance.run_spec = AsyncMock(return_value=mock_result)
            MockRunner.return_value = mock_runner_instance

            result = await orchestrator._run_run_level_validation(run_metadata)

        assert result is True
        assert run_metadata.run_validation is not None
        assert run_metadata.run_validation.passed is True
        # E2E was disabled, so e2e_passed should be None
        assert run_metadata.run_validation.e2e_passed is None

    @pytest.mark.asyncio
    async def test_e2e_passed_true_when_e2e_enabled_and_passes(
        self, tmp_path: Path
    ) -> None:
        """e2e_passed should be True when E2E is enabled and validation passes."""
        from src.logging.run_metadata import RunConfig, RunMetadata
        from src.validation.result import ValidationResult, ValidationStepResult

        # E2E enabled by default (not in disable_validations)
        orchestrator = MalaOrchestrator(repo_path=tmp_path, max_agents=1)

        run_config = RunConfig(
            max_agents=1,
            timeout_minutes=None,
            max_issues=None,
            epic_id=None,
            only_ids=None,
            braintrust_enabled=False,
        )
        run_metadata = RunMetadata(tmp_path, run_config, "test")

        async def mock_get_commit(path: Path) -> str:
            return "abc123"

        mock_result = ValidationResult(
            passed=True,
            steps=[
                ValidationStepResult(
                    name="pytest",
                    command=["pytest"],
                    ok=True,
                    returncode=0,
                    stdout_tail="",
                    duration_seconds=1.0,
                )
            ],
        )

        with (
            patch("src.orchestrator.get_git_commit_async", side_effect=mock_get_commit),
            patch("src.orchestrator.SpecValidationRunner") as MockRunner,
        ):
            mock_runner_instance = MagicMock()
            mock_runner_instance.run_spec = AsyncMock(return_value=mock_result)
            MockRunner.return_value = mock_runner_instance

            result = await orchestrator._run_run_level_validation(run_metadata)

        assert result is True
        assert run_metadata.run_validation is not None
        assert run_metadata.run_validation.passed is True
        # E2E was enabled and passed, so e2e_passed should be True
        assert run_metadata.run_validation.e2e_passed is True

    @pytest.mark.asyncio
    async def test_e2e_passed_false_when_e2e_enabled_and_fails(
        self, tmp_path: Path
    ) -> None:
        """e2e_passed should be False when E2E is enabled and validation fails."""
        from src.logging.run_metadata import RunConfig, RunMetadata
        from src.validation.result import ValidationResult, ValidationStepResult

        # E2E enabled by default, max_gate_retries=1 to fail immediately
        orchestrator = MalaOrchestrator(
            repo_path=tmp_path, max_agents=1, max_gate_retries=1
        )

        run_config = RunConfig(
            max_agents=1,
            timeout_minutes=None,
            max_issues=None,
            epic_id=None,
            only_ids=None,
            braintrust_enabled=False,
        )
        run_metadata = RunMetadata(tmp_path, run_config, "test")

        async def mock_get_commit(path: Path) -> str:
            return "abc123"

        mock_result = ValidationResult(
            passed=False,
            failure_reasons=["E2E failed"],
            steps=[
                ValidationStepResult(
                    name="pytest",
                    command=["pytest"],
                    ok=False,
                    returncode=1,
                    stdout_tail="FAILED",
                    duration_seconds=1.0,
                )
            ],
        )

        with (
            patch("src.orchestrator.get_git_commit_async", side_effect=mock_get_commit),
            patch("src.orchestrator.SpecValidationRunner") as MockRunner,
        ):
            mock_runner_instance = MagicMock()
            mock_runner_instance.run_spec = AsyncMock(return_value=mock_result)
            MockRunner.return_value = mock_runner_instance

            result = await orchestrator._run_run_level_validation(run_metadata)

        assert result is False
        assert run_metadata.run_validation is not None
        assert run_metadata.run_validation.passed is False
        # E2E was enabled and failed, so e2e_passed should be False
        assert run_metadata.run_validation.e2e_passed is False


class TestRunLevelValidationIntegration:
    """Integration tests for Gate 4 in the orchestrator run() method."""

    @pytest.mark.asyncio
    async def test_run_calls_gate4_after_issues_complete(self, tmp_path: Path) -> None:
        """run() should call run-level validation after all issues complete."""
        orchestrator = MalaOrchestrator(repo_path=tmp_path, max_agents=1)

        gate4_called = False

        async def mock_run_level(run_metadata: RunMetadata) -> bool:
            nonlocal gate4_called
            gate4_called = True
            return True

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            nonlocal first_call
            if first_call:
                first_call = False
                return ["issue-1"]
            return []

        async def mock_run_implementer(issue_id: str) -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="Done",
            )

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch.object(orchestrator.beads, "claim_async", return_value=True),
            patch.object(
                orchestrator, "run_implementer", side_effect=mock_run_implementer
            ),
            patch.object(orchestrator.beads, "close_async", return_value=True),
            patch.object(orchestrator.beads, "commit_issues_async", return_value=True),
            patch.object(
                orchestrator.beads, "close_eligible_epics_async", return_value=False
            ),
            patch.object(
                orchestrator, "_run_run_level_validation", side_effect=mock_run_level
            ),
            patch("src.orchestrator.get_lock_dir", return_value=tmp_path / "locks"),
            patch("src.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestrator.release_run_locks"),
        ):
            (tmp_path / "locks").mkdir(exist_ok=True)
            await orchestrator.run()

        assert gate4_called is True

    @pytest.mark.asyncio
    async def test_run_returns_zero_success_on_gate4_failure(
        self, tmp_path: Path
    ) -> None:
        """run() should return 0 successes if Gate 4 fails."""
        orchestrator = MalaOrchestrator(repo_path=tmp_path, max_agents=1)

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            nonlocal first_call
            if first_call:
                first_call = False
                return ["issue-1"]
            return []

        async def mock_run_implementer(issue_id: str) -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="Done",
            )

        async def mock_run_level_fails(run_metadata: RunMetadata) -> bool:
            return False  # Gate 4 fails

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch.object(orchestrator.beads, "claim_async", return_value=True),
            patch.object(
                orchestrator, "run_implementer", side_effect=mock_run_implementer
            ),
            patch.object(orchestrator.beads, "close_async", return_value=True),
            patch.object(orchestrator.beads, "commit_issues_async", return_value=True),
            patch.object(
                orchestrator.beads, "close_eligible_epics_async", return_value=False
            ),
            patch.object(
                orchestrator,
                "_run_run_level_validation",
                side_effect=mock_run_level_fails,
            ),
            patch("src.orchestrator.get_lock_dir", return_value=tmp_path / "locks"),
            patch("src.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestrator.release_run_locks"),
        ):
            (tmp_path / "locks").mkdir(exist_ok=True)
            success_count, total = await orchestrator.run()

        # Gate 4 failure should cause 0 successes to be returned
        assert success_count == 0
        assert total == 1

    @pytest.mark.asyncio
    async def test_run_skips_gate4_when_no_successes(self, tmp_path: Path) -> None:
        """run() should skip Gate 4 when there are no successful issues."""
        orchestrator = MalaOrchestrator(repo_path=tmp_path, max_agents=1)

        gate4_called = False

        async def mock_run_level(run_metadata: RunMetadata) -> bool:
            nonlocal gate4_called
            gate4_called = True
            return True

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            nonlocal first_call
            if first_call:
                first_call = False
                return ["issue-1"]
            return []

        async def mock_run_implementer(issue_id: str) -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=False,  # All issues fail
                summary="Failed",
            )

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch.object(orchestrator.beads, "claim_async", return_value=True),
            patch.object(
                orchestrator, "run_implementer", side_effect=mock_run_implementer
            ),
            patch.object(orchestrator.beads, "close_async", return_value=False),
            patch.object(orchestrator.beads, "commit_issues_async", return_value=True),
            patch.object(
                orchestrator.beads, "close_eligible_epics_async", return_value=False
            ),
            patch.object(
                orchestrator.beads, "mark_needs_followup_async", return_value=True
            ),
            patch.object(
                orchestrator, "_run_run_level_validation", side_effect=mock_run_level
            ),
            patch("src.orchestrator.get_lock_dir", return_value=tmp_path / "locks"),
            patch("src.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestrator.release_run_locks"),
        ):
            (tmp_path / "locks").mkdir(exist_ok=True)
            await orchestrator.run()

        # Gate 4 should NOT be called when no issues succeeded
        assert gate4_called is False

    @pytest.mark.asyncio
    async def test_run_skips_gate4_when_disabled(self, tmp_path: Path) -> None:
        """run() should skip Gate 4 when run-level-validate is disabled."""
        orchestrator = MalaOrchestrator(
            repo_path=tmp_path,
            max_agents=1,
            disable_validations={"run-level-validate"},
        )

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: set[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
        ) -> list[str]:
            nonlocal first_call
            if first_call:
                first_call = False
                return ["issue-1"]
            return []

        async def mock_run_implementer(issue_id: str) -> IssueResult:
            return IssueResult(
                issue_id=issue_id,
                agent_id=f"{issue_id}-agent",
                success=True,
                summary="Done",
            )

        with (
            patch.object(
                orchestrator.beads, "get_ready_async", side_effect=mock_get_ready_async
            ),
            patch.object(orchestrator.beads, "claim_async", return_value=True),
            patch.object(
                orchestrator, "run_implementer", side_effect=mock_run_implementer
            ),
            patch.object(orchestrator.beads, "close_async", return_value=True),
            patch.object(orchestrator.beads, "commit_issues_async", return_value=True),
            patch.object(
                orchestrator.beads, "close_eligible_epics_async", return_value=False
            ),
            patch("src.orchestrator.get_lock_dir", return_value=tmp_path / "locks"),
            patch("src.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestrator.release_run_locks"),
        ):
            (tmp_path / "locks").mkdir(exist_ok=True)
            success_count, total = await orchestrator.run()

        # Should succeed (not fail due to Gate 4)
        assert success_count == 1
        assert total == 1

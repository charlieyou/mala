"""Unit tests for global validation (global validation) in RunCoordinator.

Tests the implementation of global validation validation that runs after all issues complete.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

from tests.fakes import FakeEnvConfig

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.orchestration.orchestrator import MalaOrchestrator

from src.core.models import OrderPreference
from src.infra.io.log_output.run_metadata import RunConfig, RunMetadata
from src.orchestration.orchestrator import IssueResult
from src.pipeline.run_coordinator import (
    RunCoordinator,
    RunCoordinatorConfig,
    GlobalValidationInput,
    GlobalValidationOutput,
)
from tests.fakes.command_runner import FakeCommandRunner
from tests.fakes.lock_manager import FakeLockManager


@pytest.fixture
def fake_command_runner() -> FakeCommandRunner:
    """Create a FakeCommandRunner that allows unregistered commands."""
    return FakeCommandRunner(allow_unregistered=True)


@pytest.fixture
def mock_env_config() -> FakeEnvConfig:
    """Create a fake EnvConfigPort."""
    return FakeEnvConfig()


@pytest.fixture
def fake_lock_manager() -> FakeLockManager:
    """Create a FakeLockManager for testing."""
    return FakeLockManager()


@pytest.fixture
def mock_sdk_client_factory() -> MagicMock:
    """Create a mock SDKClientFactoryProtocol."""
    return MagicMock()


class TestGlobalValidation:
    """Test global validation (global validation) in RunCoordinator."""

    @pytest.mark.asyncio
    async def test_global_validation_skipped_when_disabled(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """Global validation should be skipped when disabled."""
        # Create a mock gate_checker
        mock_gate_checker = MagicMock()

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            disable_validations={"global-validate"},
        )
        coordinator = RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
        )

        run_config = RunConfig(
            max_agents=1,
            timeout_minutes=None,
            max_issues=None,
            epic_id=None,
            only_ids=None,
        )
        run_metadata = RunMetadata(tmp_path, run_config, "test")

        input_data = GlobalValidationInput(run_metadata=run_metadata)
        result = await coordinator.run_validation(input_data)

        # Should return passed=True (skipped)
        assert result.passed is True
        # run_validation should not be set
        assert run_metadata.run_validation is None

    @pytest.mark.asyncio
    async def test_global_validation_passes_when_validation_succeeds(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """Global validation should pass when validation runner succeeds."""
        from src.domain.validation.result import ValidationResult, ValidationStepResult

        mock_gate_checker = MagicMock()

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
        )
        coordinator = RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
        )

        run_config = RunConfig(
            max_agents=1,
            timeout_minutes=None,
            max_issues=None,
            epic_id=None,
            only_ids=None,
        )
        run_metadata = RunMetadata(tmp_path, run_config, "test")

        async def mock_get_commit(path: Path) -> str:
            return "abc123"

        mock_result = ValidationResult(
            passed=True,
            steps=[
                ValidationStepResult(
                    name="pytest",
                    command="pytest",
                    ok=True,
                    returncode=0,
                    stdout_tail="",
                    duration_seconds=1.0,
                )
            ],
        )

        with (
            patch(
                "src.infra.git_utils.get_git_commit_async",
                side_effect=mock_get_commit,
            ),
            patch("src.pipeline.run_coordinator.SpecValidationRunner") as MockRunner,
        ):
            mock_runner_instance = MagicMock()
            mock_runner_instance.run_spec = AsyncMock(return_value=mock_result)
            MockRunner.return_value = mock_runner_instance

            input_data = GlobalValidationInput(run_metadata=run_metadata)
            result = await coordinator.run_validation(input_data)

        assert result.passed is True
        assert run_metadata.run_validation is not None
        assert run_metadata.run_validation.passed is True

    @pytest.mark.asyncio
    async def test_global_validation_does_not_reload_config_when_missing(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """Global validation should not re-read config after startup."""
        from src.domain.validation.result import ValidationResult

        mock_gate_checker = MagicMock()

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            validation_config_missing=True,
        )
        coordinator = RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
        )

        run_config = RunConfig(
            max_agents=1,
            timeout_minutes=None,
            max_issues=None,
            epic_id=None,
            only_ids=None,
        )
        run_metadata = RunMetadata(tmp_path, run_config, "test")

        async def mock_get_commit(path: Path) -> str:
            return "abc123"

        mock_result = ValidationResult(
            passed=True,
            steps=[],
        )

        with (
            patch(
                "src.infra.git_utils.get_git_commit_async",
                side_effect=mock_get_commit,
            ),
            patch(
                "src.domain.validation.config_loader.load_config"
            ) as mock_load_config,
            patch("src.pipeline.run_coordinator.SpecValidationRunner") as MockRunner,
        ):
            mock_runner_instance = MagicMock()
            mock_runner_instance.run_spec = AsyncMock(return_value=mock_result)
            MockRunner.return_value = mock_runner_instance

            input_data = GlobalValidationInput(run_metadata=run_metadata)
            result = await coordinator.run_validation(input_data)

        assert result.passed is True
        assert mock_load_config.call_count == 0

    @pytest.mark.asyncio
    async def test_global_validation_spawns_fixer_on_failure(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """Global validation should spawn fixer agent on failure."""
        from src.domain.validation.result import ValidationResult, ValidationStepResult

        mock_gate_checker = MagicMock()

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            max_gate_retries=2,
        )
        coordinator = RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
        )

        run_config = RunConfig(
            max_agents=1,
            timeout_minutes=None,
            max_issues=None,
            epic_id=None,
            only_ids=None,
        )
        run_metadata = RunMetadata(tmp_path, run_config, "test")

        async def mock_get_commit(path: Path) -> str:
            return "abc123"

        fixer_calls: list[tuple[str, int]] = []

        async def mock_fixer(
            failure_output: str, attempt: int, spec: object = None
        ) -> bool:
            fixer_calls.append((failure_output, attempt))
            return True

        mock_result = ValidationResult(
            passed=False,
            failure_reasons=["pytest failed"],
            steps=[
                ValidationStepResult(
                    name="pytest",
                    command="pytest",
                    ok=False,
                    returncode=1,
                    stdout_tail="FAILED test_foo.py",
                    duration_seconds=1.0,
                )
            ],
        )

        with (
            patch(
                "src.infra.git_utils.get_git_commit_async",
                side_effect=mock_get_commit,
            ),
            patch.object(RunCoordinator, "_run_fixer_agent", side_effect=mock_fixer),
            patch("src.pipeline.run_coordinator.SpecValidationRunner") as MockRunner,
        ):
            mock_runner_instance = MagicMock()
            mock_runner_instance.run_spec = AsyncMock(return_value=mock_result)
            MockRunner.return_value = mock_runner_instance

            input_data = GlobalValidationInput(run_metadata=run_metadata)
            result = await coordinator.run_validation(input_data)

        # Should have called fixer once (max_gate_retries=2, fails on attempt 2)
        assert len(fixer_calls) == 1
        assert fixer_calls[0][1] == 1  # First attempt
        assert "pytest failed" in fixer_calls[0][0]

        # Should return False after exhausting retries
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_global_validation_skips_fixer_on_sigint(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """Global validation should skip fixer when interrupted by SIGINT."""
        import signal

        from src.domain.validation.result import ValidationStepResult
        from src.domain.validation.spec_executor import ValidationInterrupted

        mock_gate_checker = MagicMock()

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            max_gate_retries=2,
        )
        coordinator = RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
        )

        run_config = RunConfig(
            max_agents=1,
            timeout_minutes=None,
            max_issues=None,
            epic_id=None,
            only_ids=None,
        )
        run_metadata = RunMetadata(tmp_path, run_config, "test")

        async def mock_get_commit(path: Path) -> str:
            return "abc123"

        fixer_called = False

        async def mock_fixer(
            failure_output: str, attempt: int, spec: object = None
        ) -> bool:
            nonlocal fixer_called
            fixer_called = True
            return True

        interrupted_step = ValidationStepResult(
            name="pytest",
            command="pytest",
            ok=False,
            returncode=-signal.SIGINT,
            stdout_tail="",
            duration_seconds=1.0,
        )

        interrupt_event = asyncio.Event()

        async def interrupted_run_spec(*_args: object, **_kwargs: object) -> None:
            interrupt_event.set()
            raise ValidationInterrupted(interrupted_step, [interrupted_step])

        with (
            patch(
                "src.infra.git_utils.get_git_commit_async",
                side_effect=mock_get_commit,
            ),
            patch.object(RunCoordinator, "_run_fixer_agent", side_effect=mock_fixer),
            patch("src.pipeline.run_coordinator.SpecValidationRunner") as MockRunner,
        ):
            mock_runner_instance = MagicMock()
            mock_runner_instance.run_spec = AsyncMock(side_effect=interrupted_run_spec)
            MockRunner.return_value = mock_runner_instance

            input_data = GlobalValidationInput(run_metadata=run_metadata)
            result = await coordinator.run_validation(
                input_data, interrupt_event=interrupt_event
            )

        assert result.passed is True
        assert fixer_called is False
        assert run_metadata.run_validation is None

    @pytest.mark.asyncio
    async def test_global_validation_records_to_metadata(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """Global validation should record results to run metadata."""
        from src.domain.validation.result import ValidationResult, ValidationStepResult

        mock_gate_checker = MagicMock()

        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            max_gate_retries=1,
        )
        coordinator = RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
        )

        run_config = RunConfig(
            max_agents=1,
            timeout_minutes=None,
            max_issues=None,
            epic_id=None,
            only_ids=None,
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
                    command="pytest",
                    ok=True,
                    returncode=0,
                    stdout_tail="",
                    duration_seconds=1.0,
                ),
                ValidationStepResult(
                    name="coverage",
                    command="coverage",
                    ok=False,
                    returncode=1,
                    stdout_tail="80%",
                    duration_seconds=0.5,
                ),
            ],
        )

        with (
            patch(
                "src.infra.git_utils.get_git_commit_async",
                side_effect=mock_get_commit,
            ),
            patch("src.pipeline.run_coordinator.SpecValidationRunner") as MockRunner,
        ):
            mock_runner_instance = MagicMock()
            mock_runner_instance.run_spec = AsyncMock(return_value=mock_result)
            MockRunner.return_value = mock_runner_instance

            input_data = GlobalValidationInput(run_metadata=run_metadata)
            await coordinator.run_validation(input_data)

        # Check metadata was recorded
        assert run_metadata.run_validation is not None
        assert run_metadata.run_validation.passed is False
        assert "pytest" in run_metadata.run_validation.commands_run
        assert "coverage" in run_metadata.run_validation.commands_failed

    def test_build_validation_failure_output_with_result(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """_build_validation_failure_output should format failure details."""
        from src.domain.validation.result import ValidationResult, ValidationStepResult

        mock_gate_checker = MagicMock()
        config = RunCoordinatorConfig(repo_path=tmp_path, timeout_seconds=60)
        coordinator = RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
        )

        result = ValidationResult(
            passed=False,
            failure_reasons=["test failed", "coverage too low"],
            steps=[
                ValidationStepResult(
                    name="pytest",
                    command="pytest",
                    ok=False,
                    returncode=1,
                    stderr_tail="AssertionError: expected True",
                    duration_seconds=1.0,
                )
            ],
        )

        output = coordinator._build_validation_failure_output(result)

        assert "test failed" in output
        assert "coverage too low" in output
        assert "pytest" in output
        assert "AssertionError" in output

    def test_build_validation_failure_output_with_none(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """_build_validation_failure_output should handle None result."""
        mock_gate_checker = MagicMock()
        config = RunCoordinatorConfig(repo_path=tmp_path, timeout_seconds=60)
        coordinator = RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
        )

        output = coordinator._build_validation_failure_output(None)

        assert "crashed" in output.lower()

    @pytest.mark.asyncio
    async def test_e2e_passed_none_when_e2e_disabled(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """e2e_passed should be None when E2E is disabled via disable_validations."""
        from src.domain.validation.result import ValidationResult, ValidationStepResult

        mock_gate_checker = MagicMock()

        # Disable E2E via disable_validations
        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            disable_validations={"e2e"},
        )
        coordinator = RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
        )

        run_config = RunConfig(
            max_agents=1,
            timeout_minutes=None,
            max_issues=None,
            epic_id=None,
            only_ids=None,
        )
        run_metadata = RunMetadata(tmp_path, run_config, "test")

        async def mock_get_commit(path: Path) -> str:
            return "abc123"

        # E2E disabled, so e2e_result is None
        mock_result = ValidationResult(
            passed=True,
            steps=[
                ValidationStepResult(
                    name="pytest",
                    command="pytest",
                    ok=True,
                    returncode=0,
                    stdout_tail="",
                    duration_seconds=1.0,
                )
            ],
            e2e_result=None,  # E2E was not executed
        )

        with (
            patch(
                "src.infra.git_utils.get_git_commit_async",
                side_effect=mock_get_commit,
            ),
            patch("src.pipeline.run_coordinator.SpecValidationRunner") as MockRunner,
        ):
            mock_runner_instance = MagicMock()
            mock_runner_instance.run_spec = AsyncMock(return_value=mock_result)
            MockRunner.return_value = mock_runner_instance

            input_data = GlobalValidationInput(run_metadata=run_metadata)
            result = await coordinator.run_validation(input_data)

        assert result.passed is True
        assert run_metadata.run_validation is not None
        assert run_metadata.run_validation.passed is True
        # E2E was disabled (e2e_result=None), so e2e_passed should be None
        assert run_metadata.run_validation.e2e_passed is None

    @pytest.mark.asyncio
    async def test_e2e_passed_none_when_e2e_skipped(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """e2e_passed should be None when E2E was skipped (status=SKIPPED)."""
        from src.domain.validation.e2e import E2EResult, E2EStatus
        from src.domain.validation.result import ValidationResult, ValidationStepResult

        mock_gate_checker = MagicMock()
        config = RunCoordinatorConfig(repo_path=tmp_path, timeout_seconds=60)
        coordinator = RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
        )

        run_config = RunConfig(
            max_agents=1,
            timeout_minutes=None,
            max_issues=None,
            epic_id=None,
            only_ids=None,
        )
        run_metadata = RunMetadata(tmp_path, run_config, "test")

        async def mock_get_commit(path: Path) -> str:
            return "abc123"

        # E2E was skipped (e.g., missing prerequisites)
        mock_result = ValidationResult(
            passed=True,
            steps=[
                ValidationStepResult(
                    name="pytest",
                    command="pytest",
                    ok=True,
                    returncode=0,
                    stdout_tail="",
                    duration_seconds=1.0,
                )
            ],
            e2e_result=E2EResult(
                passed=False,  # Not passed, but skipped
                status=E2EStatus.SKIPPED,
                failure_reason="Prerequisites not met",
            ),
        )

        with (
            patch(
                "src.infra.git_utils.get_git_commit_async",
                side_effect=mock_get_commit,
            ),
            patch("src.pipeline.run_coordinator.SpecValidationRunner") as MockRunner,
        ):
            mock_runner_instance = MagicMock()
            mock_runner_instance.run_spec = AsyncMock(return_value=mock_result)
            MockRunner.return_value = mock_runner_instance

            input_data = GlobalValidationInput(run_metadata=run_metadata)
            result = await coordinator.run_validation(input_data)

        assert result.passed is True
        assert run_metadata.run_validation is not None
        assert run_metadata.run_validation.passed is True
        # E2E was skipped (status=SKIPPED), so e2e_passed should be None
        assert run_metadata.run_validation.e2e_passed is None

    @pytest.mark.asyncio
    async def test_e2e_passed_true_when_e2e_enabled_and_passes(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """e2e_passed should be True when E2E is enabled and actually passes."""
        from src.domain.validation.e2e import E2EResult, E2EStatus
        from src.domain.validation.result import ValidationResult, ValidationStepResult

        mock_gate_checker = MagicMock()
        config = RunCoordinatorConfig(repo_path=tmp_path, timeout_seconds=60)
        coordinator = RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
        )

        run_config = RunConfig(
            max_agents=1,
            timeout_minutes=None,
            max_issues=None,
            epic_id=None,
            only_ids=None,
        )
        run_metadata = RunMetadata(tmp_path, run_config, "test")

        async def mock_get_commit(path: Path) -> str:
            return "abc123"

        # E2E was executed and passed
        mock_result = ValidationResult(
            passed=True,
            steps=[
                ValidationStepResult(
                    name="pytest",
                    command="pytest",
                    ok=True,
                    returncode=0,
                    stdout_tail="",
                    duration_seconds=1.0,
                )
            ],
            e2e_result=E2EResult(
                passed=True,
                status=E2EStatus.PASSED,
            ),
        )

        with (
            patch(
                "src.infra.git_utils.get_git_commit_async",
                side_effect=mock_get_commit,
            ),
            patch("src.pipeline.run_coordinator.SpecValidationRunner") as MockRunner,
        ):
            mock_runner_instance = MagicMock()
            mock_runner_instance.run_spec = AsyncMock(return_value=mock_result)
            MockRunner.return_value = mock_runner_instance

            input_data = GlobalValidationInput(run_metadata=run_metadata)
            result = await coordinator.run_validation(input_data)

        assert result.passed is True
        assert run_metadata.run_validation is not None
        assert run_metadata.run_validation.passed is True
        # E2E was executed and passed, so e2e_passed should be True
        assert run_metadata.run_validation.e2e_passed is True

    @pytest.mark.asyncio
    async def test_e2e_passed_false_when_e2e_enabled_and_fails(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """e2e_passed should be False when E2E is enabled and actually fails."""
        from src.domain.validation.e2e import E2EResult, E2EStatus
        from src.domain.validation.result import ValidationResult, ValidationStepResult

        mock_gate_checker = MagicMock()

        # max_gate_retries=1 to fail immediately
        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            max_gate_retries=1,
        )
        coordinator = RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
        )

        run_config = RunConfig(
            max_agents=1,
            timeout_minutes=None,
            max_issues=None,
            epic_id=None,
            only_ids=None,
        )
        run_metadata = RunMetadata(tmp_path, run_config, "test")

        async def mock_get_commit(path: Path) -> str:
            return "abc123"

        # E2E was executed and failed
        mock_result = ValidationResult(
            passed=False,
            failure_reasons=["E2E failed"],
            steps=[
                ValidationStepResult(
                    name="pytest",
                    command="pytest",
                    ok=True,
                    returncode=0,
                    stdout_tail="",
                    duration_seconds=1.0,
                )
            ],
            e2e_result=E2EResult(
                passed=False,
                status=E2EStatus.FAILED,
                failure_reason="E2E test failed",
            ),
        )

        with (
            patch(
                "src.infra.git_utils.get_git_commit_async",
                side_effect=mock_get_commit,
            ),
            patch("src.pipeline.run_coordinator.SpecValidationRunner") as MockRunner,
        ):
            mock_runner_instance = MagicMock()
            mock_runner_instance.run_spec = AsyncMock(return_value=mock_result)
            MockRunner.return_value = mock_runner_instance

            input_data = GlobalValidationInput(run_metadata=run_metadata)
            result = await coordinator.run_validation(input_data)

        assert result.passed is False
        assert run_metadata.run_validation is not None
        assert run_metadata.run_validation.passed is False
        # E2E was executed and failed, so e2e_passed should be False
        assert run_metadata.run_validation.e2e_passed is False

    @pytest.mark.asyncio
    async def test_e2e_passed_none_when_earlier_step_fails(
        self,
        tmp_path: Path,
        fake_command_runner: FakeCommandRunner,
        mock_env_config: FakeEnvConfig,
        fake_lock_manager: FakeLockManager,
        mock_sdk_client_factory: MagicMock,
    ) -> None:
        """e2e_passed should be None when validation fails before E2E runs."""
        from src.domain.validation.result import ValidationResult, ValidationStepResult

        mock_gate_checker = MagicMock()

        # max_gate_retries=1 to fail immediately
        config = RunCoordinatorConfig(
            repo_path=tmp_path,
            timeout_seconds=60,
            max_gate_retries=1,
        )
        coordinator = RunCoordinator(
            config=config,
            gate_checker=mock_gate_checker,
            command_runner=fake_command_runner,
            env_config=mock_env_config,
            lock_manager=fake_lock_manager,
            sdk_client_factory=mock_sdk_client_factory,
        )

        run_config = RunConfig(
            max_agents=1,
            timeout_minutes=None,
            max_issues=None,
            epic_id=None,
            only_ids=None,
        )
        run_metadata = RunMetadata(tmp_path, run_config, "test")

        async def mock_get_commit(path: Path) -> str:
            return "abc123"

        # pytest failed, so E2E never ran (e2e_result=None)
        mock_result = ValidationResult(
            passed=False,
            failure_reasons=["pytest failed"],
            steps=[
                ValidationStepResult(
                    name="pytest",
                    command="pytest",
                    ok=False,
                    returncode=1,
                    stdout_tail="FAILED",
                    duration_seconds=1.0,
                )
            ],
            e2e_result=None,  # E2E never ran because earlier step failed
        )

        with (
            patch(
                "src.infra.git_utils.get_git_commit_async",
                side_effect=mock_get_commit,
            ),
            patch("src.pipeline.run_coordinator.SpecValidationRunner") as MockRunner,
        ):
            mock_runner_instance = MagicMock()
            mock_runner_instance.run_spec = AsyncMock(return_value=mock_result)
            MockRunner.return_value = mock_runner_instance

            input_data = GlobalValidationInput(run_metadata=run_metadata)
            result = await coordinator.run_validation(input_data)

        assert result.passed is False
        assert run_metadata.run_validation is not None
        assert run_metadata.run_validation.passed is False
        # E2E never ran (earlier step failed), so e2e_passed should be None
        assert run_metadata.run_validation.e2e_passed is None


class TestGlobalValidationIntegration:
    """Integration tests for global validation in the orchestrator run() method."""

    @pytest.mark.asyncio
    async def test_run_calls_gate4_after_issues_complete(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """run() should call global validation after all issues complete."""
        orchestrator = make_orchestrator(repo_path=tmp_path, max_agents=1)

        gate4_called = False

        async def mock_run_validation(
            input_data: GlobalValidationInput,
            **_kwargs: object,
        ) -> GlobalValidationOutput:
            nonlocal gate4_called
            gate4_called = True
            return GlobalValidationOutput(passed=True)

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: list[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
            orphans_only: bool = False,
            order_preference: OrderPreference = OrderPreference.FOCUS,
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
                orchestrator.run_coordinator,
                "run_validation",
                side_effect=mock_run_validation,
            ),
            patch(
                "src.orchestration.orchestrator.get_lock_dir",
                return_value=tmp_path / "locks",
            ),
            patch("src.orchestration.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestration.orchestrator.release_run_locks"),
        ):
            (tmp_path / "locks").mkdir(exist_ok=True)
            await orchestrator.run()

        assert gate4_called is True

    @pytest.mark.asyncio
    async def test_run_returns_zero_success_on_gate4_failure(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """run() should return 0 successes if global validation fails."""
        orchestrator = make_orchestrator(repo_path=tmp_path, max_agents=1)

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: list[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
            orphans_only: bool = False,
            order_preference: OrderPreference = OrderPreference.FOCUS,
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

        async def mock_global_fails(
            input_data: GlobalValidationInput,
            **_kwargs: object,
        ) -> GlobalValidationOutput:
            return GlobalValidationOutput(passed=False)  # global validation fails

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
                orchestrator.run_coordinator,
                "run_validation",
                side_effect=mock_global_fails,
            ),
            patch(
                "src.orchestration.orchestrator.get_lock_dir",
                return_value=tmp_path / "locks",
            ),
            patch("src.orchestration.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestration.orchestrator.release_run_locks"),
        ):
            (tmp_path / "locks").mkdir(exist_ok=True)
            success_count, total = await orchestrator.run()

        # global validation failure should cause 0 successes to be returned
        assert success_count == 0
        assert total == 1

    @pytest.mark.asyncio
    async def test_run_skips_gate4_when_no_successes(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """run() should skip global validation when there are no successful issues."""
        orchestrator = make_orchestrator(repo_path=tmp_path, max_agents=1)

        gate4_called = False

        async def mock_run_validation(
            input_data: GlobalValidationInput,
            **_kwargs: object,
        ) -> GlobalValidationOutput:
            nonlocal gate4_called
            gate4_called = True
            return GlobalValidationOutput(passed=True)

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: list[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
            orphans_only: bool = False,
            order_preference: OrderPreference = OrderPreference.FOCUS,
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
                orchestrator.run_coordinator,
                "run_validation",
                side_effect=mock_run_validation,
            ),
            patch(
                "src.orchestration.orchestrator.get_lock_dir",
                return_value=tmp_path / "locks",
            ),
            patch("src.orchestration.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestration.orchestrator.release_run_locks"),
        ):
            (tmp_path / "locks").mkdir(exist_ok=True)
            await orchestrator.run()

        # global validation should NOT be called when no issues succeeded
        assert gate4_called is False

    @pytest.mark.asyncio
    async def test_run_skips_gate4_when_disabled(
        self, tmp_path: Path, make_orchestrator: Callable[..., MalaOrchestrator]
    ) -> None:
        """run() should skip global validation when global-validate is disabled."""
        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            disable_validations={"global-validate"},
        )

        first_call = True

        async def mock_get_ready_async(
            exclude_ids: set[str] | None = None,
            epic_id: str | None = None,
            only_ids: list[str] | None = None,
            suppress_warn_ids: set[str] | None = None,
            prioritize_wip: bool = False,
            focus: bool = True,
            orphans_only: bool = False,
            order_preference: OrderPreference = OrderPreference.FOCUS,
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
            patch(
                "src.orchestration.orchestrator.get_lock_dir",
                return_value=tmp_path / "locks",
            ),
            patch("src.orchestration.orchestrator.get_runs_dir", return_value=tmp_path),
            patch("src.orchestration.orchestrator.release_run_locks"),
        ):
            (tmp_path / "locks").mkdir(exist_ok=True)
            success_count, total = await orchestrator.run()

        # Should succeed (not fail due to global validation)
        assert success_count == 1
        assert total == 1

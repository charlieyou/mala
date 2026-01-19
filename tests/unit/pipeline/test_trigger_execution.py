"""Unit tests for trigger command resolution and execution.

Tests for RunCoordinator trigger validation:
- Command resolution from base pool
- Command and timeout overrides
- Missing ref error handling
- Fail-fast behavior
- Timeout treated as failure
- Dry-run mode
- Empty command list
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from src.domain.validation.config import (
    CommandConfig,
    CommandsConfig,
    ConfigError,
    CustomCommandConfig,
    EpicCompletionTriggerConfig,
    FailureMode,
    EpicDepth,
    FireOn,
    PeriodicTriggerConfig,
    SessionEndTriggerConfig,
    TriggerCommandRef,
    TriggerType,
    ValidationConfig,
    ValidationTriggersConfig,
)
from src.infra.tools.command_runner import CommandResult
from src.pipeline.issue_result import IssueResult
from src.pipeline.run_coordinator import (
    RunCoordinator,
    RunCoordinatorConfig,
)
from src.pipeline.fixer_service import FixerService
from src.pipeline.trigger_engine import TriggerEngine
from tests.fakes import FakeEnvConfig
from tests.fakes.command_runner import FakeCommandRunner
from tests.fakes.lock_manager import FakeLockManager

if TYPE_CHECKING:
    from pathlib import Path


def make_coordinator(
    tmp_path: Path,
    *,
    validation_config: ValidationConfig | None = None,
    command_runner: FakeCommandRunner | None = None,
    fixer_prompt: str | None = None,
) -> RunCoordinator:
    """Create a RunCoordinator with minimal fakes for trigger testing."""
    config = RunCoordinatorConfig(
        repo_path=tmp_path,
        timeout_seconds=60,
        validation_config=validation_config,
        fixer_prompt=fixer_prompt
        or "Fix attempt {attempt}/{max_attempts}: {failure_output}",
    )
    trigger_engine = TriggerEngine(validation_config=validation_config)
    mock_fixer_service = MagicMock(spec=FixerService)
    mock_fixer_service.cleanup_locks = MagicMock()
    return RunCoordinator(
        config=config,
        gate_checker=MagicMock(),
        command_runner=command_runner or FakeCommandRunner(allow_unregistered=True),
        env_config=FakeEnvConfig(),
        lock_manager=FakeLockManager(),
        sdk_client_factory=MagicMock(),
        trigger_engine=trigger_engine,
        fixer_service=mock_fixer_service,
    )


def make_validation_config(
    *,
    commands: dict[str, str | None] | None = None,
    triggers: ValidationTriggersConfig | None = None,
) -> ValidationConfig:
    """Create a ValidationConfig with specified commands and triggers.

    Args:
        commands: Dict mapping base command names to command strings.
            Example: {"test": "pytest", "lint": "ruff check ."}
        triggers: ValidationTriggersConfig for validation_triggers field.
    """
    cmd_configs: dict[str, CommandConfig | None] = {}
    if commands:
        for name, cmd_str in commands.items():
            if cmd_str is not None:
                cmd_configs[name] = CommandConfig(command=cmd_str)
            else:
                cmd_configs[name] = None

    commands_config = CommandsConfig(
        test=cmd_configs.get("test"),
        lint=cmd_configs.get("lint"),
        format=cmd_configs.get("format"),
        typecheck=cmd_configs.get("typecheck"),
    )

    return ValidationConfig(
        commands=commands_config,
        validation_triggers=triggers,
    )


class TestCommandResolution:
    """Tests for _resolve_trigger_commands."""

    def test_resolves_command_from_base_pool(self, tmp_path: Path) -> None:
        """Command ref resolves to command string from base pool."""
        config = make_validation_config(
            commands={"test": "uv run pytest"},
            triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(TriggerCommandRef(ref="test"),),
                    epic_depth=EpicDepth.TOP_LEVEL,
                    fire_on=FireOn.SUCCESS,
                )
            ),
        )
        coordinator = make_coordinator(tmp_path, validation_config=config)

        # Queue a trigger
        coordinator.queue_trigger_validation(
            TriggerType.EPIC_COMPLETION, {"epic_id": "epic-1"}
        )

        # Run with dry_run to verify resolution without execution
        result = asyncio.run(coordinator.run_trigger_validation(dry_run=True))

        assert result.status == "passed"

    def test_command_override_replaces_base_command(self, tmp_path: Path) -> None:
        """Command override in trigger replaces base pool command string."""
        runner = FakeCommandRunner()
        # Register the overridden command
        runner.responses[("custom pytest command",)] = CommandResult(
            command="custom pytest command",
            returncode=0,
            stdout="",
            stderr="",
        )

        config = make_validation_config(
            commands={"test": "uv run pytest"},  # Base pool command
            triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(
                        TriggerCommandRef(ref="test", command="custom pytest command"),
                    ),
                    epic_depth=EpicDepth.TOP_LEVEL,
                    fire_on=FireOn.SUCCESS,
                )
            ),
        )
        coordinator = make_coordinator(
            tmp_path, validation_config=config, command_runner=runner
        )
        coordinator.queue_trigger_validation(
            TriggerType.EPIC_COMPLETION, {"epic_id": "epic-1"}
        )

        result = asyncio.run(coordinator.run_trigger_validation(dry_run=False))

        assert result.status == "passed"
        # Verify the overridden command was used
        assert runner.has_call_containing("custom pytest command")

    def test_timeout_override_applies(self, tmp_path: Path) -> None:
        """Timeout override in trigger ref overrides base pool timeout."""
        runner = FakeCommandRunner()
        runner.responses[("uv run pytest",)] = CommandResult(
            command="uv run pytest",
            returncode=0,
            stdout="",
            stderr="",
        )

        config = make_validation_config(
            commands={"test": "uv run pytest"},
            triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(TriggerCommandRef(ref="test", timeout=999),),
                    epic_depth=EpicDepth.TOP_LEVEL,
                    fire_on=FireOn.SUCCESS,
                )
            ),
        )
        coordinator = make_coordinator(
            tmp_path, validation_config=config, command_runner=runner
        )
        coordinator.queue_trigger_validation(
            TriggerType.EPIC_COMPLETION, {"epic_id": "epic-1"}
        )

        result = asyncio.run(coordinator.run_trigger_validation(dry_run=False))

        assert result.status == "passed"
        # Verify timeout was passed to command runner
        assert len(runner.calls) == 1
        _, kwargs = runner.calls[0]
        assert kwargs["timeout"] == 999

    def test_missing_ref_raises_config_error(self, tmp_path: Path) -> None:
        """Missing ref in base pool raises ConfigError with available commands."""
        config = make_validation_config(
            commands={"test": "uv run pytest", "lint": "ruff check ."},
            triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(TriggerCommandRef(ref="typo"),),  # Invalid ref
                    epic_depth=EpicDepth.TOP_LEVEL,
                    fire_on=FireOn.SUCCESS,
                )
            ),
        )
        coordinator = make_coordinator(tmp_path, validation_config=config)
        coordinator.queue_trigger_validation(
            TriggerType.EPIC_COMPLETION, {"epic_id": "epic-1"}
        )

        with pytest.raises(ConfigError, match=r"(?i)unknown command.*typo"):
            asyncio.run(coordinator.run_trigger_validation(dry_run=False))

    def test_base_custom_commands_included_in_pool(self, tmp_path: Path) -> None:
        """Custom commands from base commands section are included in pool."""
        runner = FakeCommandRunner()
        runner.responses[("ci_script",)] = CommandResult(
            command="ci_script", returncode=0, stdout="", stderr=""
        )

        # Create config with custom command in commands section
        commands_config = CommandsConfig(
            custom_commands={"ci": CustomCommandConfig(command="ci_script")}
        )

        config = ValidationConfig(
            commands=commands_config,
            validation_triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(TriggerCommandRef(ref="ci"),),
                    epic_depth=EpicDepth.TOP_LEVEL,
                    fire_on=FireOn.SUCCESS,
                )
            ),
        )
        coordinator = make_coordinator(
            tmp_path, validation_config=config, command_runner=runner
        )
        coordinator.queue_trigger_validation(
            TriggerType.EPIC_COMPLETION, {"epic_id": "epic-1"}
        )

        result = asyncio.run(coordinator.run_trigger_validation(dry_run=False))

        assert result.status == "passed"
        assert len(runner.calls) == 1
        args, _ = runner.calls[0]
        assert args == ("ci_script",)


class TestFailFast:
    """Tests for fail-fast execution behavior."""

    def test_fail_fast_stops_on_first_failure(self, tmp_path: Path) -> None:
        """Second command fails, third command is not executed."""
        runner = FakeCommandRunner()
        runner.responses[("cmd1",)] = CommandResult(
            command="cmd1", returncode=0, stdout="", stderr=""
        )
        runner.responses[("cmd2",)] = CommandResult(
            command="cmd2", returncode=1, stdout="", stderr="error"
        )
        runner.responses[("cmd3",)] = CommandResult(
            command="cmd3", returncode=0, stdout="", stderr=""
        )

        # Create config with custom commands in base pool
        commands_config = CommandsConfig(
            test=CommandConfig(command="cmd1"),
            lint=CommandConfig(command="cmd2"),
            format=CommandConfig(command="cmd3"),
        )
        config = ValidationConfig(
            commands=commands_config,
            validation_triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(
                        TriggerCommandRef(ref="test"),
                        TriggerCommandRef(ref="lint"),
                        TriggerCommandRef(ref="format"),
                    ),
                    epic_depth=EpicDepth.TOP_LEVEL,
                    fire_on=FireOn.SUCCESS,
                )
            ),
        )
        coordinator = make_coordinator(
            tmp_path, validation_config=config, command_runner=runner
        )
        coordinator.queue_trigger_validation(
            TriggerType.EPIC_COMPLETION, {"epic_id": "epic-1"}
        )

        result = asyncio.run(coordinator.run_trigger_validation(dry_run=False))

        assert result.status == "failed"
        # cmd1 and cmd2 should be called, but not cmd3 (fail-fast)
        assert runner.has_call_containing("cmd1")
        assert runner.has_call_containing("cmd2")
        assert not runner.has_call_containing("cmd3")

    def test_timeout_treated_as_failure(self, tmp_path: Path) -> None:
        """Command timeout is treated as failure, triggers fail-fast."""
        runner = FakeCommandRunner()
        runner.responses[("slow_cmd",)] = CommandResult(
            command="slow_cmd",
            returncode=124,  # Timeout exit code
            stdout="",
            stderr="",
            timed_out=True,
        )
        runner.responses[("next_cmd",)] = CommandResult(
            command="next_cmd", returncode=0, stdout="", stderr=""
        )

        commands_config = CommandsConfig(
            test=CommandConfig(command="slow_cmd"),
            lint=CommandConfig(command="next_cmd"),
        )
        config = ValidationConfig(
            commands=commands_config,
            validation_triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(
                        TriggerCommandRef(ref="test"),
                        TriggerCommandRef(ref="lint"),
                    ),
                    epic_depth=EpicDepth.TOP_LEVEL,
                    fire_on=FireOn.SUCCESS,
                )
            ),
        )
        coordinator = make_coordinator(
            tmp_path, validation_config=config, command_runner=runner
        )
        coordinator.queue_trigger_validation(
            TriggerType.EPIC_COMPLETION, {"epic_id": "epic-1"}
        )

        result = asyncio.run(coordinator.run_trigger_validation(dry_run=False))

        assert result.status == "failed"
        # Only slow_cmd should be called (timeout triggers fail-fast)
        assert runner.has_call_containing("slow_cmd")
        assert not runner.has_call_containing("next_cmd")


class TestDryRun:
    """Tests for dry-run mode."""

    def test_dry_run_skips_subprocess_execution(self, tmp_path: Path) -> None:
        """Dry-run mode doesn't execute subprocess commands."""
        runner = FakeCommandRunner()  # No responses registered - would fail if called

        config = make_validation_config(
            commands={"test": "uv run pytest"},
            triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(TriggerCommandRef(ref="test"),),
                    epic_depth=EpicDepth.TOP_LEVEL,
                    fire_on=FireOn.SUCCESS,
                )
            ),
        )
        coordinator = make_coordinator(
            tmp_path, validation_config=config, command_runner=runner
        )
        coordinator.queue_trigger_validation(
            TriggerType.EPIC_COMPLETION, {"epic_id": "epic-1"}
        )

        result = asyncio.run(coordinator.run_trigger_validation(dry_run=True))

        assert result.status == "passed"
        # No commands should have been called
        assert len(runner.calls) == 0

    def test_dry_run_all_commands_treated_as_passed(self, tmp_path: Path) -> None:
        """Dry-run mode treats all commands as passed."""
        config = make_validation_config(
            commands={"test": "failing_cmd", "lint": "another_cmd"},
            triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(
                        TriggerCommandRef(ref="test"),
                        TriggerCommandRef(ref="lint"),
                    ),
                    epic_depth=EpicDepth.TOP_LEVEL,
                    fire_on=FireOn.SUCCESS,
                )
            ),
        )
        coordinator = make_coordinator(tmp_path, validation_config=config)
        coordinator.queue_trigger_validation(
            TriggerType.EPIC_COMPLETION, {"epic_id": "epic-1"}
        )

        result = asyncio.run(coordinator.run_trigger_validation(dry_run=True))

        assert result.status == "passed"


class TestEmptyCommandList:
    """Tests for empty command list handling."""

    def test_empty_command_list_returns_passed(self, tmp_path: Path) -> None:
        """Empty command list returns passed status immediately."""
        config = make_validation_config(
            commands={"test": "uv run pytest"},
            triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(),  # Empty command list
                    epic_depth=EpicDepth.TOP_LEVEL,
                    fire_on=FireOn.SUCCESS,
                )
            ),
        )
        coordinator = make_coordinator(tmp_path, validation_config=config)
        coordinator.queue_trigger_validation(
            TriggerType.EPIC_COMPLETION, {"epic_id": "epic-1"}
        )

        result = asyncio.run(coordinator.run_trigger_validation(dry_run=False))

        assert result.status == "passed"

    def test_no_queued_triggers_returns_passed(self, tmp_path: Path) -> None:
        """No queued triggers returns passed status."""
        config = make_validation_config(commands={"test": "uv run pytest"})
        coordinator = make_coordinator(tmp_path, validation_config=config)
        # Don't queue any triggers

        result = asyncio.run(coordinator.run_trigger_validation(dry_run=False))

        assert result.status == "passed"


class TestFailureModeAbort:
    """Tests for abort failure mode."""

    def test_abort_sets_aborted_status(self, tmp_path: Path) -> None:
        """ABORT mode sets result status to 'aborted'."""
        runner = FakeCommandRunner()
        runner.responses[("failing_cmd",)] = CommandResult(
            command="failing_cmd", returncode=1, stdout="", stderr="error"
        )

        config = make_validation_config(
            commands={"test": "failing_cmd"},
            triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.ABORT,
                    commands=(TriggerCommandRef(ref="test"),),
                    epic_depth=EpicDepth.TOP_LEVEL,
                    fire_on=FireOn.SUCCESS,
                )
            ),
        )
        coordinator = make_coordinator(
            tmp_path, validation_config=config, command_runner=runner
        )
        coordinator.queue_trigger_validation(
            TriggerType.EPIC_COMPLETION, {"epic_id": "epic-1"}
        )

        result = asyncio.run(coordinator.run_trigger_validation(dry_run=False))

        assert result.status == "aborted"
        assert result.details is not None
        assert "test" in result.details

    def test_abort_clears_trigger_queue(self, tmp_path: Path) -> None:
        """ABORT mode clears remaining triggers from queue."""
        runner = FakeCommandRunner()
        runner.responses[("failing_cmd",)] = CommandResult(
            command="failing_cmd", returncode=1, stdout="", stderr="error"
        )

        config = make_validation_config(
            commands={"test": "failing_cmd"},
            triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.ABORT,
                    commands=(TriggerCommandRef(ref="test"),),
                    epic_depth=EpicDepth.TOP_LEVEL,
                    fire_on=FireOn.SUCCESS,
                )
            ),
        )
        coordinator = make_coordinator(
            tmp_path, validation_config=config, command_runner=runner
        )
        # Queue multiple triggers
        coordinator.queue_trigger_validation(
            TriggerType.EPIC_COMPLETION, {"epic_id": "epic-1"}
        )
        coordinator.queue_trigger_validation(
            TriggerType.EPIC_COMPLETION, {"epic_id": "epic-2"}
        )

        result = asyncio.run(coordinator.run_trigger_validation(dry_run=False))

        assert result.status == "aborted"
        # Queue should be cleared
        assert len(coordinator._trigger_queue) == 0


class TestFailureModeContinue:
    """Tests for continue failure mode."""

    def test_continue_returns_failed_status(self, tmp_path: Path) -> None:
        """CONTINUE mode returns 'failed' status on failure."""
        runner = FakeCommandRunner()
        runner.responses[("failing_cmd",)] = CommandResult(
            command="failing_cmd", returncode=1, stdout="", stderr="error"
        )

        config = make_validation_config(
            commands={"test": "failing_cmd"},
            triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(TriggerCommandRef(ref="test"),),
                    epic_depth=EpicDepth.TOP_LEVEL,
                    fire_on=FireOn.SUCCESS,
                )
            ),
        )
        coordinator = make_coordinator(
            tmp_path, validation_config=config, command_runner=runner
        )
        coordinator.queue_trigger_validation(
            TriggerType.EPIC_COMPLETION, {"epic_id": "epic-1"}
        )

        result = asyncio.run(coordinator.run_trigger_validation(dry_run=False))

        assert result.status == "failed"
        assert result.details is not None
        assert "test" in result.details

    def test_continue_processes_remaining_triggers(self, tmp_path: Path) -> None:
        """CONTINUE mode processes all queued triggers even after failure."""
        runner = FakeCommandRunner()
        runner.responses[("failing_cmd",)] = CommandResult(
            command="failing_cmd", returncode=1, stdout="", stderr="error"
        )
        runner.responses[("second_cmd",)] = CommandResult(
            command="second_cmd", returncode=0, stdout="", stderr=""
        )

        # Use CommandsConfig directly to set up two different commands
        commands_config = CommandsConfig(
            test=CommandConfig(command="failing_cmd"),
            lint=CommandConfig(command="second_cmd"),
        )
        config = ValidationConfig(
            commands=commands_config,
            validation_triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(TriggerCommandRef(ref="test"),),
                    epic_depth=EpicDepth.TOP_LEVEL,
                    fire_on=FireOn.SUCCESS,
                ),
                session_end=SessionEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(TriggerCommandRef(ref="lint"),),
                ),
            ),
        )
        coordinator = make_coordinator(
            tmp_path, validation_config=config, command_runner=runner
        )

        # Queue two triggers - first will fail, second should still run
        coordinator.queue_trigger_validation(
            TriggerType.EPIC_COMPLETION, {"epic_id": "epic-1"}
        )
        coordinator.queue_trigger_validation(TriggerType.SESSION_END, {})

        result = asyncio.run(coordinator.run_trigger_validation(dry_run=False))

        # Should report failed (from first trigger) but second should have run
        assert result.status == "failed"
        # Both commands should have been called
        assert runner.has_call_containing("failing_cmd")
        assert runner.has_call_containing("second_cmd")


class TestFailureModeRemediate:
    """Tests for remediate failure mode."""

    def test_remediate_spawns_fixer_and_retries(self, tmp_path: Path) -> None:
        """REMEDIATE mode spawns fixer and re-runs failed command."""
        config = make_validation_config(
            commands={"test": "test_cmd"},
            triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.REMEDIATE,
                    max_retries=3,
                    commands=(TriggerCommandRef(ref="test"),),
                    epic_depth=EpicDepth.TOP_LEVEL,
                    fire_on=FireOn.SUCCESS,
                )
            ),
        )

        # Create a stateful runner that fails first, then passes
        class StatefulRunner:
            def __init__(self) -> None:
                self.call_count = 0

            async def run_async(self, cmd: str, **kwargs: object) -> CommandResult:
                self.call_count += 1
                if self.call_count == 1:
                    return CommandResult(
                        command="test_cmd", returncode=1, stdout="", stderr="lint error"
                    )
                return CommandResult(
                    command="test_cmd", returncode=0, stdout="", stderr=""
                )

        stateful_runner = StatefulRunner()

        from src.pipeline.fixer_interface import FixerResult

        # Track fixer calls
        fixer_calls: list[int] = []

        async def mock_fixer(
            failure_output: str,
            attempt: int,
            spec: object = None,
            interrupt_event: object = None,
            **kwargs: object,
        ) -> FixerResult:
            fixer_calls.append(attempt)
            return FixerResult(success=True)

        coordinator = make_coordinator(tmp_path, validation_config=config)
        coordinator.command_runner = stateful_runner  # type: ignore[assignment]
        coordinator.fixer_service.run_fixer = mock_fixer  # type: ignore[method-assign]
        coordinator.queue_trigger_validation(
            TriggerType.EPIC_COMPLETION, {"epic_id": "epic-1"}
        )

        result = asyncio.run(coordinator.run_trigger_validation(dry_run=False))

        # Command should pass after fixer
        assert result.status == "passed"
        # Fixer should have been spawned once
        assert len(fixer_calls) == 1
        # Command should have been called twice (fail, then pass)
        assert stateful_runner.call_count == 2

    def test_remediate_exhaustion_aborts(self, tmp_path: Path) -> None:
        """REMEDIATE mode aborts after max_retries exhausted."""
        runner = FakeCommandRunner()
        # Always fail
        runner.responses[("test_cmd",)] = CommandResult(
            command="test_cmd", returncode=1, stdout="", stderr="error"
        )

        config = make_validation_config(
            commands={"test": "test_cmd"},
            triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.REMEDIATE,
                    max_retries=2,
                    commands=(TriggerCommandRef(ref="test"),),
                    epic_depth=EpicDepth.TOP_LEVEL,
                    fire_on=FireOn.SUCCESS,
                )
            ),
        )

        from src.pipeline.fixer_interface import FixerResult

        # Track fixer calls
        fixer_calls: list[int] = []

        async def mock_fixer(
            failure_output: str,
            attempt: int,
            spec: object = None,
            interrupt_event: object = None,
            **kwargs: object,
        ) -> FixerResult:
            fixer_calls.append(attempt)
            return FixerResult(success=True)  # Fixer "succeeds" but command still fails

        coordinator = make_coordinator(
            tmp_path, validation_config=config, command_runner=runner
        )
        coordinator.fixer_service.run_fixer = mock_fixer  # type: ignore[method-assign]
        coordinator.queue_trigger_validation(
            TriggerType.EPIC_COMPLETION, {"epic_id": "epic-1"}
        )

        result = asyncio.run(coordinator.run_trigger_validation(dry_run=False))

        assert result.status == "aborted"
        assert result.details is not None
        assert "2 remediation attempts" in result.details
        # Fixer should have been spawned twice
        assert len(fixer_calls) == 2

    def test_max_retries_zero_no_fixer_spawned(self, tmp_path: Path) -> None:
        """REMEDIATE with max_retries=0 aborts immediately without fixer."""
        runner = FakeCommandRunner()
        runner.responses[("test_cmd",)] = CommandResult(
            command="test_cmd", returncode=1, stdout="", stderr="error"
        )

        config = make_validation_config(
            commands={"test": "test_cmd"},
            triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.REMEDIATE,
                    max_retries=0,
                    commands=(TriggerCommandRef(ref="test"),),
                    epic_depth=EpicDepth.TOP_LEVEL,
                    fire_on=FireOn.SUCCESS,
                )
            ),
        )

        from src.pipeline.fixer_interface import FixerResult

        # Track fixer calls
        fixer_calls: list[int] = []

        async def mock_fixer(
            failure_output: str,
            attempt: int,
            spec: object = None,
            interrupt_event: object = None,
            **kwargs: object,
        ) -> FixerResult:
            fixer_calls.append(attempt)
            return FixerResult(success=True)

        coordinator = make_coordinator(
            tmp_path, validation_config=config, command_runner=runner
        )
        coordinator.fixer_service.run_fixer = mock_fixer  # type: ignore[method-assign]
        coordinator.queue_trigger_validation(
            TriggerType.EPIC_COMPLETION, {"epic_id": "epic-1"}
        )

        result = asyncio.run(coordinator.run_trigger_validation(dry_run=False))

        assert result.status == "aborted"
        assert result.details is not None
        assert "max_retries=0" in result.details
        # No fixer should have been spawned
        assert len(fixer_calls) == 0


class TestSigintHandling:
    """Tests for SIGINT handling during trigger validation."""

    def test_sigint_aborts_validation(self, tmp_path: Path) -> None:
        """SIGINT during validation aborts and clears queue."""
        runner = FakeCommandRunner()
        runner.responses[("slow_cmd",)] = CommandResult(
            command="slow_cmd", returncode=0, stdout="", stderr=""
        )

        config = make_validation_config(
            commands={"test": "slow_cmd"},
            triggers=ValidationTriggersConfig(
                epic_completion=EpicCompletionTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(TriggerCommandRef(ref="test"),),
                    epic_depth=EpicDepth.TOP_LEVEL,
                    fire_on=FireOn.SUCCESS,
                )
            ),
        )
        coordinator = make_coordinator(
            tmp_path, validation_config=config, command_runner=runner
        )
        # Queue multiple triggers
        coordinator.queue_trigger_validation(
            TriggerType.EPIC_COMPLETION, {"epic_id": "epic-1"}
        )
        coordinator.queue_trigger_validation(
            TriggerType.EPIC_COMPLETION, {"epic_id": "epic-2"}
        )

        # Test using the inner loop directly with a pre-set interrupt event
        validation_config = config
        triggers_config = validation_config.validation_triggers
        assert triggers_config is not None  # Satisfy type checker

        # Create an event that's already set (simulating SIGINT received)
        interrupt_event = asyncio.Event()
        interrupt_event.set()

        result = asyncio.run(
            coordinator._run_trigger_validation_loop(
                triggers_config,
                dry_run=False,
                interrupt_event=interrupt_event,
            )
        )

        assert result.status == "aborted"
        assert result.details is not None
        assert "SIGINT" in result.details
        # Queue should be cleared
        assert len(coordinator._trigger_queue) == 0


class TestEpicCompletionTriggerIntegration:
    """Tests for epic_completion trigger integration in EpicVerificationCoordinator."""

    def _make_run_metadata(self, tmp_path: Path) -> object:
        """Create a minimal RunMetadata for testing."""
        from src.infra.io.log_output.run_metadata import RunConfig, RunMetadata

        config = RunConfig(
            max_agents=1,
            timeout_minutes=60,
            max_issues=10,
            epic_id=None,
            only_ids=None,
        )
        return RunMetadata(tmp_path, config, "test")

    def _make_epic_coordinator(
        self,
        *,
        trigger_config: EpicCompletionTriggerConfig | None = None,
        parent_epic_map: dict[str, str | None] | None = None,
    ) -> tuple:
        """Create EpicVerificationCoordinator with mock callbacks for trigger testing.

        Args:
            trigger_config: The epic_completion trigger config (or None if not configured).
            parent_epic_map: Map of issue_id -> parent_epic_id (or None for top-level).

        Returns:
            Tuple of (coordinator, queued_triggers list).
        """
        from src.core.models import EpicVerificationResult, EpicVerdict
        from src.pipeline.epic_verification_coordinator import (
            EpicVerificationCallbacks,
            EpicVerificationConfig,
            EpicVerificationCoordinator,
        )

        parent_map = parent_epic_map or {}
        queued_triggers: list[tuple[TriggerType, dict]] = []

        async def get_parent_epic(issue_id: str) -> str | None:
            return parent_map.get(issue_id)

        # Default verification result: passed
        verification_result = EpicVerificationResult(
            verified_count=1,
            passed_count=1,
            failed_count=0,
            verdicts={
                "epic-1": EpicVerdict(
                    passed=True, unmet_criteria=[], reasoning="passed"
                )
            },
            remediation_issues_created=[],
        )

        async def verify_epic(
            epic_id: str, human_override: bool
        ) -> EpicVerificationResult:
            return verification_result

        callbacks = EpicVerificationCallbacks(
            get_parent_epic=get_parent_epic,
            verify_epic=verify_epic,
            spawn_remediation=lambda issue_id, flow: None,  # type: ignore[return-value, arg-type]
            finalize_remediation=lambda issue_id, result, metadata: None,  # type: ignore[return-value, arg-type]
            mark_completed=lambda issue_id: None,
            is_issue_failed=lambda issue_id: False,
            close_eligible_epics=lambda: None,  # type: ignore[return-value, arg-type]
            on_epic_closed=lambda issue_id: None,
            on_warning=lambda msg: None,
            has_epic_verifier=lambda: True,
            get_agent_id=lambda issue_id: "test-agent",
            queue_trigger_validation=lambda t, c: queued_triggers.append((t, c)),
            get_epic_completion_trigger=lambda: trigger_config,
        )

        config = EpicVerificationConfig(max_retries=0)
        coordinator = EpicVerificationCoordinator(
            config=config,
            callbacks=callbacks,
        )

        return coordinator, queued_triggers, callbacks

    @pytest.mark.asyncio
    async def test_epic_depth_top_level_nested_epic_does_not_fire(
        self, tmp_path: Path
    ) -> None:
        """epic_depth=top_level: nested epic doesn't fire trigger."""
        trigger_config = EpicCompletionTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(),
            epic_depth=EpicDepth.TOP_LEVEL,
            fire_on=FireOn.SUCCESS,
        )
        # epic-1 has a parent epic, so it's nested
        coordinator, queued, _ = self._make_epic_coordinator(
            trigger_config=trigger_config,
            parent_epic_map={"child-1": "epic-1", "epic-1": "parent-epic"},
        )

        # Run verification
        run_metadata = self._make_run_metadata(tmp_path)
        await coordinator.check_epic_closure("child-1", run_metadata)

        # No trigger should be queued for nested epic
        assert len(queued) == 0

    @pytest.mark.asyncio
    async def test_epic_depth_top_level_top_level_epic_fires(
        self, tmp_path: Path
    ) -> None:
        """epic_depth=top_level: top-level epic fires trigger."""
        trigger_config = EpicCompletionTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(),
            epic_depth=EpicDepth.TOP_LEVEL,
            fire_on=FireOn.SUCCESS,
        )
        # epic-1 has no parent, so it's top-level
        coordinator, queued, _ = self._make_epic_coordinator(
            trigger_config=trigger_config,
            parent_epic_map={"child-1": "epic-1", "epic-1": None},
        )

        run_metadata = self._make_run_metadata(tmp_path)
        await coordinator.check_epic_closure("child-1", run_metadata)

        # Trigger should be queued
        assert len(queued) == 1
        trigger_type, context = queued[0]
        assert trigger_type == TriggerType.EPIC_COMPLETION
        assert context["epic_id"] == "epic-1"
        assert context["depth"] == "top_level"
        assert context["verification_result"] == "passed"

    @pytest.mark.asyncio
    async def test_epic_depth_all_nested_epic_fires(self, tmp_path: Path) -> None:
        """epic_depth=all: nested epic fires trigger."""
        trigger_config = EpicCompletionTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(),
            epic_depth=EpicDepth.ALL,
            fire_on=FireOn.SUCCESS,
        )
        # epic-1 has a parent
        coordinator, queued, _ = self._make_epic_coordinator(
            trigger_config=trigger_config,
            parent_epic_map={"child-1": "epic-1", "epic-1": "grandparent-epic"},
        )

        run_metadata = self._make_run_metadata(tmp_path)
        await coordinator.check_epic_closure("child-1", run_metadata)

        # Trigger should be queued even for nested epic
        assert len(queued) == 1
        trigger_type, context = queued[0]
        assert trigger_type == TriggerType.EPIC_COMPLETION
        assert context["epic_id"] == "epic-1"
        assert context["depth"] == "nested"
        assert context["verification_result"] == "passed"

    @pytest.mark.asyncio
    async def test_fire_on_success_only_fires_on_pass(self, tmp_path: Path) -> None:
        """fire_on=success: only fires when verification passes."""
        from src.core.models import EpicVerificationResult, EpicVerdict

        trigger_config = EpicCompletionTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(),
            epic_depth=EpicDepth.ALL,
            fire_on=FireOn.SUCCESS,
        )
        coordinator, queued, callbacks = self._make_epic_coordinator(
            trigger_config=trigger_config,
            parent_epic_map={"child-1": "epic-1", "epic-1": None},
        )

        # Override verify_epic to return failed result
        async def verify_failed(
            epic_id: str, human_override: bool
        ) -> EpicVerificationResult:
            return EpicVerificationResult(
                verified_count=1,
                passed_count=0,
                failed_count=1,
                verdicts={
                    "epic-1": EpicVerdict(
                        passed=False,
                        unmet_criteria=[],
                        reasoning="failed",
                    )
                },
                remediation_issues_created=[],
            )

        callbacks.verify_epic = verify_failed

        run_metadata = self._make_run_metadata(tmp_path)
        await coordinator.check_epic_closure("child-1", run_metadata)

        # No trigger should be queued on failure when fire_on=success
        assert len(queued) == 0

    @pytest.mark.asyncio
    async def test_fire_on_failure_only_fires_on_fail(self, tmp_path: Path) -> None:
        """fire_on=failure: only fires when verification fails."""
        trigger_config = EpicCompletionTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(),
            epic_depth=EpicDepth.ALL,
            fire_on=FireOn.FAILURE,
        )
        coordinator, queued, _ = self._make_epic_coordinator(
            trigger_config=trigger_config,
            parent_epic_map={"child-1": "epic-1", "epic-1": None},
        )

        # Default verify_epic returns passed, so trigger should not fire
        run_metadata = self._make_run_metadata(tmp_path)
        await coordinator.check_epic_closure("child-1", run_metadata)

        # No trigger should be queued on success when fire_on=failure
        assert len(queued) == 0

    @pytest.mark.asyncio
    async def test_fire_on_failure_fires_on_fail(self, tmp_path: Path) -> None:
        """fire_on=failure: fires when verification fails."""
        from src.core.models import EpicVerificationResult, EpicVerdict

        trigger_config = EpicCompletionTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(),
            epic_depth=EpicDepth.ALL,
            fire_on=FireOn.FAILURE,
        )
        coordinator, queued, callbacks = self._make_epic_coordinator(
            trigger_config=trigger_config,
            parent_epic_map={"child-1": "epic-1", "epic-1": None},
        )

        # Override verify_epic to return failed result
        async def verify_failed(
            epic_id: str, human_override: bool
        ) -> EpicVerificationResult:
            return EpicVerificationResult(
                verified_count=1,
                passed_count=0,
                failed_count=1,
                verdicts={
                    "epic-1": EpicVerdict(
                        passed=False,
                        unmet_criteria=[],
                        reasoning="failed",
                    )
                },
                remediation_issues_created=[],
            )

        callbacks.verify_epic = verify_failed

        run_metadata = self._make_run_metadata(tmp_path)
        await coordinator.check_epic_closure("child-1", run_metadata)

        # Trigger should be queued
        assert len(queued) == 1
        trigger_type, context = queued[0]
        assert trigger_type == TriggerType.EPIC_COMPLETION
        assert context["verification_result"] == "failed"

    @pytest.mark.asyncio
    async def test_fire_on_both_always_fires(self, tmp_path: Path) -> None:
        """fire_on=both: always fires."""
        trigger_config = EpicCompletionTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(),
            epic_depth=EpicDepth.ALL,
            fire_on=FireOn.BOTH,
        )
        coordinator, queued, _ = self._make_epic_coordinator(
            trigger_config=trigger_config,
            parent_epic_map={"child-1": "epic-1", "epic-1": None},
        )

        run_metadata = self._make_run_metadata(tmp_path)
        await coordinator.check_epic_closure("child-1", run_metadata)

        # Trigger should be queued on success
        assert len(queued) == 1
        assert queued[0][1]["verification_result"] == "passed"

    @pytest.mark.asyncio
    async def test_fire_on_both_fires_on_failure(self, tmp_path: Path) -> None:
        """fire_on=both: fires on failure too."""
        from src.core.models import EpicVerificationResult, EpicVerdict

        trigger_config = EpicCompletionTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(),
            epic_depth=EpicDepth.ALL,
            fire_on=FireOn.BOTH,
        )
        coordinator, queued, callbacks = self._make_epic_coordinator(
            trigger_config=trigger_config,
            parent_epic_map={"child-1": "epic-1", "epic-1": None},
        )

        # Override verify_epic to return failed result
        async def verify_failed(
            epic_id: str, human_override: bool
        ) -> EpicVerificationResult:
            return EpicVerificationResult(
                verified_count=1,
                passed_count=0,
                failed_count=1,
                verdicts={
                    "epic-1": EpicVerdict(
                        passed=False,
                        unmet_criteria=[],
                        reasoning="failed",
                    )
                },
                remediation_issues_created=[],
            )

        callbacks.verify_epic = verify_failed

        run_metadata = self._make_run_metadata(tmp_path)
        await coordinator.check_epic_closure("child-1", run_metadata)

        # Trigger should be queued on failure too
        assert len(queued) == 1
        assert queued[0][1]["verification_result"] == "failed"

    @pytest.mark.asyncio
    async def test_leaf_epic_with_no_children_fires(self, tmp_path: Path) -> None:
        """Leaf epic (with no children) fires when verified."""
        trigger_config = EpicCompletionTriggerConfig(
            failure_mode=FailureMode.CONTINUE,
            commands=(),
            epic_depth=EpicDepth.ALL,
            fire_on=FireOn.SUCCESS,
        )
        # Simple case: child-1's parent is epic-1
        coordinator, queued, _ = self._make_epic_coordinator(
            trigger_config=trigger_config,
            parent_epic_map={"child-1": "epic-1", "epic-1": None},
        )

        run_metadata = self._make_run_metadata(tmp_path)
        await coordinator.check_epic_closure("child-1", run_metadata)

        # Trigger should be queued
        assert len(queued) == 1
        assert queued[0][1]["epic_id"] == "epic-1"

    @pytest.mark.asyncio
    async def test_no_trigger_config_does_not_fire(self, tmp_path: Path) -> None:
        """No epic_completion trigger configured: nothing fires."""
        coordinator, queued, _ = self._make_epic_coordinator(
            trigger_config=None,
            parent_epic_map={"child-1": "epic-1", "epic-1": None},
        )

        run_metadata = self._make_run_metadata(tmp_path)
        await coordinator.check_epic_closure("child-1", run_metadata)

        # No trigger should be queued
        assert len(queued) == 0


class TestPeriodicTriggerIntegration:
    """Tests for periodic trigger counter and firing logic.

    The periodic trigger fires after N non-epic issue completions (where N is the
    configured interval). These tests verify:
    - Counter increments for each non-epic issue
    - Trigger fires at exact interval multiples (5, 10, 15...)
    - interval=1 fires on every issue completion
    - Counter < interval means no trigger fired

    These tests use the REAL MalaOrchestrator._check_and_queue_periodic_trigger
    method bound to a minimal mock object to ensure production code is tested.
    """

    def _make_orchestrator_with_periodic_trigger(
        self,
        tmp_path: Path,
        interval: int,
    ) -> tuple:
        """Create a mock orchestrator using the real periodic trigger method.

        Uses MalaOrchestrator._check_and_queue_periodic_trigger bound to a
        minimal mock object, ensuring we test the actual production code.

        Returns:
            Tuple of (mock_orchestrator, trigger_queue list).
        """
        from src.orchestration.orchestrator import MalaOrchestrator

        # Create validation config with periodic trigger
        config = ValidationConfig(
            commands=CommandsConfig(),
            validation_triggers=ValidationTriggersConfig(
                periodic=PeriodicTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(),
                    interval=interval,
                )
            ),
        )

        # Create minimal mock with required attributes for _check_and_queue_periodic_trigger
        class FakeRunCoordinator:
            def __init__(self) -> None:
                self._trigger_queue: list[tuple] = []

            def queue_trigger_validation(
                self, trigger_type: TriggerType, context: dict
            ) -> None:
                self._trigger_queue.append((trigger_type, context))

        class MockOrchestrator:
            """Mock with minimal attributes needed by the real method."""

            def __init__(self) -> None:
                self._validation_config = config
                self._non_epic_completed_count = 0
                self.run_coordinator = FakeRunCoordinator()

        mock_orch = MockOrchestrator()
        # Bind the REAL method from MalaOrchestrator to our mock
        bound_method = MalaOrchestrator._check_and_queue_periodic_trigger.__get__(
            mock_orch, MockOrchestrator
        )
        mock_orch._check_and_queue_periodic_trigger = bound_method  # type: ignore[attr-defined]

        return mock_orch, mock_orch.run_coordinator._trigger_queue

    def _make_issue_result(self, issue_id: str) -> IssueResult:
        """Create a minimal IssueResult for testing."""
        return IssueResult(
            issue_id=issue_id,
            agent_id=f"{issue_id}-agent",
            success=True,
            summary="done",
        )

    def test_counter_increments_for_each_issue(self, tmp_path: Path) -> None:
        """Counter increments by 1 for each non-epic issue completion."""
        orchestrator, _ = self._make_orchestrator_with_periodic_trigger(
            tmp_path, interval=5
        )

        # Complete 3 issues
        for i in range(3):
            result = self._make_issue_result(f"issue-{i}")
            orchestrator._check_and_queue_periodic_trigger(result)

        assert orchestrator._non_epic_completed_count == 3

    def test_trigger_fires_at_exact_interval(self, tmp_path: Path) -> None:
        """Trigger fires exactly at interval multiples (5, 10, 15...)."""
        orchestrator, queue = self._make_orchestrator_with_periodic_trigger(
            tmp_path, interval=5
        )

        # Complete 12 issues (should fire at 5 and 10)
        for i in range(12):
            result = self._make_issue_result(f"issue-{i}")
            orchestrator._check_and_queue_periodic_trigger(result)

        # Should have 2 triggers (at count 5 and 10)
        assert len(queue) == 2
        assert queue[0][0] == TriggerType.PERIODIC
        assert queue[0][1]["completed_count"] == 5
        assert queue[1][0] == TriggerType.PERIODIC
        assert queue[1][1]["completed_count"] == 10

    def test_interval_one_fires_every_issue(self, tmp_path: Path) -> None:
        """interval=1 fires trigger after every issue completion."""
        orchestrator, queue = self._make_orchestrator_with_periodic_trigger(
            tmp_path, interval=1
        )

        # Complete 3 issues
        for i in range(3):
            result = self._make_issue_result(f"issue-{i}")
            orchestrator._check_and_queue_periodic_trigger(result)

        # Should have 3 triggers (one per issue)
        assert len(queue) == 3
        assert queue[0][1]["completed_count"] == 1
        assert queue[1][1]["completed_count"] == 2
        assert queue[2][1]["completed_count"] == 3

    def test_fewer_issues_than_interval_never_fires(self, tmp_path: Path) -> None:
        """Fewer completed issues than interval means no trigger fires."""
        orchestrator, queue = self._make_orchestrator_with_periodic_trigger(
            tmp_path, interval=5
        )

        # Complete only 4 issues (less than interval of 5)
        for i in range(4):
            result = self._make_issue_result(f"issue-{i}")
            orchestrator._check_and_queue_periodic_trigger(result)

        # Counter should be 4
        assert orchestrator._non_epic_completed_count == 4
        # No triggers should have fired
        assert len(queue) == 0

    def test_counter_is_continuous_across_triggers(self, tmp_path: Path) -> None:
        """Counter continues (no reset) after trigger fires."""
        orchestrator, queue = self._make_orchestrator_with_periodic_trigger(
            tmp_path, interval=3
        )

        # Complete 9 issues (should fire at 3, 6, 9)
        for i in range(9):
            result = self._make_issue_result(f"issue-{i}")
            orchestrator._check_and_queue_periodic_trigger(result)

        # Counter should be 9 (not reset)
        assert orchestrator._non_epic_completed_count == 9
        # Should have 3 triggers
        assert len(queue) == 3
        assert queue[0][1]["completed_count"] == 3
        assert queue[1][1]["completed_count"] == 6
        assert queue[2][1]["completed_count"] == 9

    def test_no_periodic_config_no_trigger(self, tmp_path: Path) -> None:
        """No periodic trigger configured means counter increments but no trigger."""
        from src.orchestration.orchestrator import MalaOrchestrator

        # Create config without periodic trigger
        config = ValidationConfig(
            commands=CommandsConfig(),
            validation_triggers=ValidationTriggersConfig(
                periodic=None,  # No periodic trigger
            ),
        )

        class FakeRunCoordinator:
            def __init__(self) -> None:
                self._trigger_queue: list[tuple] = []

            def queue_trigger_validation(
                self, trigger_type: TriggerType, context: dict
            ) -> None:
                self._trigger_queue.append((trigger_type, context))

        class MockOrchestrator:
            def __init__(self) -> None:
                self._validation_config = config
                self._non_epic_completed_count = 0
                self.run_coordinator = FakeRunCoordinator()

        mock_orch = MockOrchestrator()
        # Bind the REAL method
        bound_method = MalaOrchestrator._check_and_queue_periodic_trigger.__get__(
            mock_orch, MockOrchestrator
        )
        mock_orch._check_and_queue_periodic_trigger = bound_method  # type: ignore[attr-defined]

        # Complete 5 issues
        for i in range(5):
            result = IssueResult(
                issue_id=f"issue-{i}",
                agent_id=f"issue-{i}-agent",
                success=True,
                summary="done",
            )
            mock_orch._check_and_queue_periodic_trigger(result)  # type: ignore[arg-type]

        # Counter should increment
        assert mock_orch._non_epic_completed_count == 5
        # No triggers queued
        assert len(mock_orch.run_coordinator._trigger_queue) == 0


class TestEpicFilteringForPeriodicTrigger:
    """Tests verifying epic guard in _check_and_queue_periodic_trigger.

    The periodic trigger must only count non-epic issue completions.
    This is enforced by an explicit is_epic check in the method itself.
    """

    def _make_orchestrator_with_periodic_trigger(
        self, tmp_path: Path, interval: int
    ) -> tuple:
        """Create a mock orchestrator using the real periodic trigger method."""
        from src.orchestration.orchestrator import MalaOrchestrator

        config = ValidationConfig(
            commands=CommandsConfig(),
            validation_triggers=ValidationTriggersConfig(
                periodic=PeriodicTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(),
                    interval=interval,
                )
            ),
        )

        class FakeRunCoordinator:
            def __init__(self) -> None:
                self._trigger_queue: list[tuple] = []

            def queue_trigger_validation(
                self, trigger_type: TriggerType, context: dict
            ) -> None:
                self._trigger_queue.append((trigger_type, context))

        class MockOrchestrator:
            def __init__(self) -> None:
                self._validation_config = config
                self._non_epic_completed_count = 0
                self.run_coordinator = FakeRunCoordinator()

        mock_orch = MockOrchestrator()
        bound_method = MalaOrchestrator._check_and_queue_periodic_trigger.__get__(
            mock_orch, MockOrchestrator
        )
        mock_orch._check_and_queue_periodic_trigger = bound_method  # type: ignore[attr-defined]
        return mock_orch, mock_orch.run_coordinator._trigger_queue

    def test_epic_issue_does_not_increment_counter(self, tmp_path: Path) -> None:
        """Epic issues with is_epic=True do not increment the counter."""
        orchestrator, queue = self._make_orchestrator_with_periodic_trigger(
            tmp_path, interval=1
        )

        # Call with epic issue (is_epic=True)
        epic_result = IssueResult(
            issue_id="epic-1",
            agent_id="epic-1-agent",
            success=True,
            summary="done",
            is_epic=True,
        )
        orchestrator._check_and_queue_periodic_trigger(epic_result)

        # Counter should NOT increment for epic
        assert orchestrator._non_epic_completed_count == 0
        # No trigger should fire
        assert len(queue) == 0

    def test_non_epic_issue_increments_counter(self, tmp_path: Path) -> None:
        """Non-epic issues (is_epic=False) increment the counter."""
        orchestrator, queue = self._make_orchestrator_with_periodic_trigger(
            tmp_path, interval=1
        )

        # Call with non-epic issue (is_epic=False, the default)
        task_result = IssueResult(
            issue_id="task-1",
            agent_id="task-1-agent",
            success=True,
            summary="done",
            is_epic=False,
        )
        orchestrator._check_and_queue_periodic_trigger(task_result)

        # Counter should increment
        assert orchestrator._non_epic_completed_count == 1
        # Trigger should fire (interval=1)
        assert len(queue) == 1

    def test_mixed_epic_and_non_epic_only_counts_non_epic(self, tmp_path: Path) -> None:
        """Only non-epic issues are counted when processing a mix."""
        orchestrator, queue = self._make_orchestrator_with_periodic_trigger(
            tmp_path, interval=2
        )

        # Process: task, epic, task, epic, task (3 tasks, 2 epics)
        results = [
            IssueResult(
                issue_id="task-1", agent_id="a", success=True, summary="", is_epic=False
            ),
            IssueResult(
                issue_id="epic-1", agent_id="a", success=True, summary="", is_epic=True
            ),
            IssueResult(
                issue_id="task-2", agent_id="a", success=True, summary="", is_epic=False
            ),
            IssueResult(
                issue_id="epic-2", agent_id="a", success=True, summary="", is_epic=True
            ),
            IssueResult(
                issue_id="task-3", agent_id="a", success=True, summary="", is_epic=False
            ),
        ]

        for r in results:
            orchestrator._check_and_queue_periodic_trigger(r)

        # Only 3 non-epic issues should be counted
        assert orchestrator._non_epic_completed_count == 3
        # Trigger should fire once at count=2 (interval=2)
        assert len(queue) == 1
        assert queue[0][1]["completed_count"] == 2

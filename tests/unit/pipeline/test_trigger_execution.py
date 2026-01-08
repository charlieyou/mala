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
    EpicCompletionTriggerConfig,
    FailureMode,
    EpicDepth,
    FireOn,
    TriggerCommandRef,
    TriggerType,
    ValidationConfig,
    ValidationTriggersConfig,
)
from src.infra.tools.command_runner import CommandResult
from src.pipeline.run_coordinator import RunCoordinator, RunCoordinatorConfig
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
) -> RunCoordinator:
    """Create a RunCoordinator with minimal fakes for trigger testing."""
    config = RunCoordinatorConfig(
        repo_path=tmp_path,
        timeout_seconds=60,
        validation_config=validation_config,
    )
    return RunCoordinator(
        config=config,
        gate_checker=MagicMock(),
        command_runner=command_runner or FakeCommandRunner(allow_unregistered=True),
        env_config=FakeEnvConfig(),
        lock_manager=FakeLockManager(),
        sdk_client_factory=MagicMock(),
    )


def make_validation_config(
    *,
    commands: dict[str, str | None] | None = None,
    triggers: ValidationTriggersConfig | None = None,
) -> ValidationConfig:
    """Create a ValidationConfig with specified commands and triggers.

    Args:
        commands: Dict mapping command names to command strings.
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

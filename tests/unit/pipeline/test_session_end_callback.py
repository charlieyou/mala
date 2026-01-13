"""Unit tests for SessionCallbackFactory on_session_end_check callback.

Tests cover the full session_end callback implementation per spec R9-R11:
- R9: Commands execute sequentially; failure_mode (abort, continue)
- R10: Overall timeout wrapper; on timeout/SIGINT return appropriate result
- R11: code_review uses base_sha..HEAD range
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from src.core.session_end_result import (
    SessionEndResult,
    SessionEndRetryState,
)
from src.domain.validation.config import (
    CodeReviewConfig,
    CommandConfig,
    CommandsConfig,
    FailureMode,
    SessionEndTriggerConfig,
    TriggerCommandRef,
    ValidationConfig,
    ValidationTriggersConfig,
)
from src.pipeline.session_callback_factory import (
    SessionCallbackFactory,
    SessionRunContext,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class FakeCommandResult:
    """Fake command result for testing."""

    command: str | list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = 0.1
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def stdout_tail(self, max_chars: int = 800, max_lines: int = 20) -> str:
        return self.stdout[:max_chars]


class FakeCommandRunner:
    """Fake command runner for testing."""

    def __init__(self) -> None:
        self.commands_run: list[str] = []
        self.responses: dict[str, FakeCommandResult] = {}
        self.default_result = FakeCommandResult(command="", returncode=0)

    async def run_async(
        self,
        cmd: str | list[str],
        timeout: int | None = None,
        shell: bool = False,
        cwd: Path | None = None,
    ) -> FakeCommandResult:
        cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
        self.commands_run.append(cmd_str)
        return self.responses.get(cmd_str, self.default_result)


@dataclass
class FakeReviewFinding:
    """Fake review finding for testing."""

    file: str
    line_start: int
    line_end: int
    priority: int
    title: str
    body: str


@dataclass
class FakeCumulativeReviewResult:
    """Fake cumulative review result for testing."""

    status: str
    findings: tuple[FakeReviewFinding, ...]
    new_baseline_commit: str | None = None
    skip_reason: str | None = None


class FakeCumulativeReviewRunner:
    """Fake cumulative review runner for testing."""

    def __init__(
        self,
        result: FakeCumulativeReviewResult | None = None,
        should_raise: Exception | None = None,
    ) -> None:
        self.result = result or FakeCumulativeReviewResult(
            status="success", findings=()
        )
        self.should_raise = should_raise
        self.call_count = 0
        self.last_issue_id: str | None = None
        self.last_baseline_override: str | None = None

    async def run_review(
        self,
        trigger_type: object,
        config: object,
        run_metadata: object,
        repo_path: Path,
        interrupt_event: asyncio.Event,
        *,
        issue_id: str | None = None,
        epic_id: str | None = None,
        baseline_override: str | None = None,
    ) -> FakeCumulativeReviewResult:
        self.call_count += 1
        self.last_issue_id = issue_id
        self.last_baseline_override = baseline_override
        if self.should_raise:
            raise self.should_raise
        return self.result


@dataclass
class FakeRunMetadata:
    """Fake run metadata for testing."""

    run_id: str = "test-run-123"
    run_start_commit: str | None = "abc123"
    last_cumulative_review_commits: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.last_cumulative_review_commits is None:
            self.last_cumulative_review_commits = {}


def _create_factory(
    *,
    validation_config: ValidationConfig | None = None,
    command_runner: FakeCommandRunner | None = None,
    cumulative_review_runner: FakeCumulativeReviewRunner | None = None,
    run_metadata: FakeRunMetadata | None = None,
    base_sha_map: dict[str, str] | None = None,
    on_abort: Callable[[str], None] | None = None,
    interrupt_event: asyncio.Event | None = None,
    session_end_timeout: float = 600.0,
) -> SessionCallbackFactory:
    """Create a SessionCallbackFactory with test dependencies."""
    context = SessionRunContext(
        log_provider_getter=lambda: MagicMock(),
        evidence_check_getter=lambda: MagicMock(),
        on_session_log_path=lambda issue_id, path: None,
        on_review_log_path=lambda issue_id, path: None,
        interrupt_event_getter=lambda: interrupt_event,
        get_base_sha=lambda issue_id: (base_sha_map or {}).get(issue_id),
        get_run_metadata=lambda: run_metadata or FakeRunMetadata(),  # type: ignore[return-value]
        on_abort=on_abort or (lambda reason: None),
        abort_event_getter=lambda: None,
    )
    return SessionCallbackFactory(
        gate_async_runner=MagicMock(),
        review_runner=MagicMock(),
        context=context,
        event_sink=MagicMock(return_value=MagicMock()),
        repo_path=Path("/test/repo"),
        get_per_session_spec=MagicMock(return_value=None),
        is_verbose=MagicMock(return_value=False),
        get_validation_config=lambda: validation_config,
        command_runner=command_runner,  # type: ignore[arg-type]
        cumulative_review_runner=cumulative_review_runner,  # type: ignore[arg-type]
        session_end_timeout=session_end_timeout,
    )


def _make_validation_config(
    *,
    commands: tuple[TriggerCommandRef, ...] = (),
    failure_mode: FailureMode = FailureMode.CONTINUE,
    code_review_enabled: bool = False,
    base_commands: dict[str, str] | None = None,
) -> ValidationConfig:
    """Create a ValidationConfig with session_end trigger."""
    # Build base commands config
    cmd_configs: dict[str, CommandConfig | None] = {}
    if base_commands:
        for name, cmd_str in base_commands.items():
            cmd_configs[name] = CommandConfig(command=cmd_str)

    commands_config = CommandsConfig(
        test=cmd_configs.get("test"),
        lint=cmd_configs.get("lint"),
        format=cmd_configs.get("format"),
        typecheck=cmd_configs.get("typecheck"),
    )

    # Build code_review config if enabled
    code_review_config = None
    if code_review_enabled:
        code_review_config = CodeReviewConfig(enabled=True, failure_mode=failure_mode)

    session_end_config = SessionEndTriggerConfig(
        failure_mode=failure_mode,
        commands=commands,
        code_review=code_review_config,
    )

    return ValidationConfig(
        commands=commands_config,
        validation_triggers=ValidationTriggersConfig(session_end=session_end_config),
    )


class TestSessionEndAdapterPresence:
    """Tests for run_session_end_check in protocol adapters."""

    def test_adapter_has_run_session_end_check(self) -> None:
        """Gate runner adapter has run_session_end_check method."""
        factory = _create_factory()
        adapters = factory.build_adapters("test-issue")

        assert hasattr(adapters.gate_runner, "run_session_end_check")


class TestSessionEndNotConfigured:
    """Tests for session_end when not configured."""

    async def test_returns_skipped_when_no_validation_config(self) -> None:
        """Returns skipped when validation_config is None."""
        factory = _create_factory(validation_config=None)
        adapters = factory.build_adapters("test-issue")

        on_session_end_check = adapters.gate_runner.run_session_end_check
        result = await on_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )

        assert isinstance(result, SessionEndResult)
        assert result.status == "skipped"
        assert result.reason == "not_configured"

    async def test_returns_skipped_when_no_triggers_config(self) -> None:
        """Returns skipped when validation_triggers is None."""
        config = ValidationConfig(commands=CommandsConfig(), validation_triggers=None)
        factory = _create_factory(validation_config=config)
        adapters = factory.build_adapters("test-issue")

        on_session_end_check = adapters.gate_runner.run_session_end_check
        result = await on_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )

        assert result.status == "skipped"
        assert result.reason == "not_configured"

    async def test_returns_skipped_when_session_end_not_in_triggers(self) -> None:
        """Returns skipped when session_end is not configured in triggers."""
        config = ValidationConfig(
            commands=CommandsConfig(), validation_triggers=ValidationTriggersConfig()
        )
        factory = _create_factory(validation_config=config)
        adapters = factory.build_adapters("test-issue")

        on_session_end_check = adapters.gate_runner.run_session_end_check
        result = await on_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )

        assert result.status == "skipped"
        assert result.reason == "not_configured"


class TestCommandExecution:
    """Tests for command execution in session_end."""

    async def test_commands_execute_in_sequence(self) -> None:
        """Commands execute sequentially."""
        runner = FakeCommandRunner()
        runner.responses["ruff check ."] = FakeCommandResult(
            command="ruff check .", returncode=0
        )
        runner.responses["uv run pytest"] = FakeCommandResult(
            command="uv run pytest", returncode=0
        )

        config = _make_validation_config(
            commands=(
                TriggerCommandRef(ref="lint"),
                TriggerCommandRef(ref="test"),
            ),
            base_commands={"lint": "ruff check .", "test": "uv run pytest"},
        )
        factory = _create_factory(validation_config=config, command_runner=runner)
        adapters = factory.build_adapters("test-issue")

        on_session_end_check = adapters.gate_runner.run_session_end_check
        result = await on_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )

        assert result.status == "pass"
        assert len(runner.commands_run) == 2
        assert runner.commands_run[0] == "ruff check ."
        assert runner.commands_run[1] == "uv run pytest"

    async def test_fail_fast_stops_on_first_failure(self) -> None:
        """Fail-fast: stops on first command failure."""
        runner = FakeCommandRunner()
        runner.responses["ruff check ."] = FakeCommandResult(
            command="ruff check .", returncode=1, stderr="lint error"
        )

        config = _make_validation_config(
            commands=(
                TriggerCommandRef(ref="lint"),
                TriggerCommandRef(ref="test"),
            ),
            base_commands={"lint": "ruff check .", "test": "uv run pytest"},
        )
        factory = _create_factory(validation_config=config, command_runner=runner)
        adapters = factory.build_adapters("test-issue")

        on_session_end_check = adapters.gate_runner.run_session_end_check
        result = await on_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )

        assert result.status == "fail"
        assert result.reason == "command_failed"
        # Second command should not have run
        assert len(runner.commands_run) == 1

    async def test_empty_commands_returns_pass(self) -> None:
        """Empty command list returns pass status."""
        config = _make_validation_config(commands=(), base_commands={})
        factory = _create_factory(
            validation_config=config, command_runner=FakeCommandRunner()
        )
        adapters = factory.build_adapters("test-issue")

        on_session_end_check = adapters.gate_runner.run_session_end_check
        result = await on_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )

        assert result.status == "pass"

    async def test_timestamps_are_set(self) -> None:
        """started_at and finished_at are set for executed session_end."""
        config = _make_validation_config(commands=(), base_commands={})
        factory = _create_factory(
            validation_config=config, command_runner=FakeCommandRunner()
        )
        adapters = factory.build_adapters("test-issue")

        before = datetime.now(UTC)
        on_session_end_check = adapters.gate_runner.run_session_end_check
        result = await on_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )
        after = datetime.now(UTC)

        assert result.started_at is not None
        assert result.finished_at is not None
        assert before <= result.started_at <= after
        assert result.started_at <= result.finished_at <= after


class TestTimeoutBehavior:
    """Tests for session_end timeout behavior (spec R10)."""

    async def test_timeout_returns_empty_commands(self) -> None:
        """On timeout, returns status=timeout with empty commands."""
        runner = FakeCommandRunner()

        # Make the command take too long
        async def slow_run(
            cmd: str | list[str],
            timeout: int | None = None,
            shell: bool = False,
            cwd: Path | None = None,
        ) -> FakeCommandResult:
            await asyncio.sleep(10)
            return FakeCommandResult(command="slow", returncode=0)

        runner.run_async = slow_run  # type: ignore[method-assign]

        config = _make_validation_config(
            commands=(TriggerCommandRef(ref="lint"),),
            base_commands={"lint": "slow"},
        )
        factory = _create_factory(
            validation_config=config,
            command_runner=runner,
            session_end_timeout=0.1,  # Very short timeout
        )
        adapters = factory.build_adapters("test-issue")

        on_session_end_check = adapters.gate_runner.run_session_end_check
        result = await on_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )

        assert result.status == "timeout"
        assert result.reason == "session_end_timeout"
        assert result.commands == []


class TestInterruptBehavior:
    """Tests for session_end SIGINT behavior (spec R10)."""

    async def test_interrupt_before_first_command_returns_empty(self) -> None:
        """On SIGINT before first command, returns empty commands list."""
        interrupt_event = asyncio.Event()
        interrupt_event.set()  # Already interrupted

        config = _make_validation_config(
            commands=(TriggerCommandRef(ref="lint"),),
            base_commands={"lint": "ruff check ."},
        )
        factory = _create_factory(
            validation_config=config,
            command_runner=FakeCommandRunner(),
            interrupt_event=interrupt_event,
        )
        adapters = factory.build_adapters("test-issue")

        on_session_end_check = adapters.gate_runner.run_session_end_check
        result = await on_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )

        assert result.status == "interrupted"
        assert result.reason == "SIGINT received"
        assert result.commands == []

    async def test_interrupt_after_some_commands_returns_empty(self) -> None:
        """On SIGINT after some commands, returns empty commands (per spec R10)."""
        interrupt_event = asyncio.Event()
        runner = FakeCommandRunner()
        runner.responses["ruff check ."] = FakeCommandResult(
            command="ruff check .", returncode=0
        )

        # Custom runner that sets interrupt after first command
        original_run = runner.run_async

        async def run_then_interrupt(
            cmd: str | list[str],
            timeout: int | None = None,
            shell: bool = False,
            cwd: Path | None = None,
        ) -> FakeCommandResult:
            result = await original_run(cmd, timeout, shell, cwd)
            interrupt_event.set()  # Set interrupt after command completes
            return result

        runner.run_async = run_then_interrupt  # type: ignore[method-assign]

        config = _make_validation_config(
            commands=(
                TriggerCommandRef(ref="lint"),
                TriggerCommandRef(ref="test"),
            ),
            base_commands={"lint": "ruff check .", "test": "uv run pytest"},
        )
        factory = _create_factory(
            validation_config=config,
            command_runner=runner,
            interrupt_event=interrupt_event,
        )
        adapters = factory.build_adapters("test-issue")

        on_session_end_check = adapters.gate_runner.run_session_end_check
        result = await on_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )

        assert result.status == "interrupted"
        # Per spec R10: partial results discarded on interrupt
        assert result.commands == []

    async def test_command_completes_on_sigint_during_execution(self) -> None:
        """Command execution completes on SIGINT even if cancelled mid-execution.

        Per R10: "complete current command" on SIGINT means shield protects execution.
        The key is that interrupt_event is set, signaling SIGINT (not timeout).
        However, per R10 the partial results are discarded for consistency.
        """
        interrupt_event = asyncio.Event()
        command_started = asyncio.Event()
        command_completed = asyncio.Event()

        class SlowCommandRunner:
            """Runner that takes time to complete, allowing cancellation during execution."""

            async def run_async(
                self,
                cmd: str | list[str],
                timeout: int | None = None,
                shell: bool = False,
                cwd: Path | None = None,
            ) -> FakeCommandResult:
                command_started.set()
                # Simulate command taking time to complete
                await asyncio.sleep(0.1)
                command_completed.set()
                return FakeCommandResult(command=str(cmd), returncode=0)

        runner = SlowCommandRunner()
        config = _make_validation_config(
            commands=(TriggerCommandRef(ref="lint"),),
            base_commands={"lint": "ruff check ."},
        )
        factory = _create_factory(
            validation_config=config,
            command_runner=runner,  # type: ignore[arg-type]
            interrupt_event=interrupt_event,
        )
        adapters = factory.build_adapters("test-issue")
        on_session_end_check = adapters.gate_runner.run_session_end_check

        async def run_and_sigint() -> SessionEndResult:
            task = asyncio.create_task(
                on_session_end_check(
                    "test-issue", Path("/test/log.txt"), SessionEndRetryState()
                )
            )
            # Wait for command to start, then simulate SIGINT
            await command_started.wait()
            interrupt_event.set()  # Signal SIGINT
            task.cancel()  # Cancel simulates the task being interrupted
            return await task

        result = await run_and_sigint()

        # Command should have completed despite cancellation (SIGINT path)
        assert command_completed.is_set(), "Command did not complete (shield failed)"
        # Per spec R10: partial results discarded on interrupt
        assert result.commands == []
        assert result.status == "interrupted"
        assert result.reason == "SIGINT received"


class TestFailureModeAbort:
    """Tests for failure_mode=abort behavior."""

    async def test_abort_mode_calls_on_abort(self) -> None:
        """failure_mode=abort calls on_abort callback on failure."""
        runner = FakeCommandRunner()
        runner.responses["ruff check ."] = FakeCommandResult(
            command="ruff check .", returncode=1
        )

        abort_called = []

        def on_abort(reason: str) -> None:
            abort_called.append(reason)

        config = _make_validation_config(
            commands=(TriggerCommandRef(ref="lint"),),
            failure_mode=FailureMode.ABORT,
            base_commands={"lint": "ruff check ."},
        )
        factory = _create_factory(
            validation_config=config, command_runner=runner, on_abort=on_abort
        )
        adapters = factory.build_adapters("test-issue")

        on_session_end_check = adapters.gate_runner.run_session_end_check
        result = await on_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )

        assert result.status == "fail"
        assert len(abort_called) == 1
        assert "test-issue" in abort_called[0]


class TestFailureModeContinue:
    """Tests for failure_mode=continue behavior."""

    async def test_continue_mode_proceeds_on_failure(self) -> None:
        """failure_mode=continue proceeds regardless of failure."""
        runner = FakeCommandRunner()
        runner.responses["ruff check ."] = FakeCommandResult(
            command="ruff check .", returncode=1
        )

        abort_called = []

        def on_abort(reason: str) -> None:
            abort_called.append(reason)

        config = _make_validation_config(
            commands=(TriggerCommandRef(ref="lint"),),
            failure_mode=FailureMode.CONTINUE,
            base_commands={"lint": "ruff check ."},
        )
        factory = _create_factory(
            validation_config=config, command_runner=runner, on_abort=on_abort
        )
        adapters = factory.build_adapters("test-issue")

        on_session_end_check = adapters.gate_runner.run_session_end_check
        result = await on_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )

        assert result.status == "fail"
        assert result.reason == "command_failed"
        # on_abort should NOT be called for continue mode
        assert len(abort_called) == 0


class TestCodeReviewIntegration:
    """Tests for code_review integration (spec R11)."""

    async def test_code_review_runs_after_commands_pass(self) -> None:
        """code_review runs after commands pass."""
        runner = FakeCommandRunner()
        runner.responses["ruff check ."] = FakeCommandResult(
            command="ruff check .", returncode=0
        )

        review_runner = FakeCumulativeReviewRunner()
        base_sha_map = {"test-issue": "abc123"}

        config = _make_validation_config(
            commands=(TriggerCommandRef(ref="lint"),),
            code_review_enabled=True,
            base_commands={"lint": "ruff check ."},
        )
        factory = _create_factory(
            validation_config=config,
            command_runner=runner,
            cumulative_review_runner=review_runner,
            base_sha_map=base_sha_map,
        )
        adapters = factory.build_adapters("test-issue")

        on_session_end_check = adapters.gate_runner.run_session_end_check
        result = await on_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )

        assert result.status == "pass"
        assert review_runner.call_count == 1
        assert review_runner.last_issue_id == "test-issue"
        assert result.code_review_result is not None
        assert result.code_review_result.ran is True

    async def test_code_review_runs_with_continue_mode_on_failure(self) -> None:
        """code_review runs with continue mode even when commands fail."""
        runner = FakeCommandRunner()
        runner.responses["ruff check ."] = FakeCommandResult(
            command="ruff check .", returncode=1
        )

        review_runner = FakeCumulativeReviewRunner()
        base_sha_map = {"test-issue": "abc123"}

        config = _make_validation_config(
            commands=(TriggerCommandRef(ref="lint"),),
            failure_mode=FailureMode.CONTINUE,
            code_review_enabled=True,
            base_commands={"lint": "ruff check ."},
        )
        factory = _create_factory(
            validation_config=config,
            command_runner=runner,
            cumulative_review_runner=review_runner,
            base_sha_map=base_sha_map,
        )
        adapters = factory.build_adapters("test-issue")

        on_session_end_check = adapters.gate_runner.run_session_end_check
        result = await on_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )

        assert result.status == "fail"
        # code_review should still run with continue mode
        assert review_runner.call_count == 1

    async def test_code_review_skipped_without_base_sha(self) -> None:
        """code_review is skipped when base_sha is not available."""
        runner = FakeCommandRunner()
        review_runner = FakeCumulativeReviewRunner()

        config = _make_validation_config(
            commands=(),
            code_review_enabled=True,
            base_commands={},
        )
        factory = _create_factory(
            validation_config=config,
            command_runner=runner,
            cumulative_review_runner=review_runner,
            base_sha_map={},  # No base_sha for test-issue
        )
        adapters = factory.build_adapters("test-issue")

        on_session_end_check = adapters.gate_runner.run_session_end_check
        result = await on_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )

        assert result.status == "pass"
        assert review_runner.call_count == 0
        assert result.code_review_result is not None
        assert result.code_review_result.ran is False

    async def test_code_review_findings_are_captured(self) -> None:
        """code_review findings are included in SessionEndResult."""
        runner = FakeCommandRunner()
        review_runner = FakeCumulativeReviewRunner(
            result=FakeCumulativeReviewResult(
                status="success",
                findings=(
                    FakeReviewFinding(
                        file="test.py",
                        line_start=10,
                        line_end=15,
                        priority=1,
                        title="Test finding",
                        body="Description",
                    ),
                ),
            )
        )
        base_sha_map = {"test-issue": "abc123"}

        config = _make_validation_config(
            commands=(),
            code_review_enabled=True,
            base_commands={},
        )
        factory = _create_factory(
            validation_config=config,
            command_runner=runner,
            cumulative_review_runner=review_runner,
            base_sha_map=base_sha_map,
        )
        adapters = factory.build_adapters("test-issue")

        on_session_end_check = adapters.gate_runner.run_session_end_check
        result = await on_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )

        assert result.code_review_result is not None
        assert result.code_review_result.ran is True
        assert len(result.code_review_result.findings) == 1
        assert result.code_review_result.findings[0]["file"] == "test.py"


class TestConfigOverrideTimeout:
    """Tests for config override: non-default timeout reaches runtime."""

    async def test_custom_timeout_value_is_used(self) -> None:
        """Non-default session_end_timeout value is applied."""
        runner = FakeCommandRunner()

        config = _make_validation_config(commands=(), base_commands={})
        factory = _create_factory(
            validation_config=config,
            command_runner=runner,
            session_end_timeout=120.0,  # Custom timeout
        )
        # We can verify the timeout is stored correctly
        assert factory._session_end_timeout == 120.0

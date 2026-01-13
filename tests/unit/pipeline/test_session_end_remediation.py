"""Unit tests for session_end remediation loop (failure_mode=remediate).

Tests cover the remediation loop implementation per spec R9:
- Fixer runs before each retry (not before initial attempt)
- Total validation attempts = 1 + max_retries
- On exhausted retries: return SessionEndResult(status="fail", reason="max_retries_exhausted")
- On successful retry: return SessionEndResult(status="pass")
- code_review runs exactly once after final outcome
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from src.core.session_end_result import SessionEndRetryState

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
from src.pipeline.fixer_interface import FixerResult
from src.pipeline.session_callback_factory import (
    SessionCallbackFactory,
    SessionRunContext,
)

if TYPE_CHECKING:
    import asyncio
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
        # Sequence of results for repeated calls to same command
        self._result_sequences: dict[str, list[FakeCommandResult]] = {}
        self._call_counts: dict[str, int] = {}

    def set_result_sequence(self, cmd: str, results: list[FakeCommandResult]) -> None:
        """Configure a sequence of results for consecutive calls to the same command."""
        self._result_sequences[cmd] = results
        self._call_counts[cmd] = 0

    async def run_async(
        self,
        cmd: str | list[str],
        timeout: int | None = None,
        shell: bool = False,
        cwd: Path | None = None,
    ) -> FakeCommandResult:
        cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
        self.commands_run.append(cmd_str)

        # Check for sequence-based results
        if cmd_str in self._result_sequences:
            idx = self._call_counts.get(cmd_str, 0)
            self._call_counts[cmd_str] = idx + 1
            seq = self._result_sequences[cmd_str]
            if idx < len(seq):
                return seq[idx]
            # If exhausted, return last result
            return seq[-1] if seq else self.default_result

        return self.responses.get(cmd_str, self.default_result)


class FakeFixerInterface:
    """Fake fixer interface for testing."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.result = FixerResult(success=True)
        # Sequence of results for consecutive calls
        self._result_sequence: list[FixerResult] = []
        self._call_index = 0

    def set_result_sequence(self, results: list[FixerResult]) -> None:
        """Configure a sequence of results for consecutive calls."""
        self._result_sequence = results
        self._call_index = 0

    async def run_fixer(self, failure_output: str, issue_id: str) -> FixerResult:
        """Record the call and return the configured result."""
        self.calls.append((failure_output, issue_id))

        if self._result_sequence:
            idx = self._call_index
            self._call_index += 1
            if idx < len(self._result_sequence):
                return self._result_sequence[idx]
            return self._result_sequence[-1] if self._result_sequence else self.result

        return self.result


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
    fixer_interface: FakeFixerInterface | None = None,
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
        fixer_interface=fixer_interface,  # type: ignore[arg-type]
        cumulative_review_runner=cumulative_review_runner,  # type: ignore[arg-type]
        session_end_timeout=session_end_timeout,
    )


def _make_validation_config(
    *,
    commands: tuple[TriggerCommandRef, ...] = (),
    failure_mode: FailureMode = FailureMode.REMEDIATE,
    max_retries: int = 2,
    code_review_enabled: bool = False,
    base_commands: dict[str, str] | None = None,
) -> ValidationConfig:
    """Create a ValidationConfig with session_end trigger."""
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

    code_review_config = None
    if code_review_enabled:
        code_review_config = CodeReviewConfig(enabled=True, failure_mode=failure_mode)

    session_end_config = SessionEndTriggerConfig(
        failure_mode=failure_mode,
        commands=commands,
        max_retries=max_retries,
        code_review=code_review_config,
    )

    return ValidationConfig(
        commands=commands_config,
        validation_triggers=ValidationTriggersConfig(session_end=session_end_config),
    )


class TestRemediationLoopBasics:
    """Tests for basic remediation loop behavior."""

    @pytest.mark.asyncio
    async def test_initial_pass_no_fixer_called(self) -> None:
        """Fixer is not called when initial validation passes."""
        runner = FakeCommandRunner()
        runner.responses["ruff check ."] = FakeCommandResult(
            command="ruff check .", returncode=0
        )
        fixer = FakeFixerInterface()

        config = _make_validation_config(
            commands=(TriggerCommandRef(ref="lint"),),
            failure_mode=FailureMode.REMEDIATE,
            max_retries=2,
            base_commands={"lint": "ruff check ."},
        )
        factory = _create_factory(
            validation_config=config,
            command_runner=runner,
            fixer_interface=fixer,
        )
        adapters = factory.build_adapters("test-issue")

        result = await adapters.gate_runner.run_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )

        assert result.status == "pass"
        assert len(fixer.calls) == 0
        assert len(runner.commands_run) == 1

    @pytest.mark.asyncio
    async def test_remediation_with_max_retries_2_fixer_runs_twice(self) -> None:
        """With max_retries=2, fixer runs twice on consecutive failures."""
        runner = FakeCommandRunner()
        # All 3 attempts fail (initial + 2 retries)
        runner.set_result_sequence(
            "ruff check .",
            [
                FakeCommandResult(command="ruff check .", returncode=1, stderr="fail1"),
                FakeCommandResult(command="ruff check .", returncode=1, stderr="fail2"),
                FakeCommandResult(command="ruff check .", returncode=1, stderr="fail3"),
            ],
        )
        fixer = FakeFixerInterface()

        config = _make_validation_config(
            commands=(TriggerCommandRef(ref="lint"),),
            failure_mode=FailureMode.REMEDIATE,
            max_retries=2,
            base_commands={"lint": "ruff check ."},
        )
        factory = _create_factory(
            validation_config=config,
            command_runner=runner,
            fixer_interface=fixer,
        )
        adapters = factory.build_adapters("test-issue")

        result = await adapters.gate_runner.run_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )

        assert result.status == "fail"
        assert result.reason == "max_retries_exhausted"
        # Fixer runs before each retry, not before initial attempt
        assert len(fixer.calls) == 2
        # Total validation attempts = 1 + max_retries = 3
        assert len(runner.commands_run) == 3


class TestRemediationSuccess:
    """Tests for successful remediation scenarios."""

    @pytest.mark.asyncio
    async def test_successful_retry_stops_loop_and_returns_pass(self) -> None:
        """Successful retry stops the loop and returns pass."""
        runner = FakeCommandRunner()
        # First attempt fails, second succeeds
        runner.set_result_sequence(
            "ruff check .",
            [
                FakeCommandResult(command="ruff check .", returncode=1, stderr="fail"),
                FakeCommandResult(command="ruff check .", returncode=0),
            ],
        )
        fixer = FakeFixerInterface()

        config = _make_validation_config(
            commands=(TriggerCommandRef(ref="lint"),),
            failure_mode=FailureMode.REMEDIATE,
            max_retries=2,
            base_commands={"lint": "ruff check ."},
        )
        factory = _create_factory(
            validation_config=config,
            command_runner=runner,
            fixer_interface=fixer,
        )
        adapters = factory.build_adapters("test-issue")

        result = await adapters.gate_runner.run_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )

        assert result.status == "pass"
        assert result.reason is None
        # Only one fixer call (after first failure)
        assert len(fixer.calls) == 1
        # Two validation attempts
        assert len(runner.commands_run) == 2

    @pytest.mark.asyncio
    async def test_successful_on_last_retry_returns_pass(self) -> None:
        """Success on the last retry attempt still returns pass."""
        runner = FakeCommandRunner()
        # First two attempts fail, third succeeds
        runner.set_result_sequence(
            "ruff check .",
            [
                FakeCommandResult(command="ruff check .", returncode=1, stderr="fail1"),
                FakeCommandResult(command="ruff check .", returncode=1, stderr="fail2"),
                FakeCommandResult(command="ruff check .", returncode=0),
            ],
        )
        fixer = FakeFixerInterface()

        config = _make_validation_config(
            commands=(TriggerCommandRef(ref="lint"),),
            failure_mode=FailureMode.REMEDIATE,
            max_retries=2,
            base_commands={"lint": "ruff check ."},
        )
        factory = _create_factory(
            validation_config=config,
            command_runner=runner,
            fixer_interface=fixer,
        )
        adapters = factory.build_adapters("test-issue")

        result = await adapters.gate_runner.run_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )

        assert result.status == "pass"
        # Two fixer calls (after first and second failures)
        assert len(fixer.calls) == 2
        # Three validation attempts
        assert len(runner.commands_run) == 3


class TestRemediationExhausted:
    """Tests for exhausted retries scenarios."""

    @pytest.mark.asyncio
    async def test_exhausted_retries_returns_fail_with_reason(self) -> None:
        """Exhausted retries returns fail with reason='max_retries_exhausted'."""
        runner = FakeCommandRunner()
        runner.responses["ruff check ."] = FakeCommandResult(
            command="ruff check .", returncode=1, stderr="always fail"
        )
        fixer = FakeFixerInterface()

        config = _make_validation_config(
            commands=(TriggerCommandRef(ref="lint"),),
            failure_mode=FailureMode.REMEDIATE,
            max_retries=1,
            base_commands={"lint": "ruff check ."},
        )
        factory = _create_factory(
            validation_config=config,
            command_runner=runner,
            fixer_interface=fixer,
        )
        adapters = factory.build_adapters("test-issue")

        result = await adapters.gate_runner.run_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )

        assert result.status == "fail"
        assert result.reason == "max_retries_exhausted"
        # One fixer call for max_retries=1
        assert len(fixer.calls) == 1
        # Two validation attempts
        assert len(runner.commands_run) == 2


class TestCodeReviewAfterRemediation:
    """Tests for code_review running after remediation."""

    @pytest.mark.asyncio
    async def test_code_review_runs_after_exhausted_retries(self) -> None:
        """code_review runs after exhausted retries."""
        runner = FakeCommandRunner()
        runner.responses["ruff check ."] = FakeCommandResult(
            command="ruff check .", returncode=1
        )
        fixer = FakeFixerInterface()
        review_runner = FakeCumulativeReviewRunner()
        base_sha_map = {"test-issue": "abc123"}

        config = _make_validation_config(
            commands=(TriggerCommandRef(ref="lint"),),
            failure_mode=FailureMode.REMEDIATE,
            max_retries=1,
            code_review_enabled=True,
            base_commands={"lint": "ruff check ."},
        )
        factory = _create_factory(
            validation_config=config,
            command_runner=runner,
            fixer_interface=fixer,
            cumulative_review_runner=review_runner,
            base_sha_map=base_sha_map,
        )
        adapters = factory.build_adapters("test-issue")

        result = await adapters.gate_runner.run_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )

        assert result.status == "fail"
        assert result.reason == "max_retries_exhausted"
        # code_review should have run
        assert review_runner.call_count == 1
        assert result.code_review_result is not None
        assert result.code_review_result.ran is True

    @pytest.mark.asyncio
    async def test_code_review_runs_after_successful_retry(self) -> None:
        """code_review runs after successful retry."""
        runner = FakeCommandRunner()
        runner.set_result_sequence(
            "ruff check .",
            [
                FakeCommandResult(command="ruff check .", returncode=1),
                FakeCommandResult(command="ruff check .", returncode=0),
            ],
        )
        fixer = FakeFixerInterface()
        review_runner = FakeCumulativeReviewRunner()
        base_sha_map = {"test-issue": "abc123"}

        config = _make_validation_config(
            commands=(TriggerCommandRef(ref="lint"),),
            failure_mode=FailureMode.REMEDIATE,
            max_retries=2,
            code_review_enabled=True,
            base_commands={"lint": "ruff check ."},
        )
        factory = _create_factory(
            validation_config=config,
            command_runner=runner,
            fixer_interface=fixer,
            cumulative_review_runner=review_runner,
            base_sha_map=base_sha_map,
        )
        adapters = factory.build_adapters("test-issue")

        result = await adapters.gate_runner.run_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )

        assert result.status == "pass"
        # code_review should have run exactly once
        assert review_runner.call_count == 1

    @pytest.mark.asyncio
    async def test_code_review_runs_exactly_once(self) -> None:
        """code_review runs exactly once regardless of retry count."""
        runner = FakeCommandRunner()
        # Multiple failures before success
        runner.set_result_sequence(
            "ruff check .",
            [
                FakeCommandResult(command="ruff check .", returncode=1),
                FakeCommandResult(command="ruff check .", returncode=1),
                FakeCommandResult(command="ruff check .", returncode=0),
            ],
        )
        fixer = FakeFixerInterface()
        review_runner = FakeCumulativeReviewRunner()
        base_sha_map = {"test-issue": "abc123"}

        config = _make_validation_config(
            commands=(TriggerCommandRef(ref="lint"),),
            failure_mode=FailureMode.REMEDIATE,
            max_retries=3,
            code_review_enabled=True,
            base_commands={"lint": "ruff check ."},
        )
        factory = _create_factory(
            validation_config=config,
            command_runner=runner,
            fixer_interface=fixer,
            cumulative_review_runner=review_runner,
            base_sha_map=base_sha_map,
        )
        adapters = factory.build_adapters("test-issue")

        result = await adapters.gate_runner.run_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )

        assert result.status == "pass"
        # code_review runs exactly once after final outcome
        assert review_runner.call_count == 1


class TestRetryStateTracking:
    """Tests for SessionEndRetryState tracking during remediation."""

    @pytest.mark.asyncio
    async def test_retry_state_attempt_incremented_after_fixer(self) -> None:
        """SessionEndRetryState.attempt is incremented after each fixer run."""
        runner = FakeCommandRunner()
        runner.responses["ruff check ."] = FakeCommandResult(
            command="ruff check .", returncode=1
        )
        fixer = FakeFixerInterface()

        config = _make_validation_config(
            commands=(TriggerCommandRef(ref="lint"),),
            failure_mode=FailureMode.REMEDIATE,
            max_retries=2,
            base_commands={"lint": "ruff check ."},
        )
        factory = _create_factory(
            validation_config=config,
            command_runner=runner,
            fixer_interface=fixer,
        )
        adapters = factory.build_adapters("test-issue")

        # Create a retry_state to track attempt increments
        retry_state = SessionEndRetryState()
        assert retry_state.attempt == 1  # Initial attempt

        result = await adapters.gate_runner.run_session_end_check(
            "test-issue", Path("/test/log.txt"), retry_state
        )

        assert result.status == "fail"
        assert result.reason == "max_retries_exhausted"
        # With max_retries=2, fixer runs twice (after attempt 1 and 2)
        # After each fixer run, attempt is incremented:
        # - After fixer 1: attempt = 2
        # - After fixer 2: attempt = 3
        # Final attempt should be 3 (1 initial + 2 retries)
        assert retry_state.attempt == 3
        # Fixer calls should have the correct issue_id
        assert all(call[1] == "test-issue" for call in fixer.calls)
        assert len(fixer.calls) == 2


class TestFixerInterruption:
    """Tests for fixer interruption handling."""

    @pytest.mark.asyncio
    async def test_fixer_interrupted_stops_remediation(self) -> None:
        """Fixer interruption stops remediation and returns interrupted status."""
        runner = FakeCommandRunner()
        runner.responses["ruff check ."] = FakeCommandResult(
            command="ruff check .", returncode=1
        )
        fixer = FakeFixerInterface()
        fixer.set_result_sequence([FixerResult(success=None, interrupted=True)])

        config = _make_validation_config(
            commands=(TriggerCommandRef(ref="lint"),),
            failure_mode=FailureMode.REMEDIATE,
            max_retries=2,
            base_commands={"lint": "ruff check ."},
        )
        factory = _create_factory(
            validation_config=config,
            command_runner=runner,
            fixer_interface=fixer,
        )
        adapters = factory.build_adapters("test-issue")

        result = await adapters.gate_runner.run_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )

        assert result.status == "interrupted"
        # Fixer was called once but returned interrupted
        assert len(fixer.calls) == 1
        # Only initial validation ran
        assert len(runner.commands_run) == 1


class TestMaxRetriesZero:
    """Tests for max_retries=0 behavior."""

    @pytest.mark.asyncio
    async def test_max_retries_zero_falls_back_to_continue_mode(self) -> None:
        """With max_retries=0, falls back to continue mode (command_failed).

        Per spec R9: total validation attempts = 1 + max_retries.
        With max_retries=0, there's exactly 1 attempt (initial) and no retries.
        The initial attempt already failed with command_failed, so we report that.
        """
        runner = FakeCommandRunner()
        runner.responses["ruff check ."] = FakeCommandResult(
            command="ruff check .", returncode=1
        )
        fixer = FakeFixerInterface()

        config = _make_validation_config(
            commands=(TriggerCommandRef(ref="lint"),),
            failure_mode=FailureMode.REMEDIATE,
            max_retries=0,
            base_commands={"lint": "ruff check ."},
        )
        factory = _create_factory(
            validation_config=config,
            command_runner=runner,
            fixer_interface=fixer,
        )
        adapters = factory.build_adapters("test-issue")

        result = await adapters.gate_runner.run_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )

        # Falls back to continue mode: command_failed, not max_retries_exhausted
        assert result.status == "fail"
        assert result.reason == "command_failed"
        # No fixer calls (no retries configured)
        assert len(fixer.calls) == 0
        # Only initial validation
        assert len(runner.commands_run) == 1


class TestNoFixerInterface:
    """Tests when fixer_interface is not provided."""

    @pytest.mark.asyncio
    async def test_no_fixer_interface_falls_back_to_fail(self) -> None:
        """Without fixer_interface, remediate mode falls back to fail status."""
        runner = FakeCommandRunner()
        runner.responses["ruff check ."] = FakeCommandResult(
            command="ruff check .", returncode=1
        )

        config = _make_validation_config(
            commands=(TriggerCommandRef(ref="lint"),),
            failure_mode=FailureMode.REMEDIATE,
            max_retries=2,
            base_commands={"lint": "ruff check ."},
        )
        factory = _create_factory(
            validation_config=config,
            command_runner=runner,
            fixer_interface=None,  # No fixer interface
        )
        adapters = factory.build_adapters("test-issue")

        result = await adapters.gate_runner.run_session_end_check(
            "test-issue", Path("/test/log.txt"), SessionEndRetryState()
        )

        # Should fall back to command_failed without remediation
        assert result.status == "fail"
        assert result.reason == "command_failed"
        # Only initial validation
        assert len(runner.commands_run) == 1

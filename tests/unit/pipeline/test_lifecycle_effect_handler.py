"""Unit tests for LifecycleEffectHandler.

Tests the gate/review side-effect processing logic extracted from AgentSessionRunner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from dataclasses import dataclass, field

from src.domain.lifecycle import (
    ImplementerLifecycle,
    LifecycleConfig,
    LifecycleContext,
    LifecycleState,
)
from src.domain.evidence_check import GateResult
from src.pipeline.agent_session_runner import (
    AgentSessionConfig,
    AgentSessionInput,
    SessionPrompts,
)
from src.pipeline.lifecycle_effect_handler import LifecycleEffectHandler
from src.core.protocols.events import MalaEventSink
from tests.helpers.protocol_stubs import (
    StubGateRunner,
    StubReviewRunner,
    StubSessionLifecycle,
)

if TYPE_CHECKING:
    from pathlib import Path


@dataclass
class FakeReviewIssue:
    """Fake review issue for testing that satisfies ReviewIssue protocol."""

    file: str
    line_start: int
    line_end: int
    priority: int | None
    title: str
    body: str
    reviewer: str


@dataclass
class FakeReviewResult:
    """Fake review result for testing that satisfies ReviewOutcome protocol."""

    passed: bool
    issues: list[FakeReviewIssue] = field(default_factory=list)
    parse_error: str | None = None
    fatal_error: bool = False
    interrupted: bool = False


def make_test_prompts() -> SessionPrompts:
    """Create a SessionPrompts with stub templates for testing."""
    return SessionPrompts(
        gate_followup=(
            "Gate followup: {issue_id} attempt {attempt}/{max_attempts}\n"
            "Failures: {failure_reasons}\n"
            "Commands: {lint_command} {format_command} {typecheck_command} {test_command}"
        ),
        review_followup=(
            "Review followup: {issue_id} attempt {attempt}/{max_attempts}\n"
            "Issues: {review_issues}\n"
            "Commands: {lint_command} {format_command} {typecheck_command} "
            "{custom_commands_section} {test_command}"
        ),
        idle_resume="Continue on issue {issue_id}.",
    )


class FakeEventSink(MalaEventSink):
    """Fake event sink for testing that records all event calls."""

    def __init__(self) -> None:
        self.events: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def _record(self, name: str, *args: object, **kwargs: object) -> None:
        self.events.append((name, args, dict(kwargs)))

    def on_validation_started(self, agent_id: str, issue_id: str | None = None) -> None:
        self._record("on_validation_started", agent_id, issue_id=issue_id)

    def on_validation_result(
        self,
        agent_id: str,
        passed: bool,
        issue_id: str | None = None,
    ) -> None:
        self._record("on_validation_result", agent_id, passed=passed, issue_id=issue_id)

    def on_gate_started(
        self,
        agent_id: str | None,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        self._record(
            "on_gate_started", agent_id, attempt, max_attempts, issue_id=issue_id
        )

    def on_gate_passed(self, agent_id: str | None, issue_id: str | None = None) -> None:
        self._record("on_gate_passed", agent_id, issue_id=issue_id)

    def on_gate_failed(
        self,
        agent_id: str | None,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        self._record(
            "on_gate_failed", agent_id, attempt, max_attempts, issue_id=issue_id
        )

    def on_gate_retry(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        self._record(
            "on_gate_retry",
            agent_id,
            attempt,
            max_attempts,
            issue_id=issue_id,
        )

    def on_gate_result(
        self,
        agent_id: str | None,
        passed: bool,
        failure_reasons: list[str] | None = None,
        issue_id: str | None = None,
    ) -> None:
        self._record(
            "on_gate_result",
            agent_id,
            passed=passed,
            failure_reasons=failure_reasons,
            issue_id=issue_id,
        )

    def on_review_started(
        self,
        agent_id: str,
        attempt: int | None = None,
        max_attempts: int | None = None,
        issue_id: str | None = None,
    ) -> None:
        self._record(
            "on_review_started",
            agent_id,
            attempt=attempt,
            max_attempts=max_attempts,
            issue_id=issue_id,
        )

    def on_review_passed(self, agent_id: str, issue_id: str | None = None) -> None:
        self._record("on_review_passed", agent_id, issue_id=issue_id)

    def on_review_retry(
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        error_count: int | None = None,
        parse_error: str | None = None,
        issue_id: str | None = None,
    ) -> None:
        self._record(
            "on_review_retry",
            agent_id,
            attempt,
            max_attempts,
            error_count=error_count,
            parse_error=parse_error,
            issue_id=issue_id,
        )

    def on_review_skipped_no_progress(self, agent_id: str) -> None:
        self._record("on_review_skipped_no_progress", agent_id)

    def on_warning(self, message: str, agent_id: str | None = None) -> None:
        self._record("on_warning", message, agent_id=agent_id)

    def on_lifecycle_state(self, agent_id: str, state: str) -> None:
        self._record("on_lifecycle_state", agent_id, state)


class TestProcessGateCheck:
    """Unit tests for process_gate_check method."""

    @pytest.fixture
    def tmp_log_path(self, tmp_path: Path) -> Path:
        """Create a temporary log file path."""
        log_path = tmp_path / "session.jsonl"
        log_path.write_text("")
        return log_path

    @pytest.fixture
    def session_config(self, tmp_path: Path) -> AgentSessionConfig:
        """Create a session config for testing."""
        return AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=600,
            prompts=make_test_prompts(),
            max_gate_retries=3,
            max_review_retries=2,
            review_enabled=False,
        )

    @pytest.mark.unit
    def test_emits_validation_started_event(
        self,
        session_config: AgentSessionConfig,
    ) -> None:
        """process_gate_check should emit validation_started event."""
        fake_sink = FakeEventSink()
        handler = LifecycleEffectHandler(
            config=session_config,
            gate_runner=StubGateRunner(),  # type: ignore[arg-type]
            review_runner=StubReviewRunner(),  # type: ignore[arg-type]
            session_lifecycle=StubSessionLifecycle(),  # type: ignore[arg-type]
            event_sink=fake_sink,  # type: ignore[arg-type]
        )

        lifecycle = ImplementerLifecycle(LifecycleConfig(review_enabled=False))
        lifecycle.start()
        lifecycle._state = LifecycleState.RUNNING_GATE
        lifecycle_ctx = LifecycleContext()

        input_data = AgentSessionInput(issue_id="test-gate", prompt="Test")

        handler.process_gate_check(input_data, lifecycle, lifecycle_ctx)

        event_names = [e[0] for e in fake_sink.events]
        assert "on_validation_started" in event_names
        assert "on_gate_started" in event_names


class TestProcessGateEffect:
    """Unit tests for process_gate_effect method."""

    @pytest.fixture
    def session_config(self, tmp_path: Path) -> AgentSessionConfig:
        """Create a session config for testing."""
        return AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=600,
            prompts=make_test_prompts(),
            max_gate_retries=3,
            max_review_retries=2,
            review_enabled=False,
        )

    @pytest.mark.unit
    def test_gate_passed_emits_events(
        self,
        session_config: AgentSessionConfig,
    ) -> None:
        """process_gate_effect should emit passed events on gate pass."""
        fake_sink = FakeEventSink()
        handler = LifecycleEffectHandler(
            config=session_config,
            gate_runner=StubGateRunner(),  # type: ignore[arg-type]
            review_runner=StubReviewRunner(),  # type: ignore[arg-type]
            session_lifecycle=StubSessionLifecycle(),  # type: ignore[arg-type]
            event_sink=fake_sink,  # type: ignore[arg-type]
        )

        lifecycle = ImplementerLifecycle(
            LifecycleConfig(review_enabled=False, session_end_enabled=False)
        )
        lifecycle.start()
        lifecycle._state = LifecycleState.RUNNING_GATE
        lifecycle_ctx = LifecycleContext()

        input_data = AgentSessionInput(issue_id="test-gate", prompt="Test")
        gate_result = GateResult(passed=True, failure_reasons=[], commit_hash="abc123")

        retry_query, _should_break, _trans = handler.process_gate_effect(
            input_data, gate_result, lifecycle, lifecycle_ctx, 1000
        )

        assert retry_query is None
        event_names = [e[0] for e in fake_sink.events]
        assert "on_gate_passed" in event_names
        assert "on_validation_result" in event_names

    @pytest.mark.unit
    def test_gate_failed_with_retry_returns_query(
        self,
        session_config: AgentSessionConfig,
    ) -> None:
        """process_gate_effect should return retry query on failure with retries left."""
        fake_sink = FakeEventSink()
        handler = LifecycleEffectHandler(
            config=session_config,
            gate_runner=StubGateRunner(),  # type: ignore[arg-type]
            review_runner=StubReviewRunner(),  # type: ignore[arg-type]
            session_lifecycle=StubSessionLifecycle(),  # type: ignore[arg-type]
            event_sink=fake_sink,  # type: ignore[arg-type]
        )

        lifecycle = ImplementerLifecycle(LifecycleConfig(max_gate_retries=3))
        lifecycle.start()
        lifecycle._state = LifecycleState.RUNNING_GATE
        lifecycle_ctx = LifecycleContext()

        input_data = AgentSessionInput(issue_id="test-retry", prompt="Test")
        gate_result = GateResult(
            passed=False,
            failure_reasons=["Tests failed"],
            commit_hash=None,
        )

        retry_query, _should_break, _trans = handler.process_gate_effect(
            input_data, gate_result, lifecycle, lifecycle_ctx, 1000
        )

        assert retry_query is not None
        assert "Tests failed" in retry_query
        event_names = [e[0] for e in fake_sink.events]
        assert "on_gate_retry" in event_names
        assert "on_validation_result" in event_names


class TestProcessReviewCheck:
    """Unit tests for process_review_check method."""

    @pytest.fixture
    def session_config(self, tmp_path: Path) -> AgentSessionConfig:
        """Create a session config for testing."""
        return AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=600,
            prompts=make_test_prompts(),
            max_gate_retries=3,
            max_review_retries=3,
            review_enabled=True,
        )

    @pytest.mark.unit
    def test_emits_gate_passed_on_first_attempt(
        self,
        session_config: AgentSessionConfig,
    ) -> None:
        """process_review_check should emit gate_passed on first review attempt."""
        fake_sink = FakeEventSink()
        handler = LifecycleEffectHandler(
            config=session_config,
            gate_runner=StubGateRunner(),  # type: ignore[arg-type]
            review_runner=StubReviewRunner(),  # type: ignore[arg-type]
            session_lifecycle=StubSessionLifecycle(),  # type: ignore[arg-type]
            event_sink=fake_sink,  # type: ignore[arg-type]
        )

        lifecycle_ctx = LifecycleContext()
        lifecycle_ctx.retry_state.review_attempt = 1

        input_data = AgentSessionInput(issue_id="test-review", prompt="Test")

        handler.process_review_check(input_data, lifecycle_ctx)

        event_names = [e[0] for e in fake_sink.events]
        assert "on_gate_passed" in event_names
        assert "on_validation_result" in event_names


class TestProcessReviewEffect:
    """Unit tests for process_review_effect method."""

    @pytest.fixture
    def session_config(self, tmp_path: Path) -> AgentSessionConfig:
        """Create a session config for testing."""
        return AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=600,
            prompts=make_test_prompts(),
            max_gate_retries=3,
            max_review_retries=3,
            review_enabled=True,
        )

    @pytest.mark.unit
    def test_review_passed_returns_no_retry(
        self,
        session_config: AgentSessionConfig,
    ) -> None:
        """process_review_effect should return no retry on pass."""
        fake_sink = FakeEventSink()
        handler = LifecycleEffectHandler(
            config=session_config,
            gate_runner=StubGateRunner(),  # type: ignore[arg-type]
            review_runner=StubReviewRunner(),  # type: ignore[arg-type]
            session_lifecycle=StubSessionLifecycle(),  # type: ignore[arg-type]
            event_sink=fake_sink,  # type: ignore[arg-type]
        )

        lifecycle = ImplementerLifecycle(
            LifecycleConfig(review_enabled=True, max_review_retries=3)
        )
        lifecycle.start()
        lifecycle._state = LifecycleState.RUNNING_REVIEW
        lifecycle_ctx = LifecycleContext()
        lifecycle_ctx.retry_state.review_attempt = 1

        input_data = AgentSessionInput(issue_id="test-review", prompt="Test")
        review_result = FakeReviewResult(passed=True, issues=[])

        effect = handler.process_review_effect(
            input_data, review_result, lifecycle, lifecycle_ctx, 1000, None
        )

        assert effect.pending_query is None
        event_names = [e[0] for e in fake_sink.events]
        assert "on_review_passed" in event_names

    @pytest.mark.unit
    def test_review_failed_with_retry_returns_query(
        self,
        session_config: AgentSessionConfig,
    ) -> None:
        """process_review_effect should return retry query on failure with retries left."""
        fake_sink = FakeEventSink()
        handler = LifecycleEffectHandler(
            config=session_config,
            gate_runner=StubGateRunner(),  # type: ignore[arg-type]
            review_runner=StubReviewRunner(),  # type: ignore[arg-type]
            session_lifecycle=StubSessionLifecycle(),  # type: ignore[arg-type]
            event_sink=fake_sink,  # type: ignore[arg-type]
        )

        lifecycle = ImplementerLifecycle(
            LifecycleConfig(review_enabled=True, max_review_retries=3)
        )
        lifecycle.start()
        lifecycle._state = LifecycleState.RUNNING_REVIEW
        lifecycle_ctx = LifecycleContext()
        lifecycle_ctx.retry_state.review_attempt = 1

        input_data = AgentSessionInput(issue_id="test-rev-retry", prompt="Test")
        review_result = FakeReviewResult(
            passed=False,
            issues=[
                FakeReviewIssue(
                    title="Bug found",
                    body="A bug was found",
                    priority=1,
                    file="test.py",
                    line_start=10,
                    line_end=15,
                    reviewer="gemini",
                ),
            ],
        )

        effect = handler.process_review_effect(
            input_data, review_result, lifecycle, lifecycle_ctx, 1000, None
        )

        assert effect.pending_query is not None
        assert "Bug found" in effect.pending_query or "test.py" in effect.pending_query
        event_names = [e[0] for e in fake_sink.events]
        assert "on_review_retry" in event_names


class TestCheckReviewNoProgress:
    """Unit tests for check_review_no_progress method."""

    @pytest.fixture
    def tmp_log_path(self, tmp_path: Path) -> Path:
        """Create a temporary log file path."""
        log_path = tmp_path / "session.jsonl"
        log_path.write_text("")
        return log_path

    @pytest.fixture
    def session_config(self, tmp_path: Path) -> AgentSessionConfig:
        """Create a session config for testing."""
        return AgentSessionConfig(
            repo_path=tmp_path,
            timeout_seconds=600,
            prompts=make_test_prompts(),
            max_gate_retries=3,
            max_review_retries=3,
            review_enabled=True,
        )

    @pytest.mark.unit
    def test_skips_on_no_progress(
        self,
        session_config: AgentSessionConfig,
        tmp_log_path: Path,
    ) -> None:
        """check_review_no_progress should return result when no progress detected."""
        fake_sink = FakeEventSink()

        # Create a review runner that detects no progress
        review_runner = StubReviewRunner(no_progress_result=True)

        handler = LifecycleEffectHandler(
            config=session_config,
            gate_runner=StubGateRunner(),  # type: ignore[arg-type]
            review_runner=review_runner,  # type: ignore[arg-type]
            session_lifecycle=StubSessionLifecycle(),  # type: ignore[arg-type]
            event_sink=fake_sink,  # type: ignore[arg-type]
        )

        lifecycle = ImplementerLifecycle(
            LifecycleConfig(review_enabled=True, max_review_retries=3)
        )
        lifecycle.start()
        lifecycle._state = LifecycleState.RUNNING_REVIEW
        lifecycle_ctx = LifecycleContext()
        lifecycle_ctx.retry_state.review_attempt = 2  # Not first attempt
        lifecycle_ctx.last_gate_result = GateResult(
            passed=True, failure_reasons=[], commit_hash="same-commit"
        )

        input_data = AgentSessionInput(issue_id="test-no-progress", prompt="Test")

        result = handler.check_review_no_progress(
            input_data, tmp_log_path, lifecycle, lifecycle_ctx, None
        )

        assert result is not None
        assert result.pending_query is None
        event_names = [e[0] for e in fake_sink.events]
        assert "on_review_skipped_no_progress" in event_names

    @pytest.mark.unit
    def test_returns_none_on_first_attempt(
        self,
        session_config: AgentSessionConfig,
        tmp_log_path: Path,
    ) -> None:
        """check_review_no_progress should return None on first attempt."""
        # Review runner would detect no progress, but should be skipped on first attempt
        review_runner = StubReviewRunner(no_progress_result=True)

        handler = LifecycleEffectHandler(
            config=session_config,
            gate_runner=StubGateRunner(),  # type: ignore[arg-type]
            review_runner=review_runner,  # type: ignore[arg-type]
            session_lifecycle=StubSessionLifecycle(),  # type: ignore[arg-type]
            event_sink=None,
        )

        lifecycle = ImplementerLifecycle(
            LifecycleConfig(review_enabled=True, max_review_retries=3)
        )
        lifecycle.start()
        lifecycle_ctx = LifecycleContext()
        lifecycle_ctx.retry_state.review_attempt = 1  # First attempt

        input_data = AgentSessionInput(issue_id="test", prompt="Test")

        result = handler.check_review_no_progress(
            input_data, tmp_log_path, lifecycle, lifecycle_ctx, None
        )

        assert result is None

    @pytest.mark.unit
    def test_returns_none_when_progress_detected(
        self,
        session_config: AgentSessionConfig,
        tmp_log_path: Path,
    ) -> None:
        """check_review_no_progress should return None when progress is detected."""
        # Review runner detects progress (no_progress_result=False)
        review_runner = StubReviewRunner(no_progress_result=False)

        handler = LifecycleEffectHandler(
            config=session_config,
            gate_runner=StubGateRunner(),  # type: ignore[arg-type]
            review_runner=review_runner,  # type: ignore[arg-type]
            session_lifecycle=StubSessionLifecycle(),  # type: ignore[arg-type]
            event_sink=None,
        )

        lifecycle = ImplementerLifecycle(
            LifecycleConfig(review_enabled=True, max_review_retries=3)
        )
        lifecycle.start()
        lifecycle_ctx = LifecycleContext()
        lifecycle_ctx.retry_state.review_attempt = 2

        input_data = AgentSessionInput(issue_id="test", prompt="Test")

        result = handler.check_review_no_progress(
            input_data, tmp_log_path, lifecycle, lifecycle_ctx, None
        )

        assert result is None

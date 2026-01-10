"""Integration test for run lifecycle restructure.

This test verifies the gate -> session_end -> review flow:
- Effect.RUN_SESSION_END is handled in the lifecycle loop
- Session_end callback is invoked with correct parameters
- Event ordering: session_end_started after gate_passed, before review_start

These tests exercise the AgentSessionRunner lifecycle loop directly using
fake SDK clients, similar to tests/integration/pipeline/test_agent_session_runner.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from claude_agent_sdk import ResultMessage

from src.domain.evidence_check import GateResult
from src.pipeline.agent_session_runner import (
    AgentSessionConfig,
    AgentSessionInput,
    AgentSessionRunner,
    SessionCallbacks,
    SessionPrompts,
)
from src.pipeline.session_end_result import SessionEndResult
from tests.fakes.sdk_client import FakeSDKClient, FakeSDKClientFactory

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.core.protocols import McpServerFactory
    from src.domain.lifecycle import RetryState
    from src.pipeline.session_end_result import SessionEndRetryState


def make_noop_mcp_factory() -> McpServerFactory:
    """Create a no-op MCP server factory for tests (no locking).

    Returns a factory that returns empty servers, disabling locking.
    This is used for tests that don't need the locking MCP server.
    """

    def factory(
        agent_id: str, repo_path: Path, emit_lock_event: Callable | None
    ) -> dict[str, object]:
        return {}

    return cast("McpServerFactory", factory)


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


def make_result_message(
    session_id: str = "test-session-123",
    result: str | None = "Test completed successfully",
) -> ResultMessage:
    """Create a ResultMessage with the given fields."""
    return ResultMessage(
        subtype="result",
        duration_ms=100,
        duration_api_ms=50,
        is_error=False,
        num_turns=1,
        session_id=session_id,
        result=result,
    )


@pytest.fixture
def tmp_log_path(tmp_path: Path) -> Path:
    """Create a temporary log file path."""
    log_path = tmp_path / "session.log"
    log_path.write_text("Agent log content\n")
    return log_path


@pytest.fixture
def session_config(tmp_path: Path) -> AgentSessionConfig:
    """Create an AgentSessionConfig for testing."""
    return AgentSessionConfig(
        repo_path=tmp_path,
        timeout_seconds=60,
        prompts=make_test_prompts(),
        max_gate_retries=3,
        max_review_retries=3,
        review_enabled=False,  # Disable review to isolate session_end testing
        mcp_server_factory=make_noop_mcp_factory(),
    )


@pytest.fixture
def fake_client() -> FakeSDKClient:
    """Create a basic FakeSDKClient with a result message."""
    return FakeSDKClient(
        messages=[],
        result_message=make_result_message(),
    )


@pytest.fixture
def fake_factory(fake_client: FakeSDKClient) -> FakeSDKClientFactory:
    """Create a factory for the fake client."""
    return FakeSDKClientFactory(fake_client)


class FakeEventSink:
    """Fake event sink that tracks lifecycle events in order."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def on_gate_passed(
        self, agent_id: str | None, issue_id: str | None = None, **kwargs: object
    ) -> None:
        # issue_id is passed as keyword arg, fallback to agent_id if not provided
        eid = issue_id or agent_id
        self.events.append(f"gate_passed:{eid}")

    def on_session_end_started(self, issue_id: str) -> None:
        self.events.append(f"session_end_started:{issue_id}")

    def on_session_end_completed(self, issue_id: str, result: str) -> None:
        self.events.append(f"session_end_completed:{issue_id}:{result}")

    def on_session_end_skipped(self, issue_id: str, reason: str) -> None:
        self.events.append(f"session_end_skipped:{issue_id}:{reason}")

    def on_review_started(
        self, issue_id: str, attempt: int, max_attempts: int, **kwargs: object
    ) -> None:
        self.events.append(f"review_started:{issue_id}")

    # Stub out other event methods that may be called
    def __getattr__(self, name: str) -> object:
        # Return a no-op function for any other event method
        def noop(*args: object, **kwargs: object) -> None:
            pass

        return noop


@pytest.mark.asyncio
@pytest.mark.integration
async def test_session_end_invoked_after_gate_passes(
    session_config: AgentSessionConfig,
    fake_factory: FakeSDKClientFactory,
    tmp_log_path: Path,
) -> None:
    """Integration: session_end callback is invoked after gate passes.

    This test verifies:
    1. Gate passes
    2. session_end_started event is emitted
    3. session_end callback is invoked with correct parameters
    4. session_end_skipped event is emitted (stub callback returns skipped)
    5. Session completes successfully
    """
    # Track callback invocations
    session_end_calls: list[tuple[str, Path, SessionEndRetryState]] = []
    event_sink = FakeEventSink()

    def get_log_path(session_id: str) -> Path:
        return tmp_log_path

    async def on_gate_check(
        issue_id: str, log_path: Path, retry_state: RetryState
    ) -> tuple[GateResult, int]:
        return (
            GateResult(passed=True, failure_reasons=[], commit_hash="abc123"),
            1000,
        )

    async def on_session_end_check(
        issue_id: str, log_path: Path, retry_state: SessionEndRetryState
    ) -> SessionEndResult:
        session_end_calls.append((issue_id, log_path, retry_state))
        return SessionEndResult(status="skipped", reason="not_implemented")

    callbacks = SessionCallbacks(
        get_log_path=get_log_path,
        on_gate_check=on_gate_check,
        on_session_end_check=on_session_end_check,
    )

    runner = AgentSessionRunner(
        config=session_config,
        callbacks=callbacks,
        sdk_client_factory=fake_factory,
        event_sink=cast("Any", event_sink),
    )

    input = AgentSessionInput(
        issue_id="test-123",
        prompt="Test prompt",
    )

    output = await runner.run_session(input)

    # Session should complete successfully
    assert output.success is True

    # session_end callback should have been invoked once
    assert len(session_end_calls) == 1
    call_issue_id, call_log_path, call_retry_state = session_end_calls[0]
    assert call_issue_id == "test-123"
    assert call_log_path == tmp_log_path
    assert call_retry_state.attempt == 1

    # Verify event ordering
    assert "gate_passed:test-123" in event_sink.events
    assert "session_end_started:test-123" in event_sink.events
    assert "session_end_skipped:test-123:not_implemented" in event_sink.events

    # session_end events should come after gate_passed
    gate_idx = event_sink.events.index("gate_passed:test-123")
    session_end_started_idx = event_sink.events.index("session_end_started:test-123")
    session_end_skipped_idx = event_sink.events.index(
        "session_end_skipped:test-123:not_implemented"
    )

    assert session_end_started_idx > gate_idx
    assert session_end_skipped_idx > session_end_started_idx


@pytest.mark.asyncio
@pytest.mark.integration
async def test_session_end_completed_event_on_pass(
    session_config: AgentSessionConfig,
    fake_factory: FakeSDKClientFactory,
    tmp_log_path: Path,
) -> None:
    """Integration: session_end_completed event is emitted when session_end passes.

    When the session_end callback returns status="pass", the handler should
    emit on_session_end_completed with "pass" result.
    """
    event_sink = FakeEventSink()

    def get_log_path(session_id: str) -> Path:
        return tmp_log_path

    async def on_gate_check(
        issue_id: str, log_path: Path, retry_state: RetryState
    ) -> tuple[GateResult, int]:
        return (
            GateResult(passed=True, failure_reasons=[], commit_hash="abc123"),
            1000,
        )

    async def on_session_end_check(
        issue_id: str, log_path: Path, retry_state: SessionEndRetryState
    ) -> SessionEndResult:
        return SessionEndResult(status="pass")

    callbacks = SessionCallbacks(
        get_log_path=get_log_path,
        on_gate_check=on_gate_check,
        on_session_end_check=on_session_end_check,
    )

    runner = AgentSessionRunner(
        config=session_config,
        callbacks=callbacks,
        sdk_client_factory=fake_factory,
        event_sink=cast("Any", event_sink),
    )

    input = AgentSessionInput(
        issue_id="test-456",
        prompt="Test prompt",
    )

    output = await runner.run_session(input)

    assert output.success is True
    assert "session_end_completed:test-456:pass" in event_sink.events


@pytest.mark.asyncio
@pytest.mark.integration
async def test_session_end_not_invoked_when_gate_fails(
    session_config: AgentSessionConfig,
    fake_factory: FakeSDKClientFactory,
    tmp_log_path: Path,
) -> None:
    """Integration: session_end is not invoked when gate fails.

    When the gate check fails (and no retries succeed), the lifecycle should
    not transition to RUN_SESSION_END. The session_end callback should not
    be invoked.
    """
    session_end_calls: list[str] = []
    event_sink = FakeEventSink()

    def get_log_path(session_id: str) -> Path:
        return tmp_log_path

    async def on_gate_check(
        issue_id: str, log_path: Path, retry_state: RetryState
    ) -> tuple[GateResult, int]:
        # Gate always fails
        return (
            GateResult(passed=False, failure_reasons=["lint failed"], commit_hash=None),
            1000,
        )

    async def on_session_end_check(
        issue_id: str, log_path: Path, retry_state: SessionEndRetryState
    ) -> SessionEndResult:
        session_end_calls.append(issue_id)
        return SessionEndResult(status="skipped", reason="not_implemented")

    # Use 0 retries so the gate fails immediately
    config_no_retries = AgentSessionConfig(
        repo_path=session_config.repo_path,
        timeout_seconds=session_config.timeout_seconds,
        prompts=session_config.prompts,
        max_gate_retries=0,
        max_review_retries=0,
        review_enabled=False,
        mcp_server_factory=make_noop_mcp_factory(),
    )

    callbacks = SessionCallbacks(
        get_log_path=get_log_path,
        on_gate_check=on_gate_check,
        on_session_end_check=on_session_end_check,
    )

    runner = AgentSessionRunner(
        config=config_no_retries,
        callbacks=callbacks,
        sdk_client_factory=fake_factory,
        event_sink=cast("Any", event_sink),
    )

    input = AgentSessionInput(
        issue_id="test-789",
        prompt="Test prompt",
    )

    output = await runner.run_session(input)

    # Session should fail due to gate failure
    assert output.success is False

    # session_end callback should NOT have been invoked
    assert len(session_end_calls) == 0

    # session_end events should NOT appear
    assert not any("session_end" in e for e in event_sink.events)

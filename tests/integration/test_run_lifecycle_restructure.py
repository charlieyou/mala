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
    SessionPrompts,
)
from tests.helpers.protocol_stubs import (
    StubGateRunner,
    StubReviewRunner,
    StubSessionLifecycle,
)
from src.core.protocols.events import MalaEventSink
from src.core.session_end_result import SessionEndResult
from tests.fakes.sdk_client import FakeSDKClient, FakeSDKClientFactory

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.core.protocols.sdk import McpServerFactory
    from src.core.session_end_result import SessionEndRetryState
    from src.domain.lifecycle import RetryState


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


class FakeEventSink(MalaEventSink):
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
        self,
        agent_id: str,
        attempt: int,
        max_attempts: int,
        issue_id: str | None = None,
    ) -> None:
        # issue_id is passed as keyword arg, fallback to agent_id if not provided
        eid = issue_id or agent_id
        self.events.append(f"review_started:{eid}")

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

    runner = AgentSessionRunner(
        config=session_config,
        sdk_client_factory=fake_factory,
        gate_runner=StubGateRunner(
            on_gate_check=on_gate_check, on_session_end_check=on_session_end_check
        ),  # type: ignore[arg-type]
        review_runner=StubReviewRunner(),  # type: ignore[arg-type]
        session_lifecycle=StubSessionLifecycle(on_get_log_path=get_log_path),  # type: ignore[arg-type]
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

    runner = AgentSessionRunner(
        config=session_config,
        sdk_client_factory=fake_factory,
        gate_runner=StubGateRunner(
            on_gate_check=on_gate_check, on_session_end_check=on_session_end_check
        ),  # type: ignore[arg-type]
        review_runner=StubReviewRunner(),  # type: ignore[arg-type]
        session_lifecycle=StubSessionLifecycle(on_get_log_path=get_log_path),  # type: ignore[arg-type]
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

    runner = AgentSessionRunner(
        config=config_no_retries,
        sdk_client_factory=fake_factory,
        gate_runner=StubGateRunner(
            on_gate_check=on_gate_check, on_session_end_check=on_session_end_check
        ),  # type: ignore[arg-type]
        review_runner=StubReviewRunner(),  # type: ignore[arg-type]
        session_lifecycle=StubSessionLifecycle(on_get_log_path=get_log_path),  # type: ignore[arg-type]
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


@pytest.mark.asyncio
@pytest.mark.integration
async def test_session_end_timeout_scenario(
    session_config: AgentSessionConfig,
    fake_factory: FakeSDKClientFactory,
    tmp_log_path: Path,
) -> None:
    """Scenario: lifecycle handles session_end timeout result correctly.

    Per spec R10:
    - On timeout, status="timeout" with empty commands array
    - Issue proceeds to review with SessionEndResult containing timeout status

    This integration test verifies that the AgentSessionRunner lifecycle loop
    correctly handles a timeout result from session_end and emits the expected
    events in the correct order. The actual asyncio.timeout() behavior is
    tested in unit tests (test_session_end_callback.py::TestTimeoutBehavior).
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
        # Simulate timeout: return status="timeout" with empty commands
        # This simulates what happens when asyncio.timeout() fires
        return SessionEndResult(
            status="timeout",
            reason="session_end_timeout",
            commands=[],  # Per spec R10: partial results discarded on timeout
        )

    runner = AgentSessionRunner(
        config=session_config,
        sdk_client_factory=fake_factory,
        gate_runner=StubGateRunner(
            on_gate_check=on_gate_check, on_session_end_check=on_session_end_check
        ),  # type: ignore[arg-type]
        review_runner=StubReviewRunner(),  # type: ignore[arg-type]
        session_lifecycle=StubSessionLifecycle(on_get_log_path=get_log_path),  # type: ignore[arg-type]
        event_sink=cast("Any", event_sink),
    )

    input = AgentSessionInput(
        issue_id="test-timeout",
        prompt="Test prompt",
    )

    output = await runner.run_session(input)

    # Session should complete successfully despite timeout
    # (timeout proceeds to review per spec)
    assert output.success is True

    # Verify event ordering: gate_passed -> session_end_started -> session_end_completed:timeout
    assert "gate_passed:test-timeout" in event_sink.events
    assert "session_end_started:test-timeout" in event_sink.events
    assert "session_end_completed:test-timeout:timeout" in event_sink.events

    gate_idx = event_sink.events.index("gate_passed:test-timeout")
    session_end_started_idx = event_sink.events.index(
        "session_end_started:test-timeout"
    )
    session_end_completed_idx = event_sink.events.index(
        "session_end_completed:test-timeout:timeout"
    )

    # Verify correct ordering
    assert session_end_started_idx > gate_idx
    assert session_end_completed_idx > session_end_started_idx


@pytest.mark.asyncio
@pytest.mark.integration
async def test_session_end_interrupt_scenario(
    session_config: AgentSessionConfig,
    fake_factory: FakeSDKClientFactory,
    tmp_log_path: Path,
) -> None:
    """Scenario: SIGINT during session_end command execution.

    Per spec R10:
    - On SIGINT, complete current command and return status="interrupted"
    - Partial command results discarded (commands list is empty)
    - Issue proceeds to review with SessionEndResult containing interrupted status

    This test verifies that when SIGINT is received during session_end,
    the lifecycle correctly handles the interrupted result with empty
    command results (per spec R10) and continues to the review phase.
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
        # Simulate interrupt: per spec R10 partial results are discarded
        return SessionEndResult(
            status="interrupted",
            reason="SIGINT received",
            commands=[],
        )

    runner = AgentSessionRunner(
        config=session_config,
        sdk_client_factory=fake_factory,
        gate_runner=StubGateRunner(
            on_gate_check=on_gate_check, on_session_end_check=on_session_end_check
        ),  # type: ignore[arg-type]
        review_runner=StubReviewRunner(),  # type: ignore[arg-type]
        session_lifecycle=StubSessionLifecycle(on_get_log_path=get_log_path),  # type: ignore[arg-type]
        event_sink=cast("Any", event_sink),
    )

    input = AgentSessionInput(
        issue_id="test-interrupt",
        prompt="Test prompt",
    )

    output = await runner.run_session(input)

    # Session should complete (interrupted proceeds to review per spec)
    assert output.success is True

    # Verify event ordering: gate_passed -> session_end_started -> session_end_completed:interrupted
    assert "gate_passed:test-interrupt" in event_sink.events
    assert "session_end_started:test-interrupt" in event_sink.events
    assert "session_end_completed:test-interrupt:interrupted" in event_sink.events

    gate_idx = event_sink.events.index("gate_passed:test-interrupt")
    session_end_started_idx = event_sink.events.index(
        "session_end_started:test-interrupt"
    )
    session_end_completed_idx = event_sink.events.index(
        "session_end_completed:test-interrupt:interrupted"
    )

    # Verify correct ordering
    assert session_end_started_idx > gate_idx
    assert session_end_completed_idx > session_end_started_idx


@pytest.mark.asyncio
@pytest.mark.integration
async def test_session_end_timeout_proceeds_to_review_with_correct_result(
    tmp_path: Path,
    fake_factory: FakeSDKClientFactory,
    tmp_log_path: Path,
) -> None:
    """Verify timeout scenario proceeds to review with explicit status.

    Per spec R10: On timeout/interrupt, proceed to review with explicit status.
    This test enables review to verify the SessionEndResult is available and
    the lifecycle correctly transitions from session_end to review.
    """
    from dataclasses import dataclass as dc
    from datetime import UTC, datetime
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from collections.abc import Sequence

        from src.core.protocols.review import ReviewIssueProtocol

    @dc
    class FakeReviewOutcome:
        """Fake review outcome for testing."""

        passed: bool = True
        parse_error: str | None = None
        fatal_error: bool = False
        issues: list[object] = None  # type: ignore[assignment]
        interrupted: bool = False

        def __post_init__(self) -> None:
            if self.issues is None:
                self.issues = []

    event_sink = FakeEventSink()
    session_end_results: list[SessionEndResult] = []
    review_check_called = False

    # Enable review to verify integration
    config_with_review = AgentSessionConfig(
        repo_path=tmp_path,
        timeout_seconds=60,
        prompts=make_test_prompts(),
        max_gate_retries=3,
        max_review_retries=0,  # Don't retry review
        review_enabled=True,  # Enable review
        mcp_server_factory=make_noop_mcp_factory(),
    )

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
        result = SessionEndResult(
            status="timeout",
            started_at=datetime.now(UTC),
            finished_at=datetime.now(UTC),
            reason="session_end_timeout",
            commands=[],
        )
        session_end_results.append(result)
        return result

    async def on_review_check(
        issue_id: str,
        issue_description: str | None,
        session_id: str | None,
        retry_state: RetryState,
        author_context: str | None,
        review_issues: Sequence[ReviewIssueProtocol] | None,
        session_end_result: SessionEndResult | None,
    ) -> FakeReviewOutcome:
        nonlocal review_check_called
        review_check_called = True
        return FakeReviewOutcome(passed=True)

    runner = AgentSessionRunner(
        config=config_with_review,
        sdk_client_factory=fake_factory,
        gate_runner=StubGateRunner(
            on_gate_check=on_gate_check, on_session_end_check=on_session_end_check
        ),  # type: ignore[arg-type]
        review_runner=StubReviewRunner(on_review=on_review_check),  # type: ignore[arg-type]
        session_lifecycle=StubSessionLifecycle(on_get_log_path=get_log_path),  # type: ignore[arg-type]
        event_sink=cast("Any", event_sink),
    )

    input = AgentSessionInput(
        issue_id="test-timeout-review",
        prompt="Test prompt",
    )

    await runner.run_session(input)

    # Key verification: session_end was invoked and review was started
    assert len(session_end_results) == 1
    assert session_end_results[0].status == "timeout"
    assert session_end_results[0].commands == []

    # Verify session_end completed event with timeout status
    assert "session_end_completed:test-timeout-review:timeout" in event_sink.events

    # Verify review was invoked after session_end
    assert review_check_called, "on_review_check was not called"
    assert "review_started:test-timeout-review" in event_sink.events

    # Verify event ordering: session_end_completed comes before review_started
    session_end_idx = event_sink.events.index(
        "session_end_completed:test-timeout-review:timeout"
    )
    review_idx = event_sink.events.index("review_started:test-timeout-review")
    assert review_idx > session_end_idx


# =============================================================================
# T017: Run abort, mixed outcomes, and fire_on scenarios
# =============================================================================


class FakeEventSinkWithRunEnd(FakeEventSink):
    """Extended fake event sink that tracks run_end events."""

    def on_run_end_skipped(self, reason: str) -> None:
        self.events.append(f"run_end_skipped:{reason}")

    def on_run_end_started(self, success_count: int, total_count: int) -> None:
        self.events.append(f"run_end_started:{success_count}/{total_count}")

    def on_run_end_completed(self, result: str) -> None:
        self.events.append(f"run_end_completed:{result}")


@pytest.mark.asyncio
@pytest.mark.integration
async def test_mixed_outcomes_both_proceed_to_review(
    tmp_path: Path,
    fake_factory: FakeSDKClientFactory,
    tmp_log_path: Path,
) -> None:
    """Scenario: issues with pass/fail session_end both proceed to review.

    Per spec R5 and R9:
    - Session_end failure does not block review (informational only)
    - Both passing and failing issues proceed to review with their results
    """
    from dataclasses import dataclass as dc
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from collections.abc import Sequence

        from src.core.protocols.review import ReviewIssueProtocol

    @dc
    class FakeReviewOutcome:
        """Fake review outcome for testing."""

        passed: bool = True
        parse_error: str | None = None
        fatal_error: bool = False
        issues: list[object] = None  # type: ignore[assignment]
        interrupted: bool = False

        def __post_init__(self) -> None:
            if self.issues is None:
                self.issues = []

    event_sink = FakeEventSinkWithRunEnd()
    session_end_results: dict[str, SessionEndResult] = {}
    review_calls: list[tuple[str, SessionEndResult | None]] = []

    # Enable review
    config_with_review = AgentSessionConfig(
        repo_path=tmp_path,
        timeout_seconds=60,
        prompts=make_test_prompts(),
        max_gate_retries=0,
        max_review_retries=0,
        review_enabled=True,
        mcp_server_factory=make_noop_mcp_factory(),
    )

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
        # issue-pass returns pass, issue-fail returns fail
        if "pass" in issue_id:
            result = SessionEndResult(status="pass")
        else:
            result = SessionEndResult(status="fail", reason="validation_failed")
        session_end_results[issue_id] = result
        return result

    async def on_review_check(
        issue_id: str,
        issue_description: str | None,
        session_id: str | None,
        retry_state: RetryState,
        author_context: str | None,
        review_issues: Sequence[ReviewIssueProtocol] | None,
        session_end_result: SessionEndResult | None,
    ) -> FakeReviewOutcome:
        review_calls.append((issue_id, session_end_result))
        return FakeReviewOutcome(passed=True)

    runner = AgentSessionRunner(
        config=config_with_review,
        sdk_client_factory=fake_factory,
        gate_runner=StubGateRunner(
            on_gate_check=on_gate_check, on_session_end_check=on_session_end_check
        ),  # type: ignore[arg-type]
        review_runner=StubReviewRunner(on_review=on_review_check),  # type: ignore[arg-type]
        session_lifecycle=StubSessionLifecycle(on_get_log_path=get_log_path),  # type: ignore[arg-type]
        event_sink=cast("Any", event_sink),
    )

    # Run passing issue
    input_pass = AgentSessionInput(
        issue_id="issue-pass",
        prompt="Test prompt",
    )
    output_pass = await runner.run_session(input_pass)

    # Run failing issue
    input_fail = AgentSessionInput(
        issue_id="issue-fail",
        prompt="Test prompt",
    )
    output_fail = await runner.run_session(input_fail)

    # Both should complete (session_end failure doesn't fail the issue)
    assert output_pass.success is True
    assert output_fail.success is True

    # Both should have proceeded to review
    assert len(review_calls) == 2
    review_ids = [call[0] for call in review_calls]
    assert "issue-pass" in review_ids
    assert "issue-fail" in review_ids

    # Verify session_end results were passed to review
    for issue_id, session_end_result in review_calls:
        assert session_end_result is not None
        if "pass" in issue_id:
            assert session_end_result.status == "pass"
        else:
            assert session_end_result.status == "fail"

    # Verify events for both issues
    assert "session_end_completed:issue-pass:pass" in event_sink.events
    assert "session_end_completed:issue-fail:fail" in event_sink.events
    assert "review_started:issue-pass" in event_sink.events
    assert "review_started:issue-fail" in event_sink.events


# =============================================================================
# R7 fire_on Truth Table Tests
# =============================================================================
#
# Per spec R7, test all combinations:
#
# | success_count | failure_count | fire_on: success | fire_on: failure | fire_on: both |
# |---------------|---------------|------------------|------------------|---------------|
# | 0             | N (>0)        | skip             | fire             | fire          |
# | N (>0)        | 0             | fire             | skip             | fire          |
# | N (>0)        | M (>0)        | fire             | fire             | fire          |
#
# These tests verify the fire_on truth table at the integration level
# by exercising the actual _fire_run_end_trigger logic.


class TestFireOnTruthTable:
    """R7 fire_on truth table tests for run_end trigger.

    Tests all 6 combinations from the spec truth table:
    - 0 success, N failures (all fail)
    - N success, 0 failures (all success)
    - N success, M failures (mixed)

    For each row, tests fire_on: success, failure, and both.
    """

    @pytest.fixture
    def make_validation_config(self) -> Callable:
        """Factory for creating validation config with fire_on setting."""
        from src.domain.validation.config import (
            FailureMode,
            RunEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        def _make(fire_on: object) -> object:
            triggers = ValidationTriggersConfig(
                run_end=RunEndTriggerConfig(
                    failure_mode=FailureMode.CONTINUE,
                    commands=(),
                    fire_on=fire_on,  # type: ignore[arg-type]
                )
            )
            return ValidationConfig(validation_triggers=triggers)

        return _make

    # Row 1: 0 success, N failures (all issues failed)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_all_failures_fire_on_success_skips(
        self,
        tmp_path: Path,
        make_validation_config: Callable,
        make_orchestrator: Callable,
    ) -> None:
        """fire_on=success with all failures → run_end skipped.

        Truth table row: success_count=0, failure_count=N(>0), fire_on=success → skip
        """
        from unittest.mock import AsyncMock, MagicMock

        from src.domain.validation.config import FireOn

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
        )
        orchestrator._validation_config = make_validation_config(FireOn.SUCCESS)

        # Mock run_coordinator to track calls
        orchestrator.run_coordinator.queue_trigger_validation = MagicMock()  # type: ignore[method-assign]
        orchestrator.run_coordinator.run_trigger_validation = AsyncMock()  # type: ignore[method-assign]

        # 0 success, 3 failures
        await orchestrator._fire_run_end_trigger(success_count=0, total_count=3)

        # Should NOT queue trigger
        orchestrator.run_coordinator.queue_trigger_validation.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_all_failures_fire_on_failure_fires(
        self,
        tmp_path: Path,
        make_validation_config: Callable,
        make_orchestrator: Callable,
    ) -> None:
        """fire_on=failure with all failures → run_end fires.

        Truth table row: success_count=0, failure_count=N(>0), fire_on=failure → fire
        """
        from unittest.mock import AsyncMock, MagicMock

        from src.domain.validation.config import FireOn, TriggerType

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
        )
        orchestrator._validation_config = make_validation_config(FireOn.FAILURE)

        orchestrator.run_coordinator.queue_trigger_validation = MagicMock()  # type: ignore[method-assign]
        orchestrator.run_coordinator.run_trigger_validation = AsyncMock(  # type: ignore[method-assign]
            return_value=MagicMock(status="passed")
        )

        # 0 success, 3 failures
        await orchestrator._fire_run_end_trigger(success_count=0, total_count=3)

        # Should queue trigger
        orchestrator.run_coordinator.queue_trigger_validation.assert_called_once_with(
            TriggerType.RUN_END,
            {"success_count": 0, "total_count": 3},
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_all_failures_fire_on_both_fires(
        self,
        tmp_path: Path,
        make_validation_config: Callable,
        make_orchestrator: Callable,
    ) -> None:
        """fire_on=both with all failures → run_end fires.

        Truth table row: success_count=0, failure_count=N(>0), fire_on=both → fire
        """
        from unittest.mock import AsyncMock, MagicMock

        from src.domain.validation.config import FireOn, TriggerType

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
        )
        orchestrator._validation_config = make_validation_config(FireOn.BOTH)

        orchestrator.run_coordinator.queue_trigger_validation = MagicMock()  # type: ignore[method-assign]
        orchestrator.run_coordinator.run_trigger_validation = AsyncMock(  # type: ignore[method-assign]
            return_value=MagicMock(status="passed")
        )

        # 0 success, 3 failures
        await orchestrator._fire_run_end_trigger(success_count=0, total_count=3)

        orchestrator.run_coordinator.queue_trigger_validation.assert_called_once_with(
            TriggerType.RUN_END,
            {"success_count": 0, "total_count": 3},
        )

    # Row 2: N success, 0 failures (all issues succeeded)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_all_success_fire_on_success_fires(
        self,
        tmp_path: Path,
        make_validation_config: Callable,
        make_orchestrator: Callable,
    ) -> None:
        """fire_on=success with all success → run_end fires.

        Truth table row: success_count=N(>0), failure_count=0, fire_on=success → fire
        """
        from unittest.mock import AsyncMock, MagicMock

        from src.domain.validation.config import FireOn, TriggerType

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
        )
        orchestrator._validation_config = make_validation_config(FireOn.SUCCESS)

        orchestrator.run_coordinator.queue_trigger_validation = MagicMock()  # type: ignore[method-assign]
        orchestrator.run_coordinator.run_trigger_validation = AsyncMock(  # type: ignore[method-assign]
            return_value=MagicMock(status="passed")
        )

        # 3 success, 0 failures
        await orchestrator._fire_run_end_trigger(success_count=3, total_count=3)

        orchestrator.run_coordinator.queue_trigger_validation.assert_called_once_with(
            TriggerType.RUN_END,
            {"success_count": 3, "total_count": 3},
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_all_success_fire_on_failure_skips(
        self,
        tmp_path: Path,
        make_validation_config: Callable,
        make_orchestrator: Callable,
    ) -> None:
        """fire_on=failure with all success → run_end skipped.

        Truth table row: success_count=N(>0), failure_count=0, fire_on=failure → skip
        """
        from unittest.mock import AsyncMock, MagicMock

        from src.domain.validation.config import FireOn

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
        )
        orchestrator._validation_config = make_validation_config(FireOn.FAILURE)

        orchestrator.run_coordinator.queue_trigger_validation = MagicMock()  # type: ignore[method-assign]
        orchestrator.run_coordinator.run_trigger_validation = AsyncMock()  # type: ignore[method-assign]

        # 3 success, 0 failures
        await orchestrator._fire_run_end_trigger(success_count=3, total_count=3)

        # Should NOT queue trigger
        orchestrator.run_coordinator.queue_trigger_validation.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_all_success_fire_on_both_fires(
        self,
        tmp_path: Path,
        make_validation_config: Callable,
        make_orchestrator: Callable,
    ) -> None:
        """fire_on=both with all success → run_end fires.

        Truth table row: success_count=N(>0), failure_count=0, fire_on=both → fire
        """
        from unittest.mock import AsyncMock, MagicMock

        from src.domain.validation.config import FireOn, TriggerType

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
        )
        orchestrator._validation_config = make_validation_config(FireOn.BOTH)

        orchestrator.run_coordinator.queue_trigger_validation = MagicMock()  # type: ignore[method-assign]
        orchestrator.run_coordinator.run_trigger_validation = AsyncMock(  # type: ignore[method-assign]
            return_value=MagicMock(status="passed")
        )

        # 3 success, 0 failures
        await orchestrator._fire_run_end_trigger(success_count=3, total_count=3)

        orchestrator.run_coordinator.queue_trigger_validation.assert_called_once_with(
            TriggerType.RUN_END,
            {"success_count": 3, "total_count": 3},
        )

    # Row 3: N success, M failures (mixed outcomes)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_mixed_outcomes_fire_on_success_fires(
        self,
        tmp_path: Path,
        make_validation_config: Callable,
        make_orchestrator: Callable,
    ) -> None:
        """fire_on=success with mixed outcomes → run_end fires.

        Truth table row: success_count=N(>0), failure_count=M(>0), fire_on=success → fire
        Per spec R7: fire_on=success fires if success_count > 0
        """
        from unittest.mock import AsyncMock, MagicMock

        from src.domain.validation.config import FireOn, TriggerType

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
        )
        orchestrator._validation_config = make_validation_config(FireOn.SUCCESS)

        orchestrator.run_coordinator.queue_trigger_validation = MagicMock()  # type: ignore[method-assign]
        orchestrator.run_coordinator.run_trigger_validation = AsyncMock(  # type: ignore[method-assign]
            return_value=MagicMock(status="passed")
        )

        # 2 success, 1 failure (mixed)
        await orchestrator._fire_run_end_trigger(success_count=2, total_count=3)

        # Per spec R7: fire_on=success fires when success_count > 0
        orchestrator.run_coordinator.queue_trigger_validation.assert_called_once_with(
            TriggerType.RUN_END,
            {"success_count": 2, "total_count": 3},
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_mixed_outcomes_fire_on_failure_fires(
        self,
        tmp_path: Path,
        make_validation_config: Callable,
        make_orchestrator: Callable,
    ) -> None:
        """fire_on=failure with mixed outcomes → run_end fires.

        Truth table row: success_count=N(>0), failure_count=M(>0), fire_on=failure → fire
        Per spec: fire_on=failure fires if failure_count > 0
        """
        from unittest.mock import AsyncMock, MagicMock

        from src.domain.validation.config import FireOn, TriggerType

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
        )
        orchestrator._validation_config = make_validation_config(FireOn.FAILURE)

        orchestrator.run_coordinator.queue_trigger_validation = MagicMock()  # type: ignore[method-assign]
        orchestrator.run_coordinator.run_trigger_validation = AsyncMock(  # type: ignore[method-assign]
            return_value=MagicMock(status="passed")
        )

        # 2 success, 1 failure (mixed)
        await orchestrator._fire_run_end_trigger(success_count=2, total_count=3)

        orchestrator.run_coordinator.queue_trigger_validation.assert_called_once_with(
            TriggerType.RUN_END,
            {"success_count": 2, "total_count": 3},
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_mixed_outcomes_fire_on_both_fires(
        self,
        tmp_path: Path,
        make_validation_config: Callable,
        make_orchestrator: Callable,
    ) -> None:
        """fire_on=both with mixed outcomes → run_end fires.

        Truth table row: success_count=N(>0), failure_count=M(>0), fire_on=both → fire
        """
        from unittest.mock import AsyncMock, MagicMock

        from src.domain.validation.config import FireOn, TriggerType

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
        )
        orchestrator._validation_config = make_validation_config(FireOn.BOTH)

        orchestrator.run_coordinator.queue_trigger_validation = MagicMock()  # type: ignore[method-assign]
        orchestrator.run_coordinator.run_trigger_validation = AsyncMock(  # type: ignore[method-assign]
            return_value=MagicMock(status="passed")
        )

        # 2 success, 1 failure (mixed)
        await orchestrator._fire_run_end_trigger(success_count=2, total_count=3)

        orchestrator.run_coordinator.queue_trigger_validation.assert_called_once_with(
            TriggerType.RUN_END,
            {"success_count": 2, "total_count": 3},
        )


# =============================================================================
# Run abort scenario tests
# =============================================================================


class TestRunAbortScenarios:
    """Scenario tests for run abort behavior.

    Per spec "Run abort contract":
    - When abort is triggered, abort_event is set
    - In-flight issues finalize as interrupted
    - run_end is NOT executed; logs skipped with reason=run_aborted
    """

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_run_end_skipped_when_abort_set(
        self, tmp_path: Path, make_orchestrator: Callable
    ) -> None:
        """run_end is skipped when abort_run is set.

        Per spec R7: If run aborts before all issues finalize,
        run_end MUST log skipped with reason=run_aborted.
        """
        from unittest.mock import AsyncMock, MagicMock

        from src.domain.validation.config import (
            FailureMode,
            FireOn,
            RunEndTriggerConfig,
            ValidationConfig,
            ValidationTriggersConfig,
        )

        triggers = ValidationTriggersConfig(
            run_end=RunEndTriggerConfig(
                failure_mode=FailureMode.CONTINUE,
                commands=(),
                fire_on=FireOn.BOTH,
            )
        )
        validation_config = ValidationConfig(validation_triggers=triggers)

        # Create mock event sink to verify skip event is emitted
        mock_event_sink = MagicMock()

        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
            event_sink=mock_event_sink,
        )
        orchestrator._validation_config = validation_config

        # Set abort_run flag (simulating abort trigger)
        orchestrator.issue_coordinator.request_abort(reason="test_abort")

        # Verify abort_event is set
        assert orchestrator.issue_coordinator.abort_event.is_set()
        assert orchestrator.issue_coordinator.abort_run is True

        orchestrator.run_coordinator.queue_trigger_validation = MagicMock()  # type: ignore[method-assign]
        orchestrator.run_coordinator.run_trigger_validation = AsyncMock()  # type: ignore[method-assign]

        # Even with fire_on=BOTH and successful issues, abort should prevent run_end
        await orchestrator._fire_run_end_trigger(success_count=3, total_count=3)

        # run_end should NOT be queued
        orchestrator.run_coordinator.queue_trigger_validation.assert_not_called()

        # Verify event sink received skip event with reason=run_aborted
        mock_event_sink.on_trigger_validation_skipped.assert_called_once_with(
            "run_end", "run_aborted"
        )

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_abort_event_set_on_request_abort(
        self, tmp_path: Path, make_orchestrator: Callable
    ) -> None:
        """abort_event is set when request_abort is called.

        Per spec: abort_event allows async checking of abort status
        """
        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
        )

        # Initially not set
        assert not orchestrator.issue_coordinator.abort_event.is_set()
        assert not orchestrator.issue_coordinator.abort_run

        # Request abort
        orchestrator.issue_coordinator.request_abort(reason="session_end_failed")

        # Both should be set
        assert orchestrator.issue_coordinator.abort_event.is_set()
        assert orchestrator.issue_coordinator.abort_run is True

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_abort_idempotent(
        self, tmp_path: Path, make_orchestrator: Callable
    ) -> None:
        """Multiple request_abort calls are idempotent.

        Per spec: abort is not reverted, multiple calls don't stack.
        """
        orchestrator = make_orchestrator(
            repo_path=tmp_path,
            max_agents=1,
            timeout_minutes=1,
        )

        # First abort
        orchestrator.issue_coordinator.request_abort(reason="first_abort")
        assert orchestrator.issue_coordinator.abort_run is True

        # Second abort should be no-op
        orchestrator.issue_coordinator.request_abort(reason="second_abort")
        assert orchestrator.issue_coordinator.abort_run is True
        assert orchestrator.issue_coordinator.abort_event.is_set()

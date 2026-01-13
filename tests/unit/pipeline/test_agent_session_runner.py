"""Unit tests for AgentSessionRunner.

Tests the P0/P1 filtering logic in _build_session_output() and early interrupt
path to ensure agent_id is preserved.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from src.domain.lifecycle import (
    ImplementerLifecycle,
    LifecycleConfig,
    LifecycleContext,
    RetryState,
)
from src.pipeline.agent_session_runner import (
    AgentSessionConfig,
    AgentSessionInput,
    AgentSessionRunner,
    SessionConfig,
    SessionExecutionState,
    SessionPrompts,
)
from tests.helpers.protocol_stubs import (
    StubGateRunner,
    StubReviewRunner,
    StubSessionLifecycle,
)


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


@dataclass
class FakeOptions:
    """Fake options object."""

    pass


@dataclass
class FakeLintCache:
    """Fake lint cache."""

    pass


def make_session_config() -> SessionConfig:
    """Create a minimal SessionConfig for testing."""
    return SessionConfig(
        agent_id="test-agent",
        options=FakeOptions(),
        lint_cache=FakeLintCache(),  # type: ignore[arg-type]
        log_file_wait_timeout=10.0,
        log_file_poll_interval=0.5,
        idle_timeout_seconds=None,
    )


def make_lifecycle_ctx(
    *,
    success: bool = False,
    last_review_result: FakeReviewResult | None = None,
    review_attempt: int = 0,
) -> LifecycleContext:
    """Create a LifecycleContext with specified state."""
    ctx = LifecycleContext()
    ctx.success = success
    ctx.last_review_result = last_review_result  # type: ignore[assignment]
    ctx.retry_state = RetryState(review_attempt=review_attempt)
    return ctx


def make_state(lifecycle_ctx: LifecycleContext) -> SessionExecutionState:
    """Create a SessionExecutionState with given lifecycle context."""
    lifecycle = ImplementerLifecycle(LifecycleConfig())
    return SessionExecutionState(
        lifecycle=lifecycle,
        lifecycle_ctx=lifecycle_ctx,
    )


class TestBuildSessionOutputFiltering:
    """Tests for P0/P1 filtering in _build_session_output."""

    def test_filters_to_p0_p1_only(self) -> None:
        """P2+ issues are excluded, only P0/P1 remain."""
        review_result = FakeReviewResult(
            passed=False,
            issues=[
                FakeReviewIssue(
                    file="src/foo.py",
                    line_start=10,
                    line_end=15,
                    priority=0,  # P0 - should be included
                    title="Critical bug",
                    body="This is critical",
                    reviewer="security",
                ),
                FakeReviewIssue(
                    file="src/bar.py",
                    line_start=20,
                    line_end=25,
                    priority=1,  # P1 - should be included
                    title="Important issue",
                    body="This is important",
                    reviewer="code_quality",
                ),
                FakeReviewIssue(
                    file="src/baz.py",
                    line_start=30,
                    line_end=35,
                    priority=2,  # P2 - should be excluded
                    title="Minor issue",
                    body="This is minor",
                    reviewer="style",
                ),
                FakeReviewIssue(
                    file="src/qux.py",
                    line_start=40,
                    line_end=45,
                    priority=3,  # P3 - should be excluded
                    title="Nitpick",
                    body="This is a nitpick",
                    reviewer="docs",
                ),
            ],
        )
        lifecycle_ctx = make_lifecycle_ctx(
            success=False,
            last_review_result=review_result,
            review_attempt=1,
        )
        state = make_state(lifecycle_ctx)
        session_cfg = make_session_config()

        runner = AgentSessionRunner.__new__(AgentSessionRunner)
        output = runner._build_session_output(session_cfg, state, duration=10.0)

        assert output.last_review_issues is not None
        assert len(output.last_review_issues) == 2
        assert output.last_review_issues[0]["priority"] == 0
        assert output.last_review_issues[0]["file"] == "src/foo.py"
        assert output.last_review_issues[1]["priority"] == 1
        assert output.last_review_issues[1]["file"] == "src/bar.py"

    def test_none_when_success_true(self) -> None:
        """last_review_issues is None when success=True."""
        review_result = FakeReviewResult(
            passed=True,
            issues=[
                FakeReviewIssue(
                    file="src/foo.py",
                    line_start=10,
                    line_end=15,
                    priority=0,
                    title="Critical bug",
                    body="This is critical",
                    reviewer="security",
                ),
            ],
        )
        lifecycle_ctx = make_lifecycle_ctx(
            success=True,  # Success - should not populate issues
            last_review_result=review_result,
            review_attempt=1,
        )
        state = make_state(lifecycle_ctx)
        session_cfg = make_session_config()

        runner = AgentSessionRunner.__new__(AgentSessionRunner)
        output = runner._build_session_output(session_cfg, state, duration=10.0)

        assert output.last_review_issues is None

    def test_none_when_last_review_result_is_none(self) -> None:
        """last_review_issues is None when last_review_result is None."""
        lifecycle_ctx = make_lifecycle_ctx(
            success=False,
            last_review_result=None,  # No review result
            review_attempt=1,
        )
        state = make_state(lifecycle_ctx)
        session_cfg = make_session_config()

        runner = AgentSessionRunner.__new__(AgentSessionRunner)
        output = runner._build_session_output(session_cfg, state, duration=10.0)

        assert output.last_review_issues is None

    def test_none_when_review_attempt_zero(self) -> None:
        """last_review_issues is None when review_attempt=0 (review never ran)."""
        review_result = FakeReviewResult(
            passed=False,
            issues=[
                FakeReviewIssue(
                    file="src/foo.py",
                    line_start=10,
                    line_end=15,
                    priority=0,
                    title="Critical bug",
                    body="This is critical",
                    reviewer="security",
                ),
            ],
        )
        lifecycle_ctx = make_lifecycle_ctx(
            success=False,
            last_review_result=review_result,
            review_attempt=0,  # Review never ran
        )
        state = make_state(lifecycle_ctx)
        session_cfg = make_session_config()

        runner = AgentSessionRunner.__new__(AgentSessionRunner)
        output = runner._build_session_output(session_cfg, state, duration=10.0)

        assert output.last_review_issues is None

    def test_none_when_all_issues_filtered_out(self) -> None:
        """Empty list after filtering results in None, not []."""
        review_result = FakeReviewResult(
            passed=False,
            issues=[
                FakeReviewIssue(
                    file="src/foo.py",
                    line_start=10,
                    line_end=15,
                    priority=2,  # P2 - will be filtered out
                    title="Minor issue",
                    body="This is minor",
                    reviewer="style",
                ),
                FakeReviewIssue(
                    file="src/bar.py",
                    line_start=20,
                    line_end=25,
                    priority=3,  # P3 - will be filtered out
                    title="Nitpick",
                    body="This is a nitpick",
                    reviewer="docs",
                ),
            ],
        )
        lifecycle_ctx = make_lifecycle_ctx(
            success=False,
            last_review_result=review_result,
            review_attempt=1,
        )
        state = make_state(lifecycle_ctx)
        session_cfg = make_session_config()

        runner = AgentSessionRunner.__new__(AgentSessionRunner)
        output = runner._build_session_output(session_cfg, state, duration=10.0)

        assert output.last_review_issues is None

    def test_excludes_issues_with_none_priority(self) -> None:
        """Issues with priority=None are excluded from filtering."""
        review_result = FakeReviewResult(
            passed=False,
            issues=[
                FakeReviewIssue(
                    file="src/foo.py",
                    line_start=10,
                    line_end=15,
                    priority=None,  # No priority - should be excluded
                    title="Unknown priority",
                    body="Priority not set",
                    reviewer="unknown",
                ),
                FakeReviewIssue(
                    file="src/bar.py",
                    line_start=20,
                    line_end=25,
                    priority=1,  # P1 - should be included
                    title="Important issue",
                    body="This is important",
                    reviewer="code_quality",
                ),
            ],
        )
        lifecycle_ctx = make_lifecycle_ctx(
            success=False,
            last_review_result=review_result,
            review_attempt=1,
        )
        state = make_state(lifecycle_ctx)
        session_cfg = make_session_config()

        runner = AgentSessionRunner.__new__(AgentSessionRunner)
        output = runner._build_session_output(session_cfg, state, duration=10.0)

        assert output.last_review_issues is not None
        assert len(output.last_review_issues) == 1
        assert output.last_review_issues[0]["priority"] == 1

    def test_dict_structure_matches_protocol(self) -> None:
        """Dict structure matches ReviewIssueProtocol fields exactly."""
        review_result = FakeReviewResult(
            passed=False,
            issues=[
                FakeReviewIssue(
                    file="src/example.py",
                    line_start=42,
                    line_end=50,
                    priority=0,
                    title="Security vulnerability",
                    body="SQL injection detected",
                    reviewer="security_scanner",
                ),
            ],
        )
        lifecycle_ctx = make_lifecycle_ctx(
            success=False,
            last_review_result=review_result,
            review_attempt=1,
        )
        state = make_state(lifecycle_ctx)
        session_cfg = make_session_config()

        runner = AgentSessionRunner.__new__(AgentSessionRunner)
        output = runner._build_session_output(session_cfg, state, duration=10.0)

        assert output.last_review_issues is not None
        issue_dict = output.last_review_issues[0]
        # Verify all expected keys are present
        assert issue_dict == {
            "file": "src/example.py",
            "line_start": 42,
            "line_end": 50,
            "priority": 0,
            "title": "Security vulnerability",
            "body": "SQL injection detected",
            "reviewer": "security_scanner",
        }
        # Verify no extra keys
        assert set(issue_dict.keys()) == {
            "file",
            "line_start",
            "line_end",
            "priority",
            "title",
            "body",
            "reviewer",
        }


@dataclass
class FakeSDKClientFactory:
    """Fake SDK client factory for testing."""

    def create_client(
        self,
        options: object,
        query: str,
    ) -> object:
        raise NotImplementedError("Should not be called in early interrupt tests")

    def with_resume(self, options: object, session_id: str) -> object:
        raise NotImplementedError("Should not be called in early interrupt tests")


def make_prompts() -> SessionPrompts:
    """Create minimal SessionPrompts for testing."""
    return SessionPrompts(
        gate_followup="gate prompt",
        review_followup="review prompt",
        idle_resume="idle prompt",
        checkpoint_request="checkpoint prompt",
        continuation="continuation prompt",
    )


class TestEarlyInterruptPath:
    """Tests for early SIGINT interrupt handling in run_session."""

    @pytest.mark.asyncio
    async def test_early_interrupt_returns_agent_id_when_provided(self) -> None:
        """run_session returns provided agent_id on early interrupt."""
        # Set up pre-signaled interrupt event
        interrupt_event = asyncio.Event()
        interrupt_event.set()

        config = AgentSessionConfig(
            repo_path=Path("/tmp/test-repo"),
            timeout_seconds=300,
            prompts=make_prompts(),
        )
        runner = AgentSessionRunner(
            config=config,
            sdk_client_factory=FakeSDKClientFactory(),  # type: ignore[arg-type]
            gate_runner=StubGateRunner(),  # type: ignore[arg-type]
            review_runner=StubReviewRunner(),  # type: ignore[arg-type]
            session_lifecycle=StubSessionLifecycle(),  # type: ignore[arg-type]
        )

        session_input = AgentSessionInput(
            issue_id="test-issue",
            prompt="test prompt",
            agent_id="my-custom-agent-id",
        )

        output = await runner.run_session(
            session_input, interrupt_event=interrupt_event
        )

        assert output.interrupted is True
        assert output.agent_id == "my-custom-agent-id"
        assert output.success is False
        assert "interrupted" in output.summary.lower()

    @pytest.mark.asyncio
    async def test_early_interrupt_generates_agent_id_when_not_provided(self) -> None:
        """run_session generates and returns agent_id on early interrupt."""
        # Set up pre-signaled interrupt event
        interrupt_event = asyncio.Event()
        interrupt_event.set()

        config = AgentSessionConfig(
            repo_path=Path("/tmp/test-repo"),
            timeout_seconds=300,
            prompts=make_prompts(),
        )
        runner = AgentSessionRunner(
            config=config,
            sdk_client_factory=FakeSDKClientFactory(),  # type: ignore[arg-type]
            gate_runner=StubGateRunner(),  # type: ignore[arg-type]
            review_runner=StubReviewRunner(),  # type: ignore[arg-type]
            session_lifecycle=StubSessionLifecycle(),  # type: ignore[arg-type]
        )

        session_input = AgentSessionInput(
            issue_id="test-issue",
            prompt="test prompt",
            # agent_id not provided - should be generated
        )

        output = await runner.run_session(
            session_input, interrupt_event=interrupt_event
        )

        assert output.interrupted is True
        assert output.agent_id != ""
        assert output.agent_id.startswith("test-issue-")
        assert output.success is False

    @pytest.mark.asyncio
    async def test_early_interrupt_preserves_baseline_timestamp(self) -> None:
        """run_session preserves baseline_timestamp on early interrupt."""
        # Set up pre-signaled interrupt event
        interrupt_event = asyncio.Event()
        interrupt_event.set()

        config = AgentSessionConfig(
            repo_path=Path("/tmp/test-repo"),
            timeout_seconds=300,
            prompts=make_prompts(),
        )
        runner = AgentSessionRunner(
            config=config,
            sdk_client_factory=FakeSDKClientFactory(),  # type: ignore[arg-type]
            gate_runner=StubGateRunner(),  # type: ignore[arg-type]
            review_runner=StubReviewRunner(),  # type: ignore[arg-type]
            session_lifecycle=StubSessionLifecycle(),  # type: ignore[arg-type]
        )

        session_input = AgentSessionInput(
            issue_id="test-issue",
            prompt="test prompt",
            agent_id="agent-123",
            baseline_timestamp=1700000000,
        )

        output = await runner.run_session(
            session_input, interrupt_event=interrupt_event
        )

        assert output.interrupted is True
        assert output.baseline_timestamp == 1700000000
        assert output.agent_id == "agent-123"


class TestProtocolInterfaceAcceptance:
    """Tests for protocol interface acceptance in AgentSessionRunner."""

    def test_accepts_protocol_interfaces(self) -> None:
        """AgentSessionRunner can be initialized with protocol interfaces."""
        from dataclasses import dataclass

        @dataclass
        class FakeGateRunner:
            """Fake IGateRunner implementation."""

            async def run_gate_check(
                self, issue_id: str, log_path: object, retry_state: object
            ) -> tuple[object, int]:
                raise NotImplementedError

            async def run_session_end_check(
                self, issue_id: str, log_path: object, retry_state: object
            ) -> object:
                raise NotImplementedError

        @dataclass
        class FakeReviewRunner:
            """Fake IReviewRunner implementation."""

            async def run_review(
                self,
                issue_id: str,
                description: str | None,
                session_id: str | None,
                retry_state: object,
                author_context: str | None,
                previous_findings: object,
                session_end_result: object,
            ) -> object:
                raise NotImplementedError

            def check_no_progress(
                self,
                log_path: object,
                log_offset: int,
                prev_commit: str | None,
                curr_commit: str | None,
            ) -> bool:
                return False

        @dataclass
        class FakeSessionLifecycle:
            """Fake ISessionLifecycle implementation."""

            def get_log_path(self, session_id: str) -> Path:
                return Path("/tmp/test.jsonl")

            def get_log_offset(self, log_path: object, start_offset: int) -> int:
                return 0

            def on_abort(self, reason: str) -> None:
                pass

            def get_abort_event(self) -> None:
                return None

            def on_tool_use(
                self, agent_id: str, tool_name: str, args: dict | None
            ) -> None:
                pass

            def on_agent_text(self, agent_id: str, text: str) -> None:
                pass

        config = AgentSessionConfig(
            repo_path=Path("/tmp/test-repo"),
            timeout_seconds=300,
            prompts=make_prompts(),
        )

        # Should not raise
        runner = AgentSessionRunner(
            config=config,
            sdk_client_factory=FakeSDKClientFactory(),  # type: ignore[arg-type]
            gate_runner=FakeGateRunner(),  # type: ignore[arg-type]
            review_runner=FakeReviewRunner(),  # type: ignore[arg-type]
            session_lifecycle=FakeSessionLifecycle(),  # type: ignore[arg-type]
        )

        assert runner.gate_runner is not None
        assert runner.review_runner is not None
        assert runner.session_lifecycle is not None

    def test_bridge_methods_use_protocols(self) -> None:
        """Bridge methods use protocol interfaces."""
        from dataclasses import dataclass

        log_path_calls: list[str] = []
        log_offset_calls: list[tuple[Path, int]] = []

        @dataclass
        class FakeSessionLifecycle:
            """Fake ISessionLifecycle that tracks calls."""

            def get_log_path(self, session_id: str) -> Path:
                log_path_calls.append(session_id)
                return Path("/tmp/protocol-log.jsonl")

            def get_log_offset(self, log_path: object, start_offset: int) -> int:
                log_offset_calls.append((log_path, start_offset))  # type: ignore[arg-type]
                return 42

            def on_abort(self, reason: str) -> None:
                pass

            def get_abort_event(self) -> None:
                return None

            def on_tool_use(
                self, agent_id: str, tool_name: str, args: dict | None
            ) -> None:
                pass

            def on_agent_text(self, agent_id: str, text: str) -> None:
                pass

        @dataclass
        class FakeGateRunner:
            """Fake IGateRunner implementation."""

            async def run_gate_check(
                self, issue_id: str, log_path: object, retry_state: object
            ) -> tuple[object, int]:
                raise NotImplementedError

            async def run_session_end_check(
                self, issue_id: str, log_path: object, retry_state: object
            ) -> object:
                raise NotImplementedError

        @dataclass
        class FakeReviewRunner:
            """Fake IReviewRunner implementation."""

            async def run_review(
                self,
                issue_id: str,
                description: str | None,
                session_id: str | None,
                retry_state: object,
                author_context: str | None,
                previous_findings: object,
                session_end_result: object,
            ) -> object:
                raise NotImplementedError

            def check_no_progress(
                self,
                log_path: object,
                log_offset: int,
                prev_commit: str | None,
                curr_commit: str | None,
            ) -> bool:
                return False

        config = AgentSessionConfig(
            repo_path=Path("/tmp/test-repo"),
            timeout_seconds=300,
            prompts=make_prompts(),
        )

        runner = AgentSessionRunner(
            config=config,
            sdk_client_factory=FakeSDKClientFactory(),  # type: ignore[arg-type]
            gate_runner=FakeGateRunner(),  # type: ignore[arg-type]
            review_runner=FakeReviewRunner(),  # type: ignore[arg-type]
            session_lifecycle=FakeSessionLifecycle(),
        )

        # Test _get_log_path uses protocol
        result_path = runner._get_log_path("test-session")
        assert result_path == Path("/tmp/protocol-log.jsonl")
        assert log_path_calls == ["test-session"]

        # Test _get_log_offset uses protocol
        result_offset = runner._get_log_offset(Path("/tmp/test.jsonl"), 10)
        assert result_offset == 42
        assert log_offset_calls == [(Path("/tmp/test.jsonl"), 10)]

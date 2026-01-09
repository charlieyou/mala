"""Unit tests for AgentSessionRunner.

Tests the P0/P1 filtering logic in _build_session_output().
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.domain.lifecycle import (
    ImplementerLifecycle,
    LifecycleConfig,
    LifecycleContext,
    RetryState,
)
from src.pipeline.agent_session_runner import (
    AgentSessionRunner,
    SessionConfig,
    SessionExecutionState,
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

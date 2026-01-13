"""Stub implementations of protocols for testing.

These stubs provide minimal implementations of the IGateRunner, IReviewRunner,
and ISessionLifecycle protocols for use in unit and integration tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Awaitable, Callable, Sequence
    from pathlib import Path

    from src.core.protocols.review import ReviewIssueProtocol
    from src.core.session_end_result import SessionEndResult, SessionEndRetryState
    from src.domain.lifecycle import GateOutcome, RetryState, ReviewOutcome

    # Callback types that accept both sync and async functions
    GateCheckCallback = Callable[
        [str, Path, RetryState],
        tuple[GateOutcome, int] | Awaitable[tuple[GateOutcome, int]],
    ]
    SessionEndCheckCallback = Callable[
        [str, Path, SessionEndRetryState],
        SessionEndResult | Awaitable[SessionEndResult],
    ]


@dataclass
class StubGateOutcome:
    """Stub gate outcome that satisfies GateOutcomeProtocol."""

    passed: bool = True
    failure_reasons: list[str] = field(default_factory=list)
    commit_hash: str | None = None
    no_progress: bool = False
    resolution: Any = None


@dataclass
class StubReviewOutcome:
    """Stub review outcome that satisfies ReviewOutcomeProtocol."""

    passed: bool = True
    parse_error: str | None = None
    fatal_error: bool = False
    issues: list[Any] = field(default_factory=list)
    interrupted: bool = False


@dataclass
class StubGateRunner:
    """Stub IGateRunner implementation for testing.

    Provides default pass behavior for gate and session_end checks.
    Configure via constructor arguments or by assigning to attributes
    after construction.
    """

    gate_result: StubGateOutcome = field(default_factory=StubGateOutcome)
    gate_offset: int = 0
    session_end_result: SessionEndResult | None = None
    on_gate_check: GateCheckCallback | None = None
    on_session_end_check: SessionEndCheckCallback | None = None

    async def run_gate_check(
        self,
        issue_id: str,
        log_path: Path,
        retry_state: RetryState,
    ) -> tuple[GateOutcome, int]:
        """Run gate check - returns configured result or passes by default.

        The on_gate_check callback can be sync or async - both are supported.
        """
        import asyncio

        if self.on_gate_check is not None:
            result = self.on_gate_check(issue_id, log_path, retry_state)
            # Handle both sync and async callbacks
            if asyncio.iscoroutine(result):
                return await result  # type: ignore[return-value]
            return result  # type: ignore[return-value]
        return self.gate_result, self.gate_offset  # type: ignore[return-value]

    async def run_session_end_check(
        self,
        issue_id: str,
        log_path: Path,
        retry_state: SessionEndRetryState,
    ) -> SessionEndResult:
        """Run session_end check - returns configured result or skipped.

        The on_session_end_check callback can be sync or async - both are supported.
        """
        import asyncio

        if self.on_session_end_check is not None:
            result = self.on_session_end_check(issue_id, log_path, retry_state)
            # Handle both sync and async callbacks
            if asyncio.iscoroutine(result):
                return await result  # type: ignore[return-value]
            return result  # type: ignore[return-value]
        if self.session_end_result is not None:
            return self.session_end_result
        # Import here to avoid import issues in tests
        from src.core.session_end_result import SessionEndResult

        return SessionEndResult(status="skipped", reason="stub")


@dataclass
class StubReviewRunner:
    """Stub IReviewRunner implementation for testing.

    Provides default pass behavior for review and no-progress checks.
    Configure via constructor arguments or by assigning to attributes
    after construction.
    """

    review_result: StubReviewOutcome = field(default_factory=StubReviewOutcome)
    no_progress_result: bool = False
    on_review: (
        Callable[
            [
                str,
                str | None,
                str | None,
                RetryState,
                str | None,
                Sequence[ReviewIssueProtocol] | None,
                SessionEndResult | None,
            ],
            ReviewOutcome,
        ]
        | None
    ) = None
    on_check_no_progress: Callable[[Path, int, str | None, str | None], bool] | None = (
        None
    )

    async def run_review(
        self,
        issue_id: str,
        description: str | None,
        session_id: str | None,
        retry_state: RetryState,
        author_context: str | None,
        previous_findings: Sequence[ReviewIssueProtocol] | None,
        session_end_result: SessionEndResult | None,
    ) -> ReviewOutcome:
        """Run review - returns configured result or passes by default.

        The on_review callback can be sync or async - both are supported.
        """
        import asyncio

        if self.on_review is not None:
            result = self.on_review(
                issue_id,
                description,
                session_id,
                retry_state,
                author_context,
                previous_findings,
                session_end_result,
            )
            # Handle both sync and async callbacks
            if asyncio.iscoroutine(result):
                return await result  # type: ignore[return-value]
            return result  # type: ignore[return-value]
        return self.review_result  # type: ignore[return-value]

    def check_no_progress(
        self,
        log_path: Path,
        log_offset: int,
        prev_commit: str | None,
        curr_commit: str | None,
    ) -> bool:
        """Check no progress - returns configured result or False by default."""
        if self.on_check_no_progress is not None:
            return self.on_check_no_progress(
                log_path, log_offset, prev_commit, curr_commit
            )
        return self.no_progress_result


@dataclass
class StubSessionLifecycle:
    """Stub ISessionLifecycle implementation for testing.

    Provides default implementations for lifecycle operations.
    Configure via constructor arguments or by assigning to attributes
    after construction.

    IMPORTANT: log_path has no default - callers must provide it explicitly
    or use on_get_log_path callback. This avoids hardcoded /tmp paths that
    fail in sandboxed CI environments.
    """

    log_path: Path | None = None
    log_offset: int = 0
    abort_event: asyncio.Event | None = None
    on_get_log_path: Callable[[str], Path] | None = None
    on_get_log_offset: Callable[[Path, int], int] | None = None
    on_abort_callback: Callable[[str], None] | None = None
    on_tool_use_callback: Callable[[str, str, dict[str, Any] | None], None] | None = (
        None
    )
    on_agent_text_callback: Callable[[str, str], None] | None = None

    def get_log_path(self, session_id: str) -> Path:
        """Get log path - returns configured path or raises if not configured."""
        if self.on_get_log_path is not None:
            return self.on_get_log_path(session_id)
        if self.log_path is None:
            raise ValueError(
                "StubSessionLifecycle.log_path not configured. "
                "Pass log_path=tmp_path/'log.jsonl' or on_get_log_path callback."
            )
        return self.log_path

    def get_log_offset(self, log_path: Path, start_offset: int) -> int:
        """Get log offset - returns configured offset or passed offset."""
        if self.on_get_log_offset is not None:
            return self.on_get_log_offset(log_path, start_offset)
        return self.log_offset or start_offset

    def on_abort(self, reason: str) -> None:
        """Handle abort - calls configured callback or no-op."""
        if self.on_abort_callback is not None:
            self.on_abort_callback(reason)

    def get_abort_event(self) -> asyncio.Event | None:
        """Get abort event - returns configured event or None."""
        return self.abort_event

    def on_tool_use(
        self, agent_id: str, tool_name: str, args: dict[str, Any] | None
    ) -> None:
        """Handle tool use - calls configured callback or no-op."""
        if self.on_tool_use_callback is not None:
            self.on_tool_use_callback(agent_id, tool_name, args)

    def on_agent_text(self, agent_id: str, text: str) -> None:
        """Handle agent text - calls configured callback or no-op."""
        if self.on_agent_text_callback is not None:
            self.on_agent_text_callback(agent_id, text)

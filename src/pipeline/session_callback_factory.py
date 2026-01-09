"""SessionCallbackFactory: Builds SessionCallbacks for agent sessions.

This factory encapsulates callback construction that bridges orchestrator state
to the pipeline runners. It receives state references and returns a SessionCallbacks
instance wired to the appropriate callbacks.

Design principles:
- Single responsibility: only builds callbacks, doesn't run gates/reviews
- Protocol-based dependencies for testability
- All callback closures capture minimal state
- Late-bound lookups: dependencies are accessed via callables to support
  runtime patching (e.g., tests that swap event_sink after construction)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from src.pipeline.agent_session_runner import SessionCallbacks

if TYPE_CHECKING:
    import asyncio
    from collections.abc import Callable
    from pathlib import Path

    from src.core.protocols import (
        LogProvider,
        MalaEventSink,
    )
    from src.domain.lifecycle import (
        GateOutcome,
        RetryState,
        ReviewIssue,
        ReviewOutcome,
    )
    from src.domain.validation.spec import ValidationSpec
    from src.pipeline.review_runner import ReviewRunner


@dataclass
class _InterruptedReviewResultWrapper:
    """Wrapper for ReviewResultProtocol that ensures interrupted flag is correct.

    When a review completes but SIGINT fires before the guard is checked,
    the original result may have interrupted=False while the ReviewOutput
    has interrupted=True. This wrapper copies all fields from the original
    result but overrides interrupted to True.
    """

    passed: bool
    issues: list[ReviewIssue]
    parse_error: str | None
    fatal_error: bool
    review_log_path: Path | None = None
    interrupted: bool = True


class GateAsyncRunner(Protocol):
    """Protocol for async gate check execution."""

    async def run_gate_async(
        self,
        issue_id: str,
        log_path: Path,
        retry_state: RetryState,
        interrupt_event: asyncio.Event | None = None,
    ) -> tuple[GateOutcome, int]:
        """Run quality gate check asynchronously."""
        ...


class SessionCallbackFactory:
    """Factory for building SessionCallbacks with injected dependencies.

    This factory creates callbacks that bridge orchestrator state to the
    pipeline runners without coupling the runners to orchestrator internals.

    Usage:
        factory = SessionCallbackFactory(
            gate_async_runner=...,  # Protocol for async gate checks
            review_runner=...,
            log_provider=...,
            event_sink=...,
            repo_path=...,
            on_session_log_path=...,  # Callback for session log path
            on_review_log_path=...,   # Callback for review log path
        )
        callbacks = factory.build(issue_id)
    """

    def __init__(
        self,
        gate_async_runner: GateAsyncRunner,
        review_runner: ReviewRunner,
        log_provider: Callable[[], LogProvider],
        event_sink: Callable[[], MalaEventSink],
        evidence_check: Callable[[], GateChecker],
        repo_path: Path,
        on_session_log_path: Callable[[str, Path], None],
        on_review_log_path: Callable[[str, str], None],
        get_per_session_spec: GetPerSessionSpec,
        is_verbose: IsVerboseCheck,
        get_interrupt_event: Callable[[], asyncio.Event | None] | None = None,
    ) -> None:
        """Initialize the factory with dependencies.

        Args:
            gate_async_runner: Protocol for running async gate checks.
            review_runner: Runner for Cerberus code review.
            log_provider: Callable returning the log provider (late-bound).
            event_sink: Callable returning the event sink (late-bound).
            evidence_check: Callable returning the gate checker (late-bound).
            repo_path: Repository path for git operations.
            on_session_log_path: Callback when session log path becomes known.
            on_review_log_path: Callback when review log path becomes known.
            get_per_session_spec: Callable to get current per-session spec.
            is_verbose: Callable to check verbose mode.
            get_interrupt_event: Callable to get the interrupt event (late-bound).

        Note:
            log_provider, event_sink, evidence_check, and get_interrupt_event are
            callables to support late-bound lookups. This allows tests to patch
            orchestrator attributes after factory construction and have the
            patches take effect.
        """
        self._gate_async_runner = gate_async_runner
        self._review_runner = review_runner
        self._get_log_provider = log_provider
        self._get_event_sink = event_sink
        self._get_evidence_check = evidence_check
        self._repo_path = repo_path
        self._on_session_log_path = on_session_log_path
        self._on_review_log_path = on_review_log_path
        self._get_per_session_spec = get_per_session_spec
        self._is_verbose = is_verbose
        self._get_interrupt_event = get_interrupt_event or (lambda: None)

    def build(
        self,
        issue_id: str,
        on_abort: Callable[[str], None] | None = None,
    ) -> SessionCallbacks:
        """Build SessionCallbacks for a specific issue.

        Args:
            issue_id: The issue ID for tracking state.
            on_abort: Optional callback for fatal error signaling.

        Returns:
            SessionCallbacks with gate, review, and logging callbacks.
        """
        # Import here to avoid circular imports
        from src.core.models import ReviewInput
        from src.infra.git_utils import get_issue_commits_async
        from src.pipeline.review_runner import NoProgressInput

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateOutcome, int]:
            result, offset = await self._gate_async_runner.run_gate_async(
                issue_id, log_path, retry_state, self._get_interrupt_event()
            )
            return result, offset  # type: ignore[return-value]

        async def on_review_check(
            issue_id: str,
            issue_desc: str | None,
            session_id: str | None,
            _retry_state: RetryState,  # unused after removing timestamp filtering
        ) -> ReviewOutcome:
            self._review_runner.config.capture_session_log = self._is_verbose()
            commit_shas = await get_issue_commits_async(
                self._repo_path,
                issue_id,
            )
            review_input = ReviewInput(
                issue_id=issue_id,
                repo_path=self._repo_path,
                issue_description=issue_desc,
                commit_shas=commit_shas,
                claude_session_id=session_id,
            )
            output = await self._review_runner.run_review(
                review_input, self._get_interrupt_event()
            )
            if output.session_log_path:
                self._on_review_log_path(issue_id, output.session_log_path)

            # Propagate interrupted flag: if output.interrupted is True but
            # the result's interrupted flag is False (e.g., SIGINT fired after
            # reviewer completed), wrap the result to ensure correct flag.
            result: ReviewOutcome = output.result  # type: ignore[assignment]
            if output.interrupted and not getattr(result, "interrupted", False):
                result = _InterruptedReviewResultWrapper(
                    passed=result.passed,
                    issues=list(result.issues),
                    parse_error=result.parse_error,
                    fatal_error=result.fatal_error,
                    review_log_path=getattr(result, "review_log_path", None),
                    interrupted=True,
                )
            return result

        def on_review_no_progress(
            log_path: Path,
            log_offset: int,
            prev_commit: str | None,
            curr_commit: str | None,
        ) -> bool:
            no_progress_input = NoProgressInput(
                log_path=log_path,
                log_offset=log_offset,
                previous_commit_hash=prev_commit,
                current_commit_hash=curr_commit,
                spec=self._get_per_session_spec(),
            )
            return self._review_runner.check_no_progress(no_progress_input)

        def get_log_path(session_id: str) -> Path:
            log_path = self._get_log_provider().get_log_path(
                self._repo_path, session_id
            )
            self._on_session_log_path(issue_id, log_path)
            return log_path

        def get_log_offset(log_path: Path, start_offset: int) -> int:
            return self._get_evidence_check().get_log_end_offset(log_path, start_offset)

        def on_tool_use(agent_id: str, tool_name: str, arguments: dict | None) -> None:
            self._get_event_sink().on_tool_use(agent_id, tool_name, arguments=arguments)

        def on_agent_text(agent_id: str, text: str) -> None:
            self._get_event_sink().on_agent_text(agent_id, text)

        return SessionCallbacks(
            on_gate_check=on_gate_check,
            on_review_check=on_review_check,
            on_review_no_progress=on_review_no_progress,
            get_log_path=get_log_path,
            get_log_offset=get_log_offset,
            on_abort=on_abort,
            on_tool_use=on_tool_use,
            on_agent_text=on_agent_text,
        )


# Protocol for getting per-session spec
class GetPerSessionSpec(Protocol):
    """Protocol for getting the current per-session validation spec."""

    def __call__(self) -> ValidationSpec | None:
        """Return the current per-session spec, or None if not set."""
        ...


# Protocol for checking verbose mode
class IsVerboseCheck(Protocol):
    """Protocol for checking if verbose mode is enabled."""

    def __call__(self) -> bool:
        """Return True if verbose mode is enabled."""
        ...


# Protocol for gate checker (subset of GateChecker)
class GateChecker(Protocol):
    """Protocol for gate checking operations."""

    def get_log_end_offset(self, log_path: Path, start_offset: int) -> int:
        """Get the end offset of a log file."""
        ...

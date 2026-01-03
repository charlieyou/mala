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

from typing import TYPE_CHECKING, Protocol

from src.pipeline.agent_session_runner import SessionCallbacks

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.core.protocols import GateResultProtocol, LogProvider, ReviewResultProtocol
    from src.domain.lifecycle import RetryState
    from src.domain.quality_gate import GateResult
    from src.domain.validation.spec import ValidationSpec
    from src.infra.clients.cerberus_review import ReviewResult
    from src.infra.io.event_sink import MalaEventSink
    from src.pipeline.review_runner import ReviewRunner


class GateAsyncRunner(Protocol):
    """Protocol for async gate check execution."""

    async def run_gate_async(
        self,
        issue_id: str,
        log_path: Path,
        retry_state: RetryState,
    ) -> tuple[GateResult | GateResultProtocol, int]:
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
            session_log_paths=...,  # Dict reference for path tracking
            review_log_paths=...,   # Dict reference for review log tracking
        )
        callbacks = factory.build(issue_id)
    """

    def __init__(
        self,
        gate_async_runner: GateAsyncRunner,
        review_runner: ReviewRunner,
        log_provider: Callable[[], LogProvider],
        event_sink: Callable[[], MalaEventSink],
        quality_gate: Callable[[], GateChecker],
        repo_path: Path,
        session_log_paths: dict[str, Path],
        review_log_paths: dict[str, str],
        get_per_issue_spec: GetPerIssueSpec,
        is_verbose: IsVerboseCheck,
    ) -> None:
        """Initialize the factory with dependencies.

        Args:
            gate_async_runner: Protocol for running async gate checks.
            review_runner: Runner for Cerberus code review.
            log_provider: Callable returning the log provider (late-bound).
            event_sink: Callable returning the event sink (late-bound).
            quality_gate: Callable returning the gate checker (late-bound).
            repo_path: Repository path for git operations.
            session_log_paths: Dict to store session log paths (mutated).
            review_log_paths: Dict to store review log paths (mutated).
            get_per_issue_spec: Callable to get current per-issue spec.
            is_verbose: Callable to check verbose mode.

        Note:
            log_provider, event_sink, and quality_gate are callables to support
            late-bound lookups. This allows tests to patch orchestrator attributes
            after factory construction and have the patches take effect.
        """
        self._gate_async_runner = gate_async_runner
        self._review_runner = review_runner
        self._get_log_provider = log_provider
        self._get_event_sink = event_sink
        self._get_quality_gate = quality_gate
        self._repo_path = repo_path
        self._session_log_paths = session_log_paths
        self._review_log_paths = review_log_paths
        self._get_per_issue_spec = get_per_issue_spec
        self._is_verbose = is_verbose

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
        from src.infra.git_utils import get_git_commit_async, get_issue_commits_async
        from src.pipeline.review_runner import NoProgressInput, ReviewInput

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult | GateResultProtocol, int]:
            return await self._gate_async_runner.run_gate_async(
                issue_id, log_path, retry_state
            )

        async def on_review_check(
            issue_id: str,
            issue_desc: str | None,
            baseline: str | None,
            session_id: str | None,
            retry_state: RetryState,
        ) -> ReviewResult | ReviewResultProtocol:
            current_head = await get_git_commit_async(self._repo_path)
            self._review_runner.config.capture_session_log = self._is_verbose()
            commit_shas = await get_issue_commits_async(
                self._repo_path,
                issue_id,
                since_timestamp=retry_state.baseline_timestamp,
            )
            review_input = ReviewInput(
                issue_id=issue_id,
                repo_path=self._repo_path,
                commit_sha=current_head,
                issue_description=issue_desc,
                baseline_commit=baseline,
                commit_shas=commit_shas or None,
                claude_session_id=session_id,
            )
            output = await self._review_runner.run_review(review_input)
            if output.session_log_path:
                self._review_log_paths[issue_id] = output.session_log_path
            return output.result

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
                spec=self._get_per_issue_spec(),
            )
            return self._review_runner.check_no_progress(no_progress_input)

        def get_log_path(session_id: str) -> Path:
            log_path = self._get_log_provider().get_log_path(
                self._repo_path, session_id
            )
            self._session_log_paths[issue_id] = log_path
            return log_path

        def get_log_offset(log_path: Path, start_offset: int) -> int:
            return self._get_quality_gate().get_log_end_offset(log_path, start_offset)

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


# Protocol for getting per-issue spec
class GetPerIssueSpec(Protocol):
    """Protocol for getting the current per-issue validation spec."""

    def __call__(self) -> ValidationSpec | None:
        """Return the current per-issue spec, or None if not set."""
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

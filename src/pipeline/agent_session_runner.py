"""AgentSessionRunner: Agent session execution pipeline stage.

Extracted from MalaOrchestrator to separate SDK-specific session handling from
orchestration logic. This module handles:
- SDK streaming and message processing
- Session lifecycle transitions via ImplementerLifecycle
- Session metadata tracking (session ID, log paths)

The AgentSessionRunner receives explicit inputs and returns explicit outputs,
making it testable without SDK dependencies when using the SDKClientProtocol.

Design principles:
- Protocol-based SDK client for testability
- Explicit input/output types for clarity
- Lifecycle state machine drives policy decisions
- Callbacks for external operations (gate checks, reviews)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
    TypeVar,
    cast,
)

from src.infra.hooks import (
    FileReadCache,
    LintCache,
    block_dangerous_commands,
    block_mala_disallowed_tools,
    make_file_read_cache_hook,
    make_lint_cache_hook,
    make_lock_enforcement_hook,
    make_lock_event_hook,
    make_lock_wait_hook,
    make_stop_hook,
)
from src.infra.mcp import get_disallowed_tools, get_mcp_servers
from src.domain.lifecycle import (
    Effect,
    ImplementerLifecycle,
    LifecycleConfig,
    LifecycleContext,
    LifecycleState,
)
from src.domain.prompts import (
    build_continuation_prompt,
    extract_checkpoint,
    get_default_validation_commands as _get_default_validation_commands,
)
from src.infra.clients.cerberus_review import format_review_issues
from src.infra.clients.review_output_parser import ReviewResult
from src.infra.tools.env import SCRIPTS_DIR, get_lock_dir

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import Self

    from src.core.protocols import (
        MalaEventSink,
        SDKClientFactoryProtocol,
        SDKClientProtocol,
    )
    from src.domain.deadlock import DeadlockMonitor
    from src.domain.lifecycle import (
        GateOutcome,
        RetryState,
        ReviewIssue,
        ReviewOutcome,
        TransitionResult,
    )
    from src.core.models import IssueResolution
    from src.core.protocols import ReviewIssueProtocol
    from src.infra.telemetry import TelemetrySpan
    from src.domain.validation.config import PromptValidationCommands


# Module-level logger for idle retry messages
logger = logging.getLogger(__name__)

# Timeout for disconnect() call
DISCONNECT_TIMEOUT = 10.0


class IdleTimeoutError(Exception):
    """Raised when the SDK response stream is idle for too long."""


class ContextPressureError(Exception):
    """Raised when context usage exceeds the restart threshold.

    This exception signals that the agent session should be checkpointed
    and restarted with a fresh context to avoid context exhaustion.

    Attributes:
        session_id: SDK session ID for checkpoint query.
        input_tokens: Current input token count.
        output_tokens: Current output token count.
        cache_read_tokens: Current cache read token count.
        pressure_ratio: Ratio of usage to limit (e.g., 0.92 = 92%).
    """

    def __init__(
        self,
        session_id: str,
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        pressure_ratio: float,
    ) -> None:
        self.session_id = session_id
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_tokens = cache_read_tokens
        self.pressure_ratio = pressure_ratio
        super().__init__(
            f"Context pressure {pressure_ratio:.1%} exceeds threshold "
            f"(input={input_tokens}, output={output_tokens}, session={session_id})"
        )


_T = TypeVar("_T")


class IdleTimeoutStream(Generic[_T]):
    """Wrap an async iterator with idle timeout detection.

    Raises IdleTimeoutError if no message received within timeout,
    unless pending_tool_ids is non-empty (tool execution in progress).
    """

    def __init__(
        self,
        stream: AsyncIterator[_T],
        timeout_seconds: float | None,
        pending_tool_ids: set[str],
    ) -> None:
        self._stream: AsyncIterator[_T] = stream
        self._timeout_seconds = timeout_seconds
        self._pending_tool_ids = pending_tool_ids

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> _T:
        if self._timeout_seconds is None:
            return await self._stream.__anext__()
        # Disable timeout if tools are pending (execution in progress)
        current_timeout = None if self._pending_tool_ids else self._timeout_seconds
        try:
            return await asyncio.wait_for(
                self._stream.__anext__(),
                timeout=current_timeout,
            )
        except TimeoutError as exc:
            raise IdleTimeoutError(
                f"SDK stream idle for {self._timeout_seconds:.0f} seconds"
            ) from exc


# Type aliases for callbacks
GateCheckCallback = Callable[
    [str, Path, "RetryState"],
    Coroutine[Any, Any, tuple["GateOutcome", int]],
]
ReviewCheckCallback = Callable[
    [str, str | None, str | None, str | None, "RetryState"],
    Coroutine[Any, Any, "ReviewOutcome"],
]
ReviewNoProgressCallback = Callable[
    [Path, int, str | None, str | None],
    bool,
]
LogOffsetCallback = Callable[[Path, int], int]

# Callbacks for SDK message streaming events
ToolUseCallback = Callable[[str, str, dict[str, Any] | None], None]
AgentTextCallback = Callable[[str, str], None]


@dataclass
class MessageIterationState:
    """Mutable state for message iteration within a session.

    Used to track state that evolves during SDK message streaming
    and idle retry handling.

    Attributes:
        session_id: SDK session ID (updated when ResultMessage received).
        pending_session_id: Session ID to use for resuming after idle timeout.
        tool_calls_this_turn: Number of tool calls in the current turn.
        idle_retry_count: Number of idle timeout retries attempted.
        pending_tool_ids: Set of tool IDs awaiting results.
        pending_lint_commands: Map of tool_use_id to (lint_type, command).
        first_message_received: Whether any message was received in current turn.
    """

    session_id: str | None = None
    pending_session_id: str | None = None
    tool_calls_this_turn: int = 0
    idle_retry_count: int = 0
    pending_tool_ids: set[str] = field(default_factory=set)
    pending_lint_commands: dict[str, tuple[str, str]] = field(default_factory=dict)
    first_message_received: bool = False


@dataclass
class MessageIterationResult:
    """Result from a message iteration.

    Attributes:
        success: Whether the iteration completed successfully.
        session_id: Updated session ID (if received).
        pending_query: Next query to send (for retries), or None if complete.
        pending_session_id: Session ID to use for next query.
        idle_retry_count: Updated idle retry count.
    """

    success: bool
    session_id: str | None = None
    pending_query: str | None = None
    pending_session_id: str | None = None
    idle_retry_count: int = 0


@dataclass
class SessionConfig:
    """Derived configuration for session execution.

    Computed from AgentSessionConfig during initialization.

    Attributes:
        agent_id: Unique agent ID for this session.
        options: SDK client options.
        lint_cache: Lint command result cache.
        log_file_wait_timeout: Timeout for log file availability.
        log_file_poll_interval: Poll interval for log file.
        idle_timeout_seconds: Idle timeout for SDK stream.
    """

    agent_id: str
    options: object
    lint_cache: LintCache
    log_file_wait_timeout: float
    log_file_poll_interval: float
    idle_timeout_seconds: float | None


@dataclass
class SessionExecutionState:
    """Mutable state for session execution.

    Bundles all session state that evolves during execution, including
    lifecycle context, session identifiers, and log paths.

    Attributes:
        lifecycle: Lifecycle state machine.
        lifecycle_ctx: Lifecycle context with retry state.
        session_id: SDK session ID (updated when ResultMessage received).
        log_path: Path to session log file.
        final_result: Final result text from session.
        cerberus_review_log_path: Path to Cerberus review log (if any).
        msg_state: Message iteration state.
    """

    lifecycle: ImplementerLifecycle
    lifecycle_ctx: LifecycleContext
    session_id: str | None = None
    log_path: Path | None = None
    final_result: str = ""
    cerberus_review_log_path: str | None = None
    msg_state: MessageIterationState = field(default_factory=MessageIterationState)


@dataclass
class PromptProvider:
    """Provider for session prompts, injected at construction time.

    Holds prompt templates loaded from files. This keeps file I/O at the
    orchestration boundary and allows tests to inject custom prompts.

    Attributes:
        gate_followup: Template for gate failure follow-up prompts.
        review_followup: Template for review issues follow-up prompts.
        idle_resume: Template for idle timeout resume prompts.
        checkpoint_request: Prompt to request checkpoint from agent.
        continuation: Template for continuation prompt with checkpoint.
    """

    gate_followup: str
    review_followup: str
    idle_resume: str
    checkpoint_request: str = ""
    continuation: str = ""


@dataclass
class AgentSessionConfig:
    """Configuration for agent session execution.

    Bundles all configuration needed to run an agent session.

    Attributes:
        repo_path: Path to the repository.
        timeout_seconds: Session timeout in seconds.
        prompts: Provider for session prompts (gate, review, idle resume).
        max_gate_retries: Maximum gate retry attempts.
        max_review_retries: Maximum review retry attempts.
        review_enabled: Whether Cerberus external review is enabled.
            When enabled, code changes are reviewed after the gate passes.
        log_file_wait_timeout: Seconds to wait for log file after session
            completes. Default 60s allows time for SDK to flush logs under load.
        idle_timeout_seconds: Seconds to wait for SDK message when not waiting
            for tool execution. If None, defaults to min(900, max(300, timeout_seconds * 0.2))
            which scales with session timeout (300-900s range). During tool execution
            (after ToolUseBlock, before result), timeout is disabled. Set to 0 to
            disable timeout entirely.
        lint_tools: Set of lint tool names for LintCache. If None, uses default
            lint tools. Populated from ValidationSpec commands.
        prompt_validation_commands: Validation commands for prompt templates.
            If None, uses default Python/uv commands.
        context_restart_threshold: Ratio (0.0-1.0) at which to raise
            ContextPressureError. Default 0.90 (90% of context_limit).
        context_limit: Maximum context tokens. Default 200K for Claude.
    """

    repo_path: Path
    timeout_seconds: int
    prompts: PromptProvider
    max_gate_retries: int = 3
    max_review_retries: int = 3
    review_enabled: bool = True
    log_file_wait_timeout: float = 60.0
    idle_timeout_seconds: float | None = None
    max_idle_retries: int = 2
    idle_retry_backoff: tuple[float, ...] = (0.0, 5.0, 15.0)
    lint_tools: frozenset[str] | None = None
    prompt_validation_commands: PromptValidationCommands | None = None
    context_restart_threshold: float = 0.90
    context_limit: int = 200_000
    deadlock_monitor: DeadlockMonitor | None = None


@dataclass
class AgentSessionInput:
    """Input for running an agent session.

    Bundles all data needed to start a session for an issue.

    Attributes:
        issue_id: The issue ID being worked on.
        prompt: The initial prompt to send to the agent.
        baseline_commit: Optional baseline commit for diff comparison.
        issue_description: Issue description for scope verification.
        agent_id: Optional pre-generated agent ID for lock management.
    """

    issue_id: str
    prompt: str
    baseline_commit: str | None = None
    issue_description: str | None = None
    agent_id: str | None = None


@dataclass
class AgentSessionOutput:
    """Output from an agent session.

    Contains all results and metadata from a completed session.

    Attributes:
        success: Whether the session completed successfully.
        summary: Human-readable summary of the outcome.
        session_id: Claude SDK session ID (if available).
        log_path: Path to the session log file (if available).
        gate_attempts: Number of gate retry attempts.
        review_attempts: Number of review retry attempts.
        resolution: Issue resolution outcome (if any).
        duration_seconds: Total session duration.
        agent_id: The agent ID used for this session.
        review_log_path: Path to Cerberus review session log (if any).
        low_priority_review_issues: P2/P3 review issues to track as beads issues.
    """

    success: bool
    summary: str
    session_id: str | None = None
    log_path: Path | None = None
    gate_attempts: int = 1
    review_attempts: int = 0
    resolution: IssueResolution | None = None
    duration_seconds: float = 0.0
    agent_id: str = ""
    review_log_path: str | None = None
    low_priority_review_issues: list[ReviewIssueProtocol] | None = None


@dataclass
class SessionCallbacks:
    """Callbacks for external actions during session execution.

    These callbacks allow the orchestrator to inject behavior for
    actions that require external state (gate checks, review, etc.)
    without coupling AgentSessionRunner to those implementations.

    Attributes:
        on_gate_check: Async callback to run quality gate check.
            Args: (issue_id, log_path, retry_state) -> (GateResult, new_offset)
        on_review_check: Async callback to run external review (Cerberus).
            Args: (issue_id, issue_description, baseline_commit, session_id, retry_state) -> ReviewOutcome
        on_review_no_progress: Sync callback to check if no progress on review retry.
            Args: (log_path, log_offset, prev_commit, curr_commit) -> bool
        get_log_path: Callback to get log path from session ID.
            Args: (session_id) -> Path
        get_log_offset: Callback to get log end offset.
            Args: (log_path, start_offset) -> int
        on_abort: Callback when fatal error requires run abort.
            Args: (reason) -> None
        on_tool_use: Callback for SDK tool use events.
            Args: (agent_id, tool_name, arguments) -> None
        on_agent_text: Callback for SDK text output events.
            Args: (agent_id, text) -> None
    """

    on_gate_check: GateCheckCallback | None = None
    on_review_check: ReviewCheckCallback | None = None
    on_review_no_progress: ReviewNoProgressCallback | None = None
    get_log_path: Callable[[str], Path] | None = None
    get_log_offset: LogOffsetCallback | None = None
    on_abort: Callable[[str], None] | None = None
    on_tool_use: ToolUseCallback | None = None
    on_agent_text: AgentTextCallback | None = None


@dataclass
class ReviewEffectResult:
    """Result from _handle_review_effect method.

    Encapsulates the multi-value return from review effect handling
    with named fields for clarity.
    """

    pending_query: str | None
    """Query to send for review retry, or None if no retry needed."""

    should_break: bool
    """Whether the caller should break out of the message iteration loop."""

    cerberus_log_path: str | None
    """Path to Cerberus review log file, if captured."""

    transition_result: TransitionResult
    """Lifecycle transition result."""


def _emit_review_result_events(
    event_sink: MalaEventSink | None,
    input: AgentSessionInput,
    result: TransitionResult,
    review_result: ReviewOutcome,
    lifecycle_ctx: LifecycleContext,
    max_review_retries: int,
    blocking_count: int,
) -> None:
    """Emit events based on review result transition.

    Handles event emission for:
    - COMPLETE_SUCCESS: on_review_passed
    - RUN_REVIEW (parse error): on_warning
    - SEND_REVIEW_RETRY: on_review_retry with blocking_count
    """
    if event_sink is None:
        return

    if result.effect == Effect.COMPLETE_SUCCESS:
        event_sink.on_review_passed(
            input.issue_id,
            issue_id=input.issue_id,
        )
        return

    if result.effect == Effect.RUN_REVIEW:
        error_detail = review_result.parse_error or "unknown error"
        event_sink.on_warning(
            f"Review tool error: {error_detail}; retrying",
            agent_id=input.issue_id,
        )
        return

    if result.effect == Effect.SEND_REVIEW_RETRY:
        logger.debug(
            "Session %s: SEND_REVIEW_RETRY triggered "
            "(attempt %d/%d, %d blocking issues)",
            input.issue_id,
            lifecycle_ctx.retry_state.review_attempt,
            max_review_retries,
            blocking_count,
        )
        event_sink.on_review_retry(
            input.issue_id,
            lifecycle_ctx.retry_state.review_attempt,
            max_review_retries,
            error_count=blocking_count or None,
            parse_error=review_result.parse_error,
            issue_id=input.issue_id,
        )


def _emit_gate_passed_events(
    event_sink: MalaEventSink | None,
    issue_id: str,
    review_attempt: int,
) -> None:
    """Emit gate passed events when first entering review.

    Only emits events on the first review attempt (review_attempt == 1),
    indicating the gate has just passed.

    Args:
        event_sink: Event sink to emit to, or None to skip emission.
        issue_id: The issue identifier.
        review_attempt: Current review attempt number (1-based).
    """
    if review_attempt != 1 or event_sink is None:
        return

    event_sink.on_gate_passed(
        issue_id,
        issue_id=issue_id,
    )
    event_sink.on_validation_result(
        issue_id,
        passed=True,
        issue_id=issue_id,
    )


def _count_blocking_issues(issues: list[ReviewIssue] | None) -> int:
    """Count issues with priority <= 1 (P0 or P1).

    Args:
        issues: List of review issues, or None.

    Returns:
        Number of blocking (high-priority) issues.
    """
    if not issues:
        return 0
    return sum(1 for i in issues if i.priority is not None and i.priority <= 1)


def _make_review_effect_result(
    effect: Effect,
    cerberus_log_path: str | None,
    transition_result: TransitionResult,
    pending_query: str | None = None,
) -> ReviewEffectResult:
    """Build ReviewEffectResult based on effect type.

    Args:
        effect: The lifecycle effect to handle.
        cerberus_log_path: Path to Cerberus review log.
        transition_result: Lifecycle transition result.
        pending_query: Query string for SEND_REVIEW_RETRY effect.

    Returns:
        ReviewEffectResult with appropriate should_break flag.
    """
    should_break = effect in (Effect.COMPLETE_SUCCESS, Effect.COMPLETE_FAILURE)
    return ReviewEffectResult(
        pending_query=pending_query,
        should_break=should_break,
        cerberus_log_path=cerberus_log_path,
        transition_result=transition_result,
    )


def _build_review_retry_prompt(
    review_result: ReviewOutcome,
    lifecycle_ctx: LifecycleContext,
    issue_id: str,
    repo_path: Path,
    max_review_retries: int,
    review_followup_template: str,
) -> str:
    """Build the follow-up prompt for review retry.

    Args:
        review_result: The review outcome with issues to address.
        lifecycle_ctx: Lifecycle context with retry state.
        issue_id: The issue identifier.
        repo_path: Repository path for formatting issue paths.
        max_review_retries: Maximum number of review retries.
        review_followup_template: Template for review follow-up prompts.

    Returns:
        Formatted prompt string for the agent to address review issues.
    """
    review_issues_text = format_review_issues(
        review_result.issues,  # type: ignore[arg-type]
        base_path=repo_path,
    )
    return review_followup_template.format(
        attempt=lifecycle_ctx.retry_state.review_attempt,
        max_attempts=max_review_retries,
        review_issues=review_issues_text,
        issue_id=issue_id,
    )


@dataclass
class AgentSessionRunner:
    """Runs agent sessions with lifecycle management.

    This class encapsulates the SDK session execution logic that was previously
    inline in MalaOrchestrator.run_implementer. It manages:
    - SDK client creation and message streaming
    - Lifecycle state transitions
    - Hook setup (lock enforcement, lint cache, etc.)
    - Message logging and telemetry

    The runner uses callbacks for external operations (gate checks, reviews)
    to avoid tight coupling with the orchestrator's dependencies.

    Usage:
        runner = AgentSessionRunner(
            config=AgentSessionConfig(repo_path=repo_path, ...),
            callbacks=SessionCallbacks(on_gate_check=..., ...),
        )
        output = await runner.run_session(input)

    Attributes:
        config: Session configuration.
        callbacks: Callbacks for external operations.
        sdk_client_factory: Factory for creating SDK clients (injectable for testing).
        event_sink: Optional event sink for structured logging.
    """

    config: AgentSessionConfig
    callbacks: SessionCallbacks = field(default_factory=SessionCallbacks)
    sdk_client_factory: SDKClientFactoryProtocol | None = None
    event_sink: MalaEventSink | None = None

    def _build_agent_env(self, agent_id: str) -> dict[str, str]:
        """Build environment variables for agent execution.

        Args:
            agent_id: The agent ID for lock management.

        Returns:
            Environment dict with PATH, LOCK_DIR, AGENT_ID, etc.
        """
        return {
            **os.environ,
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
            "LOCK_DIR": str(get_lock_dir()),
            "AGENT_ID": agent_id,
            "REPO_NAMESPACE": str(self.config.repo_path),
            "MCP_TIMEOUT": "300000",
        }

    def _build_hooks(
        self,
        agent_id: str,
        file_read_cache: FileReadCache,
        lint_cache: LintCache,
    ) -> tuple[list[object], list[object], list[object]]:
        """Build PreToolUse, PostToolUse, and Stop hooks for the session.

        Args:
            agent_id: The agent ID for lock enforcement.
            file_read_cache: Cache for blocking redundant file reads.
            lint_cache: Cache for blocking redundant lint commands.

        Returns:
            Tuple of (pre_tool_hooks, post_tool_hooks, stop_hooks).
        """
        pre_tool_hooks: list[object] = [
            block_dangerous_commands,
            block_mala_disallowed_tools,
            make_lock_enforcement_hook(agent_id, str(self.config.repo_path)),
            make_file_read_cache_hook(file_read_cache),
            make_lint_cache_hook(lint_cache),
        ]
        post_tool_hooks: list[object] = []
        stop_hooks: list[object] = [make_stop_hook(agent_id)]

        # Add lock event hooks if deadlock monitor is configured
        if self.config.deadlock_monitor is not None:
            monitor = self.config.deadlock_monitor
            # PreToolUse hook for real-time WAITING detection on lock-wait.sh
            pre_tool_hooks.append(
                make_lock_wait_hook(
                    agent_id=agent_id,
                    emit_event=monitor.handle_event,
                    repo_namespace=str(self.config.repo_path),
                )
            )
            # PostToolUse hook for ACQUIRED/RELEASED events
            post_tool_hooks.append(
                make_lock_event_hook(
                    agent_id=agent_id,
                    emit_event=monitor.handle_event,
                    repo_namespace=str(self.config.repo_path),
                )
            )

        return pre_tool_hooks, post_tool_hooks, stop_hooks

    def _build_sdk_options(
        self,
        pre_tool_hooks: list[object],
        post_tool_hooks: list[object],
        stop_hooks: list[object],
        agent_env: dict[str, str],
    ) -> object:
        """Build SDK client options for the session.

        Args:
            pre_tool_hooks: PreToolUse hooks for the session.
            post_tool_hooks: PostToolUse hooks for the session.
            stop_hooks: Stop hooks for the session.
            agent_env: Environment variables for the agent.

        Returns:
            ClaudeAgentOptions configured for this session.
        """
        # Build hooks dict using factory (required)
        if self.sdk_client_factory is None:
            raise RuntimeError(
                "sdk_client_factory is required. Use SDKClientFactory from "
                "src.infra.sdk_adapter or inject via orchestration wiring."
            )
        make_matcher = self.sdk_client_factory.create_hook_matcher
        hooks_dict: dict[str, list[object]] = {
            "PreToolUse": [make_matcher(None, pre_tool_hooks)],
            "Stop": [make_matcher(None, stop_hooks)],
        }
        if post_tool_hooks:
            hooks_dict["PostToolUse"] = [make_matcher(None, post_tool_hooks)]

        return self.sdk_client_factory.create_options(
            cwd=str(self.config.repo_path),
            mcp_servers=get_mcp_servers(self.config.repo_path),
            disallowed_tools=get_disallowed_tools(),
            env=agent_env,
            hooks=hooks_dict,
        )

    def _initialize_session(
        self,
        input: AgentSessionInput,
        agent_id: str | None = None,
    ) -> tuple[SessionConfig, SessionExecutionState]:
        """Initialize session components and state.

        Creates lifecycle, hooks, SDK options, and mutable state for session.

        Args:
            input: Session input with issue_id, prompt, etc.
            agent_id: Optional agent ID. If None, generates a new one.
                Pass an existing ID to preserve lock continuity across restarts.

        Returns:
            Tuple of (SessionConfig, SessionExecutionState).
        """
        # Generate agent ID if not provided
        if agent_id is None:
            agent_id = f"{input.issue_id}-{uuid.uuid4().hex[:8]}"

        # Initialize lifecycle
        lifecycle_config = LifecycleConfig(
            max_gate_retries=self.config.max_gate_retries,
            max_review_retries=self.config.max_review_retries,
            review_enabled=self.config.review_enabled,
        )
        lifecycle = ImplementerLifecycle(lifecycle_config)
        lifecycle_ctx = LifecycleContext()
        lifecycle_ctx.retry_state.baseline_timestamp = int(time.time())

        # Build session components
        file_read_cache = FileReadCache()
        lint_cache = LintCache(
            repo_path=self.config.repo_path,
            lint_tools=self.config.lint_tools,
        )
        pre_tool_hooks, post_tool_hooks, stop_hooks = self._build_hooks(
            agent_id, file_read_cache, lint_cache
        )
        agent_env = self._build_agent_env(agent_id)
        options = self._build_sdk_options(
            pre_tool_hooks, post_tool_hooks, stop_hooks, agent_env
        )

        # Calculate idle timeout
        idle_timeout_seconds = self.config.idle_timeout_seconds
        if idle_timeout_seconds is None:
            derived = self.config.timeout_seconds * 0.2
            idle_timeout_seconds = min(900.0, max(300.0, derived))
        if idle_timeout_seconds <= 0:
            idle_timeout_seconds = None

        session_config = SessionConfig(
            agent_id=agent_id,
            options=options,
            lint_cache=lint_cache,
            log_file_wait_timeout=self.config.log_file_wait_timeout,
            log_file_poll_interval=0.5,
            idle_timeout_seconds=idle_timeout_seconds,
        )

        exec_state = SessionExecutionState(
            lifecycle=lifecycle,
            lifecycle_ctx=lifecycle_ctx,
        )

        return session_config, exec_state

    async def _run_lifecycle_loop(
        self,
        input: AgentSessionInput,
        session_cfg: SessionConfig,
        state: SessionExecutionState,
        tracer: TelemetrySpan | None = None,
    ) -> None:
        """Run the main lifecycle loop.

        Executes the message iteration, gate, and review loop until
        terminal state is reached.

        Args:
            input: Session input with issue_id, prompt, etc.
            session_cfg: Derived session configuration.
            state: Mutable session execution state.
            tracer: Optional telemetry span context.
        """
        lifecycle = state.lifecycle
        lifecycle_ctx = state.lifecycle_ctx

        # Start lifecycle
        lifecycle.start()
        if self.event_sink is not None:
            self.event_sink.on_lifecycle_state(input.issue_id, lifecycle.state.name)

        pending_query: str | None = input.prompt
        result: TransitionResult | None = None

        while not lifecycle.is_terminal:
            # === QUERY + MESSAGE ITERATION ===
            if pending_query is not None:
                iter_result = await self._run_message_iteration(
                    query=pending_query,
                    issue_id=input.issue_id,
                    options=session_cfg.options,
                    state=state.msg_state,
                    lifecycle_ctx=lifecycle_ctx,
                    lint_cache=session_cfg.lint_cache,
                    idle_timeout_seconds=session_cfg.idle_timeout_seconds,
                    tracer=tracer,
                )
                if iter_result.session_id is not None:
                    state.session_id = iter_result.session_id
                pending_query = None

                result = lifecycle.on_messages_complete(
                    lifecycle_ctx, has_session_id=bool(state.session_id)
                )
                if self.event_sink is not None:
                    self.event_sink.on_lifecycle_state(
                        input.issue_id, lifecycle.state.name
                    )
                if result.effect == Effect.COMPLETE_FAILURE:
                    state.final_result = lifecycle_ctx.final_result
                    break
            else:
                assert result is not None, (
                    "Bug: entered loop without pending_query but result not set"
                )

            # Handle WAIT_FOR_LOG
            if result.effect == Effect.WAIT_FOR_LOG:
                state.log_path, result = await self._handle_log_waiting(
                    state.session_id,
                    input.issue_id,
                    state.log_path,
                    lifecycle,
                    lifecycle_ctx,
                    session_cfg.log_file_wait_timeout,
                    session_cfg.log_file_poll_interval,
                )
                if lifecycle.state == LifecycleState.FAILED:
                    state.final_result = lifecycle_ctx.final_result
                    break

            # Handle RUN_GATE
            if result.effect == Effect.RUN_GATE:
                pending_query, gate_trans = await self._handle_gate_check(
                    input, state, lifecycle, lifecycle_ctx
                )
                if pending_query is not None:
                    state.msg_state.pending_session_id = state.session_id
                    state.msg_state.idle_retry_count = 0
                    continue
                if lifecycle.is_terminal:
                    state.final_result = lifecycle_ctx.final_result
                    break
                # Update result with gate transition for next effect check
                result = gate_trans

            # Handle RUN_REVIEW
            if result.effect == Effect.RUN_REVIEW:
                pending_query, review_result = await self._handle_review_check(
                    input, state, lifecycle, lifecycle_ctx
                )
                if review_result is not None:
                    result = review_result
                if pending_query is not None:
                    state.msg_state.pending_session_id = state.session_id
                    state.msg_state.idle_retry_count = 0
                if lifecycle.is_terminal:
                    state.final_result = lifecycle_ctx.final_result
                    break
                continue

        state.final_result = lifecycle_ctx.final_result

    async def _handle_gate_check(
        self,
        input: AgentSessionInput,
        state: SessionExecutionState,
        lifecycle: ImplementerLifecycle,
        lifecycle_ctx: LifecycleContext,
    ) -> tuple[str | None, TransitionResult]:
        """Handle RUN_GATE effect - emit events and run gate check.

        Args:
            input: Session input with issue_id.
            state: Session execution state.
            lifecycle: Lifecycle state machine.
            lifecycle_ctx: Lifecycle context.

        Returns:
            Tuple of (pending query for retry or None, transition result).
        """
        # Emit validation started BEFORE the gate check
        if self.event_sink is not None:
            self.event_sink.on_validation_started(
                input.issue_id, issue_id=input.issue_id
            )
            self.event_sink.on_gate_started(
                input.issue_id,
                lifecycle_ctx.retry_state.gate_attempt,
                self.config.max_gate_retries,
                issue_id=input.issue_id,
            )

        if self.callbacks.on_gate_check is None:
            raise ValueError("on_gate_check callback must be set")

        assert state.log_path is not None
        gate_result, new_offset = await self.callbacks.on_gate_check(
            input.issue_id, state.log_path, lifecycle_ctx.retry_state
        )

        retry_query, should_break, trans_result = self._handle_gate_effect(
            input, gate_result, lifecycle, lifecycle_ctx, new_offset
        )
        if should_break:
            return None, trans_result
        if retry_query is not None:
            if state.session_id is None:
                raise IdleTimeoutError(
                    "Cannot retry gate: session_id not received from SDK"
                )
            logger.debug(
                "Session %s: queueing gate retry prompt (%d chars, session_id=%s)",
                input.issue_id,
                len(retry_query),
                state.session_id[:8],
            )
            return retry_query, trans_result
        return None, trans_result

    async def _handle_review_check(
        self,
        input: AgentSessionInput,
        state: SessionExecutionState,
        lifecycle: ImplementerLifecycle,
        lifecycle_ctx: LifecycleContext,
    ) -> tuple[str | None, TransitionResult | None]:
        """Handle RUN_REVIEW effect - run review and process result.

        Args:
            input: Session input with issue_id, description, baseline.
            state: Session execution state.
            lifecycle: Lifecycle state machine.
            lifecycle_ctx: Lifecycle context.

        Returns:
            Tuple of (pending_query for retry, transition_result).
        """
        assert state.log_path is not None
        review_effect = await self._handle_review_effect(
            input, state.log_path, state.session_id, lifecycle, lifecycle_ctx
        )
        if review_effect.cerberus_log_path is not None:
            state.cerberus_review_log_path = review_effect.cerberus_log_path
        if review_effect.should_break:
            return None, review_effect.transition_result
        if review_effect.pending_query is not None:
            if state.session_id is None:
                raise IdleTimeoutError(
                    "Cannot retry review: session_id not received from SDK"
                )
            logger.debug(
                "Session %s: queueing review retry prompt (%d chars, session_id=%s)",
                input.issue_id,
                len(review_effect.pending_query),
                state.session_id[:8],
            )
            return review_effect.pending_query, review_effect.transition_result
        return None, review_effect.transition_result

    def _build_session_output(
        self,
        session_cfg: SessionConfig,
        state: SessionExecutionState,
        duration: float,
    ) -> AgentSessionOutput:
        """Build session output from execution state.

        Args:
            session_cfg: Session configuration with agent_id.
            state: Session execution state.
            duration: Total session duration in seconds.

        Returns:
            AgentSessionOutput with all results and metadata.
        """
        return AgentSessionOutput(
            success=state.lifecycle_ctx.success,
            summary=state.final_result,
            session_id=state.session_id,
            log_path=state.log_path,
            gate_attempts=state.lifecycle_ctx.retry_state.gate_attempt,
            review_attempts=state.lifecycle_ctx.retry_state.review_attempt,
            resolution=state.lifecycle_ctx.resolution,
            duration_seconds=duration,
            agent_id=session_cfg.agent_id,
            review_log_path=state.cerberus_review_log_path,
            low_priority_review_issues=cast(
                "list[ReviewIssueProtocol] | None",
                state.lifecycle_ctx.low_priority_review_issues or None,
            ),
        )

    async def _handle_log_waiting(
        self,
        session_id: str | None,
        issue_id: str,
        log_path: Path | None,
        lifecycle: ImplementerLifecycle,
        lifecycle_ctx: LifecycleContext,
        log_file_wait_timeout: float,
        log_file_poll_interval: float = 0.5,
    ) -> tuple[Path | None, TransitionResult]:
        """Handle WAIT_FOR_LOG effect - wait for log file to become available.

        Args:
            session_id: Current SDK session ID.
            issue_id: Issue ID for logging.
            log_path: Current log path (may be None).
            lifecycle: Lifecycle state machine.
            lifecycle_ctx: Lifecycle context.
            log_file_wait_timeout: Max seconds to wait for log file.
            log_file_poll_interval: Seconds between poll attempts.

        Returns:
            Tuple of (updated log_path, TransitionResult from log ready/timeout).
        """
        if self.callbacks.get_log_path is None:
            raise ValueError("get_log_path callback must be set")
        if session_id is None:
            raise ValueError("session_id must be set before waiting for log")
        new_log_path = self.callbacks.get_log_path(session_id)

        # Reset log_offset if log file changed (new session started)
        # This prevents using a stale offset from a previous session's
        # larger log file when parsing a new, smaller log file
        if log_path is not None and new_log_path != log_path:
            logger.info(
                "Session %s: log path changed from %s to %s, "
                "resetting log_offset from %d to 0",
                issue_id,
                log_path.name,
                new_log_path.name,
                lifecycle_ctx.retry_state.log_offset,
            )
            lifecycle_ctx.retry_state.log_offset = 0

        log_path = new_log_path
        if self.event_sink is not None:
            self.event_sink.on_log_waiting(issue_id)

        # Wait for log file
        wait_elapsed = 0.0
        while not log_path.exists():
            if wait_elapsed >= log_file_wait_timeout:
                result = lifecycle.on_log_timeout(lifecycle_ctx, str(log_path))
                if self.event_sink is not None:
                    self.event_sink.on_log_timeout(issue_id, str(log_path))
                return log_path, result
            await asyncio.sleep(log_file_poll_interval)
            wait_elapsed += log_file_poll_interval

        if log_path.exists():
            if self.event_sink is not None:
                self.event_sink.on_log_ready(issue_id)
        result = lifecycle.on_log_ready(lifecycle_ctx)
        return log_path, result

    def _handle_gate_effect(
        self,
        input: AgentSessionInput,
        gate_result: GateOutcome,
        lifecycle: ImplementerLifecycle,
        lifecycle_ctx: LifecycleContext,
        new_offset: int,
    ) -> tuple[str | None, bool, TransitionResult]:
        """Handle RUN_GATE effect - process gate result and emit events.

        Args:
            input: Session input with issue_id.
            gate_result: Result from gate check callback.
            lifecycle: Lifecycle state machine.
            lifecycle_ctx: Lifecycle context.
            new_offset: New log offset after gate check.

        Returns:
            Tuple of (pending_query for retry or None, should_break, transition_result).
        """
        result = lifecycle.on_gate_result(lifecycle_ctx, gate_result, new_offset)

        if result.effect == Effect.COMPLETE_SUCCESS:
            _emit_gate_passed_events(self.event_sink, input.issue_id, review_attempt=1)
            return None, True, result  # break

        if result.effect == Effect.COMPLETE_FAILURE:
            if self.event_sink is not None:
                self.event_sink.on_gate_failed(
                    input.issue_id,
                    lifecycle_ctx.retry_state.gate_attempt,
                    self.config.max_gate_retries,
                    issue_id=input.issue_id,
                )
                self.event_sink.on_gate_result(
                    input.issue_id,
                    passed=False,
                    failure_reasons=list(gate_result.failure_reasons),
                    issue_id=input.issue_id,
                )
                self.event_sink.on_validation_result(
                    input.issue_id,
                    passed=False,
                    issue_id=input.issue_id,
                )
            return None, True, result  # break

        if result.effect == Effect.SEND_GATE_RETRY:
            if self.event_sink is not None:
                self.event_sink.on_gate_retry(
                    input.issue_id,
                    lifecycle_ctx.retry_state.gate_attempt,
                    self.config.max_gate_retries,
                    issue_id=input.issue_id,
                )
                self.event_sink.on_gate_result(
                    input.issue_id,
                    passed=False,
                    failure_reasons=list(gate_result.failure_reasons),
                    issue_id=input.issue_id,
                )
                # Emit validation_result before retry so every
                # on_validation_started has a corresponding result
                self.event_sink.on_validation_result(
                    input.issue_id,
                    passed=False,
                    issue_id=input.issue_id,
                )
            # Build follow-up prompt
            failure_text = "\n".join(f"- {r}" for r in gate_result.failure_reasons)
            # Get validation commands or use defaults
            cmds = (
                self.config.prompt_validation_commands
                or _get_default_validation_commands()
            )
            pending_query = self.config.prompts.gate_followup.format(
                attempt=lifecycle_ctx.retry_state.gate_attempt,
                max_attempts=self.config.max_gate_retries,
                failure_reasons=failure_text,
                issue_id=input.issue_id,
                lint_command=cmds.lint,
                format_command=cmds.format,
                typecheck_command=cmds.typecheck,
                test_command=cmds.test,
            )
            return pending_query, False, result  # continue with retry

        # RUN_REVIEW or other effects - pass through
        return None, False, result

    def _check_review_no_progress(
        self,
        input: AgentSessionInput,
        log_path: Path,
        lifecycle: ImplementerLifecycle,
        lifecycle_ctx: LifecycleContext,
        cerberus_review_log_path: str | None,
    ) -> ReviewEffectResult | None:
        """Check if review retry has made no progress and should be skipped.

        Args:
            input: Session input with issue_id.
            log_path: Path to log file.
            lifecycle: Lifecycle state machine.
            lifecycle_ctx: Lifecycle context.
            cerberus_review_log_path: Path to Cerberus review log, if any.

        Returns:
            ReviewEffectResult if no progress detected (caller should return early),
            None if review should proceed normally.
        """
        # Only check on retry attempts (attempt > 1) when callback is configured
        if (
            lifecycle_ctx.retry_state.review_attempt <= 1
            or self.callbacks.on_review_no_progress is None
        ):
            return None

        current_commit = (
            lifecycle_ctx.last_gate_result.commit_hash
            if lifecycle_ctx.last_gate_result
            else None
        )
        no_progress = self.callbacks.on_review_no_progress(
            log_path,
            lifecycle_ctx.retry_state.log_offset,
            lifecycle_ctx.retry_state.previous_commit_hash,
            current_commit,
        )

        if not no_progress:
            return None

        # Emit event for no-progress skip
        if self.event_sink is not None:
            self.event_sink.on_review_skipped_no_progress(input.issue_id)

        # Create synthetic failed review
        synthetic = ReviewResult(passed=False, issues=[], parse_error=None)
        new_offset = (
            self.callbacks.get_log_offset(
                log_path,
                lifecycle_ctx.retry_state.log_offset,
            )
            if self.callbacks.get_log_offset
            else 0
        )
        no_progress_result = lifecycle.on_review_result(
            lifecycle_ctx,
            synthetic,
            new_offset,
            no_progress=True,
        )
        return ReviewEffectResult(
            pending_query=None,
            should_break=True,
            cerberus_log_path=cerberus_review_log_path,
            transition_result=no_progress_result,
        )

    async def _handle_review_effect(
        self,
        input: AgentSessionInput,
        log_path: Path,
        session_id: str | None,
        lifecycle: ImplementerLifecycle,
        lifecycle_ctx: LifecycleContext,
    ) -> ReviewEffectResult:
        """Handle RUN_REVIEW effect - run review and process result.

        Args:
            input: Session input with issue_id, description, baseline.
            log_path: Path to log file.
            session_id: Current SDK session ID.
            lifecycle: Lifecycle state machine.
            lifecycle_ctx: Lifecycle context.

        Returns:
            ReviewEffectResult with pending_query, should_break, cerberus_log_path,
            and transition_result.
        """
        cerberus_review_log_path: str | None = None

        # Emit gate passed events when first entering review
        # (review_attempt == 1 means gate just passed)
        _emit_gate_passed_events(
            self.event_sink, input.issue_id, lifecycle_ctx.retry_state.review_attempt
        )

        # Check no-progress before running review
        if no_progress_result := self._check_review_no_progress(
            input, log_path, lifecycle, lifecycle_ctx, cerberus_review_log_path
        ):
            return no_progress_result

        if self.event_sink is not None:
            self.event_sink.on_review_started(
                input.issue_id,
                lifecycle_ctx.retry_state.review_attempt,
                self.config.max_review_retries,
                issue_id=input.issue_id,
            )

        if self.callbacks.on_review_check is None:
            raise ValueError("on_review_check callback must be set")

        logger.debug(
            "Session %s: starting review (attempt %d/%d, session_id=%s)",
            input.issue_id,
            lifecycle_ctx.retry_state.review_attempt,
            self.config.max_review_retries,
            lifecycle_ctx.session_id[:8] if lifecycle_ctx.session_id else None,
        )
        review_start = time.time()
        review_result = await self.callbacks.on_review_check(
            input.issue_id,
            input.issue_description,
            input.baseline_commit,
            lifecycle_ctx.session_id,
            lifecycle_ctx.retry_state,
        )
        review_duration = time.time() - review_start
        issue_count = len(review_result.issues) if review_result.issues else 0
        blocking = _count_blocking_issues(review_result.issues)
        logger.debug(
            "Session %s: review completed in %.1fs "
            "(passed=%s, issues=%d, blocking=%d, parse_error=%s)",
            input.issue_id,
            review_duration,
            review_result.passed,
            issue_count,
            blocking,
            review_result.parse_error,
        )

        # Check for fatal error
        if review_result.fatal_error:
            if self.callbacks.on_abort is not None:
                self.callbacks.on_abort(
                    review_result.parse_error or "Unrecoverable review error"
                )

        # Capture Cerberus review log if available
        log_attr = getattr(review_result, "review_log_path", None)
        if log_attr is not None:
            cerberus_review_log_path = str(log_attr)

        # Get new log offset
        new_offset = (
            self.callbacks.get_log_offset(
                log_path,
                lifecycle_ctx.retry_state.log_offset,
            )
            if self.callbacks.get_log_offset
            else 0
        )

        result = lifecycle.on_review_result(lifecycle_ctx, review_result, new_offset)

        # Emit appropriate events based on transition result
        _emit_review_result_events(
            self.event_sink,
            input,
            result,
            review_result,
            lifecycle_ctx,
            self.config.max_review_retries,
            blocking,
        )

        # Build pending_query only for SEND_REVIEW_RETRY
        pending_query = None
        if result.effect == Effect.SEND_REVIEW_RETRY:
            pending_query = _build_review_retry_prompt(
                review_result,
                lifecycle_ctx,
                input.issue_id,
                self.config.repo_path,
                self.config.max_review_retries,
                self.config.prompts.review_followup,
            )

        return _make_review_effect_result(
            result.effect,
            cerberus_review_log_path,
            result,
            pending_query,
        )

    async def _apply_retry_backoff(self, retry_count: int) -> None:
        """Apply backoff delay before an idle retry attempt.

        Args:
            retry_count: Current retry count (1-based).
        """
        if self.config.idle_retry_backoff:
            backoff_idx = min(
                retry_count - 1,
                len(self.config.idle_retry_backoff) - 1,
            )
            backoff = self.config.idle_retry_backoff[backoff_idx]
        else:
            backoff = 0.0
        if backoff > 0:
            logger.info(f"Idle retry {retry_count}: waiting {backoff}s")
            await asyncio.sleep(backoff)

    async def _disconnect_client_safely(
        self, client: SDKClientProtocol, issue_id: str
    ) -> None:
        """Disconnect SDK client with timeout, logging any failures."""
        try:
            await asyncio.wait_for(
                client.disconnect(),
                timeout=DISCONNECT_TIMEOUT,
            )
        except TimeoutError:
            logger.warning("disconnect() timed out, subprocess abandoned")
        except Exception as e:
            logger.debug(f"Error during disconnect: {e}")

    def _prepare_idle_retry(
        self,
        state: MessageIterationState,
        lifecycle_ctx: LifecycleContext,
        issue_id: str,
    ) -> str:
        """Prepare state for idle retry and return the next query.

        Updates state.idle_retry_count, state.pending_session_id, and clears
        state.pending_tool_ids.

        Raises:
            IdleTimeoutError: If retry is not possible (max retries exceeded,
                or tool calls occurred without session context).

        Returns:
            The query to use for the retry attempt.
        """
        # Check if we can retry
        if state.idle_retry_count >= self.config.max_idle_retries:
            logger.error(
                f"Session {issue_id}: max idle retries "
                f"({self.config.max_idle_retries}) exceeded"
            )
            raise IdleTimeoutError(
                f"Max idle retries ({self.config.max_idle_retries}) exceeded"
            )

        # Prepare for retry
        state.idle_retry_count += 1
        # Clear pending state from previous attempt to avoid
        # hanging on stale tool IDs (they won't resolve on new stream)
        state.pending_tool_ids.clear()
        state.first_message_received = False
        resume_id = state.session_id or lifecycle_ctx.session_id

        if resume_id is not None:
            state.pending_session_id = resume_id
            pending_query = self.config.prompts.idle_resume.format(issue_id=issue_id)
            logger.info(
                f"Session {issue_id}: retrying with resume "
                f"(session_id={resume_id[:8]}..., "
                f"attempt {state.idle_retry_count})"
            )
            # Reset tool calls after decision to preserve safety check
            state.tool_calls_this_turn = 0
            return pending_query
        elif state.tool_calls_this_turn == 0:
            state.pending_session_id = None
            # Keep original query - caller must provide it
            logger.info(
                f"Session {issue_id}: retrying with fresh session "
                f"(no session_id, no side effects, "
                f"attempt {state.idle_retry_count})"
            )
            # Return empty string to signal caller to keep original query
            return ""
        else:
            logger.error(
                f"Session {issue_id}: cannot retry - "
                f"{state.tool_calls_this_turn} tool calls "
                "occurred without session_id"
            )
            raise IdleTimeoutError(
                f"Cannot retry: {state.tool_calls_this_turn} tool calls "
                "occurred without session context"
            )

    async def _process_message_stream(
        self,
        stream: AsyncIterator[Any],
        issue_id: str,
        state: MessageIterationState,
        lifecycle_ctx: LifecycleContext,
        lint_cache: LintCache,
        query_start: float,
        tracer: TelemetrySpan | None,
    ) -> MessageIterationResult:
        """Process SDK message stream and update state.

        Updates state.session_id, state.tool_calls_this_turn, state.pending_tool_ids,
        and lint_cache on successful lint commands.

        Args:
            stream: The message stream to process.
            issue_id: Issue ID for logging.
            state: Mutable state for the iteration.
            lifecycle_ctx: Lifecycle context for session state.
            lint_cache: Cache for lint command results.
            query_start: Timestamp when query was sent.
            tracer: Optional telemetry span context.

        Returns:
            MessageIterationResult with success status.
        """
        # Use duck typing to avoid SDK imports - check type name instead of isinstance
        async for message in stream:
            if not state.first_message_received:
                state.first_message_received = True
                latency = time.time() - query_start
                logger.debug(
                    "Session %s: first message after %.1fs",
                    issue_id,
                    latency,
                )
            if tracer is not None:
                tracer.log_message(message)

            msg_type = type(message).__name__
            if msg_type == "AssistantMessage":
                content = getattr(message, "content", [])
                for block in content:
                    block_type = type(block).__name__
                    if block_type == "TextBlock":
                        text = getattr(block, "text", "")
                        if self.callbacks.on_agent_text is not None:
                            self.callbacks.on_agent_text(issue_id, text)
                    elif block_type == "ToolUseBlock":
                        state.tool_calls_this_turn += 1
                        block_id = getattr(block, "id", "")
                        state.pending_tool_ids.add(block_id)
                        name = getattr(block, "name", "")
                        block_input = getattr(block, "input", {})
                        if self.callbacks.on_tool_use is not None:
                            self.callbacks.on_tool_use(issue_id, name, block_input)
                        if name.lower() == "bash":
                            cmd = block_input.get("command", "")
                            lint_type = lint_cache.detect_lint_command(cmd)
                            if lint_type:
                                state.pending_lint_commands[block_id] = (
                                    lint_type,
                                    cmd,
                                )
                    elif block_type == "ToolResultBlock":
                        tool_use_id = getattr(block, "tool_use_id", None)
                        if tool_use_id:
                            state.pending_tool_ids.discard(tool_use_id)
                        if tool_use_id in state.pending_lint_commands:
                            lint_type, cmd = state.pending_lint_commands.pop(
                                tool_use_id
                            )
                            if not getattr(block, "is_error", False):
                                lint_cache.mark_success(lint_type, cmd)

            elif msg_type == "ResultMessage":
                state.session_id = getattr(message, "session_id", None)
                lifecycle_ctx.session_id = state.session_id
                lifecycle_ctx.final_result = getattr(message, "result", "") or ""

                # Extract token usage from SDK for context pressure detection
                usage = getattr(message, "usage", None)
                if usage is not None:
                    # Handle both dict and object forms of usage
                    if isinstance(usage, dict):
                        input_tokens = usage.get("input_tokens", 0) or 0
                        output_tokens = usage.get("output_tokens", 0) or 0
                        cache_read = usage.get("cache_read_input_tokens", 0) or 0
                    else:
                        input_tokens = getattr(usage, "input_tokens", 0) or 0
                        output_tokens = getattr(usage, "output_tokens", 0) or 0
                        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

                    # Update lifecycle context with cumulative usage
                    lifecycle_ctx.context_usage.input_tokens = input_tokens
                    lifecycle_ctx.context_usage.output_tokens = output_tokens
                    lifecycle_ctx.context_usage.cache_read_tokens = cache_read

                    # Check context pressure threshold
                    pressure = lifecycle_ctx.context_usage.pressure_ratio(
                        self.config.context_limit
                    )
                    if pressure >= self.config.context_restart_threshold:
                        raise ContextPressureError(
                            session_id=message.session_id,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            cache_read_tokens=cache_read,
                            pressure_ratio=pressure,
                        )
                else:
                    logger.warning(
                        "Session %s: ResultMessage missing usage field, "
                        "context pressure tracking disabled",
                        issue_id,
                    )
                    lifecycle_ctx.context_usage.disable_tracking()

        # Success
        stream_duration = time.time() - query_start
        logger.debug(
            "Session %s: stream complete after %.1fs, %d tool calls",
            issue_id,
            stream_duration,
            state.tool_calls_this_turn,
        )
        return MessageIterationResult(
            success=True,
            session_id=state.session_id,
            idle_retry_count=0,
        )

    async def _run_message_iteration(
        self,
        query: str,
        issue_id: str,
        options: object,
        state: MessageIterationState,
        lifecycle_ctx: LifecycleContext,
        lint_cache: LintCache,
        idle_timeout_seconds: float | None,
        tracer: TelemetrySpan | None = None,
    ) -> MessageIterationResult:
        """Run a single message iteration with idle retry handling.

        Sends a query to the SDK and processes the response stream.
        Handles idle timeouts with automatic retry logic.

        Args:
            query: The query to send to the agent.
            issue_id: Issue ID for logging.
            options: SDK client options.
            state: Mutable state for the iteration.
            lifecycle_ctx: Lifecycle context for session state.
            lint_cache: Cache for lint command results.
            idle_timeout_seconds: Idle timeout (None to disable).
            tracer: Optional telemetry span context.

        Returns:
            MessageIterationResult with success status and updated state.

        Raises:
            IdleTimeoutError: If max idle retries exceeded.
        """
        pending_query: str | None = query
        state.tool_calls_this_turn = 0
        state.first_message_received = False
        state.pending_tool_ids.clear()

        while pending_query is not None:
            # Backoff before retry (not on first attempt)
            if state.idle_retry_count > 0:
                await self._apply_retry_backoff(state.idle_retry_count)

            # Create client for this attempt - factory required
            if self.sdk_client_factory is None:
                raise RuntimeError(
                    "sdk_client_factory is required. Use SDKClientFactory from "
                    "src.infra.sdk_adapter or inject via orchestration wiring."
                )
            client = self.sdk_client_factory.create(options)

            try:
                async with client:
                    # Send query
                    query_start = time.time()
                    if state.pending_session_id is not None:
                        logger.debug(
                            "Session %s: sending query with session_id=%s...",
                            issue_id,
                            state.pending_session_id[:8],
                        )
                        await client.query(
                            pending_query, session_id=state.pending_session_id
                        )
                    else:
                        logger.debug(
                            "Session %s: sending query (new session)",
                            issue_id,
                        )
                        await client.query(pending_query)

                    # Wrap stream with idle timeout handling
                    stream = IdleTimeoutStream(
                        client.receive_response(),
                        idle_timeout_seconds,
                        state.pending_tool_ids,
                    )

                    try:
                        return await self._process_message_stream(
                            stream,
                            issue_id,
                            state,
                            lifecycle_ctx,
                            lint_cache,
                            query_start,
                            tracer,
                        )

                    except IdleTimeoutError:
                        # Disconnect on idle timeout
                        idle_duration = time.time() - query_start
                        logger.warning(
                            f"Session {issue_id}: idle timeout after "
                            f"{idle_duration:.1f}s, first_msg={state.first_message_received}, "
                            f"{state.tool_calls_this_turn} tool calls, disconnecting subprocess"
                        )
                        await self._disconnect_client_safely(client, issue_id)

                        # Prepare state for retry (may raise IdleTimeoutError)
                        retry_query = self._prepare_idle_retry(
                            state, lifecycle_ctx, issue_id
                        )
                        # Empty string means keep original query
                        if retry_query:
                            pending_query = retry_query

            except IdleTimeoutError:
                raise

        # Should not reach here
        return MessageIterationResult(success=False)

    async def _get_checkpoint_from_agent(
        self,
        session_id: str,
        issue_id: str,
        options: object,
    ) -> str:
        """Query agent for checkpoint before context restart.

        Sends checkpoint_request prompt to the current session and extracts
        the checkpoint block from the response.

        Args:
            session_id: SDK session ID from ContextPressureError.
            issue_id: Issue ID for logging.
            options: SDK client options.

        Returns:
            Extracted checkpoint text, or fallback if extraction fails.
        """
        logger.info(
            "Session %s: requesting checkpoint from session %s...",
            issue_id,
            session_id[:8] if session_id else "unknown",
        )

        checkpoint_prompt = self.config.prompts.checkpoint_request
        if not checkpoint_prompt:
            logger.warning(
                "Session %s: no checkpoint_request prompt configured, using empty checkpoint",
                issue_id,
            )
            return ""

        # Create client to query for checkpoint - factory required
        if self.sdk_client_factory is None:
            raise RuntimeError(
                "sdk_client_factory is required. Use SDKClientFactory from "
                "src.infra.sdk_adapter or inject via orchestration wiring."
            )
        client = self.sdk_client_factory.create(options)

        response_text = ""
        try:
            async with client:
                await client.query(checkpoint_prompt, session_id=session_id)
                async for message in client.receive_response():
                    # Extract text from AssistantMessage
                    content = getattr(message, "content", None)
                    if content is not None:
                        for block in cast("list[Any]", content):
                            text = getattr(block, "text", None)
                            if text is not None:
                                response_text += text
        except Exception as e:
            logger.warning(
                "Session %s: checkpoint query failed: %s, using empty checkpoint",
                issue_id,
                e,
            )
            return ""

        # Extract checkpoint from response
        checkpoint = extract_checkpoint(response_text)
        logger.debug(
            "Session %s: extracted checkpoint (%d chars)",
            issue_id,
            len(checkpoint),
        )
        return checkpoint

    async def run_session(
        self,
        input: AgentSessionInput,
        tracer: TelemetrySpan | None = None,
    ) -> AgentSessionOutput:
        """Run an agent session for the given input.

        This method manages the full lifecycle of an agent session:
        1. Creates SDK client with appropriate options
        2. Sends initial prompt and streams responses
        3. Handles lifecycle transitions (gate, review, retries)
        4. Returns session output with results and metadata
        5. On ContextPressureError: checkpoints, restarts with fresh lifecycle

        Args:
            input: AgentSessionInput with issue_id, prompt, etc.
            tracer: Optional telemetry span context.

        Returns:
            AgentSessionOutput with success, summary, session_id, etc.
        """
        start_time = asyncio.get_event_loop().time()
        continuation_count = 0
        current_prompt = input.prompt
        # Timeout for checkpoint fetch operations (30 seconds)
        checkpoint_timeout_seconds = 30
        # Use provided agent_id or generate one to preserve lock continuity across restarts
        agent_id = input.agent_id or f"{input.issue_id}-{uuid.uuid4().hex[:8]}"

        while True:
            # Calculate remaining time to enforce overall session timeout
            loop = asyncio.get_event_loop()
            elapsed = loop.time() - start_time
            remaining = self.config.timeout_seconds - elapsed

            # Create fresh lifecycle for each iteration
            session_input = AgentSessionInput(
                issue_id=input.issue_id,
                prompt=current_prompt,
                baseline_commit=input.baseline_commit,
                issue_description=input.issue_description,
            )
            session_cfg, state = self._initialize_session(session_input, agent_id)

            try:
                # Check timeout inside try block so on_timeout cleanup runs
                if remaining <= 0:
                    raise TimeoutError("Session timeout exceeded across restarts")
                async with asyncio.timeout(remaining):
                    await self._run_lifecycle_loop(
                        session_input, session_cfg, state, tracer
                    )
                # Normal completion - exit loop
                break
            except ContextPressureError as e:
                # Get checkpoint from current session before it's gone
                # Use dedicated timeout bounded by remaining session time
                checkpoint_elapsed = loop.time() - start_time
                checkpoint_remaining = self.config.timeout_seconds - checkpoint_elapsed
                effective_checkpoint_timeout = max(
                    0, min(checkpoint_remaining, checkpoint_timeout_seconds)
                )
                try:
                    async with asyncio.timeout(effective_checkpoint_timeout):
                        checkpoint = await self._get_checkpoint_from_agent(
                            e.session_id,
                            input.issue_id,
                            session_cfg.options,
                        )
                except TimeoutError:
                    logger.warning(
                        "Session %s: checkpoint fetch timed out after %.1fs, using empty checkpoint",
                        input.issue_id,
                        effective_checkpoint_timeout,
                    )
                    checkpoint = ""
                continuation_count += 1
                logger.info(
                    "Session %s: context restart #%d at %.1f%%",
                    input.issue_id,
                    continuation_count,
                    e.pressure_ratio * 100,
                )
                # Build continuation prompt for next iteration
                continuation_template = self.config.prompts.continuation
                if continuation_template:
                    current_prompt = build_continuation_prompt(
                        continuation_template, checkpoint
                    )
                else:
                    # Fallback: just use checkpoint as prompt
                    current_prompt = f"Continue from checkpoint:\n\n{checkpoint}"
                # Loop continues with fresh lifecycle and continuation prompt
            except IdleTimeoutError as e:
                state.lifecycle.on_error(state.lifecycle_ctx, e)
                state.final_result = state.lifecycle_ctx.final_result
                break
            except TimeoutError:
                timeout_mins = self.config.timeout_seconds // 60
                state.lifecycle.on_timeout(state.lifecycle_ctx, timeout_mins)
                state.final_result = state.lifecycle_ctx.final_result
                break
            except Exception as e:
                state.lifecycle.on_error(state.lifecycle_ctx, e)
                state.final_result = state.lifecycle_ctx.final_result
                break

        duration = asyncio.get_event_loop().time() - start_time
        return self._build_session_output(session_cfg, state, duration)

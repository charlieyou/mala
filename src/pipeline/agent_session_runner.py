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
import functools
import logging
import os
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from src.infra.hooks import (
    FileReadCache,
    LintCache,
    _detect_lint_command,
    block_dangerous_commands,
    block_morph_replaced_tools,
    make_file_read_cache_hook,
    make_lint_cache_hook,
    make_lock_enforcement_hook,
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
from src.domain.prompts import get_gate_followup_prompt as _get_gate_followup_prompt
from src.infra.tools.env import SCRIPTS_DIR, get_lock_dir

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import Self

    from src.infra.io.event_sink import MalaEventSink
    from src.domain.lifecycle import (
        GateOutcome,
        RetryState,
        ReviewOutcome,
        TransitionResult,
    )
    from src.domain.quality_gate import GateResult
    from src.core.models import IssueResolution
    from src.core.protocols import ReviewIssueProtocol
    from src.infra.telemetry import TelemetrySpan


# Module-level logger for idle retry messages
logger = logging.getLogger(__name__)

# Prompt file paths
_PROMPT_DIR = Path(__file__).parent.parent / "prompts"
REVIEW_FOLLOWUP_FILE = _PROMPT_DIR / "review_followup.md"
IDLE_RESUME_PROMPT_FILE = _PROMPT_DIR / "idle_resume.md"

# Timeout for disconnect() call
DISCONNECT_TIMEOUT = 10.0


@functools.cache
def _get_review_followup_prompt() -> str:
    """Load review follow-up prompt (cached on first use)."""
    return REVIEW_FOLLOWUP_FILE.read_text()


@functools.cache
def _get_idle_resume_prompt() -> str:
    """Load idle resume prompt (cached on first use)."""
    return IDLE_RESUME_PROMPT_FILE.read_text()


class IdleTimeoutError(Exception):
    """Raised when the SDK response stream is idle for too long."""


@runtime_checkable
class SDKClientProtocol(Protocol):
    """Protocol for SDK client interactions.

    This protocol abstracts the Claude Agent SDK client, enabling testing
    with fake implementations that don't require actual SDK/API calls.

    The protocol matches the subset of ClaudeSDKClient used by AgentSessionRunner.
    """

    async def __aenter__(self) -> Self:
        """Enter async context."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        """Exit async context."""
        ...

    async def query(self, prompt: str, session_id: str | None = None) -> None:
        """Send a query to the agent.

        Args:
            prompt: The prompt to send.
            session_id: Optional session ID to resume.
        """
        ...

    def receive_response(self) -> AsyncIterator[object]:
        """Receive streaming response messages.

        Yields:
            SDK message objects (AssistantMessage, ResultMessage, etc.)
        """
        ...

    async def disconnect(self) -> None:
        """Disconnect and terminate the subprocess.

        This should be called when an idle timeout is detected to ensure
        the hung subprocess is terminated before creating a new client.
        """
        ...


@runtime_checkable
class SDKClientFactory(Protocol):
    """Protocol for creating SDK clients.

    This factory pattern allows injection of fake clients for testing.
    """

    def create(self, options: object) -> SDKClientProtocol:
        """Create a new SDK client with the given options.

        Args:
            options: SDK-specific options (ClaudeAgentOptions or equivalent).

        Returns:
            An SDK client conforming to SDKClientProtocol.
        """
        ...


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
class MessageIterationContext:
    """Mutable state for message iteration within a session.

    Passed by reference to helpers to avoid complex return types.
    Tracks state that evolves during SDK message streaming.

    Attributes:
        session_id: SDK session ID (updated when ResultMessage received).
        tool_calls_count: Number of tool calls in the current iteration.
        pending_lint_commands: Map of tool_use_id to (lint_type, command).
    """

    session_id: str | None = None
    tool_calls_count: int = 0
    pending_lint_commands: dict[str, tuple[str, str]] = field(default_factory=dict)


@dataclass
class AgentSessionConfig:
    """Configuration for agent session execution.

    Bundles all configuration needed to run an agent session.

    Attributes:
        repo_path: Path to the repository.
        timeout_seconds: Session timeout in seconds.
        max_gate_retries: Maximum gate retry attempts.
        max_review_retries: Maximum review retry attempts.
        morph_enabled: Whether Morph MCP is enabled.
        morph_api_key: Morph API key (required if morph_enabled).
        review_enabled: Whether Cerberus external review is enabled.
            When enabled, code changes are reviewed after the gate passes.
        log_file_wait_timeout: Seconds to wait for log file after session
            completes. Default 60s allows time for SDK to flush logs under load.
        idle_timeout_seconds: Seconds to wait for SDK message when not waiting
            for tool execution. If None, defaults to min(900, max(300, timeout_seconds * 0.2))
            which scales with session timeout (300-900s range). During tool execution
            (after ToolUseBlock, before result), timeout is disabled. Set to 0 to
            disable timeout entirely.
    """

    repo_path: Path
    timeout_seconds: int
    max_gate_retries: int = 3
    max_review_retries: int = 3
    morph_enabled: bool = False
    morph_api_key: str | None = None
    review_enabled: bool = True
    log_file_wait_timeout: float = 60.0
    idle_timeout_seconds: float | None = None
    max_idle_retries: int = 2
    idle_retry_backoff: tuple[float, ...] = (0.0, 5.0, 15.0)


@dataclass
class AgentSessionInput:
    """Input for running an agent session.

    Bundles all data needed to start a session for an issue.

    Attributes:
        issue_id: The issue ID being worked on.
        prompt: The initial prompt to send to the agent.
        baseline_commit: Optional baseline commit for diff comparison.
        issue_description: Issue description for scope verification.
    """

    issue_id: str
    prompt: str
    baseline_commit: str | None = None
    issue_description: str | None = None


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
    sdk_client_factory: SDKClientFactory | None = None
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
    ) -> tuple[list[object], list[object]]:
        """Build PreToolUse and Stop hooks for the session.

        Args:
            agent_id: The agent ID for lock enforcement.
            file_read_cache: Cache for blocking redundant file reads.
            lint_cache: Cache for blocking redundant lint commands.

        Returns:
            Tuple of (pre_tool_hooks, stop_hooks).
        """
        pre_tool_hooks: list[object] = [
            block_dangerous_commands,
            make_lock_enforcement_hook(agent_id, str(self.config.repo_path)),
            make_file_read_cache_hook(file_read_cache),
            make_lint_cache_hook(lint_cache),
        ]
        if self.config.morph_enabled:
            pre_tool_hooks.insert(1, block_morph_replaced_tools)

        stop_hooks: list[object] = [make_stop_hook(agent_id)]

        return pre_tool_hooks, stop_hooks

    def _build_sdk_options(
        self,
        agent_id: str,
        pre_tool_hooks: list[object],
        stop_hooks: list[object],
        agent_env: dict[str, str],
    ) -> Any:
        """Build SDK client options for the session.

        Args:
            agent_id: The agent ID for this session.
            pre_tool_hooks: PreToolUse hooks for the session.
            stop_hooks: Stop hooks for the session.
            agent_env: Environment variables for the agent.

        Returns:
            ClaudeAgentOptions configured for this session.
        """
        from claude_agent_sdk import ClaudeAgentOptions
        from claude_agent_sdk.types import HookMatcher

        return ClaudeAgentOptions(
            cwd=str(self.config.repo_path),
            permission_mode="bypassPermissions",
            model="opus",
            system_prompt={"type": "preset", "preset": "claude_code"},
            setting_sources=["project", "user"],
            mcp_servers=get_mcp_servers(
                self.config.repo_path,
                morph_api_key=self.config.morph_api_key,
                morph_enabled=self.config.morph_enabled,
            ),
            disallowed_tools=get_disallowed_tools(self.config.morph_enabled),
            env=agent_env,
            hooks={
                "PreToolUse": [
                    HookMatcher(
                        matcher=None,
                        hooks=pre_tool_hooks,  # type: ignore[arg-type]
                    )
                ],
                "Stop": [HookMatcher(matcher=None, hooks=stop_hooks)],  # type: ignore[arg-type]
            },
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
        new_log_path = self.callbacks.get_log_path(session_id)  # type: ignore[arg-type]

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
        log_path: Path,
        gate_result: GateResult,
        lifecycle: ImplementerLifecycle,
        lifecycle_ctx: LifecycleContext,
        new_offset: int,
    ) -> tuple[str | None, bool, TransitionResult]:
        """Handle RUN_GATE effect - process gate result and emit events.

        Args:
            input: Session input with issue_id.
            log_path: Path to log file.
            gate_result: Result from gate check callback.
            lifecycle: Lifecycle state machine.
            lifecycle_ctx: Lifecycle context.
            new_offset: New log offset after gate check.

        Returns:
            Tuple of (pending_query for retry or None, should_break, transition_result).
        """
        result = lifecycle.on_gate_result(lifecycle_ctx, gate_result, new_offset)

        if result.effect == Effect.COMPLETE_SUCCESS:
            if self.event_sink is not None:
                self.event_sink.on_gate_passed(
                    input.issue_id,
                    issue_id=input.issue_id,
                )
                self.event_sink.on_validation_result(
                    input.issue_id,
                    passed=True,
                    issue_id=input.issue_id,
                )
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
            pending_query = _get_gate_followup_prompt().format(
                attempt=lifecycle_ctx.retry_state.gate_attempt,
                max_attempts=self.config.max_gate_retries,
                failure_reasons=failure_text,
                issue_id=input.issue_id,
            )
            return pending_query, False, result  # continue with retry

        # RUN_REVIEW or other effects - pass through
        return None, False, result

    async def _handle_review_effect(
        self,
        input: AgentSessionInput,
        log_path: Path,
        session_id: str | None,
        lifecycle: ImplementerLifecycle,
        lifecycle_ctx: LifecycleContext,
    ) -> tuple[str | None, bool, str | None]:
        """Handle RUN_REVIEW effect - run review and process result.

        Args:
            input: Session input with issue_id, description, baseline.
            log_path: Path to log file.
            session_id: Current SDK session ID.
            lifecycle: Lifecycle state machine.
            lifecycle_ctx: Lifecycle context.

        Returns:
            Tuple of (pending_query for retry or None, should_break, cerberus_log_path).
        """
        cerberus_review_log_path: str | None = None

        # Emit gate passed events when first entering review
        # (review_attempt == 1 means gate just passed)
        if (
            lifecycle_ctx.retry_state.review_attempt == 1
            and self.event_sink is not None
        ):
            self.event_sink.on_gate_passed(
                input.issue_id,
                issue_id=input.issue_id,
            )
            self.event_sink.on_validation_result(
                input.issue_id,
                passed=True,
                issue_id=input.issue_id,
            )

        # Check no-progress before running review
        if (
            lifecycle_ctx.retry_state.review_attempt > 1
            and self.callbacks.on_review_no_progress is not None
        ):
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
            if no_progress:
                if self.event_sink is not None:
                    self.event_sink.on_review_skipped_no_progress(input.issue_id)
                # Create synthetic failed review
                from src.infra.clients.cerberus_review import ReviewResult

                synthetic = ReviewResult(passed=False, issues=[], parse_error=None)
                new_offset = (
                    self.callbacks.get_log_offset(
                        log_path,
                        lifecycle_ctx.retry_state.log_offset,
                    )
                    if self.callbacks.get_log_offset
                    else 0
                )
                lifecycle.on_review_result(
                    lifecycle_ctx,
                    synthetic,
                    new_offset,
                    no_progress=True,
                )
                return None, True, cerberus_review_log_path  # break

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
        blocking = (
            sum(
                1
                for i in review_result.issues
                if i.priority is not None and i.priority <= 1
            )
            if review_result.issues
            else 0
        )
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

        if result.effect == Effect.COMPLETE_SUCCESS:
            if self.event_sink is not None:
                self.event_sink.on_review_passed(
                    input.issue_id,
                    issue_id=input.issue_id,
                )
            return None, True, cerberus_review_log_path  # break

        if result.effect == Effect.COMPLETE_FAILURE:
            return None, True, cerberus_review_log_path  # break

        # parse_error retry: lifecycle returns RUN_REVIEW to re-run
        # review without prompting agent (no code changes needed)
        if result.effect == Effect.RUN_REVIEW:
            if self.event_sink is not None:
                self.event_sink.on_warning(
                    f"Review tool error: {review_result.parse_error}; retrying",
                    agent_id=input.issue_id,
                )
            return None, False, cerberus_review_log_path  # continue (re-run review)

        if result.effect == Effect.SEND_REVIEW_RETRY:
            # Log review retry trigger
            blocking_count = (
                sum(
                    1
                    for i in review_result.issues
                    if i.priority is not None and i.priority <= 1
                )
                if review_result.issues
                else 0
            )
            logger.debug(
                "Session %s: SEND_REVIEW_RETRY triggered "
                "(attempt %d/%d, %d blocking issues)",
                input.issue_id,
                lifecycle_ctx.retry_state.review_attempt,
                self.config.max_review_retries,
                blocking_count,
            )

            # Emit review retry event
            if self.event_sink is not None:
                self.event_sink.on_review_retry(
                    input.issue_id,
                    lifecycle_ctx.retry_state.review_attempt,
                    self.config.max_review_retries,
                    error_count=blocking_count or None,
                    parse_error=review_result.parse_error,
                    issue_id=input.issue_id,
                )

            # Build follow-up prompt for legitimate review issues
            from src.infra.clients.cerberus_review import format_review_issues

            review_issues_text = format_review_issues(
                review_result.issues,  # type: ignore[arg-type]
                base_path=self.config.repo_path,
            )
            pending_query = _get_review_followup_prompt().format(
                attempt=lifecycle_ctx.retry_state.review_attempt,
                max_attempts=self.config.max_review_retries,
                review_issues=review_issues_text,
                issue_id=input.issue_id,
            )
            return pending_query, False, cerberus_review_log_path  # continue with retry

        return None, False, cerberus_review_log_path  # continue

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

        Args:
            input: AgentSessionInput with issue_id, prompt, etc.
            tracer: Optional telemetry span context.

        Returns:
            AgentSessionOutput with success, summary, session_id, etc.
        """
        # Defer SDK imports to runtime
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeSDKClient,
            ResultMessage,
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
        )

        # Generate agent ID
        agent_id = f"{input.issue_id}-{uuid.uuid4().hex[:8]}"
        start_time = asyncio.get_event_loop().time()

        # Initialize lifecycle
        lifecycle_config = LifecycleConfig(
            max_gate_retries=self.config.max_gate_retries,
            max_review_retries=self.config.max_review_retries,
            review_enabled=self.config.review_enabled,
        )
        lifecycle = ImplementerLifecycle(lifecycle_config)
        lifecycle_ctx = LifecycleContext()

        # Set baseline timestamp
        lifecycle_ctx.retry_state.baseline_timestamp = int(time.time())

        # Build session components
        file_read_cache = FileReadCache()
        lint_cache = LintCache(repo_path=self.config.repo_path)
        pre_tool_hooks, stop_hooks = self._build_hooks(
            agent_id, file_read_cache, lint_cache
        )
        agent_env = self._build_agent_env(agent_id)

        # Build SDK options using helper
        options = self._build_sdk_options(
            agent_id, pre_tool_hooks, stop_hooks, agent_env
        )

        # Session state - session_id MUST be initialized to None here to avoid
        # UnboundLocalError if IdleTimeoutError occurs before ResultMessage
        session_id: str | None = None
        log_path: Path | None = None
        final_result = ""
        cerberus_review_log_path: str | None = None
        pending_lint_commands: dict[str, tuple[str, str]] = {}

        # Log file wait constants
        # Use configured timeout (default 60s) for log file to appear after session.
        # Claude SDK may have async delays in log writing, especially under load.
        log_file_wait_timeout = self.config.log_file_wait_timeout
        log_file_poll_interval = 0.5
        idle_timeout_seconds = self.config.idle_timeout_seconds
        if idle_timeout_seconds is None:
            # Scale idle timeout with session timeout: 20% of timeout, clamped to 300-900s
            derived = self.config.timeout_seconds * 0.2
            idle_timeout_seconds = min(900.0, max(300.0, derived))
        if idle_timeout_seconds <= 0:
            idle_timeout_seconds = None
        # Track pending tool calls to disable timeout during tool execution
        pending_tool_ids: set[str] = set()

        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                # Start lifecycle before client creation
                lifecycle.start()
                if self.event_sink is not None:
                    self.event_sink.on_lifecycle_state(
                        input.issue_id, lifecycle.state.name
                    )

                # Track state for query + message iteration
                pending_query: str | None = input.prompt
                pending_session_id: str | None = None
                idle_retry_count: int = 0
                tool_calls_this_turn: int = 0
                # result tracks the lifecycle effect from the last transition.
                # Initialized to None; will be set on first iteration since
                # pending_query is always set to input.prompt initially.
                result: TransitionResult | None = None

                # Main lifecycle loop
                while not lifecycle.is_terminal:
                    # === QUERY + MESSAGE ITERATION (only if pending_query is set) ===
                    if pending_query is not None:
                        message_iteration_complete = False
                        tool_calls_this_turn = 0
                        pending_tool_ids.clear()  # Reset for new iteration

                        while not message_iteration_complete:
                            # Backoff before retry (not on first attempt)
                            if idle_retry_count > 0:
                                # Guard against empty backoff tuple
                                if self.config.idle_retry_backoff:
                                    backoff_idx = min(
                                        idle_retry_count - 1,
                                        len(self.config.idle_retry_backoff) - 1,
                                    )
                                    backoff = self.config.idle_retry_backoff[
                                        backoff_idx
                                    ]
                                else:
                                    backoff = 0.0
                                if backoff > 0:
                                    logger.info(
                                        f"Idle retry {idle_retry_count}: "
                                        f"waiting {backoff}s"
                                    )
                                    await asyncio.sleep(backoff)

                            # Create client for this attempt
                            if self.sdk_client_factory is not None:
                                client = self.sdk_client_factory.create(options)
                            else:
                                client = ClaudeSDKClient(options=options)

                            try:
                                async with client:
                                    # Send query - only pass session_id if provided
                                    query_start = time.time()
                                    if pending_session_id is not None:
                                        logger.debug(
                                            "Session %s: sending query with session_id=%s...",
                                            input.issue_id,
                                            pending_session_id[:8],
                                        )
                                        await client.query(
                                            pending_query, session_id=pending_session_id
                                        )
                                    else:
                                        logger.debug(
                                            "Session %s: sending query (new session)",
                                            input.issue_id,
                                        )
                                        await client.query(pending_query)
                                    first_message_received = False

                                    # Define _iter_messages (captures client from scope)
                                    # Uses pending_tool_ids to disable timeout during tool execution
                                    async def _iter_messages() -> AsyncIterator[Any]:
                                        stream = client.receive_response()
                                        if idle_timeout_seconds is None:
                                            async for msg in stream:
                                                yield msg
                                            return
                                        while True:
                                            # Disable timeout while waiting for tool results
                                            current_timeout = (
                                                None
                                                if pending_tool_ids
                                                else idle_timeout_seconds
                                            )
                                            try:
                                                msg = await asyncio.wait_for(
                                                    stream.__anext__(),
                                                    timeout=current_timeout,
                                                )
                                            except StopAsyncIteration:
                                                break
                                            except TimeoutError as exc:
                                                raise IdleTimeoutError(
                                                    "SDK stream idle for "
                                                    f"{idle_timeout_seconds:.0f} seconds"
                                                ) from exc
                                            yield msg

                                    try:
                                        async for message in _iter_messages():
                                            if not first_message_received:
                                                first_message_received = True
                                                latency = time.time() - query_start
                                                logger.debug(
                                                    "Session %s: first message after %.1fs",
                                                    input.issue_id,
                                                    latency,
                                                )
                                            # Log to tracer if provided
                                            if tracer is not None:
                                                tracer.log_message(message)

                                            # Process message types
                                            if isinstance(message, AssistantMessage):
                                                for block in message.content:
                                                    if isinstance(block, TextBlock):
                                                        if (
                                                            self.callbacks.on_agent_text
                                                            is not None
                                                        ):
                                                            self.callbacks.on_agent_text(
                                                                input.issue_id,
                                                                block.text,
                                                            )
                                                    elif isinstance(
                                                        block, ToolUseBlock
                                                    ):
                                                        # Track tool calls for side effect
                                                        # detection
                                                        tool_calls_this_turn += 1
                                                        # Track pending tool for timeout logic
                                                        pending_tool_ids.add(block.id)
                                                        if (
                                                            self.callbacks.on_tool_use
                                                            is not None
                                                        ):
                                                            self.callbacks.on_tool_use(
                                                                input.issue_id,
                                                                block.name,
                                                                block.input,
                                                            )
                                                        # Track lint commands
                                                        if block.name.lower() == "bash":
                                                            cmd = block.input.get(
                                                                "command", ""
                                                            )
                                                            lint_type = (
                                                                _detect_lint_command(
                                                                    cmd
                                                                )
                                                            )
                                                            if lint_type:
                                                                pending_lint_commands[
                                                                    block.id
                                                                ] = (
                                                                    lint_type,
                                                                    cmd,
                                                                )
                                                    elif isinstance(
                                                        block, ToolResultBlock
                                                    ):
                                                        # Clear pending tool (result received)
                                                        tool_use_id = getattr(
                                                            block, "tool_use_id", None
                                                        )
                                                        if tool_use_id:
                                                            pending_tool_ids.discard(
                                                                tool_use_id
                                                            )
                                                        # Check for lint success
                                                        if (
                                                            tool_use_id
                                                            in pending_lint_commands
                                                        ):
                                                            lint_type, cmd = (
                                                                pending_lint_commands.pop(
                                                                    tool_use_id
                                                                )
                                                            )
                                                            if not getattr(
                                                                block, "is_error", False
                                                            ):
                                                                lint_cache.mark_success(
                                                                    lint_type, cmd
                                                                )

                                            elif isinstance(message, ResultMessage):
                                                session_id = message.session_id
                                                lifecycle_ctx.session_id = session_id
                                                lifecycle_ctx.final_result = (
                                                    message.result or ""
                                                )

                                        # Success! Clear pending_query and exit retry loop
                                        stream_duration = time.time() - query_start
                                        logger.debug(
                                            "Session %s: stream complete after %.1fs, "
                                            "%d tool calls",
                                            input.issue_id,
                                            stream_duration,
                                            tool_calls_this_turn,
                                        )
                                        pending_query = None
                                        idle_retry_count = 0
                                        message_iteration_complete = True

                                    except IdleTimeoutError:
                                        # === CRITICAL: Always disconnect first ===
                                        idle_duration = time.time() - query_start
                                        logger.warning(
                                            f"Session {input.issue_id}: idle timeout after "
                                            f"{idle_duration:.1f}s, {tool_calls_this_turn} tool "
                                            f"calls, first_msg={'yes' if first_message_received else 'no'}, "
                                            "disconnecting subprocess"
                                        )
                                        try:
                                            await asyncio.wait_for(
                                                client.disconnect(),
                                                timeout=DISCONNECT_TIMEOUT,
                                            )
                                        except TimeoutError:
                                            logger.warning(
                                                "disconnect() timed out, "
                                                "subprocess abandoned"
                                            )
                                        except Exception as e:
                                            logger.debug(
                                                f"Error during disconnect: {e}"
                                            )

                                        # Check if we can retry
                                        if (
                                            idle_retry_count
                                            >= self.config.max_idle_retries
                                        ):
                                            logger.error(
                                                f"Session {input.issue_id}: max idle "
                                                f"retries ({self.config.max_idle_retries})"
                                                " exceeded"
                                            )
                                            raise IdleTimeoutError(
                                                f"Max idle retries "
                                                f"({self.config.max_idle_retries}) "
                                                "exceeded"
                                            ) from None

                                        # Prepare for retry
                                        idle_retry_count += 1

                                        # Determine resume strategy
                                        resume_id = (
                                            session_id or lifecycle_ctx.session_id
                                        )
                                        if resume_id is not None:
                                            # Have session context - resume with minimal
                                            # prompt
                                            pending_session_id = resume_id
                                            pending_query = (
                                                _get_idle_resume_prompt().format(
                                                    issue_id=input.issue_id,
                                                )
                                            )
                                            logger.info(
                                                f"Session {input.issue_id}: retrying with "
                                                f"resume (session_id={resume_id[:8]}..., "
                                                f"attempt {idle_retry_count})"
                                            )
                                        elif tool_calls_this_turn == 0:
                                            # No session context AND no side effects -
                                            # safe to start fresh
                                            pending_session_id = None
                                            # Keep original pending_query (don't replace)
                                            logger.info(
                                                f"Session {input.issue_id}: retrying with "
                                                "fresh session (no session_id, no side "
                                                f"effects, attempt {idle_retry_count})"
                                            )
                                        else:
                                            # No session context BUT side effects
                                            # occurred - unsafe to retry
                                            logger.error(
                                                f"Session {input.issue_id}: cannot retry - "
                                                f"{tool_calls_this_turn} tool calls "
                                                "occurred without session_id"
                                            )
                                            raise IdleTimeoutError(
                                                f"Cannot retry: {tool_calls_this_turn} "
                                                "tool calls occurred without session "
                                                "context"
                                            ) from None

                                        # Loop continues to retry

                            except IdleTimeoutError:
                                # Re-raise if we got here (means we gave up)
                                raise

                        # === END QUERY + MESSAGE ITERATION ===

                        # Messages complete - transition lifecycle
                        result = lifecycle.on_messages_complete(
                            lifecycle_ctx, has_session_id=bool(session_id)
                        )
                        if self.event_sink is not None:
                            self.event_sink.on_lifecycle_state(
                                input.issue_id, lifecycle.state.name
                            )

                        if result.effect == Effect.COMPLETE_FAILURE:
                            final_result = lifecycle_ctx.final_result
                            break

                    else:
                        # No query to send - result is from previous iteration
                        # (e.g., RUN_REVIEW retry where lifecycle returns RUN_REVIEW
                        # again without sending a message to the agent)
                        # The 'result' variable must be set from a previous iteration
                        assert result is not None, (
                            "Bug: entered loop without pending_query but result not set"
                        )

                    # Handle WAIT_FOR_LOG
                    if result.effect == Effect.WAIT_FOR_LOG:
                        log_path, result = await self._handle_log_waiting(
                            session_id,
                            input.issue_id,
                            log_path,
                            lifecycle,
                            lifecycle_ctx,
                            log_file_wait_timeout,
                            log_file_poll_interval,
                        )
                        if lifecycle.state == LifecycleState.FAILED:
                            final_result = lifecycle_ctx.final_result
                            break

                    # Handle RUN_GATE
                    if result.effect == Effect.RUN_GATE:
                        # Emit validation started BEFORE the gate check
                        if self.event_sink is not None:
                            self.event_sink.on_validation_started(
                                input.issue_id,
                                issue_id=input.issue_id,
                            )
                            self.event_sink.on_gate_started(
                                input.issue_id,
                                lifecycle_ctx.retry_state.gate_attempt,
                                self.config.max_gate_retries,
                                issue_id=input.issue_id,
                            )

                        if self.callbacks.on_gate_check is None:
                            raise ValueError("on_gate_check callback must be set")

                        assert log_path is not None
                        gate_result, new_offset = await self.callbacks.on_gate_check(
                            input.issue_id, log_path, lifecycle_ctx.retry_state
                        )

                        retry_query, should_break, result = self._handle_gate_effect(
                            input,
                            log_path,
                            gate_result,
                            lifecycle,
                            lifecycle_ctx,
                            new_offset,
                        )
                        if should_break:
                            final_result = lifecycle_ctx.final_result
                            break
                        if retry_query is not None:
                            pending_query = retry_query
                            # session_id must be set for gate retries
                            if session_id is None:
                                raise IdleTimeoutError(
                                    "Cannot retry gate: session_id not received from SDK"
                                )
                            pending_session_id = session_id
                            logger.debug(
                                "Session %s: queueing gate retry prompt "
                                "(%d chars, session_id=%s)",
                                input.issue_id,
                                len(pending_query),
                                session_id[:8],
                            )
                            idle_retry_count = 0
                            continue

                    # Handle RUN_REVIEW
                    if result.effect == Effect.RUN_REVIEW:
                        assert log_path is not None
                        (
                            retry_query,
                            should_break,
                            review_log,
                        ) = await self._handle_review_effect(
                            input, log_path, session_id, lifecycle, lifecycle_ctx
                        )
                        if review_log is not None:
                            cerberus_review_log_path = review_log
                        if should_break:
                            final_result = lifecycle_ctx.final_result
                            break
                        if retry_query is not None:
                            pending_query = retry_query
                            # session_id must be set for review retries
                            if session_id is None:
                                raise IdleTimeoutError(
                                    "Cannot retry review: session_id not received from SDK"
                                )
                            pending_session_id = session_id
                            logger.debug(
                                "Session %s: queueing review retry prompt "
                                "(%d chars, session_id=%s)",
                                input.issue_id,
                                len(pending_query),
                                session_id[:8],
                            )
                            idle_retry_count = 0
                        continue

        except IdleTimeoutError as e:
            lifecycle.on_error(lifecycle_ctx, e)
            final_result = lifecycle_ctx.final_result
        except TimeoutError:
            timeout_mins = self.config.timeout_seconds // 60
            lifecycle.on_timeout(lifecycle_ctx, timeout_mins)
            final_result = lifecycle_ctx.final_result

        except Exception as e:
            lifecycle.on_error(lifecycle_ctx, e)
            final_result = lifecycle_ctx.final_result

        duration = asyncio.get_event_loop().time() - start_time

        return AgentSessionOutput(
            success=lifecycle_ctx.success,
            summary=final_result,
            session_id=session_id,
            log_path=log_path,
            gate_attempts=lifecycle_ctx.retry_state.gate_attempt,
            review_attempts=lifecycle_ctx.retry_state.review_attempt,
            resolution=lifecycle_ctx.resolution,
            duration_seconds=duration,
            agent_id=agent_id,
            review_log_path=cerberus_review_log_path,
            low_priority_review_issues=cast(
                "list[ReviewIssueProtocol] | None",
                lifecycle_ctx.low_priority_review_issues or None,
            ),
        )

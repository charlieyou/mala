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
import os
import time
import uuid
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

from src.hooks import (
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
from src.mcp import get_disallowed_tools, get_mcp_servers
from src.lifecycle import (
    Effect,
    ImplementerLifecycle,
    LifecycleConfig,
    LifecycleContext,
    LifecycleState,
)
from src.prompts import get_gate_followup_prompt as _get_gate_followup_prompt
from src.tools.env import SCRIPTS_DIR, get_lock_dir

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import Self

    from src.event_sink import MalaEventSink
    from src.lifecycle import GateOutcome, RetryState, ReviewOutcome
    from src.models import IssueResolution
    from src.protocols import ReviewIssueProtocol
    from src.telemetry import TelemetrySpan


# Prompt file paths
_PROMPT_DIR = Path(__file__).parent.parent / "prompts"
REVIEW_FOLLOWUP_FILE = _PROMPT_DIR / "review_followup.md"


@functools.cache
def _get_review_followup_prompt() -> str:
    """Load review follow-up prompt (cached on first use)."""
    return REVIEW_FOLLOWUP_FILE.read_text()


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
        idle_timeout_seconds: Seconds to wait for any SDK message before
            aborting the session. None derives a default based on timeout_seconds.
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
    actions that require external state (gate checking, review, etc.)
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
            ClaudeAgentOptions,
            ClaudeSDKClient,
            ResultMessage,
            TextBlock,
            ToolResultBlock,
            ToolUseBlock,
        )
        from claude_agent_sdk.types import HookMatcher

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

        # Build SDK options
        options = ClaudeAgentOptions(
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

        # Session state
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
            derived = self.config.timeout_seconds * 0.2
            idle_timeout_seconds = min(900.0, max(300.0, derived))
        if idle_timeout_seconds <= 0:
            idle_timeout_seconds = None

        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                # Use factory if provided, otherwise create real client
                if self.sdk_client_factory is not None:
                    client = self.sdk_client_factory.create(options)
                else:
                    client = ClaudeSDKClient(options=options)

                async with client:
                    # Start lifecycle
                    lifecycle.start()
                    if self.event_sink is not None:
                        self.event_sink.on_lifecycle_state(
                            input.issue_id, lifecycle.state.name
                        )

                    # Initial query
                    await client.query(input.prompt)

                    # Main message loop
                    while not lifecycle.is_terminal:

                        async def _iter_messages() -> AsyncIterator[Any]:
                            stream = client.receive_response()
                            if idle_timeout_seconds is None:
                                async for msg in stream:
                                    yield msg
                                return
                            while True:
                                try:
                                    msg = await asyncio.wait_for(
                                        stream.__anext__(),
                                        timeout=idle_timeout_seconds,
                                    )
                                except StopAsyncIteration:
                                    break
                                except TimeoutError as exc:
                                    raise IdleTimeoutError(
                                        "SDK stream idle for "
                                        f"{idle_timeout_seconds:.0f} seconds"
                                    ) from exc
                                yield msg

                        async for message in _iter_messages():
                            # Log to tracer if provided
                            if tracer is not None:
                                tracer.log_message(message)

                            # Process message types
                            if isinstance(message, AssistantMessage):
                                for block in message.content:
                                    if isinstance(block, TextBlock):
                                        if self.callbacks.on_agent_text is not None:
                                            self.callbacks.on_agent_text(
                                                input.issue_id, block.text
                                            )
                                    elif isinstance(block, ToolUseBlock):
                                        if self.callbacks.on_tool_use is not None:
                                            self.callbacks.on_tool_use(
                                                input.issue_id,
                                                block.name,
                                                block.input,
                                            )
                                        # Track lint commands
                                        if block.name.lower() == "bash":
                                            cmd = block.input.get("command", "")
                                            lint_type = _detect_lint_command(cmd)
                                            if lint_type:
                                                pending_lint_commands[block.id] = (
                                                    lint_type,
                                                    cmd,
                                                )
                                    elif isinstance(block, ToolResultBlock):
                                        # Check for lint success
                                        tool_use_id = getattr(
                                            block, "tool_use_id", None
                                        )
                                        if tool_use_id in pending_lint_commands:
                                            lint_type, cmd = pending_lint_commands.pop(
                                                tool_use_id
                                            )
                                            if not getattr(block, "is_error", False):
                                                lint_cache.mark_success(lint_type, cmd)

                            elif isinstance(message, ResultMessage):
                                session_id = message.session_id
                                lifecycle_ctx.session_id = session_id
                                lifecycle_ctx.final_result = message.result or ""

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

                        # Handle WAIT_FOR_LOG
                        if result.effect == Effect.WAIT_FOR_LOG:
                            if self.callbacks.get_log_path is None:
                                raise ValueError("get_log_path callback must be set")
                            log_path = self.callbacks.get_log_path(session_id)  # type: ignore[arg-type]
                            if self.event_sink is not None:
                                self.event_sink.on_log_waiting(input.issue_id)

                            # Wait for log file
                            wait_elapsed = 0.0
                            while not log_path.exists():
                                if wait_elapsed >= log_file_wait_timeout:
                                    result = lifecycle.on_log_timeout(
                                        lifecycle_ctx, str(log_path)
                                    )
                                    if self.event_sink is not None:
                                        self.event_sink.on_log_timeout(
                                            input.issue_id, str(log_path)
                                        )
                                    break
                                await asyncio.sleep(log_file_poll_interval)
                                wait_elapsed += log_file_poll_interval

                            if lifecycle.state == LifecycleState.FAILED:
                                final_result = lifecycle_ctx.final_result
                                break

                            if log_path.exists():
                                if self.event_sink is not None:
                                    self.event_sink.on_log_ready(input.issue_id)
                            result = lifecycle.on_log_ready(lifecycle_ctx)

                        # Handle RUN_GATE
                        if result.effect == Effect.RUN_GATE:
                            if self.event_sink is not None:
                                self.event_sink.on_gate_started(
                                    input.issue_id,
                                    lifecycle_ctx.retry_state.gate_attempt,
                                    self.config.max_gate_retries,
                                )

                            if self.callbacks.on_gate_check is None:
                                raise ValueError("on_gate_check callback must be set")

                            assert log_path is not None
                            (
                                gate_result,
                                new_offset,
                            ) = await self.callbacks.on_gate_check(
                                input.issue_id, log_path, lifecycle_ctx.retry_state
                            )

                            result = lifecycle.on_gate_result(
                                lifecycle_ctx, gate_result, new_offset
                            )

                            if result.effect == Effect.COMPLETE_SUCCESS:
                                if self.event_sink is not None:
                                    self.event_sink.on_gate_passed(input.issue_id)
                                final_result = lifecycle_ctx.final_result
                                break

                            if result.effect == Effect.COMPLETE_FAILURE:
                                if self.event_sink is not None:
                                    self.event_sink.on_gate_failed(
                                        input.issue_id,
                                        lifecycle_ctx.retry_state.gate_attempt,
                                        self.config.max_gate_retries,
                                    )
                                    self.event_sink.on_gate_result(
                                        input.issue_id,
                                        passed=False,
                                        failure_reasons=list(
                                            gate_result.failure_reasons
                                        ),
                                    )
                                final_result = lifecycle_ctx.final_result
                                break

                            if result.effect == Effect.SEND_GATE_RETRY:
                                if self.event_sink is not None:
                                    self.event_sink.on_gate_retry(
                                        input.issue_id,
                                        lifecycle_ctx.retry_state.gate_attempt,
                                        self.config.max_gate_retries,
                                    )
                                    self.event_sink.on_gate_result(
                                        input.issue_id,
                                        passed=False,
                                        failure_reasons=list(
                                            gate_result.failure_reasons
                                        ),
                                    )
                                # Build follow-up prompt
                                failure_text = "\n".join(
                                    f"- {r}" for r in gate_result.failure_reasons
                                )
                                followup = _get_gate_followup_prompt().format(
                                    attempt=lifecycle_ctx.retry_state.gate_attempt,
                                    max_attempts=self.config.max_gate_retries,
                                    failure_reasons=failure_text,
                                    issue_id=input.issue_id,
                                )
                                assert session_id is not None
                                await client.query(followup, session_id=session_id)
                                continue

                        # Handle RUN_REVIEW
                        if result.effect == Effect.RUN_REVIEW:
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
                                    log_path,  # type: ignore[arg-type]
                                    lifecycle_ctx.retry_state.log_offset,
                                    lifecycle_ctx.retry_state.previous_commit_hash,
                                    current_commit,
                                )
                                if no_progress:
                                    if self.event_sink is not None:
                                        self.event_sink.on_review_skipped_no_progress(
                                            input.issue_id
                                        )
                                    # Create synthetic failed review
                                    from src.cerberus_review import ReviewResult

                                    synthetic = ReviewResult(
                                        passed=False, issues=[], parse_error=None
                                    )
                                    new_offset = (
                                        self.callbacks.get_log_offset(
                                            log_path,  # type: ignore[arg-type]
                                            lifecycle_ctx.retry_state.log_offset,
                                        )
                                        if self.callbacks.get_log_offset
                                        else 0
                                    )
                                    result = lifecycle.on_review_result(
                                        lifecycle_ctx,
                                        synthetic,
                                        new_offset,
                                        no_progress=True,
                                    )
                                    final_result = lifecycle_ctx.final_result
                                    break

                            if self.event_sink is not None:
                                self.event_sink.on_review_started(
                                    input.issue_id,
                                    lifecycle_ctx.retry_state.review_attempt,
                                    self.config.max_review_retries,
                                )

                            if self.callbacks.on_review_check is None:
                                raise ValueError("on_review_check callback must be set")

                            review_result = await self.callbacks.on_review_check(
                                input.issue_id,
                                input.issue_description,
                                input.baseline_commit,
                                lifecycle_ctx.session_id,
                                lifecycle_ctx.retry_state,
                            )

                            # Check for fatal error
                            if review_result.fatal_error:
                                if self.callbacks.on_abort is not None:
                                    self.callbacks.on_abort(
                                        review_result.parse_error
                                        or "Unrecoverable review error"
                                    )

                            # Capture Cerberus review log if available
                            log_attr = getattr(review_result, "review_log_path", None)
                            if log_attr is not None:
                                cerberus_review_log_path = str(log_attr)

                            # Get new log offset
                            new_offset = (
                                self.callbacks.get_log_offset(
                                    log_path,  # type: ignore[arg-type]
                                    lifecycle_ctx.retry_state.log_offset,
                                )
                                if self.callbacks.get_log_offset
                                else 0
                            )

                            result = lifecycle.on_review_result(
                                lifecycle_ctx, review_result, new_offset
                            )

                            if result.effect == Effect.COMPLETE_SUCCESS:
                                if self.event_sink is not None:
                                    self.event_sink.on_review_passed(input.issue_id)
                                final_result = lifecycle_ctx.final_result
                                break

                            if result.effect == Effect.COMPLETE_FAILURE:
                                final_result = lifecycle_ctx.final_result
                                break

                            # parse_error retry: lifecycle returns RUN_REVIEW to re-run
                            # review without prompting agent (no code changes needed)
                            if result.effect == Effect.RUN_REVIEW:
                                if self.event_sink is not None:
                                    self.event_sink.on_warning(
                                        f"Review tool error: {review_result.parse_error}; retrying",
                                        agent_id=input.issue_id,
                                    )
                                continue

                            if result.effect == Effect.SEND_REVIEW_RETRY:
                                # Emit review retry event
                                if self.event_sink is not None:
                                    error_count = (
                                        sum(
                                            1
                                            for i in review_result.issues
                                            if i.priority is not None
                                            and i.priority <= 1
                                        )
                                        if review_result.issues
                                        else None
                                    )
                                    self.event_sink.on_review_retry(
                                        input.issue_id,
                                        lifecycle_ctx.retry_state.review_attempt,
                                        self.config.max_review_retries,
                                        error_count=error_count,
                                        parse_error=review_result.parse_error,
                                    )

                                # Build follow-up prompt for legitimate review issues
                                from src.cerberus_review import format_review_issues

                                review_issues_text = format_review_issues(
                                    review_result.issues,  # type: ignore[arg-type]
                                    base_path=self.config.repo_path,
                                )
                                followup = _get_review_followup_prompt().format(
                                    attempt=lifecycle_ctx.retry_state.review_attempt,
                                    max_attempts=self.config.max_review_retries,
                                    review_issues=review_issues_text,
                                    issue_id=input.issue_id,
                                )
                                assert session_id is not None
                                await client.query(followup, session_id=session_id)
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

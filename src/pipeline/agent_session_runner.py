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
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from src.hooks import (
    MORPH_DISALLOWED_TOOLS,
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
from src.lifecycle import (
    Effect,
    ImplementerLifecycle,
    LifecycleConfig,
    LifecycleContext,
    LifecycleState,
)
from src.logging.console import (
    Colors,
    log,
    log_agent_text,
    log_tool,
    log_verbose,
)
from src.tools.env import SCRIPTS_DIR, get_lock_dir

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from typing import Self

    from src.lifecycle import GateOutcome, RetryState, ReviewOutcome
    from src.models import IssueResolution
    from src.telemetry import TelemetrySpan


# Prompt file paths
_PROMPT_DIR = Path(__file__).parent.parent / "prompts"
GATE_FOLLOWUP_FILE = _PROMPT_DIR / "gate_followup.md"
REVIEW_FOLLOWUP_FILE = _PROMPT_DIR / "review_followup.md"


@functools.cache
def _get_gate_followup_prompt() -> str:
    """Load gate follow-up prompt (cached on first use)."""
    return GATE_FOLLOWUP_FILE.read_text()


@functools.cache
def _get_review_followup_prompt() -> str:
    """Load review follow-up prompt (cached on first use)."""
    return REVIEW_FOLLOWUP_FILE.read_text()


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
    [str, str | None, str | None],
    Coroutine[Any, Any, "ReviewOutcome"],
]
ReviewNoProgressCallback = Callable[
    [Path, int, str | None, str | None],
    bool,
]
LogOffsetCallback = Callable[[Path, int], int]


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
        codex_review_enabled: Whether Codex review is enabled.
    """

    repo_path: Path
    timeout_seconds: int
    max_gate_retries: int = 3
    max_review_retries: int = 2
    morph_enabled: bool = False
    morph_api_key: str | None = None
    codex_review_enabled: bool = True


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
        codex_review_log_path: Path to Codex review session log (if any).
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
    codex_review_log_path: str | None = None


@dataclass
class SessionCallbacks:
    """Callbacks for external actions during session execution.

    These callbacks allow the orchestrator to inject behavior for
    actions that require external state (gate checking, review, etc.)
    without coupling AgentSessionRunner to those implementations.

    Attributes:
        on_gate_check: Async callback to run quality gate check.
            Args: (issue_id, log_path, retry_state) -> (GateResult, new_offset)
        on_review_check: Async callback to run Codex review.
            Args: (issue_id, issue_description, baseline_commit) -> ReviewOutcome
        on_review_no_progress: Sync callback to check if no progress on review retry.
            Args: (log_path, log_offset, prev_commit, curr_commit) -> bool
        get_log_path: Callback to get log path from session ID.
            Args: (session_id) -> Path
        get_log_offset: Callback to get log end offset.
            Args: (log_path, start_offset) -> int
        on_abort: Callback when fatal error requires run abort.
            Args: (reason) -> None
    """

    on_gate_check: GateCheckCallback | None = None
    on_review_check: ReviewCheckCallback | None = None
    on_review_no_progress: ReviewNoProgressCallback | None = None
    get_log_path: Callable[[str], Path] | None = None
    get_log_offset: LogOffsetCallback | None = None
    on_abort: Callable[[str], None] | None = None


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
    """

    config: AgentSessionConfig
    callbacks: SessionCallbacks = field(default_factory=SessionCallbacks)
    sdk_client_factory: SDKClientFactory | None = None

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

    def _get_mcp_servers(self) -> dict[str, Any]:
        """Get MCP server configurations.

        Returns:
            Dict of MCP server configs, empty if Morph disabled.
        """
        if not self.config.morph_enabled:
            return {}
        if not self.config.morph_api_key:
            raise ValueError(
                "morph_api_key is required when morph_enabled=True. "
                "Pass morph_api_key from MalaConfig or set morph_enabled=False."
            )
        return {
            "morphllm": {
                "command": "npx",
                "args": ["-y", "@morphllm/morphmcp"],
                "cwd": str(self.config.repo_path),
                "env": {
                    "MORPH_API_KEY": self.config.morph_api_key,
                    "ENABLED_TOOLS": "all",
                    "WORKSPACE_MODE": "true",
                    "WORKSPACE_PATH": str(self.config.repo_path),
                },
            }
        }

    def _get_disallowed_tools(self) -> list[str]:
        """Get list of disallowed tools based on Morph enablement."""
        return list(MORPH_DISALLOWED_TOOLS) if self.config.morph_enabled else []

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
            codex_review_enabled=self.config.codex_review_enabled,
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
            mcp_servers=self._get_mcp_servers(),
            disallowed_tools=self._get_disallowed_tools(),
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
        codex_review_log_path: str | None = None
        pending_lint_commands: dict[str, tuple[str, str]] = {}

        # Log file wait constants
        # Allow 60s for log file to appear after session completes.
        # Claude SDK may have async delays in log writing, especially under load.
        log_file_wait_timeout = 60
        log_file_poll_interval = 0.5

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
                    log_verbose(
                        "->",
                        f"State: {lifecycle.state.name}",
                        agent_id=input.issue_id,
                    )

                    # Initial query
                    await client.query(input.prompt)

                    # Main message loop
                    while not lifecycle.is_terminal:
                        async for message in client.receive_response():
                            # Log to tracer if provided
                            if tracer is not None:
                                tracer.log_message(message)

                            # Process message types
                            if isinstance(message, AssistantMessage):
                                for block in message.content:
                                    if isinstance(block, TextBlock):
                                        log_agent_text(block.text, input.issue_id)
                                    elif isinstance(block, ToolUseBlock):
                                        log_tool(
                                            block.name,
                                            agent_id=input.issue_id,
                                            arguments=block.input,
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
                        log_verbose(
                            "->",
                            f"State: {lifecycle.state.name}",
                            agent_id=input.issue_id,
                        )

                        if result.effect == Effect.COMPLETE_FAILURE:
                            final_result = lifecycle_ctx.final_result
                            break

                        # Handle WAIT_FOR_LOG
                        if result.effect == Effect.WAIT_FOR_LOG:
                            if self.callbacks.get_log_path is None:
                                raise ValueError("get_log_path callback must be set")
                            log_path = self.callbacks.get_log_path(session_id)  # type: ignore[arg-type]
                            log(
                                "o",
                                "Waiting for session log...",
                                Colors.MUTED,
                                agent_id=input.issue_id,
                            )

                            # Wait for log file
                            wait_elapsed = 0.0
                            while not log_path.exists():
                                if wait_elapsed >= log_file_wait_timeout:
                                    result = lifecycle.on_log_timeout(
                                        lifecycle_ctx, str(log_path)
                                    )
                                    log(
                                        "!",
                                        f"Log file not found: {log_path.name}",
                                        Colors.YELLOW,
                                        agent_id=input.issue_id,
                                    )
                                    break
                                await asyncio.sleep(log_file_poll_interval)
                                wait_elapsed += log_file_poll_interval

                            if lifecycle.state == LifecycleState.FAILED:
                                final_result = lifecycle_ctx.final_result
                                break

                            if log_path.exists():
                                log(
                                    "v",
                                    "Log file ready",
                                    Colors.GREEN,
                                    agent_id=input.issue_id,
                                )
                            result = lifecycle.on_log_ready(lifecycle_ctx)

                        # Handle RUN_GATE
                        if result.effect == Effect.RUN_GATE:
                            log(
                                "o",
                                "Running quality gate...",
                                Colors.CYAN,
                                agent_id=input.issue_id,
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
                                log(
                                    "v",
                                    "Quality gate passed",
                                    Colors.GREEN,
                                    agent_id=input.issue_id,
                                )
                                final_result = lifecycle_ctx.final_result
                                break

                            if result.effect == Effect.COMPLETE_FAILURE:
                                reasons = "; ".join(gate_result.failure_reasons)
                                log(
                                    "x",
                                    f"Quality gate failed: {reasons}",
                                    Colors.RED,
                                    agent_id=input.issue_id,
                                )
                                final_result = lifecycle_ctx.final_result
                                break

                            if result.effect == Effect.SEND_GATE_RETRY:
                                log(
                                    "r",
                                    f"Gate retry {lifecycle_ctx.retry_state.gate_attempt}/{self.config.max_gate_retries}",
                                    Colors.YELLOW,
                                    agent_id=input.issue_id,
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
                                    log(
                                        "x",
                                        "Review skipped: No progress (commit unchanged, no working tree changes)",
                                        Colors.RED,
                                        agent_id=input.issue_id,
                                    )
                                    # Create synthetic failed review
                                    from src.codex_review import CodexReviewResult

                                    synthetic = CodexReviewResult(
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

                            log(
                                "o",
                                f"Running Codex review (attempt {lifecycle_ctx.retry_state.review_attempt}/{self.config.max_review_retries})",
                                Colors.CYAN,
                                agent_id=input.issue_id,
                            )

                            if self.callbacks.on_review_check is None:
                                raise ValueError("on_review_check callback must be set")

                            review_result = await self.callbacks.on_review_check(
                                input.issue_id,
                                input.issue_description,
                                input.baseline_commit,
                            )

                            # Check for fatal error
                            if review_result.fatal_error:
                                if self.callbacks.on_abort is not None:
                                    self.callbacks.on_abort(
                                        review_result.parse_error
                                        or "Unrecoverable Codex review error"
                                    )

                            # Capture codex review log if available
                            log_attr = getattr(review_result, "session_log_path", None)
                            if log_attr is not None:
                                codex_review_log_path = str(log_attr)

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
                                log(
                                    "v",
                                    "Codex review passed",
                                    Colors.GREEN,
                                    agent_id=input.issue_id,
                                )
                                final_result = lifecycle_ctx.final_result
                                break

                            if result.effect == Effect.COMPLETE_FAILURE:
                                final_result = lifecycle_ctx.final_result
                                break

                            if result.effect == Effect.SEND_REVIEW_RETRY:
                                # Log review failure
                                if review_result.parse_error:
                                    log(
                                        "!",
                                        f"Codex review parse error: {review_result.parse_error}",
                                        Colors.YELLOW,
                                        agent_id=input.issue_id,
                                    )
                                else:
                                    error_count = sum(
                                        1
                                        for i in review_result.issues
                                        if i.priority is not None and i.priority <= 1
                                    )
                                    log(
                                        "r",
                                        f"Review retry {lifecycle_ctx.retry_state.review_attempt}/{self.config.max_review_retries} ({error_count} errors)",
                                        Colors.YELLOW,
                                        agent_id=input.issue_id,
                                    )

                                # Build follow-up prompt
                                from src.codex_review import format_review_issues

                                review_issues_text = (
                                    format_review_issues(review_result.issues)  # type: ignore[arg-type]
                                    if review_result.issues
                                    else review_result.parse_error
                                    or "Unknown review failure"
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
            codex_review_log_path=codex_review_log_path,
        )

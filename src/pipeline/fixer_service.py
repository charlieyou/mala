"""FixerService for spawning fixer agents.

Encapsulates the logic for spawning fixer agents to address validation failures.
Extracted from RunCoordinator._run_fixer_agent to enable independent testing
and composition.

Design principles:
- Stateless service constructed per run
- Explicit dependencies via constructor injection
- Clean separation from RunCoordinator
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from src.infra.sigint_guard import InterruptGuard
from src.infra.tools.env import get_claude_log_path
from src.infra.tools.locking import cleanup_agent_locks
from src.domain.validation.spec import extract_lint_tools_from_spec
from src.pipeline.fixer_interface import FixerResult

if TYPE_CHECKING:
    from pathlib import Path

    from src.core.protocols.agent_provider import AgentProvider
    from src.core.protocols.events import MalaEventSink
    from src.core.protocols.sdk import McpServerFactory
    from src.domain.validation.spec import ValidationSpec
    from src.infra.agent_runtime import AgentRuntimeBuilder


def _noop_mcp_server_factory(
    agent_id: str, repo_path: Path, emit_lock_event: object
) -> dict[str, object]:
    """Fallback MCP server factory used when none is configured.

    See :func:`src.pipeline.agent_session_runner._noop_mcp_server_factory`
    for context. Duplicated here to avoid a cross-module dependency for a
    one-line helper.
    """
    del agent_id, repo_path, emit_lock_event
    return {}


@dataclass(frozen=True)
class FailureContext:
    """Context for a fixer agent invocation.

    Encapsulates all information needed to spawn a fixer agent:
    - What failed (failure_output, failed_command)
    - Remediation metadata (attempt number, max retries)
    - Optional validation context (spec, validation_commands)

    Attributes:
        failure_output: Human-readable description of what failed.
        attempt: Current attempt number (1-indexed).
        max_attempts: Maximum number of fixer attempts allowed.
        failed_command: The command that failed (for prompt context).
        validation_commands: Formatted list of validation commands to re-run.
            If None, FixerService builds from spec or uses a placeholder.
        spec: Optional ValidationSpec for extracting lint tool names.
    """

    failure_output: str
    attempt: int
    max_attempts: int
    failed_command: str = "unknown"
    validation_commands: str | None = None
    spec: ValidationSpec | None = None


@dataclass
class FixerServiceConfig:
    """Configuration for FixerService.

    Attributes:
        repo_path: Path to the repository.
        timeout_seconds: Session timeout in seconds.
        fixer_prompt: Template for the fixer agent prompt. Must support
            .format() with keys: attempt, max_attempts, failure_output,
            failed_command, validation_commands.
        mcp_server_factory: Optional factory for creating MCP server configs.
    """

    repo_path: Path
    timeout_seconds: int
    fixer_prompt: str
    mcp_server_factory: McpServerFactory | None = None


class FixerService:
    """Service for spawning fixer agents to address validation failures.

    Encapsulates the fixer agent lifecycle:
    - Builds runtime configuration
    - Spawns SDK client and processes messages
    - Tracks lint command results
    - Cleans up agent locks on completion

    Thread Safety:
        Not thread-safe. Each FixerService instance should be used by a
        single async task.
    """

    def __init__(
        self,
        config: FixerServiceConfig,
        agent_provider: AgentProvider,
        event_sink: MalaEventSink | None = None,
    ) -> None:
        """Initialize the fixer service.

        Args:
            config: Service configuration.
            agent_provider: AgentProvider for the run's chosen coder backend.
                Fixers spawn through the same provider as the main run, so
                ``coder=amp`` runs spawn Amp fixers (per plan L165, AC#5).
            event_sink: Optional event sink for fixer progress events.
        """
        self._config = config
        self._agent_provider = agent_provider
        self._event_sink = event_sink
        self._active_fixer_ids: list[str] = []

    async def run_fixer(
        self,
        failure_context: FailureContext,
        interrupt_event: asyncio.Event | None = None,
    ) -> FixerResult:
        """Spawn a fixer agent to address validation failures.

        Args:
            failure_context: Context describing what failed and how to fix it.
            interrupt_event: Optional event to monitor for interruption.

        Returns:
            FixerResult indicating success/failure/interrupted status.
        """
        # Check for interrupt before starting
        guard = InterruptGuard(interrupt_event)
        if guard.is_interrupted():
            return FixerResult(success=None, interrupted=True)

        agent_id = f"fixer-{uuid.uuid4().hex[:8]}"
        self._active_fixer_ids.append(agent_id)

        # Build validation commands string if not provided
        validation_commands = failure_context.validation_commands
        if validation_commands is None:
            validation_commands = self._build_validation_commands_string(
                failure_context.spec
            )

        prompt = self._config.fixer_prompt.format(
            attempt=failure_context.attempt,
            max_attempts=failure_context.max_attempts,
            failure_output=failure_context.failure_output,
            failed_command=failure_context.failed_command,
            validation_commands=validation_commands,
        )

        fixer_cwd = self._config.repo_path

        # Build runtime via the AgentProvider's runtime builder so fixers
        # follow the main coder. The fluent ``with_*`` calls are the Claude
        # builder's surface; the Amp builder will expose equivalent shaping
        # in T009. In T007, the stub Amp client_factory raises before this
        # runtime is consumed, so fixer-on-Amp surfaces NotImplementedError.
        lint_tools = extract_lint_tools_from_spec(failure_context.spec)
        mcp_factory = self._config.mcp_server_factory or _noop_mcp_server_factory
        raw_builder = self._agent_provider.runtime_builder(
            fixer_cwd,
            agent_id,
            mcp_server_factory=mcp_factory,
        )
        builder = (
            cast("AgentRuntimeBuilder", raw_builder)
            .with_hooks(
                deadlock_monitor=None,
                include_stop_hook=True,
                include_mala_disallowed_tools_hook=False,
            )
            .with_agent_timeout(self._config.timeout_seconds)
            .with_env(extra={"MALA_SDK_FLOW": "fixer"})
            .with_disallowed_tools()
        )
        # Configure MCP: use factory if available, otherwise empty (no MCP tools)
        if self._config.mcp_server_factory is not None:
            builder = builder.with_mcp()
        else:
            # No MCP tools means no lock_acquire - disable lock enforcement
            builder = builder.with_mcp(servers={}).with_hooks(
                include_lock_enforcement_hook=False
            )
        # Only set lint_tools if we have them; otherwise use builder defaults
        if lint_tools is not None:
            builder = builder.with_lint_tools(lint_tools)
        runtime = builder.build()
        client = self._agent_provider.client_factory.create(runtime)

        pending_lint_commands: dict[str, tuple[str, str]] = {}
        is_amp_provider = getattr(self._agent_provider, "name", None) == "amp"
        runtime_log_path = getattr(runtime, "log_path", None)
        log_path: str = (
            str(runtime_log_path)
            if is_amp_provider and runtime_log_path is not None
            else str(get_claude_log_path(self._config.repo_path, agent_id))
        )

        def current_log_path() -> str:
            if not is_amp_provider:
                return log_path
            client_log_path = getattr(client, "log_path", None)
            if client_log_path is not None:
                return str(client_log_path)
            return log_path

        try:
            async with asyncio.timeout(self._config.timeout_seconds):
                async with client:
                    await client.query(
                        prompt,
                        session_id=None if is_amp_provider else agent_id,
                    )

                    from src.core.protocols.agent_event import to_agent_events

                    async for event in to_agent_events(client.receive_response()):
                        # Check for interrupt between events
                        if guard.is_interrupted():
                            return FixerResult(
                                success=None,
                                interrupted=True,
                                log_path=current_log_path(),
                            )

                        self._process_event(
                            event,
                            failure_context.attempt,
                            runtime,
                            pending_lint_commands,
                        )

            return FixerResult(success=True, log_path=current_log_path())

        except TimeoutError:
            if self._event_sink is not None:
                self._event_sink.on_fixer_failed("timeout")
            return FixerResult(success=False, log_path=current_log_path())
        except Exception as e:
            if self._event_sink is not None:
                self._event_sink.on_fixer_failed(str(e))
            return FixerResult(success=False, log_path=current_log_path())
        finally:
            # Safe removal - list may have been cleared by cleanup_locks()
            if agent_id in self._active_fixer_ids:
                self._active_fixer_ids.remove(agent_id)
            cleanup_agent_locks(agent_id)

    def _process_event(
        self,
        event: object,
        attempt: int,
        runtime: object,
        pending_lint_commands: dict[str, tuple[str, str]],
    ) -> None:
        """Process one ``AgentEvent`` from the fixer agent.

        Args:
            event: ``AgentEvent`` produced by the adapter / translator.
            attempt: Current fixer attempt number.
            runtime: AgentRuntime with lint_cache.
            pending_lint_commands: Dict tracking pending lint command results.
        """
        kind = getattr(event, "kind", None)
        if kind == "text":
            if self._event_sink is not None:
                text = getattr(event, "text", "")
                self._event_sink.on_fixer_text(attempt, text)
        elif kind == "tool_use":
            name = getattr(event, "name", "")
            block_input = getattr(event, "input", {}) or {}
            if self._event_sink is not None:
                self._event_sink.on_fixer_tool_use(attempt, name, block_input)
            if isinstance(name, str) and name.lower() == "bash":
                cmd = (
                    block_input.get("command", "")
                    if isinstance(block_input, dict)
                    else ""
                )
                lint_cache = getattr(runtime, "lint_cache", None)
                if lint_cache is not None:
                    lint_type = lint_cache.detect_lint_command(cmd)
                    if lint_type:
                        block_id = getattr(event, "id", "")
                        pending_lint_commands[block_id] = (lint_type, cmd)
        elif kind == "tool_result":
            tool_use_id = getattr(event, "tool_use_id", None)
            if tool_use_id in pending_lint_commands:
                lint_type, cmd = pending_lint_commands.pop(tool_use_id)
                if not getattr(event, "is_error", False):
                    lint_cache = getattr(runtime, "lint_cache", None)
                    if lint_cache is not None:
                        lint_cache.mark_success(lint_type, cmd)
        elif kind == "result":
            result = getattr(event, "result", "") or ""
            if self._event_sink is not None:
                self._event_sink.on_fixer_completed(result)

    def _build_validation_commands_string(self, spec: ValidationSpec | None) -> str:
        """Build formatted validation commands string for fixer prompt.

        Args:
            spec: ValidationSpec containing commands. If None, returns a placeholder.

        Returns:
            Formatted string with commands as markdown list items.
        """
        if spec is None or not spec.commands:
            return "   - (Run the appropriate validation commands for this project)"

        lines = []
        for cmd in spec.commands:
            lines.append(f"   - `{cmd.command}`")
        return "\n".join(lines)

    def cleanup_locks(self) -> None:
        """Clean up any remaining fixer agent locks.

        Call this when aborting a run to ensure no orphan locks remain.
        """
        for agent_id in self._active_fixer_ids:
            cleanup_agent_locks(agent_id)
        self._active_fixer_ids.clear()

"""RunCoordinator: Main run orchestration pipeline stage.

Extracted from MalaOrchestrator to separate global coordination from
the main class. This module handles:
- Main run loop (spawning agents, waiting for completion)
- Global validation (global validation)
- Fixer agent spawning for validation failures

The RunCoordinator receives explicit inputs and returns explicit outputs,
making it testable without SDK or subprocess dependencies.

Design principles:
- Protocol-based dependencies for testability
- Explicit input/output types for clarity
- Pure functions where possible
"""

from __future__ import annotations

import asyncio
import signal
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal


from src.infra.agent_runtime import AgentRuntimeBuilder
from src.infra.tools.command_runner import CommandRunner
from src.infra.tools.locking import cleanup_agent_locks
from src.domain.validation.e2e import E2EStatus
from src.domain.validation.config import ConfigError
from src.domain.validation.spec import (
    ValidationContext,
    ValidationScope,
    build_validation_spec,
)
from src.domain.validation.spec import extract_lint_tools_from_spec
from src.domain.validation.spec_executor import ValidationInterrupted
from src.domain.validation.spec_runner import SpecValidationRunner

if TYPE_CHECKING:
    from pathlib import Path
    from types import FrameType

    from src.core.protocols import (
        CommandRunnerPort,
        EnvConfigPort,
        GateChecker,
        LockManagerPort,
        MalaEventSink,
        McpServerFactory,
        SDKClientFactoryProtocol,
    )
    from src.domain.validation.result import ValidationResult
    from src.domain.validation.config import (
        BaseTriggerConfig,
        TriggerType,
        ValidationConfig,
        ValidationTriggersConfig,
    )
    from src.domain.validation.spec import ValidationSpec
    from src.infra.io.log_output.run_metadata import (
        RunMetadata,
        ValidationResult as MetaValidationResult,
    )
    from src.pipeline.cumulative_review_runner import (
        CumulativeReviewResult,
        CumulativeReviewRunner,
    )


class _FixerPromptNotSet:
    """Sentinel indicating fixer_prompt was not set.

    Raises RuntimeError if used, preventing silent failures from empty prompts.
    """

    def format(self, **kwargs: object) -> str:
        raise RuntimeError(
            "fixer_prompt not configured. "
            "Pass fixer_prompt to RunCoordinatorConfig or use build_run_coordinator()."
        )


_FIXER_PROMPT_NOT_SET = _FixerPromptNotSet()


@dataclass
class RunCoordinatorConfig:
    """Configuration for RunCoordinator.

    Attributes:
        repo_path: Path to the repository.
        timeout_seconds: Session timeout in seconds.
        max_gate_retries: Maximum gate retry attempts.
        disable_validations: Set of validation names to disable.
        fixer_prompt: Template for the fixer agent prompt.
    """

    repo_path: Path
    timeout_seconds: int
    max_gate_retries: int = 3
    disable_validations: set[str] | None = None
    fixer_prompt: str | _FixerPromptNotSet = _FIXER_PROMPT_NOT_SET
    mcp_server_factory: McpServerFactory | None = None
    validation_config: ValidationConfig | None = None
    validation_config_missing: bool = False


@dataclass
class GlobalValidationInput:
    """Input for global validation.

    Attributes:
        run_metadata: Run metadata tracker.
    """

    run_metadata: RunMetadata


@dataclass
class GlobalValidationOutput:
    """Output from global validation.

    Attributes:
        passed: Whether validation passed.
        interrupted: Whether validation was interrupted by SIGINT.
    """

    passed: bool
    interrupted: bool = False


@dataclass(frozen=True)
class TriggerValidationResult:
    """Result from running trigger validation.

    Attributes:
        status: One of "passed", "failed", or "aborted".
        details: Optional details about the result.
    """

    status: Literal["passed", "failed", "aborted"]
    details: str | None = None


@dataclass(frozen=True)
class ResolvedCommand:
    """A trigger command resolved from the base pool with overrides applied.

    Attributes:
        ref: The command reference name (e.g., "test", "lint").
        effective_command: The resolved command string to execute.
        effective_timeout: The resolved timeout in seconds (None for system default).
    """

    ref: str
    effective_command: str
    effective_timeout: int | None


@dataclass(frozen=True)
class TriggerCommandResult:
    """Result of executing a single trigger command.

    Attributes:
        ref: The command reference name.
        passed: Whether the command succeeded.
        duration_seconds: How long the command took.
        error_message: Optional error message if the command failed.
    """

    ref: str
    passed: bool
    duration_seconds: float
    error_message: str | None = None


@dataclass
class FixerResult:
    """Result from a fixer agent session.

    Attributes:
        success: Whether the fixer completed successfully. None if not evaluated.
        interrupted: Whether the fixer was interrupted by SIGINT.
        log_path: Path to the fixer session log (if captured).
    """

    success: bool | None
    interrupted: bool = False
    log_path: str | None = None


@dataclass
class SpecResultBuilder:
    """Builds ValidationResult to MetaValidationResult conversions.

    Encapsulates the logic for deriving e2e_passed status and building
    metadata results from validation results.
    """

    @staticmethod
    def derive_e2e_passed(result: ValidationResult) -> bool | None:
        """Derive e2e_passed from E2E execution result.

        Args:
            result: ValidationResult to check.

        Returns:
            None if E2E was not executed (disabled or skipped)
            True if E2E was executed and passed
            False if E2E was executed and failed
        """
        if result.e2e_result is None:
            return None
        if result.e2e_result.status == E2EStatus.SKIPPED:
            return None
        return result.e2e_result.passed

    @staticmethod
    def build_meta_result(
        result: ValidationResult,
        passed: bool,
    ) -> MetaValidationResult:
        """Build a MetaValidationResult from a ValidationResult.

        Args:
            result: The validation result.
            passed: Whether validation passed overall.

        Returns:
            MetaValidationResult for run metadata.
        """
        from src.infra.io.log_output.run_metadata import (
            ValidationResult as MetaValidationResult,
        )

        e2e_passed = SpecResultBuilder.derive_e2e_passed(result)
        failed_commands = (
            [s.name for s in result.steps if not s.ok] if not passed else []
        )

        return MetaValidationResult(
            passed=passed,
            commands_run=[s.name for s in result.steps],
            commands_failed=failed_commands,
            artifacts=result.artifacts,
            coverage_percent=result.coverage_result.percent
            if result.coverage_result
            else None,
            e2e_passed=e2e_passed,
        )


@dataclass
class RunCoordinator:
    """Global coordination for MalaOrchestrator.

    This class encapsulates the global orchestration logic that was
    previously inline in MalaOrchestrator. It handles:
    - Global validation (global validation) with fixer retries
    - Fixer agent spawning

    The orchestrator delegates to this class for global operations
    while retaining per-session coordination.

    Attributes:
        config: Configuration for run behavior.
        gate_checker: GateChecker for global validation.
        command_runner: CommandRunner for executing validation commands.
        env_config: Environment configuration for paths.
        lock_manager: Lock manager for file locking.
        sdk_client_factory: Factory for creating SDK clients (required).
        event_sink: Optional event sink for structured logging.
    """

    config: RunCoordinatorConfig
    gate_checker: GateChecker
    command_runner: CommandRunnerPort
    env_config: EnvConfigPort
    lock_manager: LockManagerPort
    sdk_client_factory: SDKClientFactoryProtocol
    event_sink: MalaEventSink | None = None
    cumulative_review_runner: CumulativeReviewRunner | None = None
    run_metadata: RunMetadata | None = None
    _active_fixer_ids: list[str] = field(default_factory=list, init=False)
    _trigger_queue: list[tuple[TriggerType, dict[str, Any]]] = field(
        default_factory=list, init=False
    )

    async def run_validation(
        self,
        input: GlobalValidationInput,
        *,
        interrupt_event: asyncio.Event | None = None,
    ) -> GlobalValidationOutput:
        """Run global validation validation after all issues complete.

        This runs validation with GLOBAL scope, which includes E2E tests.
        On failure, spawns a fixer agent and retries up to max_gate_retries.

        Args:
            input: GlobalValidationInput with run metadata.

        Returns:
            GlobalValidationOutput indicating pass/fail.
        """
        from src.infra.git_utils import get_git_commit_async

        # Short-circuit on interrupt before doing any work
        if interrupt_event is not None and interrupt_event.is_set():
            return GlobalValidationOutput(passed=False, interrupted=True)

        # Check if global validation is disabled
        if "global-validate" in (self.config.disable_validations or set()):
            if self.event_sink is not None:
                self.event_sink.on_global_validation_disabled()
            return GlobalValidationOutput(passed=True)

        # Get current HEAD commit
        commit_hash = await get_git_commit_async(self.config.repo_path)
        if not commit_hash:
            if self.event_sink is not None:
                self.event_sink.on_warning(
                    "Could not get HEAD commit for global validation"
                )
            return GlobalValidationOutput(passed=True)

        # Build global validation spec
        spec = build_validation_spec(
            self.config.repo_path,
            scope=ValidationScope.GLOBAL,
            disable_validations=self.config.disable_validations,
            validation_config=self.config.validation_config,
            config_missing=self.config.validation_config_missing,
        )

        # Build validation context
        context = ValidationContext(
            issue_id=None,
            repo_path=self.config.repo_path,
            commit_hash=commit_hash,
            changed_files=[],
            scope=ValidationScope.GLOBAL,
        )

        # Create validation runner
        runner = SpecValidationRunner(
            self.config.repo_path,
            event_sink=self.event_sink,
            command_runner=self.command_runner,
            env_config=self.env_config,
            lock_manager=self.lock_manager,
        )

        # Retry loop with fixer agent
        for attempt in range(1, self.config.max_gate_retries + 1):
            if self.event_sink is not None:
                self.event_sink.on_gate_started(
                    None, attempt, self.config.max_gate_retries
                )

            # Run validation
            result = None
            try:
                result = await runner.run_spec(spec, context)
            except ValidationInterrupted:
                if self.event_sink is not None:
                    self.event_sink.on_warning("Validation interrupted by SIGINT")
                return GlobalValidationOutput(passed=False, interrupted=True)
            except Exception as e:
                if self.event_sink is not None:
                    self.event_sink.on_warning(f"Validation runner error: {e}")

            if interrupt_event is not None and interrupt_event.is_set():
                return GlobalValidationOutput(passed=False, interrupted=True)

            if result and result.passed:
                if self.event_sink is not None:
                    self.event_sink.on_gate_passed(None)
                meta_result = SpecResultBuilder.build_meta_result(result, passed=True)
                input.run_metadata.record_run_validation(meta_result)
                return GlobalValidationOutput(passed=True)

            # Validation failed - build failure output for fixer
            failure_output = self._build_validation_failure_output(result)

            # Record failure in metadata
            if result:
                meta_result = SpecResultBuilder.build_meta_result(result, passed=False)
                input.run_metadata.record_run_validation(meta_result)

            # Check for interrupt before spawning fixer
            if interrupt_event is not None and interrupt_event.is_set():
                return GlobalValidationOutput(passed=False, interrupted=True)

            # Check if we have retries left
            if attempt >= self.config.max_gate_retries:
                if self.event_sink is not None:
                    self.event_sink.on_gate_failed(
                        None, attempt, self.config.max_gate_retries
                    )
                    failure_reasons = (
                        result.failure_reasons
                        if result and result.failure_reasons
                        else []
                    )
                    self.event_sink.on_gate_result(
                        None, passed=False, failure_reasons=failure_reasons
                    )
                return GlobalValidationOutput(passed=False)

            # Spawn fixer agent
            if self.event_sink is not None:
                self.event_sink.on_fixer_started(attempt, self.config.max_gate_retries)

            fixer_result = await self._run_fixer_agent(
                failure_output=failure_output,
                attempt=attempt,
                spec=spec,
                interrupt_event=interrupt_event,
                failed_command=self._extract_failed_command(result),
            )

            # If interrupted during fixer run, exit early
            if fixer_result.interrupted:
                return GlobalValidationOutput(passed=False, interrupted=True)

            if not fixer_result.success:
                # Fixer failure is logged via on_fixer_failed in _run_fixer_agent
                pass  # Continue to retry validation anyway

            # Update commit hash for next validation attempt
            new_commit = await get_git_commit_async(self.config.repo_path)
            if new_commit and new_commit != commit_hash:
                commit_hash = new_commit
                context = ValidationContext(
                    issue_id=None,
                    repo_path=self.config.repo_path,
                    commit_hash=commit_hash,
                    changed_files=[],
                    scope=ValidationScope.GLOBAL,
                )

        return GlobalValidationOutput(passed=False)

    def _build_validation_failure_output(self, result: ValidationResult | None) -> str:
        """Build failure output string for fixer agent prompt.

        Args:
            result: Validation result, or None if validation crashed.

        Returns:
            Human-readable failure output.
        """
        if result is None:
            return "Validation crashed - check logs for details."

        lines: list[str] = []
        if result.failure_reasons:
            lines.append("**Failure reasons:**")
            for reason in result.failure_reasons:
                lines.append(f"- {reason}")
            lines.append("")

        # Add step details for failed steps
        failed_steps = [s for s in result.steps if not s.ok]
        if failed_steps:
            lines.append("**Failed validation steps:**")
            for step in failed_steps:
                lines.append(f"\n### {step.name} (exit {step.returncode})")
                if step.stderr_tail:
                    lines.append("```")
                    lines.append(step.stderr_tail[:2000])
                    lines.append("```")
                elif step.stdout_tail:
                    lines.append("```")
                    lines.append(step.stdout_tail[:2000])
                    lines.append("```")

        return (
            "\n".join(lines) if lines else "Validation failed (no details available)."
        )

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

    def _extract_failed_command(self, result: ValidationResult | None) -> str:
        """Extract the first failed command from a ValidationResult.

        Args:
            result: ValidationResult to extract from. If None, returns "unknown".

        Returns:
            The command string of the first failed step, or "unknown".
        """
        if result is None:
            return "unknown"

        failed_steps = [s for s in result.steps if not s.ok]
        if failed_steps:
            return failed_steps[0].command
        return "unknown"

    async def _run_fixer_agent(
        self,
        failure_output: str,
        attempt: int,
        spec: ValidationSpec | None = None,
        interrupt_event: asyncio.Event | None = None,
        *,
        failed_command: str = "unknown",
        validation_commands: str | None = None,
    ) -> FixerResult:
        """Spawn a fixer agent to address global validation failures.

        Args:
            failure_output: Human-readable description of what failed.
            attempt: Current attempt number.
            spec: Optional ValidationSpec for extracting lint tool names.
            interrupt_event: Optional event to monitor for interruption.
            failed_command: The command that failed (for prompt context).
            validation_commands: Formatted list of validation commands to re-run.
                If None, builds from spec or uses a placeholder.

        Returns:
            FixerResult indicating success/failure/interrupted status.
        """
        from src.infra.sigint_guard import InterruptGuard
        from src.infra.tools.env import get_claude_log_path

        # Check for interrupt before starting
        guard = InterruptGuard(interrupt_event)
        if guard.is_interrupted():
            return FixerResult(success=None, interrupted=True)

        agent_id = f"fixer-{uuid.uuid4().hex[:8]}"
        self._active_fixer_ids.append(agent_id)

        # Build validation commands string if not provided
        if validation_commands is None:
            validation_commands = self._build_validation_commands_string(spec)

        prompt = self.config.fixer_prompt.format(
            attempt=attempt,
            max_attempts=self.config.max_gate_retries,
            failure_output=failure_output,
            failed_command=failed_command,
            validation_commands=validation_commands,
        )

        fixer_cwd = self.config.repo_path

        # Build runtime using AgentRuntimeBuilder
        # Note: include_mala_disallowed_tools_hook=False matches original fixer behavior
        lint_tools = extract_lint_tools_from_spec(spec)
        runtime = (
            AgentRuntimeBuilder(
                fixer_cwd,
                agent_id,
                self.sdk_client_factory,
                mcp_server_factory=self.config.mcp_server_factory,
            )
            .with_hooks(
                deadlock_monitor=None,
                include_stop_hook=True,
                include_mala_disallowed_tools_hook=False,
            )
            .with_env(extra={"MALA_SDK_FLOW": "fixer"})
            .with_mcp()
            .with_disallowed_tools()
            .with_lint_tools(lint_tools)
            .build()
        )
        client = self.sdk_client_factory.create(runtime.options)

        pending_lint_commands: dict[str, tuple[str, str]] = {}
        # Use agent_id for log path to ensure logs are correctly associated with the fixer agent.
        # This ensures log_path is available even on interrupt/timeout/error.
        log_path: str = str(get_claude_log_path(self.config.repo_path, agent_id))

        try:
            async with asyncio.timeout(self.config.timeout_seconds):
                async with client:
                    await client.query(prompt, session_id=agent_id)

                    async for message in client.receive_response():
                        # Check for interrupt between messages
                        if guard.is_interrupted():
                            return FixerResult(
                                success=None, interrupted=True, log_path=log_path
                            )

                        # Use duck typing to avoid SDK imports
                        msg_type = type(message).__name__
                        if msg_type == "AssistantMessage":
                            content = getattr(message, "content", [])
                            for block in content:
                                block_type = type(block).__name__
                                if block_type == "TextBlock":
                                    text = getattr(block, "text", "")
                                    if self.event_sink is not None:
                                        self.event_sink.on_fixer_text(attempt, text)
                                elif block_type == "ToolUseBlock":
                                    name = getattr(block, "name", "")
                                    block_input = getattr(block, "input", {})
                                    if self.event_sink is not None:
                                        self.event_sink.on_fixer_tool_use(
                                            attempt, name, block_input
                                        )
                                    if name.lower() == "bash":
                                        cmd = block_input.get("command", "")
                                        lint_type = (
                                            runtime.lint_cache.detect_lint_command(cmd)
                                        )
                                        if lint_type:
                                            block_id = getattr(block, "id", "")
                                            pending_lint_commands[block_id] = (
                                                lint_type,
                                                cmd,
                                            )
                                elif block_type == "ToolResultBlock":
                                    tool_use_id = getattr(block, "tool_use_id", None)
                                    if tool_use_id in pending_lint_commands:
                                        lint_type, cmd = pending_lint_commands.pop(
                                            tool_use_id
                                        )
                                        if not getattr(block, "is_error", False):
                                            runtime.lint_cache.mark_success(
                                                lint_type, cmd
                                            )
                        elif msg_type == "ResultMessage":
                            result = getattr(message, "result", "") or ""
                            if self.event_sink is not None:
                                self.event_sink.on_fixer_completed(result)

            return FixerResult(success=True, log_path=log_path)

        except TimeoutError:
            if self.event_sink is not None:
                self.event_sink.on_fixer_failed("timeout")
            return FixerResult(success=False, log_path=log_path)
        except Exception as e:
            if self.event_sink is not None:
                self.event_sink.on_fixer_failed(str(e))
            return FixerResult(success=False, log_path=log_path)
        finally:
            self._active_fixer_ids.remove(agent_id)
            cleanup_agent_locks(agent_id)

    def cleanup_fixer_locks(self) -> None:
        """Clean up any remaining fixer agent locks."""
        for agent_id in self._active_fixer_ids:
            cleanup_agent_locks(agent_id)
        self._active_fixer_ids.clear()

    def queue_trigger_validation(
        self, trigger_type: TriggerType, context: dict[str, Any]
    ) -> None:
        """Queue a trigger for validation.

        Args:
            trigger_type: The type of trigger (epic_completion, session_end, periodic).
            context: Additional context for the trigger (e.g., issue_id, epic_id).
        """
        self._trigger_queue.append((trigger_type, context))
        # Emit queued event
        if self.event_sink is not None:
            context_str = ", ".join(f"{k}={v}" for k, v in context.items())
            self.event_sink.on_trigger_validation_queued(
                trigger_type.value, context_str
            )

    async def run_trigger_validation(
        self, *, dry_run: bool = False
    ) -> TriggerValidationResult:
        """Execute queued trigger validations.

        Processes each queued trigger sequentially. For each trigger, resolves
        commands from the base pool, applies overrides, and executes. Handles
        failure_mode (abort/continue/remediate) and SIGINT for graceful interrupts.

        Args:
            dry_run: If True, simulate execution without running commands.
                All commands are treated as passed in dry-run mode.

        Returns:
            TriggerValidationResult with status and details.

        Raises:
            ConfigError: If a command reference is not found in the base pool.
        """
        # No triggers queued - return passed
        if not self._trigger_queue:
            return TriggerValidationResult(status="passed")

        # Get validation config
        validation_config = self.config.validation_config
        if validation_config is None:
            # No validation config - clear queue and return passed
            self._trigger_queue.clear()
            return TriggerValidationResult(status="passed")

        triggers_config = validation_config.validation_triggers
        if triggers_config is None:
            # No triggers configured - clear queue and return passed
            self._trigger_queue.clear()
            return TriggerValidationResult(status="passed")

        # Build base pool from commands config
        base_pool = self._build_base_pool(validation_config)

        # Set up SIGINT handling with asyncio.Event for async cancellation
        interrupt_event = asyncio.Event()
        original_handler = signal.getsignal(signal.SIGINT)

        def sigint_handler(signum: int, frame: FrameType | None) -> None:
            interrupt_event.set()
            # Send SIGTERM to any running subprocesses for immediate termination
            CommandRunner.forward_sigint()

        signal.signal(signal.SIGINT, sigint_handler)

        try:
            return await self._run_trigger_validation_loop(
                triggers_config,
                base_pool,
                dry_run,
                interrupt_event,
            )
        finally:
            signal.signal(signal.SIGINT, original_handler)

    async def _run_trigger_validation_loop(
        self,
        triggers_config: ValidationTriggersConfig,
        base_pool: dict[str, tuple[str, int | None]],
        dry_run: bool,
        interrupt_event: asyncio.Event,
    ) -> TriggerValidationResult:
        """Inner loop for trigger validation with failure mode handling.

        Args:
            triggers_config: The trigger configuration.
            base_pool: Map of command refs to (command, timeout).
            dry_run: If True, simulate execution.
            interrupt_event: Event set when SIGINT received.

        Returns:
            TriggerValidationResult with status and details.
        """
        from src.domain.validation.config import FailureMode

        # Track last failure for CONTINUE mode
        last_failure: tuple[str, str] | None = None  # (command_ref, trigger_type)

        while self._trigger_queue:
            # Check for SIGINT before processing next trigger
            if interrupt_event.is_set():
                self.clear_trigger_queue("sigint")
                return TriggerValidationResult(
                    status="aborted", details="Validation interrupted by SIGINT"
                )

            trigger_type, context = self._trigger_queue.pop(0)

            # Get the trigger config for this type
            trigger_config = self._get_trigger_config(triggers_config, trigger_type)
            if trigger_config is None:
                continue

            # Resolve commands from base pool
            resolved_commands = self._resolve_trigger_commands(
                trigger_type, trigger_config, base_pool
            )

            # Emit validation_started event
            if self.event_sink is not None:
                command_refs = [cmd.ref for cmd in resolved_commands]
                self.event_sink.on_trigger_validation_started(
                    trigger_type.value, command_refs
                )

            # Track start time for duration calculation
            trigger_start_time = time.monotonic()

            # Execute commands with fail-fast and interrupt support
            (
                results,
                was_interrupted,
            ) = await self._execute_trigger_commands_interruptible(
                resolved_commands,
                trigger_type=trigger_type,
                dry_run=dry_run,
                interrupt_event=interrupt_event,
            )

            # Check for interrupt during command execution
            if was_interrupted:
                self.clear_trigger_queue("sigint")
                return TriggerValidationResult(
                    status="aborted", details="Validation interrupted by SIGINT"
                )

            # Check for failures
            failed_result: TriggerCommandResult | None = None
            for result in results:
                if not result.passed:
                    failed_result = result
                    break

            if failed_result is None:
                # All commands passed - run code_review if configured
                review_result = await self._run_trigger_code_review(
                    trigger_type, trigger_config, context, interrupt_event
                )

                # Handle code review failure based on code_review.failure_mode
                code_review_config = trigger_config.code_review
                if (
                    review_result is not None
                    and code_review_config is not None
                    and review_result.status == "failed"
                ):
                    review_failure_mode = code_review_config.failure_mode
                    if review_failure_mode == FailureMode.ABORT:
                        if self.event_sink is not None:
                            self.event_sink.on_trigger_validation_failed(
                                trigger_type.value, "code_review", "abort"
                            )
                        self.clear_trigger_queue("code_review_failed")
                        return TriggerValidationResult(
                            status="aborted",
                            details=f"Code review failed in trigger {trigger_type.value}",
                        )
                    elif review_failure_mode == FailureMode.CONTINUE:
                        last_failure = ("code_review", trigger_type.value)
                        if self.event_sink is not None:
                            self.event_sink.on_trigger_validation_failed(
                                trigger_type.value, "code_review", "continue"
                            )
                    # REMEDIATE is not supported for code_review - fall through to passed

                # Emit validation_passed
                trigger_duration = time.monotonic() - trigger_start_time
                if self.event_sink is not None:
                    self.event_sink.on_trigger_validation_passed(
                        trigger_type.value, trigger_duration
                    )
                continue

            # Handle failure based on failure_mode
            failure_mode = trigger_config.failure_mode

            if failure_mode == FailureMode.CONTINUE:
                # Record failure and continue to next trigger
                last_failure = (failed_result.ref, trigger_type.value)
                # Emit validation_failed event
                if self.event_sink is not None:
                    self.event_sink.on_trigger_validation_failed(
                        trigger_type.value, failed_result.ref, "continue"
                    )
                continue

            elif failure_mode == FailureMode.REMEDIATE:
                # Attempt remediation with fixer agent
                remediation_result = await self._run_trigger_remediation(
                    trigger_type,
                    trigger_config,
                    resolved_commands,
                    failed_result,
                    dry_run,
                    interrupt_event,
                )
                if remediation_result is not None:
                    # Remediation exhausted/aborted - already emitted in _run_trigger_remediation
                    return remediation_result
                # Remediation succeeded - emit validation_passed
                trigger_duration = time.monotonic() - trigger_start_time
                if self.event_sink is not None:
                    self.event_sink.on_trigger_validation_passed(
                        trigger_type.value, trigger_duration
                    )

            else:  # ABORT (default)
                # Emit validation_failed event
                if self.event_sink is not None:
                    self.event_sink.on_trigger_validation_failed(
                        trigger_type.value, failed_result.ref, "abort"
                    )
                self.clear_trigger_queue("run_aborted")
                return TriggerValidationResult(
                    status="aborted",
                    details=f"Command '{failed_result.ref}' failed in trigger {trigger_type.value}",
                )

        # All triggers processed - return failed if any CONTINUE failures occurred
        if last_failure is not None:
            cmd_ref, trigger_val = last_failure
            return TriggerValidationResult(
                status="failed",
                details=f"Command '{cmd_ref}' failed in trigger {trigger_val}",
            )

        return TriggerValidationResult(status="passed")

    async def _run_trigger_remediation(
        self,
        trigger_type: TriggerType,
        trigger_config: BaseTriggerConfig,
        resolved_commands: list[ResolvedCommand],
        failed_result: TriggerCommandResult,
        dry_run: bool,
        interrupt_event: asyncio.Event,
    ) -> TriggerValidationResult | None:
        """Run remediation loop for a failed trigger command.

        Spawns fixer agent and re-runs failed command up to max_retries times.

        Args:
            trigger_type: The trigger type that failed.
            trigger_config: The trigger's configuration.
            resolved_commands: The resolved commands for this trigger.
            failed_result: The failed command result.
            dry_run: If True, simulate execution.
            interrupt_event: Event set when SIGINT received.

        Returns:
            TriggerValidationResult if remediation failed/aborted, None if succeeded.
        """
        max_retries = trigger_config.max_retries or 0

        # max_retries=0 means no fixer spawned, immediate abort
        if max_retries == 0:
            self.clear_trigger_queue("run_aborted")
            return TriggerValidationResult(
                status="aborted",
                details=f"Command '{failed_result.ref}' failed in trigger {trigger_type.value} (max_retries=0)",
            )

        # Find the failed command to re-run
        failed_cmd = next(
            (cmd for cmd in resolved_commands if cmd.ref == failed_result.ref), None
        )
        if failed_cmd is None:
            # Shouldn't happen, but handle gracefully
            self.clear_trigger_queue("run_aborted")
            return TriggerValidationResult(
                status="aborted",
                details=f"Failed to find command '{failed_result.ref}' for remediation",
            )

        # Build failure output for fixer
        failure_output = self._build_trigger_failure_output(failed_result)

        attempts_made = 0
        for attempt in range(1, max_retries + 1):
            attempts_made = attempt
            # Check for SIGINT before fixer attempt
            if interrupt_event.is_set():
                self.clear_trigger_queue("sigint")
                return TriggerValidationResult(
                    status="aborted", details="Validation interrupted by SIGINT"
                )

            # Emit remediation_started event
            if self.event_sink is not None:
                self.event_sink.on_trigger_remediation_started(
                    trigger_type.value, attempt, max_retries
                )

            # Spawn fixer agent
            fixer_result = await self._run_fixer_agent(
                failure_output=failure_output,
                attempt=attempt,
                interrupt_event=interrupt_event,
                failed_command=failed_cmd.effective_command,
                validation_commands=f"   - `{failed_cmd.effective_command}`",
            )

            # Check for SIGINT after fixer (either via result or event)
            if fixer_result.interrupted or interrupt_event.is_set():
                self.clear_trigger_queue("sigint")
                return TriggerValidationResult(
                    status="aborted", details="Validation interrupted by SIGINT"
                )

            # Re-run the failed command with interrupt support
            (
                retry_results,
                was_interrupted,
            ) = await self._execute_trigger_commands_interruptible(
                [failed_cmd],
                trigger_type=trigger_type,
                dry_run=dry_run,
                interrupt_event=interrupt_event,
            )

            if was_interrupted:
                self.clear_trigger_queue("sigint")
                return TriggerValidationResult(
                    status="aborted", details="Validation interrupted by SIGINT"
                )

            if retry_results and retry_results[0].passed:
                # Command passed after fixer, remediation succeeded
                if self.event_sink is not None:
                    self.event_sink.on_trigger_remediation_succeeded(
                        trigger_type.value, attempt
                    )
                return None

            # Update failure output for next attempt
            if retry_results:
                failure_output = self._build_trigger_failure_output(retry_results[0])

        # Remediation exhausted - emit event and abort
        if self.event_sink is not None:
            self.event_sink.on_trigger_remediation_exhausted(
                trigger_type.value, attempts_made
            )
            self.event_sink.on_trigger_validation_failed(
                trigger_type.value, failed_result.ref, "remediate"
            )
        self.clear_trigger_queue("run_aborted")
        return TriggerValidationResult(
            status="aborted",
            details=f"Command '{failed_result.ref}' failed after {max_retries} remediation attempts",
        )

    def _build_trigger_failure_output(self, result: TriggerCommandResult) -> str:
        """Build failure output string for fixer agent from trigger command result.

        Args:
            result: The failed trigger command result.

        Returns:
            Human-readable failure output.
        """
        lines: list[str] = [f"Command '{result.ref}' failed."]
        if result.error_message:
            lines.append(f"Error: {result.error_message}")
        return "\n".join(lines)

    def _build_base_pool(
        self, validation_config: ValidationConfig
    ) -> dict[str, tuple[str, int | None]]:
        """Build the base command pool from validation config.

        The base pool maps command names to (command_string, timeout) tuples.
        Built-in commands (test, lint, format, typecheck, e2e) come from the
        commands section. Custom commands are also included.

        Args:
            validation_config: The validation configuration.

        Returns:
            Dict mapping command ref names to (command, timeout) tuples.
        """
        pool: dict[str, tuple[str, int | None]] = {}
        cmds = validation_config.commands

        # Add built-in commands
        if cmds.test is not None:
            pool["test"] = (cmds.test.command, cmds.test.timeout)
        if cmds.lint is not None:
            pool["lint"] = (cmds.lint.command, cmds.lint.timeout)
        if cmds.format is not None:
            pool["format"] = (cmds.format.command, cmds.format.timeout)
        if cmds.typecheck is not None:
            pool["typecheck"] = (cmds.typecheck.command, cmds.typecheck.timeout)
        if cmds.e2e is not None:
            pool["e2e"] = (cmds.e2e.command, cmds.e2e.timeout)
        if cmds.setup is not None:
            pool["setup"] = (cmds.setup.command, cmds.setup.timeout)

        # Add custom commands
        for name, custom_cmd in cmds.custom_commands.items():
            pool[name] = (custom_cmd.command, custom_cmd.timeout)

        return pool

    def _get_trigger_config(
        self, triggers_config: ValidationTriggersConfig, trigger_type: TriggerType
    ) -> BaseTriggerConfig | None:
        """Get the trigger configuration for a specific trigger type.

        Args:
            triggers_config: The ValidationTriggersConfig.
            trigger_type: The trigger type to look up.

        Returns:
            The trigger config if defined, None otherwise.
        """
        from src.domain.validation.config import TriggerType

        if trigger_type == TriggerType.EPIC_COMPLETION:
            return triggers_config.epic_completion
        elif trigger_type == TriggerType.SESSION_END:
            return triggers_config.session_end
        elif trigger_type == TriggerType.PERIODIC:
            return triggers_config.periodic
        elif trigger_type == TriggerType.RUN_END:
            return triggers_config.run_end
        return None

    def _resolve_trigger_commands(
        self,
        trigger_type: TriggerType,
        trigger_config: BaseTriggerConfig,
        base_pool: dict[str, tuple[str, int | None]],
    ) -> list[ResolvedCommand]:
        """Resolve trigger command refs to executable commands.

        For each TriggerCommandRef in the trigger config, looks up the command
        in the base pool and applies any overrides (command string, timeout).

        Args:
            trigger_type: The type of trigger (for error messages).
            trigger_config: The trigger configuration with command refs.
            base_pool: Dict mapping ref names to (command, timeout).

        Returns:
            List of ResolvedCommand with effective command and timeout.

        Raises:
            ConfigError: If a ref is not found in the base pool.
        """
        resolved: list[ResolvedCommand] = []

        for cmd_ref in trigger_config.commands:
            if cmd_ref.ref not in base_pool:
                available = ", ".join(sorted(base_pool.keys()))
                raise ConfigError(
                    f"trigger {trigger_type.value} references unknown command "
                    f"'{cmd_ref.ref}'. Available: {available}"
                )

            base_cmd, base_timeout = base_pool[cmd_ref.ref]

            # Apply overrides - use `is not None` to allow falsy values like timeout=0
            effective_command = (
                cmd_ref.command if cmd_ref.command is not None else base_cmd
            )
            effective_timeout = (
                cmd_ref.timeout if cmd_ref.timeout is not None else base_timeout
            )

            resolved.append(
                ResolvedCommand(
                    ref=cmd_ref.ref,
                    effective_command=effective_command,
                    effective_timeout=effective_timeout,
                )
            )

        return resolved

    async def _execute_trigger_commands(
        self,
        commands: list[ResolvedCommand],
        *,
        dry_run: bool = False,
    ) -> list[TriggerCommandResult]:
        """Execute resolved commands sequentially with fail-fast.

        Executes each command in order. Stops immediately on the first failure
        (fail-fast behavior). In dry-run mode, all commands are treated as
        passed without actually executing them.

        Args:
            commands: List of resolved commands to execute.
            dry_run: If True, skip subprocess execution.

        Returns:
            List of TriggerCommandResult for executed commands.
        """
        results: list[TriggerCommandResult] = []

        for cmd in commands:
            if dry_run:
                # Dry-run: all commands pass, no execution
                results.append(
                    TriggerCommandResult(
                        ref=cmd.ref,
                        passed=True,
                        duration_seconds=0.0,
                    )
                )
                continue

            # Execute the command
            cmd_result = await self.command_runner.run_async(
                cmd.effective_command,
                timeout=cmd.effective_timeout,
                shell=True,
                cwd=self.config.repo_path,
            )

            passed = cmd_result.returncode == 0 and not cmd_result.timed_out
            error_message = None
            if not passed:
                if cmd_result.timed_out:
                    error_message = "Command timed out"
                elif cmd_result.stderr:
                    error_message = cmd_result.stderr[:500]
                else:
                    error_message = f"Exit code {cmd_result.returncode}"

            results.append(
                TriggerCommandResult(
                    ref=cmd.ref,
                    passed=passed,
                    duration_seconds=cmd_result.duration_seconds,
                    error_message=error_message,
                )
            )

            # Fail-fast: stop on first failure
            if not passed:
                break

        return results

    async def _execute_trigger_commands_interruptible(
        self,
        commands: list[ResolvedCommand],
        *,
        trigger_type: TriggerType,
        dry_run: bool = False,
        interrupt_event: asyncio.Event,
    ) -> tuple[list[TriggerCommandResult], bool]:
        """Execute resolved commands with interrupt support.

        Wraps command execution in asyncio tasks that can be cancelled when
        SIGINT is received. This ensures blocking subprocess waits are
        immediately terminated.

        Args:
            commands: List of resolved commands to execute.
            trigger_type: The trigger type (for event emission).
            dry_run: If True, skip subprocess execution.
            interrupt_event: Event set when SIGINT received.

        Returns:
            Tuple of (results, was_interrupted) where was_interrupted is True
            if execution was cancelled due to SIGINT.
        """
        results: list[TriggerCommandResult] = []

        for index, cmd in enumerate(commands):
            # Check interrupt before each command
            if interrupt_event.is_set():
                return results, True

            # Emit command_started event
            if self.event_sink is not None:
                self.event_sink.on_trigger_command_started(
                    trigger_type.value, cmd.ref, index
                )

            if dry_run:
                # Dry-run: all commands pass, no execution
                duration = 0.0
                passed = True
                # Emit command_completed event
                if self.event_sink is not None:
                    self.event_sink.on_trigger_command_completed(
                        trigger_type.value, cmd.ref, index, passed, duration
                    )
                results.append(
                    TriggerCommandResult(
                        ref=cmd.ref,
                        passed=True,
                        duration_seconds=0.0,
                    )
                )
                continue

            # Create task for command execution
            cmd_task = asyncio.create_task(
                self.command_runner.run_async(
                    cmd.effective_command,
                    timeout=cmd.effective_timeout,
                    shell=True,
                    cwd=self.config.repo_path,
                )
            )

            # Create task for interrupt wait
            interrupt_task = asyncio.create_task(interrupt_event.wait())

            try:
                # Wait for either command completion or interrupt
                done, pending = await asyncio.wait(
                    {cmd_task, interrupt_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

                # Check if interrupted
                if interrupt_task in done:
                    # SIGINT received - cancel command and return
                    return results, True

                # Get command result
                cmd_result = cmd_task.result()

            except asyncio.CancelledError:
                # Task was cancelled externally
                return results, True

            passed = cmd_result.returncode == 0 and not cmd_result.timed_out
            error_message = None
            if not passed:
                if cmd_result.timed_out:
                    error_message = "Command timed out"
                elif cmd_result.stderr:
                    error_message = cmd_result.stderr[:500]
                else:
                    error_message = f"Exit code {cmd_result.returncode}"

            # Emit command_completed event
            if self.event_sink is not None:
                self.event_sink.on_trigger_command_completed(
                    trigger_type.value,
                    cmd.ref,
                    index,
                    passed,
                    cmd_result.duration_seconds,
                )

            results.append(
                TriggerCommandResult(
                    ref=cmd.ref,
                    passed=passed,
                    duration_seconds=cmd_result.duration_seconds,
                    error_message=error_message,
                )
            )

            # Fail-fast: stop on first failure
            if not passed:
                break

        return results, False

    async def _run_trigger_code_review(
        self,
        trigger_type: TriggerType,
        trigger_config: BaseTriggerConfig,
        context: dict[str, Any],
        interrupt_event: asyncio.Event,
    ) -> CumulativeReviewResult | None:
        """Run code review if configured for this trigger.

        Checks if code_review is enabled in the trigger config and invokes
        CumulativeReviewRunner if so.

        Args:
            trigger_type: The type of trigger firing.
            trigger_config: The trigger configuration.
            context: Trigger context with issue_id, epic_id, etc.
            interrupt_event: Event to check for SIGINT interruption.

        Returns:
            CumulativeReviewResult if review ran, None if skipped or not configured.

        Raises:
            NotImplementedError: If CumulativeReviewRunner.run_review is not implemented.
        """
        # Check if code_review is configured and enabled
        if trigger_config.code_review is None or not trigger_config.code_review.enabled:
            return None

        # Skip code_review in fixer/remediation context to prevent loops.
        # Note: Fixer agents are separate processes (MALA_SDK_FLOW=fixer) that don't
        # call run_trigger_validation, so this guard is defensive.
        if context.get("is_fixer_session"):
            return None

        # Check if CumulativeReviewRunner is wired
        if self.cumulative_review_runner is None:
            if self.event_sink is not None:
                self.event_sink.on_warning(
                    f"code_review enabled for {trigger_type.value} but "
                    "CumulativeReviewRunner not wired"
                )
            return None

        # Check if run_metadata is available
        if self.run_metadata is None:
            if self.event_sink is not None:
                self.event_sink.on_warning(
                    f"code_review enabled for {trigger_type.value} but "
                    "run_metadata not available"
                )
            return None

        # Run cumulative code review
        return await self.cumulative_review_runner.run_review(
            trigger_type=trigger_type,
            config=trigger_config.code_review,
            run_metadata=self.run_metadata,
            repo_path=self.config.repo_path,
            interrupt_event=interrupt_event,
            issue_id=context.get("issue_id"),
            epic_id=context.get("epic_id"),
        )

    def clear_trigger_queue(self, reason: str) -> None:
        """Clear the trigger queue, emitting skipped events.

        Args:
            reason: Why the queue is being cleared (for logging).
        """
        # Emit skipped events for each remaining trigger
        if self.event_sink is not None:
            for trigger_type, _context in self._trigger_queue:
                self.event_sink.on_trigger_validation_skipped(
                    trigger_type.value, reason
                )
        self._trigger_queue.clear()

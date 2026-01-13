"""RunCoordinator: Main run orchestration pipeline stage.

Extracted from MalaOrchestrator to separate global coordination from
the main class. This module handles:
- Main run loop (spawning agents, waiting for completion)
- Trigger-based validation (run_end, session_end, etc.)
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
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal


from src.infra.tools.command_runner import CommandRunner
from src.domain.validation.e2e import E2EStatus
from src.pipeline.fixer_service import FailureContext

if TYPE_CHECKING:
    from pathlib import Path
    from types import FrameType

    from src.core.protocols.events import MalaEventSink
    from src.core.protocols.infra import (
        CommandRunnerPort,
        EnvConfigPort,
        LockManagerPort,
    )
    from src.core.protocols.sdk import McpServerFactory, SDKClientFactoryProtocol
    from src.core.protocols.validation import GateChecker
    from src.domain.validation.result import ValidationResult
    from src.domain.validation.config import (
        BaseTriggerConfig,
        TriggerType,
        ValidationConfig,
        ValidationTriggersConfig,
    )
    from src.infra.io.log_output.run_metadata import (
        RunMetadata,
        ValidationResult as MetaValidationResult,
    )
    from src.pipeline.cumulative_review_runner import (
        CumulativeReviewResult,
        CumulativeReviewRunner,
        ReviewFinding,
    )
    from src.pipeline.fixer_service import FixerService
    from src.pipeline.trigger_engine import ResolvedCommand, TriggerEngine


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
    - Fixer agent spawning (via FixerService)
    - Trigger validation (via TriggerEngine for policy evaluation)

    The orchestrator delegates to this class for global operations
    while retaining per-session coordination.

    Attributes:
        config: Configuration for run behavior.
        gate_checker: GateChecker for global validation.
        command_runner: CommandRunner for executing validation commands.
        env_config: Environment configuration for paths.
        lock_manager: Lock manager for file locking.
        sdk_client_factory: Factory for creating SDK clients (required).
        trigger_engine: TriggerEngine for trigger policy evaluation.
        fixer_service: FixerService for spawning fixer agents.
        event_sink: Optional event sink for structured logging.
    """

    config: RunCoordinatorConfig
    gate_checker: GateChecker
    command_runner: CommandRunnerPort
    env_config: EnvConfigPort
    lock_manager: LockManagerPort
    sdk_client_factory: SDKClientFactoryProtocol
    trigger_engine: TriggerEngine
    fixer_service: FixerService
    event_sink: MalaEventSink | None = None
    cumulative_review_runner: CumulativeReviewRunner | None = None
    run_metadata: RunMetadata | None = None
    _trigger_queue: list[tuple[TriggerType, dict[str, Any]]] = field(
        default_factory=list, init=False
    )

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

    def cleanup_fixer_locks(self) -> None:
        """Clean up any remaining fixer agent locks."""
        self.fixer_service.cleanup_locks()

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
                dry_run,
                interrupt_event,
            )
        finally:
            signal.signal(signal.SIGINT, original_handler)

    async def _run_trigger_validation_loop(
        self,
        triggers_config: ValidationTriggersConfig,
        dry_run: bool,
        interrupt_event: asyncio.Event,
    ) -> TriggerValidationResult:
        """Inner loop for trigger validation with failure mode handling.

        Args:
            triggers_config: The trigger configuration.
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

            # Resolve commands via TriggerEngine
            resolved_commands = self.trigger_engine.resolve_commands(
                trigger_config, trigger_type
            )

            # Emit validation_started event
            if self.event_sink is not None:
                command_refs = [cmd.ref for cmd in resolved_commands]
                self.event_sink.on_trigger_validation_started(
                    trigger_type.value, command_refs
                )

            # Track start time for duration calculation
            trigger_start_time = time.monotonic()

            results: list[TriggerCommandResult] = []
            current_index = 0

            total_commands = len(resolved_commands)
            while current_index < len(resolved_commands):
                # Execute remaining commands with fail-fast and interrupt support
                (
                    partial_results,
                    was_interrupted,
                ) = await self._execute_trigger_commands_interruptible(
                    resolved_commands[current_index:],
                    trigger_type=trigger_type,
                    dry_run=dry_run,
                    interrupt_event=interrupt_event,
                    start_index=current_index,
                    total_commands=total_commands,
                )

                # Check for interrupt during command execution
                if was_interrupted:
                    self.clear_trigger_queue("sigint")
                    return TriggerValidationResult(
                        status="aborted", details="Validation interrupted by SIGINT"
                    )

                results.extend(partial_results)

                # Check for failures in the partial results
                failed_offset: int | None = None
                for offset, result in enumerate(partial_results):
                    if not result.passed:
                        failed_offset = offset
                        break

                if failed_offset is None:
                    break

                failed_index = current_index + failed_offset
                failed_result = results[failed_index]

                # Handle failure based on failure_mode BEFORE code_review
                failure_mode = trigger_config.failure_mode

                if failure_mode == FailureMode.CONTINUE:
                    # Record failure and proceed to code_review
                    last_failure = (failed_result.ref, trigger_type.value)
                    if self.event_sink is not None:
                        self.event_sink.on_trigger_validation_failed(
                            trigger_type.value, failed_result.ref, "continue"
                        )
                    break

                if failure_mode == FailureMode.REMEDIATE:
                    # Attempt remediation with fixer agent
                    (
                        remediation_result,
                        remediated_result,
                    ) = await self._run_trigger_remediation(
                        trigger_type,
                        trigger_config,
                        resolved_commands,
                        failed_result,
                        failed_index,
                        dry_run,
                        interrupt_event,
                    )
                    if remediation_result is not None:
                        # Remediation exhausted/aborted - skip code_review
                        self._record_run_validation(trigger_type, results)
                        return remediation_result

                    if remediated_result is not None:
                        results[failed_index] = remediated_result

                    current_index = failed_index + 1
                    continue

                # ABORT (default)
                if self.event_sink is not None:
                    self.event_sink.on_trigger_validation_failed(
                        trigger_type.value, failed_result.ref, "abort"
                    )
                self.clear_trigger_queue("run_aborted")
                self._record_run_validation(trigger_type, results)
                return TriggerValidationResult(
                    status="aborted",
                    details=(
                        f"Command '{failed_result.ref}' failed in trigger {trigger_type.value}"
                    ),
                )

            self._record_run_validation(trigger_type, results)

            # Run code_review if configured (regardless of command outcome,
            # unless ABORT which returned above)
            code_review_config = trigger_config.code_review
            code_review_enabled = (
                code_review_config is not None and code_review_config.enabled
            )

            # Track end status for exactly-one-end-event invariant
            # Defined outside if-else to be accessible in remediation handling
            code_review_end_status: Literal["skipped", "passed", "failed"] | None = None
            code_review_blocking_count = 0
            defer_code_review_end_event = False
            review_result: CumulativeReviewResult | None = None

            if not code_review_enabled:
                # No events when code review is disabled/not configured
                pass
            else:
                # code_review_enabled is True implies code_review_config is not None
                assert code_review_config is not None

                # Get failure mode early to determine if we should defer end event
                review_failure_mode = code_review_config.failure_mode

                # Emit started AFTER enabled check passes
                if self.event_sink is not None:
                    self.event_sink.on_trigger_code_review_started(trigger_type.value)

                try:
                    review_result = await self._run_trigger_code_review(
                        trigger_type, trigger_config, context, interrupt_event
                    )

                    # Map result to status (precedence rule from issue)
                    if review_result is None:
                        # None is only returned for defensive cases that occur before
                        # the started event would be emitted (disabled, fixer session).
                        # Wiring errors now return status="failed" instead of None.
                        # This case should rarely be reached since we checked enabled above.
                        code_review_end_status = "skipped"
                    elif review_result.status == "skipped":
                        code_review_end_status = "skipped"
                    elif review_result.status == "failed":
                        # Review execution failed internally - emit error event
                        # and abort validation. This is treated like an exception:
                        # error event is the terminal event, no further processing.
                        error_reason = review_result.skip_reason or "unknown error"
                        if self.event_sink is not None:
                            self.event_sink.on_trigger_code_review_error(
                                trigger_type.value,
                                error_reason,
                            )
                        # Abort validation - execution error is a hard failure
                        self.clear_trigger_queue("code_review_execution_error")
                        return TriggerValidationResult(
                            status="aborted",
                            details=f"Code review execution failed: {error_reason}",
                        )
                    else:
                        # Review completed successfully - check findings threshold
                        threshold = code_review_config.finding_threshold
                        if self._findings_exceed_threshold(
                            review_result.findings, threshold
                        ):
                            code_review_end_status = "failed"
                            code_review_blocking_count = sum(
                                1
                                for f in review_result.findings
                                if f.priority <= int(threshold[1])
                            )
                            # Defer end event if remediation will run - emit after
                            # remediation completes to reflect final state
                            if review_failure_mode == FailureMode.REMEDIATE:
                                defer_code_review_end_event = True
                        else:
                            code_review_end_status = "passed"
                except Exception as e:
                    # Exception bubbles up - emit error event
                    # Clear end_status to prevent finally block from emitting duplicate event
                    code_review_end_status = None
                    if self.event_sink is not None:
                        self.event_sink.on_trigger_code_review_error(
                            trigger_type.value, str(e)
                        )
                    raise
                finally:
                    # Emit exactly one end event based on status
                    # Skip emission if deferred (will be emitted after remediation)
                    if (
                        self.event_sink is not None
                        and code_review_end_status is not None
                        and not defer_code_review_end_event
                    ):
                        if code_review_end_status == "skipped":
                            skip_reason = (
                                review_result.skip_reason
                                if review_result is not None
                                else "unknown"
                            )
                            self.event_sink.on_trigger_code_review_skipped(
                                trigger_type.value, skip_reason or "unknown"
                            )
                        elif code_review_end_status == "passed":
                            self.event_sink.on_trigger_code_review_passed(
                                trigger_type.value
                            )
                        elif code_review_end_status == "failed":
                            self.event_sink.on_trigger_code_review_failed(
                                trigger_type.value, code_review_blocking_count
                            )

            # Handle code review result for trigger validation flow
            # Note: status="failed" (execution error) is handled above with early return,
            # so here we only handle status="success" or "skipped" cases.
            # Also gate on code_review_enabled to avoid handling when disabled.
            if (
                code_review_enabled
                and review_result is not None
                and code_review_config is not None
            ):
                review_failure_mode = code_review_config.failure_mode
                threshold = code_review_config.finding_threshold

                # Check finding_threshold for successful reviews
                if (
                    review_result is not None
                    and review_result.status == "success"
                    and self._findings_exceed_threshold(
                        review_result.findings, threshold
                    )
                ):
                    # Findings exceed threshold - handle based on failure_mode
                    if review_failure_mode == FailureMode.ABORT:
                        # Abort immediately without remediation
                        if self.event_sink is not None:
                            self.event_sink.on_trigger_validation_failed(
                                trigger_type.value, "code_review_findings", "abort"
                            )
                        self.clear_trigger_queue("code_review_findings_exceeded")
                        return TriggerValidationResult(
                            status="aborted",
                            details=(
                                f"Code review findings exceed threshold ({threshold}) "
                                f"in trigger {trigger_type.value}"
                            ),
                        )
                    elif review_failure_mode == FailureMode.CONTINUE:
                        # Record failure and continue without remediation
                        last_failure = ("code_review_findings", trigger_type.value)
                        if self.event_sink is not None:
                            self.event_sink.on_trigger_validation_failed(
                                trigger_type.value,
                                "code_review_findings",
                                "continue",
                            )
                    elif review_failure_mode == FailureMode.REMEDIATE:
                        # Attempt remediation via fixer agent
                        remediation_result = await self._run_code_review_remediation(
                            trigger_type,
                            trigger_config,
                            context,
                            interrupt_event,
                            review_result.findings,
                        )

                        if interrupt_event.is_set():
                            # Emit error event for SIGINT (not failed - that's for findings)
                            # to maintain exactly-one-end-event invariant
                            if (
                                defer_code_review_end_event
                                and self.event_sink is not None
                            ):
                                defer_code_review_end_event = False
                                code_review_end_status = None
                                self.event_sink.on_trigger_code_review_error(
                                    trigger_type.value, "SIGINT"
                                )
                            self.clear_trigger_queue("sigint")
                            return TriggerValidationResult(
                                status="aborted",
                                details="Validation interrupted by SIGINT",
                            )

                        # Determine final state from remediation result
                        # remediation_result is None only if interrupted or couldn't run
                        # Otherwise it contains the last review result for proper event emission
                        if remediation_result is None:
                            # Remediation couldn't run (max_retries=0, interrupted, etc.)
                            # Emit deferred code_review_failed event and clear defer flag
                            # to prevent any accidental double-emission
                            last_failure = (
                                "code_review_findings",
                                trigger_type.value,
                            )
                            if self.event_sink is not None:
                                self.event_sink.on_trigger_validation_failed(
                                    trigger_type.value,
                                    "code_review_findings",
                                    "remediate",
                                )
                            if (
                                defer_code_review_end_event
                                and self.event_sink is not None
                            ):
                                code_review_end_status = None
                                self.event_sink.on_trigger_code_review_failed(
                                    trigger_type.value, code_review_blocking_count
                                )
                            # Always clear defer flag to prevent double-emission,
                            # regardless of whether event_sink is available
                            defer_code_review_end_event = False
                        elif remediation_result.status == "failed":
                            # Last re-review had execution error - emit error event
                            # This is the terminal event, not failed/passed
                            # Clear flags to prevent duplicate terminal events
                            defer_code_review_end_event = False
                            code_review_end_status = None
                            error_reason = (
                                remediation_result.skip_reason or "unknown error"
                            )
                            if self.event_sink is not None:
                                self.event_sink.on_trigger_code_review_error(
                                    trigger_type.value,
                                    error_reason,
                                )
                            # Abort validation - execution error is a hard failure
                            self.clear_trigger_queue("code_review_execution_error")
                            return TriggerValidationResult(
                                status="aborted",
                                details=f"Code review execution failed after remediation: {error_reason}",
                            )
                        elif self._findings_exceed_threshold(
                            remediation_result.findings,
                            code_review_config.finding_threshold,
                        ):
                            # Remediation exhausted - findings still exceed threshold
                            last_failure = (
                                "code_review_findings",
                                trigger_type.value,
                            )
                            if self.event_sink is not None:
                                self.event_sink.on_trigger_validation_failed(
                                    trigger_type.value,
                                    "code_review_findings",
                                    "remediate",
                                )
                            # Emit deferred code_review_failed with updated count
                            if (
                                defer_code_review_end_event
                                and self.event_sink is not None
                            ):
                                code_review_end_status = None
                                # Count blocking findings from final review result
                                final_blocking_count = sum(
                                    1
                                    for f in remediation_result.findings
                                    if f.priority
                                    <= int(code_review_config.finding_threshold[1])
                                )
                                self.event_sink.on_trigger_code_review_failed(
                                    trigger_type.value, final_blocking_count
                                )
                            # Always clear defer flag to prevent double-emission,
                            # regardless of whether event_sink is available
                            defer_code_review_end_event = False
                        else:
                            # Remediation succeeded - findings dropped below threshold
                            # Clear code_review failure only. Command failures from this
                            # trigger (if failure_mode=CONTINUE) must be preserved.
                            if last_failure is not None and last_failure[0] in (
                                "code_review",
                                "code_review_findings",
                            ):
                                last_failure = None
                            # Emit deferred code_review_passed event
                            if (
                                defer_code_review_end_event
                                and self.event_sink is not None
                            ):
                                code_review_end_status = None
                                self.event_sink.on_trigger_code_review_passed(
                                    trigger_type.value
                                )
                            # Always clear defer flag to prevent double-emission,
                            # regardless of whether event_sink is available
                            defer_code_review_end_event = False

            # Emit validation_passed (only if no failure occurred)
            if last_failure is None:
                trigger_duration = time.monotonic() - trigger_start_time
                if self.event_sink is not None:
                    self.event_sink.on_trigger_validation_passed(
                        trigger_type.value, trigger_duration
                    )

        # All triggers processed - return failed if any CONTINUE failures occurred
        if last_failure is not None:
            cmd_ref, trigger_val = last_failure
            return TriggerValidationResult(
                status="failed",
                details=f"Command '{cmd_ref}' failed in trigger {trigger_val}",
            )

        return TriggerValidationResult(status="passed")

    def _record_run_validation(
        self,
        trigger_type: TriggerType,
        results: list[TriggerCommandResult],
    ) -> None:
        """Record run_end validation results into run metadata."""
        from src.infra.io.log_output.run_metadata import (
            ValidationResult as MetaValidationResult,
        )

        if self.run_metadata is None:
            return
        from src.domain.validation.config import TriggerType as TriggerTypeEnum

        if trigger_type != TriggerTypeEnum.RUN_END:
            return
        if not results:
            return

        coverage_percent: float | None = None
        validation_config = self.config.validation_config
        if validation_config is not None and validation_config.coverage is not None:
            from pathlib import Path

            from src.domain.validation.coverage import parse_coverage_xml

            report_path = Path(validation_config.coverage.file)
            if not report_path.is_absolute():
                report_path = self.config.repo_path / report_path
            coverage_result = parse_coverage_xml(report_path)
            if coverage_result.percent is not None:
                coverage_percent = coverage_result.percent

        meta_result = MetaValidationResult(
            passed=all(result.passed for result in results),
            commands_run=[result.ref for result in results],
            commands_failed=[result.ref for result in results if not result.passed],
            artifacts=None,
            coverage_percent=coverage_percent,
            e2e_passed=None,
        )
        self.run_metadata.record_run_validation(meta_result)

    async def _run_trigger_remediation(
        self,
        trigger_type: TriggerType,
        trigger_config: BaseTriggerConfig,
        resolved_commands: list[ResolvedCommand],
        failed_result: TriggerCommandResult,
        failed_index: int,
        dry_run: bool,
        interrupt_event: asyncio.Event,
    ) -> tuple[TriggerValidationResult | None, TriggerCommandResult | None]:
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
            Tuple of (TriggerValidationResult, TriggerCommandResult).
            TriggerValidationResult is returned when remediation fails or aborts.
            TriggerCommandResult is returned when remediation succeeds.
        """
        max_retries = trigger_config.max_retries or 0

        # max_retries=0 means no fixer spawned, immediate abort
        if max_retries == 0:
            self.clear_trigger_queue("run_aborted")
            return (
                TriggerValidationResult(
                    status="aborted",
                    details=(
                        f"Command '{failed_result.ref}' failed in trigger {trigger_type.value} (max_retries=0)"
                    ),
                ),
                None,
            )

        # Find the failed command to re-run
        failed_cmd = next(
            (cmd for cmd in resolved_commands if cmd.ref == failed_result.ref), None
        )
        if failed_cmd is None:
            # Shouldn't happen, but handle gracefully
            self.clear_trigger_queue("run_aborted")
            return (
                TriggerValidationResult(
                    status="aborted",
                    details=(
                        f"Failed to find command '{failed_result.ref}' for remediation"
                    ),
                ),
                None,
            )

        # Build failure output for fixer
        failure_output = self._build_trigger_failure_output(failed_result)

        attempts_made = 0
        for attempt in range(1, max_retries + 1):
            attempts_made = attempt
            # Check for SIGINT before fixer attempt
            if interrupt_event.is_set():
                self.clear_trigger_queue("sigint")
                return (
                    TriggerValidationResult(
                        status="aborted", details="Validation interrupted by SIGINT"
                    ),
                    None,
                )

            # Emit remediation_started event
            if self.event_sink is not None:
                self.event_sink.on_trigger_remediation_started(
                    trigger_type.value, attempt, max_retries
                )

            # Emit fixer_started event per remediation attempt
            if self.event_sink is not None:
                self.event_sink.on_fixer_started(attempt, max_retries)

            # Spawn fixer agent via FixerService
            fixer_context = FailureContext(
                failure_output=failure_output,
                attempt=attempt,
                max_attempts=max_retries,
                failed_command=failed_cmd.effective_command,
                validation_commands=f"   - `{failed_cmd.effective_command}`",
            )
            fixer_result = await self.fixer_service.run_fixer(
                fixer_context, interrupt_event
            )

            # Check for SIGINT after fixer (either via result or event)
            if fixer_result.interrupted or interrupt_event.is_set():
                self.clear_trigger_queue("sigint")
                return (
                    TriggerValidationResult(
                        status="aborted", details="Validation interrupted by SIGINT"
                    ),
                    None,
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
                start_index=failed_index,
                total_commands=len(resolved_commands),
            )

            if was_interrupted:
                self.clear_trigger_queue("sigint")
                return (
                    TriggerValidationResult(
                        status="aborted", details="Validation interrupted by SIGINT"
                    ),
                    None,
                )

            if retry_results and retry_results[0].passed:
                # Command passed after fixer, remediation succeeded
                if self.event_sink is not None:
                    self.event_sink.on_trigger_remediation_succeeded(
                        trigger_type.value, attempt
                    )
                return None, retry_results[0]

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
        return (
            TriggerValidationResult(
                status="aborted",
                details=(
                    f"Command '{failed_result.ref}' failed after {max_retries} remediation attempts"
                ),
            ),
            None,
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
        start_index: int = 0,
        total_commands: int,
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
            start_index: Offset for command indices in emitted events.
            total_commands: Total number of commands in this validation.

        Returns:
            Tuple of (results, was_interrupted) where was_interrupted is True
            if execution was cancelled due to SIGINT.
        """
        results: list[TriggerCommandResult] = []

        for index, cmd in enumerate(commands):
            command_index = start_index + index
            # Check interrupt before each command
            if interrupt_event.is_set():
                return results, True

            # Emit command_started event
            if self.event_sink is not None:
                self.event_sink.on_trigger_command_started(
                    trigger_type.value, cmd.ref, command_index, total_commands
                )

            if dry_run:
                # Dry-run: all commands pass, no execution
                duration = 0.0
                passed = True
                # Emit command_completed event
                if self.event_sink is not None:
                    self.event_sink.on_trigger_command_completed(
                        trigger_type.value,
                        cmd.ref,
                        command_index,
                        total_commands,
                        passed,
                        duration,
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
                    command_index,
                    total_commands,
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
        # Return failed status instead of None to distinguish wiring errors
        # from intentional skips. This ensures the caller emits an error event
        # rather than a skipped event (see issue: avoid started+skipped for wiring errors).
        if self.cumulative_review_runner is None:
            from src.pipeline.cumulative_review_runner import CumulativeReviewResult

            if self.event_sink is not None:
                self.event_sink.on_warning(
                    f"code_review enabled for {trigger_type.value} but "
                    "CumulativeReviewRunner not wired"
                )
            return CumulativeReviewResult(
                status="failed",
                findings=(),
                new_baseline_commit=None,
                skip_reason="CumulativeReviewRunner not wired",
            )

        # Check if run_metadata is available
        if self.run_metadata is None:
            from src.pipeline.cumulative_review_runner import CumulativeReviewResult

            if self.event_sink is not None:
                self.event_sink.on_warning(
                    f"code_review enabled for {trigger_type.value} but "
                    "run_metadata not available"
                )
            return CumulativeReviewResult(
                status="failed",
                findings=(),
                new_baseline_commit=None,
                skip_reason="run_metadata not available",
            )

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

    def _findings_exceed_threshold(
        self,
        findings: tuple[ReviewFinding, ...],
        threshold: str,
    ) -> bool:
        """Check if any findings exceed the severity threshold.

        Args:
            findings: Tuple of review findings to check.
            threshold: Threshold string ("P0", "P1", "P2", "P3", or "none").
                "none" means never fail on findings.
                "P1" means fail if any P0 or P1 findings exist.

        Returns:
            True if findings exceed threshold, False otherwise.
        """
        if threshold == "none":
            return False

        # Convert threshold to numeric: "P0" -> 0, "P1" -> 1, etc.
        threshold_priority = int(threshold[1])

        # Check if any finding has priority <= threshold
        # (P0 is priority 0, P1 is priority 1, etc.)
        for finding in findings:
            if finding.priority <= threshold_priority:
                return True

        return False

    async def _run_code_review_remediation(
        self,
        trigger_type: TriggerType,
        trigger_config: BaseTriggerConfig,
        context: dict[str, Any],
        interrupt_event: asyncio.Event,
        initial_findings: tuple[ReviewFinding, ...],
    ) -> CumulativeReviewResult | None:
        """Run remediation loop for code review findings that exceed threshold.

        Spawns fixer agent to address findings and re-runs review up to max_retries.

        Args:
            trigger_type: The trigger type for context.
            trigger_config: The trigger's configuration.
            context: Trigger context with issue_id, epic_id, etc.
            interrupt_event: Event set when SIGINT received.
            initial_findings: The findings that exceeded threshold.

        Returns:
            CumulativeReviewResult with final state after remediation attempts:
            - status="success" with findings below threshold: remediation succeeded
            - status="success" with findings above threshold: remediation exhausted
            - status="failed": last re-review had execution error
            Returns None only if remediation couldn't run (max_retries=0, interrupted,
            or review couldn't run at all).
        """
        code_review_config = trigger_config.code_review
        if code_review_config is None:
            return None

        max_retries = code_review_config.max_retries
        threshold = code_review_config.finding_threshold

        # max_retries=0 means no fixer spawned, immediate failure
        if max_retries == 0:
            return None

        # Build failure output for fixer from findings
        failure_output = self._build_findings_failure_output(
            initial_findings, threshold
        )

        # Track last review result to return on exhaustion
        last_review_result: CumulativeReviewResult | None = None

        for attempt in range(1, max_retries + 1):
            # Check for SIGINT before fixer attempt
            if interrupt_event.is_set():
                return None

            # Emit remediation_started event
            if self.event_sink is not None:
                self.event_sink.on_trigger_remediation_started(
                    trigger_type.value, attempt, max_retries
                )

            # Emit fixer_started event per remediation attempt
            if self.event_sink is not None:
                self.event_sink.on_fixer_started(attempt, max_retries)

            # Build validation commands summary for fixer
            findings_summary = "\n".join(
                f"   - {f.file}:{f.line_start}: P{f.priority} - {f.title}"
                for f in initial_findings
            )

            # Spawn fixer agent via FixerService to address findings
            fixer_context = FailureContext(
                failure_output=failure_output,
                attempt=attempt,
                max_attempts=max_retries,
                failed_command="code_review",
                validation_commands=f"Review findings:\n{findings_summary}",
            )
            fixer_result = await self.fixer_service.run_fixer(
                fixer_context, interrupt_event
            )

            # Check for SIGINT after fixer
            if fixer_result.interrupted or interrupt_event.is_set():
                return None

            # Re-run code review
            review_result = await self._run_trigger_code_review(
                trigger_type, trigger_config, context, interrupt_event
            )

            if review_result is None:
                # Review couldn't run (not configured, etc.)
                return None

            # Track this result as the last known state
            last_review_result = review_result

            if review_result.status == "failed":
                # Execution error during re-review - update failure output
                failure_output = (
                    f"Code review execution failed: {review_result.skip_reason}"
                )
                continue

            # Check if findings now below threshold
            if not self._findings_exceed_threshold(review_result.findings, threshold):
                # Remediation succeeded
                if self.event_sink is not None:
                    self.event_sink.on_trigger_remediation_succeeded(
                        trigger_type.value, attempt
                    )
                return review_result

            # Update failure output for next attempt
            failure_output = self._build_findings_failure_output(
                review_result.findings, threshold
            )
            initial_findings = review_result.findings

        # Remediation exhausted
        if self.event_sink is not None:
            self.event_sink.on_trigger_remediation_exhausted(
                trigger_type.value, max_retries
            )

        # Return last review result so caller can determine proper terminal event
        return last_review_result

    def _build_findings_failure_output(
        self,
        findings: tuple[ReviewFinding, ...],
        threshold: str,
    ) -> str:
        """Build failure output string for fixer agent from review findings.

        Args:
            findings: The review findings to format.
            threshold: The threshold that was exceeded.

        Returns:
            Human-readable failure output for fixer prompt.
        """
        lines: list[str] = [
            f"Code review found findings that exceed threshold ({threshold}):",
            "",
        ]

        for finding in findings:
            # Only include findings at or above threshold
            threshold_priority = int(threshold[1]) if threshold != "none" else 999
            if finding.priority <= threshold_priority:
                lines.append(f"## P{finding.priority}: {finding.title}")
                lines.append(
                    f"File: {finding.file}:{finding.line_start}-{finding.line_end}"
                )
                lines.append(f"{finding.body}")
                lines.append("")

        lines.append("Please fix these issues so the code review passes.")
        return "\n".join(lines)

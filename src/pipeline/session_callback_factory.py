"""SessionCallbackFactory: Builds SessionCallbacks for agent sessions.

This factory encapsulates callback construction that bridges orchestrator state
to the pipeline runners. It receives state references and returns a SessionCallbacks
instance wired to the appropriate callbacks.

Additionally, SessionCallbackFactory implements the IGateRunner, IReviewRunner,
and ISessionLifecycle protocols, allowing it to be used directly as a protocol
implementation when constructing AgentSessionRunner instances.

Design principles:
- Single responsibility: only builds callbacks, doesn't run gates/reviews
- Protocol-based dependencies for testability
- All callback closures capture minimal state
- Late-bound lookups: dependencies are accessed via callables to support
  runtime patching (e.g., tests that swap event_sink after construction)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from src.core.session_end_result import (
    CodeReviewResult,
    CommandOutcome,
    SessionEndResult,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from src.core.protocols.lifecycle import ISessionLifecycle
    from src.core.protocols.review import (
        IReviewRunner,
        ReviewIssueProtocol,
        ReviewOutcomeProtocol,
    )
    from src.core.protocols.validation import (
        GateChecker,
        GateOutcomeProtocol,
        IGateRunner,
        RetryStateProtocol,
    )
    from src.core.protocols.events import MalaEventSink
    from src.core.protocols.infra import CommandResultProtocol, CommandRunnerPort
    from src.core.protocols.log import LogProvider
    from src.domain.lifecycle import (
        GateOutcome,
        RetryState,
        ReviewIssue,
    )
    from src.domain.validation.config import (
        SessionEndTriggerConfig,
        TriggerCommandRef,
        ValidationConfig,
    )
    from src.domain.validation.spec import ValidationSpec
    from src.infra.io.log_output.run_metadata import RunMetadata
    from src.pipeline.cumulative_review_runner import CumulativeReviewRunner
    from src.pipeline.fixer_interface import FixerInterface
    from src.pipeline.review_runner import ReviewRunner
    from src.core.session_end_result import SessionEndRetryState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SessionRunContext:
    """Late-bound getters for session callback wiring.

    Bundles the 9 callable parameters currently passed individually to
    build_session_callback_factory(), consolidating "lambda soup" into a
    single typed context object.

    All fields are callables to support late-binding - values are resolved
    at call time rather than construction time.

    Attributes:
        log_provider_getter: Returns the LogProvider for session logging.
        evidence_check_getter: Returns the GateChecker for evidence validation.
        on_session_log_path: Called with (issue_id, path) when session log path is known.
        on_review_log_path: Called with (issue_id, path) when review log path is known.
        interrupt_event_getter: Returns the interrupt Event or None if not set.
        get_base_sha: Returns base SHA for an issue, or None if unavailable.
        get_run_metadata: Returns RunMetadata or None if unavailable.
        on_abort: Called with issue_id when session is aborted, or None to disable.
        abort_event_getter: Returns the abort Event or None if not set.
    """

    log_provider_getter: Callable[[], LogProvider]
    evidence_check_getter: Callable[[], GateChecker]
    on_session_log_path: Callable[[str, Path], None]
    on_review_log_path: Callable[[str, str], None]
    interrupt_event_getter: Callable[[], asyncio.Event | None]
    get_base_sha: Callable[[str], str | None]
    get_run_metadata: Callable[[], RunMetadata | None]
    on_abort: Callable[[str], None] | None
    abort_event_getter: Callable[[], asyncio.Event | None]


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
        context = SessionRunContext(
            log_provider_getter=...,
            evidence_check_getter=...,
            on_session_log_path=...,
            on_review_log_path=...,
            ...
        )
        factory = SessionCallbackFactory(
            gate_async_runner=...,
            review_runner=...,
            context=context,
            event_sink=...,
            repo_path=...,
        )
        callbacks = factory.build(issue_id)
    """

    def __init__(
        self,
        gate_async_runner: GateAsyncRunner,
        review_runner: ReviewRunner,
        context: SessionRunContext,
        event_sink: Callable[[], MalaEventSink],
        repo_path: Path,
        get_per_session_spec: GetPerSessionSpec,
        is_verbose: IsVerboseCheck,
        get_validation_config: Callable[[], ValidationConfig | None] | None = None,
        command_runner: CommandRunnerPort | None = None,
        fixer_interface: FixerInterface | None = None,
        cumulative_review_runner: CumulativeReviewRunner | None = None,
        session_end_timeout: float = 600.0,
    ) -> None:
        """Initialize the factory with dependencies.

        Args:
            gate_async_runner: Protocol for running async gate checks.
            review_runner: Runner for Cerberus code review.
            context: SessionRunContext with late-bound getters for orchestrator state.
            event_sink: Callable returning the event sink (late-bound).
            repo_path: Repository path for git operations.
            get_per_session_spec: Callable to get current per-session spec.
            is_verbose: Callable to check verbose mode.
            get_validation_config: Callable to get validation config (late-bound).
            command_runner: Command runner for executing session_end commands.
            fixer_interface: Interface for running fixer agents during remediation.
            cumulative_review_runner: Runner for session_end code review.
            session_end_timeout: Overall timeout for session_end in seconds.

        Note:
            The context bundles late-bound getters (log_provider, evidence_check,
            interrupt_event, etc.) to support runtime patching. This allows tests
            to swap orchestrator attributes after factory construction.
        """
        self._gate_async_runner = gate_async_runner
        self._review_runner = review_runner
        self._context = context
        self._get_event_sink = event_sink
        self._repo_path = repo_path
        self._get_per_session_spec = get_per_session_spec
        self._is_verbose = is_verbose
        self._get_validation_config = get_validation_config or (lambda: None)
        self._command_runner = command_runner
        self._fixer_interface = fixer_interface
        self._cumulative_review_runner = cumulative_review_runner
        self._session_end_timeout = session_end_timeout

    def build_adapters(
        self,
        issue_id: str,
        on_abort: Callable[[str], None] | None = None,
    ) -> SessionRunnerAdapters:
        """Build protocol adapters for AgentSessionRunner.

        This is the preferred method for obtaining protocol implementations
        to pass to AgentSessionRunner. Use this instead of the deprecated
        build() method which returns SessionCallbacks.

        Args:
            issue_id: The issue ID for tracking state.
            on_abort: Optional callback for fatal error signaling.

        Returns:
            SessionRunnerAdapters with IGateRunner, IReviewRunner, and
            ISessionLifecycle implementations.
        """
        return SessionRunnerAdapters(
            gate_runner=_GateRunnerAdapter(self, issue_id),
            review_runner=_ReviewRunnerAdapter(self, issue_id),
            session_lifecycle=_SessionLifecycleAdapter(self, issue_id, on_abort),
        )

    async def _execute_session_end(
        self,
        issue_id: str,
        log_path: Path,
        retry_state: SessionEndRetryState,
    ) -> SessionEndResult:
        """Execute session_end trigger with timeout wrapper.

        Per spec R9-R11:
        - Commands execute sequentially with individual timeouts
        - Overall session_end wrapped in asyncio.timeout
        - On timeout: return status="timeout" with empty commands
        - On SIGINT: complete current command, return status="interrupted"
        - code_review uses base_sha..HEAD range
        - failure_mode=abort sets abort_event; continue proceeds regardless
        """
        from src.domain.validation.config import FailureMode

        # Get session_end config
        validation_config = self._get_validation_config()
        if validation_config is None or validation_config.validation_triggers is None:
            return SessionEndResult(status="skipped", reason="not_configured")

        session_end_config = validation_config.validation_triggers.session_end
        if session_end_config is None:
            return SessionEndResult(status="skipped", reason="not_configured")

        # Check if we have the necessary dependencies
        if self._command_runner is None and session_end_config.commands:
            logger.warning("session_end has commands but no command_runner configured")
            return SessionEndResult(
                status="skipped", reason="command_runner_not_configured"
            )

        started_at = datetime.now(UTC)
        interrupt_event = self._context.interrupt_event_getter()
        command_outcomes: list[CommandOutcome] = []

        try:
            async with asyncio.timeout(self._session_end_timeout):
                # Execute commands sequentially
                all_passed = True
                for cmd_ref in session_end_config.commands:
                    # Check for interrupt before each command
                    if interrupt_event and interrupt_event.is_set():
                        logger.info(
                            "session_end interrupted before command %s for %s",
                            cmd_ref.ref,
                            issue_id,
                        )
                        # Per spec R10: discard partial results on interrupt
                        return SessionEndResult(
                            status="interrupted",
                            started_at=started_at,
                            finished_at=datetime.now(UTC),
                            commands=[],
                            reason="SIGINT received",
                        )

                    # Resolve command from base pool
                    resolved_cmd, resolved_timeout = self._resolve_command(
                        cmd_ref, validation_config
                    )
                    if resolved_cmd is None:
                        logger.warning(
                            "session_end command ref '%s' not found in base pool",
                            cmd_ref.ref,
                        )
                        all_passed = False
                        break

                    # Execute the command
                    # Per R10: "complete current command" on SIGINT (but not on timeout)
                    # Shield protects from cancellation while allowing timeout to still work
                    if self._command_runner is not None:
                        cmd_task = asyncio.create_task(
                            self._command_runner.run_async(
                                resolved_cmd,
                                timeout=resolved_timeout,
                                shell=True,
                                cwd=self._repo_path,
                            )
                        )
                        try:
                            result = await asyncio.shield(cmd_task)
                        except asyncio.CancelledError:
                            # Shield prevents cmd_task from being cancelled.
                            # Check if this is SIGINT (interrupt_event set) or timeout.
                            # Per R10: complete command on SIGINT, but not on timeout.
                            if interrupt_event and interrupt_event.is_set():
                                # SIGINT: complete the command, then discard partial results per R10
                                result = await cmd_task
                                # Log outcome but don't include in result
                                _ = self._to_command_outcome(cmd_ref.ref, result)
                                logger.info(
                                    "session_end interrupted during command %s for %s",
                                    cmd_ref.ref,
                                    issue_id,
                                )
                                # Per spec R10: discard partial results on interrupt
                                return SessionEndResult(
                                    status="interrupted",
                                    started_at=started_at,
                                    finished_at=datetime.now(UTC),
                                    commands=[],
                                    reason="SIGINT received",
                                )
                            # Not SIGINT (likely timeout): cancel the inner task and re-raise
                            # to let timeout handler return with empty commands
                            cmd_task.cancel()
                            raise
                        outcome = self._to_command_outcome(cmd_ref.ref, result)
                        command_outcomes.append(outcome)

                        # Check for interrupt after command completes
                        if interrupt_event and interrupt_event.is_set():
                            logger.info(
                                "session_end interrupted after command %s for %s",
                                cmd_ref.ref,
                                issue_id,
                            )
                            # Per spec R10: discard partial results on interrupt
                            return SessionEndResult(
                                status="interrupted",
                                started_at=started_at,
                                finished_at=datetime.now(UTC),
                                commands=[],
                                reason="SIGINT received",
                            )

                        if not result.ok:
                            all_passed = False
                            # Fail-fast: stop on first failure
                            break

                # Handle failure modes (abort, continue, remediate)
                failure_mode = session_end_config.failure_mode
                if not all_passed:
                    if failure_mode == FailureMode.ABORT:
                        logger.info(
                            "session_end failed with abort mode for %s",
                            issue_id,
                        )
                        # Set abort_event directly for immediate propagation
                        abort_event = self._context.abort_event_getter()
                        if abort_event is not None:
                            abort_event.set()
                        if self._context.on_abort:
                            self._context.on_abort(
                                f"session_end validation failed for {issue_id}"
                            )
                        return SessionEndResult(
                            status="fail",
                            started_at=started_at,
                            finished_at=datetime.now(UTC),
                            commands=list(command_outcomes),
                            reason="command_failed",
                        )
                    elif failure_mode == FailureMode.REMEDIATE:
                        # Remediation loop: run fixer and retry validation
                        remediation_result = await self._run_remediation_loop(
                            issue_id=issue_id,
                            session_end_config=session_end_config,
                            validation_config=validation_config,
                            command_outcomes=command_outcomes,
                            started_at=started_at,
                            interrupt_event=interrupt_event,
                            retry_state=retry_state,
                        )
                        if remediation_result is not None:
                            # Remediation completed (success, exhausted, or interrupted)
                            # Now run code_review once after final outcome
                            code_review_result = (
                                await self._run_session_end_code_review(
                                    issue_id=issue_id,
                                    session_end_config=session_end_config,
                                    commands_passed=(
                                        remediation_result.status == "pass"
                                    ),
                                    interrupt_event=interrupt_event,
                                )
                            )
                            # Update result with code_review
                            return SessionEndResult(
                                status=remediation_result.status,
                                started_at=remediation_result.started_at,
                                finished_at=datetime.now(UTC),
                                commands=remediation_result.commands,
                                code_review_result=code_review_result,
                                reason=remediation_result.reason,
                            )
                    # continue mode: proceed to code_review regardless
                    logger.info(
                        "session_end commands failed but continuing for %s",
                        issue_id,
                    )

                # Run code_review if configured
                code_review_result = await self._run_session_end_code_review(
                    issue_id=issue_id,
                    session_end_config=session_end_config,
                    commands_passed=all_passed,
                    interrupt_event=interrupt_event,
                )

                # Determine final status
                status = "pass" if all_passed else "fail"
                reason = None if all_passed else "command_failed"

                return SessionEndResult(
                    status=status,
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                    commands=list(command_outcomes),
                    code_review_result=code_review_result,
                    reason=reason,
                )

        except TimeoutError:
            # Per spec R10: on timeout, return with empty commands
            logger.warning(
                "session_end timed out after %.1fs for %s",
                self._session_end_timeout,
                issue_id,
            )
            return SessionEndResult(
                status="timeout",
                started_at=started_at,
                finished_at=datetime.now(UTC),
                commands=[],
                reason="session_end_timeout",
            )
        except asyncio.CancelledError:
            # Handle cancellation outside command execution (e.g., during code_review)
            # Command-level cancellation is handled inline with task awaiting
            logger.info("session_end cancelled for %s", issue_id)
            # Per spec R10: discard partial results on interrupt/cancellation
            return SessionEndResult(
                status="interrupted",
                started_at=started_at,
                finished_at=datetime.now(UTC),
                commands=[],
                reason="cancelled",
            )

    def _to_command_outcome(
        self, ref: str, result: CommandResultProtocol
    ) -> CommandOutcome:
        """Convert a CommandResult to a spec-compliant CommandOutcome.

        Per spec R5, command outcomes use the schema:
        {ref, passed, duration_seconds, error_message}

        Args:
            ref: The command reference name (e.g., "test", "lint").
            result: The raw command execution result.

        Returns:
            CommandOutcome with spec-compliant fields.
        """
        # error_message captures failure details, None if passed
        error_message: str | None = None
        if not result.ok:
            # Include stderr, stdout (for tools like pytest), or timeout message
            if result.timed_out:
                error_message = "Command timed out"
            elif result.stderr:
                error_message = result.stderr
            elif result.stdout:
                # Many tools (pytest, etc.) output failure details to stdout
                error_message = result.stdout
            else:
                error_message = f"Exit code: {result.returncode}"

        return CommandOutcome(
            ref=ref,
            passed=result.ok,
            duration_seconds=result.duration_seconds,
            error_message=error_message,
        )

    def _resolve_command(
        self,
        cmd_ref: TriggerCommandRef,
        validation_config: ValidationConfig,
    ) -> tuple[str | None, int | None]:
        """Resolve a command reference from the base pool.

        Args:
            cmd_ref: The command reference with optional overrides.
            validation_config: The validation config containing base pool.

        Returns:
            Tuple of (resolved_command, resolved_timeout) or (None, None) if not found.
        """
        # Build base pool from commands
        base_pool: dict[str, tuple[str, int | None]] = {}

        # Standard commands from commands
        commands_config = validation_config.commands

        for name in ["test", "lint", "format", "typecheck", "e2e", "setup", "build"]:
            cmd_config = getattr(commands_config, name, None)
            if cmd_config is not None:
                base_pool[name] = (
                    cmd_config.command,
                    getattr(cmd_config, "timeout", None),
                )

        # Add custom commands from commands.custom_commands (repo-level)
        if commands_config and commands_config.custom_commands:
            for name, custom_config in commands_config.custom_commands.items():
                if custom_config is not None:
                    base_pool[name] = (
                        custom_config.command,
                        getattr(custom_config, "timeout", None),
                    )

        # Look up the ref
        if cmd_ref.ref not in base_pool:
            return None, None

        base_cmd, base_timeout = base_pool[cmd_ref.ref]

        # Apply overrides
        effective_cmd = cmd_ref.command if cmd_ref.command is not None else base_cmd
        effective_timeout = (
            cmd_ref.timeout if cmd_ref.timeout is not None else base_timeout
        )

        return effective_cmd, effective_timeout

    async def _run_remediation_loop(
        self,
        issue_id: str,
        session_end_config: SessionEndTriggerConfig,
        validation_config: ValidationConfig,
        command_outcomes: list[CommandOutcome],
        started_at: datetime,
        interrupt_event: asyncio.Event | None,
        retry_state: SessionEndRetryState,
    ) -> SessionEndResult | None:
        """Run remediation loop for session_end when failure_mode=remediate.

        Per spec R9:
        - Fixer runs before each retry (not before initial attempt)
        - Total validation attempts = 1 + max_retries
        - On exhausted retries: status="fail", reason="max_retries_exhausted"
        - On successful retry: status="pass"

        Args:
            issue_id: The issue ID for context.
            session_end_config: The session_end trigger config.
            validation_config: The full validation config for command resolution.
            command_outcomes: Command outcomes from initial attempt (will be updated).
            started_at: Timestamp when session_end started.
            interrupt_event: Event to check for interruption.
            retry_state: Retry state for tracking attempt number.

        Returns:
            SessionEndResult if remediation completed/exhausted/interrupted,
            None if max_retries=0 or fixer_interface not available (falls back).
        """
        max_retries = session_end_config.max_retries or 0

        # max_retries=0 means no retries configured, fall back to continue mode
        # The initial attempt already happened and failed with command_failed
        if max_retries == 0:
            logger.info(
                "session_end failed for %s with max_retries=0, no remediation",
                issue_id,
            )
            return None

        # Check if fixer_interface is available
        if self._fixer_interface is None:
            logger.warning(
                "session_end remediate mode for %s but no fixer_interface configured",
                issue_id,
            )
            # Fall back to continue mode - return None to indicate no remediation
            return None

        # Build failure output from last command outcome
        failure_output = self._build_failure_output(command_outcomes)

        # Remediation loop: run fixer and retry validation
        for retry_num in range(1, max_retries + 1):
            # Check for interrupt before fixer
            if interrupt_event and interrupt_event.is_set():
                logger.info(
                    "session_end remediation interrupted before fixer for %s",
                    issue_id,
                )
                # Per spec R10: discard partial results on interrupt
                return SessionEndResult(
                    status="interrupted",
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                    commands=[],
                    reason="SIGINT received",
                )

            logger.info(
                "session_end remediation attempt %d/%d for %s",
                retry_num,
                max_retries,
                issue_id,
            )

            # Run fixer
            fixer_result = await self._fixer_interface.run_fixer(
                failure_output=failure_output,
                issue_id=issue_id,
            )

            # Update retry_state.attempt after fixer run (per acceptance criteria)
            # attempt = 1 is initial, so after fixer we increment to 2, 3, etc.
            retry_state.attempt = retry_num + 1

            # Check if fixer was interrupted
            if fixer_result.interrupted:
                logger.info(
                    "session_end fixer interrupted for %s",
                    issue_id,
                )
                # Per spec R10: discard partial results on interrupt
                return SessionEndResult(
                    status="interrupted",
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                    commands=[],
                    reason="fixer_interrupted",
                )

            # Check for interrupt after fixer
            if interrupt_event and interrupt_event.is_set():
                logger.info(
                    "session_end remediation interrupted after fixer for %s",
                    issue_id,
                )
                # Per spec R10: discard partial results on interrupt
                return SessionEndResult(
                    status="interrupted",
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                    commands=[],
                    reason="SIGINT received",
                )

            # Re-run validation commands
            # Require command_runner for retry - if missing, fail the remediation
            if self._command_runner is None:
                logger.error(
                    "session_end remediation: command_runner missing during retry for %s",
                    issue_id,
                )
                return SessionEndResult(
                    status="fail",
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                    commands=list(command_outcomes),
                    reason="command_runner_not_configured",
                )

            retry_passed = True
            retry_outcomes: list[CommandOutcome] = []
            for cmd_ref in session_end_config.commands:
                # Check for interrupt before each command
                if interrupt_event and interrupt_event.is_set():
                    logger.info(
                        "session_end retry interrupted before command %s for %s",
                        cmd_ref.ref,
                        issue_id,
                    )
                    # Per spec R10: discard partial results on interrupt
                    return SessionEndResult(
                        status="interrupted",
                        started_at=started_at,
                        finished_at=datetime.now(UTC),
                        commands=[],
                        reason="SIGINT received",
                    )

                # Resolve and execute command
                resolved_cmd, resolved_timeout = self._resolve_command(
                    cmd_ref, validation_config
                )
                if resolved_cmd is None:
                    # Config error: command ref not found. Cannot be fixed by retrying.
                    logger.error(
                        "session_end remediation: command ref '%s' not found for %s",
                        cmd_ref.ref,
                        issue_id,
                    )
                    return SessionEndResult(
                        status="fail",
                        started_at=started_at,
                        finished_at=datetime.now(UTC),
                        commands=list(command_outcomes),
                        reason="command_ref_not_found",
                    )

                # Execute command (command_runner guaranteed non-None from check above)
                # Shield command execution to complete current command on SIGINT
                cmd_task = asyncio.create_task(
                    self._command_runner.run_async(
                        resolved_cmd,
                        timeout=resolved_timeout,
                        shell=True,
                        cwd=self._repo_path,
                    )
                )
                try:
                    result = await asyncio.shield(cmd_task)
                except asyncio.CancelledError:
                    # Check if SIGINT: complete command. Otherwise re-raise.
                    if interrupt_event and interrupt_event.is_set():
                        result = await cmd_task
                        # Log outcome but don't include in result per spec R10
                        _ = self._to_command_outcome(cmd_ref.ref, result)
                        logger.info(
                            "session_end retry interrupted during command %s for %s",
                            cmd_ref.ref,
                            issue_id,
                        )
                        # Per spec R10: discard partial results on interrupt
                        return SessionEndResult(
                            status="interrupted",
                            started_at=started_at,
                            finished_at=datetime.now(UTC),
                            commands=[],
                            reason="SIGINT received",
                        )
                    # Not SIGINT: cancel and re-raise
                    cmd_task.cancel()
                    raise
                outcome = self._to_command_outcome(cmd_ref.ref, result)
                retry_outcomes.append(outcome)

                # Check for interrupt after command
                if interrupt_event and interrupt_event.is_set():
                    logger.info(
                        "session_end retry interrupted after command %s for %s",
                        cmd_ref.ref,
                        issue_id,
                    )
                    # Per spec R10: discard partial results on interrupt
                    return SessionEndResult(
                        status="interrupted",
                        started_at=started_at,
                        finished_at=datetime.now(UTC),
                        commands=[],
                        reason="SIGINT received",
                    )

                if not result.ok:
                    retry_passed = False
                    # Update failure output for next fixer attempt
                    failure_output = self._build_failure_output(retry_outcomes)
                    break

            # Update command_outcomes with retry outcomes
            command_outcomes.extend(retry_outcomes)

            if retry_passed:
                logger.info(
                    "session_end remediation succeeded on attempt %d for %s",
                    retry_num,
                    issue_id,
                )
                return SessionEndResult(
                    status="pass",
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                    commands=list(command_outcomes),
                )

        # Exhausted all retries
        logger.info(
            "session_end remediation exhausted after %d retries for %s",
            max_retries,
            issue_id,
        )
        return SessionEndResult(
            status="fail",
            started_at=started_at,
            finished_at=datetime.now(UTC),
            commands=list(command_outcomes),
            reason="max_retries_exhausted",
        )

    def _build_failure_output(self, command_outcomes: list[CommandOutcome]) -> str:
        """Build failure output string for fixer from command outcomes.

        Args:
            command_outcomes: List of command outcomes, last one is the failed command.

        Returns:
            Human-readable description of what failed.
        """
        if not command_outcomes:
            return "session_end validation failed (no command results)"

        last_outcome = command_outcomes[-1]
        parts = [f"Command failed: {last_outcome.ref}"]

        # Add error message if available (contains stderr for failed commands)
        if last_outcome.error_message:
            error_msg = last_outcome.error_message
            # Truncate long output
            if len(error_msg) > 1000:
                error_msg = error_msg[-1000:]
                parts.append(f"Error (truncated):\n{error_msg}")
            else:
                parts.append(f"Error:\n{error_msg}")

        return "\n".join(parts)

    async def _run_session_end_code_review(
        self,
        issue_id: str,
        session_end_config: SessionEndTriggerConfig,
        commands_passed: bool,
        interrupt_event: asyncio.Event | None,
    ) -> CodeReviewResult | None:
        """Run code review for session_end if configured.

        Per spec R11: Uses base_sha..HEAD range for the issue.

        Args:
            issue_id: The issue ID.
            session_end_config: The session_end trigger config.
            commands_passed: Whether all commands passed.
            interrupt_event: Event to check for interruption.

        Returns:
            CodeReviewResult or None if code_review not configured/enabled.
        """
        from src.domain.validation.config import FailureMode, TriggerType

        code_review_config = session_end_config.code_review
        if code_review_config is None or not code_review_config.enabled:
            return None

        # Don't run code_review if abort mode and commands failed
        if not commands_passed and session_end_config.failure_mode == FailureMode.ABORT:
            return CodeReviewResult(ran=False, passed=None, findings=[])

        if self._cumulative_review_runner is None:
            logger.warning(
                "session_end code_review enabled but no cumulative_review_runner"
            )
            return CodeReviewResult(ran=False, passed=None, findings=[])

        run_metadata = self._context.get_run_metadata()
        if run_metadata is None:
            logger.warning("session_end code_review: no run_metadata available")
            return CodeReviewResult(ran=False, passed=None, findings=[])

        # Get base_sha for the issue (per spec R11)
        base_sha = self._context.get_base_sha(issue_id)
        if base_sha is None:
            logger.warning(
                "session_end code_review: no base_sha for issue %s",
                issue_id,
            )
            return CodeReviewResult(ran=False, passed=None, findings=[])

        # Create a wrapped event for the review runner
        review_interrupt = interrupt_event or asyncio.Event()

        try:
            result = await self._cumulative_review_runner.run_review(
                trigger_type=TriggerType.SESSION_END,
                config=code_review_config,
                run_metadata=run_metadata,
                repo_path=self._repo_path,
                interrupt_event=review_interrupt,
                issue_id=issue_id,
                baseline_override=base_sha,  # Per R11: use base_sha..HEAD range
            )

            # Convert findings to dict format for SessionEndResult
            findings = [
                {
                    "file": f.file,
                    "line_start": f.line_start,
                    "line_end": f.line_end,
                    "priority": f.priority,
                    "title": f.title,
                    "body": f.body,
                }
                for f in result.findings
            ]

            # passed=True means code passed review with no issues
            # passed=False means review found issues (findings) that may need remediation
            # "skipped" status (e.g., empty diff) is treated as passed since there's nothing to review
            passed = result.status in ("success", "skipped") and len(findings) == 0
            return CodeReviewResult(ran=True, passed=passed, findings=findings)

        except Exception as e:
            logger.error("session_end code_review failed: %s", e)
            return CodeReviewResult(ran=False, passed=None, findings=[])


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


@dataclass
class SessionRunnerAdapters:
    """Bundle of protocol implementations for AgentSessionRunner.

    This class provides IGateRunner, IReviewRunner, and ISessionLifecycle
    implementations by adapting a SessionCallbackFactory instance. Use this
    to construct an AgentSessionRunner with protocol interfaces instead of
    the deprecated SessionCallbacks dataclass.

    Usage:
        factory = SessionCallbackFactory(...)
        adapters = factory.build_adapters(issue_id)
        runner = AgentSessionRunner(
            config=config,
            sdk_client_factory=sdk_factory,
            gate_runner=adapters.gate_runner,
            review_runner=adapters.review_runner,
            session_lifecycle=adapters.session_lifecycle,
        )

    Attributes:
        gate_runner: IGateRunner protocol implementation.
        review_runner: IReviewRunner protocol implementation.
        session_lifecycle: ISessionLifecycle protocol implementation.
    """

    gate_runner: IGateRunner
    review_runner: IReviewRunner
    session_lifecycle: ISessionLifecycle


class _GateRunnerAdapter:
    """IGateRunner implementation wrapping SessionCallbackFactory methods."""

    def __init__(
        self,
        factory: SessionCallbackFactory,
        issue_id: str,
    ) -> None:
        self._factory = factory
        self._issue_id = issue_id

    async def run_gate_check(
        self,
        issue_id: str,
        log_path: Path,
        retry_state: RetryStateProtocol,
    ) -> tuple[GateOutcomeProtocol, int]:
        """Run quality gate check."""
        result, offset = await self._factory._gate_async_runner.run_gate_async(
            issue_id,
            log_path,
            retry_state,  # type: ignore[arg-type] # Protocol → concrete type
            self._factory._context.interrupt_event_getter(),
        )
        return result, offset  # type: ignore[return-value] # concrete → Protocol

    async def run_session_end_check(
        self,
        issue_id: str,
        log_path: Path,
        retry_state: SessionEndRetryState,
    ) -> SessionEndResult:
        """Run session-end validation check."""
        return await self._factory._execute_session_end(
            issue_id=issue_id,
            log_path=log_path,
            retry_state=retry_state,
        )


class _ReviewRunnerAdapter:
    """IReviewRunner implementation wrapping SessionCallbackFactory methods."""

    def __init__(
        self,
        factory: SessionCallbackFactory,
        issue_id: str,
    ) -> None:
        self._factory = factory
        self._issue_id = issue_id

    async def run_review(
        self,
        issue_id: str,
        description: str | None,
        session_id: str | None,
        retry_state: RetryStateProtocol,
        author_context: str | None,
        previous_findings: Sequence[ReviewIssueProtocol] | None,
        session_end_result: SessionEndResult | None,
    ) -> ReviewOutcomeProtocol:
        """Run code review."""
        from src.core.models import ReviewInput
        from src.infra.git_utils import get_issue_commits_async

        self._factory._review_runner.config.capture_session_log = (
            self._factory._is_verbose()
        )
        commit_shas = await get_issue_commits_async(
            self._factory._repo_path,
            issue_id,
        )
        review_input = ReviewInput(
            issue_id=issue_id,
            repo_path=self._factory._repo_path,
            issue_description=description,
            commit_shas=commit_shas,
            claude_session_id=session_id,
            author_context=author_context,
            previous_findings=previous_findings,
            session_end_result=session_end_result,
        )
        output = await self._factory._review_runner.run_review(
            review_input, self._factory._context.interrupt_event_getter()
        )
        if output.session_log_path:
            self._factory._context.on_review_log_path(issue_id, output.session_log_path)

        # Propagate interrupted flag
        result = output.result
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

    def check_no_progress(
        self,
        log_path: Path,
        log_offset: int,
        prev_commit: str | None,
        curr_commit: str | None,
    ) -> bool:
        """Check if no progress was made since the last attempt."""
        from src.pipeline.review_runner import NoProgressInput

        no_progress_input = NoProgressInput(
            log_path=log_path,
            log_offset=log_offset,
            previous_commit_hash=prev_commit,
            current_commit_hash=curr_commit,
            spec=self._factory._get_per_session_spec(),
        )
        return self._factory._review_runner.check_no_progress(no_progress_input)


class _SessionLifecycleAdapter:
    """ISessionLifecycle implementation wrapping SessionCallbackFactory methods."""

    def __init__(
        self,
        factory: SessionCallbackFactory,
        issue_id: str,
        on_abort: Callable[[str], None] | None = None,
    ) -> None:
        self._factory = factory
        self._issue_id = issue_id
        self._on_abort = on_abort

    def get_log_path(self, session_id: str) -> Path:
        """Get the log file path for a session."""
        log_path = self._factory._context.log_provider_getter().get_log_path(
            self._factory._repo_path, session_id
        )
        self._factory._context.on_session_log_path(self._issue_id, log_path)
        return log_path

    def get_log_offset(self, log_path: Path, start_offset: int) -> int:
        """Get the current byte offset at the end of a log file."""
        return self._factory._context.evidence_check_getter().get_log_end_offset(
            log_path, start_offset
        )

    def on_abort(self, reason: str) -> None:
        """Handle session abort."""
        if self._on_abort is not None:
            self._on_abort(reason)

    def get_abort_event(self) -> asyncio.Event | None:
        """Get the abort event for run abort detection."""
        return self._factory._context.abort_event_getter()

    def on_tool_use(
        self, agent_id: str, tool_name: str, args: dict[str, Any] | None
    ) -> None:
        """Handle tool use event from SDK."""
        self._factory._get_event_sink().on_tool_use(agent_id, tool_name, arguments=args)

    def on_agent_text(self, agent_id: str, text: str) -> None:
        """Handle text output event from SDK."""
        self._factory._get_event_sink().on_agent_text(agent_id, text)

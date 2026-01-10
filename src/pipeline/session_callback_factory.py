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

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from src.pipeline.agent_session_runner import SessionCallbacks
from src.core.session_end_result import CodeReviewResult, SessionEndResult

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from src.core.protocols import (
        CommandResultProtocol,
        LogProvider,
        MalaEventSink,
        ReviewIssueProtocol,
    )
    from src.domain.lifecycle import (
        GateOutcome,
        RetryState,
        ReviewIssue,
        ReviewOutcome,
    )
    from src.domain.validation.config import (
        SessionEndTriggerConfig,
        TriggerCommandRef,
        ValidationConfig,
    )
    from src.domain.validation.spec import ValidationSpec
    from src.infra.io.log_output.run_metadata import RunMetadata
    from src.pipeline.cumulative_review_runner import CumulativeReviewRunner
    from src.core.protocols import CommandRunnerPort
    from src.pipeline.fixer_interface import FixerInterface
    from src.pipeline.review_runner import ReviewRunner
    from src.core.session_end_result import SessionEndRetryState

logger = logging.getLogger(__name__)


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
        get_validation_config: Callable[[], ValidationConfig | None] | None = None,
        command_runner: CommandRunnerPort | None = None,
        fixer_interface: FixerInterface | None = None,
        cumulative_review_runner: CumulativeReviewRunner | None = None,
        get_run_metadata: Callable[[], RunMetadata | None] | None = None,
        get_base_sha: Callable[[str], str | None] | None = None,
        on_abort: Callable[[str], None] | None = None,
        session_end_timeout: float = 600.0,
        get_abort_event: Callable[[], asyncio.Event | None] | None = None,
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
            get_validation_config: Callable to get validation config (late-bound).
            command_runner: Command runner for executing session_end commands.
            fixer_interface: Interface for running fixer agents during remediation.
            cumulative_review_runner: Runner for session_end code review.
            get_run_metadata: Callable to get run metadata (late-bound).
            get_base_sha: Callable to get base_sha for an issue (late-bound).
            on_abort: Callback when session_end abort mode triggers run abort.
            session_end_timeout: Overall timeout for session_end in seconds.
            get_abort_event: Callable to get the abort event (late-bound).

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
        self._get_validation_config = get_validation_config or (lambda: None)
        self._command_runner = command_runner
        self._fixer_interface = fixer_interface
        self._cumulative_review_runner = cumulative_review_runner
        self._get_run_metadata = get_run_metadata or (lambda: None)
        self._get_base_sha = get_base_sha or (lambda _: None)
        self._on_abort = on_abort
        self._session_end_timeout = session_end_timeout
        self._get_abort_event = get_abort_event or (lambda: None)

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
            author_context: str | None,
            previous_findings: Sequence[ReviewIssueProtocol] | None,
            session_end_result: SessionEndResult | None,
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
                author_context=author_context,
                previous_findings=previous_findings,
                session_end_result=session_end_result,
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

        async def on_session_end_check(
            issue_id: str, log_path: Path, retry_state: SessionEndRetryState
        ) -> SessionEndResult:
            """Execute session_end trigger: commands, timeout, and code_review.

            Implements per spec R9-R11:
            - R9: Commands execute sequentially; failure_mode (abort, continue)
            - R10: Overall asyncio.timeout wrapper; on timeout/interrupt return
              appropriate SessionEndResult with empty commands
            - R11: code_review uses base_sha..HEAD range

            Args:
                issue_id: The issue ID for context.
                log_path: Path to session log (for logging).
                retry_state: Retry state for remediation tracking.

            Returns:
                SessionEndResult with execution outcome.
            """
            return await self._execute_session_end(
                issue_id=issue_id,
                log_path=log_path,
                retry_state=retry_state,
            )

        return SessionCallbacks(
            on_gate_check=on_gate_check,
            on_review_check=on_review_check,
            on_review_no_progress=on_review_no_progress,
            get_log_path=get_log_path,
            get_log_offset=get_log_offset,
            on_abort=on_abort,
            on_tool_use=on_tool_use,
            on_agent_text=on_agent_text,
            on_session_end_check=on_session_end_check,
            get_abort_event=self._get_abort_event,
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
        interrupt_event = self._get_interrupt_event()
        command_results: list[CommandResultProtocol] = []

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
                        # Return with completed commands from previous iterations
                        return SessionEndResult(
                            status="interrupted",
                            started_at=started_at,
                            finished_at=datetime.now(UTC),
                            commands=list(command_results),
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
                                # SIGINT: complete the command, return with result
                                result = await cmd_task
                                command_results.append(result)
                                logger.info(
                                    "session_end interrupted during command %s for %s",
                                    cmd_ref.ref,
                                    issue_id,
                                )
                                return SessionEndResult(
                                    status="interrupted",
                                    started_at=started_at,
                                    finished_at=datetime.now(UTC),
                                    commands=list(command_results),
                                    reason="SIGINT received",
                                )
                            # Not SIGINT (likely timeout): cancel the inner task and re-raise
                            # to let timeout handler return with empty commands
                            cmd_task.cancel()
                            raise
                        command_results.append(result)

                        # Check for interrupt after command completes
                        if interrupt_event and interrupt_event.is_set():
                            logger.info(
                                "session_end interrupted after command %s for %s",
                                cmd_ref.ref,
                                issue_id,
                            )
                            return SessionEndResult(
                                status="interrupted",
                                started_at=started_at,
                                finished_at=datetime.now(UTC),
                                commands=list(command_results),
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
                        abort_event = self._get_abort_event()
                        if abort_event is not None:
                            abort_event.set()
                        if self._on_abort:
                            self._on_abort(
                                f"session_end validation failed for {issue_id}"
                            )
                        return SessionEndResult(
                            status="fail",
                            started_at=started_at,
                            finished_at=datetime.now(UTC),
                            commands=list(command_results),
                            reason="command_failed",
                        )
                    elif failure_mode == FailureMode.REMEDIATE:
                        # Remediation loop: run fixer and retry validation
                        remediation_result = await self._run_remediation_loop(
                            issue_id=issue_id,
                            session_end_config=session_end_config,
                            validation_config=validation_config,
                            command_results=command_results,
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
                    commands=list(command_results),
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
            return SessionEndResult(
                status="interrupted",
                started_at=started_at,
                finished_at=datetime.now(UTC),
                commands=list(command_results),
                reason="cancelled",
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
        # Build base pool from global_validation_commands with fallback to commands
        base_pool: dict[str, tuple[str, int | None]] = {}

        # Standard commands from global_validation_commands (with fallback)
        commands_config = validation_config.commands
        global_config = validation_config.global_validation_commands

        for name in ["test", "lint", "format", "typecheck", "e2e", "setup", "build"]:
            # Try global first, then fallback to base
            cmd_config = None
            if global_config:
                cmd_config = getattr(global_config, name, None)
            if cmd_config is None and commands_config:
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

        # Add custom commands from global_validation_commands.custom_commands
        # (global overrides repo-level)
        if global_config and global_config.custom_commands:
            for name, custom_config in global_config.custom_commands.items():
                if custom_config is not None:
                    base_pool[name] = (
                        custom_config.command,
                        getattr(custom_config, "timeout", None),
                    )

        # Add custom commands from global_custom_commands (deprecated, but support)
        if validation_config.global_custom_commands:
            for name, custom_config in validation_config.global_custom_commands.items():
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
        command_results: list[CommandResultProtocol],
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
            command_results: Command results from initial attempt (will be updated).
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

        # Build failure output from last command result
        failure_output = self._build_failure_output(command_results)

        # Remediation loop: run fixer and retry validation
        for retry_num in range(1, max_retries + 1):
            # Check for interrupt before fixer
            if interrupt_event and interrupt_event.is_set():
                logger.info(
                    "session_end remediation interrupted before fixer for %s",
                    issue_id,
                )
                return SessionEndResult(
                    status="interrupted",
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                    commands=list(command_results),
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
                return SessionEndResult(
                    status="interrupted",
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                    commands=list(command_results),
                    reason="fixer_interrupted",
                )

            # Check for interrupt after fixer
            if interrupt_event and interrupt_event.is_set():
                logger.info(
                    "session_end remediation interrupted after fixer for %s",
                    issue_id,
                )
                return SessionEndResult(
                    status="interrupted",
                    started_at=started_at,
                    finished_at=datetime.now(UTC),
                    commands=list(command_results),
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
                    commands=list(command_results),
                    reason="command_runner_not_configured",
                )

            retry_passed = True
            retry_results: list[CommandResultProtocol] = []
            for cmd_ref in session_end_config.commands:
                # Check for interrupt before each command
                if interrupt_event and interrupt_event.is_set():
                    logger.info(
                        "session_end retry interrupted before command %s for %s",
                        cmd_ref.ref,
                        issue_id,
                    )
                    return SessionEndResult(
                        status="interrupted",
                        started_at=started_at,
                        finished_at=datetime.now(UTC),
                        commands=list(command_results) + retry_results,
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
                        commands=list(command_results),
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
                        retry_results.append(result)
                        logger.info(
                            "session_end retry interrupted during command %s for %s",
                            cmd_ref.ref,
                            issue_id,
                        )
                        return SessionEndResult(
                            status="interrupted",
                            started_at=started_at,
                            finished_at=datetime.now(UTC),
                            commands=list(command_results) + retry_results,
                            reason="SIGINT received",
                        )
                    # Not SIGINT: cancel and re-raise
                    cmd_task.cancel()
                    raise
                retry_results.append(result)

                # Check for interrupt after command
                if interrupt_event and interrupt_event.is_set():
                    logger.info(
                        "session_end retry interrupted after command %s for %s",
                        cmd_ref.ref,
                        issue_id,
                    )
                    return SessionEndResult(
                        status="interrupted",
                        started_at=started_at,
                        finished_at=datetime.now(UTC),
                        commands=list(command_results) + retry_results,
                        reason="SIGINT received",
                    )

                if not result.ok:
                    retry_passed = False
                    # Update failure output for next fixer attempt
                    failure_output = self._build_failure_output(retry_results)
                    break

            # Update command_results with retry results
            command_results.extend(retry_results)

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
                    commands=list(command_results),
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
            commands=list(command_results),
            reason="max_retries_exhausted",
        )

    def _build_failure_output(
        self, command_results: list[CommandResultProtocol]
    ) -> str:
        """Build failure output string for fixer from command results.

        Args:
            command_results: List of command results, last one is the failed command.

        Returns:
            Human-readable description of what failed.
        """
        if not command_results:
            return "session_end validation failed (no command results)"

        last_result = command_results[-1]
        parts = [f"Command failed: {last_result.command}"]

        if hasattr(last_result, "returncode") and last_result.returncode is not None:
            parts.append(f"Exit code: {last_result.returncode}")

        # Add stderr if available
        stderr = getattr(last_result, "stderr", "") or ""
        if stderr:
            # Truncate long output
            if len(stderr) > 1000:
                stderr = stderr[-1000:]
                parts.append(f"stderr (truncated):\n{stderr}")
            else:
                parts.append(f"stderr:\n{stderr}")

        # Add stdout if stderr is empty
        stdout = getattr(last_result, "stdout", "") or ""
        if not stderr and stdout:
            if len(stdout) > 1000:
                stdout = stdout[-1000:]
                parts.append(f"stdout (truncated):\n{stdout}")
            else:
                parts.append(f"stdout:\n{stdout}")

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

        run_metadata = self._get_run_metadata()
        if run_metadata is None:
            logger.warning("session_end code_review: no run_metadata available")
            return CodeReviewResult(ran=False, passed=None, findings=[])

        # Get base_sha for the issue (per spec R11)
        base_sha = self._get_base_sha(issue_id)
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

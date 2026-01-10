"""MalaOrchestrator: Orchestrates parallel issue processing using Claude Agent SDK."""

from __future__ import annotations

import asyncio
import logging
import signal
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.domain.validation.spec import (
    ValidationScope,
    build_validation_spec,
    extract_lint_tools_from_spec,
)
from src.domain.prompts import (
    build_prompt_validation_commands,
    format_implementer_prompt,
    load_prompts,
)

from src.domain.prompts import build_custom_commands_section
from src.pipeline.review_formatter import format_review_issues
from src.infra.git_utils import get_git_branch_async, get_git_commit_async
from src.infra.io.log_output.run_metadata import (
    lookup_prior_session_info,
    remove_run_marker,
    write_run_marker,
)
from src.infra.tools.env import (
    EnvConfig,
    PROMPTS_DIR,
    encode_repo_path,
    get_lock_dir,
    get_runs_dir,
)
from src.domain.deadlock import DeadlockMonitor
from src.orchestration.orchestrator_state import OrchestratorState
from src.orchestration.deadlock_handler import DeadlockHandler, DeadlockHandlerCallbacks
from src.infra.tools.command_runner import CommandRunner
from src.infra.tools.locking import (
    LockManager,
    cleanup_agent_locks,
    release_run_locks,
)
from src.infra.sdk_adapter import SDKClientFactory
from src.pipeline.agent_session_runner import (
    AgentSessionInput,
    AgentSessionRunner,
)
from src.pipeline.issue_finalizer import (
    IssueFinalizeInput,
)
from src.domain.validation.config import FireOn, TriggerType
from src.pipeline.run_coordinator import (
    GlobalValidationInput,
)
from src.orchestration.orchestration_wiring import (
    FinalizerCallbackRefs,
    EpicCallbackRefs,
    build_gate_runner,
    build_review_runner,
    build_run_coordinator,
    build_issue_coordinator,
    build_finalizer_callbacks,
    build_epic_callbacks,
    build_session_callback_factory,
    build_session_config,
    create_mcp_server_factory,
)
from src.orchestration.types import (
    IssueFilterConfig,
    PipelineConfig,
    RuntimeDeps,
)
from src.pipeline.issue_result import IssueResult
from src.orchestration.review_tracking import create_review_tracking_issues
from src.orchestration.run_config import build_event_run_config, build_run_metadata

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from src.pipeline.issue_execution_coordinator import AbortResult
    from src.core.protocols import (
        CodeReviewer,
        EpicVerifierProtocol,
        GateChecker,
        IssueProvider,
        LogProvider,
    )
    from src.domain.prompts import PromptProvider
    from src.domain.validation.config import PromptValidationCommands
    from src.infra.io.config import MalaConfig
    from src.core.protocols import MalaEventSink
    from src.infra.io.log_output.run_metadata import RunMetadata
    from src.infra.telemetry import TelemetryProvider
    from src.pipeline.agent_session_runner import SessionCallbacks
    from src.pipeline.epic_verification_coordinator import EpicVerificationCoordinator
    from src.pipeline.issue_finalizer import IssueFinalizer

    from src.core.models import PeriodicValidationConfig, RunResult, WatchConfig

    from .types import OrchestratorConfig, _DerivedConfig


# Version (from package metadata)
from importlib.metadata import version as pkg_version

__version__ = pkg_version("mala")

logger = logging.getLogger(__name__)

# Bounded wait for log file (seconds) - used by AgentSessionRunner
# Re-exported here for backwards compatibility with tests.
# 60s allows time for Claude SDK to flush logs, especially under load.
LOG_FILE_WAIT_TIMEOUT = 60
LOG_FILE_POLL_INTERVAL = 0.5

# SIGINT escalation timing constant (5s window for counting consecutive Ctrl-C)
ESCALATION_WINDOW_SECONDS = 5.0


@dataclass(frozen=True)
class StoredReviewIssue:
    """Typed adapter for review issues loaded from metadata."""

    file: str
    line_start: int
    line_end: int
    priority: int | None
    title: str
    body: str
    reviewer: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StoredReviewIssue:
        def to_int(val: object, default: int) -> int:
            if val is None:
                return default
            try:
                return int(val)  # type: ignore[arg-type]
            except (ValueError, TypeError):
                return default

        line_start = to_int(d.get("line_start"), 0)
        return cls(
            file=str(d.get("file", "unknown")),
            line_start=line_start,
            line_end=to_int(d.get("line_end"), line_start),
            priority=d.get("priority"),
            title=str(d.get("title", "Unknown issue")),
            body=str(d.get("body", "")),
            reviewer=str(d.get("reviewer", "unknown")),
        )


def _build_resume_prompt(
    review_issues: list[dict[str, Any]],
    prompts: PromptProvider,
    validation_commands: PromptValidationCommands,
    issue_id: str,
    max_review_retries: int,
    repo_path: Path,
    prior_run_id: str,
) -> str | None:
    """Build a resume prompt with review feedback.

    Args:
        review_issues: List of review issue dicts from prior session.
        prompts: Prompt templates provider.
        validation_commands: Validation commands for the prompt.
        issue_id: The issue ID being implemented.
        max_review_retries: Maximum review retry attempts.
        repo_path: Repository path for path relativization.
        prior_run_id: Run ID of the prior session for logging.

    Returns:
        Formatted review followup prompt, or None if no issues.
    """
    if not review_issues:
        return None

    issues = [StoredReviewIssue.from_dict(d) for d in review_issues]

    logger.info(
        "Using review_followup for resume: %d issues from prior run %s",
        len(issues),
        prior_run_id,
    )

    review_issues_text = format_review_issues(issues, base_path=repo_path)

    return prompts.review_followup_prompt.format(
        attempt=1,
        max_attempts=max_review_retries,
        review_issues=review_issues_text,
        issue_id=issue_id,
        lint_command=validation_commands.lint,
        format_command=validation_commands.format,
        typecheck_command=validation_commands.typecheck,
        test_command=validation_commands.test,
        custom_commands_section=build_custom_commands_section(
            validation_commands.custom_commands
        ),
    )


class MalaOrchestrator:
    """Orchestrates parallel issue processing using Claude Agent SDK.

    Use create_orchestrator() factory function to instantiate this class.
    The factory provides clean separation of concerns and easier testing.
    """

    # Type annotations for attributes set during initialization
    _max_issues: int | None
    _deadlock_handler: DeadlockHandler

    def __init__(
        self,
        *,
        _config: OrchestratorConfig,
        _mala_config: MalaConfig,
        _derived: _DerivedConfig,
        _issue_provider: IssueProvider,
        _code_reviewer: CodeReviewer,
        _gate_checker: GateChecker,
        _log_provider: LogProvider,
        _telemetry_provider: TelemetryProvider,
        _event_sink: MalaEventSink,
        _epic_verifier: EpicVerifierProtocol | None = None,
        runs_dir: Path | None = None,
        lock_releaser: Callable[[list[str]], int] | None = None,
    ):
        self._init_from_factory(
            _config,
            _mala_config,
            _derived,
            _issue_provider,
            _code_reviewer,
            _gate_checker,
            _log_provider,
            _telemetry_provider,
            _event_sink,
            _epic_verifier,
            runs_dir=runs_dir,
            lock_releaser=lock_releaser,
        )

    def _init_from_factory(
        self,
        orch_config: OrchestratorConfig,
        mala_config: MalaConfig,
        derived: _DerivedConfig,
        issue_provider: IssueProvider,
        code_reviewer: CodeReviewer,
        gate_checker: GateChecker,
        log_provider: LogProvider,
        telemetry_provider: TelemetryProvider,
        event_sink: MalaEventSink,
        epic_verifier: EpicVerifierProtocol | None,
        *,
        runs_dir: Path | None = None,
        lock_releaser: Callable[[list[str]], int] | None = None,
    ) -> None:
        """Initialize from factory-provided config and dependencies."""
        # Store optional DI parameters, defaulting to module functions when None
        self._runs_dir = runs_dir
        self._lock_releaser = lock_releaser
        self._mala_config = mala_config
        self.repo_path = orch_config.repo_path.resolve()
        self.max_agents = orch_config.max_agents
        self.timeout_seconds = derived.timeout_seconds
        self.max_issues = orch_config.max_issues
        self.epic_id = orch_config.epic_id
        self.only_ids = orch_config.only_ids
        self.max_gate_retries = orch_config.max_gate_retries
        self.max_review_retries = orch_config.max_review_retries
        self.disable_validations = orch_config.disable_validations
        self._disabled_validations = derived.disabled_validations
        self._validation_config = derived.validation_config
        self._validation_config_missing = derived.validation_config_missing
        self.include_wip = orch_config.include_wip
        self.strict_resume = orch_config.strict_resume
        self.focus = orch_config.focus
        self.orphans_only = orch_config.orphans_only
        self.order_preference = orch_config.order_preference
        self.cli_args = orch_config.cli_args
        self.epic_override_ids = orch_config.epic_override_ids
        self.context_restart_threshold = orch_config.context_restart_threshold
        self.context_limit = orch_config.context_limit
        self.review_disabled_reason = derived.review_disabled_reason
        self._init_runtime_state()
        self.log_provider = log_provider
        self.evidence_check = gate_checker
        self.event_sink = event_sink
        self.beads = issue_provider
        self.epic_verifier = epic_verifier
        self.code_reviewer = code_reviewer
        self.telemetry_provider = telemetry_provider
        self._sdk_client_factory = SDKClientFactory()
        self._init_pipeline_runners()

    def _init_runtime_state(self) -> None:
        """Initialize runtime state during __init__.

        Called once in __init__. OrchestratorState is reset at the start of
        each run() call to ensure clean per-run state.

        Note: active_tasks, failed_issues, abort_run, and abort_reason are
        delegated to issue_coordinator to maintain a single source of truth.
        See the corresponding properties.

        Note: verified_epics and epics_being_verified are delegated to
        epic_verification_coordinator.

        Note: per_session_spec and last_gate_results are delegated to
        async_gate_runner.
        """
        self._state = OrchestratorState()
        self._lint_tools: frozenset[str] | None = None
        self._exit_code: int = 0  # Exit code from last run (0=success, 130=interrupted)
        # SIGINT escalation state (initialized here, reset per-run via _reset_sigint_state)
        self._sigint_count: int = 0
        self._sigint_last_at: float = 0.0
        self._drain_mode_active: bool = False
        self._abort_mode_active: bool = False
        self._abort_exit_code: int = 130
        self._validation_failed: bool = False
        self._shutdown_requested: bool = False
        self._run_task: asyncio.Task[Any] | None = None
        self._interrupt_event: asyncio.Event | None = None
        # Trigger state for periodic validation (see T009-T011)
        self._non_epic_completed_count: int = 0
        self._prompt_validation_commands = build_prompt_validation_commands(
            self.repo_path,
            validation_config=self._validation_config,
            config_missing=self._validation_config_missing,
        )
        self._prompts: PromptProvider = load_prompts(PROMPTS_DIR)
        # Deadlock detection is always enabled
        self.deadlock_monitor: DeadlockMonitor = DeadlockMonitor()
        logger.info("Deadlock detection enabled")

    def _init_pipeline_runners(self) -> None:
        """Initialize pipeline runner components using wiring functions."""
        runtime = self._build_runtime_deps()
        pipeline = self._build_pipeline_config()
        filters = self._build_issue_filter_config()

        # Build core runners
        self.gate_runner, self.async_gate_runner = build_gate_runner(runtime, pipeline)
        self.review_runner = build_review_runner(runtime, pipeline)
        self.run_coordinator = build_run_coordinator(
            runtime, pipeline, self._sdk_client_factory, create_mcp_server_factory()
        )
        self.issue_coordinator = build_issue_coordinator(filters, runtime)

        # Build coordinators with callbacks (callbacks need self references)
        self.issue_finalizer = self._build_issue_finalizer()
        self.epic_verification_coordinator = self._build_epic_verification_coordinator()

        # Build session infrastructure
        self.session_callback_factory = build_session_callback_factory(
            runtime,
            pipeline,
            self.async_gate_runner,
            self.review_runner,
            lambda: self.log_provider,
            lambda: self.evidence_check,
            on_session_log_path=self._on_session_log_path,
            on_review_log_path=self._on_review_log_path,
            interrupt_event_getter=lambda: self._interrupt_event,
        )
        self._session_config = build_session_config(
            pipeline,
            review_enabled=self._is_review_enabled(),
            mcp_server_factory=create_mcp_server_factory(),
            strict_resume=self.strict_resume,
        )

        # Wire deadlock handler now that all dependencies are available
        self._deadlock_handler = self._build_deadlock_handler()
        self.deadlock_monitor.on_deadlock = (
            lambda info: self._deadlock_handler.handle_deadlock(
                info, self._state, self.active_tasks
            )
        )

    def _build_runtime_deps(self) -> RuntimeDeps:
        """Build RuntimeDeps from orchestrator state."""
        return RuntimeDeps(
            evidence_check=self.evidence_check,
            code_reviewer=self.code_reviewer,
            beads=self.beads,
            event_sink=self.event_sink,
            command_runner=CommandRunner(cwd=self.repo_path),
            env_config=EnvConfig(),
            lock_manager=LockManager(),
            mala_config=self._mala_config,
        )

    def _build_pipeline_config(self) -> PipelineConfig:
        """Build PipelineConfig from orchestrator state."""
        return PipelineConfig(
            repo_path=self.repo_path,
            timeout_seconds=self.timeout_seconds,
            max_gate_retries=self.max_gate_retries,
            max_review_retries=self.max_review_retries,
            disabled_validations=set(self._disabled_validations)
            if self._disabled_validations
            else None,
            context_restart_threshold=self.context_restart_threshold,
            context_limit=self.context_limit,
            prompts=self._prompts,
            prompt_validation_commands=self._prompt_validation_commands,
            validation_config=self._validation_config,
            validation_config_missing=self._validation_config_missing,
            deadlock_monitor=self.deadlock_monitor,
        )

    def _build_issue_filter_config(self) -> IssueFilterConfig:
        """Build IssueFilterConfig from orchestrator state."""
        return IssueFilterConfig(
            max_agents=self.max_agents,
            max_issues=self._max_issues,
            epic_id=self.epic_id,
            only_ids=self.only_ids,
            include_wip=self.include_wip,
            focus=self.focus,
            orphans_only=self.orphans_only,
            epic_override_ids=self.epic_override_ids,
            order_preference=self.order_preference,
        )

    def _build_issue_finalizer(self) -> IssueFinalizer:
        """Build IssueFinalizer with callbacks."""
        from src.pipeline.issue_finalizer import IssueFinalizer, IssueFinalizeConfig

        config = IssueFinalizeConfig(
            track_review_issues=self._mala_config.track_review_issues,
        )
        callbacks = build_finalizer_callbacks(
            FinalizerCallbackRefs(
                close_issue=lambda issue_id: self.beads.close_async(issue_id),
                mark_needs_followup=lambda issue_id, summary, log_path: (
                    self.beads.mark_needs_followup_async(
                        issue_id, summary, log_path=log_path
                    )
                ),
                on_issue_closed=lambda agent_id, issue_id: (
                    self.event_sink.on_issue_closed(agent_id, issue_id)
                ),
                on_issue_completed=lambda agent_id,
                issue_id,
                success,
                duration,
                summary: (
                    self.event_sink.on_issue_completed(
                        agent_id, issue_id, success, duration, summary
                    )
                ),
                trigger_epic_closure=lambda issue_id, run_metadata: (
                    self.epic_verification_coordinator.check_epic_closure(
                        issue_id, run_metadata, interrupt_event=self._interrupt_event
                    )
                ),
                create_tracking_issues=self._create_review_tracking_issues,
            )
        )
        return IssueFinalizer(
            config=config,
            callbacks=callbacks,
            evidence_check=self.evidence_check,
            per_session_spec=None,
        )

    def _build_epic_verification_coordinator(self) -> EpicVerificationCoordinator:
        """Build EpicVerificationCoordinator with callbacks."""
        from src.pipeline.epic_verification_coordinator import (
            EpicVerificationCoordinator,
            EpicVerificationConfig,
        )

        config = EpicVerificationConfig(
            max_retries=self._mala_config.max_epic_verification_retries,
        )
        callbacks = build_epic_callbacks(
            EpicCallbackRefs(
                get_parent_epic=lambda issue_id: self.beads.get_parent_epic_async(
                    issue_id
                ),
                verify_epic=lambda epic_id, human_override: (
                    self.epic_verifier.verify_and_close_epic(  # type: ignore[union-attr]
                        epic_id, human_override=human_override
                    )
                ),
                spawn_remediation=lambda issue_id, flow="implementer": self.spawn_agent(
                    issue_id, flow=flow
                ),
                finalize_remediation=lambda issue_id, result, run_metadata: (
                    self._finalize_issue_result(issue_id, result, run_metadata)
                ),
                mark_completed=lambda issue_id: self.issue_coordinator.mark_completed(
                    issue_id
                ),
                is_issue_failed=lambda issue_id: issue_id in self.failed_issues,
                close_eligible_epics=lambda: self.beads.close_eligible_epics_async(),
                on_epic_closed=lambda issue_id: self.event_sink.on_epic_closed(
                    issue_id
                ),
                on_warning=lambda msg: self.event_sink.on_warning(msg),
                has_epic_verifier=lambda: self.epic_verifier is not None,
                get_agent_id=lambda issue_id: self._state.agent_ids.get(
                    issue_id, "unknown"
                ),
                queue_trigger_validation=(
                    lambda trigger_type,
                    context: self.run_coordinator.queue_trigger_validation(
                        trigger_type, context
                    )
                ),
                get_epic_completion_trigger=lambda: (
                    self._validation_config.validation_triggers.epic_completion
                    if self._validation_config
                    and self._validation_config.validation_triggers
                    else None
                ),
            )
        )
        return EpicVerificationCoordinator(
            config=config,
            callbacks=callbacks,
            epic_override_ids=self.epic_override_ids,
        )

    def _build_deadlock_handler(self) -> DeadlockHandler:
        """Build DeadlockHandler with callbacks."""
        callbacks = DeadlockHandlerCallbacks(
            add_dependency=self.beads.add_dependency_async,
            mark_needs_followup=lambda issue_id, summary, log_path: (
                self.beads.mark_needs_followup_async(
                    issue_id, summary, log_path=log_path
                )
            ),
            reopen_issue=self.beads.reopen_issue_async,
            on_deadlock_detected=self.event_sink.on_deadlock_detected,
            on_locks_cleaned=self.event_sink.on_locks_cleaned,
            on_tasks_aborting=self.event_sink.on_tasks_aborting,
            do_cleanup_agent_locks=cleanup_agent_locks,
            unregister_agent=self.deadlock_monitor.unregister_agent,
            finalize_issue_result=self._finalize_issue_result,
            mark_completed=lambda issue_id: self.issue_coordinator.mark_completed(
                issue_id
            ),
        )
        return DeadlockHandler(callbacks=callbacks)

    async def _create_review_tracking_issues(
        self,
        issue_id: str,
        review_issues: list,
    ) -> None:
        """Create tracking issues for P2/P3 review findings."""
        # Get parent epic so review tracking issues stay in the same epic
        parent_epic_id = await self.beads.get_parent_epic_async(issue_id)
        await create_review_tracking_issues(
            self.beads,
            self.event_sink,
            issue_id,
            review_issues,
            parent_epic_id=parent_epic_id,
        )

    # Delegate state to issue_coordinator (single source of truth)
    # These properties return empty containers if accessed before coordinator init
    @property
    def active_tasks(self) -> dict[str, asyncio.Task[IssueResult]]:
        """Active agent tasks, delegated to issue_coordinator."""
        if not hasattr(self, "issue_coordinator"):
            return {}
        return self.issue_coordinator.active_tasks  # type: ignore[return-value]

    @property
    def failed_issues(self) -> set[str]:
        """Failed issue IDs, delegated to issue_coordinator."""
        if not hasattr(self, "issue_coordinator"):
            return set()
        return self.issue_coordinator.failed_issues

    @property
    def max_issues(self) -> int | None:
        """Maximum issues to process, synced with issue_coordinator."""
        if not hasattr(self, "issue_coordinator"):
            return self._max_issues if hasattr(self, "_max_issues") else None
        return self.issue_coordinator.config.max_issues

    @max_issues.setter
    def max_issues(self, value: int | None) -> None:
        """Set max_issues and sync to issue_coordinator."""
        self._max_issues = value
        if hasattr(self, "issue_coordinator"):
            self.issue_coordinator.config.max_issues = value

    @property
    def abort_run(self) -> bool:
        """Whether run should abort, delegated to issue_coordinator."""
        if not hasattr(self, "issue_coordinator"):
            return False
        return self.issue_coordinator.abort_run

    @property
    def abort_reason(self) -> str | None:
        """Abort reason, delegated to issue_coordinator."""
        if not hasattr(self, "issue_coordinator"):
            return None
        return self.issue_coordinator.abort_reason

    @property
    def exit_code(self) -> int:
        """Exit code from the last run.

        Values:
            0: Success
            1: Validation failed
            2: Invalid arguments
            3: Abort
            130: Interrupted (SIGINT)
        """
        return self._exit_code

    def _cleanup_agent_locks(self, agent_id: str) -> None:
        """Remove locks held by a specific agent (crash/timeout cleanup).

        Delegates to DeadlockHandler.cleanup_agent_locks.
        """
        self._deadlock_handler.cleanup_agent_locks(agent_id)

    def _request_abort(self, reason: str) -> None:
        """Signal that the current run should stop due to a fatal error."""
        self.issue_coordinator.request_abort(reason)

    def _is_review_enabled(self) -> bool:
        """Return whether review should run for this orchestrator instance."""
        if "review" not in self._disabled_validations:
            return True
        if (
            self.review_disabled_reason
            and self.review_runner.code_reviewer.overrides_disabled_setting()
        ):
            return True
        return False

    def _on_session_log_path(self, issue_id: str, log_path: Path) -> None:
        """Store session log path for an active session.

        Called by session callback factory when log path becomes available.
        Used for deadlock handling during active sessions.

        Args:
            issue_id: The issue ID.
            log_path: Path to the session log file.
        """
        self._state.active_session_log_paths[issue_id] = log_path

    def _on_review_log_path(self, issue_id: str, log_path: str) -> None:
        """Handle review log path notification (currently unused).

        Review log paths are now returned via IssueResult.review_log_path.
        This callback exists for symmetry but can be a no-op.

        Args:
            issue_id: The issue ID.
            log_path: Path to the review session log file.
        """
        pass  # Review log path flows through IssueResult

    def _cleanup_active_session_path(self, issue_id: str) -> None:
        """Remove stored session log path for a completed issue.

        Args:
            issue_id: The issue ID to clean up.
        """
        self._state.active_session_log_paths.pop(issue_id, None)

    async def _finalize_issue_result(
        self,
        issue_id: str,
        result: IssueResult,
        run_metadata: RunMetadata,
    ) -> None:
        """Record an issue result, update metadata, and emit logs.

        Delegates to IssueFinalizer for the core finalization logic.
        Uses stored gate result to derive metadata, avoiding duplicate
        validation parsing. Gate result includes validation_evidence
        already parsed during quality gate check.
        """
        stored_gate_result = self.async_gate_runner.get_last_gate_result(issue_id)

        # Update finalizer's per_session_spec if it has changed
        self.issue_finalizer.per_session_spec = self.async_gate_runner.per_session_spec

        # Build finalization input - use log paths from IssueResult
        finalize_input = IssueFinalizeInput(
            issue_id=issue_id,
            result=result,
            run_metadata=run_metadata,
            log_path=result.session_log_path,
            stored_gate_result=stored_gate_result,
            review_log_path=result.review_log_path,
        )

        # Track failed issues before finalize to ensure they're recorded
        # even if finalize raises (e.g., mark_needs_followup callback fails)
        # Skip deadlock victims - they should be retried after blocker completes
        if not result.success and issue_id not in self._state.deadlock_victim_issues:
            self.failed_issues.add(issue_id)

        # Delegate to finalizer, ensuring cleanup happens even if finalize raises
        try:
            await self.issue_finalizer.finalize(finalize_input)
        finally:
            # Update tracking state (active_tasks is updated by mark_completed in finalize_callback)
            self._state.completed.append(result)

            # Cleanup active session path and gate result
            self._cleanup_active_session_path(issue_id)
            self.async_gate_runner.clear_gate_result(issue_id)

            # Remove agent_id from tracking now that finalization is complete
            # (deferred from run_implementer.finally to keep it available for get_agent_id callback)
            self._state.agent_ids.pop(issue_id, None)

    async def _abort_active_tasks(
        self, run_metadata: RunMetadata, *, is_interrupt: bool = False
    ) -> AbortResult:
        """Cancel active tasks and mark them as failed.

        Delegates to DeadlockHandler.abort_active_tasks.

        Args:
            run_metadata: Metadata for the current run.
            is_interrupt: If True, use "Interrupted" summary instead of "Aborted".

        Returns:
            AbortResult with aborted count and flag indicating unresponsive tasks.
        """
        return await self._deadlock_handler.abort_active_tasks(
            self.active_tasks,
            self.abort_reason,
            self._state,
            run_metadata,
            is_interrupt=is_interrupt,
        )

    def _build_session_callbacks(self, issue_id: str) -> SessionCallbacks:
        """Build callbacks for session operations.

        Delegates to SessionCallbackFactory for callback construction.

        Args:
            issue_id: The issue ID for tracking state.

        Returns:
            SessionCallbacks with gate, review, and logging callbacks.
        """
        return self.session_callback_factory.build(
            issue_id=issue_id,
            on_abort=self._request_abort,
        )

    async def run_implementer(
        self, issue_id: str, *, flow: str = "implementer"
    ) -> IssueResult:
        """Run implementer agent for a single issue with gate retry support.

        Delegates to AgentSessionRunner for SDK-specific session handling.
        """
        temp_agent_id = f"{issue_id}-{uuid.uuid4().hex[:8]}"
        self._state.agent_ids[issue_id] = temp_agent_id

        # Prepare session inputs (before deadlock registration for strict-resume early exit)
        issue_description = await self.beads.get_issue_description_async(issue_id)
        prompt = format_implementer_prompt(
            self._prompts.implementer_prompt,
            issue_id=issue_id,
            repo_path=self.repo_path,
            agent_id=temp_agent_id,
            validation_commands=self._prompt_validation_commands,
            lock_dir=get_lock_dir(),
        )

        # Look up prior session for resumption when include_wip is enabled
        # Must happen BEFORE deadlock registration to avoid leaking agent on strict early exit
        resume_session_id: str | None = None
        baseline_timestamp: int | None = None
        if self.include_wip:
            prior_session = lookup_prior_session_info(self.repo_path, issue_id)
            resume_session_id = prior_session.session_id if prior_session else None
            baseline_timestamp = (
                prior_session.baseline_timestamp if prior_session else None
            )
            if resume_session_id:
                logger.debug(
                    "Resuming session %s for issue %s",
                    resume_session_id,
                    issue_id,
                )
                # Build resume prompt if prior session has review issues
                # _build_resume_prompt returns None if no issues, so call unconditionally
                if prior_session:
                    resume_prompt = _build_resume_prompt(
                        prior_session.last_review_issues or [],
                        self._prompts,
                        self._prompt_validation_commands,
                        issue_id,
                        self.max_review_retries,
                        self.repo_path,
                        prior_session.run_id,
                    )
                    if resume_prompt:
                        prompt = resume_prompt
            elif self.strict_resume:
                # Strict mode: fail issue if no prior session found
                logger.warning(
                    "No prior session found for issue %s in strict mode",
                    issue_id,
                )
                return IssueResult(
                    issue_id=issue_id,
                    agent_id=temp_agent_id,
                    success=False,
                    summary="No prior session found for resume (strict mode)",
                    duration_seconds=0.0,
                    session_id=None,
                    gate_attempts=0,
                    review_attempts=0,
                    resolution=None,
                    session_log_path=None,
                )

        # Register with deadlock monitor (after strict-resume check to avoid leaks)
        self.deadlock_monitor.register_agent(
            agent_id=temp_agent_id,
            issue_id=issue_id,
            start_time=time.time(),
        )

        session_input = AgentSessionInput(
            issue_id=issue_id,
            prompt=prompt,
            issue_description=issue_description,
            agent_id=temp_agent_id,
            resume_session_id=resume_session_id,
            flow=flow,
            baseline_timestamp=baseline_timestamp,
        )

        runner = AgentSessionRunner(
            config=self._session_config,
            sdk_client_factory=self._sdk_client_factory,
            callbacks=self._build_session_callbacks(issue_id),
            event_sink=self.event_sink,
        )
        tracer = self.telemetry_provider.create_span(
            issue_id,
            metadata={
                "agent_id": temp_agent_id,
                "mala_version": __version__,
                "project_dir": self.repo_path.name,
                "git_branch": await get_git_branch_async(self.repo_path),
                "git_commit": await get_git_commit_async(self.repo_path),
                "timeout_seconds": self.timeout_seconds,
            },
        )

        try:
            with tracer:
                tracer.log_input(prompt)
                # Stale session handling is done inside run_session - it will
                # automatically retry without resume_session_id if SDK rejects it
                output = await runner.run_session(
                    session_input,
                    tracer=tracer,
                    interrupt_event=self._interrupt_event,
                )
                self._state.agent_ids[issue_id] = output.agent_id
                if output.success:
                    tracer.set_success(True)
                else:
                    tracer.set_error(output.summary)
        finally:
            # Get agent_id for lock cleanup but don't pop - entry needed for get_agent_id callback
            # until finalization completes (see _finalize_issue_result)
            agent_id = self._state.agent_ids.get(issue_id, temp_agent_id)
            # Skip cleanup if already done during deadlock handling (avoid double unregister)
            if agent_id not in self._state.deadlock_cleaned_agents:
                self._cleanup_agent_locks(agent_id)
            else:
                logger.debug(
                    "Skipped cleanup for %s (handled during deadlock)", agent_id
                )
                self._state.deadlock_cleaned_agents.discard(agent_id)

        return IssueResult(
            issue_id=issue_id,
            agent_id=output.agent_id,
            success=output.success,
            summary=output.summary,
            duration_seconds=output.duration_seconds,
            session_id=output.session_id,
            gate_attempts=output.gate_attempts,
            review_attempts=output.review_attempts,
            resolution=output.resolution,
            low_priority_review_issues=output.low_priority_review_issues,
            session_log_path=output.log_path,
            review_log_path=output.review_log_path,
            baseline_timestamp=output.baseline_timestamp,
            last_review_issues=output.last_review_issues,
        )

    async def spawn_agent(
        self, issue_id: str, flow: str = "implementer"
    ) -> asyncio.Task | None:  # type: ignore[type-arg]
        """Spawn a new agent task for an issue. Returns the Task if spawned, None otherwise."""
        if not await self.beads.claim_async(issue_id):
            self.issue_coordinator.mark_failed(issue_id)
            self.event_sink.on_claim_failed(issue_id, issue_id)
            return None

        task = asyncio.create_task(self.run_implementer(issue_id, flow=flow))
        self.event_sink.on_agent_started(issue_id, issue_id)
        return task

    async def _run_main_loop(
        self,
        run_metadata: RunMetadata,
        *,
        watch_config: WatchConfig | None = None,
        validation_config: PeriodicValidationConfig | None = None,
        drain_event: asyncio.Event | None = None,
        interrupt_event: asyncio.Event | None = None,
        validation_callback: Callable[[], Awaitable[bool]] | None = None,
    ) -> RunResult:
        """Run the main agent spawning and completion loop.

        Delegates to IssueExecutionCoordinator for the main loop logic.

        Args:
            run_metadata: Metadata for the current run.
            watch_config: Watch mode configuration.
            validation_config: Periodic validation configuration.
            drain_event: Event set to enter drain mode (1st Ctrl-C).
            interrupt_event: Event set to signal graceful shutdown (e.g., SIGINT).
            validation_callback: Called periodically in watch mode to run validation.

        Returns:
            RunResult with issues_spawned, exit_code, and exit_reason.
        """

        async def finalize_callback(
            issue_id: str,
            task: asyncio.Task,  # type: ignore[type-arg]
        ) -> None:
            """Finalize a completed task."""
            try:
                result = task.result()
            except asyncio.CancelledError:
                if issue_id in self._state.deadlock_victim_issues:
                    summary = (
                        "Killed as deadlock victim (will retry after blocker completes)"
                    )
                elif self.issue_coordinator.abort_reason:
                    summary = f"Aborted: {self.issue_coordinator.abort_reason}"
                else:
                    summary = "Aborted due to task cancellation"
                result = IssueResult(
                    issue_id=issue_id,
                    agent_id=self._state.agent_ids.get(issue_id, "unknown"),
                    success=False,
                    summary=summary,
                    session_log_path=self._state.active_session_log_paths.get(issue_id),
                )
            except Exception as e:
                result = IssueResult(
                    issue_id=issue_id,
                    agent_id=self._state.agent_ids.get(issue_id, "unknown"),
                    success=False,
                    summary=str(e),
                    session_log_path=self._state.active_session_log_paths.get(issue_id),
                )
            await self._finalize_issue_result(issue_id, result, run_metadata)
            self.issue_coordinator.mark_completed(issue_id)

            # T009: Hook for periodic trigger check after issue completion
            self._check_and_queue_periodic_trigger(result)
            # T011: Blocking wait - execute queued triggers before next issue assignment
            # This covers both PERIODIC triggers (queued above) and EPIC_COMPLETION triggers
            # (queued via epic_verification_coordinator during _finalize_issue_result)
            trigger_result = await self.run_coordinator.run_trigger_validation()
            if trigger_result.status == "aborted":
                self.issue_coordinator.request_abort(
                    reason="Trigger validation aborted"
                )

        async def abort_callback(*, is_interrupt: bool = False) -> AbortResult:
            """Abort all active tasks."""
            return await self._abort_active_tasks(
                run_metadata, is_interrupt=is_interrupt
            )

        return await self.issue_coordinator.run_loop(
            spawn_callback=self.spawn_agent,
            finalize_callback=finalize_callback,
            abort_callback=abort_callback,
            watch_config=watch_config,
            validation_config=validation_config,
            drain_event=drain_event,
            interrupt_event=interrupt_event,
            validation_callback=validation_callback,
        )

    async def _finalize_run(
        self,
        run_metadata: RunMetadata,
        run_validation_passed: bool | None,
        *,
        validation_ran: bool = True,
    ) -> tuple[int, int]:
        """Log summary and return final results.

        Args:
            run_metadata: Run metadata for saving.
            run_validation_passed: Whether validation passed (None if skipped/interrupted).
            validation_ran: Whether validation was attempted. False if skipped due to
                no issues completed. Defaults to True.
        """
        success_count = sum(1 for r in self._state.completed if r.success)
        total = len(self._state.completed)

        # Emit run completed event
        abort_reason = (
            self.abort_reason or "Unrecoverable error" if self.abort_run else None
        )
        self.event_sink.on_run_completed(
            success_count,
            total,
            run_validation_passed,
            abort_reason,
            validation_ran=validation_ran,
        )

        if success_count > 0:
            if await self.beads.commit_issues_async():
                self.event_sink.on_issues_committed()

        if total > 0:
            metadata_path = run_metadata.save()
            self.event_sink.on_run_metadata_saved(str(metadata_path))

        print()
        if self.abort_run:
            return (0, total)
        if run_validation_passed is False:
            return (0, total)
        return (success_count, total)

    def _reset_sigint_state(self) -> None:
        """Reset SIGINT escalation state for a new run."""
        self._sigint_count = 0
        self._sigint_last_at = 0.0
        self._drain_mode_active = False
        self._abort_mode_active = False
        self._abort_exit_code = 130
        self._validation_failed = False
        self._shutdown_requested = False
        self._run_task = None
        self._interrupt_event = None

    def _check_and_queue_periodic_trigger(self, result: IssueResult) -> None:
        """Check if periodic trigger should fire and queue it if so.

        Called after each issue completion. Only non-epic issues increment
        _non_epic_completed_count. Queues a PERIODIC trigger when the count
        reaches the configured interval.

        Args:
            result: The completed issue's result (checked for is_epic).
        """
        # Skip epic issues - only count non-epic completions
        if result.is_epic:
            return

        self._non_epic_completed_count += 1

        # Check if periodic trigger is configured
        triggers = (
            self._validation_config.validation_triggers
            if self._validation_config
            else None
        )
        if triggers is None or triggers.periodic is None:
            return

        # Check if count matches interval
        interval = triggers.periodic.interval
        if self._non_epic_completed_count % interval == 0:
            self.run_coordinator.queue_trigger_validation(
                TriggerType.PERIODIC,
                {"completed_count": self._non_epic_completed_count},
            )

    async def _capture_run_start_commit(self) -> None:
        """Capture HEAD commit at the start of the run.

        Called early in run() to record the starting commit SHA for later
        diff calculations (cumulative code review, run_end trigger context).

        Stores the commit in self._state.run_start_commit, which is later
        transferred to run_metadata after it's created.

        Only captures when run_end trigger is configured, since that's when
        run_start_commit capture is needed.
        """
        # Check if run_end trigger is configured
        triggers = (
            self._validation_config.validation_triggers
            if self._validation_config
            else None
        )
        if triggers is None or triggers.run_end is None:
            return

        # Already captured (should not happen in normal flow, but guard for safety)
        if self._state.run_start_commit is not None:
            return

        # Capture HEAD commit
        head = await get_git_commit_async(self.repo_path)
        if head:
            self._state.run_start_commit = head
            logger.debug(f"Captured run_start_commit: {head}")

    async def _fire_run_end_trigger(self, success_count: int, total_count: int) -> None:
        """Queue run_end trigger validation if configured.

        Called after all issues are processed (after session_end trigger,
        before global validation).

        Args:
            success_count: Number of successfully completed issues.
            total_count: Total number of issues processed.
        """
        # Skip if abort already requested
        if self.abort_run:
            return

        # Skip if no issues were processed
        if total_count == 0:
            return

        # Check if run_end trigger is configured
        triggers = (
            self._validation_config.validation_triggers
            if self._validation_config
            else None
        )
        if triggers is None or triggers.run_end is None:
            return

        trigger_config = triggers.run_end

        # Check fire_on filter
        all_success = success_count == total_count
        any_failure = success_count < total_count

        should_fire = (
            (trigger_config.fire_on == FireOn.SUCCESS and all_success)
            or (trigger_config.fire_on == FireOn.FAILURE and any_failure)
            or (trigger_config.fire_on == FireOn.BOTH)
        )

        if not should_fire:
            logger.debug(
                f"Skipping run_end trigger (fire_on={trigger_config.fire_on.value}, "
                f"success={success_count}/{total_count})"
            )
            return

        # Queue RUN_END trigger validation
        self.run_coordinator.queue_trigger_validation(
            TriggerType.RUN_END,
            {"success_count": success_count, "total_count": total_count},
        )

        # Blocking wait for trigger validation
        result = await self.run_coordinator.run_trigger_validation()

        # If trigger validation aborted, set orchestrator abort_run flag.
        if result.status == "aborted":
            self.issue_coordinator.request_abort(reason="Run end trigger aborted")

    async def _fire_session_end_trigger(self) -> None:
        """Queue session_end trigger validation if configured.

        Called at the end of a session (before final validation).
        Checks skip conditions and queues the trigger if appropriate.

        Skip conditions:
        - abort_run=True: Abort takes precedence over session end trigger
        - success_count==0: Nothing to validate if no issues succeeded
        """
        # Skip if abort already requested
        if self.abort_run:
            return

        # Skip if no successful issues
        success_count = sum(1 for r in self._state.completed if r.success)
        if success_count == 0:
            return

        # Check if session_end trigger is configured
        triggers = (
            self._validation_config.validation_triggers
            if self._validation_config
            else None
        )
        if triggers is None or triggers.session_end is None:
            return

        # Queue SESSION_END trigger
        self.run_coordinator.queue_trigger_validation(
            TriggerType.SESSION_END,
            {"success_count": success_count},
        )

        # Blocking wait for trigger validation
        result = await self.run_coordinator.run_trigger_validation()

        # If trigger validation aborted, set orchestrator abort_run flag.
        # Note: self.abort_run is a read-only property that delegates to
        # issue_coordinator.abort_run, so request_abort() sets the flag.
        if result.status == "aborted":
            self.issue_coordinator.request_abort(reason="Session end trigger aborted")

    def _handle_sigint(
        self,
        loop: asyncio.AbstractEventLoop,
        drain_event: asyncio.Event,
        interrupt_event: asyncio.Event,
    ) -> None:
        """Handle SIGINT with three-stage escalation.

        Stage 1 (1st Ctrl-C): Drain mode - stop accepting new issues, finish current
        Stage 2 (2nd Ctrl-C): Graceful abort - cancel tasks, forward SIGINT
        Stage 3 (3rd Ctrl-C): Hard abort - SIGKILL all processes, cancel run task

        The escalation window (5s) resets only when idle (not in drain/abort mode).
        """
        now = time.monotonic()
        # Reset escalation window if idle (not in drain/abort mode)
        if not self._drain_mode_active and not self._abort_mode_active:
            if now - self._sigint_last_at > ESCALATION_WINDOW_SECONDS:
                self._sigint_count = 0

        self._sigint_count += 1
        self._sigint_last_at = now

        if self._sigint_count == 1:
            # Stage 1: Drain
            self._drain_mode_active = True
            loop.call_soon_threadsafe(drain_event.set)
            # Get active task count from coordinator (capture before lambda)
            active_count = len(self.issue_coordinator.active_tasks)
            loop.call_soon_threadsafe(
                lambda: self.event_sink.on_drain_started(active_count)
            )
        elif self._sigint_count == 2:
            # Stage 2: Graceful Abort
            self._abort_mode_active = True
            self._abort_exit_code = 1 if self._validation_failed else 130
            loop.call_soon_threadsafe(interrupt_event.set)
            loop.call_soon_threadsafe(CommandRunner.forward_sigint)
            loop.call_soon_threadsafe(self.event_sink.on_abort_started)
        else:
            # Stage 3: Hard Abort - always exit 130 regardless of validation state
            self._shutdown_requested = True
            self._abort_exit_code = 130  # Hard abort always uses 130
            loop.call_soon_threadsafe(CommandRunner.kill_active_process_groups)
            loop.call_soon_threadsafe(self.event_sink.on_force_abort)
            if self._run_task:
                loop.call_soon_threadsafe(self._run_task.cancel)

    async def run(
        self,
        *,
        watch_config: WatchConfig | None = None,
        validation_config: PeriodicValidationConfig | None = None,
    ) -> tuple[int, int]:
        """Main orchestration loop. Returns (success_count, total_count).

        Args:
            watch_config: Optional watch mode configuration. If provided, enables
                watch mode with polling and interruptible idle sleep.
            validation_config: Optional periodic validation configuration. If provided
                with a validate_every value, enables periodic validation triggering.
        """
        # Reset per-run state to ensure clean state if orchestrator is reused
        self._state = OrchestratorState()
        self._exit_code = 0
        self._reset_sigint_state()

        # Clear any stale trigger queue from previous runs to prevent
        # queued triggers (e.g., run_end) from leaking into this run
        self.run_coordinator.clear_trigger_queue("run_start_cleanup")

        # Build validation triggers summary from the merged config
        validation_triggers = (
            self._validation_config.validation_triggers
            if self._validation_config
            else None
        )
        run_config = build_event_run_config(
            repo_path=self.repo_path,
            max_agents=self.max_agents,
            timeout_seconds=self.timeout_seconds,
            max_issues=self.max_issues,
            max_gate_retries=self.max_gate_retries,
            max_review_retries=self.max_review_retries,
            epic_id=self.epic_id,
            only_ids=self.only_ids,
            review_enabled=self._is_review_enabled(),
            review_disabled_reason=self.review_disabled_reason,
            include_wip=self.include_wip,
            orphans_only=self.orphans_only,
            cli_args=self.cli_args,
            validation_triggers=validation_triggers,
        )
        self.event_sink.on_run_started(run_config)

        # T010: Capture HEAD at run start for cumulative review diff calculations
        await self._capture_run_start_commit()

        get_lock_dir().mkdir(parents=True, exist_ok=True)
        base_runs_dir = self._runs_dir if self._runs_dir is not None else get_runs_dir()
        base_runs_dir.mkdir(parents=True, exist_ok=True)
        encoded_repo = encode_repo_path(self.repo_path)
        runs_dir = (
            base_runs_dir
            if base_runs_dir.name == encoded_repo
            else base_runs_dir / encoded_repo
        )
        runs_dir.mkdir(parents=True, exist_ok=True)

        run_metadata = build_run_metadata(
            repo_path=self.repo_path,
            max_agents=self.max_agents,
            timeout_seconds=self.timeout_seconds,
            max_issues=self.max_issues,
            epic_id=self.epic_id,
            only_ids=self.only_ids,
            max_gate_retries=self.max_gate_retries,
            max_review_retries=self.max_review_retries,
            review_enabled=self._is_review_enabled(),
            orphans_only=self.orphans_only,
            cli_args=self.cli_args,
            version=__version__,
            runs_dir=runs_dir,
        )

        # Transfer run_start_commit from state to metadata
        if self._state.run_start_commit is not None:
            run_metadata.run_start_commit = self._state.run_start_commit

        # Inject run_metadata into run_coordinator for cumulative code review
        self.run_coordinator.run_metadata = run_metadata

        per_session_spec = build_validation_spec(
            self.repo_path,
            scope=ValidationScope.PER_SESSION,
            disable_validations=self._disabled_validations,
            validation_config=self._validation_config,
            config_missing=self._validation_config_missing,
        )
        self.async_gate_runner.per_session_spec = per_session_spec
        self._lint_tools = extract_lint_tools_from_spec(per_session_spec)
        self._session_config.lint_tools = self._lint_tools
        write_run_marker(
            run_id=run_metadata.run_id,
            repo_path=self.repo_path,
            max_agents=self.max_agents,
        )

        # Create events for SIGINT handling
        drain_event = asyncio.Event()
        interrupt_event = asyncio.Event()
        self._interrupt_event = interrupt_event  # Store for callback access
        loop = asyncio.get_running_loop()

        # Install SIGINT handler using _handle_sigint for three-stage escalation
        def handle_sigint(sig: int, frame: object) -> None:
            self._handle_sigint(loop, drain_event, interrupt_event)

        original_handler = signal.signal(signal.SIGINT, handle_sigint)

        # Track exit code for propagation to CLI
        exit_code = 0

        try:
            # Create validation callback only if validation is configured
            validation_callback = None
            if validation_config and validation_config.validate_every is not None:

                async def _validation_callback() -> bool:
                    success_count = sum(1 for r in self._state.completed if r.success)
                    if success_count == 0 or self.abort_run:
                        return True
                    validation_input = GlobalValidationInput(run_metadata=run_metadata)
                    validation_output = await self.run_coordinator.run_validation(
                        validation_input, interrupt_event=interrupt_event
                    )
                    if validation_output.interrupted:
                        return True
                    # Update _validation_failed so SIGINT handler can snapshot it
                    if not validation_output.passed:
                        self._validation_failed = True
                    return validation_output.passed

                validation_callback = _validation_callback

            interrupted = False
            try:
                # Create task and store it so Stage 3 can cancel it
                self._run_task = asyncio.create_task(
                    self._run_main_loop(
                        run_metadata,
                        watch_config=watch_config,
                        validation_config=validation_config,
                        drain_event=drain_event,
                        interrupt_event=interrupt_event,
                        validation_callback=validation_callback,
                    )
                )
                loop_result = await self._run_task
                exit_code = loop_result.exit_code
            except asyncio.CancelledError:
                # Only treat as SIGINT if interrupt_event is set or shutdown requested
                if interrupt_event.is_set() or self._shutdown_requested:
                    # Use abort_exit_code for Stage 2/3 (set by _handle_sigint)
                    exit_code = (
                        self._abort_exit_code if self._abort_mode_active else 130
                    )
                    interrupted = True
                else:
                    raise
            finally:
                lock_releaser = (
                    self._lock_releaser
                    if self._lock_releaser is not None
                    else release_run_locks
                )
                released = lock_releaser(list(self._state.agent_ids.values()))
                # Only emit event if released is a positive integer
                # (release_run_locks returns int, but tests may mock it)
                if isinstance(released, int) and released > 0:
                    self.event_sink.on_locks_released(released)
                remove_run_marker(run_metadata.run_id)

            # Check if interrupted - use _abort_exit_code for Stage 2 (snapshotted at
            # abort entry based on whether validation had already failed), or exit_code
            # from loop_result for Stage 1 drain completion
            if interrupted or interrupt_event.is_set():
                # Stage 2 abort: use snapshotted exit code from abort entry
                # Stage 1 drain: use loop_result exit code (validation ran after drain)
                final_exit_code = (
                    self._abort_exit_code if self._abort_mode_active else exit_code
                )
                self._exit_code = final_exit_code
                return await self._finalize_run(run_metadata, None)

            # Global validation and finalization happen after lock cleanup
            # but before debug log cleanup (so they're captured in the debug log)
            success_count = sum(1 for r in self._state.completed if r.success)

            # T009: Hook for session_end trigger before global validation
            await self._fire_session_end_trigger()

            # T010: Hook for run_end trigger after all processing
            await self._fire_run_end_trigger(success_count, len(self._state.completed))

            run_validation_passed: bool | None = None
            validation_ran = False  # Track if validation was actually attempted
            if success_count > 0 and not self.abort_run:
                validation_ran = True
                try:
                    validation_input = GlobalValidationInput(run_metadata=run_metadata)
                    validation_output = await self.run_coordinator.run_validation(
                        validation_input, interrupt_event=interrupt_event
                    )
                    # Map interrupted validation to None (not False)
                    # Acceptance criteria: validation_ran=True + run_validation_passed=None → exit 130
                    if validation_output.interrupted:
                        interrupt_event.set()
                        run_validation_passed = None
                    else:
                        run_validation_passed = validation_output.passed
                except asyncio.CancelledError:
                    # SIGINT during validation - use snapshotted exit code
                    if interrupt_event.is_set() or self._shutdown_requested:
                        final_exit_code = (
                            self._abort_exit_code if self._abort_mode_active else 130
                        )
                        self._exit_code = final_exit_code
                        return await self._finalize_run(run_metadata, None)
                    raise

            # Check if SIGINT occurred during final validation phase
            # Note: this should rarely be reached since interrupt is handled earlier,
            # but keep as safety net. Use snapshotted exit code for Stage 2.
            if interrupt_event.is_set():
                final_exit_code = (
                    self._abort_exit_code if self._abort_mode_active else 130
                )
                self._exit_code = final_exit_code
                return await self._finalize_run(run_metadata, None)

            # Store exit code for CLI access
            # - validation_passed=False → exit 1 (failure)
            # - validation_passed=None (interrupted) → exit 130
            # - validation_passed=True or validation didn't run → use loop exit_code
            if run_validation_passed is False:
                self._exit_code = 1
            elif run_validation_passed is None and validation_ran:
                # Validation was attempted but interrupted
                self._exit_code = 130
            else:
                self._exit_code = exit_code
            return await self._finalize_run(
                run_metadata, run_validation_passed, validation_ran=validation_ran
            )
        finally:
            # Restore original SIGINT handler
            signal.signal(signal.SIGINT, original_handler)
            # Clean up debug logging handler after all finalization is complete
            # (idempotent - safe to call even if save() is called later)
            run_metadata.cleanup()

    def run_sync(
        self,
        *,
        watch_config: WatchConfig | None = None,
        validation_config: PeriodicValidationConfig | None = None,
    ) -> tuple[int, int]:
        """Synchronous wrapper for run(). Returns (success_count, total_count).

        Use this method when calling from synchronous code. It handles event loop
        creation and cleanup automatically.

        Args:
            watch_config: Optional watch mode configuration. If provided, enables
                watch mode with polling and interruptible idle sleep.
            validation_config: Optional periodic validation configuration. If provided
                with a validate_every value, enables periodic validation triggering.

        Raises:
            RuntimeError: If called from within an async context (e.g., inside an
                async function or when an event loop is already running). In that
                case, use `await orchestrator.run()` instead.

        Example usage::

            from src.orchestration.factory import create_orchestrator, OrchestratorConfig

            # From sync code (scripts, CLI, tests)
            config = OrchestratorConfig(repo_path=Path("."))
            orchestrator = create_orchestrator(config)
            success_count, total = orchestrator.run_sync()

            # From async code
            async def main():
                config = OrchestratorConfig(repo_path=Path("."))
                orchestrator = create_orchestrator(config)
                success_count, total = await orchestrator.run()

        Returns:
            Tuple of (success_count, total_count).
        """
        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # If we get here, there's a running event loop - can't use run_sync()
            raise RuntimeError(
                "run_sync() cannot be called from within an async context. "
                "Use 'await orchestrator.run()' instead."
            )
        except RuntimeError as e:
            # Check if it's our error or the "no running event loop" error
            if "run_sync() cannot be called" in str(e):
                raise
            # No running event loop - this is the expected case for sync usage

        # Create a new event loop and run
        return asyncio.run(
            self.run(watch_config=watch_config, validation_config=validation_config)
        )

"""MalaOrchestrator: Orchestrates parallel issue processing using Claude Agent SDK."""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING

from src.domain.validation.spec import (
    ValidationScope,
    build_validation_spec,
)
from src.domain.validation.tool_name_extractor import extract_lint_tools_from_spec
from src.domain.prompts import (
    build_prompt_validation_commands,
    format_implementer_prompt,
)
from src.infra.git_utils import (
    get_baseline_for_issue,
    get_git_branch_async,
    get_git_commit_async,
)
from src.infra.clients.cerberus_review import DefaultReviewer
from src.infra.io.log_output.run_metadata import (
    remove_run_marker,
    write_run_marker,
)
from src.infra.tools.env import (
    get_lock_dir,
    get_runs_dir,
)
from src.infra.tools.locking import (
    cleanup_agent_locks,
    release_run_locks,
)
from src.pipeline.agent_session_runner import (
    AgentSessionInput,
    AgentSessionRunner,
)
from src.pipeline.issue_finalizer import (
    IssueFinalizeInput,
)
from src.pipeline.run_coordinator import (
    RunLevelValidationInput,
)
from src.orchestration.orchestration_wiring import (
    WiringDependencies,
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
)
from src.orchestration.issue_result import IssueResult
from src.orchestration.review_tracking import create_review_tracking_issues
from src.orchestration.run_config import build_event_run_config, build_run_metadata

if TYPE_CHECKING:
    from pathlib import Path

    from src.core.protocols import (
        CodeReviewer,
        GateChecker,
        IssueProvider,
        LogProvider,
    )
    from src.infra.epic_verifier import EpicVerifier
    from src.infra.io.config import MalaConfig
    from src.infra.io.event_protocol import MalaEventSink
    from src.infra.io.log_output.run_metadata import RunMetadata
    from src.infra.telemetry import TelemetryProvider
    from src.pipeline.agent_session_runner import SessionCallbacks
    from src.pipeline.epic_verification_coordinator import EpicVerificationCoordinator
    from src.pipeline.issue_finalizer import IssueFinalizer

    from .types import OrchestratorConfig, _DerivedConfig


# Version (from package metadata)
from importlib.metadata import version as pkg_version

__version__ = pkg_version("mala")

# Bounded wait for log file (seconds) - used by AgentSessionRunner
# Re-exported here for backwards compatibility with tests.
# 60s allows time for Claude SDK to flush logs, especially under load.
LOG_FILE_WAIT_TIMEOUT = 60
LOG_FILE_POLL_INTERVAL = 0.5


class MalaOrchestrator:
    """Orchestrates parallel issue processing using Claude Agent SDK.

    Use create_orchestrator() factory function to instantiate this class.
    The factory provides clean separation of concerns and easier testing.
    """

    # Type annotations for attributes set during initialization
    _max_issues: int | None

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
        _epic_verifier: EpicVerifier | None = None,
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
        epic_verifier: EpicVerifier | None,
    ) -> None:
        """Initialize from factory-provided config and dependencies."""
        self._mala_config = mala_config
        self.repo_path = orch_config.repo_path.resolve()
        self.max_agents = orch_config.max_agents
        self.timeout_seconds = derived.timeout_seconds
        self.max_issues = orch_config.max_issues
        self.epic_id = orch_config.epic_id
        self.only_ids = orch_config.only_ids
        self.braintrust_enabled = derived.braintrust_enabled
        self.max_gate_retries = orch_config.max_gate_retries
        self.max_review_retries = orch_config.max_review_retries
        self.disable_validations = orch_config.disable_validations
        self._disabled_validations = derived.disabled_validations
        self.coverage_threshold = orch_config.coverage_threshold
        self.prioritize_wip = orch_config.prioritize_wip
        self.focus = orch_config.focus
        self.orphans_only = orch_config.orphans_only
        self.cli_args = orch_config.cli_args
        self.epic_override_ids = orch_config.epic_override_ids or set()
        self.review_disabled_reason = derived.review_disabled_reason
        self.braintrust_disabled_reason = derived.braintrust_disabled_reason
        self._init_runtime_state()
        self.log_provider = log_provider
        self.quality_gate = gate_checker
        self.event_sink = event_sink
        self.beads = issue_provider
        self.epic_verifier = epic_verifier
        self.code_reviewer = code_reviewer
        self.telemetry_provider = telemetry_provider
        self._init_pipeline_runners()

    def _init_runtime_state(self) -> None:
        """Initialize runtime state.

        Note: active_tasks, failed_issues, abort_run, and abort_reason are
        delegated to issue_coordinator to maintain a single source of truth.
        See the corresponding properties.

        Note: verified_epics and epics_being_verified are delegated to
        epic_verification_coordinator.

        Note: per_issue_spec and last_gate_results are delegated to
        async_gate_runner.
        """
        self.agent_ids: dict[str, str] = {}
        self.completed: list[IssueResult] = []
        self.session_log_paths: dict[str, Path] = {}
        self.review_log_paths: dict[str, str] = {}
        self._lint_tools: frozenset[str] | None = None
        self._prompt_validation_commands = build_prompt_validation_commands(
            self.repo_path
        )

    def _init_pipeline_runners(self) -> None:
        """Initialize pipeline runner components using wiring functions."""
        deps = self._build_wiring_dependencies()

        # Build core runners
        self.gate_runner, self.async_gate_runner = build_gate_runner(deps)
        self.review_runner = build_review_runner(deps)
        self.run_coordinator = build_run_coordinator(deps)
        self.issue_coordinator = build_issue_coordinator(deps)

        # Build coordinators with callbacks (callbacks need self references)
        self.issue_finalizer = self._build_issue_finalizer()
        self.epic_verification_coordinator = self._build_epic_verification_coordinator()

        # Build session infrastructure
        self.session_callback_factory = build_session_callback_factory(
            deps,
            self.async_gate_runner,
            self.review_runner,
            lambda: self.log_provider,
            lambda: self.quality_gate,
        )
        self._session_config = build_session_config(
            deps, review_enabled=self._is_review_enabled()
        )

    def _build_wiring_dependencies(self) -> WiringDependencies:
        """Build WiringDependencies from orchestrator state."""
        return WiringDependencies(
            repo_path=self.repo_path,
            quality_gate=self.quality_gate,
            code_reviewer=self.code_reviewer,
            beads=self.beads,
            event_sink=self.event_sink,
            mala_config=self._mala_config,
            max_agents=self.max_agents,
            max_issues=self._max_issues,
            timeout_seconds=self.timeout_seconds,
            max_gate_retries=self.max_gate_retries,
            max_review_retries=self.max_review_retries,
            coverage_threshold=self.coverage_threshold,
            disabled_validations=set(self._disabled_validations)
            if self._disabled_validations
            else None,
            epic_id=self.epic_id,
            only_ids=set(self.only_ids) if self.only_ids else None,
            prioritize_wip=self.prioritize_wip,
            focus=self.focus,
            orphans_only=self.orphans_only,
            epic_override_ids=self.epic_override_ids,
            prompt_validation_commands=self._prompt_validation_commands,
            session_log_paths=self.session_log_paths,
            review_log_paths=self.review_log_paths,
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
                        issue_id, run_metadata
                    )
                ),
                create_tracking_issues=self._create_review_tracking_issues,
            )
        )
        return IssueFinalizer(
            config=config,
            callbacks=callbacks,
            quality_gate=self.quality_gate,
            per_issue_spec=None,
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
                spawn_remediation=lambda issue_id: self.spawn_agent(issue_id),
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
                get_agent_id=lambda issue_id: self.agent_ids.get(issue_id, "unknown"),
            )
        )
        return EpicVerificationCoordinator(
            config=config,
            callbacks=callbacks,
            epic_override_ids=self.epic_override_ids,
        )

    async def _create_review_tracking_issues(
        self,
        issue_id: str,
        review_issues: list,
    ) -> None:
        """Create tracking issues for P2/P3 review findings."""
        await create_review_tracking_issues(
            self.beads,
            self.event_sink,
            issue_id,
            review_issues,
        )

    # Backward compatibility: expose _config as alias for _mala_config
    @property
    def _config(self) -> MalaConfig:
        """Backward compatibility property for accessing MalaConfig."""
        return self._mala_config

    @_config.setter
    def _config(self, value: MalaConfig) -> None:
        """Backward compatibility setter for MalaConfig."""
        self._mala_config = value

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

    def _cleanup_agent_locks(self, agent_id: str) -> None:
        """Remove locks held by a specific agent (crash/timeout cleanup)."""
        cleaned = cleanup_agent_locks(agent_id)
        if cleaned:
            self.event_sink.on_locks_cleaned(agent_id, cleaned)

    def _request_abort(self, reason: str) -> None:
        """Signal that the current run should stop due to a fatal error."""
        self.issue_coordinator.request_abort(reason)

    def _is_review_enabled(self) -> bool:
        """Return whether review should run for this orchestrator instance."""
        if "review" not in self._disabled_validations:
            return True
        if self.review_disabled_reason and not isinstance(
            self.review_runner.code_reviewer, DefaultReviewer
        ):
            return True
        return False

    def _cleanup_session_paths(self, issue_id: str) -> None:
        """Remove stored session and review log paths for an issue.

        Args:
            issue_id: The issue ID to clean up paths for.
        """
        self.session_log_paths.pop(issue_id, None)
        self.review_log_paths.pop(issue_id, None)

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
        log_path = self.session_log_paths.get(issue_id)
        stored_gate_result = self.async_gate_runner.get_last_gate_result(issue_id)

        # Update finalizer's per_issue_spec if it has changed
        self.issue_finalizer.per_issue_spec = self.async_gate_runner.per_issue_spec

        # Build finalization input
        finalize_input = IssueFinalizeInput(
            issue_id=issue_id,
            result=result,
            run_metadata=run_metadata,
            log_path=log_path,
            stored_gate_result=stored_gate_result,
            review_log_path=self.review_log_paths.get(issue_id),
        )

        # Track failed issues before finalize to ensure they're recorded
        # even if finalize raises (e.g., mark_needs_followup callback fails)
        if not result.success:
            self.failed_issues.add(issue_id)

        # Delegate to finalizer, ensuring cleanup happens even if finalize raises
        try:
            await self.issue_finalizer.finalize(finalize_input)
        finally:
            # Update tracking state (active_tasks is updated by mark_completed in finalize_callback)
            self.completed.append(result)

            # Cleanup session paths and gate result
            self._cleanup_session_paths(issue_id)
            self.async_gate_runner.clear_gate_result(issue_id)

            # Remove agent_id from tracking now that finalization is complete
            # (deferred from run_implementer.finally to keep it available for get_agent_id callback)
            self.agent_ids.pop(issue_id, None)

    async def _abort_active_tasks(self, run_metadata: RunMetadata) -> None:
        """Cancel active tasks and mark them as failed.

        Tasks that have already completed are finalized with their real results
        rather than being marked as aborted.
        """
        if not self.active_tasks:
            return
        reason = self.abort_reason or "Unrecoverable error"
        self.event_sink.on_tasks_aborting(len(self.active_tasks), reason)
        # Cancel tasks that are still running
        for task in self.active_tasks.values():
            if not task.done():
                task.cancel()

        # Finalize each remaining issue - use real result if already done
        for issue_id, task in list(self.active_tasks.items()):
            if task.done():
                # Task completed before we could cancel - use real result
                try:
                    result = task.result()
                except asyncio.CancelledError:
                    result = IssueResult(
                        issue_id=issue_id,
                        agent_id=self.agent_ids.get(issue_id, "unknown"),
                        success=False,
                        summary=f"Aborted due to unrecoverable error: {reason}",
                    )
                except Exception as e:
                    result = IssueResult(
                        issue_id=issue_id,
                        agent_id=self.agent_ids.get(issue_id, "unknown"),
                        success=False,
                        summary=str(e),
                    )
            else:
                # Task was still running - mark as aborted
                result = IssueResult(
                    issue_id=issue_id,
                    agent_id=self.agent_ids.get(issue_id, "unknown"),
                    success=False,
                    summary=f"Aborted due to unrecoverable error: {reason}",
                )
            await self._finalize_issue_result(issue_id, result, run_metadata)
            # Mark completed in coordinator to keep state consistent
            self.issue_coordinator.mark_completed(issue_id)

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

    async def run_implementer(self, issue_id: str) -> IssueResult:
        """Run implementer agent for a single issue with gate retry support.

        Delegates to AgentSessionRunner for SDK-specific session handling.
        """
        temp_agent_id = f"{issue_id}-{uuid.uuid4().hex[:8]}"
        self.agent_ids[issue_id] = temp_agent_id

        # Prepare session inputs
        issue_description = await self.beads.get_issue_description_async(issue_id)
        baseline_commit = await get_baseline_for_issue(self.repo_path, issue_id)
        if baseline_commit is None:
            baseline_commit = await get_git_commit_async(self.repo_path)

        prompt = format_implementer_prompt(
            issue_id=issue_id,
            repo_path=self.repo_path,
            agent_id=temp_agent_id,
            validation_commands=self._prompt_validation_commands,
        )

        session_input = AgentSessionInput(
            issue_id=issue_id,
            prompt=prompt,
            baseline_commit=baseline_commit,
            issue_description=issue_description,
        )

        runner = AgentSessionRunner(
            config=self._session_config,
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
                output = await runner.run_session(session_input, tracer=tracer)
                self.agent_ids[issue_id] = output.agent_id
                if output.success:
                    tracer.set_success(True)
                else:
                    tracer.set_error(output.summary)
        finally:
            # Get agent_id for lock cleanup but don't pop - entry needed for get_agent_id callback
            # until finalization completes (see _finalize_issue_result)
            agent_id = self.agent_ids.get(issue_id, temp_agent_id)
            self._cleanup_agent_locks(agent_id)

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
        )

    async def spawn_agent(self, issue_id: str) -> asyncio.Task | None:  # type: ignore[type-arg]
        """Spawn a new agent task for an issue. Returns the Task if spawned, None otherwise."""
        if not await self.beads.claim_async(issue_id):
            self.issue_coordinator.mark_failed(issue_id)
            self.event_sink.on_claim_failed(issue_id, issue_id)
            return None

        task = asyncio.create_task(self.run_implementer(issue_id))
        self.event_sink.on_agent_started(issue_id, issue_id)
        return task

    async def _run_main_loop(self, run_metadata: RunMetadata) -> int:
        """Run the main agent spawning and completion loop.

        Delegates to IssueExecutionCoordinator for the main loop logic.
        Returns the number of issues spawned.
        """

        async def finalize_callback(
            issue_id: str,
            task: asyncio.Task,  # type: ignore[type-arg]
        ) -> None:
            """Finalize a completed task."""
            try:
                result = task.result()
            except asyncio.CancelledError:
                result = IssueResult(
                    issue_id=issue_id,
                    agent_id=self.agent_ids.get(issue_id, "unknown"),
                    success=False,
                    summary=(
                        f"Aborted due to unrecoverable error: {self.issue_coordinator.abort_reason}"
                        if self.issue_coordinator.abort_reason
                        else "Aborted due to unrecoverable error"
                    ),
                )
            except Exception as e:
                result = IssueResult(
                    issue_id=issue_id,
                    agent_id=self.agent_ids.get(issue_id, "unknown"),
                    success=False,
                    summary=str(e),
                )
            await self._finalize_issue_result(issue_id, result, run_metadata)
            self.issue_coordinator.mark_completed(issue_id)

        async def abort_callback() -> None:
            """Abort all active tasks."""
            await self._abort_active_tasks(run_metadata)

        return await self.issue_coordinator.run_loop(
            spawn_callback=self.spawn_agent,
            finalize_callback=finalize_callback,
            abort_callback=abort_callback,
        )

    async def _finalize_run(
        self,
        run_metadata: RunMetadata,
        run_validation_passed: bool,
    ) -> tuple[int, int]:
        """Log summary and return final results."""
        success_count = sum(1 for r in self.completed if r.success)
        total = len(self.completed)

        # Emit run completed event
        abort_reason = (
            self.abort_reason or "Unrecoverable error" if self.abort_run else None
        )
        self.event_sink.on_run_completed(
            success_count, total, run_validation_passed, abort_reason
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
        if not run_validation_passed:
            return (0, total)
        return (success_count, total)

    async def run(self) -> tuple[int, int]:
        """Main orchestration loop. Returns (success_count, total_count)."""
        run_config = build_event_run_config(
            repo_path=self.repo_path,
            max_agents=self.max_agents,
            timeout_seconds=self.timeout_seconds,
            max_issues=self.max_issues,
            max_gate_retries=self.max_gate_retries,
            max_review_retries=self.max_review_retries,
            epic_id=self.epic_id,
            only_ids=self.only_ids,
            braintrust_enabled=self.braintrust_enabled,
            review_enabled=self._is_review_enabled(),
            review_disabled_reason=self.review_disabled_reason,
            prioritize_wip=self.prioritize_wip,
            orphans_only=self.orphans_only,
            cli_args=self.cli_args,
            braintrust_disabled_reason=self.braintrust_disabled_reason,
        )
        self.event_sink.on_run_started(run_config)

        get_lock_dir().mkdir(parents=True, exist_ok=True)
        get_runs_dir().mkdir(parents=True, exist_ok=True)

        run_metadata = build_run_metadata(
            repo_path=self.repo_path,
            max_agents=self.max_agents,
            timeout_seconds=self.timeout_seconds,
            max_issues=self.max_issues,
            epic_id=self.epic_id,
            only_ids=self.only_ids,
            braintrust_enabled=self.braintrust_enabled,
            max_gate_retries=self.max_gate_retries,
            max_review_retries=self.max_review_retries,
            review_enabled=self._is_review_enabled(),
            orphans_only=self.orphans_only,
            cli_args=self.cli_args,
            version=__version__,
        )
        per_issue_spec = build_validation_spec(
            self.repo_path,
            scope=ValidationScope.PER_ISSUE,
            disable_validations=self._disabled_validations,
        )
        self.async_gate_runner.per_issue_spec = per_issue_spec
        self._lint_tools = extract_lint_tools_from_spec(per_issue_spec)
        self._session_config.lint_tools = self._lint_tools
        write_run_marker(
            run_id=run_metadata.run_id,
            repo_path=self.repo_path,
            max_agents=self.max_agents,
        )

        try:
            try:
                await self._run_main_loop(run_metadata)
            finally:
                released = release_run_locks(list(self.agent_ids.values()))
                # Only emit event if released is a positive integer
                # (release_run_locks returns int, but tests may mock it)
                if isinstance(released, int) and released > 0:
                    self.event_sink.on_locks_released(released)
                remove_run_marker(run_metadata.run_id)

            # Run-level validation and finalization happen after lock cleanup
            # but before debug log cleanup (so they're captured in the debug log)
            success_count = sum(1 for r in self.completed if r.success)
            run_validation_passed = True
            if success_count > 0 and not self.abort_run:
                validation_input = RunLevelValidationInput(run_metadata=run_metadata)
                validation_output = await self.run_coordinator.run_validation(
                    validation_input
                )
                run_validation_passed = validation_output.passed

            return await self._finalize_run(run_metadata, run_validation_passed)
        finally:
            # Clean up debug logging handler after all finalization is complete
            # (idempotent - safe to call even if save() is called later)
            run_metadata.cleanup()

    def run_sync(self) -> tuple[int, int]:
        """Synchronous wrapper for run(). Returns (success_count, total_count).

        Use this method when calling from synchronous code. It handles event loop
        creation and cleanup automatically.

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
        return asyncio.run(self.run())

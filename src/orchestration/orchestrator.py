"""MalaOrchestrator: Orchestrates parallel issue processing using Claude Agent SDK."""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING

from src.domain.validation.spec import (
    ValidationScope,
    build_validation_spec,
)
from src.infra.git_utils import (
    get_baseline_for_issue,
    get_git_branch_async,
    get_git_commit_async,
    get_issue_commits_async,
)
from src.infra.clients.cerberus_review import DefaultReviewer
from src.infra.io.log_output.console import (
    is_verbose_enabled,
    truncate_text,
)
from src.infra.io.log_output.run_metadata import (
    IssueRun,
    remove_run_marker,
    write_run_marker,
)
from src.infra.tools.env import (
    SCRIPTS_DIR,
    get_lock_dir,
    get_runs_dir,
)
from src.infra.tools.locking import (
    cleanup_agent_locks,
    release_run_locks,
)
from src.pipeline.agent_session_runner import (
    AgentSessionConfig,
    AgentSessionInput,
    AgentSessionRunner,
    SessionCallbacks,
)
from src.pipeline.gate_runner import (
    GateRunner,
    GateRunnerConfig,
    PerIssueGateInput,
)
from src.pipeline.review_runner import (
    NoProgressInput,
    ReviewInput,
    ReviewRunner,
    ReviewRunnerConfig,
)
from src.pipeline.issue_execution_coordinator import (
    CoordinatorConfig,
    IssueExecutionCoordinator,
)
from src.pipeline.run_coordinator import (
    RunCoordinator,
    RunCoordinatorConfig,
    RunLevelValidationInput,
)
from src.orchestration.gate_metadata import (
    GateMetadata,
    build_gate_metadata,
    build_gate_metadata_from_logs,
)
from src.orchestration.issue_result import IssueResult
from src.orchestration.prompts import get_implementer_prompt
from src.orchestration.review_tracking import create_review_tracking_issues
from src.orchestration.run_config import build_event_run_config, build_run_metadata

if TYPE_CHECKING:
    from pathlib import Path

    from src.core.protocols import (
        CodeReviewer,
        GateChecker,
        GateResultProtocol,
        IssueProvider,
        LogProvider,
        ReviewResultProtocol,
    )
    from src.domain.lifecycle import RetryState
    from src.domain.quality_gate import GateResult
    from src.domain.validation.spec import ValidationSpec
    from src.infra.clients.cerberus_review import ReviewResult
    from src.infra.epic_verifier import EpicVerifier
    from src.infra.io.config import MalaConfig
    from src.infra.io.event_sink import MalaEventSink
    from src.infra.io.log_output.run_metadata import RunMetadata
    from src.infra.telemetry import TelemetryProvider

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
        self.morph_enabled = derived.morph_enabled
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
        """
        self.agent_ids: dict[str, str] = {}
        self.completed: list[IssueResult] = []
        self.session_log_paths: dict[str, Path] = {}
        self.review_log_paths: dict[str, str] = {}
        self.last_gate_results: dict[str, GateResult | GateResultProtocol] = {}
        self.verified_epics: set[str] = set()
        self.epics_being_verified: set[str] = (
            set()
        )  # Guard against re-entrant verification
        self.per_issue_spec: ValidationSpec | None = None

    def _init_pipeline_runners(self) -> None:
        """Initialize pipeline runner components."""
        # GateRunner
        gate_runner_config = GateRunnerConfig(
            max_gate_retries=self.max_gate_retries,
            disable_validations=self._disabled_validations,
            coverage_threshold=self.coverage_threshold,
        )
        self.gate_runner = GateRunner(
            gate_checker=self.quality_gate,
            repo_path=self.repo_path,
            config=gate_runner_config,
        )

        # ReviewRunner
        review_runner_config = ReviewRunnerConfig(
            max_review_retries=self.max_review_retries,
            capture_session_log=False,
            review_timeout=self._mala_config.review_timeout,
        )
        self.review_runner = ReviewRunner(
            code_reviewer=self.code_reviewer,
            config=review_runner_config,
            gate_checker=self.quality_gate,
        )

        # RunCoordinator
        run_coordinator_config = RunCoordinatorConfig(
            repo_path=self.repo_path,
            timeout_seconds=self.timeout_seconds,
            max_gate_retries=self.max_gate_retries,
            disable_validations=self._disabled_validations,
            coverage_threshold=self.coverage_threshold,
            morph_enabled=self.morph_enabled,
            morph_api_key=self._mala_config.morph_api_key,
        )
        self.run_coordinator = RunCoordinator(
            config=run_coordinator_config,
            gate_checker=self.quality_gate,
            event_sink=self.event_sink,
        )

        # IssueExecutionCoordinator
        coordinator_config = CoordinatorConfig(
            max_agents=self.max_agents,
            max_issues=self._max_issues,
            epic_id=self.epic_id,
            only_ids=self.only_ids,
            prioritize_wip=self.prioritize_wip,
            focus=self.focus,
            orphans_only=self.orphans_only,
        )
        self.issue_coordinator = IssueExecutionCoordinator(
            beads=self.beads,
            event_sink=self.event_sink,
            config=coordinator_config,
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

    def _run_quality_gate_sync(
        self,
        issue_id: str,
        log_path: Path,
        retry_state: RetryState,
    ) -> tuple[GateResult | GateResultProtocol, int]:
        """Synchronous quality gate check (blocking I/O).

        This is the blocking implementation that gets run via asyncio.to_thread.
        Delegates to GateRunner for the actual gate checking logic.
        """
        # Sync per_issue_spec with gate_runner (orchestrator may have built it at run start)
        if self.per_issue_spec is not None:
            self.gate_runner.set_cached_spec(self.per_issue_spec)

        gate_input = PerIssueGateInput(
            issue_id=issue_id,
            log_path=log_path,
            retry_state=retry_state,
            spec=self.per_issue_spec,
        )
        output = self.gate_runner.run_per_issue_gate(gate_input)

        # Sync cached spec back to orchestrator (gate_runner may have built it)
        if self.per_issue_spec is None:
            self.per_issue_spec = self.gate_runner.get_cached_spec()

        # Store gate result to avoid duplicate validation in _finalize_issue_result
        # The gate result includes validation_evidence parsed from logs
        self.last_gate_results[issue_id] = output.gate_result

        return (output.gate_result, output.new_log_offset)

    async def _run_quality_gate_async(
        self,
        issue_id: str,
        log_path: Path,
        retry_state: RetryState,
    ) -> tuple[GateResult | GateResultProtocol, int]:
        """Run quality gate check asynchronously via to_thread.

        Wraps the blocking _run_quality_gate_sync to avoid stalling the event loop.
        This allows the orchestrator to service other agents while a gate runs.

        Args:
            issue_id: The issue being checked.
            log_path: Path to the session log file.
            retry_state: Current retry state for this issue.

        Returns:
            Tuple of (GateResult, new_log_offset).
        """
        return await asyncio.to_thread(
            self._run_quality_gate_sync, issue_id, log_path, retry_state
        )

    async def _check_epic_closure(
        self, issue_id: str, run_metadata: RunMetadata
    ) -> None:
        """Check if closing this issue should also close its parent epic.

        Implements a retry loop for epic verification:
        1. Run verification
        2. If verification fails and creates remediation issues, execute them
        3. Re-verify the epic
        4. Repeat until verification passes OR max retries reached

        Args:
            issue_id: The issue that was just closed.
            run_metadata: Run metadata for recording remediation issue results.
        """
        parent_epic = await self.beads.get_parent_epic_async(issue_id)
        if parent_epic is None or parent_epic in self.verified_epics:
            return

        # Guard against re-entrant verification (e.g., when remediation tasks complete)
        if parent_epic in self.epics_being_verified:
            return

        if self.epic_verifier is not None:
            # Mark as being verified to prevent parallel verification loops
            self.epics_being_verified.add(parent_epic)
            try:
                # max_retries is the number of retries AFTER the first attempt
                # So total attempts = 1 (initial) + max_retries
                max_retries = self._mala_config.max_epic_verification_retries
                max_attempts = 1 + max_retries
                human_override = parent_epic in self.epic_override_ids

                for attempt in range(1, max_attempts + 1):
                    # Log attempt if retrying (attempt > 1)
                    if attempt > 1:
                        self.event_sink.on_warning(
                            f"Epic verification retry {attempt - 1}/{max_retries} for {parent_epic}"
                        )

                    verification_result = (
                        await self.epic_verifier.verify_and_close_epic(
                            parent_epic, human_override=human_override
                        )
                    )

                    # If epic wasn't eligible (children still open), don't mark as verified
                    # so it can be re-checked when more children close
                    if verification_result.verified_count == 0:
                        return

                    # If epic passed verification, mark as verified and return
                    if verification_result.passed_count > 0:
                        self.verified_epics.add(parent_epic)
                        return

                    # If no remediation issues were created, or max attempts reached,
                    # mark as verified (to prevent infinite loops) and return
                    if (
                        not verification_result.remediation_issues_created
                        or attempt >= max_attempts
                    ):
                        if (
                            attempt >= max_attempts
                            and verification_result.failed_count > 0
                        ):
                            self.event_sink.on_warning(
                                f"Epic verification failed after {max_retries} retries for {parent_epic}"
                            )
                        self.verified_epics.add(parent_epic)
                        return

                    # Execute remediation issues before next verification attempt
                    await self._execute_remediation_issues(
                        verification_result.remediation_issues_created,
                        run_metadata,
                    )
            finally:
                # Always remove from being_verified set when done
                self.epics_being_verified.discard(parent_epic)

        elif await self.beads.close_eligible_epics_async():
            # Fallback for mock providers without EpicVerifier
            self.event_sink.on_epic_closed(issue_id)

    async def _execute_remediation_issues(
        self, issue_ids: list[str], run_metadata: RunMetadata
    ) -> None:
        """Execute remediation issues and wait for their completion.

        Spawns agents for remediation issues, waits for completion, and finalizes
        results (closes issues, records metadata). This ensures remediation issues
        are properly tracked even though they bypass the main run_loop.

        Args:
            issue_ids: List of remediation issue IDs to execute.
            run_metadata: Run metadata for recording issue results.
        """
        if not issue_ids:
            return

        # Track (issue_id, task) pairs for finalization
        task_pairs: list[tuple[str, asyncio.Task[IssueResult]]] = []

        for issue_id in issue_ids:
            # Skip if already failed (remediation issues are freshly created, so won't be completed)
            if issue_id in self.failed_issues:
                continue

            # Spawn agent for this issue - use returned task directly
            # (spawn_agent returns the task but doesn't register it to active_tasks)
            task = await self.spawn_agent(issue_id)
            if task:
                task_pairs.append((issue_id, task))

        # Wait for all remediation tasks to complete
        if not task_pairs:
            return

        tasks = [pair[1] for pair in task_pairs]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Finalize each task result (close issue, record metadata, emit events)
        for issue_id, task in task_pairs:
            try:
                result = task.result()
            except asyncio.CancelledError:
                result = IssueResult(
                    issue_id=issue_id,
                    agent_id=self.agent_ids.get(issue_id, "unknown"),
                    success=False,
                    summary="Remediation task was cancelled",
                )
            except Exception as e:
                result = IssueResult(
                    issue_id=issue_id,
                    agent_id=self.agent_ids.get(issue_id, "unknown"),
                    success=False,
                    summary=str(e),
                )

            # Finalize (closes issue, records to run_metadata, emits events)
            # Wrap in try/except to ensure all issues are finalized even if one fails
            try:
                await self._finalize_issue_result(issue_id, result, run_metadata)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.event_sink.on_warning(
                    f"Failed to finalize remediation result for {issue_id}: {e}",
                    agent_id=issue_id,
                )

            # Mark as completed in the coordinator
            self.issue_coordinator.mark_completed(issue_id)

    def _record_issue_run(
        self,
        issue_id: str,
        result: IssueResult,
        log_path: Path | None,
        gate_metadata: GateMetadata,
        run_metadata: RunMetadata,
    ) -> None:
        """Record an issue run to the run metadata.

        Args:
            issue_id: The issue ID.
            result: The issue result.
            log_path: Path to the session log (if any).
            gate_metadata: Extracted gate metadata.
            run_metadata: The run metadata to record to.
        """
        issue_run = IssueRun(
            issue_id=result.issue_id,
            agent_id=result.agent_id,
            status="success" if result.success else "failed",
            duration_seconds=result.duration_seconds,
            session_id=result.session_id,
            log_path=str(log_path) if log_path else None,
            quality_gate=gate_metadata.quality_gate_result,
            error=result.summary if not result.success else None,
            gate_attempts=result.gate_attempts,
            review_attempts=result.review_attempts,
            validation=gate_metadata.validation_result,
            resolution=result.resolution,
            review_log_path=self.review_log_paths.get(issue_id),
        )
        run_metadata.record_issue(issue_run)

    def _cleanup_session_paths(self, issue_id: str) -> None:
        """Remove stored session and review log paths for an issue.

        Args:
            issue_id: The issue ID to clean up paths for.
        """
        self.session_log_paths.pop(issue_id, None)
        self.review_log_paths.pop(issue_id, None)

    async def _emit_completion(
        self,
        issue_id: str,
        result: IssueResult,
        log_path: Path | None,
    ) -> None:
        """Emit completion event and handle failure tracking.

        Args:
            issue_id: The issue ID.
            result: The issue result.
            log_path: Path to the session log (if any).
        """
        self.event_sink.on_issue_completed(
            issue_id,
            issue_id,
            result.success,
            result.duration_seconds,
            truncate_text(result.summary, 50) if result.success else result.summary,
        )
        if not result.success:
            self.failed_issues.add(issue_id)
            await self.beads.mark_needs_followup_async(
                issue_id, result.summary, log_path=log_path
            )

    async def _finalize_issue_result(
        self,
        issue_id: str,
        result: IssueResult,
        run_metadata: RunMetadata,
    ) -> None:
        """Record an issue result, update metadata, and emit logs.

        Uses stored gate result to derive metadata, avoiding duplicate
        validation parsing. Gate result includes validation_evidence
        already parsed during quality gate check.
        """
        log_path = self.session_log_paths.get(issue_id)
        stored_gate_result = self.last_gate_results.get(issue_id)

        # Build gate metadata from stored result or logs
        if stored_gate_result is not None:
            gate_metadata = build_gate_metadata(stored_gate_result, result.success)
        elif not result.success and log_path and log_path.exists():
            gate_metadata = build_gate_metadata_from_logs(
                log_path,
                result.summary,
                result.success,
                self.quality_gate,
                self.per_issue_spec,
            )
        else:
            gate_metadata = GateMetadata()

        # Handle successful issue closure and epic verification
        if result.success and await self.beads.close_async(issue_id):
            self.event_sink.on_issue_closed(issue_id, issue_id)
            await self._check_epic_closure(issue_id, run_metadata)

            # Create tracking issues for P2/P3 review findings (if enabled)
            if self._config.track_review_issues and result.low_priority_review_issues:
                await create_review_tracking_issues(
                    self.beads,
                    self.event_sink,
                    issue_id,
                    result.low_priority_review_issues,
                )

        # Update tracking state (active_tasks is updated by mark_completed in finalize_callback)
        self.completed.append(result)

        # Record to run metadata and cleanup
        self._record_issue_run(issue_id, result, log_path, gate_metadata, run_metadata)
        self._cleanup_session_paths(issue_id)
        await self._emit_completion(issue_id, result, log_path)

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

        Args:
            issue_id: The issue ID for tracking state.

        Returns:
            SessionCallbacks with gate, review, and logging callbacks.
        """

        async def on_gate_check(
            issue_id: str, log_path: Path, retry_state: RetryState
        ) -> tuple[GateResult | GateResultProtocol, int]:
            return await self._run_quality_gate_async(issue_id, log_path, retry_state)

        async def on_review_check(
            issue_id: str,
            issue_desc: str | None,
            baseline: str | None,
            session_id: str | None,
            retry_state: RetryState,
        ) -> ReviewResult | ReviewResultProtocol:
            current_head = await get_git_commit_async(self.repo_path)
            self.review_runner.config.capture_session_log = is_verbose_enabled()
            commit_shas = await get_issue_commits_async(
                self.repo_path,
                issue_id,
                since_timestamp=retry_state.baseline_timestamp,
            )
            review_input = ReviewInput(
                issue_id=issue_id,
                repo_path=self.repo_path,
                commit_sha=current_head,
                issue_description=issue_desc,
                baseline_commit=baseline,
                commit_shas=commit_shas or None,
                claude_session_id=session_id,
            )
            output = await self.review_runner.run_review(review_input)
            if output.session_log_path:
                self.review_log_paths[issue_id] = output.session_log_path
            return output.result

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
                spec=self.per_issue_spec,
            )
            return self.review_runner.check_no_progress(no_progress_input)

        def get_log_path(session_id: str) -> Path:
            log_path = self.log_provider.get_log_path(self.repo_path, session_id)
            self.session_log_paths[issue_id] = log_path
            return log_path

        def get_log_offset(log_path: Path, start_offset: int) -> int:
            return self.quality_gate.get_log_end_offset(log_path, start_offset)

        def on_tool_use(agent_id: str, tool_name: str, arguments: dict | None) -> None:
            self.event_sink.on_tool_use(agent_id, tool_name, arguments=arguments)

        def on_agent_text(agent_id: str, text: str) -> None:
            self.event_sink.on_agent_text(agent_id, text)

        return SessionCallbacks(
            on_gate_check=on_gate_check,
            on_review_check=on_review_check,
            on_review_no_progress=on_review_no_progress,
            get_log_path=get_log_path,
            get_log_offset=get_log_offset,
            on_abort=self._request_abort,
            on_tool_use=on_tool_use,
            on_agent_text=on_agent_text,
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

        prompt = get_implementer_prompt().format(
            issue_id=issue_id,
            repo_path=self.repo_path,
            lock_dir=get_lock_dir(),
            scripts_dir=SCRIPTS_DIR,
            agent_id=temp_agent_id,
        )

        review_enabled = self._is_review_enabled()
        session_config = AgentSessionConfig(
            repo_path=self.repo_path,
            timeout_seconds=self.timeout_seconds,
            max_gate_retries=self.max_gate_retries,
            max_review_retries=self.max_review_retries,
            morph_enabled=self.morph_enabled,
            morph_api_key=self._config.morph_api_key,
            review_enabled=review_enabled,
        )
        session_input = AgentSessionInput(
            issue_id=issue_id,
            prompt=prompt,
            baseline_commit=baseline_commit,
            issue_description=issue_description,
        )

        runner = AgentSessionRunner(
            config=session_config,
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
            agent_id = self.agent_ids.pop(issue_id, temp_agent_id)
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
            morph_enabled=self.morph_enabled,
            prioritize_wip=self.prioritize_wip,
            orphans_only=self.orphans_only,
            cli_args=self.cli_args,
            mala_config=self._mala_config,
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
        self.per_issue_spec = build_validation_spec(
            scope=ValidationScope.PER_ISSUE,
            disable_validations=self._disabled_validations,
            coverage_threshold=self.coverage_threshold,
            repo_path=self.repo_path,
        )
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

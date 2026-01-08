"""OrchestrationWiring: Pipeline component initialization and wiring.

This module extracts pipeline component creation from MalaOrchestrator,
providing a clean separation between wiring logic and runtime orchestration.

The wiring module handles:
- Creating and configuring pipeline runners (GateRunner, ReviewRunner, etc.)
- Building callback structures for coordinators
- Initializing the session callback factory

Design principles:
- Pure initialization logic, no runtime state
- All dependencies passed explicitly
- Returns fully configured components
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.infra.io.log_output.console import is_verbose_enabled
from src.pipeline.agent_session_runner import AgentSessionConfig, SessionPrompts
from src.pipeline.gate_runner import (
    AsyncGateRunner,
    GateRunner,
    GateRunnerConfig,
)
from src.pipeline.issue_finalizer import IssueFinalizeCallbacks
from src.pipeline.epic_verification_coordinator import EpicVerificationCallbacks
from src.pipeline.review_runner import (
    ReviewRunner,
    ReviewRunnerConfig,
)
from src.pipeline.session_callback_factory import SessionCallbackFactory
from src.pipeline.issue_execution_coordinator import (
    CoordinatorConfig,
    IssueExecutionCoordinator,
)
from src.pipeline.run_coordinator import (
    RunCoordinator,
    RunCoordinatorConfig,
)

if TYPE_CHECKING:
    import asyncio

    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from src.core.protocols import (
        ReviewIssueProtocol,
        SDKClientFactoryProtocol,
    )
    from src.core.models import EpicVerificationResult
    from src.infra.io.log_output.run_metadata import RunMetadata
    from src.orchestration.types import (
        IssueFilterConfig,
        PipelineConfig,
        RuntimeDeps,
    )
    from src.pipeline.issue_result import IssueResult


def build_gate_runner(
    runtime: RuntimeDeps, pipeline: PipelineConfig
) -> tuple[GateRunner, AsyncGateRunner]:
    """Build GateRunner and AsyncGateRunner."""
    config = GateRunnerConfig(
        max_gate_retries=pipeline.max_gate_retries,
        disable_validations=pipeline.disabled_validations,
        validation_config=pipeline.validation_config,
        validation_config_missing=pipeline.validation_config_missing,
    )
    gate_runner = GateRunner(
        gate_checker=runtime.evidence_check,
        repo_path=pipeline.repo_path,
        config=config,
    )
    async_gate_runner = AsyncGateRunner(gate_runner=gate_runner)
    return gate_runner, async_gate_runner


def build_review_runner(runtime: RuntimeDeps, pipeline: PipelineConfig) -> ReviewRunner:
    """Build ReviewRunner."""
    config = ReviewRunnerConfig(
        max_review_retries=pipeline.max_review_retries,
        capture_session_log=False,
        review_timeout=runtime.mala_config.review_timeout,
    )
    return ReviewRunner(
        code_reviewer=runtime.code_reviewer,
        config=config,
        gate_checker=runtime.evidence_check,
    )


def build_run_coordinator(
    runtime: RuntimeDeps,
    pipeline: PipelineConfig,
    sdk_client_factory: SDKClientFactoryProtocol,
    mcp_server_factory: Callable | None = None,
) -> RunCoordinator:
    """Build RunCoordinator."""
    config = RunCoordinatorConfig(
        repo_path=pipeline.repo_path,
        timeout_seconds=pipeline.timeout_seconds,
        max_gate_retries=pipeline.max_gate_retries,
        disable_validations=pipeline.disabled_validations,
        fixer_prompt=pipeline.prompts.fixer_prompt,
        mcp_server_factory=mcp_server_factory,
        validation_config=pipeline.validation_config,
        validation_config_missing=pipeline.validation_config_missing,
    )
    return RunCoordinator(
        config=config,
        gate_checker=runtime.evidence_check,
        command_runner=runtime.command_runner,
        env_config=runtime.env_config,
        lock_manager=runtime.lock_manager,
        sdk_client_factory=sdk_client_factory,
        event_sink=runtime.event_sink,
    )


def build_issue_coordinator(
    filters: IssueFilterConfig, runtime: RuntimeDeps
) -> IssueExecutionCoordinator:
    """Build IssueExecutionCoordinator."""
    config = CoordinatorConfig(
        max_agents=filters.max_agents,
        max_issues=filters.max_issues,
        epic_id=filters.epic_id,
        only_ids=filters.only_ids,
        prioritize_wip=filters.prioritize_wip,
        focus=filters.focus,
        orphans_only=filters.orphans_only,
        order_preference=filters.order_preference,
    )
    return IssueExecutionCoordinator(
        beads=runtime.beads,
        event_sink=runtime.event_sink,
        config=config,
    )


@dataclass
class FinalizerCallbackRefs:
    """References for building finalizer callbacks.

    These are callable getters that allow late binding to orchestrator state.
    """

    close_issue: Callable[[str], Awaitable[bool]]
    mark_needs_followup: Callable[[str, str, Path | None], Awaitable[bool]]
    on_issue_closed: Callable[[str, str], None]
    on_issue_completed: Callable[[str, str, bool, float, str], None]
    trigger_epic_closure: Callable[[str, RunMetadata], Awaitable[None]]
    create_tracking_issues: Callable[[str, list[ReviewIssueProtocol]], Awaitable[None]]


def build_finalizer_callbacks(refs: FinalizerCallbackRefs) -> IssueFinalizeCallbacks:
    """Build IssueFinalizeCallbacks from callback references."""
    return IssueFinalizeCallbacks(
        close_issue=refs.close_issue,
        mark_needs_followup=refs.mark_needs_followup,
        on_issue_closed=refs.on_issue_closed,
        on_issue_completed=refs.on_issue_completed,
        trigger_epic_closure=refs.trigger_epic_closure,
        create_tracking_issues=refs.create_tracking_issues,
    )


@dataclass
class EpicCallbackRefs:
    """References for building epic verification callbacks."""

    get_parent_epic: Callable[[str], Awaitable[str | None]]
    verify_epic: Callable[[str, bool], Awaitable[EpicVerificationResult]]
    spawn_remediation: Callable[[str, str], Awaitable[asyncio.Task[IssueResult] | None]]
    finalize_remediation: Callable[[str, IssueResult, RunMetadata], Awaitable[None]]
    mark_completed: Callable[[str], None]
    is_issue_failed: Callable[[str], bool]
    close_eligible_epics: Callable[[], Awaitable[bool]]
    on_epic_closed: Callable[[str], None]
    on_warning: Callable[[str], None]
    has_epic_verifier: Callable[[], bool]
    get_agent_id: Callable[[str], str]


def build_epic_callbacks(refs: EpicCallbackRefs) -> EpicVerificationCallbacks:
    """Build EpicVerificationCallbacks from callback references."""
    return EpicVerificationCallbacks(
        get_parent_epic=refs.get_parent_epic,
        verify_epic=refs.verify_epic,
        spawn_remediation=refs.spawn_remediation,
        finalize_remediation=refs.finalize_remediation,
        mark_completed=refs.mark_completed,
        is_issue_failed=refs.is_issue_failed,
        close_eligible_epics=refs.close_eligible_epics,
        on_epic_closed=refs.on_epic_closed,
        on_warning=refs.on_warning,
        has_epic_verifier=refs.has_epic_verifier,
        get_agent_id=refs.get_agent_id,
    )


def build_session_callback_factory(
    runtime: RuntimeDeps,
    pipeline: PipelineConfig,
    async_gate_runner: AsyncGateRunner,
    review_runner: ReviewRunner,
    log_provider_getter: Callable,
    evidence_check_getter: Callable,
    on_session_log_path: Callable[[str, Path], None],
    on_review_log_path: Callable[[str, str], None],
) -> SessionCallbackFactory:
    """Build SessionCallbackFactory."""
    return SessionCallbackFactory(
        gate_async_runner=async_gate_runner,
        review_runner=review_runner,
        log_provider=log_provider_getter,
        event_sink=lambda: runtime.event_sink,
        evidence_check=evidence_check_getter,
        repo_path=pipeline.repo_path,
        on_session_log_path=on_session_log_path,
        on_review_log_path=on_review_log_path,
        get_per_session_spec=lambda: async_gate_runner.per_session_spec,
        is_verbose=is_verbose_enabled,
    )


def build_session_config(
    pipeline: PipelineConfig,
    review_enabled: bool,
    mcp_server_factory: Callable | None = None,
    strict_resume: bool = False,
) -> AgentSessionConfig:
    """Build AgentSessionConfig for agent sessions."""
    prompts = SessionPrompts(
        gate_followup=pipeline.prompts.gate_followup_prompt,
        review_followup=pipeline.prompts.review_followup_prompt,
        idle_resume=pipeline.prompts.idle_resume_prompt,
        checkpoint_request=pipeline.prompts.checkpoint_request_prompt,
        continuation=pipeline.prompts.continuation_prompt,
    )
    return AgentSessionConfig(
        repo_path=pipeline.repo_path,
        timeout_seconds=pipeline.timeout_seconds,
        prompts=prompts,
        max_gate_retries=pipeline.max_gate_retries,
        max_review_retries=pipeline.max_review_retries,
        review_enabled=review_enabled,
        lint_tools=None,  # Set at run start
        prompt_validation_commands=pipeline.prompt_validation_commands,
        context_restart_threshold=pipeline.context_restart_threshold,
        context_limit=pipeline.context_limit,
        deadlock_monitor=pipeline.deadlock_monitor,
        mcp_server_factory=mcp_server_factory,
        strict_resume=strict_resume,
    )


def create_mcp_server_factory() -> Callable[
    [str, Path, Callable | None], dict[str, object]
]:
    """Create a factory function for MCP server configuration.

    This factory is injected into AgentRuntimeBuilder to avoid having
    the builder import SDK-dependent code directly.

    Returns:
        A factory function that creates MCP server configurations.
    """
    from src.infra.tools.locking_mcp import create_locking_mcp_server

    def factory(
        agent_id: str,
        repo_path: Path,
        emit_lock_event: Callable | None,
    ) -> dict[str, object]:
        """Create MCP servers for an agent."""

        # No-op handler if emit_lock_event is None
        def _noop_handler(event: object) -> None:
            pass

        return {
            "mala-locking": create_locking_mcp_server(
                agent_id=agent_id,
                repo_namespace=str(repo_path),
                emit_lock_event=emit_lock_event or _noop_handler,
            )
        }

    return factory

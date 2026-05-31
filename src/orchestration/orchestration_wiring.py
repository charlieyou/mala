"""OrchestrationWiring: Pipeline component initialization and wiring.

This module extracts pipeline component creation from MalaOrchestrator,
providing a clean separation between wiring logic and runtime orchestration.

The wiring module handles:
- Creating and configuring pipeline runners (GateRunner, ReviewRunner, etc.)
- Initializing the session callback factory

Design principles:
- Pure initialization logic, no runtime state
- All dependencies passed explicitly
- Returns fully configured components
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.domain.validation.config_types import LongRunningConfig
from src.infra.io.log_output.console import is_verbose_enabled
from src.pipeline.agent_session_runner import AgentSessionConfig, SessionPrompts
from src.pipeline.gate_runner import (
    AsyncGateRunner,
    GateRunner,
)
from src.pipeline.review_runner import ReviewRunner
from src.pipeline.session_callback_factory import SessionCallbackFactory
from src.pipeline.issue_execution_coordinator import (
    CoordinatorConfig,
    IssueExecutionCoordinator,
)
from src.pipeline.run_coordinator import RunCoordinator
from src.pipeline.cumulative_review_runner import CumulativeReviewRunner
from src.pipeline.trigger_engine import TriggerEngine
from src.pipeline.fixer_service import FixerService

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.orchestration.runtime_deps import RuntimeDeps
    from src.orchestration.types import (
        IssueFilterConfig,
        PipelineConfig,
    )
    from src.pipeline.session_callback_factory import SessionRunContext


def build_gate_runner(
    runtime: RuntimeDeps, pipeline: PipelineConfig
) -> tuple[GateRunner, AsyncGateRunner]:
    """Build GateRunner and AsyncGateRunner."""
    gate_runner = GateRunner(
        gate_checker=runtime.evidence_check,
        view=pipeline.gate_runner_view,
    )
    async_gate_runner = AsyncGateRunner(gate_runner=gate_runner)
    return gate_runner, async_gate_runner


def build_review_runner(runtime: RuntimeDeps, pipeline: PipelineConfig) -> ReviewRunner:
    """Build ReviewRunner."""
    return ReviewRunner(
        code_reviewer=runtime.code_reviewer,
        view=pipeline.review_runner_view,
        gate_checker=runtime.evidence_check,
    )


def build_cumulative_review_runner(
    runtime: RuntimeDeps,
    pipeline: PipelineConfig,
) -> CumulativeReviewRunner:
    """Build CumulativeReviewRunner for code_review in triggers."""
    import logging

    from src.infra.git_utils import GitUtils

    # Build ReviewRunner for code reviews
    review_runner = ReviewRunner(
        code_reviewer=runtime.code_reviewer,
        view=pipeline.review_runner_view,
        gate_checker=runtime.evidence_check,
    )

    # Build GitUtils wrapper
    git_utils = GitUtils(repo_path=pipeline.repo_path)

    return CumulativeReviewRunner(
        review_runner=review_runner,
        git_utils=git_utils,
        beads_client=runtime.beads,
        logger=logging.getLogger("mala.cumulative_review"),
    )


def build_run_coordinator(
    runtime: RuntimeDeps,
    pipeline: PipelineConfig,
    mcp_server_factory: Callable | None = None,
) -> RunCoordinator:
    """Build RunCoordinator.

    The :class:`RunCoordinator` and the :class:`FixerService` it spawns both
    consume the run's :class:`AgentProvider` (carried on ``runtime``). This is
    the wiring point that makes fixer agents follow the main coder: when
    ``coder=amp``, ``runtime.agent_provider`` is the ``AmpAgentProvider``, so
    fixer subprocesses are also spawned via Amp (per plan L165, AC#5).
    """
    # Build TriggerEngine for trigger policy evaluation
    trigger_engine = TriggerEngine(pipeline.validation_config)

    # Build FixerService for spawning fixer agents - same provider as main run
    # so fixers spawn whichever coder the main run uses (plan L165, AC#5).
    fixer_service = FixerService(
        view=pipeline.fixer_service_view,
        agent_provider=runtime.agent_provider,
        mcp_server_factory=mcp_server_factory,
        event_sink=runtime.event_sink,
    )

    # Build CumulativeReviewRunner for code_review in triggers
    cumulative_review_runner = build_cumulative_review_runner(runtime, pipeline)

    return RunCoordinator(
        view=pipeline.run_coordinator_view,
        gate_checker=runtime.evidence_check,
        command_runner=runtime.command_runner,
        env_config=runtime.env_config,
        lock_manager=runtime.lock_manager,
        agent_provider=runtime.agent_provider,
        trigger_engine=trigger_engine,
        fixer_service=fixer_service,
        event_sink=runtime.event_sink,
        cumulative_review_runner=cumulative_review_runner,
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
        include_wip=filters.include_wip,
        focus=filters.focus,
        orphans_only=filters.orphans_only,
        order_preference=filters.order_preference,
        prioritize_wip=filters.prioritize_wip,
    )
    return IssueExecutionCoordinator(
        beads=runtime.beads,
        event_sink=runtime.event_sink,
        config=config,
    )


def build_session_callback_factory(
    runtime: RuntimeDeps,
    pipeline: PipelineConfig,
    async_gate_runner: AsyncGateRunner,
    review_runner: ReviewRunner,
    context: SessionRunContext,
    cumulative_review_runner: CumulativeReviewRunner | None = None,
) -> SessionCallbackFactory:
    """Build SessionCallbackFactory.

    Args:
        runtime: Runtime dependencies.
        pipeline: Pipeline configuration.
        async_gate_runner: Async gate runner for quality checks.
        review_runner: Review runner for code reviews.
        context: Session run context with late-bound getters.
        cumulative_review_runner: Runner for session_end code review.

    Returns:
        Configured SessionCallbackFactory.
    """
    return SessionCallbackFactory(
        gate_async_runner=async_gate_runner,
        review_runner=review_runner,
        context=context,
        event_sink=lambda: runtime.event_sink,
        repo_path=pipeline.repo_path,
        get_per_session_spec=lambda: async_gate_runner.per_session_spec,
        is_verbose=is_verbose_enabled,
        get_validation_config=lambda: pipeline.validation_config,
        command_runner=runtime.command_runner,
        cumulative_review_runner=cumulative_review_runner,
    )


def build_session_config(
    pipeline: PipelineConfig,
    review_enabled: bool,
    mcp_server_factory: Callable | None = None,
    strict_resume: bool = False,
) -> AgentSessionConfig:
    """Build AgentSessionConfig for agent sessions.

    Returns only the runner-specific fields. Canonical PipelineConfig fields
    (repo_path, timeout_seconds, retry limits, etc.) are read at runtime from
    ``pipeline.agent_session_view``.
    """
    prompts = SessionPrompts(
        gate_followup=pipeline.prompts.gate_followup_prompt,
        review_followup=pipeline.prompts.review_followup_prompt,
        idle_resume=pipeline.prompts.idle_resume_prompt,
        checkpoint_request=pipeline.prompts.checkpoint_request_prompt,
        continuation=pipeline.prompts.continuation_prompt,
        await_resume=pipeline.prompts.await_resume_prompt,
    )
    # Default to enabled (LongRunningConfig defaults) when no validation config
    # is present, so repos without a mala.yaml still get the background
    # wait/resume behavior the implementer prompt advertises rather than gating
    # immediately after a backgrounded yield.
    long_running = (
        pipeline.validation_config.long_running
        if pipeline.validation_config is not None
        else LongRunningConfig()
    )
    return AgentSessionConfig(
        prompts=prompts,
        review_enabled=review_enabled,
        lint_tools=None,  # Set at run start
        mcp_server_factory=mcp_server_factory,
        strict_resume=strict_resume,
        long_running=long_running,
    )

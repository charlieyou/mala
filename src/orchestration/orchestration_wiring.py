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

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.infra.io.log_output.console import is_verbose_enabled
from src.pipeline.agent_session_runner import AgentSessionConfig, PromptProvider
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
from src.domain.prompts import (
    get_fixer_prompt,
    get_gate_followup_prompt,
    get_idle_resume_prompt,
    get_review_followup_prompt,
)

if TYPE_CHECKING:
    import asyncio

    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from src.core.protocols import (
        CodeReviewer,
        GateChecker,
        IssueProvider,
        ReviewIssueProtocol,
    )
    from src.domain.prompts import PromptValidationCommands
    from src.infra.epic_verifier import EpicVerificationResult
    from src.infra.io.config import MalaConfig
    from src.infra.io.event_protocol import MalaEventSink
    from src.infra.io.log_output.run_metadata import RunMetadata
    from src.orchestration.issue_result import IssueResult


@dataclass
class WiringDependencies:
    """Dependencies required for pipeline component wiring.

    All dependencies are provided by the orchestrator factory and passed
    through to the wiring functions.
    """

    repo_path: Path
    quality_gate: GateChecker
    code_reviewer: CodeReviewer
    beads: IssueProvider
    event_sink: MalaEventSink
    mala_config: MalaConfig
    # Config values
    max_agents: int | None
    max_issues: int | None
    timeout_seconds: int
    max_gate_retries: int
    max_review_retries: int
    coverage_threshold: float | None
    disabled_validations: set[str] | None
    epic_id: str | None
    only_ids: set[str] | None
    prioritize_wip: bool
    focus: bool
    orphans_only: bool
    epic_override_ids: set[str]
    prompt_validation_commands: PromptValidationCommands
    # Mutable state references (shared with orchestrator)
    session_log_paths: dict[str, Path]
    review_log_paths: dict[str, str]


def build_gate_runner(deps: WiringDependencies) -> tuple[GateRunner, AsyncGateRunner]:
    """Build GateRunner and AsyncGateRunner."""
    config = GateRunnerConfig(
        max_gate_retries=deps.max_gate_retries,
        disable_validations=deps.disabled_validations,
        coverage_threshold=deps.coverage_threshold,
    )
    gate_runner = GateRunner(
        gate_checker=deps.quality_gate,
        repo_path=deps.repo_path,
        config=config,
    )
    async_gate_runner = AsyncGateRunner(gate_runner=gate_runner)
    return gate_runner, async_gate_runner


def build_review_runner(deps: WiringDependencies) -> ReviewRunner:
    """Build ReviewRunner."""
    config = ReviewRunnerConfig(
        max_review_retries=deps.max_review_retries,
        capture_session_log=False,
        review_timeout=deps.mala_config.review_timeout,
    )
    return ReviewRunner(
        code_reviewer=deps.code_reviewer,
        config=config,
        gate_checker=deps.quality_gate,
    )


def build_run_coordinator(deps: WiringDependencies) -> RunCoordinator:
    """Build RunCoordinator."""
    config = RunCoordinatorConfig(
        repo_path=deps.repo_path,
        timeout_seconds=deps.timeout_seconds,
        max_gate_retries=deps.max_gate_retries,
        disable_validations=deps.disabled_validations,
        coverage_threshold=deps.coverage_threshold,
        fixer_prompt=get_fixer_prompt(),
    )
    return RunCoordinator(
        config=config,
        gate_checker=deps.quality_gate,
        event_sink=deps.event_sink,
    )


def build_issue_coordinator(deps: WiringDependencies) -> IssueExecutionCoordinator:
    """Build IssueExecutionCoordinator."""
    config = CoordinatorConfig(
        max_agents=deps.max_agents,
        max_issues=deps.max_issues,
        epic_id=deps.epic_id,
        only_ids=deps.only_ids,
        prioritize_wip=deps.prioritize_wip,
        focus=deps.focus,
        orphans_only=deps.orphans_only,
    )
    return IssueExecutionCoordinator(
        beads=deps.beads,
        event_sink=deps.event_sink,
        config=config,
    )


@dataclass
class FinalizerCallbackRefs:
    """References for building finalizer callbacks.

    These are callable getters that allow late binding to orchestrator state.
    """

    close_issue: Callable[[str], Awaitable[bool]]
    mark_needs_followup: Callable[[str, str, Path | None], Awaitable[None]]
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
    spawn_remediation: Callable[[str], Awaitable[asyncio.Task[IssueResult] | None]]
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
    deps: WiringDependencies,
    async_gate_runner: AsyncGateRunner,
    review_runner: ReviewRunner,
    log_provider_getter: Callable,
    quality_gate_getter: Callable,
) -> SessionCallbackFactory:
    """Build SessionCallbackFactory."""
    return SessionCallbackFactory(
        gate_async_runner=async_gate_runner,
        review_runner=review_runner,
        log_provider=log_provider_getter,
        event_sink=lambda: deps.event_sink,
        quality_gate=quality_gate_getter,
        repo_path=deps.repo_path,
        session_log_paths=deps.session_log_paths,
        review_log_paths=deps.review_log_paths,
        get_per_issue_spec=lambda: async_gate_runner.per_issue_spec,
        is_verbose=is_verbose_enabled,
    )


def build_session_config(
    deps: WiringDependencies,
    review_enabled: bool,
) -> AgentSessionConfig:
    """Build AgentSessionConfig for agent sessions."""
    prompts = PromptProvider(
        gate_followup=get_gate_followup_prompt(),
        review_followup=get_review_followup_prompt(),
        idle_resume=get_idle_resume_prompt(),
    )
    return AgentSessionConfig(
        repo_path=deps.repo_path,
        timeout_seconds=deps.timeout_seconds,
        prompts=prompts,
        max_gate_retries=deps.max_gate_retries,
        max_review_retries=deps.max_review_retries,
        review_enabled=review_enabled,
        lint_tools=None,  # Set at run start
        prompt_validation_commands=deps.prompt_validation_commands,
    )

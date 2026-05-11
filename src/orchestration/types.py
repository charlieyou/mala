"""Shared types for orchestrator components.

This module contains dataclasses and constants shared between
orchestrator.py and orchestrator_factory.py to break circular imports.

Design principles:
- OrchestratorConfig: All scalar configuration (timeouts, flags, limits)
- OrchestratorDependencies: All protocol implementations (DI for testability)
- _DerivedConfig: Internal computed configuration values
- PipelineConfig: Configuration for pipeline stages
- IssueFilterConfig: Filtering criteria for issue selection
- *View: Narrow cached views derived from PipelineConfig (one per runner) so
  runner wiring reads canonical fields without duplicating state.
- DEFAULT_AGENT_TIMEOUT_MINUTES: Default timeout constant

RuntimeDeps lives in `src/orchestration/runtime_deps.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path  # noqa: TC003 - needed at runtime for dataclass field
from typing import TYPE_CHECKING

# Private import for runtime use in dataclass defaults (not re-exported)
from src.core.models import OrderPreference as _OrderPreference

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.core.models import OrderPreference
    from src.core.protocols.agent_provider import AgentProvider
    from src.core.protocols.evidence import EvidenceProvider
    from src.core.protocols.events import MalaEventSink
    from src.core.protocols.infra import (
        CommandRunnerPort,
        EnvConfigPort,
        LockManagerPort,
    )
    from src.core.protocols.issue import IssueProvider
    from src.core.protocols.review import CodeReviewer
    from src.core.protocols.validation import GateChecker
    from src.domain.deadlock import DeadlockMonitor
    from src.domain.prompts import PromptProvider
    from src.domain.validation.config_types import (
        CodeReviewConfig,
        PromptValidationCommands,
        ValidationConfig,
    )
    from src.infra.telemetry import TelemetryProvider

# Default timeout for agent execution (protects against hung MCP server subprocesses)
DEFAULT_AGENT_TIMEOUT_MINUTES = 30

# Default idle timeout retry configuration
DEFAULT_MAX_IDLE_RETRIES = 2


@dataclass
class OrchestratorConfig:
    """Configuration for MalaOrchestrator.

    All scalar configuration values that control orchestrator behavior.
    These are typically derived from CLI arguments or environment.

    Attributes:
        repo_path: Path to the repository with beads issues.
        max_agents: Maximum concurrent agents (None = unlimited).
        timeout_minutes: Timeout per agent in minutes (None = default 30).
        max_issues: Maximum issues to process (None = unlimited).
        epic_id: Only process tasks under this epic.
        only_ids: List of issue IDs to process exclusively.
        max_gate_retries: Maximum quality gate retry attempts per issue.
        max_review_retries: Maximum code review retry attempts per issue.
        disable_validations: Set of validation types to disable.
        include_wip: Include in_progress issues in scope (no ordering changes).
        focus: Only work on one epic at a time.
        order_preference: Issue ordering preference (focus, epic-priority, issue-priority, or input).
        cli_args: CLI arguments for logging and metadata.
        epic_override_ids: Epic IDs to close without verification.
        orphans_only: Only process issues with no parent epic.
    """

    repo_path: Path
    max_agents: int | None = None
    timeout_minutes: int | None = None
    max_issues: int | None = None
    epic_id: str | None = None
    only_ids: list[str] | None = None
    max_gate_retries: int = 3
    max_review_retries: int = 3
    disable_validations: set[str] | None = None
    include_wip: bool = False
    focus: bool = True
    order_preference: OrderPreference = _OrderPreference.EPIC_PRIORITY
    cli_args: dict[str, object] | None = None
    epic_override_ids: set[str] = field(default_factory=set)
    orphans_only: bool = False
    # Session resume strict mode: fail issue if no prior session found
    strict_resume: bool = False
    # Fresh session mode: start new SDK session instead of resuming (requires --resume)
    fresh_session: bool = False


@dataclass
class OrchestratorDependencies:
    """Protocol implementations for MalaOrchestrator.

    All injected dependencies that implement the orchestrator's protocols.
    When None, the factory creates default implementations.

    Note: AgentSessionRunner is NOT included here because it's constructed
    per-session in run_implementer with issue-specific callbacks.

    Attributes:
        issue_provider: IssueProvider for issue tracking operations.
        code_reviewer: CodeReviewer for post-commit code reviews.
        gate_checker: GateChecker for quality gate validation.
        evidence_provider: EvidenceProvider for session log access.
        telemetry_provider: TelemetryProvider for tracing.
        event_sink: MalaEventSink for run lifecycle logging.
        agent_provider: AgentProvider bundling client_factory + runtime_builder
            + evidence_provider for the chosen coder backend (Claude or Amp).
            Selected once at orchestrator construction in
            ``src/orchestration/factory.py`` and threaded into
            AgentSessionRunner / FixerService / RunCoordinator.
        command_runner: CommandRunnerPort for executing shell commands.
        env_config: EnvConfigPort for environment configuration.
        lock_manager: LockManagerPort for file locking coordination.
        runs_dir: Directory for run markers (for testing).
        lock_releaser: Function to release locks (for testing).
    """

    issue_provider: IssueProvider | None = None
    code_reviewer: CodeReviewer | None = None
    gate_checker: GateChecker | None = None
    evidence_provider: EvidenceProvider | None = None
    telemetry_provider: TelemetryProvider | None = None
    event_sink: MalaEventSink | None = None
    agent_provider: AgentProvider | None = None
    command_runner: CommandRunnerPort | None = None
    env_config: EnvConfigPort | None = None
    lock_manager: LockManagerPort | None = None
    runs_dir: Path | None = None
    lock_releaser: Callable[[list[str]], int] | None = None


@dataclass
class _DerivedConfig:
    """Derived configuration values computed from OrchestratorConfig and MalaConfig.

    Internal class used to pass computed values to the orchestrator.
    """

    timeout_seconds: int
    disabled_validations: set[str]
    max_idle_retries: int
    idle_timeout_seconds: float | None
    max_gate_retries: int | None = None
    max_review_retries: int | None = None
    review_timeout_seconds: int | None = None
    max_epic_verification_retries: int | None = None
    max_diff_size_kb: int | None = None
    epic_verify_lock_timeout_seconds: int | None = None
    review_disabled_reason: str | None = None
    per_issue_review: CodeReviewConfig | None = None
    validation_config: ValidationConfig | None = None
    validation_config_missing: bool = False


@dataclass(frozen=True)
class GateRunnerView:
    """Narrow view of PipelineConfig for gate runner wiring.

    Holds exactly the PipelineConfig fields the GateRunner needs. Produced by
    ``PipelineConfig.gate_runner_view``; do not construct directly outside the
    cached_property getter so the source remains canonical.
    """

    repo_path: Path
    max_gate_retries: int
    disabled_validations: set[str] | None
    validation_config: ValidationConfig | None
    validation_config_missing: bool


@dataclass(frozen=True)
class ReviewRunnerView:
    """Narrow view of PipelineConfig for review runner wiring."""

    max_review_retries: int
    review_timeout_seconds: int | None


@dataclass(frozen=True)
class CumulativeReviewView:
    """Narrow view of PipelineConfig for cumulative review runner wiring."""

    repo_path: Path
    review_timeout_seconds: int | None


@dataclass(frozen=True)
class FixerServiceView:
    """Narrow view of PipelineConfig for fixer service wiring."""

    repo_path: Path
    timeout_seconds: int
    fixer_prompt: str


@dataclass(frozen=True)
class RunCoordinatorView:
    """Narrow view of PipelineConfig for run coordinator wiring."""

    repo_path: Path
    timeout_seconds: int
    max_gate_retries: int
    disabled_validations: set[str] | None
    fixer_prompt: str
    validation_config: ValidationConfig | None
    validation_config_missing: bool


@dataclass(frozen=True)
class AgentSessionView:
    """Narrow view of PipelineConfig for agent session runner wiring."""

    repo_path: Path
    timeout_seconds: int
    prompts: PromptProvider
    max_gate_retries: int
    max_review_retries: int
    prompt_validation_commands: PromptValidationCommands
    max_idle_retries: int
    idle_timeout_seconds: float | None
    deadlock_monitor: DeadlockMonitor | None


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration values for pipeline behavior.

    Scalar configuration controlling pipeline execution.

    Attributes:
        repo_path: Path to the repository with beads issues.
        timeout_seconds: Timeout per agent in seconds.
        max_gate_retries: Maximum quality gate retry attempts per issue.
        max_review_retries: Maximum code review retry attempts per issue.
        review_timeout_seconds: Timeout for code reviews. None delegates to reviewer default.
        disabled_validations: Set of validation types to disable.
        max_idle_retries: Maximum number of idle timeout retries.
        idle_timeout_seconds: Idle timeout for SDK stream (None = derive from timeout).
        prompts: PromptProvider with loaded prompt templates.
        prompt_validation_commands: Validation commands for prompt substitution.
        validation_config: Loaded ValidationConfig from startup, if any.
        validation_config_missing: True if mala.yaml was missing at startup.
        deadlock_monitor: DeadlockMonitor for deadlock detection (None until wired).
    """

    repo_path: Path
    timeout_seconds: int
    max_gate_retries: int
    max_review_retries: int
    disabled_validations: set[str] | None
    max_idle_retries: int
    idle_timeout_seconds: float | None
    prompts: PromptProvider
    prompt_validation_commands: PromptValidationCommands
    validation_config: ValidationConfig | None = None
    validation_config_missing: bool = False
    deadlock_monitor: DeadlockMonitor | None = None
    review_timeout_seconds: int | None = None

    @cached_property
    def gate_runner_view(self) -> GateRunnerView:
        """Fields needed to wire a GateRunner (see build_gate_runner)."""
        return GateRunnerView(
            repo_path=self.repo_path,
            max_gate_retries=self.max_gate_retries,
            disabled_validations=self.disabled_validations,
            validation_config=self.validation_config,
            validation_config_missing=self.validation_config_missing,
        )

    @cached_property
    def review_runner_view(self) -> ReviewRunnerView:
        """Fields needed to wire a ReviewRunner (see build_review_runner)."""
        return ReviewRunnerView(
            max_review_retries=self.max_review_retries,
            review_timeout_seconds=self.review_timeout_seconds,
        )

    @cached_property
    def cumulative_review_view(self) -> CumulativeReviewView:
        """Fields needed to wire a CumulativeReviewRunner."""
        return CumulativeReviewView(
            repo_path=self.repo_path,
            review_timeout_seconds=self.review_timeout_seconds,
        )

    @cached_property
    def fixer_service_view(self) -> FixerServiceView:
        """Fields needed to wire a FixerService (see build_run_coordinator)."""
        return FixerServiceView(
            repo_path=self.repo_path,
            timeout_seconds=self.timeout_seconds,
            fixer_prompt=self.prompts.fixer_prompt,
        )

    @cached_property
    def run_coordinator_view(self) -> RunCoordinatorView:
        """Fields needed to wire a RunCoordinator (see build_run_coordinator)."""
        return RunCoordinatorView(
            repo_path=self.repo_path,
            timeout_seconds=self.timeout_seconds,
            max_gate_retries=self.max_gate_retries,
            disabled_validations=self.disabled_validations,
            fixer_prompt=self.prompts.fixer_prompt,
            validation_config=self.validation_config,
            validation_config_missing=self.validation_config_missing,
        )

    @cached_property
    def agent_session_view(self) -> AgentSessionView:
        """Fields needed to wire an AgentSessionConfig (see build_session_config)."""
        return AgentSessionView(
            repo_path=self.repo_path,
            timeout_seconds=self.timeout_seconds,
            prompts=self.prompts,
            max_gate_retries=self.max_gate_retries,
            max_review_retries=self.max_review_retries,
            prompt_validation_commands=self.prompt_validation_commands,
            max_idle_retries=self.max_idle_retries,
            idle_timeout_seconds=self.idle_timeout_seconds,
            deadlock_monitor=self.deadlock_monitor,
        )


@dataclass(frozen=True)
class IssueFilterConfig:
    """Configuration for issue filtering and selection.

    Controls which issues are selected for processing.

    Attributes:
        max_agents: Maximum concurrent agents (None = unlimited).
        max_issues: Maximum issues to process (None = unlimited).
        epic_id: Only process tasks under this epic.
        only_ids: List of issue IDs to process exclusively.
        include_wip: Include in_progress issues in scope (no ordering changes).
        focus: Only work on one epic at a time.
        orphans_only: Only process issues with no parent epic.
        epic_override_ids: Epic IDs to close without verification.
        order_preference: Issue ordering preference (focus, epic-priority, issue-priority, or input).
    """

    max_agents: int | None = None
    max_issues: int | None = None
    epic_id: str | None = None
    only_ids: list[str] | None = None
    include_wip: bool = False
    focus: bool = True
    orphans_only: bool = False
    epic_override_ids: set[str] = field(default_factory=set)
    order_preference: OrderPreference = _OrderPreference.EPIC_PRIORITY

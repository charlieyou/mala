"""Shared types for orchestrator components.

This module contains dataclasses and constants shared between
orchestrator.py and orchestrator_factory.py to break circular imports.

Design principles:
- OrchestratorConfig: All scalar configuration (timeouts, flags, limits)
- OrchestratorDependencies: All protocol implementations (DI for testability)
- _DerivedConfig: Internal computed configuration values
- RuntimeDeps: Protocol implementations for pipeline components
- PipelineConfig: Configuration for pipeline stages
- IssueFilterConfig: Filtering criteria for issue selection
- DEFAULT_AGENT_TIMEOUT_MINUTES: Default timeout constant
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003 - needed at runtime for dataclass field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.core.protocols import (
        CodeReviewer,
        CommandRunnerPort,
        EnvConfigPort,
        GateChecker,
        IssueProvider,
        LockManagerPort,
        LogProvider,
        MalaEventSink,
    )
    from src.domain.deadlock import DeadlockMonitor
    from src.domain.prompts import PromptProvider
    from src.domain.validation.config import PromptValidationCommands
    from src.infra.io.config import MalaConfig
    from src.infra.telemetry import TelemetryProvider

# Default timeout for agent execution (protects against hung MCP server subprocesses)
DEFAULT_AGENT_TIMEOUT_MINUTES = 60


@dataclass
class OrchestratorConfig:
    """Configuration for MalaOrchestrator.

    All scalar configuration values that control orchestrator behavior.
    These are typically derived from CLI arguments or environment.

    Attributes:
        repo_path: Path to the repository with beads issues.
        max_agents: Maximum concurrent agents (None = unlimited).
        timeout_minutes: Timeout per agent in minutes (None = default 60).
        max_issues: Maximum issues to process (None = unlimited).
        epic_id: Only process tasks under this epic.
        only_ids: Comma-separated list of issue IDs to process exclusively.
        braintrust_enabled: Enable Braintrust tracing.
        max_gate_retries: Maximum quality gate retry attempts per issue.
        max_review_retries: Maximum code review retry attempts per issue.
        disable_validations: Set of validation types to disable.
        coverage_threshold: Minimum coverage percentage (None = no-decrease mode).
        prioritize_wip: Prioritize in_progress issues before open issues.
        focus: Group tasks by epic for focused work.
        cli_args: CLI arguments for logging and metadata.
        epic_override_ids: Epic IDs to close without verification.
        orphans_only: Only process issues with no parent epic.
        context_restart_threshold: Ratio (0.0-1.0) at which to restart agent.
        context_limit: Maximum context tokens (default 200K for Claude).
    """

    repo_path: Path
    max_agents: int | None = None
    timeout_minutes: int | None = None
    max_issues: int | None = None
    epic_id: str | None = None
    only_ids: set[str] | None = None
    braintrust_enabled: bool | None = None
    max_gate_retries: int = 3
    max_review_retries: int = 3
    disable_validations: set[str] | None = None
    coverage_threshold: float | None = None
    prioritize_wip: bool = False
    focus: bool = True
    cli_args: dict[str, object] | None = None
    epic_override_ids: set[str] = field(default_factory=set)
    orphans_only: bool = False
    # Context exhaustion handling thresholds
    context_restart_threshold: float = 0.90
    context_limit: int = 200_000


@dataclass
class OrchestratorDependencies:
    """Protocol implementations for MalaOrchestrator.

    All injected dependencies that implement the orchestrator's protocols.
    When None, the factory creates default implementations.

    Note: AgentSessionRunner is NOT included here because it's constructed
    per-issue in run_implementer with issue-specific callbacks.

    Attributes:
        issue_provider: IssueProvider for issue tracking operations.
        code_reviewer: CodeReviewer for post-commit code reviews.
        gate_checker: GateChecker for quality gate validation.
        log_provider: LogProvider for session log access.
        telemetry_provider: TelemetryProvider for tracing.
        event_sink: MalaEventSink for run lifecycle logging.
        runs_dir: Directory for run markers (for testing).
        lock_releaser: Function to release locks (for testing).
    """

    issue_provider: IssueProvider | None = None
    code_reviewer: CodeReviewer | None = None
    gate_checker: GateChecker | None = None
    log_provider: LogProvider | None = None
    telemetry_provider: TelemetryProvider | None = None
    event_sink: MalaEventSink | None = None
    runs_dir: Path | None = None
    lock_releaser: Callable[[list[str]], int] | None = None


@dataclass
class _DerivedConfig:
    """Derived configuration values computed from OrchestratorConfig and MalaConfig.

    Internal class used to pass computed values to the orchestrator.
    """

    timeout_seconds: int
    braintrust_enabled: bool
    disabled_validations: set[str]
    review_disabled_reason: str | None = None
    braintrust_disabled_reason: str | None = None


@dataclass(frozen=True)
class RuntimeDeps:
    """Runtime dependencies for pipeline wiring.

    Protocol implementations and service objects injected at startup.

    Attributes:
        quality_gate: GateChecker for quality gate validation.
        code_reviewer: CodeReviewer for post-commit code reviews.
        beads: IssueProvider for issue tracking operations.
        event_sink: MalaEventSink for run lifecycle logging.
        command_runner: CommandRunnerPort for executing shell commands.
        env_config: EnvConfigPort for environment configuration.
        lock_manager: LockManagerPort for file locking coordination.
        mala_config: MalaConfig for orchestrator configuration.
    """

    quality_gate: GateChecker
    code_reviewer: CodeReviewer
    beads: IssueProvider
    event_sink: MalaEventSink
    command_runner: CommandRunnerPort
    env_config: EnvConfigPort
    lock_manager: LockManagerPort
    mala_config: MalaConfig


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration values for pipeline behavior.

    Scalar configuration controlling pipeline execution.

    Attributes:
        repo_path: Path to the repository with beads issues.
        timeout_seconds: Timeout per agent in seconds.
        max_gate_retries: Maximum quality gate retry attempts per issue.
        max_review_retries: Maximum code review retry attempts per issue.
        coverage_threshold: Minimum coverage percentage (None = no-decrease mode).
        disabled_validations: Set of validation types to disable.
        context_restart_threshold: Ratio (0.0-1.0) at which to restart agent.
        context_limit: Maximum context tokens (default 200K for Claude).
        prompts: PromptProvider with loaded prompt templates.
        prompt_validation_commands: Validation commands for prompt substitution.
        deadlock_monitor: DeadlockMonitor for deadlock detection (None until wired).
    """

    repo_path: Path
    timeout_seconds: int
    max_gate_retries: int
    max_review_retries: int
    coverage_threshold: float | None
    disabled_validations: set[str] | None
    context_restart_threshold: float
    context_limit: int
    prompts: PromptProvider
    prompt_validation_commands: PromptValidationCommands
    deadlock_monitor: DeadlockMonitor | None = None


@dataclass(frozen=True)
class IssueFilterConfig:
    """Configuration for issue filtering and selection.

    Controls which issues are selected for processing.

    Attributes:
        max_agents: Maximum concurrent agents (None = unlimited).
        max_issues: Maximum issues to process (None = unlimited).
        epic_id: Only process tasks under this epic.
        only_ids: Set of issue IDs to process exclusively.
        prioritize_wip: Prioritize in_progress issues before open issues.
        focus: Group tasks by epic for focused work.
        orphans_only: Only process issues with no parent epic.
        epic_override_ids: Epic IDs to close without verification.
    """

    max_agents: int | None = None
    max_issues: int | None = None
    epic_id: str | None = None
    only_ids: set[str] | None = None
    prioritize_wip: bool = False
    focus: bool = True
    orphans_only: bool = False
    epic_override_ids: set[str] = field(default_factory=set)


@dataclass
class WatchConfig:
    """Configuration for watch mode behavior.

    Attributes:
        enabled: Whether watch mode is active.
        validate_every: Run validation after this many issues complete.
        poll_interval_seconds: Seconds between poll cycles (internal, not CLI-exposed).
    """

    enabled: bool = False
    validate_every: int = 10
    poll_interval_seconds: float = 60.0


@dataclass
class WatchState:
    """Runtime state for watch mode (internal to Coordinator).

    Attributes:
        completed_count: Total issues completed in this watch session.
        last_validation_at: Completed count at last validation.
        next_validation_threshold: Run validation when completed_count reaches this.
            Must be initialized from WatchConfig.validate_every to stay in sync.
        consecutive_poll_failures: Count of consecutive poll failures.
        last_idle_log_time: Timestamp of last idle log message.
        was_idle: Whether the previous poll found no ready issues.
    """

    # No default - must be explicitly set from WatchConfig.validate_every
    next_validation_threshold: int
    completed_count: int = 0
    last_validation_at: int = 0
    consecutive_poll_failures: int = 0
    last_idle_log_time: float = 0.0
    was_idle: bool = False


@dataclass
class RunResult:
    """Result of a coordinator run loop.

    Attributes:
        issues_spawned: Number of issues spawned during the run.
        exit_code: Exit code (0=success, 1=validation_failed, 2=invalid_args, 3=abort, 130=interrupted).
        exit_reason: Human-readable exit reason.
    """

    issues_spawned: int
    exit_code: int
    exit_reason: str

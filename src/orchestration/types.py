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

# Private import for runtime use in dataclass defaults (not re-exported)
from src.core.models import OrderPreference as _OrderPreference

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.core.models import OrderPreference
    from src.core.protocols.events import MalaEventSink
    from src.core.protocols.infra import (
        CommandRunnerPort,
        EnvConfigPort,
        LockManagerPort,
    )
    from src.core.protocols.issue import IssueProvider
    from src.core.protocols.log import LogProvider
    from src.core.protocols.review import CodeReviewer
    from src.core.protocols.validation import GateChecker
    from src.domain.deadlock import DeadlockMonitor
    from src.domain.prompts import PromptProvider
    from src.domain.validation.config import (
        CodeReviewConfig,
        PromptValidationCommands,
        ValidationConfig,
    )
    from src.infra.io.config import MalaConfig
    from src.infra.telemetry import TelemetryProvider

# Default timeout for agent execution (protects against hung MCP server subprocesses)
DEFAULT_AGENT_TIMEOUT_MINUTES = 60

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
        timeout_minutes: Timeout per agent in minutes (None = default 60).
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
        log_provider: LogProvider for session log access.
        telemetry_provider: TelemetryProvider for tracing.
        event_sink: MalaEventSink for run lifecycle logging.
        command_runner: CommandRunnerPort for executing shell commands.
        env_config: EnvConfigPort for environment configuration.
        lock_manager: LockManagerPort for file locking coordination.
        runs_dir: Directory for run markers (for testing).
        lock_releaser: Function to release locks (for testing).
    """

    issue_provider: IssueProvider | None = None
    code_reviewer: CodeReviewer | None = None
    gate_checker: GateChecker | None = None
    log_provider: LogProvider | None = None
    telemetry_provider: TelemetryProvider | None = None
    event_sink: MalaEventSink | None = None
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
    max_epic_verification_retries: int | None = None
    max_diff_size_kb: int | None = None
    epic_verify_lock_timeout_seconds: int | None = None
    review_disabled_reason: str | None = None
    per_issue_review: CodeReviewConfig | None = None
    validation_config: ValidationConfig | None = None
    validation_config_missing: bool = False


@dataclass(frozen=True)
class RuntimeDeps:
    """Runtime dependencies for pipeline wiring.

    Protocol implementations and service objects injected at startup.

    Attributes:
        evidence_check: GateChecker for quality gate validation.
        code_reviewer: CodeReviewer for post-commit code reviews.
        beads: IssueProvider for issue tracking operations.
        event_sink: MalaEventSink for run lifecycle logging.
        command_runner: CommandRunnerPort for executing shell commands.
        env_config: EnvConfigPort for environment configuration.
        lock_manager: LockManagerPort for file locking coordination.
        mala_config: MalaConfig for orchestrator configuration.
    """

    evidence_check: GateChecker
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

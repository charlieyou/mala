"""Shared types for orchestrator components.

This module contains dataclasses and constants shared between
orchestrator.py and orchestrator_factory.py to break circular imports.

Design principles:
- OrchestratorConfig: All scalar configuration (timeouts, flags, limits)
- OrchestratorDependencies: All protocol implementations (DI for testability)
- _DerivedConfig: Internal computed configuration values
- DEFAULT_AGENT_TIMEOUT_MINUTES: Default timeout constant
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - needed at runtime for dataclass field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.infra.io.event_protocol import MalaEventSink
    from src.core.protocols import CodeReviewer, GateChecker, IssueProvider, LogProvider
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
    epic_override_ids: set[str] | None = None
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
    """

    issue_provider: IssueProvider | None = None
    code_reviewer: CodeReviewer | None = None
    gate_checker: GateChecker | None = None
    log_provider: LogProvider | None = None
    telemetry_provider: TelemetryProvider | None = None
    event_sink: MalaEventSink | None = None


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

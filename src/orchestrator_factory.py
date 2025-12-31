"""Factory function for MalaOrchestrator initialization.

This module provides a clean factory interface for creating MalaOrchestrator
instances, extracting the complex initialization logic from __init__ into
composable, testable pieces.

The factory pattern supports:
- Clean separation of configuration and dependencies
- Easy mocking for unit tests
- Progressive migration from legacy API
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - Path is used at runtime in dataclass
from typing import TYPE_CHECKING

from .config import MalaConfig
from .beads_client import BeadsClient
from .event_sink_console import ConsoleEventSink
from .telemetry import BraintrustProvider, NullTelemetryProvider
from .cerberus_review import DefaultReviewer
from .session_log_parser import FileSystemLogProvider
from .quality_gate import QualityGate
from .epic_verifier import ClaudeEpicVerificationModel, EpicVerifier
from .models import RetryConfig
from .pipeline.gate_runner import GateRunner, GateRunnerConfig
from .pipeline.review_runner import ReviewRunner, ReviewRunnerConfig
from .pipeline.run_coordinator import RunCoordinator, RunCoordinatorConfig

if TYPE_CHECKING:
    from .event_sink import MalaEventSink
    from .orchestrator import MalaOrchestrator
    from .protocols import CodeReviewer, GateChecker, IssueProvider, LogProvider
    from .telemetry import TelemetryProvider


# Default timeout for agent execution (matches orchestrator.py)
DEFAULT_AGENT_TIMEOUT_MINUTES = 60


@dataclass
class OrchestratorConfig:
    """Configuration for MalaOrchestrator initialization.

    Contains all configuration values that control orchestrator behavior.
    This separates configuration from runtime dependencies.

    Attributes:
        repo_path: Path to the repository to process.
        max_agents: Maximum concurrent agents (None = unlimited).
        timeout_minutes: Timeout per agent in minutes (default: 60).
        max_issues: Maximum issues to process (None = unlimited).
        epic_id: Only process tasks that are children of this epic.
        only_ids: Set of issue IDs to process exclusively.
        braintrust_enabled: Whether Braintrust tracing is enabled.
        max_gate_retries: Maximum quality gate retry attempts per issue.
        max_review_retries: Maximum codex review retry attempts per issue.
        disable_validations: Set of validation types to disable.
        coverage_threshold: Minimum coverage percentage (0-100).
        morph_enabled: Whether Morph MCP features are enabled.
        prioritize_wip: Prioritize in_progress issues before open issues.
        focus: Group tasks by epic for focused work.
        cli_args: CLI arguments for logging/metadata.
        epic_override_ids: Epic IDs to close without verification.
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
    morph_enabled: bool | None = None
    prioritize_wip: bool = False
    focus: bool = True
    cli_args: dict[str, object] | None = None
    epic_override_ids: set[str] | None = None


@dataclass
class OrchestratorDependencies:
    """Protocol-based dependencies for MalaOrchestrator.

    Contains all external dependencies that the orchestrator interacts with.
    All fields are optional - the factory will create defaults for missing deps.

    Attributes:
        issue_provider: IssueProvider for issue tracking (default: BeadsClient).
        code_reviewer: CodeReviewer for Cerberus review (default: DefaultReviewer).
        gate_checker: GateChecker for quality gate (default: QualityGate).
        log_provider: LogProvider for session logs (default: FileSystemLogProvider).
        telemetry_provider: TelemetryProvider for tracing (default: based on config).
        event_sink: MalaEventSink for run events (default: ConsoleEventSink).
    """

    issue_provider: IssueProvider | None = None
    code_reviewer: CodeReviewer | None = None
    gate_checker: GateChecker | None = None
    log_provider: LogProvider | None = None
    telemetry_provider: TelemetryProvider | None = None
    event_sink: MalaEventSink | None = None


@dataclass
class _ResolvedDeps:
    """Internal: Fully resolved dependencies after factory construction."""

    issue_provider: IssueProvider
    code_reviewer: CodeReviewer
    gate_checker: GateChecker
    log_provider: LogProvider
    telemetry_provider: TelemetryProvider
    event_sink: MalaEventSink
    epic_verifier: EpicVerifier | None
    gate_runner: GateRunner
    review_runner: ReviewRunner
    run_coordinator: RunCoordinator
    review_disabled_reason: str | None = None


def _resolve_feature_flags(
    config: OrchestratorConfig,
    mala_config: MalaConfig,
) -> tuple[bool, bool]:
    """Resolve feature flags from config hierarchy.

    Precedence: explicit OrchestratorConfig > MalaConfig defaults.

    Returns:
        Tuple of (braintrust_enabled, morph_enabled).
    """
    braintrust_enabled = (
        config.braintrust_enabled
        if config.braintrust_enabled is not None
        else mala_config.braintrust_enabled
    )
    morph_enabled = (
        config.morph_enabled
        if config.morph_enabled is not None
        else mala_config.morph_enabled
    )
    return braintrust_enabled, morph_enabled


def _check_review_availability(
    mala_config: MalaConfig,
    disabled_validations: set[str],
) -> str | None:
    """Check if review-gate is available, return disabled reason if not."""
    if "review" in disabled_validations:
        return None  # Already disabled, no need to explain

    review_gate_path = (
        mala_config.cerberus_bin_path / "review-gate"
        if mala_config.cerberus_bin_path
        else None
    )

    if review_gate_path is None:
        # No explicit bin_path - check PATH (respecting cerberus_env if set)
        cerberus_env_dict = dict(mala_config.cerberus_env)
        if "PATH" in cerberus_env_dict:
            effective_path = (
                cerberus_env_dict["PATH"] + os.pathsep + os.environ.get("PATH", "")
            )
        else:
            effective_path = os.environ.get("PATH", "")
        if shutil.which("review-gate", path=effective_path) is None:
            return "cerberus plugin not detected (review-gate unavailable)"
    elif not review_gate_path.exists():
        return f"review-gate missing at {review_gate_path}"
    elif not review_gate_path.is_file():
        return f"review-gate path is not a file: {review_gate_path}"
    elif not os.access(review_gate_path, os.X_OK):
        return f"review-gate not executable at {review_gate_path}"

    return None


def _build_dependencies(
    config: OrchestratorConfig,
    mala_config: MalaConfig,
    deps: OrchestratorDependencies | None,
    braintrust_enabled: bool,
    morph_enabled: bool,
) -> _ResolvedDeps:
    """Build all dependencies, using provided deps or creating defaults."""
    deps = deps or OrchestratorDependencies()
    disabled_validations = set(config.disable_validations or [])

    # Timeout in seconds
    timeout_minutes = (
        config.timeout_minutes
        if config.timeout_minutes is not None
        else DEFAULT_AGENT_TIMEOUT_MINUTES
    )
    timeout_seconds = timeout_minutes * 60

    # EventSink: ConsoleEventSink by default (needed early for log_warning)
    event_sink: MalaEventSink = (
        deps.event_sink if deps.event_sink is not None else ConsoleEventSink()
    )

    def log_warning(msg: str) -> None:
        event_sink.on_warning(msg)

    # LogProvider: FileSystemLogProvider for reading Claude SDK logs
    log_provider: LogProvider = (
        deps.log_provider if deps.log_provider is not None else FileSystemLogProvider()
    )

    # GateChecker: QualityGate for post-run validation
    gate_checker: GateChecker = (
        deps.gate_checker
        if deps.gate_checker is not None
        else QualityGate(config.repo_path, log_provider=log_provider)
    )

    # IssueProvider: BeadsClient for issue tracking
    if deps.issue_provider is not None:
        issue_provider = deps.issue_provider
    else:
        issue_provider = BeadsClient(config.repo_path, log_warning=log_warning)

    # EpicVerifier: For epic verification before closure
    epic_verifier: EpicVerifier | None = None
    if isinstance(issue_provider, BeadsClient):
        verification_model = ClaudeEpicVerificationModel(
            api_key=mala_config.llm_api_key,
            base_url=mala_config.llm_base_url,
            timeout_ms=timeout_seconds * 1000,
            retry_config=RetryConfig(),
        )
        epic_verifier = EpicVerifier(
            beads=issue_provider,
            model=verification_model,
            repo_path=config.repo_path,
            max_diff_size_kb=mala_config.max_diff_size_kb,
            event_sink=event_sink,
            lock_manager=True,
        )

    # Check review availability
    review_disabled_reason = _check_review_availability(
        mala_config, disabled_validations
    )
    if review_disabled_reason:
        disabled_validations.add("review")

    # CodeReviewer: DefaultReviewer using Cerberus review-gate
    code_reviewer: CodeReviewer = (
        deps.code_reviewer
        if deps.code_reviewer is not None
        else DefaultReviewer(
            repo_path=config.repo_path,
            bin_path=mala_config.cerberus_bin_path,
            spawn_args=mala_config.cerberus_spawn_args,
            wait_args=mala_config.cerberus_wait_args,
            env=dict(mala_config.cerberus_env),
        )
    )

    # TelemetryProvider: BraintrustProvider when enabled, else NullTelemetryProvider
    telemetry_provider: TelemetryProvider
    if deps.telemetry_provider is not None:
        telemetry_provider = deps.telemetry_provider
    elif braintrust_enabled:
        telemetry_provider = BraintrustProvider()
    else:
        telemetry_provider = NullTelemetryProvider()

    # GateRunner: Extracted gate checking logic
    gate_runner_config = GateRunnerConfig(
        max_gate_retries=config.max_gate_retries,
        disable_validations=disabled_validations,
        coverage_threshold=config.coverage_threshold,
    )
    gate_runner = GateRunner(
        gate_checker=gate_checker,
        repo_path=config.repo_path,
        config=gate_runner_config,
    )

    # ReviewRunner: Extracted review orchestration logic
    review_runner_config = ReviewRunnerConfig(
        max_review_retries=config.max_review_retries,
        capture_session_log=False,  # Set per-issue based on verbose
        review_timeout=mala_config.review_timeout,
    )
    review_runner = ReviewRunner(
        code_reviewer=code_reviewer,
        config=review_runner_config,
        gate_checker=gate_checker,
    )

    # RunCoordinator: Extracted run-level validation and fixer logic
    run_coordinator_config = RunCoordinatorConfig(
        repo_path=config.repo_path,
        timeout_seconds=timeout_seconds,
        max_gate_retries=config.max_gate_retries,
        disable_validations=disabled_validations,
        coverage_threshold=config.coverage_threshold,
        morph_enabled=morph_enabled,
        morph_api_key=mala_config.morph_api_key,
    )
    run_coordinator = RunCoordinator(
        config=run_coordinator_config,
        gate_checker=gate_checker,
        event_sink=event_sink,
    )

    return _ResolvedDeps(
        issue_provider=issue_provider,  # type: ignore[arg-type]
        code_reviewer=code_reviewer,
        gate_checker=gate_checker,
        log_provider=log_provider,
        telemetry_provider=telemetry_provider,
        event_sink=event_sink,
        epic_verifier=epic_verifier,
        gate_runner=gate_runner,
        review_runner=review_runner,
        run_coordinator=run_coordinator,
        review_disabled_reason=review_disabled_reason,
    )


def create_orchestrator(
    config: OrchestratorConfig,
    *,
    mala_config: MalaConfig | None = None,
    deps: OrchestratorDependencies | None = None,
) -> MalaOrchestrator:
    """Create a MalaOrchestrator instance with the given configuration.

    This factory function encapsulates all initialization logic, providing:
    - Clean separation of configuration and dependencies
    - Easy mocking for unit tests
    - Consistent initialization regardless of entry point

    Config precedence:
    - explicit OrchestratorConfig fields > MalaConfig > defaults

    Args:
        config: OrchestratorConfig with all configuration values.
        mala_config: Optional MalaConfig. If None, loads from environment.
        deps: Optional OrchestratorDependencies for testing. If None, uses defaults.

    Returns:
        Configured MalaOrchestrator instance.

    Example:
        # Production usage (from CLI)
        config = OrchestratorConfig(
            repo_path=Path("/path/to/repo"),
            max_agents=4,
            timeout_minutes=30,
        )
        orchestrator = create_orchestrator(config)

        # Testing usage (with mocked dependencies)
        deps = OrchestratorDependencies(
            issue_provider=mock_issue_provider,
            gate_checker=mock_gate_checker,
        )
        orchestrator = create_orchestrator(config, deps=deps)
    """
    # Import here to avoid circular imports
    from .orchestrator import MalaOrchestrator

    # Use provided MalaConfig or load from environment
    effective_mala_config = (
        mala_config if mala_config is not None else MalaConfig.from_env(validate=False)
    )

    # Resolve feature flags
    braintrust_enabled, morph_enabled = _resolve_feature_flags(
        config, effective_mala_config
    )

    # Build all dependencies
    resolved = _build_dependencies(
        config,
        effective_mala_config,
        deps,
        braintrust_enabled,
        morph_enabled,
    )

    # Create orchestrator with resolved config and dependencies
    return MalaOrchestrator(
        _config=config,
        _mala_config=effective_mala_config,
        _resolved_deps=resolved,
        _braintrust_enabled=braintrust_enabled,
        _morph_enabled=morph_enabled,
    )

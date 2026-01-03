"""Factory function for MalaOrchestrator initialization.

This module encapsulates the ~250-line __init__ logic into a clean factory
pattern with explicit configuration and dependency dataclasses.

Design principles:
- OrchestratorConfig: All scalar configuration (timeouts, flags, limits)
- OrchestratorDependencies: All protocol implementations (DI for testability)
- create_orchestrator(): Factory function encapsulating initialization logic

Usage:
    # Simple usage with defaults
    config = OrchestratorConfig(repo_path=Path("."))
    orchestrator = create_orchestrator(config)

    # With explicit MalaConfig for API keys
    mala_config = MalaConfig.from_env()
    orchestrator = create_orchestrator(config, mala_config=mala_config)

    # With custom dependencies for testing
    deps = OrchestratorDependencies(
        issue_provider=mock_beads,
        code_reviewer=mock_reviewer,
    )
    orchestrator = create_orchestrator(config, deps=deps)
"""

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING, cast

# Import shared types from types module (breaks circular import)
from .types import (
    DEFAULT_AGENT_TIMEOUT_MINUTES,
    OrchestratorConfig,
    OrchestratorDependencies,
    _DerivedConfig,
)

# Re-export for backwards compatibility
__all__ = [
    "DEFAULT_AGENT_TIMEOUT_MINUTES",
    "OrchestratorConfig",
    "OrchestratorDependencies",
    "create_orchestrator",
]

if TYPE_CHECKING:
    from src.core.protocols import (
        CodeReviewer,
        EpicVerificationModel,
        GateChecker,
        IssueProvider,
        LogProvider,
    )
    from src.infra.epic_verifier import EpicVerifier
    from src.infra.io.config import MalaConfig
    from src.infra.io.event_protocol import MalaEventSink
    from src.infra.telemetry import TelemetryProvider

    from .orchestrator import MalaOrchestrator


def _derive_config(
    config: OrchestratorConfig,
    mala_config: MalaConfig,
) -> _DerivedConfig:
    """Derive computed configuration values from config sources.

    Config precedence: OrchestratorConfig > MalaConfig > defaults

    Args:
        config: User-provided orchestrator configuration.
        mala_config: MalaConfig with API keys and feature flags.

    Returns:
        _DerivedConfig with computed values.
    """
    # Compute timeout - match legacy behavior where 0 is treated as "use default"
    # (truthiness check: `if timeout_minutes` treats 0 as falsy)
    effective_timeout = (
        config.timeout_minutes
        if config.timeout_minutes
        else DEFAULT_AGENT_TIMEOUT_MINUTES
    )
    timeout_seconds = effective_timeout * 60

    # Derive feature flags from mala_config if not explicitly set
    if config.braintrust_enabled is not None:
        braintrust_enabled = config.braintrust_enabled
    else:
        braintrust_enabled = mala_config.braintrust_enabled

    if config.morph_enabled is not None:
        morph_enabled = config.morph_enabled
    else:
        morph_enabled = mala_config.morph_enabled

    # Build disabled validations set
    disabled_validations = (
        set(config.disable_validations) if config.disable_validations else set()
    )

    return _DerivedConfig(
        timeout_seconds=timeout_seconds,
        braintrust_enabled=braintrust_enabled,
        morph_enabled=morph_enabled,
        disabled_validations=disabled_validations,
    )


def _check_review_availability(
    mala_config: MalaConfig,
    disabled_validations: set[str],
) -> str | None:
    """Check if code review is available.

    Returns the reason review is disabled, or None if available.
    """
    if "review" in disabled_validations:
        return None  # Explicitly disabled, no warning needed

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
    derived: _DerivedConfig,
    deps: OrchestratorDependencies | None,
) -> tuple[
    IssueProvider,
    CodeReviewer,
    GateChecker,
    LogProvider,
    TelemetryProvider,
    MalaEventSink,
    EpicVerifier | None,
]:
    """Build all dependencies, using provided ones or creating defaults.

    Args:
        config: Orchestrator configuration.
        mala_config: MalaConfig with API keys.
        derived: Derived configuration values.
        deps: Optional pre-built dependencies.

    Returns:
        Tuple of all required dependencies.
    """
    from src.core.models import RetryConfig
    from src.domain.quality_gate import QualityGate
    from src.infra.clients.beads_client import BeadsClient
    from src.infra.clients.braintrust_integration import BraintrustProvider
    from src.infra.clients.cerberus_review import DefaultReviewer
    from src.infra.epic_verifier import ClaudeEpicVerificationModel, EpicVerifier
    from src.infra.io.event_sink import ConsoleEventSink
    from src.infra.io.session_log_parser import FileSystemLogProvider
    from src.infra.telemetry import NullTelemetryProvider

    # Get resolved path
    repo_path = config.repo_path.resolve()

    # Event sink (needed for log_warning in BeadsClient)
    if deps is not None and deps.event_sink is not None:
        event_sink = deps.event_sink
    else:
        event_sink = ConsoleEventSink()

    # Log provider
    log_provider: LogProvider
    if deps is not None and deps.log_provider is not None:
        log_provider = deps.log_provider
    else:
        log_provider = cast("LogProvider", FileSystemLogProvider())

    # Gate checker (needs log_provider)
    gate_checker: GateChecker
    if deps is not None and deps.gate_checker is not None:
        gate_checker = deps.gate_checker
    else:
        gate_checker = cast(
            "GateChecker", QualityGate(repo_path, log_provider=log_provider)
        )

    # Issue provider (needs event_sink for warnings)
    issue_provider: IssueProvider
    if deps is not None and deps.issue_provider is not None:
        issue_provider = deps.issue_provider
    else:

        def log_warning(msg: str) -> None:
            event_sink.on_warning(msg)

        beads_client = BeadsClient(repo_path, log_warning=log_warning)
        # BeadsClient implements IssueProvider protocol
        issue_provider = beads_client  # type: ignore[assignment]

    # Epic verifier (only when using real BeadsClient - either created or injected)
    epic_verifier: EpicVerifier | None = None
    if isinstance(issue_provider, BeadsClient):
        verification_model = ClaudeEpicVerificationModel(
            timeout_ms=derived.timeout_seconds * 1000,
            retry_config=RetryConfig(),
            repo_path=repo_path,
        )
        epic_verifier = EpicVerifier(
            beads=issue_provider,
            model=cast("EpicVerificationModel", verification_model),
            repo_path=repo_path,
            event_sink=event_sink,
            lock_manager=True,
        )

    # Code reviewer
    code_reviewer: CodeReviewer
    if deps is not None and deps.code_reviewer is not None:
        code_reviewer = deps.code_reviewer
    else:
        code_reviewer = cast(
            "CodeReviewer",
            DefaultReviewer(
                repo_path=repo_path,
                bin_path=mala_config.cerberus_bin_path,
                spawn_args=mala_config.cerberus_spawn_args,
                wait_args=mala_config.cerberus_wait_args,
                env=dict(mala_config.cerberus_env),
                event_sink=event_sink,
            ),
        )

    # Telemetry provider
    telemetry_provider: TelemetryProvider
    if deps is not None and deps.telemetry_provider is not None:
        telemetry_provider = deps.telemetry_provider
    elif derived.braintrust_enabled:
        telemetry_provider = cast("TelemetryProvider", BraintrustProvider())
    else:
        telemetry_provider = cast("TelemetryProvider", NullTelemetryProvider())

    return (
        issue_provider,
        code_reviewer,
        gate_checker,
        log_provider,
        telemetry_provider,
        event_sink,
        epic_verifier,
    )


def create_orchestrator(
    config: OrchestratorConfig,
    *,
    mala_config: MalaConfig | None = None,
    deps: OrchestratorDependencies | None = None,
) -> MalaOrchestrator:
    """Create a MalaOrchestrator with the given configuration.

    This factory function encapsulates all initialization logic that was
    previously in MalaOrchestrator.__init__. It:
    1. Derives computed configuration from config sources
    2. Checks review availability
    3. Builds dependencies (using provided or creating defaults)
    4. Constructs and returns the orchestrator

    Config precedence: config > mala_config > defaults

    Args:
        config: OrchestratorConfig with all scalar configuration.
        mala_config: Optional MalaConfig for API keys and feature flags.
            If None, loads from environment.
        deps: Optional OrchestratorDependencies for custom implementations.
            If None, creates default implementations.

    Returns:
        Configured MalaOrchestrator ready for run().

    Example:
        # Simple usage
        config = OrchestratorConfig(repo_path=Path("."))
        orchestrator = create_orchestrator(config)
        success, total = await orchestrator.run()

        # With custom dependencies for testing
        deps = OrchestratorDependencies(
            issue_provider=mock_beads,
            gate_checker=mock_gate,
        )
        orchestrator = create_orchestrator(config, deps=deps)
    """
    from src.infra.io.config import MalaConfig

    from .orchestrator import MalaOrchestrator

    # Load MalaConfig if not provided
    if mala_config is None:
        mala_config = MalaConfig.from_env(validate=False)

    # Derive computed configuration
    derived = _derive_config(config, mala_config)

    # Check review availability and update disabled_validations
    review_disabled_reason = _check_review_availability(
        mala_config, derived.disabled_validations
    )
    if review_disabled_reason:
        derived.disabled_validations.add("review")
        derived.review_disabled_reason = review_disabled_reason

    # Build dependencies
    (
        issue_provider,
        code_reviewer,
        gate_checker,
        log_provider,
        telemetry_provider,
        event_sink,
        epic_verifier,
    ) = _build_dependencies(config, mala_config, derived, deps)

    # Create orchestrator using internal constructor
    return MalaOrchestrator(
        _config=config,
        _mala_config=mala_config,
        _derived=derived,
        _issue_provider=issue_provider,
        _code_reviewer=code_reviewer,
        _gate_checker=gate_checker,
        _log_provider=log_provider,
        _telemetry_provider=telemetry_provider,
        _event_sink=event_sink,
        _epic_verifier=epic_verifier,
    )

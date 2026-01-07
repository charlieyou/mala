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

import logging
import os
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast


# Import shared types from types module (breaks circular import)
from .types import (
    DEFAULT_AGENT_TIMEOUT_MINUTES,
    OrchestratorConfig,
    OrchestratorDependencies,
    _DerivedConfig,
)

__all__ = [
    "DEFAULT_AGENT_TIMEOUT_MINUTES",
    "OrchestratorConfig",
    "OrchestratorDependencies",
    "create_issue_provider",
    "create_orchestrator",
]


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.core.protocols import (
        CodeReviewer,
        EpicVerificationModel,
        GateChecker,
        IssueProvider,
        LogProvider,
    )
    from src.infra.io.config import MalaConfig
    from src.core.protocols import EpicVerifierProtocol, MalaEventSink
    from src.infra.telemetry import TelemetryProvider

    from .orchestrator import MalaOrchestrator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ReviewerConfig:
    """Reviewer configuration loaded from mala.yaml.

    Consolidates all reviewer-related settings to avoid loading
    ValidationConfig multiple times during orchestrator creation.
    """

    reviewer_type: str = "agent_sdk"
    agent_sdk_review_timeout: int = 600
    agent_sdk_reviewer_model: str = "sonnet"


def create_issue_provider(
    repo_path: Path,
    log_warning: Callable[[str], None] | None = None,
) -> IssueProvider:
    """Create an IssueProvider instance (BeadsClient).

    This factory function allows CLI code to create an IssueProvider without
    importing BeadsClient directly, maintaining the architectural boundary.

    Args:
        repo_path: Path to the repository.
        log_warning: Optional callback for logging warnings.

    Returns:
        An IssueProvider instance (BeadsClient).
    """
    from src.infra.clients.beads_client import BeadsClient

    return BeadsClient(repo_path, log_warning=log_warning)  # type: ignore[return-value]


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

    # Build disabled validations set
    disabled_validations = (
        set(config.disable_validations) if config.disable_validations else set()
    )

    logger.debug(
        "Derived config: timeout=%ds",
        timeout_seconds,
    )
    return _DerivedConfig(
        timeout_seconds=timeout_seconds,
        disabled_validations=disabled_validations,
    )


def _get_reviewer_config(repo_path: Path) -> _ReviewerConfig:
    """Get reviewer configuration from ValidationConfig.

    Loads mala.yaml once to get all reviewer-related settings.
    Returns defaults if config is missing or loading fails.

    Args:
        repo_path: Path to the repository.

    Returns:
        _ReviewerConfig with reviewer settings.
    """
    from src.domain.validation.config_loader import ConfigMissingError, load_config

    try:
        validation_config = load_config(repo_path)
        return _ReviewerConfig(
            reviewer_type=getattr(validation_config, "reviewer_type", "agent_sdk"),
            agent_sdk_review_timeout=getattr(
                validation_config, "agent_sdk_review_timeout", 600
            ),
            agent_sdk_reviewer_model=getattr(
                validation_config, "agent_sdk_reviewer_model", "sonnet"
            ),
        )
    except ConfigMissingError:
        return _ReviewerConfig()
    except Exception as e:
        logger.warning("Failed to load reviewer config from mala.yaml: %s", e)
        return _ReviewerConfig()


def _check_review_availability(
    mala_config: MalaConfig,
    disabled_validations: set[str],
    reviewer_type: str = "agent_sdk",
) -> str | None:
    """Check if code review is available.

    Returns the reason review is disabled, or None if available.
    For agent_sdk reviewer, always available (no external dependencies).
    For cerberus reviewer, checks if review-gate binary is available.
    For unknown reviewer_type, returns disabled with warning.

    Args:
        mala_config: MalaConfig with Cerberus settings.
        disabled_validations: Set of validations explicitly disabled.
        reviewer_type: Type of reviewer ('agent_sdk' or 'cerberus').

    Returns:
        Reason review is disabled, or None if available.
    """
    if "review" in disabled_validations:
        return None  # Explicitly disabled, no warning needed

    # Agent SDK reviewer is always available (no external dependencies)
    if reviewer_type == "agent_sdk":
        return None

    # Unknown reviewer_type - disable review to prevent crashes
    if reviewer_type not in ("cerberus", "agent_sdk"):
        reason = f"unknown reviewer_type '{reviewer_type}', expected 'agent_sdk' or 'cerberus'"
        logger.warning("Review disabled: reason=%s", reason)
        return reason

    # Cerberus reviewer requires review-gate binary
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
            reason = "cerberus plugin not detected (review-gate unavailable)"
            logger.info("Review disabled: reason=%s", reason)
            return reason
    elif not review_gate_path.exists():
        reason = f"review-gate missing at {review_gate_path}"
        logger.info("Review disabled: reason=%s", reason)
        return reason
    elif not review_gate_path.is_file():
        reason = f"review-gate path is not a file: {review_gate_path}"
        logger.info("Review disabled: reason=%s", reason)
        return reason
    elif not os.access(review_gate_path, os.X_OK):
        reason = f"review-gate not executable at {review_gate_path}"
        logger.info("Review disabled: reason=%s", reason)
        return reason

    return None


def _create_code_reviewer(
    repo_path: Path,
    mala_config: MalaConfig,
    event_sink: MalaEventSink,
    reviewer_config: _ReviewerConfig,
) -> CodeReviewer:
    """Create the code reviewer based on reviewer configuration.

    Uses the pre-loaded _ReviewerConfig to avoid redundant mala.yaml reads.
    Defaults to AgentSDKReviewer when reviewer_type is 'agent_sdk' (default)
    or falls back to DefaultReviewer (Cerberus) when reviewer_type is 'cerberus'.

    Args:
        repo_path: Path to the repository.
        mala_config: MalaConfig with Cerberus settings (for fallback).
        event_sink: Event sink for telemetry and warnings.
        reviewer_config: Pre-loaded reviewer configuration from mala.yaml.

    Returns:
        CodeReviewer implementation (AgentSDKReviewer or DefaultReviewer).
    """
    from src.domain.prompts import load_prompts
    from src.infra.clients.agent_sdk_review import AgentSDKReviewer
    from src.infra.clients.cerberus_review import DefaultReviewer
    from src.infra.sdk_adapter import SDKClientFactory

    if reviewer_config.reviewer_type == "cerberus":
        logger.info("Using DefaultReviewer (Cerberus) for code review")
        return cast(
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
    else:
        # Default: agent_sdk
        logger.info("Using AgentSDKReviewer for code review")
        # Load prompts to get review_agent_prompt
        from src.infra.tools.env import PROMPTS_DIR

        prompts = load_prompts(PROMPTS_DIR)
        sdk_client_factory = SDKClientFactory()

        return cast(
            "CodeReviewer",
            AgentSDKReviewer(
                repo_path=repo_path,
                review_agent_prompt=prompts.review_agent_prompt,
                sdk_client_factory=sdk_client_factory,
                event_sink=event_sink,
                model=reviewer_config.agent_sdk_reviewer_model,
                default_timeout=reviewer_config.agent_sdk_review_timeout,
            ),
        )


def _build_dependencies(
    config: OrchestratorConfig,
    mala_config: MalaConfig,
    derived: _DerivedConfig,
    deps: OrchestratorDependencies | None,
    reviewer_config: _ReviewerConfig,
) -> tuple[
    IssueProvider,
    CodeReviewer,
    GateChecker,
    LogProvider,
    TelemetryProvider,
    MalaEventSink,
    EpicVerifierProtocol | None,
]:
    """Build all dependencies, using provided ones or creating defaults.

    Args:
        config: Orchestrator configuration.
        mala_config: MalaConfig with API keys.
        derived: Derived configuration values.
        deps: Optional pre-built dependencies.
        reviewer_config: Pre-loaded reviewer configuration from mala.yaml.

    Returns:
        Tuple of all required dependencies.
    """
    from src.core.models import RetryConfig
    from src.domain.evidence_check import EvidenceCheck
    from src.infra.clients.beads_client import BeadsClient
    from src.infra.epic_verifier import ClaudeEpicVerificationModel, EpicVerifier
    from src.infra.io.console_sink import ConsoleEventSink
    from src.infra.io.session_log_parser import FileSystemLogProvider
    from src.infra.telemetry import NullTelemetryProvider
    from src.infra.tools.command_runner import CommandRunner

    # Get resolved path
    repo_path = config.repo_path.resolve()

    # Command runner (shared by components that need it)
    command_runner = CommandRunner(cwd=repo_path)

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

    # Gate checker (needs log_provider and command_runner)
    gate_checker: GateChecker
    if deps is not None and deps.gate_checker is not None:
        gate_checker = deps.gate_checker
    else:
        gate_checker = cast(
            "GateChecker",
            EvidenceCheck(
                repo_path, log_provider=log_provider, command_runner=command_runner
            ),
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
    epic_verifier: EpicVerifierProtocol | None = None
    if isinstance(issue_provider, BeadsClient):
        verification_model = ClaudeEpicVerificationModel(
            timeout_ms=derived.timeout_seconds * 1000,
            retry_config=RetryConfig(),
            repo_path=repo_path,
        )
        from src.infra.tools.locking import LockManager

        epic_verifier = cast(
            "EpicVerifierProtocol",
            EpicVerifier(
                beads=issue_provider,
                model=cast("EpicVerificationModel", verification_model),
                repo_path=repo_path,
                command_runner=command_runner,
                event_sink=event_sink,
                lock_manager=LockManager(),
            ),
        )

    # Code reviewer - select based on reviewer_config.reviewer_type
    code_reviewer: CodeReviewer
    if deps is not None and deps.code_reviewer is not None:
        code_reviewer = deps.code_reviewer
    else:
        code_reviewer = _create_code_reviewer(
            repo_path=repo_path,
            mala_config=mala_config,
            event_sink=event_sink,
            reviewer_config=reviewer_config,
        )

    # Telemetry provider
    telemetry_provider: TelemetryProvider
    if deps is not None and deps.telemetry_provider is not None:
        telemetry_provider = deps.telemetry_provider
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

    # Load reviewer config once (avoids redundant mala.yaml reads)
    reviewer_config = _get_reviewer_config(config.repo_path)

    # Check review availability and update disabled_validations
    review_disabled_reason = _check_review_availability(
        mala_config, derived.disabled_validations, reviewer_config.reviewer_type
    )
    if review_disabled_reason:
        derived.disabled_validations.add("review")
        derived.review_disabled_reason = review_disabled_reason

    # Build dependencies (pass reviewer_config to avoid second config load)
    (
        issue_provider,
        code_reviewer,
        gate_checker,
        log_provider,
        telemetry_provider,
        event_sink,
        epic_verifier,
    ) = _build_dependencies(config, mala_config, derived, deps, reviewer_config)

    # Create orchestrator using internal constructor
    logger.info(
        "Orchestrator created: max_agents=%d timeout=%ds",
        config.max_agents or 0,
        derived.timeout_seconds,
    )
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
        runs_dir=deps.runs_dir if deps else None,
        lock_releaser=deps.lock_releaser if deps else None,
    )

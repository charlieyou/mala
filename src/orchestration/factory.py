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
    DEFAULT_MAX_IDLE_RETRIES,
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

    from src.core.protocols.events import MalaEventSink
    from src.core.protocols.infra import (
        CommandRunnerPort,
        EnvConfigPort,
        LockManagerPort,
    )
    from src.core.protocols.issue import IssueProvider
    from src.core.protocols.log import LogProvider
    from src.core.protocols.review import CodeReviewer
    from src.core.protocols.validation import (
        EpicVerificationModel,
        EpicVerifierProtocol,
        GateChecker,
    )
    from src.domain.validation.config import (
        CerberusConfig,
        ValidationConfig,
        VerificationRetryPolicy,
    )
    from src.infra.io.config import MalaConfig
    from src.infra.telemetry import TelemetryProvider

    from .orchestrator import MalaOrchestrator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ReviewerConfig:
    """Reviewer configuration loaded from mala.yaml.

    Consolidates all reviewer-related settings to avoid loading
    ValidationConfig multiple times during orchestrator creation.

    When reviewer_type='cerberus', the cerberus_config field contains
    settings from code_review.cerberus in mala.yaml. Falls back to
    MalaConfig.cerberus_* env vars for backward compatibility.
    """

    reviewer_type: str = "agent_sdk"
    agent_sdk_review_timeout: int = 600
    agent_sdk_reviewer_model: str = "sonnet"
    cerberus_config: CerberusConfig | None = None


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
    *,
    validation_config: ValidationConfig | None,
    validation_config_missing: bool,
) -> _DerivedConfig:
    """Derive computed configuration values from config sources.

    Config precedence: OrchestratorConfig > MalaConfig > defaults

    Args:
        config: User-provided orchestrator configuration.
        mala_config: MalaConfig with API keys and feature flags.
        validation_config: Loaded ValidationConfig from startup, if any.
        validation_config_missing: True if mala.yaml was missing at startup.

    Returns:
        _DerivedConfig with computed values.
    """
    # Compute timeout - precedence: CLI > mala.yaml > default
    # Legacy behavior: CLI 0 is treated as "use default" (skips mala.yaml)
    # - CLI positive value -> use CLI value
    # - CLI 0 or None -> check mala.yaml, else default
    # - CLI 0 specifically bypasses mala.yaml (for backward compat)
    yaml_timeout: int | None = (
        getattr(validation_config, "timeout_minutes", None)
        if validation_config is not None
        else None
    )
    if config.timeout_minutes:
        # CLI provided positive timeout_minutes
        timeout_seconds = config.timeout_minutes * 60
    elif config.timeout_minutes == 0:
        # CLI explicitly passed 0 -> use default (legacy behavior)
        timeout_seconds = DEFAULT_AGENT_TIMEOUT_MINUTES * 60
    elif yaml_timeout:
        # mala.yaml has timeout_minutes
        timeout_seconds = yaml_timeout * 60
    else:
        # Fall back to default
        timeout_seconds = DEFAULT_AGENT_TIMEOUT_MINUTES * 60

    # Build disabled validations set
    disabled_validations = (
        set(config.disable_validations) if config.disable_validations else set()
    )

    # Extract max_review_retries - precedence:
    # 1. per_issue_review.max_retries (only when per_issue_review.enabled=True)
    # 2. First enabled trigger code_review.max_retries (existing behavior)
    max_review_retries: int | None = None
    per_issue_review = (
        getattr(validation_config, "per_issue_review", None)
        if validation_config is not None
        else None
    )
    if per_issue_review is not None and getattr(per_issue_review, "enabled", False):
        max_review_retries = getattr(per_issue_review, "max_retries", None)
    if max_review_retries is None:
        # Fall back to trigger config
        code_review = _get_first_enabled_code_review(validation_config)
        if code_review is not None:
            max_review_retries = getattr(code_review, "max_retries", None)

    # Extract max_gate_retries from session_end trigger config
    max_gate_retries: int | None = None
    if validation_config is not None:
        triggers = getattr(validation_config, "validation_triggers", None)
        if triggers is not None:
            session_end = getattr(triggers, "session_end", None)
            if session_end is not None:
                max_gate_retries = getattr(session_end, "max_retries", None)

    # Extract max_epic_verification_retries and epic_verify_lock_timeout_seconds
    # from epic_completion trigger config
    max_epic_verification_retries: int | None = None
    epic_verify_lock_timeout_seconds: int | None = None
    if validation_config is not None:
        triggers = getattr(validation_config, "validation_triggers", None)
        if triggers is not None:
            epic_completion = getattr(triggers, "epic_completion", None)
            if epic_completion is not None:
                max_epic_verification_retries = getattr(
                    epic_completion, "max_epic_verification_retries", None
                )
                epic_verify_lock_timeout_seconds = getattr(
                    epic_completion, "epic_verify_lock_timeout_seconds", None
                )

    # Compute max_idle_retries - only from mala.yaml (no CLI option)
    yaml_max_idle_retries: int | None = (
        getattr(validation_config, "max_idle_retries", None)
        if validation_config is not None
        else None
    )
    max_idle_retries = (
        yaml_max_idle_retries
        if yaml_max_idle_retries is not None
        else DEFAULT_MAX_IDLE_RETRIES
    )

    # Compute idle_timeout_seconds - only from mala.yaml (no CLI option)
    # None means derive from agent timeout at runtime; 0 means disabled
    yaml_idle_timeout_seconds: float | None = (
        getattr(validation_config, "idle_timeout_seconds", None)
        if validation_config is not None
        else None
    )
    idle_timeout_seconds = yaml_idle_timeout_seconds  # None = derive at runtime

    # Compute max_diff_size_kb - only from mala.yaml (no CLI option)
    # None means no limit on diff size for epic verification
    max_diff_size_kb: int | None = (
        getattr(validation_config, "max_diff_size_kb", None)
        if validation_config is not None
        else None
    )

    logger.debug(
        "Derived config: timeout=%ds max_idle_retries=%d idle_timeout_seconds=%s",
        timeout_seconds,
        max_idle_retries,
        idle_timeout_seconds,
    )
    return _DerivedConfig(
        timeout_seconds=timeout_seconds,
        disabled_validations=disabled_validations,
        max_idle_retries=max_idle_retries,
        idle_timeout_seconds=idle_timeout_seconds,
        max_gate_retries=max_gate_retries,
        max_review_retries=max_review_retries,
        max_epic_verification_retries=max_epic_verification_retries,
        max_diff_size_kb=max_diff_size_kb,
        epic_verify_lock_timeout_seconds=epic_verify_lock_timeout_seconds,
        per_issue_review=per_issue_review,
        validation_config=validation_config,
        validation_config_missing=validation_config_missing,
    )


def _get_first_enabled_code_review(
    validation_config: object | None,
) -> object | None:
    """Get first enabled code_review config from validation triggers.

    Args:
        validation_config: A ValidationConfig instance, or None.

    Returns:
        The first CodeReviewConfig with enabled=True, or None if none exists.
    """
    if validation_config is None:
        return None

    triggers = getattr(validation_config, "validation_triggers", None)
    if triggers is None:
        return None

    for trigger_name in ("session_end", "epic_completion", "run_end", "periodic"):
        trigger = getattr(triggers, trigger_name, None)
        if trigger is not None:
            code_review = getattr(trigger, "code_review", None)
            if code_review is not None and getattr(code_review, "enabled", False):
                return code_review
    return None


def _has_new_code_review_config(validation_config: object | None) -> bool:
    """Check if validation_config has any enabled code_review in triggers.

    Args:
        validation_config: A ValidationConfig instance, or None.

    Returns:
        True if any trigger has code_review.enabled=True.
    """
    return _get_first_enabled_code_review(validation_config) is not None


def _extract_reviewer_config(
    validation_config: object,
) -> _ReviewerConfig:
    """Extract reviewer configuration from a ValidationConfig.

    Priority order (only when enabled=True):
    1. per_issue_review - highest priority
    2. First enabled trigger code_review config
    3. Defaults (agent_sdk with timeout=600, model=sonnet)

    When reviewer_type='cerberus', also extracts code_review.cerberus config.

    Args:
        validation_config: A ValidationConfig instance.

    Returns:
        _ReviewerConfig with reviewer settings.
    """
    # Check per_issue_review first (only when enabled)
    if validation_config is not None:
        per_issue_review = getattr(validation_config, "per_issue_review", None)
        if per_issue_review is not None and getattr(per_issue_review, "enabled", False):
            return _ReviewerConfig(
                reviewer_type=getattr(per_issue_review, "reviewer_type", "agent_sdk"),
                agent_sdk_review_timeout=getattr(
                    per_issue_review, "agent_sdk_timeout", 600
                ),
                agent_sdk_reviewer_model=getattr(
                    per_issue_review, "agent_sdk_model", "sonnet"
                ),
                cerberus_config=getattr(per_issue_review, "cerberus", None),
            )

    # Fall back to trigger configs
    code_review = _get_first_enabled_code_review(validation_config)
    if code_review is None:
        # No enabled code_review config - use defaults
        return _ReviewerConfig()

    return _ReviewerConfig(
        reviewer_type=getattr(code_review, "reviewer_type", "agent_sdk"),
        agent_sdk_review_timeout=getattr(code_review, "agent_sdk_timeout", 600),
        agent_sdk_reviewer_model=getattr(code_review, "agent_sdk_model", "sonnet"),
        cerberus_config=getattr(code_review, "cerberus", None),
    )


def _get_reviewer_config(repo_path: Path) -> _ReviewerConfig:
    """Get reviewer configuration from mala.yaml.

    Convenience wrapper that loads mala.yaml and extracts reviewer settings.
    Returns defaults if config is missing or loading fails.

    Note: If you already have a loaded ValidationConfig, use
    _extract_reviewer_config() directly to avoid redundant I/O.

    Args:
        repo_path: Path to the repository.

    Returns:
        _ReviewerConfig with reviewer settings.
    """
    from src.domain.validation.config import ConfigError
    from src.domain.validation.config_loader import ConfigMissingError, load_config

    try:
        validation_config = load_config(repo_path)
        return _extract_reviewer_config(validation_config)
    except ConfigMissingError:
        return _ReviewerConfig()
    except ConfigError as e:
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


def _check_epic_verifier_availability(
    reviewer_type: str = "agent_sdk",
    mala_config: MalaConfig | None = None,
    cerberus_config: CerberusConfig | None = None,
) -> str | None:
    """Check if epic verifier is available for the specified reviewer type.

    Returns the reason verifier is disabled, or None if available.
    For agent_sdk reviewer, always available (no external dependencies).
    For cerberus reviewer, probes `spawn-epic-verify --help` per R5 spec.

    Args:
        reviewer_type: Type of epic verifier ('agent_sdk' or 'cerberus').
        mala_config: MalaConfig with Cerberus settings (bin_path, env).
        cerberus_config: CerberusConfig from epic_verification.cerberus (env).

    Returns:
        Reason verifier is disabled, or None if available.
    """
    import subprocess

    if reviewer_type == "agent_sdk":
        return None  # Always available

    if reviewer_type not in ("cerberus", "agent_sdk"):
        return f"unknown epic verification reviewer_type '{reviewer_type}'"

    # Cerberus reviewer: probe spawn-epic-verify --help
    review_gate_path = (
        mala_config.cerberus_bin_path / "review-gate"
        if mala_config and mala_config.cerberus_bin_path
        else None
    )

    # Build effective env: prefer cerberus_config.env, fall back to mala_config.cerberus_env
    if cerberus_config is not None:
        cerberus_env_dict = dict(cerberus_config.env)
    elif mala_config is not None:
        cerberus_env_dict = dict(mala_config.cerberus_env)
    else:
        cerberus_env_dict = {}

    # Check if review-gate exists
    if review_gate_path is None:
        # No explicit bin_path - check PATH (respecting cerberus env if set)
        if "PATH" in cerberus_env_dict:
            effective_path = (
                cerberus_env_dict["PATH"] + os.pathsep + os.environ.get("PATH", "")
            )
        else:
            effective_path = os.environ.get("PATH", "")
        review_gate_bin = shutil.which("review-gate", path=effective_path)
        if review_gate_bin is None:
            reason = "cerberus plugin not detected (review-gate unavailable)"
            logger.info("Epic verifier disabled: reason=%s", reason)
            return reason
        review_gate_cmd = review_gate_bin
    elif not review_gate_path.exists():
        reason = f"review-gate missing at {review_gate_path}"
        logger.info("Epic verifier disabled: reason=%s", reason)
        return reason
    elif not review_gate_path.is_file():
        reason = f"review-gate path is not a file: {review_gate_path}"
        logger.info("Epic verifier disabled: reason=%s", reason)
        return reason
    elif not os.access(review_gate_path, os.X_OK):
        reason = f"review-gate not executable at {review_gate_path}"
        logger.info("Epic verifier disabled: reason=%s", reason)
        return reason
    else:
        review_gate_cmd = str(review_gate_path)

    # Probe spawn-epic-verify --help to verify subcommand support (R5)
    env = dict(os.environ)
    env.update(cerberus_env_dict)

    try:
        result = subprocess.run(
            [review_gate_cmd, "spawn-epic-verify", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
            env=env,
        )
        if result.returncode != 0:
            stderr_snippet = result.stderr[:200] if result.stderr else "(no output)"
            reason = (
                f"Cerberus plugin does not support epic verification: {stderr_snippet}. "
                "Update plugin or use reviewer_type: agent_sdk."
            )
            logger.info("Epic verifier disabled: reason=%s", reason)
            return reason
    except FileNotFoundError:
        reason = "cerberus plugin not detected (review-gate unavailable)"
        logger.info("Epic verifier disabled: reason=%s", reason)
        return reason
    except subprocess.TimeoutExpired:
        reason = "review-gate spawn-epic-verify --help timed out"
        logger.info("Epic verifier disabled: reason=%s", reason)
        return reason
    except OSError as e:
        reason = f"Failed to probe epic verification support: {e}"
        logger.info("Epic verifier disabled: reason=%s", reason)
        return reason

    return None


def _create_epic_verification_model(
    reviewer_type: str,
    repo_path: Path,
    timeout_ms: int,
    mala_config: MalaConfig | None = None,
    cerberus_config: CerberusConfig | None = None,
) -> EpicVerificationModel:
    """Create the epic verification model based on reviewer type.

    Args:
        reviewer_type: Type of verifier ('agent_sdk' or 'cerberus').
        repo_path: Path to the repository.
        timeout_ms: Timeout in milliseconds for verification.
        mala_config: MalaConfig with Cerberus settings (for cerberus reviewer).
        cerberus_config: Optional CerberusConfig from epic_verification.cerberus.

    Returns:
        EpicVerificationModel implementation.

    Raises:
        ValueError: If reviewer_type is unknown.
    """
    from src.core.models import RetryConfig
    from src.infra.epic_verifier import ClaudeEpicVerificationModel

    if reviewer_type == "cerberus":
        from src.infra.clients.cerberus_epic_verifier import CerberusEpicVerifier

        # Prefer cerberus_config from epic_verification block, fall back to mala_config
        if cerberus_config is not None:
            timeout_seconds = cerberus_config.timeout
            env = dict(cerberus_config.env)
            spawn_args = cerberus_config.spawn_args
            wait_args = cerberus_config.wait_args
        elif mala_config is not None:
            timeout_seconds = timeout_ms // 1000
            env = dict(mala_config.cerberus_env)
            spawn_args = ()
            wait_args = ()
        else:
            timeout_seconds = timeout_ms // 1000
            env = {}
            spawn_args = ()
            wait_args = ()

        bin_path = mala_config.cerberus_bin_path if mala_config else None

        return cast(
            "EpicVerificationModel",
            CerberusEpicVerifier(
                repo_path=repo_path,
                bin_path=bin_path,
                timeout=timeout_seconds,
                spawn_args=spawn_args,
                wait_args=wait_args,
                env=env if env else None,
            ),
        )

    if reviewer_type != "agent_sdk":
        raise ValueError(
            f"Unknown epic verification reviewer_type '{reviewer_type}', "
            f"expected 'agent_sdk' or 'cerberus'"
        )

    # agent_sdk - use existing ClaudeEpicVerificationModel
    return cast(
        "EpicVerificationModel",
        ClaudeEpicVerificationModel(
            timeout_ms=timeout_ms,
            retry_config=RetryConfig(),
            repo_path=repo_path,
        ),
    )


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

    For Cerberus reviewer, prefers settings from code_review.cerberus in mala.yaml,
    falling back to mala_config.cerberus_* env vars for backward compatibility.

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
    from src.infra.clients.cerberus_gate_cli import CerberusGateCLI
    from src.infra.clients.cerberus_review import DefaultReviewer
    from src.infra.sdk_adapter import SDKClientFactory

    if reviewer_config.reviewer_type == "cerberus":
        logger.info("Using DefaultReviewer (Cerberus) for code review")
        # Prefer code_review.cerberus settings from mala.yaml, fall back to env vars
        cerb = reviewer_config.cerberus_config
        if cerb is not None:
            spawn_args = cerb.spawn_args
            wait_args = cerb.wait_args
            # Inject cerberus.timeout into wait_args if not already specified
            if CerberusGateCLI.extract_wait_timeout(wait_args) is None:
                wait_args = ("--timeout", str(cerb.timeout), *wait_args)
            env = dict(cerb.env)
        else:
            spawn_args = mala_config.cerberus_spawn_args
            wait_args = mala_config.cerberus_wait_args
            env = dict(mala_config.cerberus_env)
        return cast(
            "CodeReviewer",
            DefaultReviewer(
                repo_path=repo_path,
                bin_path=mala_config.cerberus_bin_path,
                spawn_args=spawn_args,
                wait_args=wait_args,
                env=env,
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
    epic_verifier_reviewer_type: str = "agent_sdk",
    epic_verifier_cerberus_config: CerberusConfig | None = None,
    epic_verifier_timeout_seconds: int = 600,
    epic_verifier_retry_policy: VerificationRetryPolicy | None = None,
) -> tuple[
    IssueProvider,
    CodeReviewer,
    GateChecker,
    LogProvider,
    TelemetryProvider,
    MalaEventSink,
    EpicVerifierProtocol | None,
    CommandRunnerPort,
    EnvConfigPort,
    LockManagerPort,
]:
    """Build all dependencies, using provided ones or creating defaults.

    Args:
        config: Orchestrator configuration.
        mala_config: MalaConfig with API keys.
        derived: Derived configuration values.
        deps: Optional pre-built dependencies.
        reviewer_config: Pre-loaded reviewer configuration from mala.yaml.
        epic_verifier_reviewer_type: Type of epic verifier ('agent_sdk' or 'cerberus').
        epic_verifier_cerberus_config: Optional CerberusConfig for epic_verification.cerberus.
        epic_verifier_timeout_seconds: Timeout for epic verification (from config).
        epic_verifier_retry_policy: Per-category retry limits for verification failures.

    Returns:
        Tuple of all required dependencies.
    """
    from src.domain.evidence_check import EvidenceCheck
    from src.infra.clients.beads_client import BeadsClient
    from src.infra.epic_verifier import EpicVerifier
    from src.infra.io.console_sink import ConsoleEventSink
    from src.infra.io.session_log_parser import FileSystemLogProvider
    from src.infra.telemetry import NullTelemetryProvider
    from src.infra.tools.command_runner import CommandRunner
    from src.infra.tools.env import EnvConfig
    from src.infra.tools.locking import LockManager

    # Get resolved path
    repo_path = config.repo_path.resolve()

    # Command runner (shared by components that need it)
    command_runner: CommandRunnerPort
    if deps is not None and deps.command_runner is not None:
        command_runner = deps.command_runner
    else:
        command_runner = CommandRunner(cwd=repo_path)

    # Env config
    env_config: EnvConfigPort
    if deps is not None and deps.env_config is not None:
        env_config = deps.env_config
    else:
        env_config = EnvConfig()

    # Lock manager
    lock_manager: LockManagerPort
    if deps is not None and deps.lock_manager is not None:
        lock_manager = deps.lock_manager
    else:
        lock_manager = LockManager()

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
        # Check epic verifier availability based on reviewer_type
        epic_unavailable_reason = _check_epic_verifier_availability(
            epic_verifier_reviewer_type,
            mala_config=mala_config,
            cerberus_config=epic_verifier_cerberus_config,
        )
        if epic_unavailable_reason is None:
            verification_model = _create_epic_verification_model(
                reviewer_type=epic_verifier_reviewer_type,
                repo_path=repo_path,
                timeout_ms=epic_verifier_timeout_seconds * 1000,
                mala_config=mala_config,
                cerberus_config=epic_verifier_cerberus_config,
            )
            epic_verifier = cast(
                "EpicVerifierProtocol",
                EpicVerifier(
                    beads=issue_provider,
                    model=verification_model,
                    repo_path=repo_path,
                    command_runner=command_runner,
                    event_sink=event_sink,
                    lock_manager=lock_manager,
                    max_diff_size_kb=derived.max_diff_size_kb,
                    lock_timeout_seconds=derived.epic_verify_lock_timeout_seconds,
                    reviewer_type=epic_verifier_reviewer_type,
                    retry_policy=epic_verifier_retry_policy,
                ),
            )
        else:
            logger.info(
                "Epic verifier disabled: reason=%s",
                epic_unavailable_reason,
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
        command_runner,
        env_config,
        lock_manager,
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

    reviewer_type is determined from validation_triggers.*.code_review config.
    If no code_review config is enabled, defaults to 'agent_sdk'.

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
    from src.domain.validation.config import ConfigError
    from src.domain.validation.config_loader import ConfigMissingError, load_config
    from src.domain.validation.config_merger import merge_configs
    from src.domain.validation.preset_registry import PresetRegistry
    from src.infra.io.config import MalaConfig

    from .orchestrator import MalaOrchestrator

    # Load ValidationConfig once from mala.yaml for all settings that need to flow
    # to MalaConfig and reviewer configuration (avoids redundant I/O and parsing)
    yaml_claude_settings_sources: tuple[str, ...] | None = None
    reviewer_config = _ReviewerConfig()  # Default
    validation_config = None
    validation_config_missing = False
    try:
        user_config = load_config(config.repo_path)
        # Merge with preset if specified (required for trigger base pool)
        if user_config.preset is not None:
            registry = PresetRegistry()
            preset_config = registry.get(user_config.preset)
            validation_config = merge_configs(preset_config, user_config)
        else:
            validation_config = user_config
        yaml_claude_settings_sources = validation_config.claude_settings_sources
        reviewer_config = _extract_reviewer_config(validation_config)
    except ConfigMissingError:
        # mala.yaml not present - use defaults
        validation_config_missing = True
    except ConfigError:
        # Invalid config (syntax error, schema violation, etc.) - fail fast
        raise

    # Load MalaConfig if not provided
    if mala_config is None:
        mala_config = MalaConfig.from_env(
            validate=False,
            yaml_claude_settings_sources=yaml_claude_settings_sources,
        )

    # Derive computed configuration
    derived = _derive_config(
        config,
        mala_config,
        validation_config=validation_config,
        validation_config_missing=validation_config_missing,
    )

    # Check review availability and update disabled_validations
    review_disabled_reason = _check_review_availability(
        mala_config, derived.disabled_validations, reviewer_config.reviewer_type
    )
    if review_disabled_reason:
        derived.disabled_validations.add("review")
        derived.review_disabled_reason = review_disabled_reason

    # Extract epic_verifier settings from validation_config
    epic_verifier_reviewer_type = (
        validation_config.epic_verification.reviewer_type
        if validation_config is not None
        else "agent_sdk"
    )
    epic_verifier_cerberus_config = (
        validation_config.epic_verification.cerberus
        if validation_config is not None
        else None
    )
    # Compute timeout:
    # - cerberus: use cerberus.timeout if config exists, else generic timeout
    # - agent_sdk: use agent_sdk_timeout
    # Generic timeout serves as fallback for cerberus when cerberus config is absent
    if epic_verifier_reviewer_type == "cerberus" and epic_verifier_cerberus_config:
        epic_verifier_timeout_seconds = epic_verifier_cerberus_config.timeout
    elif epic_verifier_reviewer_type == "cerberus" and validation_config is not None:
        # Cerberus without cerberus config: use generic timeout
        epic_verifier_timeout_seconds = validation_config.epic_verification.timeout
    elif validation_config is not None:
        # agent_sdk: use agent_sdk_timeout
        epic_verifier_timeout_seconds = (
            validation_config.epic_verification.agent_sdk_timeout
        )
    else:
        epic_verifier_timeout_seconds = 600  # default

    # Extract retry_policy for per-category retry limits (R6)
    epic_verifier_retry_policy = (
        validation_config.epic_verification.retry_policy
        if validation_config is not None
        else None
    )

    # Build dependencies (pass reviewer_config to avoid second config load)
    (
        issue_provider,
        code_reviewer,
        gate_checker,
        log_provider,
        telemetry_provider,
        event_sink,
        epic_verifier,
        command_runner,
        env_config,
        lock_manager,
    ) = _build_dependencies(
        config,
        mala_config,
        derived,
        deps,
        reviewer_config,
        epic_verifier_reviewer_type=epic_verifier_reviewer_type,
        epic_verifier_cerberus_config=epic_verifier_cerberus_config,
        epic_verifier_timeout_seconds=epic_verifier_timeout_seconds,
        epic_verifier_retry_policy=epic_verifier_retry_policy,
    )

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
        _command_runner=command_runner,
        _env_config=env_config,
        _lock_manager=lock_manager,
        runs_dir=deps.runs_dir if deps else None,
        lock_releaser=deps.lock_releaser if deps else None,
    )

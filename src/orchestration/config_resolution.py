"""Pure helpers for orchestrator configuration resolution.

Holds the precedence-resolution, timeout policy, and reviewer-type logic
that previously lived in :mod:`src.orchestration.factory`. Every public
function here is pure (no file/network/subprocess I/O); the factory wires
them into the DI flow.

Precedence layers handled:

- ``OrchestratorConfig`` (CLI-derived) > ``ValidationConfig`` (mala.yaml) >
  defaults, for timeout/idle/retry settings.
- ``per_issue_review`` > first enabled trigger ``code_review`` > defaults,
  for reviewer selection.
- Legacy ``MalaConfig.review_timeout`` programmatic override > reviewer
  YAML/default timeout.

The full ``CLI > env > yaml > default`` chain is realized by combining
these helpers with ``MalaConfig.from_env`` (which owns the env vs yaml vs
default tiers for coder/effort/etc.).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.core.constants import (
    DEFAULT_AGENT_SDK_REVIEW_TIMEOUT_SECONDS,
    DEFAULT_CERBERUS_REVIEW_TIMEOUT_SECONDS,
)

from .types import (
    DEFAULT_AGENT_TIMEOUT_MINUTES,
    DEFAULT_MAX_IDLE_RETRIES,
    _DerivedConfig,
)

if TYPE_CHECKING:
    from src.domain.validation.config_types import (
        CerberusConfig,
        ValidationConfig,
    )
    from src.infra.io.config import CodexOptions, MalaConfig

    from .types import OrchestratorConfig

logger = logging.getLogger(__name__)

__all__ = [
    "_DerivedConfig",
    "_ReviewerConfig",
    "_derive_config",
    "_extract_reviewer_config",
    "_get_first_enabled_code_review",
    "_has_new_code_review_config",
    "_resolve_review_timeout_seconds",
    "_resolve_yaml_codex_options",
]


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
    agent_sdk_review_timeout: int = DEFAULT_AGENT_SDK_REVIEW_TIMEOUT_SECONDS
    agent_sdk_reviewer_model: str = "opus"
    cerberus_config: CerberusConfig | None = None


def _resolve_yaml_codex_options(
    yaml_codex_options: object | None,
) -> CodexOptions | None:
    """Convert :class:`CodexOptionsConfig` (yaml) to :class:`CodexOptions` (runtime).

    Returns ``None`` when the yaml block is absent or all fields are unset
    (so :meth:`MalaConfig.from_env` keeps using its defaults). Otherwise
    fills in defaults for any fields the user did not specify.
    """
    if yaml_codex_options is None:
        return None
    from src.domain.validation.config_types import CodexOptionsConfig
    from src.infra.io.config import CodexOptions

    if not isinstance(yaml_codex_options, CodexOptionsConfig):
        return None

    has_value = (
        yaml_codex_options.model is not None
        or yaml_codex_options.effort is not None
        or yaml_codex_options.approval_policy is not None
        or yaml_codex_options.sandbox is not None
        or bool(yaml_codex_options.mcp_servers)
    )
    if not has_value:
        return None

    base = CodexOptions()
    return CodexOptions(
        model=yaml_codex_options.model
        if yaml_codex_options.model is not None
        else base.model,
        effort=yaml_codex_options.effort
        if yaml_codex_options.effort is not None
        else base.effort,
        approval_policy=yaml_codex_options.approval_policy
        if yaml_codex_options.approval_policy is not None
        else base.approval_policy,
        sandbox=yaml_codex_options.sandbox
        if yaml_codex_options.sandbox is not None
        else base.sandbox,
        mcp_servers=yaml_codex_options.mcp_servers,
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
    3. Defaults (agent_sdk with timeout=300, model=opus)

    When reviewer_type='cerberus', also extracts code_review.cerberus config.

    Args:
        validation_config: A ValidationConfig instance.

    Returns:
        _ReviewerConfig with reviewer settings.
    """
    if validation_config is not None:
        per_issue_review = getattr(validation_config, "per_issue_review", None)
        if per_issue_review is not None and getattr(per_issue_review, "enabled", False):
            return _ReviewerConfig(
                reviewer_type=getattr(per_issue_review, "reviewer_type", "agent_sdk"),
                agent_sdk_review_timeout=getattr(
                    per_issue_review,
                    "agent_sdk_timeout",
                    DEFAULT_AGENT_SDK_REVIEW_TIMEOUT_SECONDS,
                ),
                agent_sdk_reviewer_model=getattr(
                    per_issue_review, "agent_sdk_model", "opus"
                ),
                cerberus_config=getattr(per_issue_review, "cerberus", None),
            )

    code_review = _get_first_enabled_code_review(validation_config)
    if code_review is None:
        return _ReviewerConfig()

    return _ReviewerConfig(
        reviewer_type=getattr(code_review, "reviewer_type", "agent_sdk"),
        agent_sdk_review_timeout=getattr(
            code_review,
            "agent_sdk_timeout",
            DEFAULT_AGENT_SDK_REVIEW_TIMEOUT_SECONDS,
        ),
        agent_sdk_reviewer_model=getattr(code_review, "agent_sdk_model", "opus"),
        cerberus_config=getattr(code_review, "cerberus", None),
    )


def _resolve_review_timeout_seconds(
    mala_config: MalaConfig,
    reviewer_config: _ReviewerConfig,
) -> int:
    """Resolve the timeout passed to code reviewers.

    ``MalaConfig.review_timeout`` is a legacy programmatic override. When it is
    still at the old default, use the reviewer-specific YAML/default timeout.
    """
    legacy_default = DEFAULT_CERBERUS_REVIEW_TIMEOUT_SECONDS
    if mala_config.review_timeout != legacy_default:
        return mala_config.review_timeout

    if reviewer_config.reviewer_type == "cerberus":
        if reviewer_config.cerberus_config is not None:
            return reviewer_config.cerberus_config.timeout
        return DEFAULT_CERBERUS_REVIEW_TIMEOUT_SECONDS

    return reviewer_config.agent_sdk_review_timeout


def _derive_config(
    config: OrchestratorConfig,
    mala_config: MalaConfig,
    *,
    validation_config: ValidationConfig | None,
    validation_config_missing: bool,
) -> _DerivedConfig:
    """Derive computed configuration values from config sources.

    Config precedence: OrchestratorConfig (CLI) > ValidationConfig (yaml) > defaults.

    Args:
        config: User-provided orchestrator configuration.
        mala_config: MalaConfig with API keys and feature flags.
        validation_config: Loaded ValidationConfig from startup, if any.
        validation_config_missing: True if mala.yaml was missing at startup.

    Returns:
        _DerivedConfig with computed values.
    """
    # Legacy behavior: CLI 0 is treated as "use default" (skips mala.yaml)
    yaml_timeout: int | None = (
        getattr(validation_config, "timeout_minutes", None)
        if validation_config is not None
        else None
    )
    if config.timeout_minutes:
        timeout_seconds = config.timeout_minutes * 60
    elif config.timeout_minutes == 0:
        timeout_seconds = DEFAULT_AGENT_TIMEOUT_MINUTES * 60
    elif yaml_timeout:
        timeout_seconds = yaml_timeout * 60
    else:
        timeout_seconds = DEFAULT_AGENT_TIMEOUT_MINUTES * 60

    disabled_validations = (
        set(config.disable_validations) if config.disable_validations else set()
    )

    # max_review_retries precedence:
    # 1. per_issue_review.max_retries (only when per_issue_review.enabled=True)
    # 2. First enabled trigger code_review.max_retries
    max_review_retries: int | None = None
    per_issue_review = (
        getattr(validation_config, "per_issue_review", None)
        if validation_config is not None
        else None
    )
    if per_issue_review is not None and getattr(per_issue_review, "enabled", False):
        max_review_retries = getattr(per_issue_review, "max_retries", None)
    if max_review_retries is None:
        code_review = _get_first_enabled_code_review(validation_config)
        if code_review is not None:
            max_review_retries = getattr(code_review, "max_retries", None)

    max_gate_retries: int | None = None
    if validation_config is not None:
        triggers = getattr(validation_config, "validation_triggers", None)
        if triggers is not None:
            session_end = getattr(triggers, "session_end", None)
            if session_end is not None:
                max_gate_retries = getattr(session_end, "max_retries", None)

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

    # None => derive from agent timeout at runtime; 0 => disabled
    yaml_idle_timeout_seconds: float | None = (
        getattr(validation_config, "idle_timeout_seconds", None)
        if validation_config is not None
        else None
    )
    idle_timeout_seconds = yaml_idle_timeout_seconds

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

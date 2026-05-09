"""Shared constants for configuration and validation.

Claude settings sources map to configuration files in the Claude Code settings hierarchy:
- local: .claude/settings.local.json (gitignored, machine-specific)
- project: .claude/settings.json (checked in, shared with team)
- user: ~/.claude/settings.json (user's global settings)
"""

from __future__ import annotations

import logging
from typing import Literal

_logger = logging.getLogger(__name__)

# All valid Claude settings source identifiers
VALID_CLAUDE_SETTINGS_SOURCES: frozenset[str] = frozenset({"local", "project", "user"})

# Default sources to load when not explicitly configured (order matters for override priority)
DEFAULT_CLAUDE_SETTINGS_SOURCES: tuple[str, ...] = ("local", "project")

DEFAULT_CERBERUS_REVIEW_TIMEOUT_SECONDS = 600
DEFAULT_AGENT_SDK_REVIEW_TIMEOUT_SECONDS = 300

# Reasoning-effort values forwarded to supported coder backends. Claude SDK
# types effort as ``low | medium | high | max``, while local Claude and Amp also
# accept ``xhigh``; mala validates that superset at the input boundary.
VALID_EFFORTS: frozenset[str] = frozenset({"low", "medium", "high", "xhigh", "max"})
VALID_AMP_DEEP_EFFORTS: frozenset[str] = frozenset({"low", "medium", "xhigh"})
VALID_AMP_SMART_EFFORTS: frozenset[str] = frozenset({"medium", "high", "xhigh"})
DEFAULT_CLAUDE_EFFORT = "xhigh"
DEFAULT_AMP_EFFORT_BY_MODE: dict[Literal["smart", "rush", "deep"], str | None] = {
    "smart": "xhigh",
    "rush": None,
    "deep": "medium",
}


def default_effort_for(
    *,
    coder: Literal["claude", "amp", "codex"],
    mode: Literal["smart", "rush", "deep"],
) -> str | None:
    """Return mala's default reasoning effort for the selected backend."""
    if coder == "claude":
        return DEFAULT_CLAUDE_EFFORT
    if coder == "codex":
        return None
    return DEFAULT_AMP_EFFORT_BY_MODE[mode]


def validate_amp_effort_for_mode(
    *,
    coder: Literal["claude", "amp", "codex"],
    mode: Literal["smart", "rush", "deep"],
    effort: str | None,
    source: str,
) -> None:
    """Validate effort values that are meaningful to Amp modes."""
    if coder != "amp" or mode == "rush" or effort is None:
        return
    valid_efforts = {
        "smart": VALID_AMP_SMART_EFFORTS,
        "deep": VALID_AMP_DEEP_EFFORTS,
    }[mode]
    if effort not in valid_efforts:
        valid = ", ".join(sorted(valid_efforts)) or "none"
        raise ValueError(
            f"{source}: Amp {mode} mode only accepts effort values: {valid}; "
            f"got '{effort}'"
        )


# Documented frozen fallback for ``codex_app_server.ReasoningEffort`` when the
# SDK is not installed. Mirrors the values exposed by the OpenAI Codex
# app-server SDK (decision #13). Sourced from the SDK at parser-build time
# when available; this fallback is the safety net for environments that have
# not yet installed ``openai-codex-app-server-sdk``.
_FALLBACK_CODEX_EFFORTS: frozenset[str] = frozenset(
    {"minimal", "low", "medium", "high"}
)


def _resolve_codex_efforts() -> tuple[frozenset[str], bool]:
    """Return ``(valid_efforts, sourced_from_sdk)`` for Codex effort validation.

    Lazily imports ``codex_app_server.ReasoningEffort`` at call time so the
    parser stays in sync with the installed SDK version. Falls back to the
    documented frozen list when the SDK is not installed (Phase B can run
    without the codex_app_server dependency); the fallback path emits a
    warning so operators know to install the SDK for authoritative
    validation.
    """
    try:
        # SDK is optional in Phase B; the Codex client lands in Phase C.
        sdk = __import__("codex_app_server")
    except ImportError:
        _logger.warning(
            "codex_app_server SDK not installed; falling back to frozen "
            "ReasoningEffort list %s. Install openai-codex-app-server-sdk "
            "for SDK-sourced validation.",
            sorted(_FALLBACK_CODEX_EFFORTS),
        )
        return _FALLBACK_CODEX_EFFORTS, False

    sdk_enum = getattr(sdk, "ReasoningEffort", None)
    if sdk_enum is None:
        _logger.warning(
            "codex_app_server.ReasoningEffort not found; falling back to "
            "frozen list %s.",
            sorted(_FALLBACK_CODEX_EFFORTS),
        )
        return _FALLBACK_CODEX_EFFORTS, False

    members: set[str] = set()
    for member in sdk_enum:
        value = getattr(member, "value", None)
        if isinstance(value, str):
            members.add(value)
    if not members:
        _logger.warning(
            "codex_app_server.ReasoningEffort exposed no string members; "
            "falling back to frozen list %s.",
            sorted(_FALLBACK_CODEX_EFFORTS),
        )
        return _FALLBACK_CODEX_EFFORTS, False
    return frozenset(members), True


def validate_codex_effort(effort: str | None, *, source: str) -> None:
    """Validate ``effort`` against Codex's ``ReasoningEffort`` enum.

    Mirrors :func:`validate_amp_effort_for_mode`'s shape: ``None`` means no
    explicit value to validate and is always accepted; non-None values must
    match an enum member sourced lazily from ``codex_app_server`` (or the
    documented fallback list when the SDK is absent).
    """
    if effort is None:
        return
    valid, _ = _resolve_codex_efforts()
    if effort not in valid:
        valid_list = ", ".join(sorted(valid))
        raise ValueError(
            f"{source}: Invalid Codex effort '{effort}'. Valid efforts: {valid_list}"
        )


# Codex-coder specific enums (decisions #2, #3). Strict-validated at config
# parse time (issue AC#13). Defaults match the unattended-run posture of
# `--coder codex`: `sandbox=danger-full-access` + `approval_policy=never`
# (the bundled hook is the safety gate).
VALID_CODEX_APPROVAL_POLICIES: frozenset[str] = frozenset(
    {"never", "on-request", "on-failure", "untrusted"}
)
VALID_CODEX_SANDBOXES: frozenset[str] = frozenset(
    {"read-only", "workspace-write", "danger-full-access"}
)
DEFAULT_CODEX_MODEL: str = "gpt-5.5"
DEFAULT_CODEX_EFFORT: str = "medium"
DEFAULT_CODEX_APPROVAL_POLICY: Literal[
    "never", "on-request", "on-failure", "untrusted"
] = "never"
DEFAULT_CODEX_SANDBOX: Literal["read-only", "workspace-write", "danger-full-access"] = (
    "danger-full-access"
)

"""Shared constants for configuration and validation.

Claude settings sources map to configuration files in the Claude Code settings hierarchy:
- local: .claude/settings.local.json (gitignored, machine-specific)
- project: .claude/settings.json (checked in, shared with team)
- user: ~/.claude/settings.json (user's global settings)
"""

from __future__ import annotations
from typing import Literal

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
DEFAULT_CLAUDE_MODEL = "opus[1m]"
DEFAULT_AMP_EFFORT_BY_MODE: dict[Literal["smart", "rush", "deep"], str | None] = {
    "smart": "xhigh",
    "rush": None,
    "deep": "medium",
}


def default_model_for(*, coder: Literal["claude", "amp", "codex"]) -> str | None:
    """Return mala's default model for the selected backend."""
    if coder == "claude":
        return DEFAULT_CLAUDE_MODEL
    if coder == "codex":
        return DEFAULT_CODEX_MODEL
    return None


def default_effort_for(
    *,
    coder: Literal["claude", "amp", "codex"],
    mode: Literal["smart", "rush", "deep"],
) -> str | None:
    """Return mala's default reasoning effort for the selected backend."""
    if coder == "claude":
        return DEFAULT_CLAUDE_EFFORT
    if coder == "codex":
        return "medium"
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
DEFAULT_CODEX_APPROVAL_POLICY: Literal[
    "never", "on-request", "on-failure", "untrusted"
] = "never"
DEFAULT_CODEX_SANDBOX: Literal["read-only", "workspace-write", "danger-full-access"] = (
    "danger-full-access"
)

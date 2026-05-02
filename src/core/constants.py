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

# Reasoning-effort values forwarded to supported coder backends. Claude SDK
# types effort as ``low | medium | high | max``, while local Claude and Amp also
# accept ``xhigh``; mala validates that superset at the input boundary.
VALID_EFFORTS: frozenset[str] = frozenset({"low", "medium", "high", "xhigh", "max"})
VALID_AMP_DEEP_EFFORTS: frozenset[str] = frozenset({"medium", "high", "xhigh"})


def validate_amp_effort_for_mode(
    *,
    coder: Literal["claude", "amp"],
    mode: Literal["smart", "rush", "deep"],
    effort: str | None,
    source: str,
) -> None:
    """Validate effort values that are meaningful to Amp deep mode."""
    if coder != "amp" or mode != "deep" or effort is None:
        return
    if effort not in VALID_AMP_DEEP_EFFORTS:
        valid = ", ".join(sorted(VALID_AMP_DEEP_EFFORTS))
        raise ValueError(
            f"{source}: Amp deep mode only accepts effort values: {valid}; "
            f"got '{effort}'"
        )

"""Shared constants for configuration and validation.

Claude settings sources map to configuration files in the Claude Code settings hierarchy:
- local: .claude/settings.local.json (gitignored, machine-specific)
- project: .claude/settings.json (checked in, shared with team)
- user: ~/.claude/settings.json (user's global settings)
"""

# All valid Claude settings source identifiers
VALID_CLAUDE_SETTINGS_SOURCES: frozenset[str] = frozenset({"local", "project", "user"})

# Default sources to load when not explicitly configured (order matters for override priority)
DEFAULT_CLAUDE_SETTINGS_SOURCES: tuple[str, ...] = ("local", "project")

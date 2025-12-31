"""Hook logic for Claude Agent SDK.

Contains PreToolUse hooks and related constants for blocking dangerous commands
and managing tool restrictions.

This module re-exports all public symbols from submodules for backward compatibility:
- dangerous_commands: Security patterns and blocking hooks
- file_cache: File read caching logic
- lint_cache: Lint command caching logic
- locking: Lock enforcement hooks
"""

from __future__ import annotations

# Re-export from dangerous_commands
from .dangerous_commands import (
    BASH_TOOL_NAMES,
    DANGEROUS_PATTERNS,
    DESTRUCTIVE_GIT_PATTERNS,
    SAFE_GIT_ALTERNATIVES,
    block_dangerous_commands,
    block_morph_replaced_tools,
)

# Re-export from file_cache
from .file_cache import (
    FILE_PATH_KEYS,
    FILE_WRITE_TOOLS,
    CachedFileInfo,
    FileReadCache,
    PreToolUseHook,
    make_file_read_cache_hook,
)

# Re-export from lint_cache
from .lint_cache import (
    LINT_COMMAND_PATTERNS,
    LintCache,
    LintCacheEntry,
    _detect_lint_command,
    _get_git_state,
    make_lint_cache_hook,
)

# Re-export from locking
from .locking import (
    StopHook,
    make_lock_enforcement_hook,
    make_stop_hook,
)

# Re-export MORPH_DISALLOWED_TOOLS from mcp for backward compatibility
from ..mcp import MORPH_DISALLOWED_TOOLS

# Re-export get_lock_holder for test patching compatibility
from ..tools.locking import get_lock_holder

__all__ = [
    "BASH_TOOL_NAMES",
    "DANGEROUS_PATTERNS",
    "DESTRUCTIVE_GIT_PATTERNS",
    "FILE_PATH_KEYS",
    "FILE_WRITE_TOOLS",
    "LINT_COMMAND_PATTERNS",
    "MORPH_DISALLOWED_TOOLS",
    "SAFE_GIT_ALTERNATIVES",
    "CachedFileInfo",
    "FileReadCache",
    "LintCache",
    "LintCacheEntry",
    "PreToolUseHook",
    "StopHook",
    "_detect_lint_command",
    "_get_git_state",
    "block_dangerous_commands",
    "block_morph_replaced_tools",
    "get_lock_holder",
    "make_file_read_cache_hook",
    "make_lint_cache_hook",
    "make_lock_enforcement_hook",
    "make_stop_hook",
]

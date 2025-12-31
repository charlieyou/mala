"""Hook logic for Claude Agent SDK.

Contains PreToolUse hooks and related constants for blocking dangerous commands
and managing tool restrictions.

This package provides:
- Security hooks: block_dangerous_commands, block_morph_replaced_tools
- File caching: FileReadCache, make_file_read_cache_hook
- Lint caching: LintCache, make_lint_cache_hook
- Locking: make_lock_enforcement_hook, make_stop_hook
"""

from __future__ import annotations

# Re-export all public symbols for backward compatibility
from .dangerous_commands import (
    BASH_TOOL_NAMES,
    DANGEROUS_PATTERNS,
    DESTRUCTIVE_GIT_PATTERNS,
    PreToolUseHook,
    SAFE_GIT_ALTERNATIVES,
    block_dangerous_commands,
    block_morph_replaced_tools,
)
from .file_cache import (
    FILE_PATH_KEYS,
    FILE_WRITE_TOOLS,
    CachedFileInfo,
    FileReadCache,
    make_file_read_cache_hook,
)
from .lint_cache import (
    LINT_COMMAND_PATTERNS,
    LintCache,
    LintCacheEntry,
    _detect_lint_command,
    _get_git_state,
    make_lint_cache_hook,
)
from .locking import (
    StopHook,
    get_lock_holder,
    make_lock_enforcement_hook,
    make_stop_hook,
    run_command,
)

# Re-export from mcp for backward compatibility
from ..mcp import MORPH_DISALLOWED_TOOLS

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
    "run_command",
]

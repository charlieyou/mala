"""Hook logic for Claude Agent SDK.

Contains PreToolUse hooks and related constants for blocking dangerous commands
and managing tool restrictions.

This package provides:
- Security hooks: block_dangerous_commands
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
    PostToolUseHook,
    PreToolUseHook,
    SAFE_GIT_ALTERNATIVES,
    block_dangerous_commands,
    block_mala_disallowed_tools,
    deny_pretool_use,
)
from .file_cache import (
    FILE_PATH_KEYS,
    FILE_WRITE_TOOLS,
    CachedFileInfo,
    FileReadCache,
    make_file_read_cache_hook,
)
from .lint_cache import (
    DEFAULT_LINT_TOOLS,
    LintCache,
    LintCacheEntry,
    _detect_lint_command,
    _get_git_state,
    make_lint_cache_hook,
)
from .deadlock import (
    make_lock_event_hook,
)
from .commit_guard import (
    make_commit_guard_hook,
)
from .precompact import (
    make_precompact_hook,
)
from .locking import (
    StopHook,
    get_lock_holder,
    make_lock_enforcement_hook,
    make_stop_hook,
)

__all__ = [
    "BASH_TOOL_NAMES",
    "DANGEROUS_PATTERNS",
    "DEFAULT_LINT_TOOLS",
    "DESTRUCTIVE_GIT_PATTERNS",
    "FILE_PATH_KEYS",
    "FILE_WRITE_TOOLS",
    "SAFE_GIT_ALTERNATIVES",
    "CachedFileInfo",
    "FileReadCache",
    "LintCache",
    "LintCacheEntry",
    "PostToolUseHook",
    "PreToolUseHook",
    "StopHook",
    "_detect_lint_command",
    "_get_git_state",
    "block_dangerous_commands",
    "block_mala_disallowed_tools",
    "deny_pretool_use",
    "get_lock_holder",
    "make_commit_guard_hook",
    "make_file_read_cache_hook",
    "make_lint_cache_hook",
    "make_lock_enforcement_hook",
    "make_lock_event_hook",
    "make_precompact_hook",
    "make_stop_hook",
]

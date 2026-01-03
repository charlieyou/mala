"""Security patterns and hooks for blocking dangerous commands.

Contains patterns for detecting dangerous bash commands and destructive git
operations, plus hooks for enforcing these restrictions.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claude_agent_sdk.types import (
        HookContext,
        PreToolUseHookInput,
        SyncHookJSONOutput,
    )

from ..mcp import MALA_DISALLOWED_TOOLS, MORPH_DISALLOWED_TOOLS

# Type alias for PreToolUse hooks (using string annotations to avoid import)
PreToolUseHook = Callable[
    ["PreToolUseHookInput", str | None, "HookContext"],
    Awaitable["SyncHookJSONOutput"],
]

# Dangerous bash command patterns to block
DANGEROUS_PATTERNS = [
    "rm -rf /",
    "rm -rf ~",
    "rm -rf $HOME",
    ":(){:|:&};:",  # fork bomb
    "mkfs.",
    "dd if=",
    "> /dev/sd",
    "chmod -R 777 /",
    "curl | bash",
    "wget | bash",
    "curl | sh",
    "wget | sh",
]

# Destructive git command patterns to block in multi-agent contexts.
# These operations modify working tree or history in ways that can conflict
# between concurrent agents.
DESTRUCTIVE_GIT_PATTERNS = [
    # Hard reset - discards uncommitted changes silently
    "git reset --hard",
    # Clean - removes untracked files
    "git clean -fd",
    "git clean -f",
    "git clean -df",
    "git clean -d -f",
    # Force checkout - discards local changes
    "git checkout -- .",
    "git checkout -f",
    "git checkout --force",
    # Restore - discards uncommitted changes without confirmation
    "git restore",
    # Rebase - can rewrite history and requires conflict resolution
    "git rebase",
    # Force delete branches
    "git branch -D",
    "git branch -d -f",
    # Stash - hides changes that other agents cannot see
    "git stash",
    # Abort operations - may discard other agents' work in progress
    "git merge --abort",
    "git rebase --abort",
    "git cherry-pick --abort",
]

# Safe alternatives to blocked git operations (for error messages)
SAFE_GIT_ALTERNATIVES: dict[str, str] = {
    "git stash": "commit changes instead: git add . && git commit -m 'WIP: ...'",
    "git reset --hard": "use git checkout <file> to revert specific files, or commit first",
    "git rebase": "use git merge instead, or coordinate with other agents",
    "git checkout -f": "commit or stash changes first (in non-agent context)",
    "git checkout --force": "commit or stash changes first (in non-agent context)",
    "git restore": "commit changes first, or use git diff to review before discarding",
    "git clean -f": "manually remove specific untracked files with rm",
    "git merge --abort": "resolve merge conflicts instead of aborting",
    "git rebase --abort": "resolve rebase conflicts instead of aborting",
    "git cherry-pick --abort": "resolve cherry-pick conflicts instead of aborting",
}

# Tool names that should be treated as bash (case-insensitive matching)
BASH_TOOL_NAMES = frozenset(["bash"])


async def block_dangerous_commands(
    hook_input: PreToolUseHookInput,
    stderr: str | None,
    context: HookContext,
) -> SyncHookJSONOutput:
    """PreToolUse hook to block dangerous bash commands.

    In multi-agent contexts, certain git operations are blocked because they
    can cause conflicts between concurrent agents. Blocked operations include:
    - git stash (all subcommands) - hides changes other agents cannot see
    - git reset --hard - discards uncommitted changes silently
    - git rebase - requires human input and can rewrite history
    - git checkout -f/--force - discards local changes
    - git clean -f - removes untracked files without warning
    - git merge/rebase/cherry-pick --abort - may discard other agents' work

    When a blocked operation is detected, the error message includes a safe
    alternative when available.
    """
    tool_name = hook_input["tool_name"].lower()
    if tool_name not in BASH_TOOL_NAMES:
        return {}  # Allow non-Bash tools

    command = hook_input["tool_input"].get("command", "")

    # Block dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if pattern in command:
            return {
                "decision": "block",
                "reason": f"Blocked dangerous command pattern: {pattern}",
            }

    # Block destructive git patterns with safe alternatives
    for pattern in DESTRUCTIVE_GIT_PATTERNS:
        if pattern in command:
            alternative = SAFE_GIT_ALTERNATIVES.get(pattern, "")
            reason = f"Blocked destructive git command: {pattern}"
            if alternative:
                reason = f"{reason}. Safe alternative: {alternative}"
            return {
                "decision": "block",
                "reason": reason,
            }

    # Block force push to ALL branches (--force-with-lease is allowed as safer alternative)
    if "git push" in command:
        # Allow --force-with-lease (safer alternative)
        if "--force-with-lease" in command:
            pass  # Allow
        elif "--force" in command or "-f " in command:
            return {
                "decision": "block",
                "reason": "Blocked force push (use --force-with-lease if needed)",
            }

    return {}  # Allow the command


async def block_morph_replaced_tools(
    hook_input: PreToolUseHookInput,
    stderr: str | None,
    context: HookContext,
) -> SyncHookJSONOutput:
    """PreToolUse hook to block tools replaced by MorphLLM MCP.

    Note: We use a hook instead of the SDK's `disallowed_tools` parameter because
    it has a known bug where it's sometimes ignored.
    See: https://github.com/anthropics/claude-agent-sdk-python/issues/361
    """
    tool_name = hook_input["tool_name"]
    if tool_name in MORPH_DISALLOWED_TOOLS:
        return {
            "decision": "block",
            "reason": f"Use MorphLLM MCP tool instead of {tool_name}. "
            "Available: edit_file, warpgrep_codebase_search",
        }
    return {}


async def block_mala_disallowed_tools(
    hook_input: PreToolUseHookInput,
    stderr: str | None,
    context: HookContext,
) -> SyncHookJSONOutput:
    """PreToolUse hook to block tools disabled for mala agents.

    Blocks tools that cause excessive token usage without proportional value.
    This hook is always active regardless of MorphLLM configuration.

    Note: We use a hook instead of the SDK's `disallowed_tools` parameter because
    it has a known bug where it's sometimes ignored.
    See: https://github.com/anthropics/claude-agent-sdk-python/issues/361
    """
    tool_name = hook_input["tool_name"]
    if tool_name in MALA_DISALLOWED_TOOLS:
        return {
            "decision": "block",
            "reason": f"Tool {tool_name} is disabled for mala agents to reduce token waste.",
        }
    return {}

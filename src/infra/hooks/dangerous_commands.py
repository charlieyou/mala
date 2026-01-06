"""Security patterns and hooks for blocking dangerous commands.

Contains patterns for detecting dangerous bash commands and destructive git
operations, plus hooks for enforcing these restrictions.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from ..tool_config import MALA_DISALLOWED_TOOLS

# Type alias for PreToolUse hooks (using Any to avoid SDK import)
PreToolUseHook = Callable[
    [Any, str | None, Any],
    Awaitable[dict[str, Any]],
]

# Type alias for PostToolUse hooks (mirrors PreToolUseHook pattern)
PostToolUseHook = Callable[
    [Any, str | None, Any],
    Awaitable[dict[str, Any]],
]


def deny_pretool_use(reason: str) -> dict[str, Any]:
    """Create a hook response that denies a PreToolUse request.

    Args:
        reason: Explanation shown to Claude about why the tool use was denied.

    Returns:
        Hook response dict with hookSpecificOutput in the format expected by
        the Claude Agent SDK.
    """
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


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
    "git reset --mixed",
    "git reset --soft",
    # Soft reset that unstages files
    "git reset HEAD",
    # Clean - removes untracked files
    "git clean -fd",
    "git clean -f",
    "git clean -df",
    "git clean -d -f",
    # Force checkout - discards local changes (catches both "-- ." and "-- <file>")
    "git checkout --",
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
    # Amend - rewrites history
    "git commit --amend",
    # Abort operations - may discard other agents' work in progress
    "git merge --abort",
    "git rebase --abort",
    "git cherry-pick --abort",
    # Worktree removal - can discard uncommitted changes in worktree
    "git worktree remove",
    # Force submodule operations
    "git submodule deinit -f",
]

# Safe alternatives to blocked git operations (for error messages)
SAFE_GIT_ALTERNATIVES: dict[str, str] = {
    "git stash": "commit changes instead: git add . && git commit -m 'WIP: ...'",
    "git reset --hard": "commit first, or use git diff to review changes before discarding",
    "git reset --mixed": "commit staged changes first",
    "git reset --soft": "create a new commit instead of rewriting history",
    "git reset HEAD": "commit staged changes first",
    "git rebase": "use git merge instead, or coordinate with other agents",
    "git checkout --": "commit changes first, or use git diff to review before discarding",
    "git checkout -f": "commit changes first",
    "git checkout --force": "commit changes first",
    "git restore": "commit changes first, or use git diff to review before discarding",
    "git clean -f": "manually remove specific untracked files with rm",
    "git merge --abort": "resolve merge conflicts instead of aborting",
    "git rebase --abort": "resolve rebase conflicts instead of aborting",
    "git cherry-pick --abort": "resolve cherry-pick conflicts instead of aborting",
    "git worktree remove": "commit changes in worktree first",
    "git submodule deinit -f": "use git submodule deinit without -f",
    "git commit --amend": "create a new commit instead of amending history",
}

# Tool names that should be treated as bash (case-insensitive matching)
BASH_TOOL_NAMES = frozenset(["bash"])


async def block_dangerous_commands(
    hook_input: Any,  # noqa: ANN401 - SDK type, avoid import
    stderr: str | None,
    context: Any,  # noqa: ANN401 - SDK type, avoid import
) -> dict[str, Any]:
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
            return deny_pretool_use(f"Blocked dangerous command pattern: {pattern}")

    # Block destructive git patterns with safe alternatives
    for pattern in DESTRUCTIVE_GIT_PATTERNS:
        if pattern in command:
            alternative = SAFE_GIT_ALTERNATIVES.get(pattern, "")
            reason = f"Blocked destructive git command: {pattern}"
            if alternative:
                reason = f"{reason}. Safe alternative: {alternative}"
            return deny_pretool_use(reason)

    # Enforce atomic add+commit and explicit staging (no -a/--all).
    command_lower = command.lower()
    if "git commit" in command_lower:
        if " --all" in command_lower or "git commit -a" in command_lower:
            return deny_pretool_use(
                "Blocked git commit -a/--all. "
                'Stage explicit files: git add <files> && git commit -m "...".'
            )

        commit_index = command_lower.find("git commit")
        add_index = command_lower.find("git add")
        if add_index == -1 or add_index > commit_index:
            return deny_pretool_use(
                "Atomic add+commit required. "
                'Use `git add <files> && git commit -m "..."`.'
            )

    # Block force push to ALL branches (--force-with-lease is allowed as safer alternative)
    if "git push" in command:
        # Allow --force-with-lease (safer alternative)
        if "--force-with-lease" in command:
            pass  # Allow
        elif "--force" in command or "-f " in command:
            return deny_pretool_use(
                "Blocked force push (use --force-with-lease if needed)"
            )

    return {}  # Allow the command


async def block_mala_disallowed_tools(
    hook_input: Any,  # noqa: ANN401 - SDK type, avoid import
    stderr: str | None,
    context: Any,  # noqa: ANN401 - SDK type, avoid import
) -> dict[str, Any]:
    """PreToolUse hook to block tools disabled for mala agents.

    Blocks tools that cause excessive token usage without proportional value.

    Note: We use a hook instead of the SDK's `disallowed_tools` parameter because
    it has a known bug where it's sometimes ignored.
    See: https://github.com/anthropics/claude-agent-sdk-python/issues/361
    """
    tool_name = hook_input["tool_name"]
    if tool_name in MALA_DISALLOWED_TOOLS:
        return deny_pretool_use(
            f"Tool {tool_name} is disabled for mala agents to reduce token waste."
        )
    return {}

"""
Hook logic for Claude Agent SDK.

Contains PreToolUse hooks and related constants for blocking dangerous commands
and managing tool restrictions.
"""

from claude_agent_sdk.types import (
    PreToolUseHookInput,
    HookContext,
    SyncHookJSONOutput,
)

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

# Destructive git command patterns to block
DESTRUCTIVE_GIT_PATTERNS = [
    "git reset --hard",
    "git clean -fd",
    "git clean -f",
    "git checkout -- .",
    "git rebase",
    "git branch -D",
    "git branch -d -f",
]

# Tool names that should be treated as bash (case-insensitive matching)
BASH_TOOL_NAMES = frozenset(["bash"])

# Tools replaced by MorphLLM MCP (use disallowed_tools parameter, not hooks)
MORPH_DISALLOWED_TOOLS = ["Edit", "Grep"]


async def block_dangerous_commands(
    hook_input: PreToolUseHookInput,
    stderr: str | None,
    context: HookContext,
) -> SyncHookJSONOutput:
    """PreToolUse hook to block dangerous bash commands."""
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

    # Block destructive git patterns
    for pattern in DESTRUCTIVE_GIT_PATTERNS:
        if pattern in command:
            return {
                "decision": "block",
                "reason": f"Blocked destructive git command: {pattern}",
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

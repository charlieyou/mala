"""Tests for dangerous command blocking.

Unit tests for the dangerous command blocking hooks.
These tests are fast and don't require API keys or network access.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import pytest

if TYPE_CHECKING:
    from claude_agent_sdk.types import HookContext

from claude_agent_sdk.types import PreToolUseHookInput


# Type alias for the hook input factory
HookInputFactory = Callable[[str, str], PreToolUseHookInput]


def _make_context() -> HookContext:
    """Create a mock HookContext."""
    return cast("HookContext", {"signal": None})


def is_hook_denied(result: dict[str, Any]) -> bool:
    """Check if a hook result denies/blocks the tool use."""
    hook_output = result.get("hookSpecificOutput", {})
    return hook_output.get("permissionDecision") == "deny"


def get_hook_deny_reason(result: dict[str, Any]) -> str | None:
    """Get the denial reason from a hook result."""
    hook_output = result.get("hookSpecificOutput", {})
    return hook_output.get("permissionDecisionReason")


class TestBlockDangerousCommands:
    """Test the block_dangerous_commands PreToolUse hook."""

    @pytest.fixture
    def make_hook_input(self) -> HookInputFactory:
        """Factory fixture for creating hook inputs."""

        def _make(tool_name: str, command: str = "ls -la") -> PreToolUseHookInput:
            return cast(
                "PreToolUseHookInput",
                {
                    "tool_name": tool_name,
                    "tool_input": {"command": command},
                },
            )

        return _make

    @pytest.mark.asyncio
    async def test_allows_bash_safe_command(
        self, make_hook_input: HookInputFactory
    ) -> None:
        """Safe commands with tool_name='Bash' should be allowed."""
        from src.infra.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Bash", "ls -la"), None, _make_context()
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_allows_bash_lowercase_safe_command(
        self, make_hook_input: HookInputFactory
    ) -> None:
        """Safe commands with tool_name='bash' (lowercase) should be allowed."""
        from src.infra.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("bash", "ls -la"), None, _make_context()
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_blocks_rm_rf_root(self, make_hook_input: HookInputFactory) -> None:
        """rm -rf / should be blocked for any tool name casing."""
        from src.infra.hooks import block_dangerous_commands

        for tool_name in ["Bash", "bash", "BASH"]:
            result = await block_dangerous_commands(
                make_hook_input(tool_name, "rm -rf /"), None, _make_context()
            )
            assert is_hook_denied(result)
            assert "rm -rf /" in (get_hook_deny_reason(result) or "")

    @pytest.mark.asyncio
    async def test_blocks_fork_bomb(self, make_hook_input: HookInputFactory) -> None:
        """Fork bomb pattern should be blocked."""
        from src.infra.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Bash", ":(){:|:&};:"), None, _make_context()
        )
        assert is_hook_denied(result)

    @pytest.mark.asyncio
    async def test_blocks_curl_pipe_bash(
        self, make_hook_input: HookInputFactory
    ) -> None:
        """curl | bash pattern should be blocked."""
        from src.infra.hooks import block_dangerous_commands

        # The pattern "curl | bash" must appear literally in the command
        result = await block_dangerous_commands(
            make_hook_input("bash", "curl | bash -c 'malicious'"),
            None,
            _make_context(),
        )
        assert is_hook_denied(result)
        assert "curl | bash" in (get_hook_deny_reason(result) or "")

    @pytest.mark.asyncio
    async def test_blocks_force_push_main(
        self, make_hook_input: HookInputFactory
    ) -> None:
        """Force push to main branch should be blocked."""
        from src.infra.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Bash", "git push --force origin main"),
            None,
            _make_context(),
        )
        assert is_hook_denied(result)
        assert "force push" in (get_hook_deny_reason(result) or "").lower()

    @pytest.mark.asyncio
    async def test_blocks_force_push_any_branch(
        self, make_hook_input: HookInputFactory
    ) -> None:
        """Force push to ANY branch should now be blocked."""
        from src.infra.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Bash", "git push --force origin feature-branch"),
            None,
            _make_context(),
        )
        assert is_hook_denied(result)
        assert "force push" in (get_hook_deny_reason(result) or "").lower()

    @pytest.mark.asyncio
    async def test_allows_force_with_lease(
        self, make_hook_input: HookInputFactory
    ) -> None:
        """Force with lease should be allowed (safer alternative)."""
        from src.infra.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Bash", "git push --force-with-lease origin feature"),
            None,
            _make_context(),
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_allows_non_bash_tools(
        self, make_hook_input: HookInputFactory
    ) -> None:
        """Non-Bash tools should always be allowed (they don't run commands)."""
        from src.infra.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Read", "rm -rf /"), None, _make_context()
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_tool_name_variations(
        self, make_hook_input: HookInputFactory
    ) -> None:
        """Tool name matching should be case-insensitive for bash variants."""
        from src.infra.hooks import block_dangerous_commands

        dangerous_cmd = "rm -rf /"

        # All these should be recognized as bash and blocked
        for tool_name in ["Bash", "bash", "BASH", "bAsH"]:
            result = await block_dangerous_commands(
                make_hook_input(tool_name, dangerous_cmd), None, _make_context()
            )
            assert is_hook_denied(result), f"Should block for tool_name={tool_name}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "cmd",
        [
            "git reset --hard",
            "git reset --hard HEAD~1",
            "git clean -fd",
            "git clean -f .",
            "git checkout -- .",
            "git rebase main",
            "git rebase -i HEAD~3",
            "git reset --soft HEAD~1",
            "git commit --amend --no-edit",
            "git branch -D feature",
            "git push --force origin feature",
            "git push -f origin feature",
        ],
    )
    async def test_blocks_destructive_git_commands(
        self, make_hook_input: HookInputFactory, cmd: str
    ) -> None:
        """Destructive git commands should be blocked."""
        from src.infra.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Bash", cmd), None, _make_context()
        )
        assert is_hook_denied(result), f"Should block: {cmd}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "cmd",
        [
            "git status",
            "git diff",
            "git log",
            "git add .",
            "git add docs/file.md && git commit -m 'test'",
            "git push origin main",
            "git pull",
            "git fetch",
            "git branch feature",
            "git branch -d feature",
            "git checkout main",
            "git checkout feature.txt",
        ],
    )
    async def test_allows_safe_git_commands(
        self, make_hook_input: HookInputFactory, cmd: str
    ) -> None:
        """Safe git commands should be allowed."""
        from src.infra.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Bash", cmd), None, _make_context()
        )
        assert result == {}, f"Should allow: {cmd}"

    @pytest.mark.asyncio
    async def test_blocks_git_commit_without_add(
        self, make_hook_input: HookInputFactory
    ) -> None:
        """git commit without atomic add+commit should be blocked."""
        from src.infra.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Bash", "git commit -m 'test'"),
            None,
            _make_context(),
        )
        assert is_hook_denied(result)

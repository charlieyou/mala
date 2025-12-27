"""Unit tests for PreToolUse hooks in src/hooks.py."""

from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import pytest

from claude_agent_sdk.types import PreToolUseHookInput, HookContext

from src.hooks import (
    make_lock_enforcement_hook,
    block_dangerous_commands,
    DESTRUCTIVE_GIT_PATTERNS,
    SAFE_GIT_ALTERNATIVES,
    FILE_WRITE_TOOLS,
)


def make_hook_input(tool_name: str, tool_input: dict[str, Any]) -> PreToolUseHookInput:
    """Create a mock PreToolUseHookInput."""
    return cast(
        "PreToolUseHookInput",
        {
            "tool_name": tool_name,
            "tool_input": tool_input,
        },
    )


def make_context(agent_id: str = "test-agent") -> HookContext:
    """Create a mock HookContext."""
    return cast("HookContext", {"agent_id": agent_id})


class TestMakeLockEnforcementHook:
    """Tests for the make_lock_enforcement_hook factory function."""

    @pytest.mark.asyncio
    async def test_captures_agent_id_via_closure(self, tmp_path: Path) -> None:
        """Hook created by factory should use the captured agent_id."""
        test_file = str(tmp_path / "test.py")
        hook = make_lock_enforcement_hook("captured-agent-id")
        hook_input = make_hook_input("Write", {"file_path": test_file})
        context = make_context()  # Context agent_id is ignored

        with patch(
            "src.hooks.get_lock_holder", return_value="captured-agent-id"
        ) as mock:
            result = await hook(hook_input, None, context)

        assert result == {}  # Allowed because lock holder matches captured ID
        mock.assert_called_once_with(test_file, repo_namespace=None)

    @pytest.mark.asyncio
    async def test_blocks_when_different_agent_holds_lock(self, tmp_path: Path) -> None:
        """Factory-created hook should block when another agent holds lock."""
        test_file = str(tmp_path / "test.py")
        hook = make_lock_enforcement_hook("my-agent")
        hook_input = make_hook_input("Write", {"file_path": test_file})
        context = make_context()

        with patch("src.hooks.get_lock_holder", return_value="other-agent"):
            result = await hook(hook_input, None, context)

        assert result["decision"] == "block"
        assert "other-agent" in result["reason"]

    @pytest.mark.asyncio
    async def test_blocks_when_no_lock_exists(self, tmp_path: Path) -> None:
        """Factory-created hook should block when file is not locked."""
        test_file = str(tmp_path / "test.py")
        hook = make_lock_enforcement_hook("my-agent")
        hook_input = make_hook_input("Write", {"file_path": test_file})
        context = make_context()

        with patch("src.hooks.get_lock_holder", return_value=None):
            result = await hook(hook_input, None, context)

        assert result["decision"] == "block"
        assert "not locked" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_allows_non_write_tools(self) -> None:
        """Non-write tools should be allowed without lock check."""
        hook = make_lock_enforcement_hook("test-agent")
        hook_input = make_hook_input("Bash", {"command": "ls -la"})
        context = make_context()

        result = await hook(hook_input, None, context)

        assert result == {}  # Empty dict means allow

    @pytest.mark.asyncio
    async def test_handles_edit_file_mcp_tool(self, tmp_path: Path) -> None:
        """MCP edit_file tool should also check lock ownership."""
        test_file = str(tmp_path / "test.py")
        hook_input = make_hook_input(
            "mcp__morphllm__edit_file", {"path": test_file, "code_edit": "..."}
        )
        agent_id = "test-agent"
        hook = make_lock_enforcement_hook(agent_id)
        context = make_context(agent_id)

        with patch("src.hooks.get_lock_holder", return_value=agent_id):
            result = await hook(hook_input, None, context)

        assert result == {}  # Allowed when agent holds lock

    @pytest.mark.asyncio
    async def test_handles_notebook_edit_tool(self, tmp_path: Path) -> None:
        """NotebookEdit tool should also check lock ownership."""
        notebook_file = str(tmp_path / "notebook.ipynb")
        hook_input = make_hook_input(
            "NotebookEdit",
            {"notebook_path": notebook_file, "new_source": "print('hello')"},
        )
        agent_id = "notebook-agent"
        hook = make_lock_enforcement_hook(agent_id)
        context = make_context(agent_id)

        with patch("src.hooks.get_lock_holder", return_value=agent_id):
            result = await hook(hook_input, None, context)

        assert result == {}  # Allowed

    @pytest.mark.asyncio
    async def test_file_write_tools_constant_contains_expected_tools(self) -> None:
        """FILE_WRITE_TOOLS should contain expected write tools."""
        # These are the tools we expect to be file-write tools
        expected_tools = {"Write", "NotebookEdit", "mcp__morphllm__edit_file"}
        assert expected_tools.issubset(FILE_WRITE_TOOLS)

    @pytest.mark.asyncio
    async def test_handles_missing_file_path_gracefully(self) -> None:
        """Should handle malformed tool input without crashing."""
        hook = make_lock_enforcement_hook("test-agent")
        hook_input = make_hook_input("Write", {})  # Missing file_path
        context = make_context()

        # Should not raise, should allow (or handle gracefully)
        result = await hook(hook_input, None, context)
        # Without a path to check, we allow (can't block)
        assert result == {}

    @pytest.mark.asyncio
    async def test_repo_path_passed_to_get_lock_holder(self, tmp_path: Path) -> None:
        """repo_path should be passed to get_lock_holder as repo_namespace."""
        test_file = str(tmp_path / "test.py")
        repo_path = "/home/user/my-repo"
        hook = make_lock_enforcement_hook("my-agent", repo_path=repo_path)
        hook_input = make_hook_input("Write", {"file_path": test_file})
        context = make_context()

        with patch("src.hooks.get_lock_holder", return_value="my-agent") as mock:
            result = await hook(hook_input, None, context)

        assert result == {}  # Allowed
        mock.assert_called_once_with(test_file, repo_namespace=repo_path)


class TestBlockDangerousCommands:
    """Tests for block_dangerous_commands hook."""

    @pytest.mark.asyncio
    async def test_allows_safe_git_commands(self) -> None:
        """Safe git commands should be allowed."""
        safe_commands = [
            "git status",
            "git log",
            "git diff",
            "git add .",
            "git commit -m 'test'",
            "git pull",
            "git fetch",
            "git branch feature",
            "git checkout feature",
            "git merge feature",
        ]
        context = make_context()

        for cmd in safe_commands:
            hook_input = make_hook_input("Bash", {"command": cmd})
            result = await block_dangerous_commands(hook_input, None, context)
            assert result == {}, f"Expected {cmd!r} to be allowed"

    @pytest.mark.asyncio
    async def test_blocks_git_stash(self) -> None:
        """git stash (all subcommands) should be blocked."""
        stash_commands = [
            "git stash",
            "git stash push",
            "git stash pop",
            "git stash apply",
            "git stash list",
            "git stash drop",
        ]
        context = make_context()

        for cmd in stash_commands:
            hook_input = make_hook_input("Bash", {"command": cmd})
            result = await block_dangerous_commands(hook_input, None, context)
            assert result.get("decision") == "block", f"Expected {cmd!r} to be blocked"
            assert "git stash" in result["reason"]

    @pytest.mark.asyncio
    async def test_blocks_git_reset_hard(self) -> None:
        """git reset --hard should be blocked."""
        hook_input = make_hook_input("Bash", {"command": "git reset --hard HEAD~1"})
        context = make_context()

        result = await block_dangerous_commands(hook_input, None, context)

        assert result["decision"] == "block"
        assert "git reset --hard" in result["reason"]

    @pytest.mark.asyncio
    async def test_blocks_git_rebase(self) -> None:
        """git rebase (all forms) should be blocked."""
        rebase_commands = [
            "git rebase main",
            "git rebase -i HEAD~3",
            "git rebase --onto main feature",
        ]
        context = make_context()

        for cmd in rebase_commands:
            hook_input = make_hook_input("Bash", {"command": cmd})
            result = await block_dangerous_commands(hook_input, None, context)
            assert result.get("decision") == "block", f"Expected {cmd!r} to be blocked"
            assert "git rebase" in result["reason"]

    @pytest.mark.asyncio
    async def test_blocks_force_checkout(self) -> None:
        """git checkout -f/--force should be blocked."""
        force_checkouts = [
            "git checkout -f",
            "git checkout --force",
            "git checkout -f main",
            "git checkout --force feature",
        ]
        context = make_context()

        for cmd in force_checkouts:
            hook_input = make_hook_input("Bash", {"command": cmd})
            result = await block_dangerous_commands(hook_input, None, context)
            assert result.get("decision") == "block", f"Expected {cmd!r} to be blocked"

    @pytest.mark.asyncio
    async def test_blocks_git_clean(self) -> None:
        """git clean -f should be blocked."""
        clean_commands = [
            "git clean -f",
            "git clean -fd",
            "git clean -df",
            "git clean -d -f",
        ]
        context = make_context()

        for cmd in clean_commands:
            hook_input = make_hook_input("Bash", {"command": cmd})
            result = await block_dangerous_commands(hook_input, None, context)
            assert result.get("decision") == "block", f"Expected {cmd!r} to be blocked"

    @pytest.mark.asyncio
    async def test_blocks_abort_operations(self) -> None:
        """git merge/rebase/cherry-pick --abort should be blocked."""
        abort_commands = [
            "git merge --abort",
            "git rebase --abort",
            "git cherry-pick --abort",
        ]
        context = make_context()

        for cmd in abort_commands:
            hook_input = make_hook_input("Bash", {"command": cmd})
            result = await block_dangerous_commands(hook_input, None, context)
            assert result.get("decision") == "block", f"Expected {cmd!r} to be blocked"

    @pytest.mark.asyncio
    async def test_includes_safe_alternatives_in_error(self) -> None:
        """Error messages should include safe alternatives when available."""
        context = make_context()

        # Test git stash - should suggest commit instead
        hook_input = make_hook_input("Bash", {"command": "git stash"})
        result = await block_dangerous_commands(hook_input, None, context)
        assert "commit" in result["reason"].lower()

        # Test git reset --hard - should suggest checkout for specific files
        hook_input = make_hook_input("Bash", {"command": "git reset --hard"})
        result = await block_dangerous_commands(hook_input, None, context)
        assert (
            "checkout" in result["reason"].lower()
            or "commit" in result["reason"].lower()
        )

        # Test git rebase - should suggest merge
        hook_input = make_hook_input("Bash", {"command": "git rebase main"})
        result = await block_dangerous_commands(hook_input, None, context)
        assert "merge" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_allows_non_bash_tools(self) -> None:
        """Non-Bash tools should not be affected by the hook."""
        context = make_context()
        hook_input = make_hook_input("Write", {"file_path": "/test.py"})

        result = await block_dangerous_commands(hook_input, None, context)

        assert result == {}

    @pytest.mark.asyncio
    async def test_blocks_force_push(self) -> None:
        """git push --force should be blocked."""
        context = make_context()
        hook_input = make_hook_input(
            "Bash", {"command": "git push --force origin main"}
        )

        result = await block_dangerous_commands(hook_input, None, context)

        assert result["decision"] == "block"
        assert "force push" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_allows_force_with_lease(self) -> None:
        """git push --force-with-lease should be allowed (safer alternative)."""
        context = make_context()
        hook_input = make_hook_input(
            "Bash", {"command": "git push --force-with-lease origin main"}
        )

        result = await block_dangerous_commands(hook_input, None, context)

        assert result == {}


class TestDestructiveGitPatternsConstant:
    """Tests for DESTRUCTIVE_GIT_PATTERNS constant coverage."""

    def test_contains_all_required_patterns(self) -> None:
        """DESTRUCTIVE_GIT_PATTERNS should contain all required blocked operations."""
        required = [
            "git stash",
            "git reset --hard",
            "git rebase",
            "git checkout -f",
            "git checkout --force",
            "git clean -f",
            "git merge --abort",
            "git rebase --abort",
            "git cherry-pick --abort",
        ]
        for pattern in required:
            assert any(pattern in p for p in DESTRUCTIVE_GIT_PATTERNS), (
                f"Missing required pattern: {pattern}"
            )


class TestSafeGitAlternatives:
    """Tests for SAFE_GIT_ALTERNATIVES documentation."""

    def test_provides_alternatives_for_common_operations(self) -> None:
        """SAFE_GIT_ALTERNATIVES should have alternatives for common blocked ops."""
        assert "git stash" in SAFE_GIT_ALTERNATIVES
        assert "git reset --hard" in SAFE_GIT_ALTERNATIVES
        assert "git rebase" in SAFE_GIT_ALTERNATIVES

    def test_alternatives_are_non_empty_strings(self) -> None:
        """All alternatives should be non-empty strings."""
        for pattern, alternative in SAFE_GIT_ALTERNATIVES.items():
            assert isinstance(alternative, str), (
                f"Alternative for {pattern} is not a string"
            )
            assert len(alternative) > 0, f"Alternative for {pattern} is empty"

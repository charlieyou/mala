"""Unit tests for PreToolUse hooks in src/hooks.py."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.hooks import (
    make_lock_enforcement_hook,
    FILE_WRITE_TOOLS,
)


def make_hook_input(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Create a mock PreToolUseHookInput."""
    return {
        "tool_name": tool_name,
        "tool_input": tool_input,
    }


def make_context(agent_id: str = "test-agent") -> dict[str, Any]:
    """Create a mock HookContext."""
    return {"agent_id": agent_id}


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

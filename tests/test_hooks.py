"""Unit tests for PreToolUse hooks in src/hooks.py."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.hooks import (
    block_unlocked_file_writes,
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


class TestBlockUnlockedFileWrites:
    """Tests for the block_unlocked_file_writes PreToolUse hook."""

    @pytest.mark.asyncio
    async def test_allows_non_write_tools(self):
        """Non-write tools should be allowed without lock check."""
        hook_input = make_hook_input("Bash", {"command": "ls -la"})
        context = make_context()

        result = await block_unlocked_file_writes(hook_input, None, context)

        assert result == {}  # Empty dict means allow

    @pytest.mark.asyncio
    async def test_allows_write_when_agent_holds_lock(self, tmp_path: Path):
        """Write tool should be allowed when agent holds the lock."""
        test_file = str(tmp_path / "test.py")
        hook_input = make_hook_input("Write", {"file_path": test_file})
        agent_id = "test-agent-123"
        context = make_context(agent_id)

        with patch(
            "src.hooks.get_lock_holder", return_value=agent_id
        ) as mock_get_holder:
            result = await block_unlocked_file_writes(hook_input, None, context)

        mock_get_holder.assert_called_once_with(test_file)
        assert result == {}  # Allowed

    @pytest.mark.asyncio
    async def test_blocks_write_when_no_lock_held(self, tmp_path: Path):
        """Write tool should be blocked when no one holds the lock."""
        test_file = str(tmp_path / "test.py")
        hook_input = make_hook_input("Write", {"file_path": test_file})
        context = make_context("test-agent")

        with patch("src.hooks.get_lock_holder", return_value=None):
            result = await block_unlocked_file_writes(hook_input, None, context)

        assert result["decision"] == "block"
        assert test_file in result["reason"]
        assert "not locked" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_blocks_write_when_different_agent_holds_lock(self, tmp_path: Path):
        """Write tool should be blocked when another agent holds the lock."""
        test_file = str(tmp_path / "test.py")
        hook_input = make_hook_input("Write", {"file_path": test_file})
        context = make_context("agent-1")

        with patch("src.hooks.get_lock_holder", return_value="agent-2"):
            result = await block_unlocked_file_writes(hook_input, None, context)

        assert result["decision"] == "block"
        assert test_file in result["reason"]
        assert "agent-2" in result["reason"]

    @pytest.mark.asyncio
    async def test_handles_edit_file_mcp_tool(self, tmp_path: Path):
        """MCP edit_file tool should also check lock ownership."""
        test_file = str(tmp_path / "test.py")
        hook_input = make_hook_input(
            "mcp__morphllm__edit_file", {"path": test_file, "code_edit": "..."}
        )
        agent_id = "test-agent"
        context = make_context(agent_id)

        with patch("src.hooks.get_lock_holder", return_value=agent_id):
            result = await block_unlocked_file_writes(hook_input, None, context)

        assert result == {}  # Allowed when agent holds lock

    @pytest.mark.asyncio
    async def test_handles_notebook_edit_tool(self, tmp_path: Path):
        """NotebookEdit tool should also check lock ownership."""
        notebook_file = str(tmp_path / "notebook.ipynb")
        hook_input = make_hook_input(
            "NotebookEdit",
            {"notebook_path": notebook_file, "new_source": "print('hello')"},
        )
        agent_id = "notebook-agent"
        context = make_context(agent_id)

        with patch("src.hooks.get_lock_holder", return_value=agent_id):
            result = await block_unlocked_file_writes(hook_input, None, context)

        assert result == {}  # Allowed

    @pytest.mark.asyncio
    async def test_blocks_notebook_edit_without_lock(self, tmp_path: Path):
        """NotebookEdit should be blocked without lock."""
        notebook_file = str(tmp_path / "notebook.ipynb")
        hook_input = make_hook_input(
            "NotebookEdit",
            {"notebook_path": notebook_file, "new_source": "print('hello')"},
        )
        context = make_context("agent-a")

        with patch("src.hooks.get_lock_holder", return_value="agent-b"):
            result = await block_unlocked_file_writes(hook_input, None, context)

        assert result["decision"] == "block"

    @pytest.mark.asyncio
    async def test_file_write_tools_constant_contains_expected_tools(self):
        """FILE_WRITE_TOOLS should contain expected write tools."""
        # These are the tools we expect to be file-write tools
        expected_tools = {"Write", "NotebookEdit", "mcp__morphllm__edit_file"}
        assert expected_tools.issubset(FILE_WRITE_TOOLS)

    @pytest.mark.asyncio
    async def test_extracts_path_correctly_for_each_tool_type(self, tmp_path: Path):
        """Each tool type should have its path extracted correctly."""
        agent_id = "test-agent"
        context = make_context(agent_id)

        test_cases = [
            ("Write", {"file_path": "/a/b/c.py", "content": "x"}),
            ("mcp__morphllm__edit_file", {"path": "/d/e/f.py", "code_edit": "y"}),
            ("NotebookEdit", {"notebook_path": "/g/h/i.ipynb", "new_source": "z"}),
        ]

        for tool_name, tool_input in test_cases:
            hook_input = make_hook_input(tool_name, tool_input)
            with patch("src.hooks.get_lock_holder", return_value=agent_id):
                result = await block_unlocked_file_writes(hook_input, None, context)
                assert result == {}, f"Failed for tool {tool_name}"

    @pytest.mark.asyncio
    async def test_handles_missing_file_path_gracefully(self):
        """Should handle malformed tool input without crashing."""
        hook_input = make_hook_input("Write", {})  # Missing file_path
        context = make_context()

        # Should not raise, should allow (or handle gracefully)
        result = await block_unlocked_file_writes(hook_input, None, context)
        # Without a path to check, we allow (can't block)
        assert result == {}

    @pytest.mark.asyncio
    async def test_context_agent_id_used_for_lock_check(self, tmp_path: Path):
        """The agent_id from context should be used for ownership check."""
        test_file = str(tmp_path / "test.py")
        hook_input = make_hook_input("Write", {"file_path": test_file})
        context = make_context("specific-agent-id")

        # Lock holder matches the agent in context
        with patch(
            "src.hooks.get_lock_holder", return_value="specific-agent-id"
        ) as mock:
            result = await block_unlocked_file_writes(hook_input, None, context)

        assert result == {}  # Allowed
        mock.assert_called_with(test_file)


class TestMakeLockEnforcementHook:
    """Tests for the make_lock_enforcement_hook factory function."""

    @pytest.mark.asyncio
    async def test_captures_agent_id_via_closure(self, tmp_path: Path):
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
        mock.assert_called_once_with(test_file)

    @pytest.mark.asyncio
    async def test_blocks_when_different_agent_holds_lock(self, tmp_path: Path):
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
    async def test_blocks_when_no_lock_exists(self, tmp_path: Path):
        """Factory-created hook should block when file is not locked."""
        test_file = str(tmp_path / "test.py")
        hook = make_lock_enforcement_hook("my-agent")
        hook_input = make_hook_input("Write", {"file_path": test_file})
        context = make_context()

        with patch("src.hooks.get_lock_holder", return_value=None):
            result = await hook(hook_input, None, context)

        assert result["decision"] == "block"
        assert "not locked" in result["reason"].lower()

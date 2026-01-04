"""Unit tests for PreToolUse hooks in src/hooks.py."""

import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import patch

import pytest

from claude_agent_sdk.types import PreToolUseHookInput, HookContext

if TYPE_CHECKING:
    from claude_agent_sdk.types import StopHookInput

from src.infra.hooks import (
    make_lock_enforcement_hook,
    block_dangerous_commands,
    DESTRUCTIVE_GIT_PATTERNS,
    SAFE_GIT_ALTERNATIVES,
    FILE_WRITE_TOOLS,
    FileReadCache,
    CachedFileInfo,
    make_file_read_cache_hook,
    make_stop_hook,
    _get_git_state,
)
from src.infra.tools.command_runner import CommandResult


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
            "src.infra.hooks.locking.get_lock_holder", return_value="captured-agent-id"
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

        with patch(
            "src.infra.hooks.locking.get_lock_holder", return_value="other-agent"
        ):
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

        with patch("src.infra.hooks.locking.get_lock_holder", return_value=None):
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

        with patch("src.infra.hooks.locking.get_lock_holder", return_value=agent_id):
            result = await hook(hook_input, None, context)

        assert result == {}  # Allowed

    @pytest.mark.asyncio
    async def test_handles_edit_tool(self, tmp_path: Path) -> None:
        """Edit tool should check lock ownership."""
        test_file = str(tmp_path / "test.py")
        hook_input = make_hook_input(
            "Edit",
            {"file_path": test_file, "old_string": "a", "new_string": "b"},
        )
        agent_id = "edit-agent"
        hook = make_lock_enforcement_hook(agent_id)
        context = make_context(agent_id)

        with patch("src.infra.hooks.locking.get_lock_holder", return_value=agent_id):
            result = await hook(hook_input, None, context)

        assert result == {}  # Allowed

    @pytest.mark.asyncio
    async def test_file_write_tools_constant_contains_expected_tools(self) -> None:
        """FILE_WRITE_TOOLS should contain expected write tools."""
        # These are the tools we expect to be file-write tools
        expected_tools = {"Write", "Edit", "NotebookEdit"}
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

        with patch(
            "src.infra.hooks.locking.get_lock_holder", return_value="my-agent"
        ) as mock:
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
    async def test_blocks_git_restore(self) -> None:
        """git restore should be blocked."""
        restore_commands = [
            "git restore .",
            "git restore file.py",
            "git restore --staged file.py",
            "git restore --source HEAD~1 file.py",
        ]
        context = make_context()

        for cmd in restore_commands:
            hook_input = make_hook_input("Bash", {"command": cmd})
            result = await block_dangerous_commands(hook_input, None, context)
            assert result.get("decision") == "block", f"Expected {cmd!r} to be blocked"
            assert "git restore" in result["reason"]

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
            "git restore",
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
        assert "git restore" in SAFE_GIT_ALTERNATIVES

    def test_alternatives_are_non_empty_strings(self) -> None:
        """All alternatives should be non-empty strings."""
        for pattern, alternative in SAFE_GIT_ALTERNATIVES.items():
            assert isinstance(alternative, str), (
                f"Alternative for {pattern} is not a string"
            )
            assert len(alternative) > 0, f"Alternative for {pattern} is empty"


class TestFileReadCache:
    """Tests for the FileReadCache class."""

    def test_first_read_is_allowed(self, tmp_path: Path) -> None:
        """First read of a file should be allowed and cached."""
        cache = FileReadCache()
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        is_redundant, message = cache.check_and_update(str(test_file))

        assert is_redundant is False
        assert message == ""
        assert cache.cache_size == 1
        assert cache.blocked_count == 0

    def test_second_read_unchanged_file_is_blocked(self, tmp_path: Path) -> None:
        """Second read of unchanged file should be blocked."""
        cache = FileReadCache()
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        # First read - should be allowed
        cache.check_and_update(str(test_file))

        # Second read - should be blocked
        is_redundant, message = cache.check_and_update(str(test_file))

        assert is_redundant is True
        assert "unchanged" in message.lower()
        assert "read 2x" in message
        assert cache.blocked_count == 1

    def test_third_read_shows_correct_count(self, tmp_path: Path) -> None:
        """Third read should show read 3x in message."""
        cache = FileReadCache()
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        cache.check_and_update(str(test_file))  # 1st
        cache.check_and_update(str(test_file))  # 2nd (blocked)
        is_redundant, message = cache.check_and_update(str(test_file))  # 3rd (blocked)

        assert is_redundant is True
        assert "read 3x" in message
        assert cache.blocked_count == 2

    def test_read_after_file_modification_is_allowed(self, tmp_path: Path) -> None:
        """Read after file modification should be allowed."""
        cache = FileReadCache()
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")

        # First read
        cache.check_and_update(str(test_file))

        # Modify the file (change content and mtime)
        time.sleep(0.01)  # Ensure mtime changes
        test_file.write_text("print('world')")

        # Second read after modification - should be allowed
        is_redundant, message = cache.check_and_update(str(test_file))

        assert is_redundant is False
        assert message == ""
        assert cache.blocked_count == 0

    def test_read_after_size_change_is_allowed(self, tmp_path: Path) -> None:
        """Read after file size change should be allowed."""
        cache = FileReadCache()
        test_file = tmp_path / "test.py"
        test_file.write_text("short")

        cache.check_and_update(str(test_file))

        # Change size
        time.sleep(0.01)
        test_file.write_text("much longer content now")

        is_redundant, _ = cache.check_and_update(str(test_file))
        assert is_redundant is False

    def test_nonexistent_file_is_allowed(self, tmp_path: Path) -> None:
        """Read of nonexistent file should be allowed (tool will report error)."""
        cache = FileReadCache()
        nonexistent = tmp_path / "does_not_exist.py"

        is_redundant, message = cache.check_and_update(str(nonexistent))

        assert is_redundant is False
        assert message == ""
        assert cache.cache_size == 0

    def test_invalidate_removes_cache_entry(self, tmp_path: Path) -> None:
        """Invalidating a file should remove it from cache."""
        cache = FileReadCache()
        test_file = tmp_path / "test.py"
        test_file.write_text("content")

        cache.check_and_update(str(test_file))
        assert cache.cache_size == 1

        cache.invalidate(str(test_file))
        assert cache.cache_size == 0

        # Next read should be allowed (first read after invalidation)
        is_redundant, _ = cache.check_and_update(str(test_file))
        assert is_redundant is False

    def test_invalidate_nonexistent_entry_is_safe(self) -> None:
        """Invalidating a file not in cache should not raise."""
        cache = FileReadCache()
        cache.invalidate("/path/to/nonexistent/file.py")
        # Should not raise

    def test_different_files_cached_separately(self, tmp_path: Path) -> None:
        """Different files should be cached independently."""
        cache = FileReadCache()
        file1 = tmp_path / "file1.py"
        file2 = tmp_path / "file2.py"
        file1.write_text("content1")
        file2.write_text("content2")

        # Read both files
        cache.check_and_update(str(file1))
        cache.check_and_update(str(file2))

        assert cache.cache_size == 2

        # Second read of file1 blocked, file2 still allowed on first re-read
        is_redundant_1, _ = cache.check_and_update(str(file1))
        is_redundant_2, _ = cache.check_and_update(str(file2))

        assert is_redundant_1 is True
        assert is_redundant_2 is True
        assert cache.blocked_count == 2

    def test_cached_file_info_dataclass(self) -> None:
        """CachedFileInfo dataclass should work correctly."""
        info = CachedFileInfo(
            mtime_ns=1234567890,
            size=100,
            content_hash="abc123",
            read_count=1,
        )
        assert info.mtime_ns == 1234567890
        assert info.size == 100
        assert info.content_hash == "abc123"
        assert info.read_count == 1

    def test_different_offset_limit_allowed(self, tmp_path: Path) -> None:
        """Reading same file with different offset/limit should be allowed."""
        cache = FileReadCache()
        test_file = tmp_path / "test.py"
        test_file.write_text("line1\nline2\nline3\nline4\nline5")

        # First read - full file (no offset/limit)
        is_redundant1, _ = cache.check_and_update(str(test_file))
        assert is_redundant1 is False

        # Second read with offset=0, limit=2 - different range, should be allowed
        is_redundant2, _ = cache.check_and_update(str(test_file), offset=0, limit=2)
        assert is_redundant2 is False

        # Third read with offset=2, limit=2 - different range, should be allowed
        is_redundant3, _ = cache.check_and_update(str(test_file), offset=2, limit=2)
        assert is_redundant3 is False

        # Fourth read - same full file read - should be blocked
        is_redundant4, _ = cache.check_and_update(str(test_file))
        assert is_redundant4 is True

        # Fifth read with offset=0, limit=2 again - same range as second, should be blocked
        is_redundant5, _ = cache.check_and_update(str(test_file), offset=0, limit=2)
        assert is_redundant5 is True

        # Cache should have 3 entries (full, offset=0/limit=2, offset=2/limit=2)
        assert cache.cache_size == 3

    def test_invalidate_clears_all_ranges(self, tmp_path: Path) -> None:
        """Invalidating a file should clear all offset/limit cache entries."""
        cache = FileReadCache()
        test_file = tmp_path / "test.py"
        test_file.write_text("line1\nline2\nline3\nline4\nline5")

        # Cache multiple ranges
        cache.check_and_update(str(test_file))  # full file
        cache.check_and_update(str(test_file), offset=0, limit=2)
        cache.check_and_update(str(test_file), offset=2, limit=2)
        assert cache.cache_size == 3

        # Invalidate the file
        cache.invalidate(str(test_file))
        assert cache.cache_size == 0

        # All reads should now be allowed
        is_redundant1, _ = cache.check_and_update(str(test_file))
        is_redundant2, _ = cache.check_and_update(str(test_file), offset=0, limit=2)
        assert is_redundant1 is False
        assert is_redundant2 is False


class TestMakeFileReadCacheHook:
    """Tests for the make_file_read_cache_hook factory function."""

    @pytest.mark.asyncio
    async def test_allows_first_read(self, tmp_path: Path) -> None:
        """First read through hook should be allowed."""
        cache = FileReadCache()
        hook = make_file_read_cache_hook(cache)
        test_file = tmp_path / "test.py"
        test_file.write_text("content")

        hook_input = make_hook_input("Read", {"file_path": str(test_file)})
        context = make_context()

        result = await hook(hook_input, None, context)

        assert result == {}  # Empty dict means allow

    @pytest.mark.asyncio
    async def test_blocks_second_read_unchanged(self, tmp_path: Path) -> None:
        """Second read through hook should be blocked."""
        cache = FileReadCache()
        hook = make_file_read_cache_hook(cache)
        test_file = tmp_path / "test.py"
        test_file.write_text("content")

        hook_input = make_hook_input("Read", {"file_path": str(test_file)})
        context = make_context()

        # First read
        await hook(hook_input, None, context)

        # Second read
        result = await hook(hook_input, None, context)

        assert result["decision"] == "block"
        assert "unchanged" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_allows_non_read_tools(self, tmp_path: Path) -> None:
        """Non-Read tools should be allowed without cache check."""
        cache = FileReadCache()
        hook = make_file_read_cache_hook(cache)

        hook_input = make_hook_input("Bash", {"command": "ls -la"})
        context = make_context()

        result = await hook(hook_input, None, context)

        assert result == {}

    @pytest.mark.asyncio
    async def test_invalidates_cache_on_write(self, tmp_path: Path) -> None:
        """Write tool should invalidate cache for that file."""
        cache = FileReadCache()
        hook = make_file_read_cache_hook(cache)
        test_file = tmp_path / "test.py"
        test_file.write_text("content")

        # First read - caches the file
        read_input = make_hook_input("Read", {"file_path": str(test_file)})
        context = make_context()
        await hook(read_input, None, context)
        assert cache.cache_size == 1

        # Write tool - should invalidate cache
        write_input = make_hook_input("Write", {"file_path": str(test_file)})
        await hook(write_input, None, context)
        assert cache.cache_size == 0

    @pytest.mark.asyncio
    async def test_invalidates_cache_on_notebook_edit(self, tmp_path: Path) -> None:
        """NotebookEdit tool should invalidate cache."""
        cache = FileReadCache()
        hook = make_file_read_cache_hook(cache)
        notebook = tmp_path / "notebook.ipynb"
        notebook.write_text("{}")

        # Read to cache
        read_input = make_hook_input("Read", {"file_path": str(notebook)})
        context = make_context()
        await hook(read_input, None, context)

        # NotebookEdit - should invalidate
        edit_input = make_hook_input(
            "NotebookEdit",
            {"notebook_path": str(notebook), "new_source": "cell content"},
        )
        await hook(edit_input, None, context)
        assert cache.cache_size == 0

    @pytest.mark.asyncio
    async def test_read_missing_file_path_allowed(self) -> None:
        """Read with missing file_path should be allowed (will fail later)."""
        cache = FileReadCache()
        hook = make_file_read_cache_hook(cache)

        hook_input = make_hook_input("Read", {})  # Missing file_path
        context = make_context()

        result = await hook(hook_input, None, context)

        assert result == {}  # Allowed (tool will fail anyway)

    @pytest.mark.asyncio
    async def test_allows_different_offset_limit(self, tmp_path: Path) -> None:
        """Read with different offset/limit should be allowed."""
        cache = FileReadCache()
        hook = make_file_read_cache_hook(cache)
        test_file = tmp_path / "test.py"
        test_file.write_text("line1\nline2\nline3\nline4\nline5")

        context = make_context()

        # First read - full file
        full_read = make_hook_input("Read", {"file_path": str(test_file)})
        result1 = await hook(full_read, None, context)
        assert result1 == {}  # allowed

        # Second read - with offset and limit (different range)
        partial_read = make_hook_input(
            "Read", {"file_path": str(test_file), "offset": 0, "limit": 2}
        )
        result2 = await hook(partial_read, None, context)
        assert result2 == {}  # allowed (different range)

        # Third read - full file again (same as first)
        result3 = await hook(full_read, None, context)
        assert result3["decision"] == "block"  # blocked (same as first read)

        # Fourth read - different offset
        partial_read2 = make_hook_input(
            "Read", {"file_path": str(test_file), "offset": 2, "limit": 2}
        )
        result4 = await hook(partial_read2, None, context)
        assert result4 == {}  # allowed (different range)

        # Fifth read - same as second (offset=0, limit=2)
        result5 = await hook(partial_read, None, context)
        assert result5["decision"] == "block"  # blocked (same as second read)


class TestDetectLintCommand:
    """Tests for the _detect_lint_command function and LintCache.detect_lint_command.

    The detection uses extract_tool_name to dynamically extract tool names
    from commands, making it language-agnostic (supports eslint, golangci-lint, etc.).
    """

    def test_detects_ruff(self) -> None:
        """Should detect ruff commands."""
        from src.infra.hooks import DEFAULT_LINT_TOOLS, _detect_lint_command

        assert _detect_lint_command("uvx ruff check .", DEFAULT_LINT_TOOLS) == "ruff"
        assert _detect_lint_command("ruff check src/", DEFAULT_LINT_TOOLS) == "ruff"
        assert _detect_lint_command("uvx ruff format .", DEFAULT_LINT_TOOLS) == "ruff"
        assert (
            _detect_lint_command("ruff format --check .", DEFAULT_LINT_TOOLS) == "ruff"
        )

    def test_detects_ty(self) -> None:
        """Should detect ty commands."""
        from src.infra.hooks import DEFAULT_LINT_TOOLS, _detect_lint_command

        assert _detect_lint_command("uvx ty check", DEFAULT_LINT_TOOLS) == "ty"
        assert _detect_lint_command("ty check src/", DEFAULT_LINT_TOOLS) == "ty"

    def test_detects_eslint(self) -> None:
        """Should detect eslint commands."""
        from src.infra.hooks import DEFAULT_LINT_TOOLS, _detect_lint_command

        assert _detect_lint_command("npx eslint .", DEFAULT_LINT_TOOLS) == "eslint"
        assert _detect_lint_command("eslint src/", DEFAULT_LINT_TOOLS) == "eslint"
        assert (
            _detect_lint_command("npx eslint --fix .", DEFAULT_LINT_TOOLS) == "eslint"
        )

    def test_detects_golangci_lint(self) -> None:
        """Should detect golangci-lint commands."""
        from src.infra.hooks import DEFAULT_LINT_TOOLS, _detect_lint_command

        assert (
            _detect_lint_command("golangci-lint run", DEFAULT_LINT_TOOLS)
            == "golangci-lint"
        )
        assert (
            _detect_lint_command("golangci-lint run ./...", DEFAULT_LINT_TOOLS)
            == "golangci-lint"
        )

    def test_detects_custom_lint_tools(self) -> None:
        """Should detect custom lint tools when configured."""
        from src.infra.hooks import _detect_lint_command

        custom_tools = frozenset({"mypy", "flake8", "cargo clippy"})
        assert _detect_lint_command("mypy src/", custom_tools) == "mypy"
        assert _detect_lint_command("flake8 .", custom_tools) == "flake8"
        assert (
            _detect_lint_command("cargo clippy -- -D warnings", custom_tools)
            == "cargo clippy"
        )

    def test_case_insensitive_lint_tools(self) -> None:
        """Should match lint tools case-insensitively via LintCache.

        Note: _detect_lint_command expects pre-lowercased lint_tools for
        efficiency. LintCache.detect_lint_command handles the normalization.
        """
        from src.infra.hooks import LintCache, _detect_lint_command

        # _detect_lint_command expects pre-lowercased lint_tools
        lowercase_tools = frozenset({"ruff", "ty", "eslint"})
        assert _detect_lint_command("ruff check .", lowercase_tools) == "ruff"
        assert _detect_lint_command("uvx ty check", lowercase_tools) == "ty"
        assert _detect_lint_command("eslint src/", lowercase_tools) == "eslint"

        # LintCache handles case-insensitive matching via pre-lowercasing
        cache = LintCache(lint_tools=frozenset({"RUFF", "TY", "ESLINT"}))
        assert cache.detect_lint_command("ruff check .") == "ruff"
        assert cache.detect_lint_command("uvx ty check") == "ty"
        assert cache.detect_lint_command("eslint src/") == "eslint"

        # Uppercase commands also work (extract_tool_name normalizes to lowercase)
        assert _detect_lint_command("RUFF CHECK .", lowercase_tools) == "ruff"

    def test_returns_none_for_non_lint_commands(self) -> None:
        """Should return None for non-lint commands."""
        from src.infra.hooks import DEFAULT_LINT_TOOLS, _detect_lint_command

        assert _detect_lint_command("git status", DEFAULT_LINT_TOOLS) is None
        assert _detect_lint_command("uv run pytest", DEFAULT_LINT_TOOLS) is None
        assert _detect_lint_command("ls -la", DEFAULT_LINT_TOOLS) is None
        assert _detect_lint_command("echo hello", DEFAULT_LINT_TOOLS) is None

    def test_lint_cache_detect_method(self, tmp_path: Path) -> None:
        """LintCache.detect_lint_command should use configured lint tools."""
        from src.infra.hooks import LintCache

        # Test with default lint tools
        cache = LintCache(repo_path=tmp_path)
        assert cache.detect_lint_command("uvx ruff check .") == "ruff"
        assert cache.detect_lint_command("npx eslint .") == "eslint"
        assert cache.detect_lint_command("git status") is None

        # Test with custom lint tools
        custom_cache = LintCache(
            repo_path=tmp_path, lint_tools={"mypy", "cargo clippy"}
        )
        assert custom_cache.detect_lint_command("mypy src/") == "mypy"
        assert custom_cache.detect_lint_command("cargo clippy") == "cargo clippy"
        # ruff not in custom tools, should return None
        assert custom_cache.detect_lint_command("uvx ruff check .") is None


class TestLintCache:
    """Tests for the LintCache class."""

    def test_first_lint_is_allowed(self, tmp_path: Path) -> None:
        """First lint of a type should be allowed (recorded as pending)."""
        from src.infra.hooks import LintCache

        # Create a git repo for testing
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        # Create initial commit
        (tmp_path / "file.py").write_text("print('hello')")
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        cache = LintCache(repo_path=tmp_path)
        is_redundant, message = cache.check_and_update("ruff")

        assert is_redundant is False
        assert message == ""
        # First call only records pending state, not confirmed cache
        assert cache.cache_size == 0
        assert cache.skipped_count == 0

    def test_second_lint_unchanged_is_blocked(self, tmp_path: Path) -> None:
        """Second lint with unchanged state promotes pending to confirmed and blocks."""
        from src.infra.hooks import LintCache

        # Create a git repo
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        (tmp_path / "file.py").write_text("print('hello')")
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        cache = LintCache(repo_path=tmp_path)

        # First lint - allowed
        is_redundant_first, _ = cache.check_and_update("ruff")
        assert is_redundant_first is False

        # Mark success after lint passes
        cache.mark_success("ruff")
        assert cache.cache_size == 1

        # Second lint - blocked (cached success)
        is_redundant, message = cache.check_and_update("ruff")

        assert is_redundant is True
        assert "no changes" in message.lower()
        assert "skipped 1x" in message
        assert cache.skipped_count == 1

    def test_lint_after_file_change_is_allowed(self, tmp_path: Path) -> None:
        """Lint after file modification should be allowed."""
        from src.infra.hooks import LintCache

        # Create a git repo
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        test_file = tmp_path / "file.py"
        test_file.write_text("print('hello')")
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        cache = LintCache(repo_path=tmp_path)

        # First lint
        cache.check_and_update("ruff")

        # Modify file (creates uncommitted change)
        test_file.write_text("print('world')")

        # Second lint after modification - should be allowed
        is_redundant, message = cache.check_and_update("ruff")

        assert is_redundant is False
        assert message == ""
        assert cache.skipped_count == 0

    def test_different_lint_types_cached_separately(self, tmp_path: Path) -> None:
        """Different lint types should be cached independently."""
        from src.infra.hooks import LintCache

        # Create a git repo
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        (tmp_path / "file.py").write_text("print('hello')")
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        cache = LintCache(repo_path=tmp_path)

        # Run different lint types and mark each as successful
        cache.check_and_update("ruff")
        cache.mark_success("ruff")
        cache.check_and_update("eslint")
        cache.mark_success("eslint")
        cache.check_and_update("ty")
        cache.mark_success("ty")

        # All 3 are cached after mark_success
        assert cache.cache_size == 3

        # Second run of each - blocked due to cached success
        is_redundant_1, _ = cache.check_and_update("ruff")
        is_redundant_2, _ = cache.check_and_update("eslint")
        is_redundant_3, _ = cache.check_and_update("ty")

        assert is_redundant_1 is True
        assert is_redundant_2 is True
        assert is_redundant_3 is True
        assert cache.skipped_count == 3

    def test_invalidate_clears_specific_type(self, tmp_path: Path) -> None:
        """Invalidating a specific lint type should only clear that type."""
        from src.infra.hooks import LintCache

        # Create a git repo
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        (tmp_path / "file.py").write_text("print('hello')")
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        cache = LintCache(repo_path=tmp_path)
        # Mark each lint type as successful
        cache.check_and_update("ruff")
        cache.mark_success("ruff")
        cache.check_and_update("eslint")
        cache.mark_success("eslint")
        cache.check_and_update("ty")
        cache.mark_success("ty")
        assert cache.cache_size == 3

        cache.invalidate("ruff")
        assert cache.cache_size == 2

        # ruff should be allowed again (cache cleared)
        is_redundant, _ = cache.check_and_update("ruff")
        assert is_redundant is False

    def test_invalidate_all_clears_cache(self, tmp_path: Path) -> None:
        """Invalidating without type should clear all entries."""
        from src.infra.hooks import LintCache

        # Create a git repo
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        (tmp_path / "file.py").write_text("print('hello')")
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        cache = LintCache(repo_path=tmp_path)
        # Mark each lint type as successful
        cache.check_and_update("ruff")
        cache.mark_success("ruff")
        cache.check_and_update("eslint")
        cache.mark_success("eslint")
        cache.check_and_update("ty")
        cache.mark_success("ty")
        assert cache.cache_size == 3

        cache.invalidate()
        assert cache.cache_size == 0

    def test_non_git_repo_allows_all_lints(self, tmp_path: Path) -> None:
        """In a non-git directory, all lints should be allowed."""
        from src.infra.hooks import LintCache

        # tmp_path is not a git repo
        cache = LintCache(repo_path=tmp_path)

        # First lint
        is_redundant_1, _ = cache.check_and_update("ruff")
        # Second lint (would normally be blocked)
        is_redundant_2, _ = cache.check_and_update("ruff")

        # Both should be allowed since we can't determine git state
        assert is_redundant_1 is False
        assert is_redundant_2 is False


class TestMakeLintCacheHook:
    """Tests for the make_lint_cache_hook factory function."""

    @pytest.mark.asyncio
    async def test_allows_first_lint(self, tmp_path: Path) -> None:
        """First lint through hook should be allowed."""
        from src.infra.hooks import LintCache, make_lint_cache_hook

        # Create a git repo
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        (tmp_path / "file.py").write_text("print('hello')")
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        cache = LintCache(repo_path=tmp_path)
        hook = make_lint_cache_hook(cache)

        hook_input = make_hook_input("Bash", {"command": "uvx ruff check ."})
        context = make_context()

        result = await hook(hook_input, None, context)

        assert result == {}  # Empty dict means allow

    @pytest.mark.asyncio
    async def test_blocks_second_lint_unchanged(self, tmp_path: Path) -> None:
        """Second lint through hook should be blocked if unchanged."""
        from src.infra.hooks import LintCache, make_lint_cache_hook

        # Create a git repo
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        (tmp_path / "file.py").write_text("print('hello')")
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        cache = LintCache(repo_path=tmp_path)
        hook = make_lint_cache_hook(cache)

        hook_input = make_hook_input("Bash", {"command": "uvx ruff check ."})
        context = make_context()

        # First lint - allowed
        result = await hook(hook_input, None, context)
        assert result == {}  # Allow

        # Simulate successful lint completion (lint_type is now "ruff" via extract_tool_name)
        cache.mark_success("ruff", "uvx ruff check .")

        # Second lint - blocked due to cached success
        result = await hook(hook_input, None, context)

        assert result["decision"] == "block"
        assert "no changes" in result["reason"].lower()

    @pytest.mark.asyncio
    async def test_allows_non_lint_bash_commands(self, tmp_path: Path) -> None:
        """Non-lint Bash commands should be allowed."""
        from src.infra.hooks import LintCache, make_lint_cache_hook

        cache = LintCache(repo_path=tmp_path)
        hook = make_lint_cache_hook(cache)

        hook_input = make_hook_input("Bash", {"command": "git status"})
        context = make_context()

        result = await hook(hook_input, None, context)

        assert result == {}

    @pytest.mark.asyncio
    async def test_allows_non_bash_tools(self, tmp_path: Path) -> None:
        """Non-Bash tools should be allowed."""
        from src.infra.hooks import LintCache, make_lint_cache_hook

        cache = LintCache(repo_path=tmp_path)
        hook = make_lint_cache_hook(cache)

        hook_input = make_hook_input("Read", {"file_path": "/some/file.py"})
        context = make_context()

        result = await hook(hook_input, None, context)

        assert result == {}

    @pytest.mark.asyncio
    async def test_allows_compound_commands_with_lint(self, tmp_path: Path) -> None:
        """Compound commands containing lint should not be blocked.

        When an agent runs 'ruff check . && pytest', the entire command should
        be allowed even if ruff check was previously cached. This ensures the
        test portion runs.
        """
        from src.infra.hooks import LintCache, make_lint_cache_hook

        # Create a git repo
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        (tmp_path / "file.py").write_text("print('hello')")
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        cache = LintCache(repo_path=tmp_path)
        hook = make_lint_cache_hook(cache)
        context = make_context()

        # First lint (simple) - allowed
        simple_lint = make_hook_input("Bash", {"command": "uvx ruff check ."})
        result = await hook(simple_lint, None, context)
        assert result == {}  # Allow

        # Simulate successful lint completion (lint_type is now "ruff" via extract_tool_name)
        cache.mark_success("ruff", "uvx ruff check .")

        # Second simple lint - blocked due to cached success
        result = await hook(simple_lint, None, context)
        assert result["decision"] == "block"

        # Compound command with && - should NOT be blocked
        compound_and = make_hook_input(
            "Bash", {"command": "uvx ruff check . && pytest"}
        )
        result = await hook(compound_and, None, context)
        assert result == {}  # Allow compound commands

        # Compound command with || - should NOT be blocked
        compound_or = make_hook_input(
            "Bash", {"command": "uvx ruff check . || echo 'lint failed'"}
        )
        result = await hook(compound_or, None, context)
        assert result == {}  # Allow compound commands

        # Compound command with ; - should NOT be blocked
        compound_semi = make_hook_input("Bash", {"command": "uvx ruff check .; pytest"})
        result = await hook(compound_semi, None, context)
        assert result == {}  # Allow compound commands

    @pytest.mark.asyncio
    async def test_invalidates_cache_on_write(self, tmp_path: Path) -> None:
        """Write tool should invalidate lint cache."""
        from src.infra.hooks import LintCache, make_lint_cache_hook

        # Create a git repo
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        (tmp_path / "file.py").write_text("print('hello')")
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        cache = LintCache(repo_path=tmp_path)
        hook = make_lint_cache_hook(cache)
        context = make_context()

        # First lint - allowed
        lint_input = make_hook_input("Bash", {"command": "uvx ruff check ."})
        await hook(lint_input, None, context)
        # Simulate successful lint completion
        cache.mark_success("ruff", "uvx ruff check .")
        assert cache.cache_size == 1

        # Write tool - should invalidate cache
        write_input = make_hook_input("Write", {"file_path": str(tmp_path / "file.py")})
        await hook(write_input, None, context)
        assert cache.cache_size == 0

    @pytest.mark.asyncio
    async def test_invalidates_cache_on_notebook_edit(self, tmp_path: Path) -> None:
        """NotebookEdit tool should invalidate lint cache."""
        from src.infra.hooks import LintCache, make_lint_cache_hook

        # Create a git repo
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        (tmp_path / "file.py").write_text("print('hello')")
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        cache = LintCache(repo_path=tmp_path)
        hook = make_lint_cache_hook(cache)
        context = make_context()

        # First lint - allowed
        lint_input = make_hook_input("Bash", {"command": "uvx ruff check ."})
        await hook(lint_input, None, context)
        # Simulate successful lint completion
        cache.mark_success("ruff", "uvx ruff check .")
        assert cache.cache_size == 1

        # NotebookEdit - should invalidate
        edit_input = make_hook_input(
            "NotebookEdit",
            {
                "notebook_path": str(tmp_path / "notebook.ipynb"),
                "new_source": "cell content",
            },
        )
        await hook(edit_input, None, context)
        assert cache.cache_size == 0


class TestMakeStopHookWithCommandRunner:
    """Tests for make_stop_hook that verify CommandRunner integration."""

    @pytest.mark.asyncio
    async def test_uses_run_command_for_lock_cleanup(self) -> None:
        """Stop hook should use run_command instead of subprocess.run."""
        hook = make_stop_hook("test-agent-123")
        context = make_context()
        hook_input = cast("StopHookInput", {"stop_hook_type": "natural"})

        with patch("src.infra.hooks.locking.run_command") as mock_run_command:
            mock_run_command.return_value = CommandResult(
                command=["lock-release-all.sh"],
                returncode=0,
            )
            result = await hook(hook_input, None, context)

        assert result == {}  # Hook returns empty dict
        mock_run_command.assert_called_once()
        # Verify the script path is passed
        call_args = mock_run_command.call_args
        assert "lock-release-all.sh" in call_args[0][0][0]

    @pytest.mark.asyncio
    async def test_passes_agent_id_in_env(self) -> None:
        """Stop hook should pass AGENT_ID in environment."""
        agent_id = "my-special-agent"
        hook = make_stop_hook(agent_id)
        context = make_context()
        hook_input = cast("StopHookInput", {"stop_hook_type": "natural"})

        with patch("src.infra.hooks.locking.run_command") as mock_run_command:
            mock_run_command.return_value = CommandResult(
                command=["lock-release-all.sh"],
                returncode=0,
            )
            await hook(hook_input, None, context)

        call_kwargs = mock_run_command.call_args[1]
        assert call_kwargs["env"]["AGENT_ID"] == agent_id

    @pytest.mark.asyncio
    async def test_ignores_run_command_failure(self) -> None:
        """Stop hook should not raise on run_command failure."""
        hook = make_stop_hook("test-agent")
        context = make_context()
        hook_input = cast("StopHookInput", {"stop_hook_type": "natural"})

        with patch("src.infra.hooks.locking.run_command") as mock_run_command:
            mock_run_command.side_effect = Exception("Command failed")
            result = await hook(hook_input, None, context)

        assert result == {}  # Should return empty dict despite failure


class TestGetGitStateWithCommandRunner:
    """Tests for _get_git_state that verify CommandRunner integration."""

    def test_uses_run_command_for_git_operations(self, tmp_path: Path) -> None:
        """_get_git_state should use run_command instead of subprocess.run."""
        with patch("src.infra.hooks.lint_cache.run_command") as mock_run_command:
            # Mock successful git commands
            mock_run_command.side_effect = [
                CommandResult(
                    command=["git", "rev-parse", "HEAD"],
                    returncode=0,
                    stdout="abc123def456\n",
                ),
                CommandResult(
                    command=["git", "diff", "HEAD"],
                    returncode=0,
                    stdout="",
                ),
                CommandResult(
                    command=["git", "ls-files", "--others", "--exclude-standard"],
                    returncode=0,
                    stdout="",
                ),
            ]

            result = _get_git_state(tmp_path)

        assert result is not None
        assert len(result) == 16  # SHA-256 hash truncated to 16 chars
        assert mock_run_command.call_count == 3

    def test_returns_none_on_git_failure(self, tmp_path: Path) -> None:
        """_get_git_state should return None if git command fails."""
        with patch("src.infra.hooks.lint_cache.run_command") as mock_run_command:
            mock_run_command.return_value = CommandResult(
                command=["git", "rev-parse", "HEAD"],
                returncode=128,
                stdout="",
                stderr="fatal: not a git repository",
            )

            result = _get_git_state(tmp_path)

        assert result is None

    def test_respects_commandrunner_timeout_behavior(self, tmp_path: Path) -> None:
        """_get_git_state uses CommandRunner which handles timeouts."""
        # This test verifies that run_command is used, which means
        # CommandRunner's timeout and process-group termination apply
        with patch("src.infra.hooks.lint_cache.run_command") as mock_run_command:
            mock_run_command.side_effect = [
                CommandResult(
                    command=["git", "rev-parse", "HEAD"],
                    returncode=0,
                    stdout="abc123\n",
                ),
                CommandResult(
                    command=["git", "diff", "HEAD"],
                    returncode=124,  # TIMEOUT_EXIT_CODE
                    stdout="",
                    timed_out=True,
                ),
            ]

            result = _get_git_state(tmp_path)

        # Should return None because diff command "failed" (timed out)
        assert result is None

    def test_handles_untracked_files_failure_gracefully(self, tmp_path: Path) -> None:
        """_get_git_state should handle ls-files failure gracefully."""
        with patch("src.infra.hooks.lint_cache.run_command") as mock_run_command:
            mock_run_command.side_effect = [
                CommandResult(
                    command=["git", "rev-parse", "HEAD"],
                    returncode=0,
                    stdout="abc123\n",
                ),
                CommandResult(
                    command=["git", "diff", "HEAD"],
                    returncode=0,
                    stdout="",
                ),
                CommandResult(
                    command=["git", "ls-files", "--others", "--exclude-standard"],
                    returncode=1,  # Failed
                    stdout="",
                ),
            ]

            result = _get_git_state(tmp_path)

        # Should still return a hash, just without untracked files
        assert result is not None

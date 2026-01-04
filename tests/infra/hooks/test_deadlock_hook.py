"""Unit tests for the deadlock PostToolUse hook."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from claude_agent_sdk.types import HookContext, PostToolUseHookInput

    from src.domain.deadlock import LockEvent

from src.domain.deadlock import LockEventType
from src.infra.hooks.deadlock import (
    _extract_all_lock_paths,
    _extract_lock_path,
    _get_exit_code,
    _is_safe_batch_command,
    make_lock_event_hook,
)


def make_post_hook_input(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_result: str = "",
    exit_code: int | None = None,
) -> PostToolUseHookInput:
    """Create a mock PostToolUseHookInput."""
    result: dict[str, Any] = {
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_result": tool_result,
    }
    if exit_code is not None:
        result["exit_code"] = exit_code
    return cast("PostToolUseHookInput", result)


def make_context(agent_id: str = "test-agent") -> HookContext:
    """Create a mock HookContext."""
    return cast("HookContext", {"agent_id": agent_id})


@pytest.mark.unit
class TestExtractLockPath:
    """Tests for _extract_lock_path helper."""

    def test_lock_try_simple_path(self) -> None:
        """Extract path from lock-try.sh command."""
        result = _extract_lock_path("lock-try.sh /path/to/file.py")
        assert result == ("try", "/path/to/file.py")

    def test_lock_try_with_whitespace(self) -> None:
        """Extract path with trailing whitespace."""
        result = _extract_lock_path("lock-try.sh /path/to/file.py  ")
        assert result == ("try", "/path/to/file.py")

    def test_lock_wait_with_timeout(self) -> None:
        """Extract path from lock-wait.sh, ignoring timeout args."""
        result = _extract_lock_path("lock-wait.sh /path/to/file.py 300 1000")
        assert result == ("wait", "/path/to/file.py")

    def test_lock_release_simple(self) -> None:
        """Extract path from lock-release.sh command."""
        result = _extract_lock_path("lock-release.sh /path/to/file.py")
        assert result == ("release", "/path/to/file.py")

    def test_non_lock_command(self) -> None:
        """Return None for non-lock commands."""
        assert _extract_lock_path("ls -la") is None
        assert _extract_lock_path("git status") is None
        assert _extract_lock_path("echo lock-try.sh") is None

    def test_partial_match_not_extracted(self) -> None:
        """Patterns must fully match lock command."""
        # lock-release-all.sh should not match lock-release.sh pattern
        result = _extract_lock_path("lock-release-all.sh")
        assert result is None

    def test_lock_command_in_path(self) -> None:
        """Lock command embedded in a longer command."""
        result = _extract_lock_path("cd /foo && lock-try.sh bar.py")
        assert result == ("try", "bar.py")

    def test_trailing_shell_operators_not_captured(self) -> None:
        """Shell operators after path should not be captured."""
        result = _extract_lock_path("lock-try.sh file.py && echo done")
        assert result == ("try", "file.py")

        result = _extract_lock_path("lock-release.sh file.py || exit 1")
        assert result == ("release", "file.py")

        result = _extract_lock_path("lock-try.sh file.py; ls")
        assert result == ("try", "file.py")

    def test_quoted_paths_stripped(self) -> None:
        """Shell quotes around paths should be stripped."""
        result = _extract_lock_path('lock-try.sh "file.py"')
        assert result == ("try", "file.py")

        result = _extract_lock_path("lock-try.sh 'file.py'")
        assert result == ("try", "file.py")

        result = _extract_lock_path('lock-try.sh "/path/with spaces/file.py"')
        assert result == ("try", "/path/with spaces/file.py")

    def test_lock_wait_with_spaces_quoted(self) -> None:
        """lock-wait.sh with quoted path containing spaces."""
        result = _extract_lock_path('lock-wait.sh "/path/with spaces/file.py" 300')
        assert result == ("wait", "/path/with spaces/file.py")


@pytest.mark.unit
class TestExtractAllLockPaths:
    """Tests for _extract_all_lock_paths helper."""

    def test_single_command(self) -> None:
        """Extract single lock command."""
        result = _extract_all_lock_paths("lock-try.sh /path/to/file.py")
        assert result == [("try", "/path/to/file.py")]

    def test_multiple_commands(self) -> None:
        """Extract multiple lock commands from batched bash."""
        cmd = "lock-try.sh a.py && lock-try.sh b.py && lock-try.sh c.py"
        result = _extract_all_lock_paths(cmd)
        assert result == [("try", "a.py"), ("try", "b.py"), ("try", "c.py")]

    def test_mixed_command_types(self) -> None:
        """Extract mixed lock command types."""
        cmd = "lock-try.sh a.py && lock-release.sh b.py"
        result = _extract_all_lock_paths(cmd)
        assert result == [("try", "a.py"), ("release", "b.py")]

    def test_multiline_commands(self) -> None:
        """Extract commands from multiline bash."""
        cmd = "lock-try.sh a.py\nlock-try.sh b.py"
        result = _extract_all_lock_paths(cmd)
        assert result == [("try", "a.py"), ("try", "b.py")]

    def test_no_lock_commands(self) -> None:
        """Return empty list for non-lock commands."""
        assert _extract_all_lock_paths("ls -la") == []


@pytest.mark.unit
class TestGetExitCode:
    """Tests for _get_exit_code helper."""

    def test_exit_code_in_text(self) -> None:
        """Parse exit code from text format."""
        assert _get_exit_code("exit code: 0") == 0
        assert _get_exit_code("exit code: 1") == 1
        assert _get_exit_code("Some output\nexit code: 2\n") == 2

    def test_case_insensitive(self) -> None:
        """Parse exit code case insensitively."""
        assert _get_exit_code("Exit Code: 0") == 0
        assert _get_exit_code("EXIT CODE: 1") == 1

    def test_no_exit_code(self) -> None:
        """Return None when no exit code found."""
        assert _get_exit_code("Command completed successfully") is None
        assert _get_exit_code("") is None


@pytest.mark.unit
class TestMakeLockEventHook:
    """Tests for the make_lock_event_hook factory."""

    @pytest.mark.asyncio
    async def test_lock_try_exit_0_emits_acquired(self) -> None:
        """lock-try.sh with exit 0 emits ACQUIRED event."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        with patch(
            "src.infra.hooks.deadlock.canonicalize_path",
            return_value="/canonical/path/file.py",
        ):
            hook = make_lock_event_hook("agent-1", emit, "/repo")
            hook_input = make_post_hook_input(
                "Bash",
                {"command": "lock-try.sh /path/file.py"},
                exit_code=0,
            )
            await hook(hook_input, None, make_context())

        assert len(events) == 1
        assert events[0].event_type == LockEventType.ACQUIRED
        assert events[0].agent_id == "agent-1"
        assert events[0].lock_path == "/canonical/path/file.py"

    @pytest.mark.asyncio
    async def test_lock_try_exit_1_emits_waiting(self) -> None:
        """lock-try.sh with exit 1 emits WAITING event."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        with patch(
            "src.infra.hooks.deadlock.canonicalize_path",
            return_value="/canonical/path/file.py",
        ):
            hook = make_lock_event_hook("agent-1", emit, "/repo")
            hook_input = make_post_hook_input(
                "Bash",
                {"command": "lock-try.sh /path/file.py"},
                exit_code=1,
            )
            await hook(hook_input, None, make_context())

        assert len(events) == 1
        assert events[0].event_type == LockEventType.WAITING
        assert events[0].agent_id == "agent-1"

    @pytest.mark.asyncio
    async def test_lock_release_exit_0_emits_released(self) -> None:
        """lock-release.sh with exit 0 emits RELEASED event."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        with patch(
            "src.infra.hooks.deadlock.canonicalize_path",
            return_value="/canonical/path/file.py",
        ):
            hook = make_lock_event_hook("agent-1", emit, "/repo")
            hook_input = make_post_hook_input(
                "Bash",
                {"command": "lock-release.sh /path/file.py"},
                exit_code=0,
            )
            await hook(hook_input, None, make_context())

        assert len(events) == 1
        assert events[0].event_type == LockEventType.RELEASED
        assert events[0].agent_id == "agent-1"

    @pytest.mark.asyncio
    async def test_lock_wait_exit_0_emits_acquired(self) -> None:
        """lock-wait.sh with exit 0 emits ACQUIRED event."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        with patch(
            "src.infra.hooks.deadlock.canonicalize_path",
            return_value="/canonical/path/file.py",
        ):
            hook = make_lock_event_hook("agent-1", emit, "/repo")
            hook_input = make_post_hook_input(
                "Bash",
                {"command": "lock-wait.sh /path/file.py 300"},
                exit_code=0,
            )
            await hook(hook_input, None, make_context())

        assert len(events) == 1
        assert events[0].event_type == LockEventType.ACQUIRED

    @pytest.mark.asyncio
    async def test_lock_wait_exit_1_no_event(self) -> None:
        """lock-wait.sh timeout (exit 1) emits no event."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        with patch(
            "src.infra.hooks.deadlock.canonicalize_path",
            return_value="/canonical/path/file.py",
        ):
            hook = make_lock_event_hook("agent-1", emit, "/repo")
            hook_input = make_post_hook_input(
                "Bash",
                {"command": "lock-wait.sh /path/file.py 300"},
                exit_code=1,
            )
            await hook(hook_input, None, make_context())

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_non_lock_bash_command_ignored(self) -> None:
        """Non-lock bash commands don't emit events."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        hook = make_lock_event_hook("agent-1", emit, "/repo")
        hook_input = make_post_hook_input(
            "Bash",
            {"command": "ls -la"},
            exit_code=0,
        )
        await hook(hook_input, None, make_context())

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_non_bash_tool_ignored(self) -> None:
        """Non-bash tools don't emit events."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        hook = make_lock_event_hook("agent-1", emit, "/repo")
        hook_input = make_post_hook_input(
            "Write",
            {"file_path": "/path/file.py", "content": "..."},
            exit_code=0,
        )
        await hook(hook_input, None, make_context())

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_exit_code_2_logs_warning_no_event(self) -> None:
        """Exit code 2 (error) logs warning and emits no event."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        with (
            patch(
                "src.infra.hooks.deadlock.canonicalize_path",
                return_value="/canonical/path/file.py",
            ),
            patch("src.infra.hooks.deadlock.logger") as mock_logger,
        ):
            hook = make_lock_event_hook("agent-1", emit, "/repo")
            hook_input = make_post_hook_input(
                "Bash",
                {"command": "lock-try.sh /path/file.py"},
                exit_code=2,
            )
            await hook(hook_input, None, make_context())

        assert len(events) == 0
        mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_path_canonicalization_failure_logs_warning(self) -> None:
        """Failed path canonicalization logs warning, no event."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        with (
            patch(
                "src.infra.hooks.deadlock.canonicalize_path",
                side_effect=ValueError("bad path"),
            ),
            patch("src.infra.hooks.deadlock.logger") as mock_logger,
        ):
            hook = make_lock_event_hook("agent-1", emit, "/repo")
            hook_input = make_post_hook_input(
                "Bash",
                {"command": "lock-try.sh /bad/path"},
                exit_code=0,
            )
            await hook(hook_input, None, make_context())

        assert len(events) == 0
        mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_emit_callback(self) -> None:
        """Async emit callback is awaited."""
        events: list[LockEvent] = []

        async def async_emit(event: LockEvent) -> None:
            events.append(event)

        emit = AsyncMock(side_effect=async_emit)

        with patch(
            "src.infra.hooks.deadlock.canonicalize_path",
            return_value="/canonical/path/file.py",
        ):
            hook = make_lock_event_hook("agent-1", emit, "/repo")
            hook_input = make_post_hook_input(
                "Bash",
                {"command": "lock-try.sh /path/file.py"},
                exit_code=0,
            )
            await hook(hook_input, None, make_context())

        emit.assert_awaited_once()
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_exit_code_from_tool_result_text(self) -> None:
        """Parse exit code from tool result when not in hook_input."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        with patch(
            "src.infra.hooks.deadlock.canonicalize_path",
            return_value="/canonical/path/file.py",
        ):
            hook = make_lock_event_hook("agent-1", emit, "/repo")
            hook_input = make_post_hook_input(
                "Bash",
                {"command": "lock-try.sh /path/file.py"},
                tool_result="Lock acquired\nexit code: 0",
                # No exit_code key
            )
            await hook(hook_input, None, make_context())

        assert len(events) == 1
        assert events[0].event_type == LockEventType.ACQUIRED

    @pytest.mark.asyncio
    async def test_event_has_timestamp(self) -> None:
        """Emitted events have a timestamp."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        with patch(
            "src.infra.hooks.deadlock.canonicalize_path",
            return_value="/canonical/path/file.py",
        ):
            hook = make_lock_event_hook("agent-1", emit, "/repo")
            hook_input = make_post_hook_input(
                "Bash",
                {"command": "lock-try.sh /path/file.py"},
                exit_code=0,
            )
            await hook(hook_input, None, make_context())

        assert len(events) == 1
        assert events[0].timestamp > 0


@pytest.mark.unit
class TestIsSafeBatchCommand:
    """Tests for _is_safe_batch_command helper."""

    def test_single_command_safe(self) -> None:
        """Single commands are safe."""
        assert _is_safe_batch_command("lock-try.sh file.py") is True

    def test_and_chain_safe(self) -> None:
        """Commands chained with && are safe."""
        assert _is_safe_batch_command("lock-try.sh a.py && lock-try.sh b.py") is True
        assert (
            _is_safe_batch_command("lock-try.sh a && lock-try.sh b && lock-try.sh c")
            is True
        )

    def test_semicolon_unsafe(self) -> None:
        """Commands with ; are unsafe."""
        assert _is_safe_batch_command("lock-try.sh a.py; lock-try.sh b.py") is False

    def test_or_chain_unsafe(self) -> None:
        """Commands with || are unsafe."""
        assert _is_safe_batch_command("lock-try.sh a.py || lock-try.sh b.py") is False

    def test_pipe_unsafe(self) -> None:
        """Commands with | are unsafe."""
        assert _is_safe_batch_command("lock-try.sh a.py | cat") is False

    def test_background_unsafe(self) -> None:
        """Commands with & (background) are unsafe."""
        assert _is_safe_batch_command("lock-try.sh a.py & lock-try.sh b.py") is False

    def test_operators_in_quotes_safe(self) -> None:
        """Operators inside quotes don't make command unsafe."""
        assert _is_safe_batch_command('lock-try.sh "file;name.py"') is True
        assert _is_safe_batch_command("lock-try.sh 'file|name.py'") is True
        assert _is_safe_batch_command('echo "a || b" && lock-try.sh c.py') is True

    def test_redirection_ampersand_safe(self) -> None:
        """Ampersand in redirections is safe, not a background operator."""
        # 2>&1 redirect stderr to stdout
        assert (
            _is_safe_batch_command("lock-try.sh a.py 2>&1 && lock-try.sh b.py") is True
        )
        # >&2 redirect stdout to stderr
        assert _is_safe_batch_command("lock-try.sh a.py >&2") is True
        # &> redirect both stdout and stderr to file
        assert _is_safe_batch_command("lock-try.sh a.py &>/dev/null") is True
        # Combined with && chaining
        assert (
            _is_safe_batch_command(
                "lock-try.sh a.py 2>&1 && lock-try.sh b.py 2>&1 && lock-try.sh c.py"
            )
            is True
        )


@pytest.mark.unit
class TestExtractAllLockPathsOrder:
    """Tests for _extract_all_lock_paths order preservation (Finding 2)."""

    def test_preserves_command_order(self) -> None:
        """Lock commands should be returned in their original order."""
        # release before try in command should preserve that order
        cmd = "lock-release.sh a && lock-try.sh b"
        result = _extract_all_lock_paths(cmd)
        assert result == [("release", "a"), ("try", "b")]

    def test_mixed_types_ordered(self) -> None:
        """Mixed command types maintain positional order."""
        cmd = "lock-try.sh a && lock-wait.sh b && lock-release.sh c"
        result = _extract_all_lock_paths(cmd)
        assert result == [("try", "a"), ("wait", "b"), ("release", "c")]


@pytest.mark.unit
class TestGetExitCodeNoSpace:
    """Tests for _get_exit_code guard fix (Finding 1)."""

    def test_exit_code_without_space(self) -> None:
        """Parse exit code without space after colon."""
        assert _get_exit_code("exit code:0") == 0
        assert _get_exit_code("exit code:1") == 1

    def test_exit_code_with_space(self) -> None:
        """Parse exit code with space after colon."""
        assert _get_exit_code("exit code: 0") == 0
        assert _get_exit_code("exit code:  1") == 1


@pytest.mark.unit
class TestUnsafeBatchEventEmission:
    """Tests for unsafe batch handling (Finding 3)."""

    @pytest.mark.asyncio
    async def test_unsafe_batch_only_emits_last_command(self) -> None:
        """Unsafe batch with || only emits event for the last command."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        with patch(
            "src.infra.hooks.deadlock.canonicalize_path",
            side_effect=lambda p, _: f"/canonical/{p}",
        ):
            hook = make_lock_event_hook("agent-1", emit, "/repo")
            # || is unsafe - only last command event should be emitted
            hook_input = make_post_hook_input(
                "Bash",
                {"command": "lock-try.sh a.py || lock-try.sh b.py"},
                exit_code=0,
            )
            await hook(hook_input, None, make_context())

        # Only b.py (last command) should emit
        assert len(events) == 1
        assert events[0].lock_path == "/canonical/b.py"

    @pytest.mark.asyncio
    async def test_safe_batch_emits_all_commands(self) -> None:
        """Safe batch with && emits events for all commands."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        with patch(
            "src.infra.hooks.deadlock.canonicalize_path",
            side_effect=lambda p, _: f"/canonical/{p}",
        ):
            hook = make_lock_event_hook("agent-1", emit, "/repo")
            # && is safe - all commands should emit
            hook_input = make_post_hook_input(
                "Bash",
                {"command": "lock-try.sh a.py && lock-try.sh b.py"},
                exit_code=0,
            )
            await hook(hook_input, None, make_context())

        assert len(events) == 2
        assert events[0].lock_path == "/canonical/a.py"
        assert events[1].lock_path == "/canonical/b.py"

    @pytest.mark.asyncio
    async def test_semicolon_batch_only_emits_last(self) -> None:
        """Semicolon batch only emits event for the last command."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        with patch(
            "src.infra.hooks.deadlock.canonicalize_path",
            side_effect=lambda p, _: f"/canonical/{p}",
        ):
            hook = make_lock_event_hook("agent-1", emit, "/repo")
            hook_input = make_post_hook_input(
                "Bash",
                {"command": "lock-try.sh a.py; lock-try.sh b.py"},
                exit_code=0,
            )
            await hook(hook_input, None, make_context())

        assert len(events) == 1
        assert events[0].lock_path == "/canonical/b.py"

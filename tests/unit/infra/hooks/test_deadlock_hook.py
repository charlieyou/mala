"""Unit tests for the deadlock PostToolUse hook."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

if TYPE_CHECKING:
    from claude_agent_sdk.types import HookContext, PostToolUseHookInput

from src.core.models import LockEvent, LockEventType
from src.infra.hooks.deadlock import make_lock_event_hook


def make_mcp_response(data: dict[str, Any]) -> dict[str, Any]:
    """Create an MCP tool response structure."""
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(data),
            }
        ]
    }


def make_post_hook_input(
    tool_name: str,
    tool_input: dict[str, Any],
    tool_response: dict[str, Any] | None = None,
) -> PostToolUseHookInput:
    """Create a mock PostToolUseHookInput."""
    result: dict[str, Any] = {
        "tool_name": tool_name,
        "tool_input": tool_input,
    }
    if tool_response is not None:
        result["tool_response"] = tool_response
    return cast("PostToolUseHookInput", result)


def make_context(agent_id: str = "test-agent") -> HookContext:
    """Create a mock HookContext."""
    return cast("HookContext", {"agent_id": agent_id})


@pytest.mark.unit
class TestMakeLockEventHook:
    """Tests for the make_lock_event_hook factory."""

    @pytest.mark.asyncio
    async def test_lock_acquire_success_emits_acquired(self) -> None:
        """lock_acquire with acquired=true emits ACQUIRED event."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        with patch(
            "src.infra.hooks.deadlock.canonicalize_path",
            return_value="/canonical/path/file.py",
        ):
            hook = make_lock_event_hook(
                "agent-1",
                emit,
                "/repo",
                lock_event_class=LockEvent,
                lock_event_type_enum=LockEventType,
            )
            response = make_mcp_response(
                {"results": [{"filepath": "/path/file.py", "acquired": True}]}
            )
            hook_input = make_post_hook_input(
                "mcp__mala-locking__lock_acquire",
                {"filepaths": ["/path/file.py"]},
                tool_response=response,
            )
            await hook(hook_input, None, make_context())

        assert len(events) == 1
        assert events[0].event_type == LockEventType.ACQUIRED
        assert events[0].agent_id == "agent-1"
        assert events[0].lock_path == "/canonical/path/file.py"

    @pytest.mark.asyncio
    async def test_lock_acquire_not_acquired_no_event(self) -> None:
        """lock_acquire with acquired=false emits no event (WAITING emitted by tool)."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        with patch(
            "src.infra.hooks.deadlock.canonicalize_path",
            return_value="/canonical/path/file.py",
        ):
            hook = make_lock_event_hook(
                "agent-1",
                emit,
                "/repo",
                lock_event_class=LockEvent,
                lock_event_type_enum=LockEventType,
            )
            response = make_mcp_response(
                {"results": [{"filepath": "/path/file.py", "acquired": False}]}
            )
            hook_input = make_post_hook_input(
                "mcp__mala-locking__lock_acquire",
                {"filepaths": ["/path/file.py"]},
                tool_response=response,
            )
            await hook(hook_input, None, make_context())

        # WAITING is emitted by the MCP tool handler, not the hook
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_lock_release_emits_released(self) -> None:
        """lock_release emits RELEASED events for released files."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        with patch(
            "src.infra.hooks.deadlock.canonicalize_path",
            return_value="/canonical/path/file.py",
        ):
            hook = make_lock_event_hook(
                "agent-1",
                emit,
                "/repo",
                lock_event_class=LockEvent,
                lock_event_type_enum=LockEventType,
            )
            response = make_mcp_response({"released": ["/path/file.py"]})
            hook_input = make_post_hook_input(
                "mcp__mala-locking__lock_release",
                {"filepaths": ["/path/file.py"]},
                tool_response=response,
            )
            await hook(hook_input, None, make_context())

        assert len(events) == 1
        assert events[0].event_type == LockEventType.RELEASED
        assert events[0].agent_id == "agent-1"
        assert events[0].lock_path == "/canonical/path/file.py"

    @pytest.mark.asyncio
    async def test_non_mcp_tool_ignored(self) -> None:
        """Non-MCP locking tools don't emit events."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        hook = make_lock_event_hook(
            "agent-1",
            emit,
            "/repo",
            lock_event_class=LockEvent,
            lock_event_type_enum=LockEventType,
        )
        hook_input = make_post_hook_input(
            "Bash",
            {"command": "ls -la"},
        )
        await hook(hook_input, None, make_context())

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_non_locking_mcp_tool_ignored(self) -> None:
        """Non-locking MCP tools don't emit events."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        hook = make_lock_event_hook(
            "agent-1",
            emit,
            "/repo",
            lock_event_class=LockEvent,
            lock_event_type_enum=LockEventType,
        )
        hook_input = make_post_hook_input(
            "mcp__other__some_tool",
            {"arg": "value"},
        )
        await hook(hook_input, None, make_context())

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_error_response_no_event(self) -> None:
        """Error response from MCP tool emits no event."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        hook = make_lock_event_hook(
            "agent-1",
            emit,
            "/repo",
            lock_event_class=LockEvent,
            lock_event_type_enum=LockEventType,
        )
        response = make_mcp_response({"error": "Something went wrong"})
        hook_input = make_post_hook_input(
            "mcp__mala-locking__lock_acquire",
            {"filepaths": ["/path/file.py"]},
            tool_response=response,
        )
        await hook(hook_input, None, make_context())

        assert len(events) == 0

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
            hook = make_lock_event_hook(
                "agent-1",
                emit,
                "/repo",
                lock_event_class=LockEvent,
                lock_event_type_enum=LockEventType,
            )
            response = make_mcp_response(
                {"results": [{"filepath": "/bad/path", "acquired": True}]}
            )
            hook_input = make_post_hook_input(
                "mcp__mala-locking__lock_acquire",
                {"filepaths": ["/bad/path"]},
                tool_response=response,
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
            hook = make_lock_event_hook(
                "agent-1",
                emit,
                "/repo",
                lock_event_class=LockEvent,
                lock_event_type_enum=LockEventType,
            )
            response = make_mcp_response(
                {"results": [{"filepath": "/path/file.py", "acquired": True}]}
            )
            hook_input = make_post_hook_input(
                "mcp__mala-locking__lock_acquire",
                {"filepaths": ["/path/file.py"]},
                tool_response=response,
            )
            await hook(hook_input, None, make_context())

        emit.assert_awaited_once()
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_event_has_timestamp(self) -> None:
        """Emitted events have a timestamp."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        with patch(
            "src.infra.hooks.deadlock.canonicalize_path",
            return_value="/canonical/path/file.py",
        ):
            hook = make_lock_event_hook(
                "agent-1",
                emit,
                "/repo",
                lock_event_class=LockEvent,
                lock_event_type_enum=LockEventType,
            )
            response = make_mcp_response(
                {"results": [{"filepath": "/path/file.py", "acquired": True}]}
            )
            hook_input = make_post_hook_input(
                "mcp__mala-locking__lock_acquire",
                {"filepaths": ["/path/file.py"]},
                tool_response=response,
            )
            await hook(hook_input, None, make_context())

        assert len(events) == 1
        assert events[0].timestamp > 0

    @pytest.mark.asyncio
    async def test_multiple_files_acquired(self) -> None:
        """Multiple files acquired in one call emit multiple events."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        with patch(
            "src.infra.hooks.deadlock.canonicalize_path",
            side_effect=lambda p, _: f"/canonical{p}",
        ):
            hook = make_lock_event_hook(
                "agent-1",
                emit,
                "/repo",
                lock_event_class=LockEvent,
                lock_event_type_enum=LockEventType,
            )
            response = make_mcp_response(
                {
                    "results": [
                        {"filepath": "/a.py", "acquired": True},
                        {"filepath": "/b.py", "acquired": True},
                        {"filepath": "/c.py", "acquired": False},  # Not acquired
                    ]
                }
            )
            hook_input = make_post_hook_input(
                "mcp__mala-locking__lock_acquire",
                {"filepaths": ["/a.py", "/b.py", "/c.py"]},
                tool_response=response,
            )
            await hook(hook_input, None, make_context())

        # Only 2 events (c.py was not acquired)
        assert len(events) == 2
        assert events[0].lock_path == "/canonical/a.py"
        assert events[1].lock_path == "/canonical/b.py"

    @pytest.mark.asyncio
    async def test_multiple_files_released(self) -> None:
        """Multiple files released in one call emit multiple events."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        with patch(
            "src.infra.hooks.deadlock.canonicalize_path",
            side_effect=lambda p, _: f"/canonical{p}",
        ):
            hook = make_lock_event_hook(
                "agent-1",
                emit,
                "/repo",
                lock_event_class=LockEvent,
                lock_event_type_enum=LockEventType,
            )
            response = make_mcp_response({"released": ["/a.py", "/b.py"]})
            hook_input = make_post_hook_input(
                "mcp__mala-locking__lock_release",
                {"filepaths": ["/a.py", "/b.py"]},
                tool_response=response,
            )
            await hook(hook_input, None, make_context())

        assert len(events) == 2
        assert events[0].lock_path == "/canonical/a.py"
        assert events[1].lock_path == "/canonical/b.py"
        assert all(e.event_type == LockEventType.RELEASED for e in events)

    @pytest.mark.asyncio
    async def test_empty_response_no_event(self) -> None:
        """Empty tool response emits no events."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        hook = make_lock_event_hook(
            "agent-1",
            emit,
            "/repo",
            lock_event_class=LockEvent,
            lock_event_type_enum=LockEventType,
        )
        hook_input = make_post_hook_input(
            "mcp__mala-locking__lock_acquire",
            {"filepaths": ["/path/file.py"]},
            tool_response={"content": []},
        )
        await hook(hook_input, None, make_context())

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_non_text_content_ignored(self) -> None:
        """Non-text content blocks are ignored."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        hook = make_lock_event_hook(
            "agent-1",
            emit,
            "/repo",
            lock_event_class=LockEvent,
            lock_event_type_enum=LockEventType,
        )
        hook_input = make_post_hook_input(
            "mcp__mala-locking__lock_acquire",
            {"filepaths": ["/path/file.py"]},
            tool_response={"content": [{"type": "image", "data": "..."}]},
        )
        await hook(hook_input, None, make_context())

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_invalid_json_logs_warning(self) -> None:
        """Invalid JSON in response logs warning."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        with patch("src.infra.hooks.deadlock.logger") as mock_logger:
            hook = make_lock_event_hook(
                "agent-1",
                emit,
                "/repo",
                lock_event_class=LockEvent,
                lock_event_type_enum=LockEventType,
            )
            hook_input = make_post_hook_input(
                "mcp__mala-locking__lock_acquire",
                {"filepaths": ["/path/file.py"]},
                tool_response={"content": [{"type": "text", "text": "not json"}]},
            )
            await hook(hook_input, None, make_context())

        assert len(events) == 0
        mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_tool_response_no_event(self) -> None:
        """Missing tool_response emits no events."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        hook = make_lock_event_hook(
            "agent-1",
            emit,
            "/repo",
            lock_event_class=LockEvent,
            lock_event_type_enum=LockEventType,
        )
        hook_input = make_post_hook_input(
            "mcp__mala-locking__lock_acquire",
            {"filepaths": ["/path/file.py"]},
            # No tool_response
        )
        await hook(hook_input, None, make_context())

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_empty_text_in_response_no_event(self) -> None:
        """Empty text in response content emits no events."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        hook = make_lock_event_hook(
            "agent-1",
            emit,
            "/repo",
            lock_event_class=LockEvent,
            lock_event_type_enum=LockEventType,
        )
        hook_input = make_post_hook_input(
            "mcp__mala-locking__lock_acquire",
            {"filepaths": ["/path/file.py"]},
            tool_response={"content": [{"type": "text", "text": ""}]},
        )
        await hook(hook_input, None, make_context())

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_empty_filepath_in_acquire_ignored(self) -> None:
        """Empty filepath in acquire results is skipped."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        with patch(
            "src.infra.hooks.deadlock.canonicalize_path",
            return_value="/canonical/path/file.py",
        ):
            hook = make_lock_event_hook(
                "agent-1",
                emit,
                "/repo",
                lock_event_class=LockEvent,
                lock_event_type_enum=LockEventType,
            )
            response = make_mcp_response(
                {
                    "results": [
                        {"filepath": "", "acquired": True},  # Empty filepath
                        {"filepath": "/valid.py", "acquired": True},  # Valid
                    ]
                }
            )
            hook_input = make_post_hook_input(
                "mcp__mala-locking__lock_acquire",
                {"filepaths": ["/valid.py"]},
                tool_response=response,
            )
            await hook(hook_input, None, make_context())

        # Only the valid filepath should emit an event
        assert len(events) == 1
        assert events[0].lock_path == "/canonical/path/file.py"

    @pytest.mark.asyncio
    async def test_empty_filepath_in_release_ignored(self) -> None:
        """Empty filepath in release results is skipped."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        with patch(
            "src.infra.hooks.deadlock.canonicalize_path",
            return_value="/canonical/path/file.py",
        ):
            hook = make_lock_event_hook(
                "agent-1",
                emit,
                "/repo",
                lock_event_class=LockEvent,
                lock_event_type_enum=LockEventType,
            )
            response = make_mcp_response(
                {
                    "released": ["", "/valid.py"]  # First is empty
                }
            )
            hook_input = make_post_hook_input(
                "mcp__mala-locking__lock_release",
                {"filepaths": ["/valid.py"]},
                tool_response=response,
            )
            await hook(hook_input, None, make_context())

        # Only the valid filepath should emit an event
        assert len(events) == 1
        assert events[0].lock_path == "/canonical/path/file.py"
        assert events[0].event_type == LockEventType.RELEASED

    @pytest.mark.asyncio
    async def test_release_canonicalization_failure_logs_warning(self) -> None:
        """Failed path canonicalization in release logs warning, no event."""
        events: list[LockEvent] = []
        emit = MagicMock(side_effect=lambda e: events.append(e))

        with (
            patch(
                "src.infra.hooks.deadlock.canonicalize_path",
                side_effect=ValueError("bad path"),
            ),
            patch("src.infra.hooks.deadlock.logger") as mock_logger,
        ):
            hook = make_lock_event_hook(
                "agent-1",
                emit,
                "/repo",
                lock_event_class=LockEvent,
                lock_event_type_enum=LockEventType,
            )
            response = make_mcp_response({"released": ["/bad/path"]})
            hook_input = make_post_hook_input(
                "mcp__mala-locking__lock_release",
                {"filepaths": ["/bad/path"]},
                tool_response=response,
            )
            await hook(hook_input, None, make_context())

        assert len(events) == 0
        mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_release_async_emit_callback(self) -> None:
        """Async emit callback is awaited for release events."""
        events: list[LockEvent] = []

        async def async_emit(event: LockEvent) -> None:
            events.append(event)

        emit = AsyncMock(side_effect=async_emit)

        with patch(
            "src.infra.hooks.deadlock.canonicalize_path",
            return_value="/canonical/path/file.py",
        ):
            hook = make_lock_event_hook(
                "agent-1",
                emit,
                "/repo",
                lock_event_class=LockEvent,
                lock_event_type_enum=LockEventType,
            )
            response = make_mcp_response({"released": ["/path/file.py"]})
            hook_input = make_post_hook_input(
                "mcp__mala-locking__lock_release",
                {"filepaths": ["/path/file.py"]},
                tool_response=response,
            )
            await hook(hook_input, None, make_context())

        emit.assert_awaited_once()
        assert len(events) == 1
        assert events[0].event_type == LockEventType.RELEASED

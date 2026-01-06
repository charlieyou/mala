"""Unit tests for the deadlock PostToolUse hook."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, cast

import pytest

if TYPE_CHECKING:
    from pathlib import Path

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
    tool_response: dict[str, Any] | list[dict[str, Any]] | None = None,
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


def collect_events() -> tuple[list[LockEvent], Any]:
    """Create an event collector and emit callback for tests."""
    events: list[LockEvent] = []

    def emit(event: LockEvent) -> None:
        events.append(event)

    return events, emit


@pytest.mark.unit
class TestMakeLockEventHook:
    """Tests for the make_lock_event_hook factory."""

    @pytest.mark.asyncio
    async def test_lock_acquire_success_emits_acquired(self, tmp_path: Path) -> None:
        """lock_acquire with acquired=true emits ACQUIRED event."""
        events, emit = collect_events()
        test_file = tmp_path / "file.py"
        test_file.touch()
        filepath = str(test_file)

        hook = make_lock_event_hook(
            "agent-1",
            emit,
            str(tmp_path),
            lock_event_class=LockEvent,
            lock_event_type_enum=LockEventType,
        )
        response = make_mcp_response(
            {"results": [{"filepath": filepath, "acquired": True}]}
        )
        hook_input = make_post_hook_input(
            "mcp__mala-locking__lock_acquire",
            {"filepaths": [filepath]},
            tool_response=response,
        )
        await hook(hook_input, None, make_context())

        assert len(events) == 1
        assert events[0].event_type == LockEventType.ACQUIRED
        assert events[0].agent_id == "agent-1"
        # Path should be canonicalized (resolved)
        assert events[0].lock_path == str(test_file.resolve())

    @pytest.mark.asyncio
    async def test_lock_acquire_list_tool_response_emits_acquired(
        self, tmp_path: Path
    ) -> None:
        """lock_acquire handles list-shaped tool_response content."""
        events, emit = collect_events()
        test_file = tmp_path / "file.py"
        test_file.touch()
        filepath = str(test_file)

        hook = make_lock_event_hook(
            "agent-1",
            emit,
            str(tmp_path),
            lock_event_class=LockEvent,
            lock_event_type_enum=LockEventType,
        )
        response = [
            {
                "type": "text",
                "text": json.dumps(
                    {"results": [{"filepath": filepath, "acquired": True}]}
                ),
            }
        ]
        hook_input = make_post_hook_input(
            "mcp__mala-locking__lock_acquire",
            {"filepaths": [filepath]},
            tool_response=response,
        )
        await hook(hook_input, None, make_context())

        assert len(events) == 1
        assert events[0].event_type == LockEventType.ACQUIRED
        assert events[0].agent_id == "agent-1"

    @pytest.mark.asyncio
    async def test_lock_acquire_not_acquired_no_event(self, tmp_path: Path) -> None:
        """lock_acquire with acquired=false emits no event (WAITING emitted by tool)."""
        events, emit = collect_events()
        test_file = tmp_path / "file.py"
        test_file.touch()
        filepath = str(test_file)

        hook = make_lock_event_hook(
            "agent-1",
            emit,
            str(tmp_path),
            lock_event_class=LockEvent,
            lock_event_type_enum=LockEventType,
        )
        response = make_mcp_response(
            {"results": [{"filepath": filepath, "acquired": False}]}
        )
        hook_input = make_post_hook_input(
            "mcp__mala-locking__lock_acquire",
            {"filepaths": [filepath]},
            tool_response=response,
        )
        await hook(hook_input, None, make_context())

        # WAITING is emitted by the MCP tool handler, not the hook
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_lock_release_emits_released(self, tmp_path: Path) -> None:
        """lock_release emits RELEASED events for released files."""
        events, emit = collect_events()
        test_file = tmp_path / "file.py"
        test_file.touch()
        filepath = str(test_file)

        hook = make_lock_event_hook(
            "agent-1",
            emit,
            str(tmp_path),
            lock_event_class=LockEvent,
            lock_event_type_enum=LockEventType,
        )
        response = make_mcp_response({"released": [filepath]})
        hook_input = make_post_hook_input(
            "mcp__mala-locking__lock_release",
            {"filepaths": [filepath]},
            tool_response=response,
        )
        await hook(hook_input, None, make_context())

        assert len(events) == 1
        assert events[0].event_type == LockEventType.RELEASED
        assert events[0].agent_id == "agent-1"
        assert events[0].lock_path == str(test_file.resolve())

    @pytest.mark.asyncio
    async def test_non_mcp_tool_ignored(self) -> None:
        """Non-MCP locking tools don't emit events."""
        events, emit = collect_events()

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
        events, emit = collect_events()

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
        events, emit = collect_events()

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
    async def test_async_emit_callback(self, tmp_path: Path) -> None:
        """Async emit callback is awaited."""
        events: list[LockEvent] = []

        async def async_emit(event: LockEvent) -> None:
            events.append(event)

        test_file = tmp_path / "file.py"
        test_file.touch()
        filepath = str(test_file)

        hook = make_lock_event_hook(
            "agent-1",
            async_emit,
            str(tmp_path),
            lock_event_class=LockEvent,
            lock_event_type_enum=LockEventType,
        )
        response = make_mcp_response(
            {"results": [{"filepath": filepath, "acquired": True}]}
        )
        hook_input = make_post_hook_input(
            "mcp__mala-locking__lock_acquire",
            {"filepaths": [filepath]},
            tool_response=response,
        )
        await hook(hook_input, None, make_context())

        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_event_has_timestamp(self, tmp_path: Path) -> None:
        """Emitted events have a timestamp."""
        events, emit = collect_events()
        test_file = tmp_path / "file.py"
        test_file.touch()
        filepath = str(test_file)

        hook = make_lock_event_hook(
            "agent-1",
            emit,
            str(tmp_path),
            lock_event_class=LockEvent,
            lock_event_type_enum=LockEventType,
        )
        response = make_mcp_response(
            {"results": [{"filepath": filepath, "acquired": True}]}
        )
        hook_input = make_post_hook_input(
            "mcp__mala-locking__lock_acquire",
            {"filepaths": [filepath]},
            tool_response=response,
        )
        await hook(hook_input, None, make_context())

        assert len(events) == 1
        assert events[0].timestamp > 0

    @pytest.mark.asyncio
    async def test_multiple_files_acquired(self, tmp_path: Path) -> None:
        """Multiple files acquired in one call emit multiple events."""
        events, emit = collect_events()

        # Create test files
        file_a = tmp_path / "a.py"
        file_b = tmp_path / "b.py"
        file_c = tmp_path / "c.py"
        file_a.touch()
        file_b.touch()
        file_c.touch()

        hook = make_lock_event_hook(
            "agent-1",
            emit,
            str(tmp_path),
            lock_event_class=LockEvent,
            lock_event_type_enum=LockEventType,
        )
        response = make_mcp_response(
            {
                "results": [
                    {"filepath": str(file_a), "acquired": True},
                    {"filepath": str(file_b), "acquired": True},
                    {"filepath": str(file_c), "acquired": False},  # Not acquired
                ]
            }
        )
        hook_input = make_post_hook_input(
            "mcp__mala-locking__lock_acquire",
            {"filepaths": [str(file_a), str(file_b), str(file_c)]},
            tool_response=response,
        )
        await hook(hook_input, None, make_context())

        # Only 2 events (c.py was not acquired)
        assert len(events) == 2
        lock_paths = {e.lock_path for e in events}
        assert str(file_a.resolve()) in lock_paths
        assert str(file_b.resolve()) in lock_paths

    @pytest.mark.asyncio
    async def test_multiple_files_released(self, tmp_path: Path) -> None:
        """Multiple files released in one call emit multiple events."""
        events, emit = collect_events()

        file_a = tmp_path / "a.py"
        file_b = tmp_path / "b.py"
        file_a.touch()
        file_b.touch()

        hook = make_lock_event_hook(
            "agent-1",
            emit,
            str(tmp_path),
            lock_event_class=LockEvent,
            lock_event_type_enum=LockEventType,
        )
        response = make_mcp_response({"released": [str(file_a), str(file_b)]})
        hook_input = make_post_hook_input(
            "mcp__mala-locking__lock_release",
            {"filepaths": [str(file_a), str(file_b)]},
            tool_response=response,
        )
        await hook(hook_input, None, make_context())

        assert len(events) == 2
        lock_paths = {e.lock_path for e in events}
        assert str(file_a.resolve()) in lock_paths
        assert str(file_b.resolve()) in lock_paths
        assert all(e.event_type == LockEventType.RELEASED for e in events)

    @pytest.mark.asyncio
    async def test_empty_response_no_event(self) -> None:
        """Empty tool response emits no events."""
        events, emit = collect_events()

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
        events, emit = collect_events()

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
    async def test_invalid_json_no_event(self) -> None:
        """Invalid JSON in response emits no events."""
        events, emit = collect_events()

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

    @pytest.mark.asyncio
    async def test_no_tool_response_no_event(self) -> None:
        """Missing tool_response emits no events."""
        events, emit = collect_events()

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
        events, emit = collect_events()

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
    async def test_empty_filepath_in_acquire_ignored(self, tmp_path: Path) -> None:
        """Empty filepath in acquire results is skipped."""
        events, emit = collect_events()
        valid_file = tmp_path / "valid.py"
        valid_file.touch()

        hook = make_lock_event_hook(
            "agent-1",
            emit,
            str(tmp_path),
            lock_event_class=LockEvent,
            lock_event_type_enum=LockEventType,
        )
        response = make_mcp_response(
            {
                "results": [
                    {"filepath": "", "acquired": True},  # Empty filepath
                    {"filepath": str(valid_file), "acquired": True},  # Valid
                ]
            }
        )
        hook_input = make_post_hook_input(
            "mcp__mala-locking__lock_acquire",
            {"filepaths": [str(valid_file)]},
            tool_response=response,
        )
        await hook(hook_input, None, make_context())

        # Only the valid filepath should emit an event
        assert len(events) == 1
        assert events[0].lock_path == str(valid_file.resolve())

    @pytest.mark.asyncio
    async def test_empty_filepath_in_release_ignored(self, tmp_path: Path) -> None:
        """Empty filepath in release results is skipped."""
        events, emit = collect_events()
        valid_file = tmp_path / "valid.py"
        valid_file.touch()

        hook = make_lock_event_hook(
            "agent-1",
            emit,
            str(tmp_path),
            lock_event_class=LockEvent,
            lock_event_type_enum=LockEventType,
        )
        response = make_mcp_response(
            {
                "released": ["", str(valid_file)]  # First is empty
            }
        )
        hook_input = make_post_hook_input(
            "mcp__mala-locking__lock_release",
            {"filepaths": [str(valid_file)]},
            tool_response=response,
        )
        await hook(hook_input, None, make_context())

        # Only the valid filepath should emit an event
        assert len(events) == 1
        assert events[0].lock_path == str(valid_file.resolve())
        assert events[0].event_type == LockEventType.RELEASED

    @pytest.mark.asyncio
    async def test_release_async_emit_callback(self, tmp_path: Path) -> None:
        """Async emit callback is awaited for release events."""
        events: list[LockEvent] = []

        async def async_emit(event: LockEvent) -> None:
            events.append(event)

        test_file = tmp_path / "file.py"
        test_file.touch()
        filepath = str(test_file)

        hook = make_lock_event_hook(
            "agent-1",
            async_emit,
            str(tmp_path),
            lock_event_class=LockEvent,
            lock_event_type_enum=LockEventType,
        )
        response = make_mcp_response({"released": [filepath]})
        hook_input = make_post_hook_input(
            "mcp__mala-locking__lock_release",
            {"filepaths": [filepath]},
            tool_response=response,
        )
        await hook(hook_input, None, make_context())

        assert len(events) == 1
        assert events[0].event_type == LockEventType.RELEASED

    @pytest.mark.asyncio
    async def test_path_canonicalization_failure_logs_warning_no_event(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When canonicalize_path raises an exception, log warning and skip event.

        This tests the fail-soft behavior in _process_acquire_response where
        path canonicalization failures are caught, logged, and the event is
        skipped rather than crashing the hook.
        """
        import logging

        from src.infra.hooks import deadlock

        events, emit = collect_events()

        # Make canonicalize_path raise an exception
        def failing_canonicalize(
            filepath: str, repo_namespace: str | None = None
        ) -> str:
            raise ValueError(f"Cannot canonicalize path: {filepath}")

        monkeypatch.setattr(deadlock, "canonicalize_path", failing_canonicalize)

        hook = make_lock_event_hook(
            "agent-1",
            emit,
            str(tmp_path),
            lock_event_class=LockEvent,
            lock_event_type_enum=LockEventType,
        )
        response = make_mcp_response(
            {"results": [{"filepath": "/some/path/file.py", "acquired": True}]}
        )
        hook_input = make_post_hook_input(
            "mcp__mala-locking__lock_acquire",
            {"filepaths": ["/some/path/file.py"]},
            tool_response=response,
        )

        with caplog.at_level(logging.WARNING):
            await hook(hook_input, None, make_context())

        # No event should be emitted due to canonicalization failure
        assert len(events) == 0
        # Warning should be logged
        assert any(
            "Failed to canonicalize" in record.message for record in caplog.records
        )

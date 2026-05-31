"""Unit tests for SDK adapter settings."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, cast

import pytest

from src.core.protocols.agent_event import (
    AgentResultEvent,
    AgentTaskCompletedEvent,
    AgentTaskStartedEvent,
)
from src.infra.sdk_adapter import (
    CLAUDE_STDOUT_MAX_BUFFER_SIZE,
    SDKClientFactory,
    _ClaudeAgentEventClient,
)

if TYPE_CHECKING:
    from claude_agent_sdk.types import ClaudeAgentOptions

    from src.core.protocols.sdk import SDKClientProtocol


@pytest.fixture
def factory() -> SDKClientFactory:
    """Create an SDKClientFactory instance for testing."""
    return SDKClientFactory()


class TestSettingSources:
    """Tests for setting_sources behavior in create_options."""

    def test_default_setting_sources(
        self, factory: SDKClientFactory, tmp_path: Path
    ) -> None:
        """Default setting_sources is ['local', 'project'] when not specified."""
        options = cast("ClaudeAgentOptions", factory.create_options(cwd=str(tmp_path)))
        assert options.setting_sources == ["local", "project"]

    def test_default_setting_sources_explicit_none(
        self, factory: SDKClientFactory, tmp_path: Path
    ) -> None:
        """Explicit None for setting_sources uses default ['local', 'project']."""
        options = cast(
            "ClaudeAgentOptions",
            factory.create_options(cwd=str(tmp_path), setting_sources=None),
        )
        assert options.setting_sources == ["local", "project"]

    def test_override_setting_sources(
        self, factory: SDKClientFactory, tmp_path: Path
    ) -> None:
        """Explicit setting_sources overrides the default."""
        options = cast(
            "ClaudeAgentOptions",
            factory.create_options(cwd=str(tmp_path), setting_sources=["user"]),
        )
        assert options.setting_sources == ["user"]

    def test_override_setting_sources_multiple(
        self, factory: SDKClientFactory, tmp_path: Path
    ) -> None:
        """Multiple sources can be provided as override."""
        options = cast(
            "ClaudeAgentOptions",
            factory.create_options(
                cwd=str(tmp_path), setting_sources=["project", "user"]
            ),
        )
        assert options.setting_sources == ["project", "user"]

    def test_override_setting_sources_empty_list(
        self, factory: SDKClientFactory, tmp_path: Path
    ) -> None:
        """Empty list preserves explicit override (no defaults)."""
        options = cast(
            "ClaudeAgentOptions",
            factory.create_options(cwd=str(tmp_path), setting_sources=[]),
        )
        assert options.setting_sources == []


class TestMaxBufferSize:
    """Tests for Claude subprocess stdout buffer sizing."""

    def test_create_options_raises_sdk_stdout_buffer_limit(
        self, factory: SDKClientFactory, tmp_path: Path
    ) -> None:
        """Large TaskOutput JSON events should not hit the SDK's 1 MiB default."""
        options = cast("ClaudeAgentOptions", factory.create_options(cwd=str(tmp_path)))

        assert options.max_buffer_size == CLAUDE_STDOUT_MAX_BUFFER_SIZE


def _make_message(class_name: str, **attrs: object) -> object:
    """Build a message instance whose ``type(...).__name__`` is ``class_name``.

    ``to_agent_events`` detects SDK messages by class name, so each call must
    produce a distinct type (a single shared class mutated in place would
    report the last name set for every instance).
    """
    cls = type(class_name, (), {})
    instance = cls()
    for key, value in attrs.items():
        setattr(instance, key, value)
    return instance


class _FakeInner:
    """Inner SDK client exposing a continuous ``receive_messages`` stream."""

    def __init__(self, messages: list[object]) -> None:
        self._messages = messages
        self.stopped: list[str] = []

    async def receive_messages(self) -> object:
        for msg in self._messages:
            yield msg

    async def stop_task(self, task_id: str) -> None:
        self.stopped.append(task_id)


class TestBackgroundTaskSurface:
    """The Claude wrapper exposes the background-task capability + stream."""

    def test_supports_background_tasks_is_true(self) -> None:
        client = _ClaudeAgentEventClient(cast("SDKClientProtocol", _FakeInner([])))
        assert client.supports_background_tasks() is True

    @pytest.mark.asyncio
    async def test_receive_messages_translates_continuous_stream(self) -> None:
        inner = _FakeInner(
            [
                _make_message("TaskStartedMessage", task_id="bg-1", tool_use_id="t1"),
                _make_message(
                    "TaskNotificationMessage",
                    task_id="bg-1",
                    tool_use_id="t1",
                    status="completed",
                    summary="exit code 0",
                    output_file="/tmp/out.log",
                ),
                _make_message("ResultMessage", session_id="s1", is_error=False),
            ]
        )
        client = _ClaudeAgentEventClient(cast("SDKClientProtocol", inner))
        events = [event async for event in client.receive_messages()]
        assert events == [
            AgentTaskStartedEvent(task_id="bg-1", tool_use_id="t1"),
            AgentTaskCompletedEvent(
                task_id="bg-1",
                tool_use_id="t1",
                status="completed",
                summary="exit code 0",
                output_file="/tmp/out.log",
            ),
            AgentResultEvent(session_id="s1", is_error=False),
        ]

    @pytest.mark.asyncio
    async def test_stop_task_delegates_to_inner(self) -> None:
        inner = _FakeInner([])
        client = _ClaudeAgentEventClient(cast("SDKClientProtocol", inner))
        await client.stop_task("bg-1")
        assert inner.stopped == ["bg-1"]

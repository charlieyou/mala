"""Unit tests for SDK adapter settings."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, cast

import pytest

from src.core.protocols.agent_event import (
    AgentResultEvent,
    AgentTaskCompletedEvent,
    AgentTaskStartedEvent,
    AgentToolResultEvent,
    AgentToolUseEvent,
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


class _FakeRawQuery:
    """Private Claude Query look-alike yielding raw CLI message dicts."""

    def __init__(self, messages: list[dict[str, object]]) -> None:
        self._messages = messages

    async def receive_messages(self) -> object:
        for msg in self._messages:
            yield msg


class _FakeRawInner(_FakeInner):
    """Inner client exposing the private raw ``_query`` Mala uses for Claude."""

    def __init__(self, messages: list[dict[str, object]]) -> None:
        super().__init__([])
        self._query = _FakeRawQuery(messages)


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
    async def test_receive_response_translates_raw_queue_notifications(self) -> None:
        """Claude queued task notifications are recovered before SDK parsing drops them."""
        inner = _FakeRawInner(
            [
                {
                    "type": "assistant",
                    "session_id": "s1",
                    "message": {
                        "model": "claude-opus",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tool-bg",
                                "name": "Bash",
                                "input": {
                                    "command": "slow.sh",
                                    "run_in_background": True,
                                },
                            }
                        ],
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-bg",
                                "content": "Command running in background with ID: task-1.",
                                "is_error": False,
                            }
                        ],
                    },
                },
                {
                    "type": "queue-operation",
                    "operation": "enqueue",
                    "content": "\n".join(
                        [
                            "<task-notification>",
                            "<task-id>task-1</task-id>",
                            "<tool-use-id>tool-bg</tool-use-id>",
                            "<output-file>/tmp/task-1.output</output-file>",
                            "<status>completed</status>",
                            "<summary>Background command completed</summary>",
                            "</task-notification>",
                        ]
                    ),
                },
                {
                    "type": "result",
                    "subtype": "success",
                    "duration_ms": 1,
                    "duration_api_ms": 1,
                    "is_error": False,
                    "num_turns": 1,
                    "session_id": "s1",
                    "result": "done",
                },
                {
                    "type": "assistant",
                    "session_id": "s1",
                    "message": {
                        "model": "claude-opus",
                        "content": [{"type": "text", "text": "after result"}],
                    },
                },
            ]
        )
        client = _ClaudeAgentEventClient(cast("SDKClientProtocol", inner))

        events = [event async for event in client.receive_response()]

        assert events == [
            AgentToolUseEvent(
                id="tool-bg",
                name="Bash",
                input={"command": "slow.sh", "run_in_background": True},
            ),
            AgentToolResultEvent(
                tool_use_id="tool-bg",
                is_error=False,
                content="Command running in background with ID: task-1.",
            ),
            AgentTaskCompletedEvent(
                task_id="task-1",
                tool_use_id="tool-bg",
                status="completed",
                summary="Background command completed",
                output_file="/tmp/task-1.output",
            ),
            AgentResultEvent(
                session_id="s1",
                is_error=False,
                subtype="success",
                result="done",
            ),
        ]

    @pytest.mark.asyncio
    async def test_receive_response_recovers_local_transcript_task_notifications(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Transcript-only task notifications clear background launches before result."""
        claude_config = tmp_path / "claude"
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_config))
        transcript_dir = claude_config / "projects" / "-repo"
        transcript_dir.mkdir(parents=True)
        session_id = "s-transcript"
        notification_text = "\n".join(
            [
                "<task-notification>",
                "<task-id>task-transcript</task-id>",
                "<tool-use-id>tool-bg-transcript</tool-use-id>",
                "<output-file>/tmp/task-transcript.output</output-file>",
                "<status>completed</status>",
                "<summary>Background command completed</summary>",
                "</task-notification>",
            ]
        )
        (transcript_dir / f"{session_id}.jsonl").write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "type": "queue-operation",
                            "operation": "enqueue",
                            "content": notification_text,
                        }
                    ),
                    json.dumps(
                        {
                            "type": "attachment",
                            "attachment": {
                                "type": "queued_command",
                                "commandMode": "task-notification",
                                "prompt": notification_text,
                            },
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        inner = _FakeRawInner(
            [
                {
                    "type": "assistant",
                    "session_id": session_id,
                    "message": {
                        "model": "claude-opus",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tool-bg-transcript",
                                "name": "Bash",
                                "input": {
                                    "command": "slow.sh",
                                    "run_in_background": True,
                                },
                            }
                        ],
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-bg-transcript",
                                "content": "Command running in background with ID: task-transcript.",
                                "is_error": False,
                            }
                        ],
                    },
                },
                {
                    "type": "result",
                    "subtype": "success",
                    "duration_ms": 1,
                    "duration_api_ms": 1,
                    "is_error": False,
                    "num_turns": 1,
                    "session_id": session_id,
                    "result": "done",
                },
            ]
        )
        client = _ClaudeAgentEventClient(cast("SDKClientProtocol", inner))

        events = [event async for event in client.receive_response()]

        assert events == [
            AgentToolUseEvent(
                id="tool-bg-transcript",
                name="Bash",
                input={"command": "slow.sh", "run_in_background": True},
            ),
            AgentToolResultEvent(
                tool_use_id="tool-bg-transcript",
                is_error=False,
                content="Command running in background with ID: task-transcript.",
            ),
            AgentTaskCompletedEvent(
                task_id="task-transcript",
                tool_use_id="tool-bg-transcript",
                status="completed",
                summary="Background command completed",
                output_file="/tmp/task-transcript.output",
            ),
            AgentResultEvent(
                session_id=session_id,
                is_error=False,
                subtype="success",
                result="done",
            ),
        ]

    @pytest.mark.asyncio
    async def test_receive_messages_drains_transcript_notifications_after_result(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Close the race where transcript-only completion appears after result."""
        claude_config = tmp_path / "claude"
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_config))
        transcript_dir = claude_config / "projects" / "-repo"
        transcript_dir.mkdir(parents=True)
        session_id = "s-race"
        transcript = transcript_dir / f"{session_id}.jsonl"
        transcript.write_text("", encoding="utf-8")
        inner = _FakeRawInner(
            [
                {
                    "type": "assistant",
                    "session_id": session_id,
                    "message": {
                        "model": "claude-opus",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "tool-bg-race",
                                "name": "Bash",
                                "input": {
                                    "command": "slow.sh",
                                    "run_in_background": True,
                                },
                            }
                        ],
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "tool-bg-race",
                                "content": "Command running in background with ID: task-race.",
                                "is_error": False,
                            }
                        ],
                    },
                },
                {
                    "type": "result",
                    "subtype": "success",
                    "duration_ms": 1,
                    "duration_api_ms": 1,
                    "is_error": False,
                    "num_turns": 1,
                    "session_id": session_id,
                    "result": "done",
                },
            ]
        )
        client = _ClaudeAgentEventClient(cast("SDKClientProtocol", inner))

        response_events = [event async for event in client.receive_response()]
        assert response_events[-1] == AgentResultEvent(
            session_id=session_id,
            is_error=False,
            subtype="success",
            result="done",
        )

        notification_text = "\n".join(
            [
                "<task-notification>",
                "<task-id>task-race</task-id>",
                "<tool-use-id>tool-bg-race</tool-use-id>",
                "<output-file>/tmp/task-race.output</output-file>",
                "<status>completed</status>",
                "<summary>Background command completed</summary>",
                "</task-notification>",
            ]
        )
        transcript.write_text(
            json.dumps(
                {
                    "type": "queue-operation",
                    "operation": "enqueue",
                    "content": notification_text,
                }
            )
            + "\n",
            encoding="utf-8",
        )

        first_event: AgentTaskCompletedEvent | None = None
        async for event in client.receive_messages():
            first_event = cast("AgentTaskCompletedEvent", event)
            break

        assert first_event == AgentTaskCompletedEvent(
            task_id="task-race",
            tool_use_id="tool-bg-race",
            status="completed",
            summary="Background command completed",
            output_file="/tmp/task-race.output",
        )

    @pytest.mark.asyncio
    async def test_receive_messages_translates_raw_task_notification_attachment(
        self,
    ) -> None:
        inner = _FakeRawInner(
            [
                {
                    "type": "attachment",
                    "attachment": {
                        "type": "queued_command",
                        "commandMode": "task-notification",
                        "prompt": "\n".join(
                            [
                                "<task-notification>",
                                "<task-id>task-2</task-id>",
                                "<tool-use-id>tool-bg-2</tool-use-id>",
                                "<status>failed</status>",
                                "<summary>exit code 100</summary>",
                                "</task-notification>",
                            ]
                        ),
                    },
                }
            ]
        )
        client = _ClaudeAgentEventClient(cast("SDKClientProtocol", inner))

        events = [event async for event in client.receive_messages()]

        assert events == [
            AgentTaskCompletedEvent(
                task_id="task-2",
                tool_use_id="tool-bg-2",
                status="failed",
                summary="exit code 100",
            )
        ]

    @pytest.mark.asyncio
    async def test_stop_task_delegates_to_inner(self) -> None:
        inner = _FakeInner([])
        client = _ClaudeAgentEventClient(cast("SDKClientProtocol", inner))
        await client.stop_task("bg-1")
        assert inner.stopped == ["bg-1"]

"""Unit tests for ContextPressureHandler.

Tests the extracted context pressure handling logic using fake SDK clients,
without actual SDK/API dependencies.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import pytest
from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

from src.pipeline.context_pressure_handler import (
    ContextPressureConfig,
    ContextPressureHandler,
)
from src.pipeline.message_stream_processor import ContextPressureError
from tests.fakes.sdk_client import FakeSDKClient, FakeSDKClientFactory

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


# --- Fake SDK Client and Factory ---


def make_result_message(session_id: str = "test-session") -> ResultMessage:
    """Create a ResultMessage for testing."""
    return ResultMessage(
        subtype="result",
        duration_ms=100,
        duration_api_ms=50,
        is_error=False,
        num_turns=1,
        session_id=session_id,
        result="Checkpoint complete",
    )


class DelayedSDKClient(FakeSDKClient):
    """SDK client that delays responses to test timeouts."""

    def __init__(
        self,
        delay_seconds: float,
        messages: list[Any] | None = None,
        result_message: ResultMessage | None = None,
        query_error: Exception | None = None,
    ):
        super().__init__(
            messages=messages or [],
            result_message=result_message,
            query_error=query_error,
        )
        self.delay_seconds = delay_seconds

    async def receive_response(self) -> AsyncIterator[Any]:
        await asyncio.sleep(self.delay_seconds)
        async for msg in super().receive_response():
            yield msg


# --- Test Fixtures ---


@pytest.fixture
def default_config() -> ContextPressureConfig:
    """Create default config with checkpoint prompts."""
    return ContextPressureConfig(
        checkpoint_request_prompt="Please provide a checkpoint summary.",
        continuation_template="Continue from checkpoint:\n\n{checkpoint}",
        checkpoint_timeout_seconds=5.0,
    )


@pytest.fixture
def empty_prompt_config() -> ContextPressureConfig:
    """Create config with empty checkpoint prompt."""
    return ContextPressureConfig(
        checkpoint_request_prompt="",
        continuation_template="Continue from checkpoint:\n\n{checkpoint}",
    )


# --- Test Classes ---


class TestCheckpointFetch:
    """Tests for fetch_checkpoint method."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_extracts_checkpoint_from_response(
        self, default_config: ContextPressureConfig
    ) -> None:
        """Checkpoint is extracted from <checkpoint> tags in response."""
        checkpoint_msg = AssistantMessage(
            content=[TextBlock(text="<checkpoint>step 3 of 5</checkpoint>")],
            model="test",
        )
        client = FakeSDKClient(messages=[checkpoint_msg])
        factory = FakeSDKClientFactory(client)

        handler = ContextPressureHandler(default_config, factory)

        result = await handler.fetch_checkpoint(
            session_id="test-session-123",
            issue_id="issue-1",
            options={},
        )

        assert result.checkpoint == "step 3 of 5"
        assert result.timed_out is False
        assert len(client.queries) == 1
        assert client.queries[0] == (
            "Please provide a checkpoint summary.",
            None,
        )
        assert len(factory.with_resume_calls) == 1
        assert factory.with_resume_calls[0][1] == "test-session-123"

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_extracts_multiline_checkpoint(
        self, default_config: ContextPressureConfig
    ) -> None:
        """Multiline checkpoint content is preserved."""
        checkpoint_msg = AssistantMessage(
            content=[TextBlock(text="<checkpoint>Line 1\nLine 2\nLine 3</checkpoint>")],
            model="test",
        )
        client = FakeSDKClient(messages=[checkpoint_msg])
        factory = FakeSDKClientFactory(client)

        handler = ContextPressureHandler(default_config, factory)

        result = await handler.fetch_checkpoint(
            session_id="sess-1",
            issue_id="issue-1",
            options={},
        )

        assert result.checkpoint == "Line 1\nLine 2\nLine 3"
        assert result.timed_out is False

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_returns_empty_when_no_checkpoint_prompt(
        self, empty_prompt_config: ContextPressureConfig
    ) -> None:
        """Returns empty checkpoint when no checkpoint_request_prompt configured."""
        factory = FakeSDKClientFactory()

        handler = ContextPressureHandler(empty_prompt_config, factory)

        result = await handler.fetch_checkpoint(
            session_id="sess-1",
            issue_id="issue-1",
            options={},
        )

        assert result.checkpoint == ""
        assert result.timed_out is False
        # Should not have created any client
        assert len(factory.create_calls) == 0

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_returns_empty_on_query_error(
        self, default_config: ContextPressureConfig
    ) -> None:
        """Returns empty checkpoint when SDK query fails."""
        client = FakeSDKClient(query_error=RuntimeError("Connection failed"))
        factory = FakeSDKClientFactory(client)

        handler = ContextPressureHandler(default_config, factory)

        result = await handler.fetch_checkpoint(
            session_id="sess-1",
            issue_id="issue-1",
            options={},
        )

        assert result.checkpoint == ""
        assert result.timed_out is False

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_timeout_returns_empty_checkpoint(
        self, default_config: ContextPressureConfig
    ) -> None:
        """Timeout during fetch returns empty checkpoint with timed_out=True."""
        # Client delays 2 seconds, but timeout is 0.1 seconds
        client = DelayedSDKClient(delay_seconds=2.0)
        factory = FakeSDKClientFactory(client)

        handler = ContextPressureHandler(default_config, factory)

        result = await handler.fetch_checkpoint(
            session_id="sess-1",
            issue_id="issue-1",
            options={},
            timeout_seconds=0.1,
        )

        assert result.checkpoint == ""
        assert result.timed_out is True

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_uses_fallback_when_no_tags(
        self, default_config: ContextPressureConfig
    ) -> None:
        """When no <checkpoint> tags, uses full response as fallback."""
        checkpoint_msg = AssistantMessage(
            content=[TextBlock(text="Current progress: completed steps 1-3")],
            model="test",
        )
        client = FakeSDKClient(messages=[checkpoint_msg])
        factory = FakeSDKClientFactory(client)

        handler = ContextPressureHandler(default_config, factory)

        result = await handler.fetch_checkpoint(
            session_id="sess-1",
            issue_id="issue-1",
            options={},
        )

        assert result.checkpoint == "Current progress: completed steps 1-3"
        assert result.timed_out is False


class TestContinuationPrompt:
    """Tests for build_continuation_prompt method."""

    @pytest.mark.unit
    def test_builds_prompt_with_template(
        self, default_config: ContextPressureConfig
    ) -> None:
        """Continuation prompt uses template with checkpoint."""
        factory = FakeSDKClientFactory()
        handler = ContextPressureHandler(default_config, factory)

        result = handler.build_continuation_prompt("step 3 of 5")

        assert result == "Continue from checkpoint:\n\nstep 3 of 5"

    @pytest.mark.unit
    def test_builds_fallback_without_template(self) -> None:
        """Continuation prompt falls back when no template configured."""
        config = ContextPressureConfig(
            checkpoint_request_prompt="Get checkpoint",
            continuation_template="",  # Empty template
        )
        factory = FakeSDKClientFactory()
        handler = ContextPressureHandler(config, factory)

        result = handler.build_continuation_prompt("current state")

        assert result == "Continue from checkpoint:\n\ncurrent state"

    @pytest.mark.unit
    def test_handles_curly_braces_in_checkpoint(
        self, default_config: ContextPressureConfig
    ) -> None:
        """Checkpoint with curly braces (JSON/code) doesn't break formatting."""
        factory = FakeSDKClientFactory()
        handler = ContextPressureHandler(default_config, factory)

        checkpoint = '{"step": 3, "data": {"key": "value"}}'
        result = handler.build_continuation_prompt(checkpoint)

        assert '{"step": 3, "data": {"key": "value"}}' in result


class TestHandlePressureError:
    """Tests for handle_pressure_error method."""

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_handles_pressure_error_end_to_end(
        self, default_config: ContextPressureConfig
    ) -> None:
        """handle_pressure_error fetches checkpoint and builds continuation."""
        checkpoint_msg = AssistantMessage(
            content=[TextBlock(text="<checkpoint>step 3</checkpoint>")],
            model="test",
        )
        client = FakeSDKClient(messages=[checkpoint_msg])
        factory = FakeSDKClientFactory(client)

        handler = ContextPressureHandler(default_config, factory)

        error = ContextPressureError(
            session_id="session-abc",
            input_tokens=180_000,
            output_tokens=10_000,
            cache_read_tokens=0,
            pressure_ratio=0.95,
        )

        continuation_prompt, new_count = await handler.handle_pressure_error(
            error=error,
            issue_id="issue-1",
            options={},
            continuation_count=0,
            remaining_time=30.0,
        )

        assert "step 3" in continuation_prompt
        assert new_count == 1

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_increments_continuation_count(
        self, default_config: ContextPressureConfig
    ) -> None:
        """Continuation count is incremented from current value."""
        client = FakeSDKClient(
            messages=[
                AssistantMessage(
                    content=[TextBlock(text="<checkpoint>x</checkpoint>")],
                    model="test",
                )
            ]
        )
        factory = FakeSDKClientFactory(client)
        handler = ContextPressureHandler(default_config, factory)

        error = ContextPressureError(
            session_id="s1",
            input_tokens=180_000,
            output_tokens=10_000,
            cache_read_tokens=0,
            pressure_ratio=0.95,
        )

        _, count = await handler.handle_pressure_error(
            error=error,
            issue_id="issue-1",
            options={},
            continuation_count=5,
            remaining_time=30.0,
        )

        assert count == 6

    @pytest.mark.asyncio
    @pytest.mark.unit
    async def test_respects_remaining_time_for_timeout(
        self, default_config: ContextPressureConfig
    ) -> None:
        """Timeout is bounded by remaining_time when it's less than config timeout."""
        # Delayed client would timeout with a 0.1s remaining time
        client = DelayedSDKClient(delay_seconds=2.0)
        factory = FakeSDKClientFactory(client)
        handler = ContextPressureHandler(default_config, factory)

        error = ContextPressureError(
            session_id="s1",
            input_tokens=180_000,
            output_tokens=10_000,
            cache_read_tokens=0,
            pressure_ratio=0.95,
        )

        continuation_prompt, _ = await handler.handle_pressure_error(
            error=error,
            issue_id="issue-1",
            options={},
            continuation_count=0,
            remaining_time=0.1,  # Very short remaining time
        )

        # Should have used fallback due to timeout
        assert "Continue from checkpoint" in continuation_prompt

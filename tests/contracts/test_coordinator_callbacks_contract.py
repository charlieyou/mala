"""Contract tests for coordinator callback protocol implementations.

Ensures FakeSpawnCallback, FakeFinalizeCallback, and FakeAbortCallback
implement the coordinator callback protocols correctly.
"""

import asyncio

import pytest

from src.pipeline.issue_execution_coordinator import (
    AbortCallback,
    AbortResult,
    FinalizeCallback,
    SpawnCallback,
)
from tests.contracts import get_protocol_members
from tests.fakes.coordinator_callbacks import (
    FakeAbortCallback,
    FakeFinalizeCallback,
    FakeSpawnCallback,
)


@pytest.mark.unit
def test_fake_spawn_callback_implements_protocol() -> None:
    """FakeSpawnCallback must implement SpawnCallback protocol."""
    protocol_methods = get_protocol_members(SpawnCallback)

    # SpawnCallback requires __call__, check it exists
    assert "__call__" in protocol_methods
    assert hasattr(FakeSpawnCallback, "__call__")


@pytest.mark.unit
def test_fake_finalize_callback_implements_protocol() -> None:
    """FakeFinalizeCallback must implement FinalizeCallback protocol."""
    protocol_methods = get_protocol_members(FinalizeCallback)

    assert "__call__" in protocol_methods
    assert hasattr(FakeFinalizeCallback, "__call__")


@pytest.mark.unit
def test_fake_abort_callback_implements_protocol() -> None:
    """FakeAbortCallback must implement AbortCallback protocol."""
    protocol_methods = get_protocol_members(AbortCallback)

    assert "__call__" in protocol_methods
    assert hasattr(FakeAbortCallback, "__call__")


class TestFakeSpawnCallbackBehavior:
    """Behavioral tests for FakeSpawnCallback."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_returns_task_by_default(self) -> None:
        """Default spawn returns an asyncio.Task."""
        callback = FakeSpawnCallback()
        result = await callback("issue-1")
        assert isinstance(result, asyncio.Task)
        await result  # Clean up task

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_records_spawned_issues(self) -> None:
        """Spawned issue IDs are recorded."""
        callback = FakeSpawnCallback()
        await callback("issue-1")
        await callback("issue-2")
        assert callback.spawned_issues == ["issue-1", "issue-2"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_fail_issues_returns_none(self) -> None:
        """Issues in fail_issues set return None."""
        callback = FakeSpawnCallback(fail_issues={"issue-bad"})
        result = await callback("issue-bad")
        assert result is None
        assert "issue-bad" in callback.spawned_issues


class TestFakeFinalizeCallbackBehavior:
    """Behavioral tests for FakeFinalizeCallback."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_records_finalizations(self) -> None:
        """Finalized issues are recorded."""
        callback = FakeFinalizeCallback()

        async def dummy() -> None:
            pass

        task = asyncio.create_task(dummy())
        await task

        await callback("issue-1", task)
        assert len(callback.finalized_issues) == 1
        assert callback.finalized_issues[0][0] == "issue-1"


class TestFakeAbortCallbackBehavior:
    """Behavioral tests for FakeAbortCallback."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_records_abort_calls(self) -> None:
        """Abort calls are recorded with kwargs."""
        callback = FakeAbortCallback()
        await callback(is_interrupt=True)
        await callback(is_interrupt=False)
        assert len(callback.abort_calls) == 2
        assert callback.abort_calls[0] == {"is_interrupt": True}
        assert callback.abort_calls[1] == {"is_interrupt": False}

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_returns_zero_without_coordinator(self) -> None:
        """Returns AbortResult with aborted_count=0 when no coordinator is set."""
        callback = FakeAbortCallback()
        result = await callback()
        assert result == AbortResult(aborted_count=0)

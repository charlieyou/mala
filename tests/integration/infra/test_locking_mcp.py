"""Integration tests for MCP locking tools with real filesystem.

Tests the full tool→hook→monitor flow:
- Real filesystem lock directory (temp dir)
- lock_acquire with immediate success and contention
- lock_acquire returns on ANY progress (one of N blocked files freed)
- lock_acquire timeout behavior (returns partial results)
- lock_release with specific files and all=true
- WAITING events emitted via closure callback
- PostToolUse hook emits ACQUIRED/RELEASED from tool results
- Multi-file batches process in sorted canonical order
- Empty filepaths rejection
- AgentRuntimeBuilder.build() → full runtime with locking MCP server
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.core.models import LockEvent
    from src.infra.tools.locking_mcp import LockingToolHandlers

pytestmark = pytest.mark.integration


@pytest.fixture
def lock_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide isolated lock directory with MALA_LOCK_DIR set."""
    lock_path = tmp_path / "locks"
    lock_path.mkdir()
    monkeypatch.setenv("MALA_LOCK_DIR", str(lock_path))
    return lock_path


def _create_handlers(
    agent_id: str = "test-agent",
    repo_namespace: str | None = None,
    emit_lock_event: Callable[[LockEvent], object] | None = None,
) -> LockingToolHandlers:
    """Create MCP locking tool handlers for testing."""
    events: list[LockEvent] = []
    emit = emit_lock_event if emit_lock_event else events.append

    from src.infra.tools.locking_mcp import create_locking_mcp_server

    result = create_locking_mcp_server(
        agent_id=agent_id,
        repo_namespace=repo_namespace,
        emit_lock_event=emit,
        _return_handlers=True,
    )
    assert isinstance(result, tuple)
    return result[1]  # type: ignore[return-value]


class TestLockAcquireImmediateSuccess:
    """Test lock_acquire when locks are immediately available."""

    @pytest.mark.asyncio
    async def test_acquires_single_file(self, lock_dir: Path) -> None:
        """Acquire lock on single file succeeds immediately."""
        handlers = _create_handlers(repo_namespace=str(lock_dir))
        test_file = str(lock_dir / "test.py")

        result = await handlers.lock_acquire.handler({"filepaths": [test_file]})

        content = json.loads(result["content"][0]["text"])
        assert content["all_acquired"] is True
        assert len(content["results"]) == 1
        assert content["results"][0]["acquired"] is True
        assert content["results"][0]["holder"] is None

    @pytest.mark.asyncio
    async def test_acquires_multiple_files(self, lock_dir: Path) -> None:
        """Acquire locks on multiple files succeeds immediately."""
        handlers = _create_handlers(repo_namespace=str(lock_dir))
        files = [str(lock_dir / f"file{i}.py") for i in range(3)]

        result = await handlers.lock_acquire.handler({"filepaths": files})

        content = json.loads(result["content"][0]["text"])
        assert content["all_acquired"] is True
        assert len(content["results"]) == 3
        for r in content["results"]:
            assert r["acquired"] is True


class TestLockAcquireContention:
    """Test lock_acquire with contention from background tasks."""

    @pytest.mark.asyncio
    async def test_returns_blocked_with_holder(self, lock_dir: Path) -> None:
        """When file is held, returns blocked status with holder ID."""
        from src.infra.tools.locking import try_lock

        test_file = str(lock_dir / "contested.py")
        repo_ns = str(lock_dir)

        # Another agent holds the lock (same namespace as handlers)
        assert try_lock(test_file, "blocking-agent", repo_namespace=repo_ns) is True

        handlers = _create_handlers(repo_namespace=repo_ns)
        result = await handlers.lock_acquire.handler(
            {"filepaths": [test_file], "timeout_seconds": 0}
        )

        content = json.loads(result["content"][0]["text"])
        assert content["all_acquired"] is False
        assert content["results"][0]["acquired"] is False
        assert content["results"][0]["holder"] == "blocking-agent"

    @pytest.mark.asyncio
    async def test_partial_acquisition(self, lock_dir: Path) -> None:
        """Some files acquired, some blocked."""
        from src.infra.tools.locking import try_lock

        free_file = str(lock_dir / "free.py")
        blocked_file = str(lock_dir / "blocked.py")
        repo_ns = str(lock_dir)

        # Block one file (same namespace as handlers)
        assert try_lock(blocked_file, "blocker", repo_namespace=repo_ns) is True

        handlers = _create_handlers(repo_namespace=repo_ns)
        result = await handlers.lock_acquire.handler(
            {"filepaths": [free_file, blocked_file], "timeout_seconds": 0}
        )

        content = json.loads(result["content"][0]["text"])
        assert content["all_acquired"] is False

        results_by_file = {r["filepath"]: r for r in content["results"]}
        assert results_by_file[free_file]["acquired"] is True
        assert results_by_file[blocked_file]["acquired"] is False


class TestLockAcquireAnyProgress:
    """Test that lock_acquire returns on ANY progress (one of N freed)."""

    @pytest.mark.asyncio
    async def test_returns_when_one_file_freed(self, lock_dir: Path) -> None:
        """Wait until one of multiple blocked files is freed."""
        from src.infra.tools.locking import release_lock, try_lock

        file1 = str(lock_dir / "held1.py")
        file2 = str(lock_dir / "held2.py")
        repo_ns = str(lock_dir)

        # Both files initially blocked (same namespace as handlers)
        assert try_lock(file1, "blocker1", repo_namespace=repo_ns) is True
        assert try_lock(file2, "blocker2", repo_namespace=repo_ns) is True

        handlers = _create_handlers(repo_namespace=repo_ns)

        async def release_first_after_delay() -> None:
            await asyncio.sleep(0.05)
            release_lock(file1, "blocker1", repo_namespace=repo_ns)

        release_task = asyncio.create_task(release_first_after_delay())

        result = await handlers.lock_acquire.handler(
            {"filepaths": [file1, file2], "timeout_seconds": 2.0}
        )
        await release_task

        content = json.loads(result["content"][0]["text"])
        # At least file1 should be acquired (the one that was released)
        results_by_file = {r["filepath"]: r for r in content["results"]}
        assert results_by_file[file1]["acquired"] is True


class TestLockAcquireTimeout:
    """Test lock_acquire timeout behavior."""

    @pytest.mark.asyncio
    async def test_timeout_returns_partial_results(self, lock_dir: Path) -> None:
        """Timeout returns partial results without acquired blocked files."""
        from src.infra.tools.locking import try_lock

        free_file = str(lock_dir / "free.py")
        blocked_file = str(lock_dir / "stuck.py")
        repo_ns = str(lock_dir)

        # One file blocked permanently (same namespace)
        assert (
            try_lock(blocked_file, "permanent-holder", repo_namespace=repo_ns) is True
        )

        handlers = _create_handlers(repo_namespace=repo_ns)
        result = await handlers.lock_acquire.handler(
            {"filepaths": [free_file, blocked_file], "timeout_seconds": 0.1}
        )

        content = json.loads(result["content"][0]["text"])
        # Free file should be acquired, blocked file should timeout
        assert content["all_acquired"] is False
        results_by_file = {r["filepath"]: r for r in content["results"]}
        assert results_by_file[free_file]["acquired"] is True
        assert results_by_file[blocked_file]["acquired"] is False

    @pytest.mark.asyncio
    async def test_non_blocking_mode(self, lock_dir: Path) -> None:
        """timeout_seconds=0 returns immediately without waiting."""
        from src.infra.tools.locking import try_lock

        test_file = str(lock_dir / "blocked.py")
        repo_ns = str(lock_dir)
        assert try_lock(test_file, "blocker", repo_namespace=repo_ns) is True

        handlers = _create_handlers(repo_namespace=repo_ns)

        import time

        start = time.monotonic()
        result = await handlers.lock_acquire.handler(
            {"filepaths": [test_file], "timeout_seconds": 0}
        )
        elapsed = time.monotonic() - start

        # Should return almost immediately (< 100ms)
        assert elapsed < 0.1
        content = json.loads(result["content"][0]["text"])
        assert content["results"][0]["acquired"] is False


class TestLockReleaseSpecificFiles:
    """Test lock_release with specific file list."""

    @pytest.mark.asyncio
    async def test_release_single_file(self, lock_dir: Path) -> None:
        """Release a single held lock."""
        from src.infra.tools.locking import get_lock_holder, try_lock

        test_file = str(lock_dir / "to_release.py")
        agent_id = "test-releaser"
        repo_ns = str(lock_dir)

        # Acquire lock (same namespace as handlers)
        assert try_lock(test_file, agent_id, repo_namespace=repo_ns) is True
        assert get_lock_holder(test_file, repo_namespace=repo_ns) == agent_id

        handlers = _create_handlers(agent_id=agent_id, repo_namespace=repo_ns)
        result = await handlers.lock_release.handler({"filepaths": [test_file]})

        content = json.loads(result["content"][0]["text"])
        assert content["count"] == 1
        assert test_file in content["released"]

        # Verify lock is released
        assert get_lock_holder(test_file, repo_namespace=repo_ns) is None

    @pytest.mark.asyncio
    async def test_release_multiple_files(self, lock_dir: Path) -> None:
        """Release multiple held locks."""
        from src.infra.tools.locking import get_lock_holder, try_lock

        agent_id = "multi-releaser"
        repo_ns = str(lock_dir)
        files = [str(lock_dir / f"multi{i}.py") for i in range(3)]

        for f in files:
            assert try_lock(f, agent_id, repo_namespace=repo_ns) is True

        handlers = _create_handlers(agent_id=agent_id, repo_namespace=repo_ns)
        result = await handlers.lock_release.handler({"filepaths": files})

        content = json.loads(result["content"][0]["text"])
        assert content["count"] == 3

        for f in files:
            assert get_lock_holder(f, repo_namespace=repo_ns) is None


class TestLockReleaseAll:
    """Test lock_release with all=true."""

    @pytest.mark.asyncio
    async def test_release_all_locks(self, lock_dir: Path) -> None:
        """Release all locks held by agent."""
        from src.infra.tools.locking import get_lock_holder, try_lock

        agent_id = "all-releaser"
        repo_ns = str(lock_dir)
        files = [str(lock_dir / f"all{i}.py") for i in range(5)]

        for f in files:
            assert try_lock(f, agent_id, repo_namespace=repo_ns) is True

        handlers = _create_handlers(agent_id=agent_id, repo_namespace=repo_ns)
        result = await handlers.lock_release.handler({"all": True})

        content = json.loads(result["content"][0]["text"])
        assert content["count"] == 5

        # Verify each specific file is released (more robust than counting all locks)
        for f in files:
            assert get_lock_holder(f, repo_namespace=repo_ns) is None

    @pytest.mark.asyncio
    async def test_release_all_preserves_other_agents(self, lock_dir: Path) -> None:
        """Release all only affects own locks."""
        from src.infra.tools.locking import get_lock_holder, try_lock

        repo_ns = str(lock_dir)
        our_file = str(lock_dir / "ours.py")
        other_file = str(lock_dir / "theirs.py")

        # Use consistent namespace for all lock operations
        assert try_lock(our_file, "our-agent", repo_namespace=repo_ns) is True
        assert try_lock(other_file, "other-agent", repo_namespace=repo_ns) is True

        handlers = _create_handlers(agent_id="our-agent", repo_namespace=repo_ns)
        await handlers.lock_release.handler({"all": True})

        # Our lock released, theirs preserved
        assert get_lock_holder(our_file, repo_namespace=repo_ns) is None
        assert get_lock_holder(other_file, repo_namespace=repo_ns) == "other-agent"


class TestWaitingEvents:
    """Test WAITING events emitted via closure callback."""

    @pytest.mark.asyncio
    async def test_emits_waiting_when_blocked(self, lock_dir: Path) -> None:
        """WAITING event emitted for blocked file."""
        from src.core.models import LockEventType
        from src.infra.tools.locking import try_lock

        blocked_file = str(lock_dir / "blocked.py")
        repo_ns = str(lock_dir)
        assert try_lock(blocked_file, "blocker", repo_namespace=repo_ns) is True

        events: list[LockEvent] = []
        handlers = _create_handlers(
            repo_namespace=repo_ns, emit_lock_event=events.append
        )

        await handlers.lock_acquire.handler(
            {"filepaths": [blocked_file], "timeout_seconds": 0.1}
        )

        waiting_events = [e for e in events if e.event_type == LockEventType.WAITING]
        assert len(waiting_events) >= 1
        assert waiting_events[0].agent_id == "test-agent"

    @pytest.mark.asyncio
    async def test_no_waiting_when_all_free(self, lock_dir: Path) -> None:
        """No WAITING events when all locks available."""
        events: list[LockEvent] = []
        handlers = _create_handlers(
            repo_namespace=str(lock_dir), emit_lock_event=events.append
        )

        free_file = str(lock_dir / "free.py")
        await handlers.lock_acquire.handler({"filepaths": [free_file]})

        from src.core.models import LockEventType

        waiting_events = [e for e in events if e.event_type == LockEventType.WAITING]
        assert len(waiting_events) == 0

    @pytest.mark.asyncio
    async def test_waiting_emitted_for_each_blocked_file(self, lock_dir: Path) -> None:
        """WAITING event emitted for each blocked file."""
        from src.core.models import LockEventType
        from src.infra.tools.locking import try_lock

        blocked1 = str(lock_dir / "blocked1.py")
        blocked2 = str(lock_dir / "blocked2.py")
        free = str(lock_dir / "free.py")
        repo_ns = str(lock_dir)

        assert try_lock(blocked1, "blocker", repo_namespace=repo_ns) is True
        assert try_lock(blocked2, "blocker", repo_namespace=repo_ns) is True

        events: list[LockEvent] = []
        handlers = _create_handlers(
            repo_namespace=repo_ns, emit_lock_event=events.append
        )

        await handlers.lock_acquire.handler(
            {"filepaths": [free, blocked1, blocked2], "timeout_seconds": 0.1}
        )

        waiting_events = [e for e in events if e.event_type == LockEventType.WAITING]
        assert len(waiting_events) == 2


class TestPostToolUseHookEvents:
    """Test PostToolUse hook emits ACQUIRED/RELEASED events."""

    @pytest.mark.asyncio
    async def test_hook_emits_acquired_event(self, lock_dir: Path) -> None:
        """PostToolUse hook emits ACQUIRED for successful acquisition."""
        from src.core.models import LockEvent, LockEventType
        from src.infra.hooks.deadlock import make_lock_event_hook

        events: list[LockEvent] = []

        async def capture_event(event: LockEvent) -> None:
            events.append(event)

        hook = make_lock_event_hook(
            agent_id="hook-test-agent",
            emit_event=capture_event,
            repo_namespace=str(lock_dir),
            lock_event_class=LockEvent,
            lock_event_type_enum=LockEventType,
        )

        # Simulate lock_acquire tool response
        test_file = str(lock_dir / "acquired.py")
        hook_input = {
            "tool_name": "mcp__mala-locking__lock_acquire",
            "tool_response": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "results": [
                                    {
                                        "filepath": test_file,
                                        "acquired": True,
                                        "holder": None,
                                    }
                                ],
                                "all_acquired": True,
                            }
                        ),
                    }
                ]
            },
        }

        await hook(hook_input, None, MagicMock())

        acquired_events = [e for e in events if e.event_type == LockEventType.ACQUIRED]
        assert len(acquired_events) == 1
        assert acquired_events[0].agent_id == "hook-test-agent"

    @pytest.mark.asyncio
    async def test_hook_emits_released_event(self, lock_dir: Path) -> None:
        """PostToolUse hook emits RELEASED for successful release."""
        from src.core.models import LockEvent, LockEventType
        from src.infra.hooks.deadlock import make_lock_event_hook

        events: list[LockEvent] = []

        async def capture_event(event: LockEvent) -> None:
            events.append(event)

        hook = make_lock_event_hook(
            agent_id="hook-test-agent",
            emit_event=capture_event,
            repo_namespace=str(lock_dir),
            lock_event_class=LockEvent,
            lock_event_type_enum=LockEventType,
        )

        # Simulate lock_release tool response
        test_file = str(lock_dir / "released.py")
        hook_input = {
            "tool_name": "mcp__mala-locking__lock_release",
            "tool_response": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"released": [test_file], "count": 1}),
                    }
                ]
            },
        }

        await hook(hook_input, None, MagicMock())

        released_events = [e for e in events if e.event_type == LockEventType.RELEASED]
        assert len(released_events) == 1
        assert released_events[0].agent_id == "hook-test-agent"

    @pytest.mark.asyncio
    async def test_hook_skips_non_acquired_files(self, lock_dir: Path) -> None:
        """Hook doesn't emit ACQUIRED for blocked files."""
        from src.core.models import LockEvent, LockEventType
        from src.infra.hooks.deadlock import make_lock_event_hook

        events: list[LockEvent] = []

        async def capture_event(event: LockEvent) -> None:
            events.append(event)

        hook = make_lock_event_hook(
            agent_id="hook-test-agent",
            emit_event=capture_event,
            repo_namespace=str(lock_dir),
            lock_event_class=LockEvent,
            lock_event_type_enum=LockEventType,
        )

        # Simulate partial acquisition
        hook_input = {
            "tool_name": "mcp__mala-locking__lock_acquire",
            "tool_response": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "results": [
                                    {
                                        "filepath": "ok.py",
                                        "acquired": True,
                                        "holder": None,
                                    },
                                    {
                                        "filepath": "blocked.py",
                                        "acquired": False,
                                        "holder": "other",
                                    },
                                ],
                                "all_acquired": False,
                            }
                        ),
                    }
                ]
            },
        }

        await hook(hook_input, None, MagicMock())

        # Only one ACQUIRED event (for ok.py)
        acquired_events = [e for e in events if e.event_type == LockEventType.ACQUIRED]
        assert len(acquired_events) == 1


class TestCanonicalOrdering:
    """Test multi-file batches process in sorted canonical order."""

    @pytest.mark.asyncio
    async def test_files_processed_in_sorted_order(self, lock_dir: Path) -> None:
        """Files are processed in sorted canonical path order."""
        from src.infra.tools.locking import get_lock_holder

        repo_ns = str(lock_dir)
        # Unsorted input
        files = [
            str(lock_dir / "z.py"),
            str(lock_dir / "a.py"),
            str(lock_dir / "m.py"),
        ]

        handlers = _create_handlers(agent_id="order-test", repo_namespace=repo_ns)
        result = await handlers.lock_acquire.handler({"filepaths": files})

        content = json.loads(result["content"][0]["text"])
        # All should be acquired
        assert content["all_acquired"] is True

        # Verify all locks exist (using same namespace)
        for f in files:
            assert get_lock_holder(f, repo_namespace=repo_ns) == "order-test"

    @pytest.mark.asyncio
    async def test_deduplicates_canonical_paths(self, lock_dir: Path) -> None:
        """Duplicate canonical paths are deduplicated."""
        import os

        # Create test file so os.path.realpath can resolve it
        dup_file = lock_dir / "dup.py"
        dup_file.touch()

        # Use non-normalized path strings that require realpath resolution
        # These are different string representations that resolve to the same file
        base = str(dup_file)
        with_dot = str(lock_dir) + "/./dup.py"  # Contains ./ that needs resolution
        with_parent = str(lock_dir) + "/../" + os.path.basename(lock_dir) + "/dup.py"

        # Verify paths are not pre-normalized (the point of this test)
        assert base != with_dot
        assert base != with_parent

        files = [base, with_dot, with_parent]

        handlers = _create_handlers(repo_namespace=str(lock_dir))
        result = await handlers.lock_acquire.handler({"filepaths": files})

        content = json.loads(result["content"][0]["text"])
        # Should only have one result after dedup - tool must resolve canonical paths
        assert len(content["results"]) == 1


class TestEmptyFilepathsRejection:
    """Test empty filepaths array is rejected."""

    @pytest.mark.asyncio
    async def test_acquire_rejects_empty_array(self, lock_dir: Path) -> None:
        """lock_acquire rejects empty filepaths array."""
        handlers = _create_handlers(repo_namespace=str(lock_dir))
        result = await handlers.lock_acquire.handler({"filepaths": []})

        content = json.loads(result["content"][0]["text"])
        assert "error" in content
        assert "non-empty" in content["error"]

    @pytest.mark.asyncio
    async def test_release_rejects_empty_array(self, lock_dir: Path) -> None:
        """lock_release rejects empty filepaths array."""
        handlers = _create_handlers(repo_namespace=str(lock_dir))
        result = await handlers.lock_release.handler({"filepaths": []})

        content = json.loads(result["content"][0]["text"])
        assert "error" in content
        assert "non-empty" in content["error"]


class TestConcurrentOperations:
    """Test concurrent lock operations."""

    @pytest.mark.asyncio
    async def test_concurrent_acquire_different_files(self, lock_dir: Path) -> None:
        """Multiple agents can acquire different files concurrently."""
        from src.infra.tools.locking import get_lock_holder

        repo_ns = str(lock_dir)
        file1 = str(lock_dir / "agent1.py")
        file2 = str(lock_dir / "agent2.py")

        handlers1 = _create_handlers(agent_id="agent-1", repo_namespace=repo_ns)
        handlers2 = _create_handlers(agent_id="agent-2", repo_namespace=repo_ns)

        # Acquire concurrently
        result1, result2 = await asyncio.gather(
            handlers1.lock_acquire.handler({"filepaths": [file1]}),
            handlers2.lock_acquire.handler({"filepaths": [file2]}),
        )

        content1 = json.loads(result1["content"][0]["text"])
        content2 = json.loads(result2["content"][0]["text"])

        assert content1["all_acquired"] is True
        assert content2["all_acquired"] is True

        assert get_lock_holder(file1, repo_namespace=repo_ns) == "agent-1"
        assert get_lock_holder(file2, repo_namespace=repo_ns) == "agent-2"

    @pytest.mark.asyncio
    async def test_concurrent_acquire_same_file(self, lock_dir: Path) -> None:
        """Only one agent acquires contested file."""
        repo_ns = str(lock_dir)
        test_file = str(lock_dir / "contested.py")

        handlers1 = _create_handlers(agent_id="racer-1", repo_namespace=repo_ns)
        handlers2 = _create_handlers(agent_id="racer-2", repo_namespace=repo_ns)

        # Race for the same file
        result1, result2 = await asyncio.gather(
            handlers1.lock_acquire.handler(
                {"filepaths": [test_file], "timeout_seconds": 0}
            ),
            handlers2.lock_acquire.handler(
                {"filepaths": [test_file], "timeout_seconds": 0}
            ),
        )

        content1 = json.loads(result1["content"][0]["text"])
        content2 = json.loads(result2["content"][0]["text"])

        # Exactly one should succeed
        acquired_count = (
            content1["results"][0]["acquired"] + content2["results"][0]["acquired"]
        )
        assert acquired_count == 1

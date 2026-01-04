"""Unit tests for MCP tool handlers in locking_mcp.py.

Tests tool input validation, return format, wiring to locking.py, and
WAITING event emission. Uses mocks for fast execution (<1s each).
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, patch

import pytest

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from claude_agent_sdk.types import McpSdkServerConfig

    from src.core.models import LockEvent
    from src.infra.tools.locking_mcp import LockingToolHandlers

pytestmark = pytest.mark.unit


def _create_handlers(
    agent_id: str = "test-agent",
    repo_namespace: str | None = "/test/repo",
    emit_lock_event: Callable[[LockEvent], None | Awaitable[None]] | None = None,
) -> LockingToolHandlers:
    """Create MCP server and return the handlers object.

    Uses the _return_handlers=True parameter to get direct access to handlers.
    """
    events: list[Any] = []
    emit = emit_lock_event if emit_lock_event else events.append

    from src.infra.tools.locking_mcp import create_locking_mcp_server

    result = create_locking_mcp_server(
        agent_id=agent_id,
        repo_namespace=repo_namespace,
        emit_lock_event=emit,
        _return_handlers=True,
    )
    # _return_handlers=True returns tuple (server, handlers)
    assert isinstance(result, tuple)
    handlers: LockingToolHandlers = result[1]  # type: ignore[assignment]

    return handlers


class TestLockAcquireInputValidation:
    """Test lock_acquire JSON schema validation."""

    @pytest.mark.asyncio
    async def test_empty_filepaths_rejected(self) -> None:
        """Empty filepaths array should return error."""
        with patch("src.infra.tools.locking_mcp.try_lock", return_value=True):
            handlers = _create_handlers()

        result = await handlers.lock_acquire.handler({"filepaths": []})

        assert "content" in result
        content = json.loads(result["content"][0]["text"])
        assert "error" in content
        assert "non-empty" in content["error"]

    @pytest.mark.asyncio
    async def test_missing_filepaths_rejected(self) -> None:
        """Missing filepaths should return error."""
        with patch("src.infra.tools.locking_mcp.try_lock", return_value=True):
            handlers = _create_handlers()

        result = await handlers.lock_acquire.handler({})

        assert "content" in result
        content = json.loads(result["content"][0]["text"])
        assert "error" in content


class TestLockReleaseInputValidation:
    """Test lock_release JSON schema validation."""

    @pytest.mark.asyncio
    async def test_mutually_exclusive_params_rejected(self) -> None:
        """Cannot specify both filepaths and all=true."""
        with patch("src.infra.tools.locking_mcp.try_lock", return_value=True):
            handlers = _create_handlers()

        result = await handlers.lock_release.handler(
            {"filepaths": ["file.py"], "all": True}
        )

        content = json.loads(result["content"][0]["text"])
        assert "error" in content
        assert "both" in content["error"].lower()

    @pytest.mark.asyncio
    async def test_neither_param_rejected(self) -> None:
        """Must specify either filepaths or all=true."""
        with patch("src.infra.tools.locking_mcp.try_lock", return_value=True):
            handlers = _create_handlers()

        result = await handlers.lock_release.handler({})

        content = json.loads(result["content"][0]["text"])
        assert "error" in content
        assert "either" in content["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_filepaths_rejected(self) -> None:
        """Empty filepaths array should return error."""
        with patch("src.infra.tools.locking_mcp.try_lock", return_value=True):
            handlers = _create_handlers()

        result = await handlers.lock_release.handler({"filepaths": []})

        content = json.loads(result["content"][0]["text"])
        assert "error" in content
        assert "non-empty" in content["error"]


class TestLockAcquireReturnFormat:
    """Test lock_acquire return structure matches spec."""

    @pytest.mark.asyncio
    async def test_return_format_all_acquired(self) -> None:
        """Result contains per-file results and all_acquired flag."""
        with patch("src.infra.tools.locking_mcp.try_lock", return_value=True):
            handlers = _create_handlers()
            result = await handlers.lock_acquire.handler(
                {"filepaths": ["a.py", "b.py"]}
            )

        content = json.loads(result["content"][0]["text"])

        assert "results" in content
        assert "all_acquired" in content
        assert content["all_acquired"] is True
        assert len(content["results"]) == 2

        for r in content["results"]:
            assert "filepath" in r
            assert "acquired" in r
            assert "holder" in r
            assert "canonical" not in r  # Internal field stripped

    @pytest.mark.asyncio
    async def test_return_format_partial_acquired(self) -> None:
        """Result shows partial acquisition with holder info."""

        def mock_try_lock(filepath: str, agent_id: str, namespace: str | None) -> bool:
            return "blocked" not in filepath

        def mock_get_holder(filepath: str, namespace: str | None) -> str | None:
            if "blocked" in filepath:
                return "other-agent"
            return None

        with (
            patch("src.infra.tools.locking_mcp.try_lock", side_effect=mock_try_lock),
            patch(
                "src.infra.tools.locking_mcp.get_lock_holder",
                side_effect=mock_get_holder,
            ),
            patch(
                "src.infra.tools.locking_mcp.wait_for_lock_async",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            handlers = _create_handlers()

            result = await handlers.lock_acquire.handler(
                {"filepaths": ["ok.py", "blocked.py"], "timeout_seconds": 0}
            )

        content = json.loads(result["content"][0]["text"])

        assert content["all_acquired"] is False
        results_by_file = {r["filepath"]: r for r in content["results"]}

        assert results_by_file["ok.py"]["acquired"] is True
        assert results_by_file["ok.py"]["holder"] is None

        assert results_by_file["blocked.py"]["acquired"] is False
        assert results_by_file["blocked.py"]["holder"] == "other-agent"


class TestLockAcquireWiring:
    """Test lock_acquire wiring to locking.py functions."""

    @pytest.mark.asyncio
    async def test_calls_try_lock_for_each_file_sorted(self) -> None:
        """try_lock called for each file in sorted canonical order."""
        try_lock_calls: list[str] = []

        def mock_try_lock(filepath: str, agent_id: str, namespace: str | None) -> bool:
            try_lock_calls.append(filepath)
            return True

        with (
            patch("src.infra.tools.locking_mcp.try_lock", side_effect=mock_try_lock),
            patch(
                "src.infra.tools.locking_mcp.canonicalize_path",
                side_effect=lambda p, ns: f"/{p}",
            ),
        ):
            handlers = _create_handlers()

            await handlers.lock_acquire.handler({"filepaths": ["z.py", "a.py", "m.py"]})

        # Should be sorted by canonical path
        assert try_lock_calls == ["/a.py", "/m.py", "/z.py"]

    @pytest.mark.asyncio
    async def test_deduplicates_by_canonical_path(self) -> None:
        """Duplicate canonical paths are deduplicated."""
        try_lock_calls: list[str] = []

        def mock_try_lock(filepath: str, agent_id: str, namespace: str | None) -> bool:
            try_lock_calls.append(filepath)
            return True

        # Both paths canonicalize to the same path
        def mock_canonicalize(path: str, namespace: str | None) -> str:
            return "/same/path.py"

        with (
            patch("src.infra.tools.locking_mcp.try_lock", side_effect=mock_try_lock),
            patch(
                "src.infra.tools.locking_mcp.canonicalize_path",
                side_effect=mock_canonicalize,
            ),
        ):
            handlers = _create_handlers()

            await handlers.lock_acquire.handler(
                {"filepaths": ["a.py", "./a.py", "b/../a.py"]}
            )

        # Only one try_lock call despite 3 input paths
        assert len(try_lock_calls) == 1


class TestLockAcquireWaitingEvents:
    """Test WAITING event emission when blocked."""

    @pytest.mark.asyncio
    async def test_emits_waiting_for_blocked_files(self) -> None:
        """WAITING event emitted for each blocked file."""
        from src.core.models import LockEventType

        events: list[Any] = []

        def mock_try_lock(filepath: str, agent_id: str, namespace: str | None) -> bool:
            return "free" in filepath

        def mock_canonicalize(path: str, namespace: str | None) -> str:
            return f"/canonical/{path}"

        with (
            patch("src.infra.tools.locking_mcp.try_lock", side_effect=mock_try_lock),
            patch(
                "src.infra.tools.locking_mcp.get_lock_holder", return_value="blocker"
            ),
            patch(
                "src.infra.tools.locking_mcp.canonicalize_path",
                side_effect=mock_canonicalize,
            ),
            patch(
                "src.infra.tools.locking_mcp.wait_for_lock_async",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            handlers = _create_handlers(emit_lock_event=events.append)

            await handlers.lock_acquire.handler(
                {
                    "filepaths": ["free.py", "blocked1.py", "blocked2.py"],
                    "timeout_seconds": 0.1,
                }
            )

        # Should have WAITING events for the two blocked files
        waiting_events = [e for e in events if e.event_type == LockEventType.WAITING]
        assert len(waiting_events) == 2

        blocked_paths = {e.lock_path for e in waiting_events}
        assert "/canonical/blocked1.py" in blocked_paths
        assert "/canonical/blocked2.py" in blocked_paths

        # All events should have correct agent_id
        for e in waiting_events:
            assert e.agent_id == "test-agent"

    @pytest.mark.asyncio
    async def test_no_waiting_when_all_acquired(self) -> None:
        """No WAITING events when all locks acquired immediately."""
        events: list[Any] = []

        with patch("src.infra.tools.locking_mcp.try_lock", return_value=True):
            handlers = _create_handlers(emit_lock_event=events.append)

            await handlers.lock_acquire.handler({"filepaths": ["a.py", "b.py"]})

        assert len(events) == 0


class TestAsyncEmitLockEvent:
    """Test async emit_lock_event callback support."""

    @pytest.mark.asyncio
    async def test_async_emit_lock_event_awaited(self) -> None:
        """Async emit_lock_event callback is awaited, not dropped."""
        from src.core.models import LockEventType

        events: list[Any] = []
        emit_awaited = False

        async def async_emit(event: LockEvent) -> None:
            nonlocal emit_awaited
            emit_awaited = True
            events.append(event)

        def mock_try_lock(filepath: str, agent_id: str, namespace: str | None) -> bool:
            return False  # All blocked to trigger WAITING

        with (
            patch("src.infra.tools.locking_mcp.try_lock", side_effect=mock_try_lock),
            patch("src.infra.tools.locking_mcp.get_lock_holder", return_value="other"),
            patch(
                "src.infra.tools.locking_mcp.wait_for_lock_async",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            handlers = _create_handlers(emit_lock_event=async_emit)

            await handlers.lock_acquire.handler(
                {"filepaths": ["blocked.py"], "timeout_seconds": 0.1}
            )

        # Verify async callback was awaited
        assert emit_awaited, "async emit_lock_event was not awaited"
        assert len(events) == 1
        assert events[0].event_type == LockEventType.WAITING

    @pytest.mark.asyncio
    async def test_async_emit_error_logged_not_raised(self) -> None:
        """Errors in async emit_lock_event are logged, not raised."""

        async def failing_emit(event: LockEvent) -> None:
            raise RuntimeError("emit failed")

        def mock_try_lock(filepath: str, agent_id: str, namespace: str | None) -> bool:
            return False

        with (
            patch("src.infra.tools.locking_mcp.try_lock", side_effect=mock_try_lock),
            patch("src.infra.tools.locking_mcp.get_lock_holder", return_value="other"),
            patch(
                "src.infra.tools.locking_mcp.wait_for_lock_async",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            handlers = _create_handlers(emit_lock_event=failing_emit)

            # Should not raise even though emit callback raises
            result = await handlers.lock_acquire.handler(
                {"filepaths": ["file.py"], "timeout_seconds": 0.1}
            )

        # Should return a result (even if emit failed)
        assert "content" in result


class TestLockReleaseWiring:
    """Test lock_release wiring to locking.py functions."""

    @pytest.mark.asyncio
    async def test_release_all_calls_cleanup_agent_locks(self) -> None:
        """all=true calls cleanup_agent_locks and returns released paths."""
        cleanup_called: list[str] = []
        mock_released_paths = ["/path/to/file1.py", "/path/to/file2.py"]

        def mock_cleanup(agent_id: str) -> tuple[int, list[str]]:
            cleanup_called.append(agent_id)
            return 5, mock_released_paths

        with patch(
            "src.infra.tools.locking_mcp.cleanup_agent_locks", side_effect=mock_cleanup
        ):
            handlers = _create_handlers()

            result = await handlers.lock_release.handler({"all": True})

        assert cleanup_called == ["test-agent"]
        content = json.loads(result["content"][0]["text"])
        assert content["count"] == 5
        assert content["released"] == mock_released_paths

    @pytest.mark.asyncio
    async def test_release_specific_calls_release_lock(self) -> None:
        """Specific filepaths call release_lock for each."""
        release_calls: list[tuple[str, str]] = []

        def mock_release(filepath: str, agent_id: str, namespace: str | None) -> bool:
            release_calls.append((filepath, agent_id))
            return True

        with (
            patch("src.infra.tools.locking_mcp.release_lock", side_effect=mock_release),
            patch(
                "src.infra.tools.locking_mcp.canonicalize_path",
                side_effect=lambda p, ns: f"/{p}",
            ),
        ):
            handlers = _create_handlers()

            result = await handlers.lock_release.handler(
                {"filepaths": ["a.py", "b.py"]}
            )

        assert len(release_calls) == 2
        content = json.loads(result["content"][0]["text"])
        assert content["count"] == 2

    @pytest.mark.asyncio
    async def test_release_idempotent_behavior(self) -> None:
        """Release succeeds even when lock not held (idempotent)."""
        release_calls: list[str] = []

        def mock_release(filepath: str, agent_id: str, namespace: str | None) -> bool:
            release_calls.append(filepath)
            return False  # Lock was not held

        with (
            patch("src.infra.tools.locking_mcp.release_lock", side_effect=mock_release),
            patch(
                "src.infra.tools.locking_mcp.canonicalize_path",
                side_effect=lambda p, ns: f"/{p}",
            ),
        ):
            handlers = _create_handlers()

            result = await handlers.lock_release.handler({"filepaths": ["not_held.py"]})

        # Should succeed despite release_lock returning False
        content = json.loads(result["content"][0]["text"])
        assert "error" not in content
        assert content["count"] == 1


class TestLockAcquireAsyncWaiting:
    """Test async waiting behavior in lock_acquire."""

    @pytest.mark.asyncio
    async def test_waits_for_blocked_files(self) -> None:
        """Spawns wait tasks for blocked files."""
        wait_calls: list[str] = []

        async def mock_wait(
            filepath: str,
            agent_id: str,
            namespace: str | None,
            timeout_seconds: float,
            poll_interval_ms: int,
        ) -> bool:
            wait_calls.append(filepath)
            return True

        def mock_try_lock(filepath: str, agent_id: str, namespace: str | None) -> bool:
            return "blocked" not in filepath

        with (
            patch("src.infra.tools.locking_mcp.try_lock", side_effect=mock_try_lock),
            patch("src.infra.tools.locking_mcp.get_lock_holder", return_value="other"),
            patch(
                "src.infra.tools.locking_mcp.wait_for_lock_async", side_effect=mock_wait
            ),
            patch(
                "src.infra.tools.locking_mcp.canonicalize_path",
                side_effect=lambda p, ns: f"/{p}",
            ),
        ):
            handlers = _create_handlers()

            await handlers.lock_acquire.handler(
                {
                    "filepaths": ["ok.py", "blocked.py"],
                    "timeout_seconds": 5.0,
                }
            )

        assert "/blocked.py" in wait_calls

    @pytest.mark.asyncio
    async def test_non_blocking_mode_skips_waiting(self) -> None:
        """timeout_seconds=0 returns immediately without waiting."""
        wait_calls: list[str] = []

        with (
            patch("src.infra.tools.locking_mcp.try_lock", return_value=False),
            patch("src.infra.tools.locking_mcp.get_lock_holder", return_value="other"),
            patch(
                "src.infra.tools.locking_mcp.wait_for_lock_async",
                new_callable=AsyncMock,
                side_effect=lambda *a, **k: wait_calls.append("called") or True,
            ),
        ):
            handlers = _create_handlers()

            await handlers.lock_acquire.handler(
                {"filepaths": ["file.py"], "timeout_seconds": 0}
            )

        assert len(wait_calls) == 0

    @pytest.mark.asyncio
    async def test_cancels_pending_waits_on_first_success(self) -> None:
        """Cancels remaining wait tasks when first completes."""
        cancel_count = 0

        async def slow_wait(
            filepath: str,
            agent_id: str,
            namespace: str | None,
            timeout_seconds: float,
            poll_interval_ms: int,
        ) -> bool:
            nonlocal cancel_count
            try:
                await asyncio.sleep(10)  # Will be cancelled
                return False
            except asyncio.CancelledError:
                cancel_count += 1
                raise

        async def fast_wait(
            filepath: str,
            agent_id: str,
            namespace: str | None,
            timeout_seconds: float,
            poll_interval_ms: int,
        ) -> bool:
            await asyncio.sleep(0.01)
            return True

        call_index = 0

        async def mock_wait(
            filepath: str,
            agent_id: str,
            namespace: str | None,
            timeout_seconds: float,
            poll_interval_ms: int,
        ) -> bool:
            nonlocal call_index
            call_index += 1
            if call_index == 1:
                return await fast_wait(
                    filepath, agent_id, namespace, timeout_seconds, poll_interval_ms
                )
            return await slow_wait(
                filepath, agent_id, namespace, timeout_seconds, poll_interval_ms
            )

        with (
            patch("src.infra.tools.locking_mcp.try_lock", return_value=False),
            patch("src.infra.tools.locking_mcp.get_lock_holder", return_value="other"),
            patch(
                "src.infra.tools.locking_mcp.wait_for_lock_async", side_effect=mock_wait
            ),
            patch(
                "src.infra.tools.locking_mcp.canonicalize_path",
                side_effect=lambda p, ns: f"/{p}",
            ),
        ):
            handlers = _create_handlers()

            await handlers.lock_acquire.handler(
                {
                    "filepaths": ["a.py", "b.py"],
                    "timeout_seconds": 5.0,
                }
            )

        # The slow wait should have been cancelled
        assert cancel_count == 1


class TestMCPServerConfiguration:
    """Test MCP server configuration."""

    def test_server_has_correct_name(self) -> None:
        """Server has expected name."""
        from src.infra.tools.locking_mcp import create_locking_mcp_server

        with patch("src.infra.tools.locking_mcp.try_lock", return_value=True):
            result = create_locking_mcp_server("test", None, lambda e: None)

        # Without _return_handlers, returns just the server config
        server: McpSdkServerConfig = result  # type: ignore[assignment]
        assert server["name"] == "mala-locking"

    def test_server_has_instance(self) -> None:
        """Server exposes an MCP server instance."""
        from src.infra.tools.locking_mcp import create_locking_mcp_server

        with patch("src.infra.tools.locking_mcp.try_lock", return_value=True):
            result = create_locking_mcp_server("test", None, lambda e: None)

        server: McpSdkServerConfig = result  # type: ignore[assignment]
        assert "instance" in server
        assert server["type"] == "sdk"

    def test_handlers_have_both_tools(self) -> None:
        """Handlers object has lock_acquire and lock_release SdkMcpTool objects."""
        with patch("src.infra.tools.locking_mcp.try_lock", return_value=True):
            handlers = _create_handlers()

        assert hasattr(handlers, "lock_acquire")
        assert hasattr(handlers, "lock_release")
        # SdkMcpTool objects have .handler attribute that is callable
        assert callable(handlers.lock_acquire.handler)
        assert callable(handlers.lock_release.handler)

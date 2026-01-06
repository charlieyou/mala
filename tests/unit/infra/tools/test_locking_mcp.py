"""Unit tests for MCP tool handlers in locking_mcp.py.

Tests tool input validation, return format, wiring to LockingBackend, and
WAITING event emission. Uses FakeLockingBackend for fast, observable tests.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any

import pytest

from tests.fakes.locking_backend import FakeLockingBackend

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
    locking_backend: FakeLockingBackend | None = None,
) -> LockingToolHandlers:
    """Create MCP server and return the handlers object.

    Uses the _return_handlers=True parameter to get direct access to handlers.
    """
    events: list[Any] = []
    emit = emit_lock_event if emit_lock_event else events.append
    backend = locking_backend if locking_backend is not None else FakeLockingBackend()

    from src.infra.tools.locking_mcp import create_locking_mcp_server

    result = create_locking_mcp_server(
        agent_id=agent_id,
        repo_namespace=repo_namespace,
        emit_lock_event=emit,
        locking_backend=backend,
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
        handlers = _create_handlers()

        result = await handlers.lock_acquire.handler({"filepaths": []})

        assert "content" in result
        content = json.loads(result["content"][0]["text"])
        assert "error" in content
        assert "non-empty" in content["error"]

    @pytest.mark.asyncio
    async def test_missing_filepaths_rejected(self) -> None:
        """Missing filepaths should return error."""
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
        handlers = _create_handlers()

        result = await handlers.lock_release.handler({})

        content = json.loads(result["content"][0]["text"])
        assert "error" in content
        assert "either" in content["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_filepaths_rejected(self) -> None:
        """Empty filepaths array should return error."""
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
        handlers = _create_handlers()
        result = await handlers.lock_acquire.handler({"filepaths": ["a.py", "b.py"]})

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
        backend = FakeLockingBackend()
        backend.try_lock_results["blocked.py"] = False
        backend.holders["blocked.py"] = "other-agent"

        handlers = _create_handlers(locking_backend=backend)

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
    """Test lock_acquire wiring to LockingBackend."""

    @pytest.mark.asyncio
    async def test_calls_try_lock_for_each_file_sorted(self) -> None:
        """try_lock called for each file in sorted canonical order."""
        backend = FakeLockingBackend()
        backend.canonicalize_fn = lambda p, ns: f"/{p}"

        handlers = _create_handlers(locking_backend=backend)

        await handlers.lock_acquire.handler({"filepaths": ["z.py", "a.py", "m.py"]})

        # Should be sorted by canonical path
        try_lock_paths = [call[0] for call in backend.try_lock_calls]
        assert try_lock_paths == ["/a.py", "/m.py", "/z.py"]

    @pytest.mark.asyncio
    async def test_deduplicates_by_canonical_path(self) -> None:
        """Duplicate canonical paths are deduplicated."""
        backend = FakeLockingBackend()
        backend.canonicalize_fn = lambda p, ns: "/same/path.py"

        handlers = _create_handlers(locking_backend=backend)

        await handlers.lock_acquire.handler(
            {"filepaths": ["a.py", "./a.py", "b/../a.py"]}
        )

        # Only one try_lock call despite 3 input paths
        assert len(backend.try_lock_calls) == 1


class TestLockAcquireWaitingEvents:
    """Test WAITING event emission when blocked."""

    @pytest.mark.asyncio
    async def test_emits_waiting_for_blocked_files(self) -> None:
        """WAITING event emitted for each blocked file."""
        from src.core.models import LockEventType

        events: list[Any] = []

        backend = FakeLockingBackend()
        backend.canonicalize_fn = lambda p, ns: f"/canonical/{p}"
        backend.try_lock_results["/canonical/blocked1.py"] = False
        backend.try_lock_results["/canonical/blocked2.py"] = False
        backend.holders["/canonical/blocked1.py"] = "blocker"
        backend.holders["/canonical/blocked2.py"] = "blocker"
        backend.wait_for_lock_results["/canonical/blocked1.py"] = False
        backend.wait_for_lock_results["/canonical/blocked2.py"] = False

        handlers = _create_handlers(
            emit_lock_event=events.append, locking_backend=backend
        )

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

        handlers = _create_handlers(emit_lock_event=events.append)

        await handlers.lock_acquire.handler({"filepaths": ["a.py", "b.py"]})

        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_no_waiting_for_reentrant_acquire(self) -> None:
        """No WAITING events when same agent re-acquires lock it already holds.

        Uses FakeLockingBackend to verify idempotent re-acquire behavior:
        same agent holding a lock can re-acquire it without blocking or
        emitting WAITING events.
        """
        events: list[Any] = []
        backend = FakeLockingBackend()
        test_file = "/test/reentrant_test.py"
        handlers = _create_handlers(
            agent_id="reentrant-agent",
            emit_lock_event=events.append,
            locking_backend=backend,
        )

        # First acquire
        result1 = await handlers.lock_acquire.handler(
            {"filepaths": [test_file], "timeout_seconds": 0.1}
        )
        content1 = json.loads(result1["content"][0]["text"])
        assert content1["all_acquired"] is True
        assert len(events) == 0  # No WAITING on first acquire

        # Second acquire (re-entrant) - should succeed without WAITING
        result2 = await handlers.lock_acquire.handler(
            {"filepaths": [test_file], "timeout_seconds": 0.1}
        )
        content2 = json.loads(result2["content"][0]["text"])
        assert content2["all_acquired"] is True

        # No WAITING events should be emitted for re-entrant acquire
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

        backend = FakeLockingBackend()
        backend.try_lock_results["blocked.py"] = False
        backend.holders["blocked.py"] = "other"
        backend.wait_for_lock_results["blocked.py"] = False

        handlers = _create_handlers(emit_lock_event=async_emit, locking_backend=backend)

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

        backend = FakeLockingBackend()
        backend.try_lock_results["file.py"] = False
        backend.holders["file.py"] = "other"
        backend.wait_for_lock_results["file.py"] = False

        handlers = _create_handlers(
            emit_lock_event=failing_emit, locking_backend=backend
        )

        # Should not raise even though emit callback raises
        result = await handlers.lock_acquire.handler(
            {"filepaths": ["file.py"], "timeout_seconds": 0.1}
        )

        # Should return a result (even if emit failed)
        assert "content" in result


class TestLockReleaseWiring:
    """Test lock_release wiring to LockingBackend."""

    @pytest.mark.asyncio
    async def test_release_all_calls_cleanup_agent_locks(self) -> None:
        """all=true calls cleanup_agent_locks and returns released paths."""
        backend = FakeLockingBackend()
        backend.cleanup_result = (5, ["/path/to/file1.py", "/path/to/file2.py"])

        handlers = _create_handlers(locking_backend=backend)

        result = await handlers.lock_release.handler({"all": True})

        assert backend.cleanup_calls == ["test-agent"]
        content = json.loads(result["content"][0]["text"])
        assert content["count"] == 5
        assert content["released"] == ["/path/to/file1.py", "/path/to/file2.py"]

    @pytest.mark.asyncio
    async def test_release_specific_calls_release_lock(self) -> None:
        """Specific filepaths call release_lock for each."""
        backend = FakeLockingBackend()
        backend.canonicalize_fn = lambda p, ns: f"/{p}"
        # Pre-acquire locks so release returns True
        backend.locks["/a.py"] = "test-agent"
        backend.locks["/b.py"] = "test-agent"

        handlers = _create_handlers(locking_backend=backend)

        result = await handlers.lock_release.handler({"filepaths": ["a.py", "b.py"]})

        assert len(backend.release_lock_calls) == 2
        content = json.loads(result["content"][0]["text"])
        assert content["count"] == 2

    @pytest.mark.asyncio
    async def test_release_idempotent_behavior(self) -> None:
        """Release succeeds even when lock not held (idempotent)."""
        backend = FakeLockingBackend()
        backend.canonicalize_fn = lambda p, ns: f"/{p}"
        # Don't pre-acquire lock, so release_lock returns False

        handlers = _create_handlers(locking_backend=backend)

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
        backend = FakeLockingBackend()
        backend.canonicalize_fn = lambda p, ns: f"/{p}"
        backend.try_lock_results["/blocked.py"] = False
        backend.holders["/blocked.py"] = "other"

        handlers = _create_handlers(locking_backend=backend)

        await handlers.lock_acquire.handler(
            {
                "filepaths": ["ok.py", "blocked.py"],
                "timeout_seconds": 5.0,
            }
        )

        wait_paths = [call["filepath"] for call in backend.wait_for_lock_calls]
        assert "/blocked.py" in wait_paths

    @pytest.mark.asyncio
    async def test_non_blocking_mode_skips_waiting(self) -> None:
        """timeout_seconds=0 returns immediately without waiting."""
        backend = FakeLockingBackend()
        backend.try_lock_results["file.py"] = False
        backend.holders["file.py"] = "other"

        handlers = _create_handlers(locking_backend=backend)

        await handlers.lock_acquire.handler(
            {"filepaths": ["file.py"], "timeout_seconds": 0}
        )

        assert len(backend.wait_for_lock_calls) == 0

    @pytest.mark.asyncio
    async def test_cancels_pending_waits_on_first_success(self) -> None:
        """Cancels remaining wait tasks when first completes.

        This test requires custom async behavior that can't be represented
        in FakeLockingBackend's simple result dicts. We use a custom backend
        subclass with async timing behavior.
        """
        cancel_count = 0
        call_index = 0

        class TimingBackend(FakeLockingBackend):
            async def wait_for_lock_async(
                self,
                filepath: str,
                agent_id: str,
                ns: str | None,
                timeout_seconds: float,
                poll_interval_ms: int,
            ) -> bool:
                nonlocal call_index, cancel_count
                call_index += 1
                if call_index == 1:
                    # Fast wait - completes quickly
                    await asyncio.sleep(0.01)
                    self.locks[filepath] = agent_id
                    return True
                else:
                    # Slow wait - will be cancelled
                    try:
                        await asyncio.sleep(10)
                        return False
                    except asyncio.CancelledError:
                        cancel_count += 1
                        raise

        backend = TimingBackend()
        backend.canonicalize_fn = lambda p, ns: f"/{p}"
        backend.try_lock_results["/a.py"] = False
        backend.try_lock_results["/b.py"] = False
        backend.holders["/a.py"] = "other"
        backend.holders["/b.py"] = "other"

        handlers = _create_handlers(locking_backend=backend)

        await handlers.lock_acquire.handler(
            {
                "filepaths": ["a.py", "b.py"],
                "timeout_seconds": 5.0,
            }
        )

        # The slow wait should have been cancelled
        assert cancel_count == 1


class TestToolSchemaClaudeAPICompatibility:
    """Test tool schemas are compatible with Claude API constraints.

    Claude API does not support oneOf, allOf, or anyOf at the top level
    of tool input_schema. These tests ensure our schemas are valid.
    """

    UNSUPPORTED_TOP_LEVEL_KEYS: frozenset[str] = frozenset({"oneOf", "allOf", "anyOf"})

    def _get_tool_schemas(self) -> list[dict[str, Any]]:
        """Extract input_schema from both tools."""
        from src.infra.tools.locking_mcp import create_locking_mcp_server

        result = create_locking_mcp_server(
            agent_id="schema-test",
            repo_namespace=None,
            emit_lock_event=lambda e: None,
            _return_handlers=True,
        )
        assert isinstance(result, tuple)
        handlers = result[1]
        # SdkMcpTool.input_schema is typed loosely in the SDK
        schemas: list[Any] = [
            handlers.lock_acquire.input_schema,
            handlers.lock_release.input_schema,
        ]
        return schemas  # type: ignore[return-value]

    def test_no_unsupported_keywords_at_top_level(self) -> None:
        """Schemas must not use oneOf/allOf/anyOf at top level."""
        for schema in self._get_tool_schemas():
            top_level_keys = set(schema.keys())
            forbidden = top_level_keys & self.UNSUPPORTED_TOP_LEVEL_KEYS
            assert not forbidden, (
                f"Schema uses unsupported top-level keywords: {forbidden}. "
                "Claude API rejects oneOf/allOf/anyOf at the top level."
            )

    def test_schema_is_object_type(self) -> None:
        """Schemas must have type: object at top level."""
        for schema in self._get_tool_schemas():
            assert schema.get("type") == "object", "Schema must be type: object"

    def test_schema_has_properties(self) -> None:
        """Schemas must have properties defined."""
        for schema in self._get_tool_schemas():
            assert "properties" in schema, "Schema must have properties"
            assert isinstance(schema["properties"], dict)


class TestMCPServerConfiguration:
    """Test MCP server configuration."""

    def test_server_has_correct_name(self) -> None:
        """Server has expected name."""
        from src.infra.tools.locking_mcp import create_locking_mcp_server

        result = create_locking_mcp_server(
            "test", None, lambda e: None, locking_backend=FakeLockingBackend()
        )

        # Without _return_handlers, returns just the server config
        server: McpSdkServerConfig = result  # type: ignore[assignment]
        assert server["name"] == "mala-locking"

    def test_server_has_instance(self) -> None:
        """Server exposes an MCP server instance."""
        from src.infra.tools.locking_mcp import create_locking_mcp_server

        result = create_locking_mcp_server(
            "test", None, lambda e: None, locking_backend=FakeLockingBackend()
        )

        server: McpSdkServerConfig = result  # type: ignore[assignment]
        assert "instance" in server
        assert server["type"] == "sdk"

    def test_handlers_have_both_tools(self) -> None:
        """Handlers object has lock_acquire and lock_release SdkMcpTool objects."""
        handlers = _create_handlers()

        assert hasattr(handlers, "lock_acquire")
        assert hasattr(handlers, "lock_release")
        # SdkMcpTool objects have .handler attribute that is callable
        assert callable(handlers.lock_acquire.handler)
        assert callable(handlers.lock_release.handler)

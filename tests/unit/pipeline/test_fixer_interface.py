"""Unit tests for FixerInterface protocol and adapter.

Tests the FixerInterface protocol contract and FixerServiceAdapter
without subprocess or SDK dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.pipeline.fixer_interface import (
    FixerResult,
    FixerServiceAdapter,
)

if TYPE_CHECKING:
    from src.pipeline.fixer_interface import FixerInterface


class MockFixerImplementation:
    """Mock implementation of FixerInterface for testing."""

    def __init__(self, result: FixerResult) -> None:
        self.result = result
        self.calls: list[tuple[str, str]] = []

    async def run_fixer(self, failure_output: str, issue_id: str) -> FixerResult:
        """Record the call and return the configured result."""
        self.calls.append((failure_output, issue_id))
        return self.result


class TestFixerInterfaceProtocol:
    """Test that FixerInterface protocol works correctly with implementations."""

    @pytest.mark.asyncio
    async def test_mock_implementation_satisfies_protocol(self) -> None:
        """Verify a mock implementation satisfies FixerInterface."""
        result = FixerResult(success=True, log_path="/tmp/fixer.log")
        mock_fixer: FixerInterface = MockFixerImplementation(result)

        actual = await mock_fixer.run_fixer("Test failure", "issue-1")

        assert actual.success is True
        assert actual.log_path == "/tmp/fixer.log"

    @pytest.mark.asyncio
    async def test_protocol_accepts_success_result(self) -> None:
        """Verify protocol works with successful fixer result."""
        result = FixerResult(success=True, interrupted=False, log_path="/tmp/log")
        mock_fixer: FixerInterface = MockFixerImplementation(result)

        actual = await mock_fixer.run_fixer("Lint failed", "bd-123")

        assert actual.success is True
        assert actual.interrupted is False

    @pytest.mark.asyncio
    async def test_protocol_accepts_failure_result(self) -> None:
        """Verify protocol works with failed fixer result."""
        result = FixerResult(success=False, log_path="/tmp/failed.log")
        mock_fixer: FixerInterface = MockFixerImplementation(result)

        actual = await mock_fixer.run_fixer("Tests failed", "bd-456")

        assert actual.success is False
        assert actual.log_path == "/tmp/failed.log"

    @pytest.mark.asyncio
    async def test_protocol_accepts_interrupted_result(self) -> None:
        """Verify protocol works with interrupted fixer result."""
        result = FixerResult(success=None, interrupted=True)
        mock_fixer: FixerInterface = MockFixerImplementation(result)

        actual = await mock_fixer.run_fixer("Build failed", "bd-789")

        assert actual.success is None
        assert actual.interrupted is True

    @pytest.mark.asyncio
    async def test_mock_records_calls(self) -> None:
        """Verify mock implementation records all calls correctly."""
        result = FixerResult(success=True)
        mock_fixer = MockFixerImplementation(result)

        await mock_fixer.run_fixer("Error 1", "issue-a")
        await mock_fixer.run_fixer("Error 2", "issue-b")

        assert len(mock_fixer.calls) == 2
        assert mock_fixer.calls[0] == ("Error 1", "issue-a")
        assert mock_fixer.calls[1] == ("Error 2", "issue-b")


class TestFixerServiceAdapter:
    """Test the adapter that wraps FixerService for FixerInterface."""

    def _create_mock_fixer_service(
        self, fixer_result: FixerResult | None = None
    ) -> MagicMock:
        """Create a mock FixerService with configurable run_fixer."""
        from src.pipeline.fixer_service import FixerService

        fixer_service = MagicMock(spec=FixerService)

        if fixer_result is None:
            # Create a default successful result
            fixer_result = FixerResult(success=True, log_path="/tmp/log")

        fixer_service.run_fixer = AsyncMock(return_value=fixer_result)
        return fixer_service

    @pytest.mark.asyncio
    async def test_adapter_satisfies_protocol(self) -> None:
        """Verify adapter satisfies FixerInterface protocol."""
        fixer_service = self._create_mock_fixer_service()
        adapter: FixerInterface = FixerServiceAdapter(fixer_service)

        result = await adapter.run_fixer("Test failure", "issue-1")

        assert result.success is True
        fixer_service.run_fixer.assert_called_once()

    @pytest.mark.asyncio
    async def test_adapter_passes_failure_output(self) -> None:
        """Verify adapter passes failure_output to fixer_service."""
        fixer_service = self._create_mock_fixer_service()
        adapter = FixerServiceAdapter(fixer_service)

        await adapter.run_fixer("Lint check failed: unused import", "bd-test")

        call_args = fixer_service.run_fixer.call_args[0]
        context = call_args[0]
        assert context.failure_output == "Lint check failed: unused import"

    @pytest.mark.asyncio
    async def test_adapter_tracks_attempts_per_issue(self) -> None:
        """Verify adapter increments attempt counter per issue_id."""
        fixer_service = self._create_mock_fixer_service()
        adapter = FixerServiceAdapter(fixer_service)

        # First call for issue-a
        await adapter.run_fixer("Error 1", "issue-a")
        call1 = fixer_service.run_fixer.call_args[0][0]
        assert call1.attempt == 1

        # Second call for issue-a
        await adapter.run_fixer("Error 2", "issue-a")
        call2 = fixer_service.run_fixer.call_args[0][0]
        assert call2.attempt == 2

        # First call for issue-b (separate counter)
        await adapter.run_fixer("Error 3", "issue-b")
        call3 = fixer_service.run_fixer.call_args[0][0]
        assert call3.attempt == 1

    @pytest.mark.asyncio
    async def test_adapter_sets_failed_command(self) -> None:
        """Verify adapter sets appropriate failed_command context."""
        fixer_service = self._create_mock_fixer_service()
        adapter = FixerServiceAdapter(fixer_service)

        await adapter.run_fixer("Validation error", "issue-1")

        call_args = fixer_service.run_fixer.call_args[0]
        context = call_args[0]
        assert context.failed_command == "session_end validation"

    @pytest.mark.asyncio
    async def test_adapter_converts_success_result(self) -> None:
        """Verify adapter converts FixerResult correctly."""
        service_result = FixerResult(
            success=True, interrupted=False, log_path="/tmp/success.log"
        )
        fixer_service = self._create_mock_fixer_service(service_result)
        adapter = FixerServiceAdapter(fixer_service)

        result = await adapter.run_fixer("Error", "issue-1")

        assert isinstance(result, FixerResult)
        assert result.success is True
        assert result.interrupted is False
        assert result.log_path == "/tmp/success.log"

    @pytest.mark.asyncio
    async def test_adapter_converts_failure_result(self) -> None:
        """Verify adapter converts failed FixerResult correctly."""
        service_result = FixerResult(success=False, log_path="/tmp/failed.log")
        fixer_service = self._create_mock_fixer_service(service_result)
        adapter = FixerServiceAdapter(fixer_service)

        result = await adapter.run_fixer("Error", "issue-1")

        assert result.success is False
        assert result.log_path == "/tmp/failed.log"

    @pytest.mark.asyncio
    async def test_adapter_converts_interrupted_result(self) -> None:
        """Verify adapter converts interrupted FixerResult correctly."""
        service_result = FixerResult(
            success=None, interrupted=True, log_path="/tmp/interrupted.log"
        )
        fixer_service = self._create_mock_fixer_service(service_result)
        adapter = FixerServiceAdapter(fixer_service)

        result = await adapter.run_fixer("Error", "issue-1")

        assert result.success is None
        assert result.interrupted is True
        assert result.log_path == "/tmp/interrupted.log"


class TestFixerResultDataclass:
    """Test the FixerResult dataclass."""

    def test_default_values(self) -> None:
        """Verify FixerResult has correct defaults."""
        result = FixerResult(success=True)

        assert result.success is True
        assert result.interrupted is False
        assert result.log_path is None

    def test_all_fields(self) -> None:
        """Verify all FixerResult fields can be set."""
        result = FixerResult(success=False, interrupted=True, log_path="/tmp/test.log")

        assert result.success is False
        assert result.interrupted is True
        assert result.log_path == "/tmp/test.log"

    def test_none_success(self) -> None:
        """Verify success can be None (for unevaluated/interrupted)."""
        result = FixerResult(success=None, interrupted=True)

        assert result.success is None
        assert result.interrupted is True

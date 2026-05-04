"""Unit tests for FixerInterface protocol and adapter.

Tests the FixerInterface protocol contract and FixerServiceAdapter
without subprocess or SDK dependencies.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.pipeline.fixer_interface import FixerResult, FixerServiceAdapter


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

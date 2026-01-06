"""Tests for periodic validation in non-watch mode.

Verifies that ValidationConfig controls validation triggering
independently of WatchConfig.enabled.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from src.core.models import PeriodicValidationConfig, WatchConfig
from src.pipeline.issue_execution_coordinator import (
    AbortResult,
    CoordinatorConfig,
    IssueExecutionCoordinator,
)
from tests.fakes.event_sink import FakeEventSink
from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider


class TestPeriodicValidationThresholds:
    """Tests for validation triggering at N, 2N, 3N intervals."""

    @pytest.fixture
    def event_sink(self) -> FakeEventSink:
        return FakeEventSink()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "validate_every,completions,expected_validations",
        [
            # Note: +1 for final validation on exit when completed > last_validation_at
            (
                1,
                3,
                3,
            ),  # Validate after each issue (no extra final - 3 == last validation)
            (2, 4, 2),  # Validate at 2, 4 (no extra final - 4 == last validation)
            (3, 9, 3),  # Validate at 3, 6, 9 (no extra final - 9 == last validation)
            (3, 8, 3),  # Validate at 3, 6, plus final at 8 (8 > 6)
            (5, 5, 1),  # Validate at 5 (no extra final - 5 == last validation)
        ],
    )
    async def test_periodic_validation_thresholds(
        self,
        event_sink: FakeEventSink,
        validate_every: int,
        completions: int,
        expected_validations: int,
    ) -> None:
        """Validation triggers at N, 2N, 3N intervals."""
        provider = FakeIssueProvider(
            issues={
                f"issue-{i}": FakeIssue(id=f"issue-{i}", status="ready")
                for i in range(completions)
            }
        )
        # No watch mode - validation works independently
        validation_config = PeriodicValidationConfig(validate_every=validate_every)
        validation_call_count = 0

        async def validation_callback() -> bool:
            nonlocal validation_call_count
            validation_call_count += 1
            return True

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_agents=1),
        )

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            async def complete_immediately() -> None:
                pass

            return asyncio.create_task(complete_immediately())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)

        async def get_ready_side_effect(*args: object, **kwargs: object) -> list[str]:
            return [
                f"issue-{i}"
                for i in range(completions)
                if f"issue-{i}" not in coord.completed_ids
            ]

        provider.get_ready_async = AsyncMock(side_effect=get_ready_side_effect)  # type: ignore[method-assign]

        result = await asyncio.wait_for(
            coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
                validation_config=validation_config,
                validation_callback=validation_callback,
            ),
            timeout=5.0,
        )

        assert result.exit_code == 0
        assert result.exit_reason == "success"
        assert validation_call_count == expected_validations, (
            f"Expected {expected_validations} validations for validate_every={validate_every} "
            f"with {completions} completions, got {validation_call_count}"
        )


class TestValidationWithoutWatchMode:
    """Tests for validation behavior when watch_config.enabled=False."""

    @pytest.fixture
    def event_sink(self) -> FakeEventSink:
        return FakeEventSink()

    @pytest.mark.asyncio
    async def test_validation_without_watch_mode(
        self,
        event_sink: FakeEventSink,
    ) -> None:
        """Validation works when watch_config.enabled=False."""
        provider = FakeIssueProvider(
            issues={
                f"issue-{i}": FakeIssue(id=f"issue-{i}", status="ready")
                for i in range(6)
            }
        )
        # Explicitly disable watch mode
        watch_config = WatchConfig(enabled=False)
        validation_config = PeriodicValidationConfig(validate_every=3)
        validation_call_count = 0

        async def validation_callback() -> bool:
            nonlocal validation_call_count
            validation_call_count += 1
            return True

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_agents=1),
        )

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            async def complete_immediately() -> None:
                pass

            return asyncio.create_task(complete_immediately())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)

        async def get_ready_side_effect(*args: object, **kwargs: object) -> list[str]:
            return [
                f"issue-{i}"
                for i in range(6)
                if f"issue-{i}" not in coord.completed_ids
            ]

        provider.get_ready_async = AsyncMock(side_effect=get_ready_side_effect)  # type: ignore[method-assign]

        result = await asyncio.wait_for(
            coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
                watch_config=watch_config,
                validation_config=validation_config,
                validation_callback=validation_callback,
            ),
            timeout=5.0,
        )

        assert result.exit_code == 0
        assert result.exit_reason == "success"
        # Should validate at 3 and 6 completions
        assert validation_call_count == 2


class TestNoValidationWhenConfigNone:
    """Tests for no periodic validation when validation_config=None."""

    @pytest.fixture
    def event_sink(self) -> FakeEventSink:
        return FakeEventSink()

    @pytest.mark.asyncio
    async def test_no_periodic_validation_when_config_none(
        self,
        event_sink: FakeEventSink,
    ) -> None:
        """No periodic validation when validation_config=None.

        Note: Final validation still runs on exit when validation_callback is provided,
        but no periodic validation at intervals should occur.
        """
        provider = FakeIssueProvider(
            issues={
                f"issue-{i}": FakeIssue(id=f"issue-{i}", status="ready")
                for i in range(5)
            }
        )
        validation_call_count = 0
        validation_completed_counts: list[int] = []

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_agents=1),
        )

        async def validation_callback() -> bool:
            nonlocal validation_call_count
            validation_call_count += 1
            validation_completed_counts.append(len(coord.completed_ids))
            return True

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            async def complete_immediately() -> None:
                pass

            return asyncio.create_task(complete_immediately())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)

        async def get_ready_side_effect(*args: object, **kwargs: object) -> list[str]:
            return [
                f"issue-{i}"
                for i in range(5)
                if f"issue-{i}" not in coord.completed_ids
            ]

        provider.get_ready_async = AsyncMock(side_effect=get_ready_side_effect)  # type: ignore[method-assign]

        result = await asyncio.wait_for(
            coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
                # validation_config=None (not passed)
                validation_callback=validation_callback,
            ),
            timeout=5.0,
        )

        assert result.exit_code == 0
        assert result.exit_reason == "success"
        # Only final validation should have occurred (at 5 completions), not periodic
        assert validation_call_count == 1
        assert validation_completed_counts == [5]  # Only at exit


class TestExactBoundaryValidation:
    """Tests for validation at exact threshold boundaries."""

    @pytest.fixture
    def event_sink(self) -> FakeEventSink:
        return FakeEventSink()

    @pytest.mark.asyncio
    async def test_exact_boundary_validation(
        self,
        event_sink: FakeEventSink,
    ) -> None:
        """Validate at exact threshold (e.g., 5 issues with validate_every=5)."""
        provider = FakeIssueProvider(
            issues={
                f"issue-{i}": FakeIssue(id=f"issue-{i}", status="ready")
                for i in range(5)
            }
        )
        validation_config = PeriodicValidationConfig(validate_every=5)
        validation_call_count = 0

        async def validation_callback() -> bool:
            nonlocal validation_call_count
            validation_call_count += 1
            return True

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_agents=1),
        )

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            async def complete_immediately() -> None:
                pass

            return asyncio.create_task(complete_immediately())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)

        async def get_ready_side_effect(*args: object, **kwargs: object) -> list[str]:
            return [
                f"issue-{i}"
                for i in range(5)
                if f"issue-{i}" not in coord.completed_ids
            ]

        provider.get_ready_async = AsyncMock(side_effect=get_ready_side_effect)  # type: ignore[method-assign]

        result = await asyncio.wait_for(
            coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
                validation_config=validation_config,
                validation_callback=validation_callback,
            ),
            timeout=5.0,
        )

        assert result.exit_code == 0
        assert result.exit_reason == "success"
        # Should validate exactly once at the 5-completion boundary
        assert validation_call_count == 1

    @pytest.mark.asyncio
    async def test_validation_at_multiples_of_threshold(
        self,
        event_sink: FakeEventSink,
    ) -> None:
        """Validate at 10, 20 when validate_every=10 and 20 issues complete.

        With exactly 20 completions, no extra final validation runs since
        last_validation_at == completed_count.
        """
        provider = FakeIssueProvider(
            issues={
                f"issue-{i}": FakeIssue(id=f"issue-{i}", status="ready")
                for i in range(20)
            }
        )
        validation_config = PeriodicValidationConfig(validate_every=10)
        validation_call_count = 0

        async def validation_callback() -> bool:
            nonlocal validation_call_count
            validation_call_count += 1
            return True

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_agents=1),
        )

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            async def complete_immediately() -> None:
                pass

            return asyncio.create_task(complete_immediately())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)

        async def get_ready_side_effect(*args: object, **kwargs: object) -> list[str]:
            return [
                f"issue-{i}"
                for i in range(20)
                if f"issue-{i}" not in coord.completed_ids
            ]

        provider.get_ready_async = AsyncMock(side_effect=get_ready_side_effect)  # type: ignore[method-assign]

        result = await asyncio.wait_for(
            coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
                validation_config=validation_config,
                validation_callback=validation_callback,
            ),
            timeout=10.0,
        )

        assert result.exit_code == 0
        assert result.exit_reason == "success"
        # Should validate at 10 and 20 (no extra final since 20 == last_validation_at)
        assert validation_call_count == 2


class TestValidationFailureInNonWatchMode:
    """Tests for validation failure behavior in non-watch mode."""

    @pytest.fixture
    def event_sink(self) -> FakeEventSink:
        return FakeEventSink()

    @pytest.mark.asyncio
    async def test_validation_failure_exits_with_code_1(
        self,
        event_sink: FakeEventSink,
    ) -> None:
        """Validation failure should return exit_code=1 in non-watch mode."""
        provider = FakeIssueProvider(
            issues={
                f"issue-{i}": FakeIssue(id=f"issue-{i}", status="ready")
                for i in range(5)
            }
        )
        validation_config = PeriodicValidationConfig(validate_every=3)

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_agents=1),
        )

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            async def complete_immediately() -> None:
                pass

            return asyncio.create_task(complete_immediately())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)

        async def get_ready_side_effect(*args: object, **kwargs: object) -> list[str]:
            return [
                f"issue-{i}"
                for i in range(5)
                if f"issue-{i}" not in coord.completed_ids
            ]

        provider.get_ready_async = AsyncMock(side_effect=get_ready_side_effect)  # type: ignore[method-assign]

        result = await asyncio.wait_for(
            coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=AsyncMock(return_value=AbortResult(aborted_count=0)),
                validation_config=validation_config,
                validation_callback=AsyncMock(return_value=False),  # Validation fails
            ),
            timeout=5.0,
        )

        assert result.exit_code == 1
        assert result.exit_reason == "validation_failed"
        # Should have stopped at 3 completions when validation failed
        assert len(coord.completed_ids) == 3

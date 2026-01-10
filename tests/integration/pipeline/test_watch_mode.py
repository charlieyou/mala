"""Integration tests for watch mode behavior.

These tests verify the full watch mode flow including:
- SIGINT handling during idle and active states
- Issues picked up after poll interval
- --max-issues interaction with watch mode
- periodic validation triggers at correct intervals

Unlike unit tests, these use real asyncio timing (with short intervals)
and verify the complete coordinator flow.
"""

import asyncio
import signal
import subprocess
import sys
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest

from src.core.models import PeriodicValidationConfig, WatchConfig
from src.pipeline.issue_execution_coordinator import (
    AbortResult,
    CoordinatorConfig,
    IssueExecutionCoordinator,
)
from tests.fakes.coordinator_callbacks import (
    FakeAbortCallback,
    FakeFinalizeCallback,
    FakeSpawnCallback,
)
from tests.fakes.event_sink import FakeEventSink
from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.integration
class TestSigintHandling:
    """Integration tests for SIGINT handling in watch mode."""

    @pytest.mark.asyncio
    async def test_sigint_during_idle_exits_with_code_130(self) -> None:
        """SIGINT during idle sleep exits cleanly with code 130.

        This test verifies that when watch mode is idle (no ready issues,
        no active agents), receiving SIGINT triggers a clean exit with
        the standard interrupt exit code (130 = 128 + SIGINT).
        """
        provider = FakeIssueProvider()  # No issues - will enter idle
        event_sink = FakeEventSink()
        watch_config = WatchConfig(
            enabled=True,
            poll_interval_seconds=0.1,  # Short interval for fast test
        )
        interrupt_event = asyncio.Event()

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(),
        )

        async def set_interrupt_after_idle() -> None:
            """Wait for idle state then trigger interrupt."""
            # Wait for watch_idle event (indicates we're in idle sleep)
            for _ in range(100):  # 100 * 0.01s = 1.0s max wait
                if event_sink.has_event("watch_idle"):
                    interrupt_event.set()
                    return
                await asyncio.sleep(0.01)
            pytest.fail("watch_idle event never observed before timeout")

        # Run coordinator and interrupt task concurrently using TaskGroup
        # so helper failure cancels run_loop immediately
        async with asyncio.timeout(2.0):
            async with asyncio.TaskGroup() as tg:
                run_task = tg.create_task(
                    coord.run_loop(
                        spawn_callback=FakeSpawnCallback(),
                        finalize_callback=FakeFinalizeCallback(),
                        abort_callback=FakeAbortCallback(),
                        watch_config=watch_config,
                        interrupt_event=interrupt_event,
                    )
                )
                tg.create_task(set_interrupt_after_idle())
            result = run_task.result()

        assert result.exit_code == 130, (
            f"Expected exit code 130, got {result.exit_code}"
        )
        assert result.exit_reason == "interrupted"
        # watch_idle is guaranteed by set_interrupt_after_idle() (fails if not seen)

    @pytest.mark.asyncio
    async def test_sigint_during_active_processing_exits_with_code_130(self) -> None:
        """SIGINT during active issue processing exits cleanly with code 130.

        When agents are actively processing issues and SIGINT is received,
        the coordinator should abort active tasks and exit with code 130.
        """
        provider = FakeIssueProvider(
            issues={"issue-1": FakeIssue(id="issue-1", status="open")}
        )
        event_sink = FakeEventSink()
        watch_config = WatchConfig(enabled=True, poll_interval_seconds=0.1)
        interrupt_event = asyncio.Event()

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_agents=1),
        )

        task_started = asyncio.Event()

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            """Spawn a task that signals when started then waits."""

            async def long_running() -> None:
                task_started.set()
                await asyncio.sleep(10.0)  # Will be interrupted

            return asyncio.create_task(long_running())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            coord.mark_completed(issue_id)

        async def abort_callback(*, is_interrupt: bool = False) -> AbortResult:
            # Cancel and await all active tasks to avoid "Task destroyed" warnings
            tasks = list(coord.active_tasks.values())
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            return AbortResult(aborted_count=len(tasks))

        async def set_interrupt_when_active() -> None:
            """Wait for task to start then trigger interrupt."""
            await task_started.wait()
            await asyncio.sleep(0.05)  # Brief delay to ensure processing
            interrupt_event.set()

        async with asyncio.timeout(2.0):
            interrupt_task = asyncio.create_task(set_interrupt_when_active())
            result = await coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=abort_callback,
                watch_config=watch_config,
                interrupt_event=interrupt_event,
            )
            await interrupt_task

        assert result.exit_code == 130
        assert result.exit_reason == "interrupted"


@pytest.mark.integration
class TestIssuePolling:
    """Integration tests for issue polling behavior."""

    @pytest.mark.asyncio
    async def test_new_issues_picked_up_after_poll(self) -> None:
        """Issues added during watch are processed on next poll.

        Verifies that when new issues become ready while watch mode is
        waiting, they are picked up and processed after the poll interval.
        """
        provider = FakeIssueProvider()  # Start with no issues
        event_sink = FakeEventSink()
        watch_config = WatchConfig(
            enabled=True,
            poll_interval_seconds=0.1,  # Short poll for fast test
        )
        interrupt_event = asyncio.Event()

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_agents=1),
        )

        issues_processed: list[str] = []

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            async def complete_immediately() -> None:
                pass

            return asyncio.create_task(complete_immediately())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            issues_processed.append(issue_id)
            coord.mark_completed(issue_id)
            # Exit after processing the issue we added
            if issue_id == "new-issue":
                interrupt_event.set()

        async def add_issue_after_idle() -> None:
            """Wait for idle then add a new issue."""
            # Wait for idle state
            for _ in range(100):  # 100 * 0.01s = 1.0s max wait
                if event_sink.has_event("watch_idle"):
                    # Add a new issue while idle
                    provider.issues["new-issue"] = FakeIssue(
                        id="new-issue", status="open"
                    )
                    return
                await asyncio.sleep(0.01)
            pytest.fail("watch_idle event never observed before timeout")

        # Use TaskGroup so helper failure cancels run_loop immediately
        async with asyncio.timeout(2.0):
            async with asyncio.TaskGroup() as tg:
                run_task = tg.create_task(
                    coord.run_loop(
                        spawn_callback=spawn_callback,
                        finalize_callback=finalize_callback,
                        abort_callback=FakeAbortCallback(),
                        watch_config=watch_config,
                        interrupt_event=interrupt_event,
                    )
                )
                tg.create_task(add_issue_after_idle())
            result = run_task.result()

        assert "new-issue" in issues_processed, (
            f"Expected 'new-issue' to be processed, got {issues_processed}"
        )
        assert result.issues_spawned >= 1


@pytest.mark.integration
class TestMaxIssuesWithWatch:
    """Integration tests for --max-issues interaction with watch mode."""

    @pytest.mark.asyncio
    async def test_max_issues_stops_watch_mode(self) -> None:
        """--max-issues limit stops watch even with more work available.

        When max_issues is reached, watch mode should exit cleanly
        rather than continuing to wait for more issues.
        """
        # Create more issues than the limit
        provider = FakeIssueProvider(
            issues={
                f"issue-{i}": FakeIssue(id=f"issue-{i}", status="open")
                for i in range(10)
            }
        )
        event_sink = FakeEventSink()
        watch_config = WatchConfig(enabled=True, poll_interval_seconds=0.1)

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_issues=2, max_agents=1),
        )

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            async def complete_immediately() -> None:
                pass

            return asyncio.create_task(complete_immediately())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            # Update provider status so issue doesn't reappear as ready
            await provider.close_async(issue_id)
            coord.mark_completed(issue_id)

        async with asyncio.timeout(2.0):
            result = await coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=FakeAbortCallback(),
                watch_config=watch_config,
            )

        assert result.exit_code == 0
        assert result.exit_reason == "limit_reached"
        assert len(coord.completed_ids) == 2, (
            f"Expected exactly 2 completed, got {len(coord.completed_ids)}"
        )

    @pytest.mark.asyncio
    async def test_max_issues_with_parallel_agents(self) -> None:
        """--max-issues works correctly with multiple parallel agents.

        Even with parallel spawning, the loop should exit when the
        completed count reaches max_issues.
        """
        provider = FakeIssueProvider(
            issues={
                f"issue-{i}": FakeIssue(id=f"issue-{i}", status="open")
                for i in range(10)
            }
        )
        event_sink = FakeEventSink()
        watch_config = WatchConfig(enabled=True, poll_interval_seconds=0.1)

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_issues=3, max_agents=3),
        )

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            async def complete_immediately() -> None:
                pass

            return asyncio.create_task(complete_immediately())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            # Update provider status so issue doesn't reappear as ready
            await provider.close_async(issue_id)
            coord.mark_completed(issue_id)

        async with asyncio.timeout(2.0):
            result = await coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=FakeAbortCallback(),
                watch_config=watch_config,
            )

        assert result.exit_code == 0
        assert result.exit_reason == "limit_reached"
        # With parallel agents, we may complete exactly 3 or slightly more
        # due to task timing, but the loop exits based on completed_count
        assert len(coord.completed_ids) >= 3


@pytest.mark.integration
class TestValidationTriggers:
    """Integration tests for periodic validation behavior."""

    @pytest.mark.asyncio
    async def test_validation_runs_at_threshold(self) -> None:
        """Periodic validation triggers at correct intervals.

        Validation callback should be called when completed_count reaches
        the validate_every threshold.
        """
        provider = FakeIssueProvider(
            issues={
                f"issue-{i}": FakeIssue(id=f"issue-{i}", status="open")
                for i in range(5)
            }
        )
        event_sink = FakeEventSink()
        watch_config = WatchConfig(
            enabled=True,
            poll_interval_seconds=0.1,
        )
        validation_config = PeriodicValidationConfig(
            validate_every=2
        )  # Trigger at 2 completions
        validation_calls: list[int] = []  # Record completed count at each call

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            # Use max_issues to exit after processing 5 issues
            config=CoordinatorConfig(max_agents=1, max_issues=5),
        )

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            async def complete_immediately() -> None:
                pass

            return asyncio.create_task(complete_immediately())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            # Update provider status so issue doesn't reappear as ready
            await provider.close_async(issue_id)
            coord.mark_completed(issue_id)

        async def validation_callback() -> bool:
            validation_calls.append(len(coord.completed_ids))
            return True

        async with asyncio.timeout(5.0):
            await coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=FakeAbortCallback(),
                watch_config=watch_config,
                validation_config=validation_config,
                validation_callback=validation_callback,
            )

        # Should trigger validation at 2, 4, and final (5)
        assert len(validation_calls) >= 2, (
            f"Expected at least 2 validation calls, got {len(validation_calls)}: {validation_calls}"
        )
        # First validation should be exactly at threshold (2)
        assert validation_calls[0] == 2, (
            f"First validation at {validation_calls[0]}, expected == 2"
        )

    @pytest.mark.asyncio
    async def test_validation_failure_stops_watch(self) -> None:
        """Validation failure exits watch mode with code 1.

        When validation_callback returns False, watch mode should
        exit immediately with exit_code=1.
        """
        provider = FakeIssueProvider(
            issues={
                f"issue-{i}": FakeIssue(id=f"issue-{i}", status="open")
                for i in range(5)
            }
        )
        event_sink = FakeEventSink()
        watch_config = WatchConfig(
            enabled=True,
            poll_interval_seconds=0.1,
        )
        validation_config = PeriodicValidationConfig(validate_every=2)

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
            # Update provider status so issue doesn't reappear as ready
            await provider.close_async(issue_id)
            coord.mark_completed(issue_id)

        async def validation_callback() -> bool:
            return False  # Always fail

        async with asyncio.timeout(2.0):
            result = await coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=FakeAbortCallback(),
                watch_config=watch_config,
                validation_config=validation_config,
                validation_callback=validation_callback,
            )

        assert result.exit_code == 1
        assert result.exit_reason == "validation_failed"
        # Should have stopped after first validation (at 2 completions)
        assert len(coord.completed_ids) <= 3  # Some tolerance for timing


@pytest.mark.integration
class TestWatchModeFullFlow:
    """End-to-end integration tests for complete watch mode scenarios."""

    @pytest.mark.asyncio
    async def test_full_flow_process_complete_exit(self) -> None:
        """Full watch flow: process issues, enter idle, interrupt, exit.

        This test verifies the complete lifecycle:
        1. Start with issues
        2. Process all issues
        3. Enter idle state (no more issues)
        4. Receive SIGINT (Stage 2)
        5. Exit cleanly (Stage 2 does NOT run validation per spec)
        """
        provider = FakeIssueProvider(
            issues={
                f"issue-{i}": FakeIssue(id=f"issue-{i}", status="open")
                for i in range(3)
            }
        )
        event_sink = FakeEventSink()
        watch_config = WatchConfig(
            enabled=True,
            poll_interval_seconds=0.1,
        )
        interrupt_event = asyncio.Event()
        validation_called = False

        coord = IssueExecutionCoordinator(
            beads=provider,
            event_sink=event_sink,
            config=CoordinatorConfig(max_agents=2),
        )

        async def spawn_callback(issue_id: str) -> asyncio.Task[None]:
            async def complete_immediately() -> None:
                pass

            return asyncio.create_task(complete_immediately())

        async def finalize_callback(issue_id: str, task: asyncio.Task[None]) -> None:
            # Update provider status so issue doesn't reappear as ready
            await provider.close_async(issue_id)
            coord.mark_completed(issue_id)

        async def validation_callback() -> bool:
            nonlocal validation_called
            validation_called = True
            return True

        async def interrupt_after_idle() -> None:
            """Wait for all issues processed then interrupt on idle."""
            for _ in range(100):
                # Wait until all 3 issues are complete
                if len(coord.completed_ids) >= 3:
                    # Wait for idle state
                    await asyncio.sleep(0.15)  # Allow time for idle detection
                    break
                await asyncio.sleep(0.02)
            interrupt_event.set()

        async with asyncio.timeout(3.0):
            interrupt_task = asyncio.create_task(interrupt_after_idle())
            result = await coord.run_loop(
                spawn_callback=spawn_callback,
                finalize_callback=finalize_callback,
                abort_callback=FakeAbortCallback(),
                watch_config=watch_config,
                validation_callback=validation_callback,
                interrupt_event=interrupt_event,
            )
            await interrupt_task

        assert result.exit_code == 130
        assert result.exit_reason == "interrupted"
        assert len(coord.completed_ids) == 3
        # Stage 2 interrupt does NOT call validation - exit code is determined by
        # orchestrator's _abort_exit_code snapshot, not by validation during abort
        assert not validation_called, (
            "Validation should NOT be called during Stage 2 interrupt"
        )


@pytest.mark.integration
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="SIGINT subprocess test not supported on Windows",
)
class TestSigintSubprocess:
    """Tests that verify SIGINT handling via actual signal delivery.

    These tests run the watch loop in a subprocess and send real SIGINT
    signals to verify signal handling works correctly in practice.
    """

    def test_sigint_subprocess_exits_130(self, tmp_path: "Path") -> None:
        """Real SIGINT signal causes exit code 130 in subprocess.

        This test creates a minimal script that runs watch mode and
        sends it SIGINT, verifying the real signal handling path.
        """
        import os
        import select
        from pathlib import Path

        # Derive repo root from this test file's location
        repo_root = Path(__file__).resolve().parent.parent.parent.parent

        # Create a test script that runs watch mode
        script = tmp_path / "watch_test.py"
        script.write_text(
            """
import asyncio
import sys
import signal

from src.core.models import PeriodicValidationConfig, WatchConfig
from src.pipeline.issue_execution_coordinator import (
    CoordinatorConfig,
    IssueExecutionCoordinator,
)
from tests.fakes.coordinator_callbacks import (
    FakeAbortCallback,
    FakeFinalizeCallback,
    FakeSpawnCallback,
)
from tests.fakes.event_sink import FakeEventSink
from tests.fakes.issue_provider import FakeIssueProvider

async def main():
    provider = FakeIssueProvider()
    event_sink = FakeEventSink()
    watch_config = WatchConfig(enabled=True, poll_interval_seconds=60.0)
    interrupt_event = asyncio.Event()

    # Set up SIGINT handler
    def sigint_handler(signum, frame):
        interrupt_event.set()
    signal.signal(signal.SIGINT, sigint_handler)

    coord = IssueExecutionCoordinator(
        beads=provider,
        event_sink=event_sink,
        config=CoordinatorConfig(),
    )

    # Print ready signal
    print("READY", flush=True)

    result = await coord.run_loop(
        spawn_callback=FakeSpawnCallback(),
        finalize_callback=FakeFinalizeCallback(),
        abort_callback=FakeAbortCallback(),
        watch_config=watch_config,
        interrupt_event=interrupt_event,
    )

    sys.exit(result.exit_code)

asyncio.run(main())
"""
        )

        # Set up environment with PYTHONPATH pointing to repo root
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root)

        # Run the script in a subprocess (stderr=DEVNULL to avoid pipe deadlock)
        proc = subprocess.Popen(
            [sys.executable, str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )

        try:
            # Wait for "READY" signal with early exit detection
            ready_received = False
            for _ in range(50):  # 5 second timeout
                if proc.poll() is not None:
                    # Process exited early (likely import error)
                    pytest.fail(f"Subprocess exited early with code {proc.returncode}")
                # Non-blocking read
                if select.select([proc.stdout], [], [], 0.1)[0]:
                    line = proc.stdout.readline()  # type: ignore[union-attr]
                    if "READY" in line:
                        ready_received = True
                        break

            if not ready_received:
                proc.kill()
                proc.wait()
                pytest.fail("Subprocess never sent READY signal")

            # Send SIGINT
            proc.send_signal(signal.SIGINT)

            # Wait for exit with timeout
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                pytest.fail("Process did not exit after SIGINT")

            assert proc.returncode == 130, (
                f"Expected exit code 130, got {proc.returncode}"
            )

        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


@pytest.mark.integration
class TestPeriodicValidationWithoutWatch:
    """Integration tests for periodic validation without watch mode.

    These tests verify that periodic validation works independently of watch.
    They are expected to FAIL until T003 removes the watch_enabled guards.
    """

    @pytest.mark.asyncio
    async def test_periodic_validation_without_watch_mode(self) -> None:
        """Verify periodic validation triggers in non-watch mode.

        This test verifies that when watch_config.enabled=False but
        validation_config.validate_every is set, validation still triggers.
        """
        # Set up issues that will complete
        provider = FakeIssueProvider(
            issues={
                f"issue-{i}": FakeIssue(id=f"issue-{i}", status="ready")
                for i in range(5)
            }
        )
        event_sink = FakeEventSink()
        # Watch mode disabled, but validation enabled
        watch_config = WatchConfig(enabled=False)
        validation_config = PeriodicValidationConfig(validate_every=2)

        # Track validation callback invocations
        validation_calls: list[bool] = []

        async def validation_callback() -> bool:
            validation_calls.append(True)
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
                abort_callback=FakeAbortCallback(),
                watch_config=watch_config,
                validation_config=validation_config,
                validation_callback=validation_callback,
            ),
            timeout=5.0,
        )

        # Should complete successfully (no watch mode = exits when no more issues)
        assert result.exit_code == 0

        # Validation callback should have been called at least once
        # (completes 5 issues with validate_every=2, so should trigger at 2, 4)
        assert len(validation_calls) >= 1, (
            "Expected validation_callback to be called at least once "
            "but it was never called. This fails because watch_enabled guards "
            "(T003) are still present."
        )

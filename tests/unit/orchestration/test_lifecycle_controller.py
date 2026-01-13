"""Unit tests for LifecycleController.

Tests verify the three-stage SIGINT escalation behavior:
- Stage 1: Drain mode
- Stage 2: Graceful abort
- Stage 3: Hard abort

Also tests state reset, interrupt checking, and callback wiring.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from src.orchestration.lifecycle_controller import (
    ESCALATION_WINDOW_SECONDS,
    LifecycleController,
)


@pytest.fixture
def controller() -> LifecycleController:
    """Create a fresh LifecycleController for testing."""
    return LifecycleController()


@pytest.fixture
def loop() -> asyncio.AbstractEventLoop:
    """Get the current event loop."""
    return asyncio.new_event_loop()


@pytest.mark.unit
class TestLifecycleControllerInitialState:
    """Tests for initial controller state."""

    def test_initial_state_not_interrupted(
        self, controller: LifecycleController
    ) -> None:
        """Controller starts in non-interrupted state."""
        assert not controller.is_interrupted()
        assert not controller.is_drain_mode()
        assert not controller.is_abort_mode()
        assert not controller.is_shutdown_requested()

    def test_initial_events_not_set(self, controller: LifecycleController) -> None:
        """Events are not set initially."""
        assert not controller.interrupt_event.is_set()
        assert not controller.drain_event.is_set()

    def test_initial_exit_code(self, controller: LifecycleController) -> None:
        """Default abort exit code is 130."""
        assert controller.abort_exit_code == 130

    def test_validation_failed_default(self, controller: LifecycleController) -> None:
        """Validation failed is False by default."""
        assert not controller.validation_failed


@pytest.mark.unit
class TestSigintEscalation:
    """Tests for SIGINT three-stage escalation."""

    def test_stage1_drain_mode(
        self, controller: LifecycleController, loop: asyncio.AbstractEventLoop
    ) -> None:
        """First SIGINT enters drain mode."""
        # Configure callbacks
        drain_started_calls: list[int] = []
        controller.configure_callbacks(
            get_active_task_count=lambda: 3,
            on_drain_started=lambda count: drain_started_calls.append(count),
            on_abort_started=MagicMock(),
            on_force_abort=MagicMock(),
            forward_sigint=MagicMock(),
            kill_processes=MagicMock(),
        )

        controller.handle_sigint(loop)
        loop.run_until_complete(asyncio.sleep(0))  # Process callbacks

        assert controller.is_drain_mode()
        assert not controller.is_abort_mode()
        assert not controller.is_shutdown_requested()
        assert controller.drain_event.is_set()
        assert not controller.interrupt_event.is_set()
        assert drain_started_calls == [3]

    def test_stage2_graceful_abort(
        self, controller: LifecycleController, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Second SIGINT enters abort mode."""
        forward_sigint = MagicMock()
        abort_started = MagicMock()
        controller.configure_callbacks(
            get_active_task_count=lambda: 2,
            on_drain_started=MagicMock(),
            on_abort_started=abort_started,
            on_force_abort=MagicMock(),
            forward_sigint=forward_sigint,
            kill_processes=MagicMock(),
        )

        controller.handle_sigint(loop)  # Stage 1
        controller.handle_sigint(loop)  # Stage 2
        loop.run_until_complete(asyncio.sleep(0))

        assert controller.is_drain_mode()
        assert controller.is_abort_mode()
        assert not controller.is_shutdown_requested()
        assert controller.interrupt_event.is_set()
        forward_sigint.assert_called_once()
        abort_started.assert_called_once()

    def test_stage3_hard_abort(
        self, controller: LifecycleController, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Third SIGINT triggers hard abort."""
        kill_processes = MagicMock()
        force_abort = MagicMock()
        controller.configure_callbacks(
            get_active_task_count=lambda: 1,
            on_drain_started=MagicMock(),
            on_abort_started=MagicMock(),
            on_force_abort=force_abort,
            forward_sigint=MagicMock(),
            kill_processes=kill_processes,
        )

        controller.handle_sigint(loop)  # Stage 1
        controller.handle_sigint(loop)  # Stage 2
        controller.handle_sigint(loop)  # Stage 3
        loop.run_until_complete(asyncio.sleep(0))

        assert controller.is_shutdown_requested()
        assert controller.abort_exit_code == 130  # Hard abort always 130
        kill_processes.assert_called_once()
        force_abort.assert_called_once()

    def test_stage3_cancels_run_task(
        self, controller: LifecycleController, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Stage 3 cancels the run task if set."""
        controller.configure_callbacks(
            get_active_task_count=lambda: 0,
            on_drain_started=MagicMock(),
            on_abort_started=MagicMock(),
            on_force_abort=MagicMock(),
            forward_sigint=MagicMock(),
            kill_processes=MagicMock(),
        )

        # Create a mock task
        mock_task = MagicMock(spec=asyncio.Task)
        controller.run_task = mock_task

        controller.handle_sigint(loop)  # Stage 1
        controller.handle_sigint(loop)  # Stage 2
        controller.handle_sigint(loop)  # Stage 3
        loop.run_until_complete(asyncio.sleep(0))

        mock_task.cancel.assert_called_once()


@pytest.mark.unit
class TestEscalationWindow:
    """Tests for escalation window behavior."""

    def test_escalation_resets_after_window(
        self, controller: LifecycleController, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Escalation count resets if idle longer than window."""
        controller.configure_callbacks(
            get_active_task_count=lambda: 0,
            on_drain_started=MagicMock(),
            on_abort_started=MagicMock(),
            on_force_abort=MagicMock(),
            forward_sigint=MagicMock(),
            kill_processes=MagicMock(),
        )

        # Manually set state as if first SIGINT happened long ago
        controller._sigint_last_at = time.monotonic() - ESCALATION_WINDOW_SECONDS - 1

        # This should be treated as first SIGINT (count reset)
        controller.handle_sigint(loop)
        loop.run_until_complete(asyncio.sleep(0))

        # Should be in drain mode, not abort (count was reset to 1)
        assert controller.is_drain_mode()
        assert not controller.is_abort_mode()

    def test_escalation_preserved_in_drain_mode(
        self, controller: LifecycleController, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Escalation count preserved once in drain mode, regardless of time."""
        controller.configure_callbacks(
            get_active_task_count=lambda: 0,
            on_drain_started=MagicMock(),
            on_abort_started=MagicMock(),
            on_force_abort=MagicMock(),
            forward_sigint=MagicMock(),
            kill_processes=MagicMock(),
        )

        # Enter drain mode
        controller.handle_sigint(loop)
        loop.run_until_complete(asyncio.sleep(0))
        assert controller.is_drain_mode()

        # Simulate time passing beyond window
        controller._sigint_last_at = time.monotonic() - ESCALATION_WINDOW_SECONDS - 1

        # Next SIGINT should still escalate to stage 2
        controller.handle_sigint(loop)
        loop.run_until_complete(asyncio.sleep(0))

        assert controller.is_abort_mode()


@pytest.mark.unit
class TestValidationFailedExitCode:
    """Tests for validation-aware exit codes."""

    def test_abort_exit_code_130_when_validation_passed(
        self, controller: LifecycleController, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Exit code is 130 when validation passed before abort."""
        controller.configure_callbacks(
            get_active_task_count=lambda: 0,
            on_drain_started=MagicMock(),
            on_abort_started=MagicMock(),
            on_force_abort=MagicMock(),
            forward_sigint=MagicMock(),
            kill_processes=MagicMock(),
        )
        controller.validation_failed = False

        controller.handle_sigint(loop)  # Stage 1
        controller.handle_sigint(loop)  # Stage 2
        loop.run_until_complete(asyncio.sleep(0))

        assert controller.abort_exit_code == 130

    def test_abort_exit_code_1_when_validation_failed(
        self, controller: LifecycleController, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Exit code is 1 when validation failed before abort."""
        controller.configure_callbacks(
            get_active_task_count=lambda: 0,
            on_drain_started=MagicMock(),
            on_abort_started=MagicMock(),
            on_force_abort=MagicMock(),
            forward_sigint=MagicMock(),
            kill_processes=MagicMock(),
        )
        controller.validation_failed = True

        controller.handle_sigint(loop)  # Stage 1
        controller.handle_sigint(loop)  # Stage 2
        loop.run_until_complete(asyncio.sleep(0))

        assert controller.abort_exit_code == 1


@pytest.mark.unit
class TestReset:
    """Tests for state reset."""

    def test_reset_clears_all_state(
        self, controller: LifecycleController, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Reset clears all escalation state."""
        controller.configure_callbacks(
            get_active_task_count=lambda: 0,
            on_drain_started=MagicMock(),
            on_abort_started=MagicMock(),
            on_force_abort=MagicMock(),
            forward_sigint=MagicMock(),
            kill_processes=MagicMock(),
        )

        # Escalate to abort mode
        controller.handle_sigint(loop)
        controller.handle_sigint(loop)
        controller.validation_failed = True
        mock_task = MagicMock(spec=asyncio.Task)
        controller.run_task = mock_task
        loop.run_until_complete(asyncio.sleep(0))

        assert controller.is_abort_mode()

        # Reset
        controller.reset()

        # Verify clean state
        assert not controller.is_interrupted()
        assert not controller.is_drain_mode()
        assert not controller.is_abort_mode()
        assert not controller.is_shutdown_requested()
        assert not controller.interrupt_event.is_set()
        assert not controller.drain_event.is_set()
        assert controller.abort_exit_code == 130
        assert not controller.validation_failed
        assert controller.run_task is None

    def test_reset_creates_new_events(self, controller: LifecycleController) -> None:
        """Reset creates fresh event objects."""
        old_interrupt = controller.interrupt_event
        old_drain = controller.drain_event

        controller.reset()

        assert controller.interrupt_event is not old_interrupt
        assert controller.drain_event is not old_drain


@pytest.mark.unit
class TestRequestAbort:
    """Tests for programmatic abort request."""

    def test_request_abort_sets_abort_mode(
        self, controller: LifecycleController
    ) -> None:
        """request_abort sets abort mode and exit code."""
        controller.request_abort("Test reason")

        assert controller.is_abort_mode()
        assert controller.abort_exit_code == 1
        assert controller.is_interrupted()

    def test_request_abort_does_not_require_drain(
        self, controller: LifecycleController
    ) -> None:
        """request_abort can be called without going through drain."""
        assert not controller.is_drain_mode()

        controller.request_abort("Fatal error")

        assert controller.is_abort_mode()
        assert not controller.is_drain_mode()  # Skipped drain


@pytest.mark.unit
class TestCallbackConfiguration:
    """Tests for callback configuration."""

    def test_configure_callbacks(self, controller: LifecycleController) -> None:
        """configure_callbacks sets all callback functions."""
        get_count = MagicMock(return_value=5)
        drain_cb = MagicMock()
        abort_cb = MagicMock()
        force_cb = MagicMock()
        forward_cb = MagicMock()
        kill_cb = MagicMock()

        controller.configure_callbacks(
            get_active_task_count=get_count,
            on_drain_started=drain_cb,
            on_abort_started=abort_cb,
            on_force_abort=force_cb,
            forward_sigint=forward_cb,
            kill_processes=kill_cb,
        )

        assert controller._get_active_task_count is get_count
        assert controller._on_drain_started is drain_cb
        assert controller._on_abort_started is abort_cb
        assert controller._on_force_abort is force_cb
        assert controller._forward_sigint is forward_cb
        assert controller._kill_processes is kill_cb

    def test_handle_sigint_without_callbacks(
        self, controller: LifecycleController, loop: asyncio.AbstractEventLoop
    ) -> None:
        """handle_sigint works without callbacks (no-op for notifications)."""
        # Don't configure callbacks - should still work for state changes
        controller.handle_sigint(loop)
        loop.run_until_complete(asyncio.sleep(0))

        assert controller.is_drain_mode()
        assert controller.drain_event.is_set()

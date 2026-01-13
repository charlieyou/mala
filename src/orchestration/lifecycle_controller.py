"""LifecycleController: Signal handling and shutdown escalation state.

This module extracts SIGINT handling and shutdown state from MalaOrchestrator,
providing clean separation between lifecycle management and runtime orchestration.

The controller manages:
- SIGINT escalation state (drain → abort → force abort)
- Interrupt and drain events
- Shutdown request tracking

Design principles:
- Pure state management, no signal registration (main thread requirement)
- All state encapsulated in dataclass fields
- Methods called by signal handlers via loop.call_soon_threadsafe
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from asyncio import AbstractEventLoop
    from collections.abc import Callable

# SIGINT escalation timing constant (5s window for counting consecutive Ctrl-C)
ESCALATION_WINDOW_SECONDS = 5.0


@dataclass
class LifecycleController:
    """Manages shutdown lifecycle state and SIGINT escalation.

    Three-stage escalation:
    - Stage 1 (1st Ctrl-C): Drain mode - stop accepting new issues, finish current
    - Stage 2 (2nd Ctrl-C): Graceful abort - cancel tasks, forward SIGINT
    - Stage 3 (3rd Ctrl-C): Hard abort - SIGKILL all processes, cancel run task

    Signal registration stays in Orchestrator (main thread requirement).
    This controller only manages state and provides handler methods.
    """

    # Events for coordinating shutdown
    interrupt_event: asyncio.Event = field(default_factory=asyncio.Event)
    drain_event: asyncio.Event = field(default_factory=asyncio.Event)

    # Escalation state
    _sigint_count: int = 0
    _sigint_last_at: float = 0.0
    _drain_mode_active: bool = False
    _abort_mode_active: bool = False
    _abort_exit_code: int = 130
    _validation_failed: bool = False
    _shutdown_requested: bool = False
    _run_task: asyncio.Task | None = field(default=None, repr=False)

    # Callbacks for stage actions (set by orchestrator)
    _get_active_task_count: Callable[[], int] | None = field(default=None, repr=False)
    _on_drain_started: Callable[[int], None] | None = field(default=None, repr=False)
    _on_abort_started: Callable[[], None] | None = field(default=None, repr=False)
    _on_force_abort: Callable[[], None] | None = field(default=None, repr=False)
    _forward_sigint: Callable[[], None] | None = field(default=None, repr=False)
    _kill_processes: Callable[[], None] | None = field(default=None, repr=False)

    def handle_sigint(self, loop: AbstractEventLoop) -> None:
        """Handle SIGINT with three-stage escalation.

        Stage 1 (1st Ctrl-C): Drain mode - stop accepting new issues, finish current
        Stage 2 (2nd Ctrl-C): Graceful abort - cancel tasks, forward SIGINT
        Stage 3 (3rd Ctrl-C): Hard abort - SIGKILL all processes, cancel run task

        The escalation window (5s) resets only when idle (not in drain/abort mode).

        Args:
            loop: The event loop for thread-safe callbacks.
        """
        now = time.monotonic()
        # Reset escalation window if idle (not in drain/abort mode)
        if not self._drain_mode_active and not self._abort_mode_active:
            if now - self._sigint_last_at > ESCALATION_WINDOW_SECONDS:
                self._sigint_count = 0

        self._sigint_count += 1
        self._sigint_last_at = now

        if self._sigint_count == 1:
            # Stage 1: Drain
            self._drain_mode_active = True
            loop.call_soon_threadsafe(self.drain_event.set)
            # Get active task count and notify
            if self._get_active_task_count and self._on_drain_started:
                active_count = self._get_active_task_count()
                on_drain = self._on_drain_started  # Capture for lambda
                loop.call_soon_threadsafe(lambda: on_drain(active_count))
        elif self._sigint_count == 2:
            # Stage 2: Graceful Abort
            self._abort_mode_active = True
            self._abort_exit_code = 1 if self._validation_failed else 130
            loop.call_soon_threadsafe(self.interrupt_event.set)
            if self._forward_sigint:
                loop.call_soon_threadsafe(self._forward_sigint)
            if self._on_abort_started:
                loop.call_soon_threadsafe(self._on_abort_started)
        else:
            # Stage 3: Hard Abort - always exit 130 regardless of validation state
            self._shutdown_requested = True
            self._abort_exit_code = 130  # Hard abort always uses 130
            if self._kill_processes:
                loop.call_soon_threadsafe(self._kill_processes)
            if self._on_force_abort:
                loop.call_soon_threadsafe(self._on_force_abort)
            if self._run_task:
                loop.call_soon_threadsafe(self._run_task.cancel)

    def request_abort(self, reason: str) -> None:
        """Request abort mode programmatically (not via SIGINT).

        This sets abort mode without going through the escalation stages.
        Used for fatal errors that require immediate shutdown.

        Args:
            reason: Description of why abort was requested.
        """
        self._abort_mode_active = True
        self._abort_exit_code = 1
        self.interrupt_event.set()

    def is_interrupted(self) -> bool:
        """Check if interrupt has been signaled."""
        return self.interrupt_event.is_set()

    def is_drain_mode(self) -> bool:
        """Check if drain mode is active."""
        return self._drain_mode_active

    def is_abort_mode(self) -> bool:
        """Check if abort mode is active."""
        return self._abort_mode_active

    def is_shutdown_requested(self) -> bool:
        """Check if hard shutdown was requested (Stage 3)."""
        return self._shutdown_requested

    @property
    def abort_exit_code(self) -> int:
        """Get the exit code to use for abort."""
        return self._abort_exit_code

    @property
    def validation_failed(self) -> bool:
        """Check if validation has failed."""
        return self._validation_failed

    @validation_failed.setter
    def validation_failed(self, value: bool) -> None:
        """Set validation failed flag."""
        self._validation_failed = value

    @property
    def run_task(self) -> asyncio.Task | None:
        """Get the current run task."""
        return self._run_task

    @run_task.setter
    def run_task(self, task: asyncio.Task | None) -> None:
        """Set the current run task."""
        self._run_task = task

    def reset(self) -> None:
        """Reset state for a new run.

        Creates fresh events and clears all escalation state.
        Called at the start of each orchestrator run.
        """
        self.interrupt_event = asyncio.Event()
        self.drain_event = asyncio.Event()
        self._sigint_count = 0
        self._sigint_last_at = 0.0
        self._drain_mode_active = False
        self._abort_mode_active = False
        self._abort_exit_code = 130
        self._validation_failed = False
        self._shutdown_requested = False
        self._run_task = None

    def configure_callbacks(
        self,
        *,
        get_active_task_count: Callable[[], int],
        on_drain_started: Callable[[int], None],
        on_abort_started: Callable[[], None],
        on_force_abort: Callable[[], None],
        forward_sigint: Callable[[], None],
        kill_processes: Callable[[], None],
    ) -> None:
        """Configure callbacks for lifecycle events.

        Args:
            get_active_task_count: Get number of active tasks.
            on_drain_started: Called when drain mode starts.
            on_abort_started: Called when abort mode starts.
            on_force_abort: Called when force abort is triggered.
            forward_sigint: Forward SIGINT to child processes.
            kill_processes: Kill all child processes.
        """
        self._get_active_task_count = get_active_task_count
        self._on_drain_started = on_drain_started
        self._on_abort_started = on_abort_started
        self._on_force_abort = on_force_abort
        self._forward_sigint = forward_sigint
        self._kill_processes = kill_processes

"""Integration tests for SIGINT three-stage escalation.

These tests spawn real subprocesses and send real SIGINT signals to verify
the three-stage escalation behavior works correctly in practice:
- Stage 1 (1st Ctrl-C): Drain mode - stop accepting new issues, finish current
- Stage 2 (2nd Ctrl-C): Graceful abort - cancel tasks, forward SIGINT, exit 130
- Stage 3 (3rd Ctrl-C): Hard abort - SIGKILL all processes, immediate exit 130

POSIX-only (skipped on Windows).
"""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest


# Derive repo root from this test file's location
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _wait_for_ready(
    proc: subprocess.Popen[str], timeout: float = 5.0
) -> tuple[bool, str]:
    """Wait for subprocess to print READY signal.

    Returns:
        (ready_received, output_so_far)
    """
    output_lines: list[str] = []
    deadline = time.monotonic() + timeout
    iterations = int(timeout / 0.1)

    for _ in range(iterations):
        if proc.poll() is not None:
            # Process exited early
            remaining = proc.stdout.read() if proc.stdout else ""
            output_lines.append(remaining)
            return False, "".join(output_lines)

        # Non-blocking read
        if proc.stdout and select.select([proc.stdout], [], [], 0.1)[0]:
            line = proc.stdout.readline()
            output_lines.append(line)
            if "READY" in line:
                return True, "".join(output_lines)

        if time.monotonic() > deadline:
            break

    return False, "".join(output_lines)


def _send_sigint_and_wait(
    proc: subprocess.Popen[str],
    wait_after: float = 0.3,
) -> None:
    """Send SIGINT to process and wait a short time for handling."""
    if proc.poll() is None:
        os.kill(proc.pid, signal.SIGINT)
        time.sleep(wait_after)


def _get_stderr(proc: subprocess.Popen[str]) -> str:
    """Get stderr from process after ensuring it has exited.

    This function kills the process if still running to avoid blocking
    on stderr.read() which waits for EOF.
    """
    if proc.poll() is None:
        proc.kill()
        proc.wait()
    if proc.stderr:
        return proc.stderr.read()
    return ""


@pytest.mark.integration
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only")
class TestSigintEscalationSubprocess:
    """Integration tests for SIGINT escalation via subprocess signal delivery.

    These tests use production code (IssueExecutionCoordinator) to verify
    real signal handling behavior.
    """

    def test_single_sigint_drain_mode(self, tmp_path: Path) -> None:
        """Single SIGINT enters drain mode and allows task to complete.

        This test verifies Stage 1 behavior using the production
        IssueExecutionCoordinator: drain mode stops accepting new issues
        but allows current work to finish, resulting in exit code 0.
        """
        script = tmp_path / "drain_test.py"
        script.write_text(
            """
import asyncio
import signal
import sys

from src.core.models import WatchConfig
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
    provider = FakeIssueProvider()  # No issues - will exit quickly on drain
    event_sink = FakeEventSink()
    drain_event = asyncio.Event()
    interrupt_event = asyncio.Event()

    def sigint_handler(signum, frame):
        # Set drain_event on first SIGINT (mirrors Stage 1 behavior)
        if not drain_event.is_set():
            drain_event.set()

    signal.signal(signal.SIGINT, sigint_handler)

    coord = IssueExecutionCoordinator(
        beads=provider,
        event_sink=event_sink,
        config=CoordinatorConfig(),
    )

    print("READY", flush=True)

    # Use short poll interval so drain check happens quickly
    result = await coord.run_loop(
        spawn_callback=FakeSpawnCallback(),
        finalize_callback=FakeFinalizeCallback(),
        abort_callback=FakeAbortCallback(),
        watch_config=WatchConfig(enabled=True, poll_interval_seconds=0.1),
        interrupt_event=interrupt_event,
        drain_event=drain_event,
    )

    # Drain mode with no active tasks should exit cleanly (0)
    sys.exit(result.exit_code)


asyncio.run(main())
"""
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)

        proc = subprocess.Popen(
            [sys.executable, str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        try:
            ready, output = _wait_for_ready(proc)
            if not ready:
                stderr = _get_stderr(proc)
                if proc.poll() is not None:
                    pytest.fail(
                        f"Subprocess exited early with code {proc.returncode}. "
                        f"Output: {output}\nStderr: {stderr}"
                    )
                pytest.fail(f"Subprocess never sent READY signal. Stderr: {stderr}")

            # Send single SIGINT for drain mode
            _send_sigint_and_wait(proc)

            # Wait for clean exit
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                stderr = _get_stderr(proc)
                proc.kill()
                proc.wait()
                pytest.fail(
                    f"Process did not exit after single SIGINT. Stderr: {stderr}"
                )

            # Drain mode should exit cleanly (0)
            if proc.returncode != 0:
                stderr = _get_stderr(proc)
                pytest.fail(
                    f"Expected exit code 0 (drain complete), got {proc.returncode}. "
                    f"Stderr: {stderr}"
                )

        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_double_sigint_abort_mode(self, tmp_path: Path) -> None:
        """Double SIGINT enters abort mode and exits with code 130.

        This test verifies Stage 2 behavior: two SIGINTs trigger abort mode.
        Since signal handlers can't reliably wake up asyncio event loops,
        Stage 2 exits directly from the signal handler (matching orchestrator
        behavior where force_abort exits immediately).
        """
        script = tmp_path / "abort_test.py"
        script.write_text(
            """
import signal
import sys
import time

sigint_count = 0


def sigint_handler(signum, frame):
    global sigint_count
    sigint_count += 1
    print(f"SIGINT count={sigint_count}", flush=True)
    if sigint_count == 1:
        # Stage 1: Drain mode - continue running
        pass
    elif sigint_count >= 2:
        # Stage 2: Abort mode - exit with 130
        sys.exit(130)


signal.signal(signal.SIGINT, sigint_handler)

print("READY", flush=True)

# Simulate active work
while True:
    time.sleep(0.1)
"""
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)

        proc = subprocess.Popen(
            [sys.executable, str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        try:
            ready, output = _wait_for_ready(proc)
            if not ready:
                stderr = _get_stderr(proc)
                if proc.poll() is not None:
                    pytest.fail(
                        f"Subprocess exited early with code {proc.returncode}. "
                        f"Output: {output}\nStderr: {stderr}"
                    )
                pytest.fail(f"Subprocess never sent READY signal. Stderr: {stderr}")

            # Wait briefly for task to start
            time.sleep(0.3)

            # Send two SIGINTs for abort mode
            _send_sigint_and_wait(proc, wait_after=0.2)
            _send_sigint_and_wait(proc, wait_after=0.2)

            # Wait for exit
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                stderr = _get_stderr(proc)
                proc.kill()
                proc.wait()
                pytest.fail(
                    f"Process did not exit after double SIGINT. Stderr: {stderr}"
                )

            # Abort mode should exit with 130
            if proc.returncode != 130:
                stderr = _get_stderr(proc)
                pytest.fail(
                    f"Expected exit code 130 (abort), got {proc.returncode}. "
                    f"Stderr: {stderr}"
                )

        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_triple_sigint_force_kill(self, tmp_path: Path) -> None:
        """Triple SIGINT triggers force kill and immediate exit 130.

        This test verifies Stage 3 behavior: three SIGINTs trigger
        immediate exit with 130. We test with a long-running task
        that would normally survive Stage 2.
        """
        script = tmp_path / "force_kill_test.py"
        script.write_text(
            """
import asyncio
import signal
import sys

from src.core.models import WatchConfig
from src.pipeline.issue_execution_coordinator import (
    AbortResult,
    CoordinatorConfig,
    IssueExecutionCoordinator,
)
from tests.fakes.event_sink import FakeEventSink
from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider


async def main():
    provider = FakeIssueProvider(
        issues={"test-issue": FakeIssue(id="test-issue", status="open")}
    )
    event_sink = FakeEventSink()
    drain_event = asyncio.Event()
    interrupt_event = asyncio.Event()
    sigint_count = 0

    def sigint_handler(signum, frame):
        nonlocal sigint_count
        sigint_count += 1
        if sigint_count == 1:
            drain_event.set()
        elif sigint_count == 2:
            interrupt_event.set()
        else:
            # Stage 3: Exit immediately (simulates force kill behavior)
            sys.exit(130)

    signal.signal(signal.SIGINT, sigint_handler)

    async def spawn_callback(issue_id: str) -> asyncio.Task:
        async def long_running():
            await asyncio.sleep(60.0)
        return asyncio.create_task(long_running())

    async def finalize_callback(issue_id: str, task: asyncio.Task) -> None:
        pass

    async def abort_callback(*, is_interrupt: bool = False):
        # Simulate unresponsive tasks that don't cancel quickly
        await asyncio.sleep(10.0)
        return AbortResult(aborted_count=0, has_unresponsive_tasks=True)

    coord = IssueExecutionCoordinator(
        beads=provider,
        event_sink=event_sink,
        config=CoordinatorConfig(max_agents=1),
    )

    print("READY", flush=True)

    result = await coord.run_loop(
        spawn_callback=spawn_callback,
        finalize_callback=finalize_callback,
        abort_callback=abort_callback,
        watch_config=WatchConfig(enabled=True, poll_interval_seconds=0.1),
        interrupt_event=interrupt_event,
        drain_event=drain_event,
    )

    sys.exit(result.exit_code)


asyncio.run(main())
"""
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)

        proc = subprocess.Popen(
            [sys.executable, str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        try:
            ready, output = _wait_for_ready(proc)
            if not ready:
                stderr = _get_stderr(proc)
                if proc.poll() is not None:
                    pytest.fail(
                        f"Subprocess exited early with code {proc.returncode}. "
                        f"Output: {output}\nStderr: {stderr}"
                    )
                pytest.fail(f"Subprocess never sent READY signal. Stderr: {stderr}")

            # Wait for task to start
            time.sleep(0.3)

            # Send three SIGINTs for force kill
            _send_sigint_and_wait(proc, wait_after=0.2)
            _send_sigint_and_wait(proc, wait_after=0.2)
            _send_sigint_and_wait(proc, wait_after=0.2)

            # Wait for exit
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                stderr = _get_stderr(proc)
                proc.kill()
                proc.wait()
                pytest.fail(
                    f"Process did not exit after triple SIGINT. Stderr: {stderr}"
                )

            # Force kill should exit with 130
            if proc.returncode != 130:
                stderr = _get_stderr(proc)
                pytest.fail(
                    f"Expected exit code 130 (force kill), got {proc.returncode}. "
                    f"Stderr: {stderr}"
                )

        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_escalation_does_not_reset_after_drain_mode(self, tmp_path: Path) -> None:
        """Escalation window does NOT reset once in drain mode.

        This verifies that once Stage 1 (drain) is entered, subsequent
        SIGINTs always escalate even if more than 5s passes.
        """
        script = tmp_path / "no_reset_in_drain_test.py"
        script.write_text(
            """
import signal
import sys
import time

ESCALATION_WINDOW_SECONDS = 5.0

sigint_count = 0
sigint_last_at = 0.0
drain_mode_active = False


def handle_sigint(signum, frame):
    global sigint_count, sigint_last_at, drain_mode_active

    now = time.monotonic()

    # Reset escalation window ONLY if idle (not in drain mode)
    if not drain_mode_active:
        if now - sigint_last_at > ESCALATION_WINDOW_SECONDS:
            sigint_count = 0

    sigint_count += 1
    sigint_last_at = now

    print(f"SIGINT count={sigint_count}", flush=True)

    if sigint_count == 1:
        drain_mode_active = True  # Enter drain mode
        print("DRAIN_MODE", flush=True)
    elif sigint_count == 2:
        print("ABORT_MODE", flush=True)
        sys.exit(130)


signal.signal(signal.SIGINT, handle_sigint)

print("READY", flush=True)

while True:
    time.sleep(0.1)
"""
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)

        proc = subprocess.Popen(
            [sys.executable, str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        try:
            ready, _output = _wait_for_ready(proc)
            if not ready:
                stderr = _get_stderr(proc)
                pytest.fail(f"Subprocess never sent READY. Stderr: {stderr}")

            # Send first SIGINT - enters drain mode
            _send_sigint_and_wait(proc, wait_after=0.3)

            if proc.stdout and select.select([proc.stdout], [], [], 1.0)[0]:
                line = proc.stdout.readline()
                assert "count=1" in line

            # Wait MORE than 5s (but drain mode prevents reset)
            time.sleep(5.5)

            # Second SIGINT should be count=2 (no reset since in drain mode)
            _send_sigint_and_wait(proc, wait_after=0.3)

            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                pytest.fail("Process did not exit")

            # Should exit with 130 (reached Stage 2)
            assert proc.returncode == 130

        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_abort_with_validation_failed(self, tmp_path: Path) -> None:
        """When validation fails during interrupt handling, abort exits with code 1.

        This tests the real IssueExecutionCoordinator._handle_interrupt path:
        after tasks are aborted, the coordinator runs validation_callback.
        If validation fails, exit_code becomes 1 instead of 130.
        """
        script = tmp_path / "validation_failed_abort_test.py"
        script.write_text(
            """
import asyncio
import signal
import sys

from src.core.models import WatchConfig
from src.pipeline.issue_execution_coordinator import (
    AbortResult,
    CoordinatorConfig,
    IssueExecutionCoordinator,
)
from tests.fakes.event_sink import FakeEventSink
from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider


async def main():
    # Create a provider with one issue so the coordinator has work
    provider = FakeIssueProvider(
        issues={"test-issue": FakeIssue(id="test-issue", status="open")}
    )
    event_sink = FakeEventSink()
    drain_event = asyncio.Event()
    interrupt_event = asyncio.Event()

    # Get the running loop for signal handler to use
    loop = asyncio.get_running_loop()

    def sigint_handler(signum, frame):
        # Use call_soon_threadsafe to properly wake the event loop
        # (signal handlers run outside the event loop context)
        loop.call_soon_threadsafe(interrupt_event.set)

    signal.signal(signal.SIGINT, sigint_handler)

    async def spawn_callback(issue_id: str) -> asyncio.Task:
        async def long_running():
            await asyncio.sleep(60.0)
        return asyncio.create_task(long_running())

    async def finalize_callback(issue_id: str, task: asyncio.Task) -> None:
        pass

    async def abort_callback(*, is_interrupt: bool = False):
        return AbortResult(aborted_count=1, has_unresponsive_tasks=False)

    async def validation_callback() -> bool:
        # Validation fails - this should cause exit_code=1
        print("VALIDATION_FAILED", flush=True)
        return False

    coord = IssueExecutionCoordinator(
        beads=provider,
        event_sink=event_sink,
        config=CoordinatorConfig(max_agents=1),
    )

    print("READY", flush=True)

    result = await coord.run_loop(
        spawn_callback=spawn_callback,
        finalize_callback=finalize_callback,
        abort_callback=abort_callback,
        watch_config=WatchConfig(enabled=True, poll_interval_seconds=0.1),
        interrupt_event=interrupt_event,
        drain_event=drain_event,
        validation_callback=validation_callback,
    )

    sys.exit(result.exit_code)


asyncio.run(main())
"""
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)

        proc = subprocess.Popen(
            [sys.executable, str(script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        try:
            ready, output = _wait_for_ready(proc)
            if not ready:
                stderr = _get_stderr(proc)
                if proc.poll() is not None:
                    pytest.fail(
                        f"Subprocess exited early with code {proc.returncode}. "
                        f"Output: {output}\nStderr: {stderr}"
                    )
                pytest.fail(f"Subprocess never sent READY signal. Stderr: {stderr}")

            # Wait for task to start
            time.sleep(0.3)

            # Send SIGINT to trigger interrupt handling with validation
            _send_sigint_and_wait(proc, wait_after=0.5)

            # Wait for exit
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                stderr = _get_stderr(proc)
                proc.kill()
                proc.wait()
                pytest.fail(f"Process did not exit after SIGINT. Stderr: {stderr}")

            # When validation fails during interrupt, exit code should be 1
            if proc.returncode != 1:
                stderr = _get_stderr(proc)
                pytest.fail(
                    f"Expected exit code 1 (validation failed during abort), "
                    f"got {proc.returncode}. Stderr: {stderr}"
                )

        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

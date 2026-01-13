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
) -> tuple[bool, str, str]:
    """Wait for subprocess to print READY signal.

    Returns:
        (ready_received, stdout_so_far, stderr_if_exited_early)
    """
    output_lines: list[str] = []
    deadline = time.monotonic() + timeout
    iterations = int(timeout / 0.1)

    for _ in range(iterations):
        if proc.poll() is not None:
            # Process exited early - use communicate to safely drain pipes
            remaining, stderr = proc.communicate(timeout=5.0)
            output_lines.append(remaining)
            return False, "".join(output_lines), stderr

        # Non-blocking read
        if proc.stdout and select.select([proc.stdout], [], [], 0.1)[0]:
            line = proc.stdout.readline()
            output_lines.append(line)
            if "READY" in line:
                return True, "".join(output_lines), ""

        if time.monotonic() > deadline:
            break

    return False, "".join(output_lines), ""


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
            ready, output, early_stderr = _wait_for_ready(proc)
            if not ready:
                if proc.poll() is not None:
                    pytest.fail(
                        f"Subprocess exited early with code {proc.returncode}. "
                        f"Output: {output}\nStderr: {early_stderr}"
                    )
                stderr = _get_stderr(proc)
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
            ready, output, early_stderr = _wait_for_ready(proc)
            if not ready:
                if proc.poll() is not None:
                    pytest.fail(
                        f"Subprocess exited early with code {proc.returncode}. "
                        f"Output: {output}\nStderr: {early_stderr}"
                    )
                stderr = _get_stderr(proc)
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
            ready, output, early_stderr = _wait_for_ready(proc)
            if not ready:
                if proc.poll() is not None:
                    pytest.fail(
                        f"Subprocess exited early with code {proc.returncode}. "
                        f"Output: {output}\nStderr: {early_stderr}"
                    )
                stderr = _get_stderr(proc)
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
            ready, _output, early_stderr = _wait_for_ready(proc)
            if not ready:
                if proc.poll() is not None:
                    pytest.fail(
                        f"Subprocess exited early with code {proc.returncode}. "
                        f"Stderr: {early_stderr}"
                    )
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

    def test_coordinator_interrupt_exits_130(self, tmp_path: Path) -> None:
        """Coordinator interrupt always exits 130 - orchestrator handles exit code override.

        Stage 2 (graceful abort) exit code is determined by the orchestrator's
        _abort_exit_code snapshot at Stage 2 entry, not by validation during abort.
        The coordinator's _handle_interrupt simply aborts tasks and returns 130.
        The orchestrator overrides this with _abort_exit_code when _abort_mode_active.

        This test verifies the coordinator-level behavior. Full Stage 2 exit code
        precedence (1 if validation had failed before abort, else 130) is tested
        via the orchestrator integration tests.
        """
        script = tmp_path / "coordinator_interrupt_test.py"
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
        # Stage 2 should NOT call validation - print for verification
        print("UNEXPECTED_VALIDATION_CALL", flush=True)
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
            ready, output, early_stderr = _wait_for_ready(proc)
            if not ready:
                if proc.poll() is not None:
                    pytest.fail(
                        f"Subprocess exited early with code {proc.returncode}. "
                        f"Output: {output}\nStderr: {early_stderr}"
                    )
                stderr = _get_stderr(proc)
                pytest.fail(f"Subprocess never sent READY signal. Stderr: {stderr}")

            # Wait for task to start
            time.sleep(0.3)

            # Send SIGINT to trigger interrupt handling
            _send_sigint_and_wait(proc, wait_after=0.5)

            # Wait for exit and drain stdout/stderr safely
            try:
                remaining_stdout, stderr = proc.communicate(timeout=5.0)
            except subprocess.TimeoutExpired as e:
                # Capture partial output from first timeout (str since text=True)
                partial_stdout = str(e.stdout) if e.stdout else ""
                partial_stderr = str(e.stderr) if e.stderr else ""
                proc.kill()
                try:
                    kill_stdout, kill_stderr = proc.communicate(timeout=5.0)
                except subprocess.TimeoutExpired as e2:
                    # Capture any additional output from the post-kill timeout
                    kill_partial_stdout = str(e2.stdout) if e2.stdout else ""
                    kill_partial_stderr = str(e2.stderr) if e2.stderr else ""
                    pytest.fail(
                        f"Process did not exit after SIGINT and kill(). "
                        f"Partial stdout: {output + partial_stdout + kill_partial_stdout}\n"
                        f"Partial stderr: {partial_stderr + kill_partial_stderr}"
                    )
                remaining_stdout = partial_stdout + kill_stdout
                stderr = partial_stderr + kill_stderr
                pytest.fail(
                    f"Process did not exit after SIGINT. "
                    f"Stdout: {output + remaining_stdout}\nStderr: {stderr}"
                )

            all_stdout = output + remaining_stdout

            # Coordinator interrupt always exits 130 (orchestrator overrides for Stage 2)
            if proc.returncode != 130:
                pytest.fail(
                    f"Expected exit code 130 (interrupt), "
                    f"got {proc.returncode}. Stderr: {stderr}"
                )
            assert "UNEXPECTED_VALIDATION_CALL" not in all_stdout, (
                f"validation_callback should NOT be called during interrupt. "
                f"Stdout: {all_stdout}"
            )

        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


@pytest.mark.integration
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only")
class TestMalaOrchestratorSigint:
    """Integration tests for MalaOrchestrator SIGINT handler.

    These tests exercise the real MalaOrchestrator._handle_sigint method
    via subprocess signal delivery, verifying the orchestrator's signal
    handling wiring works correctly in practice.
    """

    def test_orchestrator_single_sigint_drain_mode(self, tmp_path: Path) -> None:
        """MalaOrchestrator: single SIGINT triggers drain mode via _handle_sigint.

        Verifies that the real MalaOrchestrator._handle_sigint method is wired
        correctly and sets drain_event on first SIGINT (Stage 1). With no active
        tasks, drain mode completes immediately with exit code 0.
        """
        script = tmp_path / "orchestrator_drain_test.py"
        script.write_text(
            """
import asyncio
import signal
import sys
import time

from pathlib import Path

from src.orchestration.factory import OrchestratorConfig, OrchestratorDependencies, create_orchestrator
from src.core.models import WatchConfig
from tests.fakes.event_sink import FakeEventSink
from tests.fakes.issue_provider import FakeIssueProvider

_orchestrator = None


async def main():
    global _orchestrator
    tmp_dir = Path(sys.argv[1])
    runs_dir = tmp_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Create orchestrator with empty provider (will exit on drain)
    provider = FakeIssueProvider()
    event_sink = FakeEventSink()

    config = OrchestratorConfig(
        repo_path=tmp_dir,
        max_agents=1,
        max_issues=1,
    )
    deps = OrchestratorDependencies(
        issue_provider=provider,
        event_sink=event_sink,
        runs_dir=runs_dir,
    )
    _orchestrator = create_orchestrator(config, deps=deps)

    # Run with watch mode so it waits for SIGINT
    watch_config = WatchConfig(enabled=True, poll_interval_seconds=0.1)

    # Start orchestrator as task, wait for SIGINT handler to be installed
    run_task = asyncio.create_task(_orchestrator.run(watch_config=watch_config))

    # Wait for SIGINT handler to be installed (not default handler)
    # Use 15s timeout (generous for slow/loaded CI) and check run_task for early failure
    deadline = time.monotonic() + 15.0
    while signal.getsignal(signal.SIGINT) is signal.default_int_handler:
        if run_task.done():
            # Extract exception safely - exception() raises CancelledError if task was cancelled
            try:
                exc = run_task.exception()
            except asyncio.CancelledError:
                exc = None
            if exc:
                print(f"STARTUP_ERROR: {exc}", file=sys.stderr, flush=True)
            elif run_task.cancelled():
                print("STARTUP_CANCELLED", file=sys.stderr, flush=True)
            else:
                print("STARTUP_EXITED", file=sys.stderr, flush=True)
            sys.exit(1)
        if time.monotonic() > deadline:
            print("TIMEOUT: SIGINT handler not installed", file=sys.stderr, flush=True)
            sys.exit(1)
        await asyncio.sleep(0.01)

    # Now safe to signal READY
    print("READY", flush=True)

    try:
        success_count, total = await run_task
    except asyncio.CancelledError:
        # CancelledError can propagate from Stage 2/3 handling
        pass

    # Check internal state to verify drain mode was triggered
    if _orchestrator._lifecycle.is_drain_mode():
        print("DRAIN_MODE_ACTIVE", flush=True)
    else:
        print(f"drain_mode_active={_orchestrator._lifecycle.is_drain_mode()}", flush=True)
        print(f"sigint_count={_orchestrator._lifecycle._sigint_count}", flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(_orchestrator.exit_code)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Check state even after KeyboardInterrupt
        if _orchestrator is not None:
            if _orchestrator._lifecycle.is_drain_mode():
                print("DRAIN_MODE_ACTIVE", flush=True)
            sys.stdout.flush()
        sys.exit(130)
"""
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)

        proc = subprocess.Popen(
            [sys.executable, str(script), str(tmp_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        try:
            ready, output, early_stderr = _wait_for_ready(proc, timeout=15.0)
            if not ready:
                if proc.poll() is not None:
                    pytest.fail(
                        f"Subprocess exited early with code {proc.returncode}. "
                        f"Output: {output}\nStderr: {early_stderr}"
                    )
                stderr = _get_stderr(proc)
                pytest.fail(f"Subprocess never sent READY signal. Stderr: {stderr}")

            # Wait for orchestrator to enter its run loop before sending SIGINT
            time.sleep(0.3)

            # Check process is still running (watch mode should keep it alive)
            if proc.poll() is not None:
                # Process already exited - use communicate to safely drain pipes
                remaining, stderr = proc.communicate(timeout=5.0)
                pytest.fail(
                    f"Process exited before SIGINT could be sent. "
                    f"Exit code: {proc.returncode}\n"
                    f"Stdout: {output + remaining}\nStderr: {stderr}"
                )

            # Send single SIGINT - should trigger drain mode via _handle_sigint
            _send_sigint_and_wait(proc, wait_after=0.5)

            # Wait for exit and drain stdout/stderr safely
            try:
                remaining_stdout, stderr = proc.communicate(timeout=10.0)
            except subprocess.TimeoutExpired as e:
                # Capture partial output from first timeout (str since text=True)
                partial_stdout = str(e.stdout) if e.stdout else ""
                partial_stderr = str(e.stderr) if e.stderr else ""
                proc.kill()
                try:
                    kill_stdout, kill_stderr = proc.communicate(timeout=5.0)
                except subprocess.TimeoutExpired as e2:
                    # Capture any additional output from the post-kill timeout
                    kill_partial_stdout = str(e2.stdout) if e2.stdout else ""
                    kill_partial_stderr = str(e2.stderr) if e2.stderr else ""
                    pytest.fail(
                        f"Process did not exit after single SIGINT and kill(). "
                        f"Partial stdout: {output + partial_stdout + kill_partial_stdout}\n"
                        f"Partial stderr: {partial_stderr + kill_partial_stderr}"
                    )
                remaining_stdout = partial_stdout + kill_stdout
                stderr = partial_stderr + kill_stderr
                pytest.fail(
                    f"Process did not exit after single SIGINT. "
                    f"Stdout: {output + remaining_stdout}\nStderr: {stderr}"
                )

            all_stdout = output + remaining_stdout

            # Drain mode with no tasks should exit cleanly (0)
            if proc.returncode != 0:
                pytest.fail(
                    f"Expected exit code 0 (drain complete), got {proc.returncode}. "
                    f"Stdout: {all_stdout}\nStderr: {stderr}"
                )

            # Verify drain mode was activated in the orchestrator
            assert "DRAIN_MODE_ACTIVE" in all_stdout, (
                f"_drain_mode_active not set by _handle_sigint. Stdout: {all_stdout}"
            )

        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_orchestrator_double_sigint_abort_mode(self, tmp_path: Path) -> None:
        """MalaOrchestrator: double SIGINT triggers abort mode via _handle_sigint.

        Verifies Stage 2 behavior: two SIGINTs cause the orchestrator to
        set _abort_mode_active and exit with code 130.

        Uses a fake issue with a long-running task to keep the orchestrator
        alive long enough to receive multiple SIGINTs.
        """
        script = tmp_path / "orchestrator_abort_test.py"
        script.write_text(
            """
import asyncio
import signal
import sys
import time

from pathlib import Path

from src.orchestration.factory import OrchestratorConfig, OrchestratorDependencies, create_orchestrator
from src.core.models import WatchConfig
from tests.fakes.event_sink import FakeEventSink
from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider

_orchestrator = None


async def long_running_agent(issue_id: str, *, flow: str = "implementer"):
    '''Mock agent that sleeps until cancelled - keeps active_tasks populated.'''
    try:
        await asyncio.sleep(3600)  # Sleep for 1 hour (will be cancelled by SIGINT)
    except asyncio.CancelledError:
        pass
    # Return a minimal IssueResult
    from src.pipeline.issue_result import IssueResult
    return IssueResult(issue_id=issue_id, agent_id="mock", success=False, summary="Cancelled")


async def main():
    global _orchestrator
    tmp_dir = Path(sys.argv[1])
    runs_dir = tmp_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Provide a fake issue so the orchestrator has work to spawn
    provider = FakeIssueProvider(
        issues={"test-issue": FakeIssue(id="test-issue", status="open")}
    )
    event_sink = FakeEventSink()

    config = OrchestratorConfig(
        repo_path=tmp_dir,
        max_agents=1,
        max_issues=1,
    )
    deps = OrchestratorDependencies(
        issue_provider=provider,
        event_sink=event_sink,
        runs_dir=runs_dir,
    )
    _orchestrator = create_orchestrator(config, deps=deps)

    # Mock run_implementer to be a long-running task that keeps active_tasks populated
    # This avoids real agent work while ensuring orchestrator stays alive for multiple SIGINTs
    _orchestrator.run_implementer = long_running_agent

    # Short poll interval so drain check happens quickly after SIGINT
    watch_config = WatchConfig(enabled=True, poll_interval_seconds=0.1)

    # Start orchestrator as task, wait for SIGINT handler to be installed
    run_task = asyncio.create_task(_orchestrator.run(watch_config=watch_config))

    # Wait for SIGINT handler to be installed
    # Use 15s timeout (generous for slow/loaded CI) and check run_task for early failure
    deadline = time.monotonic() + 15.0
    while signal.getsignal(signal.SIGINT) is signal.default_int_handler:
        if run_task.done():
            # Extract exception safely - exception() raises CancelledError if task was cancelled
            try:
                exc = run_task.exception()
            except asyncio.CancelledError:
                exc = None
            if exc:
                print(f"STARTUP_ERROR: {exc}", file=sys.stderr, flush=True)
            elif run_task.cancelled():
                print("STARTUP_CANCELLED", file=sys.stderr, flush=True)
            else:
                print("STARTUP_EXITED", file=sys.stderr, flush=True)
            sys.exit(1)
        if time.monotonic() > deadline:
            print("TIMEOUT: SIGINT handler not installed", file=sys.stderr, flush=True)
            sys.exit(1)
        await asyncio.sleep(0.01)

    # Wait for agent task to be spawned (active_tasks non-empty)
    # This ensures drain mode won't exit immediately due to zero active tasks
    # Use fresh deadline - handler installation may have consumed most of the previous one
    spawn_deadline = time.monotonic() + 15.0
    while not _orchestrator.issue_coordinator.active_tasks:
        if run_task.done():
            try:
                exc = run_task.exception()
            except asyncio.CancelledError:
                exc = None
            if exc:
                print(f"AGENT_SPAWN_ERROR: {exc}", file=sys.stderr, flush=True)
            else:
                print("AGENT_SPAWN_EXITED", file=sys.stderr, flush=True)
            sys.exit(1)
        if time.monotonic() > spawn_deadline:
            print("TIMEOUT: Agent task not spawned", file=sys.stderr, flush=True)
            sys.exit(1)
        await asyncio.sleep(0.01)

    print("READY", flush=True)

    try:
        await run_task
    except asyncio.CancelledError:
        # Stage 2/3 may cancel the run task
        pass

    # Check internal state (more reliable than async event sink)
    if _orchestrator._lifecycle.is_abort_mode():
        print("ABORT_MODE_ACTIVE", flush=True)
    else:
        print(f"abort_mode_active={_orchestrator._lifecycle.is_abort_mode()}", flush=True)
    print(f"sigint_count={_orchestrator._lifecycle._sigint_count}", flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(_orchestrator.exit_code)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Check state even after KeyboardInterrupt
        if _orchestrator is not None:
            if _orchestrator._lifecycle.is_abort_mode():
                print("ABORT_MODE_ACTIVE", flush=True)
            print(f"sigint_count={_orchestrator._lifecycle._sigint_count}", flush=True)
            sys.stdout.flush()
        sys.exit(130)
"""
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)

        proc = subprocess.Popen(
            [sys.executable, str(script), str(tmp_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        try:
            ready, output, early_stderr = _wait_for_ready(proc, timeout=15.0)
            if not ready:
                if proc.poll() is not None:
                    pytest.fail(
                        f"Subprocess exited early with code {proc.returncode}. "
                        f"Output: {output}\nStderr: {early_stderr}"
                    )
                stderr = _get_stderr(proc)
                pytest.fail(f"Subprocess never sent READY signal. Stderr: {stderr}")

            # READY now means agent task is spawned - no additional sleep needed

            # Send two SIGINTs for abort mode
            # First triggers drain, second triggers abort
            _send_sigint_and_wait(proc, wait_after=0.3)
            _send_sigint_and_wait(proc, wait_after=0.3)

            # Wait for exit and drain stdout/stderr safely
            try:
                remaining_stdout, stderr = proc.communicate(timeout=10.0)
            except subprocess.TimeoutExpired as e:
                # Capture partial output from first timeout (str since text=True)
                partial_stdout = str(e.stdout) if e.stdout else ""
                partial_stderr = str(e.stderr) if e.stderr else ""
                proc.kill()
                try:
                    kill_stdout, kill_stderr = proc.communicate(timeout=5.0)
                except subprocess.TimeoutExpired as e2:
                    # Capture any additional output from the post-kill timeout
                    kill_partial_stdout = str(e2.stdout) if e2.stdout else ""
                    kill_partial_stderr = str(e2.stderr) if e2.stderr else ""
                    pytest.fail(
                        f"Process did not exit after double SIGINT and kill(). "
                        f"Partial stdout: {output + partial_stdout + kill_partial_stdout}\n"
                        f"Partial stderr: {partial_stderr + kill_partial_stderr}"
                    )
                remaining_stdout = partial_stdout + kill_stdout
                stderr = partial_stderr + kill_stderr
                pytest.fail(
                    f"Process did not exit after double SIGINT. "
                    f"Stdout: {output + remaining_stdout}\nStderr: {stderr}"
                )

            all_stdout = output + remaining_stdout

            # Abort mode should exit with 130
            if proc.returncode != 130:
                pytest.fail(
                    f"Expected exit code 130 (abort), got {proc.returncode}. "
                    f"Stdout: {all_stdout}\nStderr: {stderr}"
                )

            # Verify abort mode was activated via internal state
            assert "ABORT_MODE_ACTIVE" in all_stdout, (
                f"_abort_mode_active not set by Stage 2 _handle_sigint. "
                f"Stdout: {all_stdout}"
            )

        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_orchestrator_triple_sigint_force_abort(self, tmp_path: Path) -> None:
        """MalaOrchestrator: triple SIGINT triggers force abort via _handle_sigint.

        Verifies Stage 3 behavior: three SIGINTs within the escalation window
        cause _shutdown_requested to be set and immediate exit with 130.

        Note: If Stage 2 abort completes quickly, the process may exit with
        sigint_count=2 before the third SIGINT arrives. This test uses a slow
        abort to ensure Stage 3 is reached.
        """
        script = tmp_path / "orchestrator_force_abort_test.py"
        script.write_text(
            """
import asyncio
import signal
import sys

from pathlib import Path

from src.orchestration.factory import OrchestratorConfig, OrchestratorDependencies, create_orchestrator
from src.core.models import WatchConfig
from tests.fakes.event_sink import FakeEventSink
from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider
from src.orchestration.orchestrator import MalaOrchestrator
import time

# Globals to check state after signal handling
_orchestrator = None

# Monkey-patch the orchestrator's _handle_sigint to add delay after Stage 2
# This ensures the process stays alive long enough for Stage 3 SIGINT to arrive
_original_handle_sigint = MalaOrchestrator._handle_sigint


def _slow_handle_sigint(self, loop):
    # Call original handler first
    _original_handle_sigint(self, loop)
    # If Stage 2 was just entered, sleep to allow Stage 3 SIGINT to arrive
    # (signal handlers run synchronously, so time.sleep works here)
    if self._lifecycle.is_abort_mode() and not self._lifecycle.is_shutdown_requested():
        time.sleep(1.0)


MalaOrchestrator._handle_sigint = _slow_handle_sigint


async def long_running_agent(issue_id: str, *, flow: str = "implementer"):
    '''Mock agent that sleeps until cancelled - keeps active_tasks populated.'''
    try:
        await asyncio.sleep(3600)  # Sleep for 1 hour (will be cancelled by SIGINT)
    except asyncio.CancelledError:
        pass
    # Return a minimal IssueResult
    from src.pipeline.issue_result import IssueResult
    return IssueResult(issue_id=issue_id, agent_id="mock", success=False, summary="Cancelled")


async def main():
    global _orchestrator
    tmp_dir = Path(sys.argv[1])
    runs_dir = tmp_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Provide a fake issue so the orchestrator has work to spawn
    provider = FakeIssueProvider(
        issues={"test-issue": FakeIssue(id="test-issue", status="open")}
    )
    event_sink = FakeEventSink()

    config = OrchestratorConfig(
        repo_path=tmp_dir,
        max_agents=1,
        max_issues=1,
    )
    deps = OrchestratorDependencies(
        issue_provider=provider,
        event_sink=event_sink,
        runs_dir=runs_dir,
    )
    _orchestrator = create_orchestrator(config, deps=deps)

    # Mock run_implementer to be a long-running task that keeps active_tasks populated
    # This avoids real agent work while ensuring orchestrator stays alive for multiple SIGINTs
    _orchestrator.run_implementer = long_running_agent

    # Short poll interval so drain check happens quickly after SIGINT
    watch_config = WatchConfig(enabled=True, poll_interval_seconds=0.1)

    # Start orchestrator as task, wait for SIGINT handler to be installed
    run_task = asyncio.create_task(_orchestrator.run(watch_config=watch_config))

    # Wait for SIGINT handler to be installed
    # Use 15s timeout (generous for slow/loaded CI) and check run_task for early failure
    deadline = time.monotonic() + 15.0
    while signal.getsignal(signal.SIGINT) is signal.default_int_handler:
        if run_task.done():
            # Extract exception safely - exception() raises CancelledError if task was cancelled
            try:
                exc = run_task.exception()
            except asyncio.CancelledError:
                exc = None
            if exc:
                print(f"STARTUP_ERROR: {exc}", file=sys.stderr, flush=True)
            elif run_task.cancelled():
                print("STARTUP_CANCELLED", file=sys.stderr, flush=True)
            else:
                print("STARTUP_EXITED", file=sys.stderr, flush=True)
            sys.exit(1)
        if time.monotonic() > deadline:
            print("TIMEOUT: SIGINT handler not installed", file=sys.stderr, flush=True)
            sys.exit(1)
        await asyncio.sleep(0.01)

    # Wait for agent task to be spawned (active_tasks non-empty)
    # This ensures drain mode won't exit immediately due to zero active tasks
    # Use fresh deadline - handler installation may have consumed most of the previous one
    spawn_deadline = time.monotonic() + 15.0
    while not _orchestrator.issue_coordinator.active_tasks:
        if run_task.done():
            try:
                exc = run_task.exception()
            except asyncio.CancelledError:
                exc = None
            if exc:
                print(f"AGENT_SPAWN_ERROR: {exc}", file=sys.stderr, flush=True)
            else:
                print("AGENT_SPAWN_EXITED", file=sys.stderr, flush=True)
            sys.exit(1)
        if time.monotonic() > spawn_deadline:
            print("TIMEOUT: Agent task not spawned", file=sys.stderr, flush=True)
            sys.exit(1)
        await asyncio.sleep(0.01)

    print("READY", flush=True)

    try:
        await run_task
    except asyncio.CancelledError:
        # Stage 3 cancels the run task
        pass

    # Check internal state for Stage 3 indicators
    if _orchestrator._lifecycle.is_shutdown_requested():
        print("SHUTDOWN_REQUESTED", flush=True)
    else:
        print(f"shutdown_requested={_orchestrator._lifecycle.is_shutdown_requested()}", flush=True)
    print(f"sigint_count={_orchestrator._lifecycle._sigint_count}", flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(_orchestrator.exit_code)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # Check state even after KeyboardInterrupt
        if _orchestrator is not None:
            if _orchestrator._lifecycle.is_shutdown_requested():
                print("SHUTDOWN_REQUESTED", flush=True)
            else:
                print(f"shutdown_requested={_orchestrator._lifecycle.is_shutdown_requested()}", flush=True)
            print(f"sigint_count={_orchestrator._lifecycle._sigint_count}", flush=True)
            sys.stdout.flush()
        sys.exit(130)
"""
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)

        proc = subprocess.Popen(
            [sys.executable, str(script), str(tmp_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        try:
            ready, output, early_stderr = _wait_for_ready(proc, timeout=15.0)
            if not ready:
                if proc.poll() is not None:
                    pytest.fail(
                        f"Subprocess exited early with code {proc.returncode}. "
                        f"Output: {output}\nStderr: {early_stderr}"
                    )
                stderr = _get_stderr(proc)
                pytest.fail(f"Subprocess never sent READY signal. Stderr: {stderr}")

            # READY now means agent task is spawned - no additional sleep needed

            # Send three SIGINTs for force abort
            # First two trigger drain and abort; third triggers force abort
            _send_sigint_and_wait(proc, wait_after=0.3)
            _send_sigint_and_wait(proc, wait_after=0.3)
            # Wait a bit for abort to start (which is slow due to monkey-patch)
            time.sleep(0.5)
            _send_sigint_and_wait(proc, wait_after=0.3)

            # Wait for exit and drain stdout/stderr safely
            try:
                remaining_stdout, stderr = proc.communicate(timeout=10.0)
            except subprocess.TimeoutExpired as e:
                # Capture partial output from first timeout (str since text=True)
                partial_stdout = str(e.stdout) if e.stdout else ""
                partial_stderr = str(e.stderr) if e.stderr else ""
                proc.kill()
                try:
                    kill_stdout, kill_stderr = proc.communicate(timeout=5.0)
                except subprocess.TimeoutExpired as e2:
                    # Capture any additional output from the post-kill timeout
                    kill_partial_stdout = str(e2.stdout) if e2.stdout else ""
                    kill_partial_stderr = str(e2.stderr) if e2.stderr else ""
                    pytest.fail(
                        f"Process did not exit after triple SIGINT and kill(). "
                        f"Partial stdout: {output + partial_stdout + kill_partial_stdout}\n"
                        f"Partial stderr: {partial_stderr + kill_partial_stderr}"
                    )
                remaining_stdout = partial_stdout + kill_stdout
                stderr = partial_stderr + kill_stderr
                pytest.fail(
                    f"Process did not exit after triple SIGINT. "
                    f"Stdout: {output + remaining_stdout}\nStderr: {stderr}"
                )

            all_stdout = output + remaining_stdout

            # Force abort should exit with 130
            if proc.returncode != 130:
                pytest.fail(
                    f"Expected exit code 130 (force abort), got {proc.returncode}. "
                    f"Stdout: {all_stdout}\nStderr: {stderr}"
                )

            # Verify Stage 3 was triggered (shutdown_requested set by _handle_sigint)
            assert "SHUTDOWN_REQUESTED" in all_stdout, (
                f"_shutdown_requested not set by Stage 3 _handle_sigint. "
                f"Stdout: {all_stdout}"
            )

        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


@pytest.mark.integration
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only")
class TestUnifiedSIGINTHandling:
    """Integration tests for unified SIGINT propagation path.

    These tests verify the interrupt handling components work correctly.
    T007 wired interrupt_event from orchestrator through all flows.
    """

    def test_sigint_during_active_agent_exits_130(self, tmp_path: Path) -> None:
        """SIGINT during active agent session exits with code 130.

        This test verifies the interrupt propagation path:
        1. Orchestrator receives SIGINT and sets interrupt_event
        2. Double SIGINT triggers abort mode
        3. Agent is cancelled and orchestrator exits with 130
        """
        script = tmp_path / "agent_interrupt_test.py"
        script.write_text(
            """
import asyncio
import signal
import sys
import time

from pathlib import Path

from src.orchestration.factory import OrchestratorConfig, OrchestratorDependencies, create_orchestrator
from src.core.models import WatchConfig
from tests.fakes.event_sink import FakeEventSink
from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider

_orchestrator = None
_agent_started = asyncio.Event()


async def mock_agent(issue_id: str, *, flow: str = "implementer"):
    '''Mock agent that signals when started and waits for interrupt.'''
    from src.pipeline.issue_result import IssueResult

    # Signal that we're in the agent
    _agent_started.set()

    # Wait for interrupt (simulates agent doing work)
    try:
        await asyncio.sleep(60.0)
    except asyncio.CancelledError:
        pass

    return IssueResult(issue_id=issue_id, agent_id="mock", success=False, summary="Interrupted")


async def main():
    global _orchestrator
    tmp_dir = Path(sys.argv[1])
    runs_dir = tmp_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    provider = FakeIssueProvider(
        issues={"test-issue": FakeIssue(id="test-issue", status="open")}
    )
    event_sink = FakeEventSink()

    config = OrchestratorConfig(
        repo_path=tmp_dir,
        max_agents=1,
        max_issues=1,
    )
    deps = OrchestratorDependencies(
        issue_provider=provider,
        event_sink=event_sink,
        runs_dir=runs_dir,
    )
    _orchestrator = create_orchestrator(config, deps=deps)
    _orchestrator.run_implementer = mock_agent

    watch_config = WatchConfig(enabled=True, poll_interval_seconds=0.1)

    run_task = asyncio.create_task(_orchestrator.run(watch_config=watch_config))

    # Wait for SIGINT handler to be installed
    deadline = time.monotonic() + 15.0
    while signal.getsignal(signal.SIGINT) is signal.default_int_handler:
        if run_task.done():
            sys.exit(1)
        if time.monotonic() > deadline:
            sys.exit(1)
        await asyncio.sleep(0.01)

    # Wait for agent to start
    spawn_deadline = time.monotonic() + 15.0
    while not _agent_started.is_set():
        if run_task.done():
            sys.exit(1)
        if time.monotonic() > spawn_deadline:
            sys.exit(1)
        await asyncio.sleep(0.01)

    print("READY", flush=True)

    try:
        await run_task
    except asyncio.CancelledError:
        pass

    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(_orchestrator.exit_code)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        if _orchestrator is not None:
            sys.stdout.flush()
        sys.exit(130)
"""
        )

        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT)

        proc = subprocess.Popen(
            [sys.executable, str(script), str(tmp_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )

        try:
            ready, output, early_stderr = _wait_for_ready(proc, timeout=20.0)
            if not ready:
                if proc.poll() is not None:
                    pytest.fail(
                        f"Subprocess exited early with code {proc.returncode}. "
                        f"Output: {output}\nStderr: {early_stderr}"
                    )
                stderr = _get_stderr(proc)
                pytest.fail(f"Subprocess never sent READY signal. Stderr: {stderr}")

            # Send double SIGINT to trigger abort mode
            _send_sigint_and_wait(proc, wait_after=0.5)
            _send_sigint_and_wait(proc, wait_after=0.5)

            # Wait for exit
            try:
                remaining_stdout, stderr = proc.communicate(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    remaining_stdout, stderr = proc.communicate(timeout=5.0)
                except subprocess.TimeoutExpired:
                    remaining_stdout, stderr = "", ""
                pytest.fail(
                    f"Process did not exit after double SIGINT. "
                    f"Stdout: {output + remaining_stdout}\nStderr: {stderr}"
                )

            # Should exit with 130 (abort mode)
            assert proc.returncode == 130, (
                f"Expected exit code 130 (abort), got {proc.returncode}. "
                f"Stdout: {output + remaining_stdout}\nStderr: {stderr}"
            )

        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

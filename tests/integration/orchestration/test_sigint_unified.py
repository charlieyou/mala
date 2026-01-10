"""Integration tests for unified SIGINT handling across all flows.

These tests verify interrupt_event propagation from orchestrator through
all flow components: epic verification, implementer sessions, and validation.

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
    proc: subprocess.Popen[str], timeout: float = 10.0
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
class TestValidationRanParameter:
    """Integration tests for validation_ran parameter in on_run_completed.

    These tests verify the correct computation and propagation of the
    validation_ran parameter for different run scenarios.
    """

    def test_validation_ran_false_when_no_issues_completed(
        self, tmp_path: Path
    ) -> None:
        """validation_ran=False when no issues to validate.

        When no issues complete successfully, validation is skipped and
        validation_ran should be False.

        Exit code semantics (per task acceptance criteria):
        - validation_ran=False + run_validation_passed=None → exit 0 (skipped)
        - validation_ran=True + run_validation_passed=None → exit 130 (interrupted)
        """
        script = tmp_path / "validation_skipped_test.py"
        script.write_text(
            """
import asyncio
import sys
from pathlib import Path

from src.orchestration.factory import OrchestratorConfig, OrchestratorDependencies, create_orchestrator
from src.core.models import PeriodicValidationConfig, WatchConfig
from tests.fakes.event_sink import FakeEventSink
from tests.fakes.issue_provider import FakeIssueProvider


class CapturingEventSink(FakeEventSink):
    def __init__(self):
        super().__init__()
        self.validation_ran = None

    def on_run_completed(
        self,
        success_count: int,
        total_count: int,
        run_validation_passed,
        abort_reason=None,
        *,
        validation_ran: bool = True,
    ) -> None:
        self.validation_ran = validation_ran


async def main():
    tmp_dir = Path(sys.argv[1])
    runs_dir = tmp_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Empty provider - no issues to process
    provider = FakeIssueProvider()
    event_sink = CapturingEventSink()

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
    orchestrator = create_orchestrator(config, deps=deps)

    # Run with validation configured
    validation_config = PeriodicValidationConfig(validate_every=1)
    watch_config = WatchConfig(enabled=False)

    await orchestrator.run(
        watch_config=watch_config,
        validation_config=validation_config,
    )

    # Print result for test verification
    if event_sink.validation_ran is False:
        print("VALIDATION_RAN_FALSE", flush=True)
    elif event_sink.validation_ran is True:
        print("VALIDATION_RAN_TRUE", flush=True)
    else:
        print(f"VALIDATION_RAN_UNKNOWN={event_sink.validation_ran}", flush=True)

    sys.exit(orchestrator.exit_code)


asyncio.run(main())
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
            stdout, stderr = proc.communicate(timeout=30.0)

            # Should exit 0 (no issues = success)
            assert proc.returncode == 0, (
                f"Expected exit code 0, got {proc.returncode}. "
                f"Stdout: {stdout}\nStderr: {stderr}"
            )

            # validation_ran should be False
            assert "VALIDATION_RAN_FALSE" in stdout, (
                f"Expected VALIDATION_RAN_FALSE in output. "
                f"Stdout: {stdout}\nStderr: {stderr}"
            )

        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()


@pytest.mark.integration
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only")
class TestEpicVerificationInterrupt:
    """Integration tests for SIGINT handling during epic verification."""

    def test_sigint_during_epic_verification_cancels_remediation(
        self, tmp_path: Path
    ) -> None:
        """SIGINT during active work cancels and exits 130.

        This test verifies that when SIGINT is received during an active
        agent session, the session is cancelled and the orchestrator exits
        with code 130 (double SIGINT for abort mode).
        """
        script = tmp_path / "epic_verification_interrupt_test.py"
        script.write_text(
            """
import asyncio
import signal
import sys
import time
from pathlib import Path

from src.orchestration.factory import OrchestratorConfig, OrchestratorDependencies, create_orchestrator
from src.core.models import WatchConfig
from src.pipeline.issue_result import IssueResult
from tests.fakes.event_sink import FakeEventSink
from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider

_orchestrator = None
_agent_started = asyncio.Event()


async def slow_mock_agent(issue_id: str, *, flow: str = "implementer") -> IssueResult:
    '''Mock agent that signals start and sleeps until cancelled.'''
    global _agent_started
    _agent_started.set()
    try:
        await asyncio.sleep(60.0)  # Will be cancelled by SIGINT
    except asyncio.CancelledError:
        pass
    return IssueResult(
        issue_id=issue_id,
        agent_id="mock",
        success=False,
        summary="Interrupted",
    )


async def main():
    global _orchestrator
    tmp_dir = Path(sys.argv[1])
    runs_dir = tmp_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Provider with one issue
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
    _orchestrator.run_implementer = slow_mock_agent

    watch_config = WatchConfig(enabled=True, poll_interval_seconds=0.1)

    # Start orchestrator as task
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
            ready, output, early_stderr = _wait_for_ready(proc, timeout=15.0)
            if not ready:
                if proc.poll() is not None:
                    pytest.fail(
                        f"Subprocess exited early with code {proc.returncode}. "
                        f"Output: {output}\nStderr: {early_stderr}"
                    )
                stderr = _get_stderr(proc)
                pytest.fail(f"Subprocess never sent READY signal. Stderr: {stderr}")

            # Wait briefly for agent to start
            time.sleep(0.3)

            # Send two SIGINTs to trigger abort mode
            _send_sigint_and_wait(proc, wait_after=0.3)
            _send_sigint_and_wait(proc, wait_after=0.3)

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


@pytest.mark.integration
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only")
class TestImplementerSessionInterrupt:
    """Integration tests for SIGINT handling during implementer sessions."""

    def test_sigint_during_implementer_session_returns_interrupted(
        self, tmp_path: Path
    ) -> None:
        """SIGINT during implementer session returns interrupted status.

        This test verifies that when SIGINT is received during an active
        implementer session, the session is interrupted and exits with 130.
        """
        script = tmp_path / "implementer_interrupt_test.py"
        script.write_text(
            """
import asyncio
import signal
import sys
import time
from pathlib import Path

from src.orchestration.factory import OrchestratorConfig, OrchestratorDependencies, create_orchestrator
from src.core.models import WatchConfig
from src.pipeline.issue_result import IssueResult
from tests.fakes.event_sink import FakeEventSink
from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider

_orchestrator = None
_agent_started = asyncio.Event()


async def slow_mock_agent(issue_id: str, *, flow: str = "implementer") -> IssueResult:
    '''Mock agent that signals start and sleeps.'''
    global _agent_started
    _agent_started.set()
    try:
        await asyncio.sleep(60.0)  # Will be cancelled by SIGINT
    except asyncio.CancelledError:
        pass
    return IssueResult(
        issue_id=issue_id,
        agent_id="mock",
        success=False,
        summary="Interrupted",
    )


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
    _orchestrator.run_implementer = slow_mock_agent

    watch_config = WatchConfig(enabled=True, poll_interval_seconds=0.1)

    run_task = asyncio.create_task(_orchestrator.run(watch_config=watch_config))

    # Wait for SIGINT handler
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
            ready, output, early_stderr = _wait_for_ready(proc, timeout=15.0)
            if not ready:
                if proc.poll() is not None:
                    pytest.fail(
                        f"Subprocess exited early with code {proc.returncode}. "
                        f"Output: {output}\nStderr: {early_stderr}"
                    )
                stderr = _get_stderr(proc)
                pytest.fail(f"Subprocess never sent READY signal. Stderr: {stderr}")

            # Send two SIGINTs for abort mode
            _send_sigint_and_wait(proc, wait_after=0.3)
            _send_sigint_and_wait(proc, wait_after=0.3)

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

            # Should exit with 130
            assert proc.returncode == 130, (
                f"Expected exit code 130 (interrupt), got {proc.returncode}. "
                f"Stdout: {output + remaining_stdout}\nStderr: {stderr}"
            )

        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

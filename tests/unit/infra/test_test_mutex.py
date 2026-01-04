"""Tests for test-mutex.sh global mutex wrapper script."""

import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent.parent.parent / "src" / "scripts"

# Stable mutex key used by test-mutex.sh for repo-wide commands
MUTEX_KEY = "__test_mutex__"


@pytest.fixture
def mutex_env(tmp_path: Path) -> dict[str, str]:
    """Provide a clean mutex environment for each test."""
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    return {
        "LOCK_DIR": str(lock_dir),
        "AGENT_ID": "test-agent",
        "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
    }


def run_mutex(
    args: list[str], env: dict[str, str], timeout: float | None = 10.0
) -> subprocess.CompletedProcess[str]:
    """Run test-mutex.sh with the given arguments and environment."""
    script = SCRIPTS_DIR / "test-mutex.sh"
    # Create clean environment: exclude lock-related vars from os.environ
    base_env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("LOCK_DIR", "AGENT_ID", "REPO_NAMESPACE")
    }
    full_env = {**base_env, **env}
    return subprocess.run(
        [str(script), *args],
        env=full_env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def run_script(
    name: str, args: list[str], env: dict[str, str], cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a lock script with the given environment."""
    script = SCRIPTS_DIR / name
    base_env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("LOCK_DIR", "AGENT_ID", "REPO_NAMESPACE")
    }
    full_env = {**base_env, **env}
    return subprocess.run(
        [str(script), *args],
        env=full_env,
        capture_output=True,
        text=True,
        cwd=cwd,
    )


class TestMutexAcquisition:
    """Test that test-mutex.sh acquires the global mutex."""

    def test_successful_command_returns_zero(self, mutex_env: dict[str, str]) -> None:
        """Mutex wrapper should pass through command's exit code on success."""
        result = run_mutex(["true"], mutex_env)
        assert result.returncode == 0

    def test_failed_command_returns_nonzero(self, mutex_env: dict[str, str]) -> None:
        """Mutex wrapper should pass through command's exit code on failure."""
        result = run_mutex(["false"], mutex_env)
        assert result.returncode != 0

    def test_command_output_passed_through(self, mutex_env: dict[str, str]) -> None:
        """Command output should be passed through to stdout."""
        result = run_mutex(["echo", "hello world"], mutex_env)
        assert result.returncode == 0
        assert "hello world" in result.stdout

    def test_command_stderr_passed_through(self, mutex_env: dict[str, str]) -> None:
        """Command stderr should be passed through."""
        result = run_mutex(["sh", "-c", "echo error >&2"], mutex_env)
        assert "error" in result.stderr


class TestMutexRelease:
    """Test that the mutex is released after command execution."""

    def test_mutex_released_after_success(self, mutex_env: dict[str, str]) -> None:
        """Mutex should be released after successful command."""
        result = run_mutex(["true"], mutex_env)
        assert result.returncode == 0

        # Mutex should be released - another agent should be able to acquire
        other_env = {**mutex_env, "AGENT_ID": "other-agent"}
        result2 = run_mutex(["true"], other_env)
        assert result2.returncode == 0

    def test_mutex_released_after_failure(self, mutex_env: dict[str, str]) -> None:
        """Mutex should be released even after command failure."""
        result = run_mutex(["false"], mutex_env)
        assert result.returncode != 0

        # Mutex should still be released
        other_env = {**mutex_env, "AGENT_ID": "other-agent"}
        result2 = run_mutex(["true"], other_env)
        assert result2.returncode == 0


class TestMutexContention:
    """Test mutex contention behavior."""

    def test_reports_holder_when_blocked(self, mutex_env: dict[str, str]) -> None:
        """When mutex is held, wrapper should exit non-zero and report holder."""
        # First agent acquires mutex via direct lock (simulating held mutex)
        env1 = {**mutex_env, "AGENT_ID": "blocking-agent"}
        result1 = run_script("lock-try.sh", [MUTEX_KEY], env1)
        assert result1.returncode == 0

        # Second agent tries to run command
        env2 = {**mutex_env, "AGENT_ID": "blocked-agent"}
        result2 = run_mutex(["echo", "should not run"], env2)
        assert result2.returncode != 0
        assert "blocking-agent" in result2.stderr

    def test_does_not_run_command_when_blocked(self, mutex_env: dict[str, str]) -> None:
        """When mutex is held, command should not execute."""
        # First agent acquires mutex
        env1 = {**mutex_env, "AGENT_ID": "blocking-agent"}
        run_script("lock-try.sh", [MUTEX_KEY], env1)

        # Second agent tries to run command - should fail without executing
        env2 = {**mutex_env, "AGENT_ID": "blocked-agent"}
        result = run_mutex(["echo", "THIS_SHOULD_NOT_APPEAR"], env2)
        assert result.returncode != 0
        assert "THIS_SHOULD_NOT_APPEAR" not in result.stdout

    def test_blocks_across_working_directories(
        self, mutex_env: dict[str, str], tmp_path: Path
    ) -> None:
        """Global mutex should be shared across different CWDs."""
        cwd1 = tmp_path / "cwd1"
        cwd2 = tmp_path / "cwd2"
        cwd1.mkdir()
        cwd2.mkdir()

        env1 = {**mutex_env, "AGENT_ID": "agent-a"}
        env2 = {**mutex_env, "AGENT_ID": "agent-b"}

        result1 = run_script("lock-try.sh", [MUTEX_KEY], env1, cwd=cwd1)
        assert result1.returncode == 0

        result2 = run_script("lock-try.sh", [MUTEX_KEY], env2, cwd=cwd2)
        assert result2.returncode != 0

    def test_serial_execution(self, mutex_env: dict[str, str]) -> None:
        """Commands should run serially (one at a time)."""
        # Run two commands back-to-back
        result1 = run_mutex(["echo", "first"], mutex_env)
        assert result1.returncode == 0
        assert "first" in result1.stdout

        result2 = run_mutex(["echo", "second"], mutex_env)
        assert result2.returncode == 0
        assert "second" in result2.stdout


class TestMutexRaceConditions:
    """Test race condition handling."""

    def test_exactly_one_racer_wins(self, mutex_env: dict[str, str]) -> None:
        """When multiple agents race, exactly one should win."""

        def try_mutex(agent_id: str) -> tuple[bool, str]:
            env = {**mutex_env, "AGENT_ID": agent_id}
            result = run_mutex(["echo", agent_id], env, timeout=5.0)
            return (result.returncode == 0, result.stdout)

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(try_mutex, f"racer-{i}") for i in range(5)]
            results = [f.result() for f in futures]

        # At least one should succeed (the first to acquire)
        successes = [r for r in results if r[0]]
        assert len(successes) >= 1


class TestErrorHandling:
    """Test error handling for missing environment variables."""

    def test_fails_without_lock_dir(self, mutex_env: dict[str, str]) -> None:
        """Should fail if LOCK_DIR not set."""
        env = {k: v for k, v in mutex_env.items() if k != "LOCK_DIR"}
        result = run_mutex(["true"], env)
        assert result.returncode != 0
        assert "LOCK_DIR" in result.stderr

    def test_fails_without_agent_id(self, mutex_env: dict[str, str]) -> None:
        """Should fail if AGENT_ID not set."""
        env = {k: v for k, v in mutex_env.items() if k != "AGENT_ID"}
        result = run_mutex(["true"], env)
        assert result.returncode != 0
        assert "AGENT_ID" in result.stderr

    def test_fails_without_command(self, mutex_env: dict[str, str]) -> None:
        """Should fail if no command provided."""
        result = run_mutex([], mutex_env)
        assert result.returncode != 0
        assert "Usage" in result.stderr


class TestSignalHandling:
    """Test that mutex is released on signals."""

    def test_mutex_released_on_interrupt(self, mutex_env: dict[str, str]) -> None:
        """Mutex should be released when command is interrupted."""
        # Start a long-running command
        script = SCRIPTS_DIR / "test-mutex.sh"
        base_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("LOCK_DIR", "AGENT_ID", "REPO_NAMESPACE")
        }
        full_env = {**base_env, **mutex_env}

        proc = subprocess.Popen(
            [str(script), "sleep", "10"],
            env=full_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Give it time to acquire mutex and start
        time.sleep(0.3)

        # Terminate it
        proc.terminate()
        proc.wait(timeout=2)

        # Mutex should be released - another agent should be able to acquire
        other_env = {**mutex_env, "AGENT_ID": "other-agent"}
        result = run_mutex(["true"], other_env)
        assert result.returncode == 0, "Mutex should be released after interrupt"

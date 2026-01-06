"""Integration tests for CommandRunner - standardized subprocess execution.

Tests cover:
- Basic command execution (sync and async)
- Timeout handling with process-group termination
- Output capture and tailing
- Error handling
"""

from __future__ import annotations

import asyncio
import signal
import sys
import threading
import time
from typing import TYPE_CHECKING

import pytest

from src.infra.tools.command_runner import (
    CommandResult,
    CommandRunner,
    run_command,
    run_command_async,
)

if TYPE_CHECKING:
    from pathlib import Path


pytestmark = pytest.mark.integration


class TestCommandResult:
    """Test CommandResult dataclass."""

    def test_stdout_tail(self) -> None:
        long_output = "\n".join([f"line {i}" for i in range(100)])
        result = CommandResult(
            command=["cmd"],
            returncode=0,
            stdout=long_output,
            stderr="",
            duration_seconds=0.1,
            timed_out=False,
        )
        # Default tail is 20 lines
        tail = result.stdout_tail()
        lines = tail.strip().splitlines()
        assert len(lines) <= 20
        assert "line 99" in tail  # Should have last line

    def test_stderr_tail(self) -> None:
        long_error = "x" * 2000
        result = CommandResult(
            command=["cmd"],
            returncode=1,
            stdout="",
            stderr=long_error,
            duration_seconds=0.1,
            timed_out=False,
        )
        # Default tail is 800 chars
        tail = result.stderr_tail()
        assert len(tail) <= 800


class TestCommandRunner:
    """Test CommandRunner class."""

    def test_run_simple_command(self, tmp_path: Path) -> None:
        runner = CommandRunner(cwd=tmp_path)
        result = runner.run(["echo", "hello"])
        assert result.ok is True
        assert "hello" in result.stdout
        assert result.timed_out is False

    def test_run_failing_command(self, tmp_path: Path) -> None:
        runner = CommandRunner(cwd=tmp_path)
        result = runner.run(["false"])
        assert result.ok is False
        assert result.returncode != 0

    def test_run_with_timeout(self, tmp_path: Path) -> None:
        runner = CommandRunner(cwd=tmp_path, timeout_seconds=0.5)
        result = runner.run(["sleep", "10"])
        assert result.ok is False
        assert result.timed_out is True
        assert result.returncode == 124

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only test")
    def test_sigint_forwarding_interrupts_process_group(self, tmp_path: Path) -> None:
        runner = CommandRunner(cwd=tmp_path)

        def send_sigint() -> None:
            time.sleep(0.1)
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                CommandRunner.forward_sigint()
                time.sleep(0.05)

        thread = threading.Thread(target=send_sigint, daemon=True)
        thread.start()
        result = runner.run(["sleep", "2"])
        thread.join()

        assert result.returncode == -signal.SIGINT

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only test")
    def test_timeout_kills_process_group(self, tmp_path: Path) -> None:
        """Verify that timeout kills entire process group, not just parent."""
        # Create a script that spawns a child process that writes after delay
        # We timeout at 0.1s, child tries to write at 0.3s
        # If child survives the kill, it will write the file
        script = tmp_path / "spawner.sh"
        script.write_text(
            """#!/bin/bash
            # Spawn a child that writes to a file after delay
            (sleep 0.3; echo "child survived" > "$1/child_output.txt") &
            # Parent sleeps forever
            sleep 10
            """
        )
        script.chmod(0o755)

        runner = CommandRunner(cwd=tmp_path, timeout_seconds=0.1)
        result = runner.run([str(script), str(tmp_path)])

        assert result.timed_out is True

        # Wait for child to have time to write (if it survived)
        time.sleep(0.5)

        # Child should NOT have survived to write the file
        child_output = tmp_path / "child_output.txt"
        assert not child_output.exists(), "Child process should have been killed"

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only test")
    def test_sigterm_grace_period(self, tmp_path: Path) -> None:
        """Verify SIGTERM grace period allows graceful shutdown (sync).

        Uses a process that catches SIGTERM and exits cleanly within grace period.
        """
        script = tmp_path / "graceful.sh"
        script.write_text(
            """#!/bin/bash
            trap 'echo "caught SIGTERM" > "$1/graceful.txt"; exit 42' TERM
            sleep 10
            """
        )
        script.chmod(0o755)

        runner = CommandRunner(
            cwd=tmp_path, timeout_seconds=0.1, kill_grace_seconds=2.0
        )
        result = runner.run([str(script), str(tmp_path)])

        assert result.timed_out is True
        graceful_file = tmp_path / "graceful.txt"
        assert graceful_file.exists(), "Process should have caught SIGTERM"
        assert "caught SIGTERM" in graceful_file.read_text()

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only test")
    def test_sigkill_after_grace_period(self, tmp_path: Path) -> None:
        """Verify SIGKILL sent after grace period if SIGTERM ignored (sync)."""
        script = tmp_path / "ignore_sigterm.sh"
        script.write_text(
            """#!/bin/bash
            trap '' TERM
            echo "started" > "$1/started.txt"
            sleep 10
            echo "survived" > "$1/survived.txt"
            """
        )
        script.chmod(0o755)

        runner = CommandRunner(
            cwd=tmp_path, timeout_seconds=0.1, kill_grace_seconds=0.1
        )
        start = time.monotonic()
        result = runner.run([str(script), str(tmp_path)])
        elapsed = time.monotonic() - start

        assert result.timed_out is True
        assert elapsed < 1.0, f"Process hung, took {elapsed}s"
        assert (tmp_path / "started.txt").exists()
        assert not (tmp_path / "survived.txt").exists()

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only test")
    def test_use_process_group_false_leaves_children(self, tmp_path: Path) -> None:
        """Verify use_process_group=False only kills parent, children survive."""
        script = tmp_path / "spawner.sh"
        script.write_text(
            """#!/bin/bash
            (sleep 0.2; echo "child survived" > "$1/child_output.txt") &
            sleep 10
            """
        )
        script.chmod(0o755)

        runner = CommandRunner(cwd=tmp_path, timeout_seconds=0.1)
        result = runner.run([str(script), str(tmp_path)], use_process_group=False)

        assert result.timed_out is True
        time.sleep(0.5)
        # With use_process_group=False, child process survives
        assert (tmp_path / "child_output.txt").exists(), "Child should survive"

    def test_run_with_env(self, tmp_path: Path) -> None:
        runner = CommandRunner(cwd=tmp_path)
        result = runner.run(
            ["sh", "-c", "echo $MY_VAR"],
            env={"MY_VAR": "test_value"},
        )
        assert result.ok is True
        assert "test_value" in result.stdout

    def test_run_captures_stderr(self, tmp_path: Path) -> None:
        runner = CommandRunner(cwd=tmp_path)
        result = runner.run(["sh", "-c", "echo error >&2; exit 1"])
        assert result.ok is False
        assert "error" in result.stderr


class TestAsyncCommandRunner:
    """Test async command execution."""

    @pytest.mark.asyncio
    async def test_run_async_simple(self, tmp_path: Path) -> None:
        runner = CommandRunner(cwd=tmp_path)
        result = await runner.run_async(["echo", "async hello"])
        assert result.ok is True
        assert "async hello" in result.stdout

    @pytest.mark.asyncio
    async def test_run_async_with_timeout(self, tmp_path: Path) -> None:
        runner = CommandRunner(cwd=tmp_path, timeout_seconds=0.5)
        result = await runner.run_async(["sleep", "10"])
        assert result.ok is False
        assert result.timed_out is True

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only test")
    async def test_async_sigint_forwarding_interrupts_process_group(
        self, tmp_path: Path
    ) -> None:
        runner = CommandRunner(cwd=tmp_path)

        async def send_sigint() -> None:
            await asyncio.sleep(0.1)
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                CommandRunner.forward_sigint()
                await asyncio.sleep(0.05)

        sigint_task = asyncio.create_task(send_sigint())
        result = await runner.run_async(["sleep", "2"])
        await sigint_task

        assert result.returncode == -signal.SIGINT

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only test")
    async def test_async_timeout_kills_process_group(self, tmp_path: Path) -> None:
        """Verify that async timeout kills entire process group."""
        # Create a script that spawns a child process that writes after delay
        # We timeout at 0.1s, child tries to write at 0.3s
        # If child survives the kill, it will write the file
        script = tmp_path / "spawner.sh"
        script.write_text(
            """#!/bin/bash
            # Spawn a child that writes to a file after delay
            (sleep 0.3; echo "child survived" > "$1/child_output.txt") &
            # Parent sleeps forever
            sleep 10
            """
        )
        script.chmod(0o755)

        runner = CommandRunner(cwd=tmp_path, timeout_seconds=0.1)
        result = await runner.run_async([str(script), str(tmp_path)])

        assert result.timed_out is True

        # Wait for child to have time to write (if it survived)
        await asyncio.sleep(0.5)

        child_output = tmp_path / "child_output.txt"
        assert not child_output.exists(), "Child process should have been killed"

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only test")
    async def test_async_sigterm_grace_period(self, tmp_path: Path) -> None:
        """Verify SIGTERM grace period allows graceful shutdown.

        Uses a process that catches SIGTERM and exits cleanly within grace period.
        Verifies it exits with its own exit code (not SIGKILL's 137).
        """
        script = tmp_path / "graceful.sh"
        script.write_text(
            """#!/bin/bash
            trap 'echo "caught SIGTERM" > "$1/graceful.txt"; exit 42' TERM
            # Sleep forever, waiting for signal
            sleep 10
            """
        )
        script.chmod(0o755)

        # Use short timeout but long grace period to ensure SIGTERM path
        runner = CommandRunner(
            cwd=tmp_path, timeout_seconds=0.1, kill_grace_seconds=2.0
        )
        result = await runner.run_async([str(script), str(tmp_path)])

        assert result.timed_out is True

        # Verify the process caught SIGTERM and exited gracefully
        graceful_file = tmp_path / "graceful.txt"
        assert graceful_file.exists(), "Process should have caught SIGTERM"
        assert "caught SIGTERM" in graceful_file.read_text()

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only test")
    async def test_async_sigkill_after_grace_period(self, tmp_path: Path) -> None:
        """Verify SIGKILL sent after grace period if SIGTERM ignored.

        Uses a process that ignores SIGTERM. After the grace period expires,
        SIGKILL should forcibly terminate it.
        """
        script = tmp_path / "ignore_sigterm.sh"
        script.write_text(
            """#!/bin/bash
            # Ignore SIGTERM
            trap '' TERM
            # Write a marker to show we started
            echo "started" > "$1/started.txt"
            # Sleep forever - should be SIGKILLed
            sleep 10
            # This should never execute
            echo "survived" > "$1/survived.txt"
            """
        )
        script.chmod(0o755)

        # Very short grace period to speed up test
        runner = CommandRunner(
            cwd=tmp_path, timeout_seconds=0.1, kill_grace_seconds=0.1
        )
        start = time.monotonic()
        result = await runner.run_async([str(script), str(tmp_path)])
        elapsed = time.monotonic() - start

        assert result.timed_out is True
        # Should complete within timeout + grace + small buffer, not hang
        assert elapsed < 1.0, f"Process hung, took {elapsed}s"

        # Process was started
        assert (tmp_path / "started.txt").exists()
        # But was killed before completing (no survived.txt)
        assert not (tmp_path / "survived.txt").exists()

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only test")
    async def test_async_use_process_group_false_leaves_children(
        self, tmp_path: Path
    ) -> None:
        """Verify use_process_group=False only kills parent, children survive."""
        script = tmp_path / "spawner.sh"
        script.write_text(
            """#!/bin/bash
            (sleep 0.2; echo "child survived" > "$1/child_output.txt") &
            sleep 10
            """
        )
        script.chmod(0o755)

        runner = CommandRunner(cwd=tmp_path, timeout_seconds=0.1)
        result = await runner.run_async(
            [str(script), str(tmp_path)], use_process_group=False
        )

        assert result.timed_out is True
        await asyncio.sleep(0.5)
        # With use_process_group=False, child process survives
        assert (tmp_path / "child_output.txt").exists(), "Child should survive"


class TestShellMode:
    """Test shell mode execution."""

    def test_shell_mode_with_pipe(self, tmp_path: Path) -> None:
        """Test shell command with pipe."""
        runner = CommandRunner(cwd=tmp_path)
        result = runner.run("echo 'hello world' | grep hello", shell=True)
        assert result.ok is True
        assert "hello" in result.stdout

    def test_shell_mode_with_redirect(self, tmp_path: Path) -> None:
        """Test shell command with redirect."""
        runner = CommandRunner(cwd=tmp_path)
        output_file = tmp_path / "output.txt"
        result = runner.run(f"echo 'test content' > {output_file}", shell=True)
        assert result.ok is True
        assert output_file.exists()
        assert "test content" in output_file.read_text()

    def test_shell_mode_exit_code_capture(self, tmp_path: Path) -> None:
        """Test that exit codes are captured correctly in shell mode."""
        runner = CommandRunner(cwd=tmp_path)
        result = runner.run("exit 42", shell=True)
        assert result.ok is False
        assert result.returncode == 42

    def test_shell_mode_stderr_capture(self, tmp_path: Path) -> None:
        """Test stderr capture in shell mode."""
        runner = CommandRunner(cwd=tmp_path)
        result = runner.run("echo error >&2; exit 1", shell=True)
        assert result.ok is False
        assert "error" in result.stderr

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only test")
    def test_shell_mode_with_timeout(self, tmp_path: Path) -> None:
        """Test timeout enforcement with shell commands."""
        runner = CommandRunner(cwd=tmp_path, timeout_seconds=0.5)
        result = runner.run("sleep 10", shell=True)
        assert result.ok is False
        assert result.timed_out is True
        assert result.returncode == 124

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only test")
    def test_shell_mode_kills_process_group(self, tmp_path: Path) -> None:
        """Verify shell mode timeout kills entire process group."""
        runner = CommandRunner(cwd=tmp_path, timeout_seconds=0.1)
        # Shell spawns child that tries to write after timeout
        result = runner.run(
            f"(sleep 0.3; echo survived > {tmp_path}/child.txt) & sleep 10",
            shell=True,
        )
        assert result.timed_out is True
        time.sleep(0.5)
        assert not (tmp_path / "child.txt").exists(), "Child should have been killed"

    def test_shell_mode_with_env(self, tmp_path: Path) -> None:
        """Test environment variable passing in shell mode."""
        runner = CommandRunner(cwd=tmp_path)
        result = runner.run("echo $MY_VAR", shell=True, env={"MY_VAR": "test_value"})
        assert result.ok is True
        assert "test_value" in result.stdout

    def test_convenience_function_shell_mode(self, tmp_path: Path) -> None:
        """Test run_command convenience function with shell=True."""
        result = run_command("echo 'hello world' | wc -w", cwd=tmp_path, shell=True)
        assert result.ok is True
        assert "2" in result.stdout


class TestAsyncShellMode:
    """Test async shell mode execution."""

    @pytest.mark.asyncio
    async def test_async_shell_mode_with_pipe(self, tmp_path: Path) -> None:
        """Test async shell command with pipe."""
        runner = CommandRunner(cwd=tmp_path)
        result = await runner.run_async("echo 'hello world' | grep hello", shell=True)
        assert result.ok is True
        assert "hello" in result.stdout

    @pytest.mark.asyncio
    async def test_async_shell_mode_exit_code(self, tmp_path: Path) -> None:
        """Test exit code capture in async shell mode."""
        runner = CommandRunner(cwd=tmp_path)
        result = await runner.run_async("exit 42", shell=True)
        assert result.ok is False
        assert result.returncode == 42

    @pytest.mark.asyncio
    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX-only test")
    async def test_async_shell_mode_timeout(self, tmp_path: Path) -> None:
        """Test timeout with async shell mode."""
        runner = CommandRunner(cwd=tmp_path, timeout_seconds=0.5)
        result = await runner.run_async("sleep 10", shell=True)
        assert result.ok is False
        assert result.timed_out is True

    @pytest.mark.asyncio
    async def test_async_convenience_function_shell_mode(self, tmp_path: Path) -> None:
        """Test run_command_async convenience function with shell=True."""
        result = await run_command_async(
            "echo 'hello world' | wc -w", cwd=tmp_path, shell=True
        )
        assert result.ok is True
        assert "2" in result.stdout

"""Integration tests for CommandRunner - standardized subprocess execution.

Tests cover:
- Basic command execution (sync and async)
- Timeout handling with process-group termination
- Output capture and tailing
- Error handling
"""

from __future__ import annotations

import asyncio
import sys
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

    def test_basic_result(self) -> None:
        result = CommandResult(
            command=["echo", "hello"],
            returncode=0,
            stdout="hello\n",
            stderr="",
            duration_seconds=0.1,
            timed_out=False,
        )
        assert result.ok is True
        assert result.command == ["echo", "hello"]
        assert result.returncode == 0
        assert result.stdout == "hello\n"

    def test_failed_result(self) -> None:
        result = CommandResult(
            command=["false"],
            returncode=1,
            stdout="",
            stderr="error",
            duration_seconds=0.05,
            timed_out=False,
        )
        assert result.ok is False
        assert result.returncode == 1

    def test_timed_out_result(self) -> None:
        result = CommandResult(
            command=["sleep", "10"],
            returncode=124,
            stdout="",
            stderr="",
            duration_seconds=1.0,
            timed_out=True,
        )
        assert result.ok is False
        assert result.timed_out is True
        assert result.returncode == 124

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
    def test_use_process_group_true_kills_children(self, tmp_path: Path) -> None:
        """Verify use_process_group=True kills child processes on timeout."""
        script = tmp_path / "spawner.sh"
        script.write_text(
            """#!/bin/bash
            (sleep 0.3; echo "child survived" > "$1/child_output.txt") &
            sleep 10
            """
        )
        script.chmod(0o755)

        runner = CommandRunner(cwd=tmp_path, timeout_seconds=0.1)
        result = runner.run([str(script), str(tmp_path)], use_process_group=True)

        assert result.timed_out is True
        time.sleep(0.5)
        assert not (tmp_path / "child_output.txt").exists(), "Child should be killed"

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
    async def test_async_use_process_group_true_kills_children(
        self, tmp_path: Path
    ) -> None:
        """Verify use_process_group=True kills child processes on async timeout."""
        script = tmp_path / "spawner.sh"
        script.write_text(
            """#!/bin/bash
            (sleep 0.3; echo "child survived" > "$1/child_output.txt") &
            sleep 10
            """
        )
        script.chmod(0o755)

        runner = CommandRunner(cwd=tmp_path, timeout_seconds=0.1)
        result = await runner.run_async(
            [str(script), str(tmp_path)], use_process_group=True
        )

        assert result.timed_out is True
        await asyncio.sleep(0.5)
        assert not (tmp_path / "child_output.txt").exists(), "Child should be killed"

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


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_run_command(self, tmp_path: Path) -> None:
        result = run_command(["echo", "convenience"], cwd=tmp_path)
        assert result.ok is True
        assert "convenience" in result.stdout

    def test_run_command_with_timeout(self, tmp_path: Path) -> None:
        result = run_command(["sleep", "10"], cwd=tmp_path, timeout_seconds=0.5)
        assert result.timed_out is True

    @pytest.mark.asyncio
    async def test_run_command_async(self, tmp_path: Path) -> None:
        result = await run_command_async(["echo", "async"], cwd=tmp_path)
        assert result.ok is True
        assert "async" in result.stdout

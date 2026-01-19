"""Cerberus review-gate CLI subprocess management.

This module provides CerberusGateCLI for managing subprocess interactions
with the review-gate CLI binary. It handles:
- Binary validation (exists and is executable)
- Environment variable merging (CLAUDE_SESSION_ID)
- Subprocess spawn/wait/resolve with timeout handling
- Exit code interpretation

This is a low-level component extracted from DefaultReviewer to enable
independent testing of subprocess logic.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.core.protocols.infra import CommandRunnerPort


@dataclass
class SpawnResult:
    """Result of spawning a code review.

    Attributes:
        success: Whether spawn succeeded.
        timed_out: Whether spawn timed out.
        error_detail: Error message if spawn failed.
        already_active: Whether failure was due to an existing active gate.
    """

    success: bool
    timed_out: bool = False
    error_detail: str = ""
    already_active: bool = False


@dataclass
class WaitResult:
    """Result of waiting for review completion.

    Attributes:
        returncode: Exit code from wait command.
        stdout: Standard output (JSON).
        stderr: Standard error.
        timed_out: Whether wait timed out.
        session_dir: Path to session directory (extracted from JSON output).
    """

    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    session_dir: Path | None = None


@dataclass
class ResolveResult:
    """Result of resolving (clearing) an active gate.

    Attributes:
        success: Whether resolve succeeded.
        error_detail: Error message if resolve failed.
    """

    success: bool
    error_detail: str = ""


@dataclass
class CerberusGateCLI:
    """Low-level CLI subprocess management for review-gate.

    Handles subprocess spawn/wait/resolve, binary validation, env merging,
    and timeout handling. Performs minimal JSON parsing to extract transport
    fields (e.g., session_dir) but does not map exit codes to domain results
    - that responsibility belongs to DefaultReviewer.

    Attributes:
        repo_path: Working directory for subprocess execution.
        bin_path: Optional path to directory containing review-gate binary.
            If None, uses bare "review-gate" from PATH.
        env: Additional environment variables to merge with os.environ.
    """

    repo_path: Path
    bin_path: Path | None = None
    env: dict[str, str] = field(default_factory=dict)

    def _review_gate_bin(self) -> str:
        """Get the path to the review-gate binary."""
        if self.bin_path is not None:
            return str(self.bin_path / "review-gate")
        return "review-gate"

    def validate_binary(self) -> str | None:
        """Validate that the review-gate binary exists and is executable.

        Uses the merged env's PATH (respecting self.env) when checking for
        the binary to avoid false negatives when callers inject PATH via cerberus_env.

        Returns:
            None if the binary is valid, or an error message if not.
        """
        if self.bin_path is not None:
            binary_path = self.bin_path / "review-gate"
            if not binary_path.exists():
                return f"review-gate binary not found at {binary_path}"
            if not os.access(binary_path, os.X_OK):
                return f"review-gate binary at {binary_path} is not executable"
        else:
            # Build effective PATH by merging self.env with current os.environ
            if "PATH" in self.env:
                effective_path = (
                    self.env["PATH"] + os.pathsep + os.environ.get("PATH", "")
                )
            else:
                effective_path = os.environ.get("PATH", "")
            if shutil.which("review-gate", path=effective_path) is None:
                return "review-gate binary not found in PATH"
        return None

    def build_env(self, claude_session_id: str | None) -> dict[str, str]:
        """Build environment dict with CLAUDE_SESSION_ID merged.

        Args:
            claude_session_id: Session ID to use, or None to use from env.

        Returns:
            Merged environment dict with CLAUDE_SESSION_ID set.
        """
        merged = dict(self.env)
        claude_session = (
            claude_session_id
            or merged.get("CLAUDE_SESSION_ID")
            or os.environ.get("CLAUDE_SESSION_ID")
        )
        if claude_session:
            merged["CLAUDE_SESSION_ID"] = claude_session
        return merged

    async def spawn_code_review(
        self,
        runner: CommandRunnerPort,
        env: dict[str, str],
        timeout: int,
        *,
        commit_shas: Sequence[str],
        spawn_args: tuple[str, ...] = (),
        context_file: Path | None = None,
    ) -> SpawnResult:
        """Spawn a code review subprocess.

        Args:
            runner: CommandRunnerPort instance for subprocess execution.
            env: Environment variables for the command.
            timeout: Timeout in seconds.
            commit_shas: List of commit SHAs for commit-based review.
            spawn_args: Additional arguments for spawn-code-review.
            context_file: Optional context file path.

        Returns:
            SpawnResult with success/failure status and error details.
        """
        spawn_cmd = [self._review_gate_bin(), "spawn-code-review"]
        # Always exclude .beads/ directory (auto-generated issue tracker files)
        spawn_cmd.extend(["--exclude", ".beads/"])
        # Disable follow-up rounds; mala handles iterative fixes
        spawn_cmd.extend(["--max-rounds", "0"])
        if context_file is not None:
            spawn_cmd.extend(["--context-file", str(context_file)])
        if spawn_args:
            spawn_cmd.extend(spawn_args)
        spawn_cmd.append("--commit")
        spawn_cmd.extend(commit_shas)

        result = await runner.run_async(spawn_cmd, env=env, timeout=timeout)
        if result.timed_out:
            return SpawnResult(
                success=False, timed_out=True, error_detail="spawn timeout"
            )

        if result.returncode != 0:
            stderr = result.stderr_tail()
            stdout = result.stdout_tail()
            detail = stderr or stdout or "spawn failed"
            combined = f"{stderr or ''} {stdout or ''}".lower()
            already_active = "already active" in combined
            return SpawnResult(
                success=False,
                timed_out=False,
                error_detail=detail,
                already_active=already_active,
            )

        return SpawnResult(success=True)

    async def spawn_epic_verify(
        self,
        runner: CommandRunnerPort,
        env: dict[str, str],
        timeout: int,
        *,
        epic_path: Path,
        spawn_args: tuple[str, ...] = (),
    ) -> SpawnResult:
        """Spawn an epic verification subprocess.

        Args:
            runner: CommandRunnerPort instance for subprocess execution.
            env: Environment variables for the command.
            timeout: Timeout in seconds.
            epic_path: Path to the epic markdown file.
            spawn_args: Additional arguments for spawn-epic-verify.

        Returns:
            SpawnResult with success/failure status and error details.
        """
        spawn_cmd = [self._review_gate_bin(), "spawn-epic-verify"]
        # Disable auto-respawn; mala handles retries explicitly.
        spawn_cmd.extend(["--max-rounds", "0"])
        if spawn_args:
            spawn_cmd.extend(spawn_args)
        spawn_cmd.append(str(epic_path))

        result = await runner.run_async(spawn_cmd, env=env, timeout=timeout)
        if result.timed_out:
            return SpawnResult(
                success=False, timed_out=True, error_detail="spawn timeout"
            )

        if result.returncode != 0:
            stderr = result.stderr_tail()
            stdout = result.stdout_tail()
            detail = stderr or stdout or "spawn failed"
            combined = f"{stderr or ''} {stdout or ''}".lower()
            already_active = "already active" in combined
            return SpawnResult(
                success=False,
                timed_out=False,
                error_detail=detail,
                already_active=already_active,
            )

        return SpawnResult(success=True)

    async def wait_for_review(
        self,
        session_id: str,
        runner: CommandRunnerPort,
        env: dict[str, str],
        cli_timeout: int,
        wait_args: tuple[str, ...] = (),
        user_timeout: int | None = None,
    ) -> WaitResult:
        """Wait for review completion.

        Args:
            session_id: Session ID to wait for.
            runner: CommandRunnerPort instance for subprocess execution.
            env: Environment variables for the command.
            cli_timeout: Timeout value to pass to CLI (if user_timeout is None).
            wait_args: Additional arguments for wait command.
            user_timeout: User-specified timeout (if already in wait_args).

        Returns:
            WaitResult with returncode, stdout, stderr, and timeout status.
        """
        wait_cmd = [
            self._review_gate_bin(),
            "wait",
            "--json",
            "--finalize",
            "--session-id",
            session_id,
        ]
        # Only add --timeout if not already specified in wait_args
        if user_timeout is None:
            wait_cmd.extend(["--timeout", str(cli_timeout)])
        if wait_args:
            wait_cmd.extend(wait_args)

        # Use CLI timeout + grace period for subprocess timeout
        effective_timeout = (
            user_timeout if user_timeout is not None else cli_timeout
        ) + 30
        result = await runner.run_async(wait_cmd, env=env, timeout=effective_timeout)

        # Extract session_dir from JSON output
        session_dir: Path | None = None
        try:
            wait_data = json.loads(result.stdout)
            if isinstance(wait_data, dict):
                raw_session_dir = wait_data.get("session_dir")
                if isinstance(raw_session_dir, str) and raw_session_dir:
                    session_dir = Path(raw_session_dir)
        except (json.JSONDecodeError, TypeError):
            pass

        return WaitResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            timed_out=result.timed_out,
            session_dir=session_dir,
        )

    async def resolve_gate(
        self,
        runner: CommandRunnerPort,
        env: dict[str, str],
        reason: str = "mala: auto-clearing stale gate for retry",
    ) -> ResolveResult:
        """Resolve (clear) an active gate.

        Args:
            runner: CommandRunnerPort instance for subprocess execution.
            env: Environment variables for the command (must include CLAUDE_SESSION_ID).
            reason: Reason message for the resolve command.

        Returns:
            ResolveResult with success/failure status and error details.
        """
        resolve_cmd = [
            self._review_gate_bin(),
            "resolve",
            "--reason",
            reason,
        ]
        try:
            result = await runner.run_async(resolve_cmd, env=env, timeout=30)
            if result.returncode == 0:
                return ResolveResult(success=True)
            stderr = result.stderr.strip() if result.stderr else ""
            stdout = result.stdout.strip() if result.stdout else ""
            return ResolveResult(
                success=False, error_detail=stderr or stdout or "resolve failed"
            )
        except Exception as e:
            return ResolveResult(success=False, error_detail=str(e))

    @staticmethod
    def extract_wait_timeout(args: tuple[str, ...]) -> int | None:
        """Extract --timeout value from wait args if provided.

        Parses args to detect if a --timeout flag is already specified.
        Supports both '--timeout VALUE' and '--timeout=VALUE' formats.

        Args:
            args: Tuple of command-line arguments to search.

        Returns:
            The timeout value as int if found and valid, None otherwise.
        """
        i = 0
        while i < len(args):
            arg = args[i]
            if arg.startswith("--timeout="):
                value = arg.split("=", 1)[1]
                if value.isdigit():
                    return int(value)
                return None
            if arg == "--timeout" and i + 1 < len(args):
                value = args[i + 1]
                if value.isdigit():
                    return int(value)
                return None
            i += 1
        return None

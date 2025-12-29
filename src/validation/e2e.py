"""E2E fixture runner for mala validation.

This module provides end-to-end validation using a fixture repository.
It creates a temporary repo with a known bug, runs mala to fix it,
and validates the result.

Key types:
- E2EResult: Result of an E2E validation run
- E2EConfig: Configuration for E2E validation
- E2ERunner: Orchestrates the E2E validation flow
"""

from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from src.tools.command_runner import CommandRunner
from .helpers import (
    annotate_issue,
    get_ready_issue_id,
    init_fixture_repo,
    write_fixture_repo,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


class E2EStatus(Enum):
    """Status of E2E validation."""

    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"  # Skipped due to missing prerequisites


@dataclass
class E2EPrereqResult:
    """Result of prerequisite check.

    Attributes:
        ok: Whether all prerequisites are met.
        missing: List of missing prerequisites.
        can_skip: Whether E2E can be skipped rather than failed.
    """

    ok: bool
    missing: list[str] = field(default_factory=list)
    can_skip: bool = False

    def failure_reason(self) -> str | None:
        """Return failure reason string, or None if ok."""
        if self.ok:
            return None
        if not self.missing:
            return "E2E prerequisites not met"
        return f"E2E prereq missing: {', '.join(self.missing)}"


@dataclass
class E2EResult:
    """Result of an E2E validation run.

    Attributes:
        passed: Whether E2E validation passed.
        status: Status code for the E2E run.
        failure_reason: Explanation for failure (None if passed).
        fixture_path: Path to the fixture repo (if created).
        duration_seconds: How long the E2E run took.
        command_output: Output from the mala command (truncated).
        returncode: Exit code from the mala command.
    """

    passed: bool
    status: E2EStatus
    failure_reason: str | None = None
    fixture_path: Path | None = None
    duration_seconds: float = 0.0
    command_output: str = ""
    returncode: int = 0

    def short_summary(self) -> str:
        """One-line summary for logs/prompts."""
        if self.status == E2EStatus.SKIPPED:
            return f"E2E skipped: {self.failure_reason or 'prerequisites not met'}"
        if self.passed:
            return "E2E passed"
        return f"E2E failed: {self.failure_reason or 'unknown error'}"


@dataclass
class E2EConfig:
    """Configuration for E2E validation.

    Attributes:
        enabled: Whether E2E is enabled.
        skip_if_no_keys: Deprecated, kept for backward compatibility.
            MORPH_API_KEY is no longer a hard prereq for E2E validation.
        keep_fixture: Keep fixture repo after completion (for debugging).
        timeout_seconds: Timeout for the mala run command (default 300s/5min).
        max_agents: Maximum agents for the mala run.
        max_issues: Maximum issues to process in the mala run.
    """

    enabled: bool = True
    skip_if_no_keys: bool = False
    keep_fixture: bool = False
    timeout_seconds: float = 300.0
    max_agents: int = 1
    max_issues: int = 1


class E2ERunner:
    """Orchestrates E2E validation using a fixture repository."""

    def __init__(self, config: E2EConfig | None = None):
        """Initialize the E2E runner.

        Args:
            config: E2E configuration. Uses defaults if None.
        """
        self.config = config or E2EConfig()

    def check_prereqs(self, env: Mapping[str, str] | None = None) -> E2EPrereqResult:
        """Check if all E2E prerequisites are met.

        Args:
            env: Environment variables to check. Uses os.environ if None.

        Returns:
            E2EPrereqResult with details about missing prerequisites.

        Note:
            MORPH_API_KEY is NOT checked here - it's not a hard prerequisite
            for E2E validation. Morph-specific tests will skip when the key
            is missing, but the overall E2E validation can still run.
        """
        import os

        if env is None:
            env = os.environ

        missing: list[str] = []

        # Check for mala CLI
        if not shutil.which("mala"):
            missing.append("mala CLI not found in PATH")

        # Check for bd CLI
        if not shutil.which("bd"):
            missing.append("bd CLI not found in PATH")

        # Note: MORPH_API_KEY is intentionally NOT checked here.
        # E2E validation should not fail just because the key is missing.
        # Morph-specific tests will skip when the key is absent.

        if missing:
            return E2EPrereqResult(ok=False, missing=missing, can_skip=False)

        return E2EPrereqResult(ok=True)

    def run(
        self, env: Mapping[str, str] | None = None, cwd: Path | None = None
    ) -> E2EResult:
        """Run E2E validation.

        Creates a fixture repo, runs mala on it, and validates the result.
        Cleans up the fixture repo unless keep_fixture is True.

        Args:
            env: Environment variables for subprocess. Uses os.environ if None.
            cwd: Working directory for mala command. Uses current directory if None.

        Returns:
            E2EResult with details about the validation.
        """
        import os

        if env is None:
            env = dict(os.environ)
        else:
            env = dict(env)

        if cwd is None:
            cwd = Path.cwd()

        # Check prerequisites
        prereq = self.check_prereqs(env)
        if not prereq.ok:
            if prereq.can_skip:
                return E2EResult(
                    passed=True,  # Skipped is considered "not failed"
                    status=E2EStatus.SKIPPED,
                    failure_reason=prereq.failure_reason(),
                )
            return E2EResult(
                passed=False,
                status=E2EStatus.FAILED,
                failure_reason=prereq.failure_reason(),
            )

        # Create fixture repo
        fixture_path = Path(tempfile.mkdtemp(prefix="mala-e2e-fixture-"))
        start_time = time.monotonic()

        try:
            # Write fixture files
            setup_error = self._setup_fixture(fixture_path)
            if setup_error:
                duration = time.monotonic() - start_time
                return E2EResult(
                    passed=False,
                    status=E2EStatus.FAILED,
                    failure_reason=setup_error,
                    fixture_path=fixture_path if self.config.keep_fixture else None,
                    duration_seconds=duration,
                )

            # Run mala
            result = self._run_mala(fixture_path, env, cwd)
            result.fixture_path = fixture_path if self.config.keep_fixture else None

            return result

        finally:
            duration = time.monotonic() - start_time
            # Cleanup fixture unless keeping it
            if not self.config.keep_fixture and fixture_path.exists():
                shutil.rmtree(fixture_path, ignore_errors=True)

    def _setup_fixture(self, repo_path: Path) -> str | None:
        """Set up the fixture repository.

        Args:
            repo_path: Path to create the fixture repo in.

        Returns:
            Error message if setup failed, None on success.
        """
        # Write fixture files using shared helper
        write_fixture_repo(repo_path)

        # Initialize git and beads using shared helper
        return init_fixture_repo(repo_path)

    def _run_mala(
        self, fixture_path: Path, env: Mapping[str, str], cwd: Path
    ) -> E2EResult:
        """Run mala on the fixture repo.

        Args:
            fixture_path: Path to the fixture repository.
            env: Environment variables for subprocess.
            cwd: Working directory for the mala command.

        Returns:
            E2EResult with command execution details.
        """
        # Annotate the issue with context using shared helper
        issue_id = get_ready_issue_id(fixture_path)
        if issue_id:
            annotate_issue(fixture_path, issue_id)

        cmd = [
            "mala",
            "run",
            str(fixture_path),
            "--max-agents",
            str(self.config.max_agents),
            "--max-issues",
            str(self.config.max_issues),
            "--timeout",
            str(int(self.config.timeout_seconds)),
        ]

        # Use CommandRunner with buffer for cleanup time
        runner = CommandRunner(
            cwd=cwd, timeout_seconds=self.config.timeout_seconds + 30
        )
        result = runner.run(cmd, env=dict(env))

        if result.ok:
            return E2EResult(
                passed=True,
                status=E2EStatus.PASSED,
                duration_seconds=result.duration_seconds,
                command_output=result.stdout_tail(),
                returncode=0,
            )

        if result.timed_out:
            return E2EResult(
                passed=False,
                status=E2EStatus.FAILED,
                failure_reason=f"mala timed out after {self.config.timeout_seconds}s",
                duration_seconds=result.duration_seconds,
                command_output=result.stderr_tail() or result.stdout_tail(),
                returncode=124,
            )

        output = result.stderr_tail() or result.stdout_tail()
        return E2EResult(
            passed=False,
            status=E2EStatus.FAILED,
            failure_reason=f"mala exited {result.returncode}: {output}",
            duration_seconds=result.duration_seconds,
            command_output=output,
            returncode=result.returncode,
        )


# For backwards compatibility, export the prereq checker with the old name
def check_e2e_prereqs(env: Mapping[str, str]) -> str | None:
    """Check E2E prerequisites (legacy interface).

    Args:
        env: Environment variables to check.

    Returns:
        Error message if prerequisites not met, None if all ok.
    """
    runner = E2ERunner()
    result = runner.check_prereqs(env)
    return result.failure_reason()

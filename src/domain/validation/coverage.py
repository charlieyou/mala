"""Coverage parsing and threshold handling for mala validation.

This module provides:
- CoverageResult: result of parsing a coverage report
- parse_coverage_xml: parse coverage.xml and return CoverageResult
- check_coverage_threshold: compare coverage against minimum threshold
- get_baseline_coverage: extract coverage percentage from existing baseline file
- is_baseline_stale: check if baseline file is older than last commit or repo is dirty
- BaselineCoverageService: service for refreshing baseline coverage in isolated worktree
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from .config import YamlCoverageConfig  # noqa: TC001 - used at runtime
from .coverage_args import rewrite_coverage_command

if TYPE_CHECKING:
    from collections.abc import Iterator

    from src.core.protocols.infra import (
        CommandRunnerPort,
        EnvConfigPort,
        LockManagerPort,
    )

    from .spec import ValidationSpec


def _infer_coverage_base_command(original_cmd: list[str]) -> list[str]:
    """Infer the coverage base command from the original coverage command.

    Determines whether to use 'coverage', 'uv run coverage', or 'python -m coverage'
    based on how the original command invoked pytest.

    Args:
        original_cmd: The original coverage command (e.g., ['uv', 'run', 'pytest', ...])

    Returns:
        Base command for running coverage subcommands like 'combine' or 'xml'.
    """
    if len(original_cmd) >= 3 and original_cmd[0] == "uv" and original_cmd[1] == "run":
        return ["uv", "run", "coverage"]
    elif (
        len(original_cmd) >= 3
        and original_cmd[1] == "-m"
        and original_cmd[2] == "pytest"
    ):
        return [original_cmd[0], "-m", "coverage"]
    else:
        return ["coverage"]


class CoverageStatus(Enum):
    """Status of coverage parsing/validation."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    PARSED = "parsed"  # Successfully parsed, but threshold not yet checked


@dataclass(frozen=True)
class CoverageResult:
    """Result of parsing and validating a coverage report.

    Attributes:
        percent: Coverage percentage (0.0-100.0), or None if parsing failed.
        passed: Whether coverage meets the threshold (False until threshold checked).
        status: Status of the coverage check.
        report_path: Path to the coverage report file.
        failure_reason: Explanation for failure/error (None if passed).
        line_rate: Raw line rate from XML (0.0-1.0), or None if unavailable.
        branch_rate: Raw branch rate from XML (0.0-1.0), or None if unavailable.
    """

    percent: float | None
    passed: bool
    status: CoverageStatus
    report_path: Path | None
    failure_reason: str | None = None
    line_rate: float | None = None
    branch_rate: float | None = None

    def short_summary(self) -> str:
        """One-line summary for logs/prompts."""
        if self.passed:
            return f"coverage {self.percent:.1f}% passed"
        if self.failure_reason:
            return self.failure_reason
        if self.status == CoverageStatus.PARSED:
            return f"coverage {self.percent:.1f}% (threshold not checked)"
        return f"coverage {self.percent:.1f}% failed"


def parse_coverage_xml(report_path: Path) -> CoverageResult:
    """Parse a coverage.xml file and extract coverage metrics.

    Note: This function returns status=PARSED with passed=False. Callers must
    use check_coverage_threshold() to determine if coverage meets requirements.

    Args:
        report_path: Path to the coverage.xml file.

    Returns:
        CoverageResult with parsed metrics (status=PARSED) or error information.
    """
    if not report_path.exists():
        return CoverageResult(
            percent=None,
            passed=False,
            status=CoverageStatus.ERROR,
            report_path=report_path,
            failure_reason=f"Coverage report not found: {report_path}",
        )

    try:
        tree = ET.parse(report_path)
    except ET.ParseError as e:
        return CoverageResult(
            percent=None,
            passed=False,
            status=CoverageStatus.ERROR,
            report_path=report_path,
            failure_reason=f"Invalid coverage XML: {e}",
        )
    except OSError as e:
        return CoverageResult(
            percent=None,
            passed=False,
            status=CoverageStatus.ERROR,
            report_path=report_path,
            failure_reason=f"Cannot read coverage report: {e}",
        )

    root = tree.getroot()

    # Check for expected root element
    if root.tag != "coverage":
        return CoverageResult(
            percent=None,
            passed=False,
            status=CoverageStatus.ERROR,
            report_path=report_path,
            failure_reason=f"Invalid coverage XML: expected <coverage> root, got <{root.tag}>",
        )

    # Extract line-rate and branch-rate from coverage element
    line_rate_str = root.get("line-rate")
    branch_rate_str = root.get("branch-rate")

    if line_rate_str is None:
        return CoverageResult(
            percent=None,
            passed=False,
            status=CoverageStatus.ERROR,
            report_path=report_path,
            failure_reason="Invalid coverage XML: missing line-rate attribute",
        )

    try:
        line_rate = float(line_rate_str)
    except ValueError:
        return CoverageResult(
            percent=None,
            passed=False,
            status=CoverageStatus.ERROR,
            report_path=report_path,
            failure_reason=f"Invalid coverage XML: line-rate '{line_rate_str}' is not a number",
        )

    branch_rate: float | None = None
    if branch_rate_str is not None:
        try:
            branch_rate = float(branch_rate_str)
        except ValueError:
            pass  # Branch rate is optional, ignore parse errors

    # Convert line rate (0.0-1.0) to percentage (0.0-100.0)
    percent = line_rate * 100.0

    return CoverageResult(
        percent=percent,
        passed=False,  # Must call check_coverage_threshold to set passed=True
        status=CoverageStatus.PARSED,
        report_path=report_path,
        line_rate=line_rate,
        branch_rate=branch_rate,
    )


def check_coverage_threshold(
    result: CoverageResult,
    min_percent: float | None,
) -> CoverageResult:
    """Check if coverage meets the minimum threshold.

    Args:
        result: A CoverageResult from parse_coverage_xml.
        min_percent: Minimum required coverage percentage (0.0-100.0), or None
            to skip threshold checking (always passes).

    Returns:
        A new CoverageResult with passed/status updated based on threshold.
    """
    # If parsing failed, return as-is
    if result.status == CoverageStatus.ERROR or result.percent is None:
        return result

    # If no threshold specified, consider it passed
    if min_percent is None:
        return CoverageResult(
            percent=result.percent,
            passed=True,
            status=CoverageStatus.PASSED,
            report_path=result.report_path,
            failure_reason=None,
            line_rate=result.line_rate,
            branch_rate=result.branch_rate,
        )

    # Use small epsilon for floating-point comparison to avoid precision issues
    # where coverage like 88.79999 fails against threshold 88.8 even though
    # they display as the same value
    epsilon = 1e-9
    passed = result.percent >= min_percent - epsilon

    if passed:
        return CoverageResult(
            percent=result.percent,
            passed=True,
            status=CoverageStatus.PASSED,
            report_path=result.report_path,
            failure_reason=None,
            line_rate=result.line_rate,
            branch_rate=result.branch_rate,
        )

    return CoverageResult(
        percent=result.percent,
        passed=False,
        status=CoverageStatus.FAILED,
        report_path=result.report_path,
        failure_reason=f"Coverage {result.percent:.1f}% is below threshold {min_percent:.1f}%",
        line_rate=result.line_rate,
        branch_rate=result.branch_rate,
    )


def parse_and_check_coverage(
    report_path: Path,
    min_percent: float | None,
) -> CoverageResult:
    """Parse coverage XML and check against threshold in one call.

    This is a convenience function that combines parse_coverage_xml
    and check_coverage_threshold.

    Args:
        report_path: Path to the coverage.xml file.
        min_percent: Minimum required coverage percentage (0.0-100.0), or None
            to skip threshold checking (always passes).

    Returns:
        CoverageResult with parsing and threshold check results.
    """
    result = parse_coverage_xml(report_path)
    return check_coverage_threshold(result, min_percent)


def check_coverage_from_config(
    coverage_config: YamlCoverageConfig | None,
    cwd: Path,
) -> CoverageResult | None:
    """Check coverage using YamlCoverageConfig settings.

    This is the primary entry point for config-driven coverage checking.
    It uses the config's file path and threshold to perform the check.

    Args:
        coverage_config: Coverage configuration from mala.yaml, or None to skip.
        cwd: Working directory to resolve relative paths against.

    Returns:
        CoverageResult if coverage_config is provided, None if coverage is disabled.
        When the coverage file is missing, returns CoverageResult with ERROR status.
    """
    if coverage_config is None:
        return None

    # Resolve file path against cwd
    report_path = Path(coverage_config.file)
    if not report_path.is_absolute():
        report_path = cwd / report_path

    # Check for missing coverage file
    if not report_path.exists():
        return CoverageResult(
            percent=None,
            passed=False,
            status=CoverageStatus.ERROR,
            report_path=report_path,
            failure_reason=f"Coverage report not found: {report_path}",
        )

    return parse_and_check_coverage(report_path, coverage_config.threshold)


def get_baseline_coverage(report_path: Path) -> float | None:
    """Extract coverage percentage from a baseline coverage report.

    This function is used to read a previously saved coverage baseline file
    to get the minimum coverage threshold for "no decrease" checking.

    Args:
        report_path: Path to the coverage.xml baseline file.

    Returns:
        Coverage percentage (0.0-100.0) if file exists and is valid, None if
        the file is missing.

    Raises:
        ValueError: If the file exists but cannot be parsed (malformed XML,
            missing required attributes, etc.).
    """
    if not report_path.exists():
        return None

    result = parse_coverage_xml(report_path)

    if result.status == CoverageStatus.ERROR:
        raise ValueError(result.failure_reason)

    return result.percent


def is_baseline_stale(
    report_path: Path,
    repo_path: Path,
    command_runner: CommandRunnerPort,
) -> bool:
    """Check if the coverage baseline file is stale and needs refresh.

    A baseline is considered stale if:
    - The baseline file doesn't exist
    - The repo has uncommitted changes (dirty working tree)
    - The baseline file's mtime is older than the last commit time
    - Git commands fail (non-git repo or git errors)

    Args:
        report_path: Path to the coverage.xml baseline file.
        repo_path: Path to the git repository root.
        command_runner: CommandRunnerPort for running git commands.

    Returns:
        True if baseline is stale or doesn't exist, False if baseline is fresh.
    """
    # Missing baseline is considered stale
    if not report_path.exists():
        return True

    runner = command_runner

    try:
        # Check for dirty working tree
        dirty_result = runner.run(["git", "status", "--porcelain"], cwd=repo_path)
        if not dirty_result.ok:
            # Git command failed - treat as stale
            return True
        if dirty_result.stdout.strip():
            # Has uncommitted changes
            return True

        # Get last commit timestamp (Unix epoch seconds)
        commit_time_result = runner.run(
            ["git", "log", "-1", "--format=%ct"], cwd=repo_path
        )
        if not commit_time_result.ok:
            # Git command failed - treat as stale
            return True
        commit_time_str = commit_time_result.stdout.strip()
        if not commit_time_str:
            # No commits in repo
            return True

        commit_time = int(commit_time_str)

        # Get baseline file mtime
        baseline_mtime = report_path.stat().st_mtime

        # Stale if baseline is older than last commit
        return baseline_mtime < commit_time

    except (ValueError, OSError):
        # Path error or parse error - treat as stale
        return True


# Lock file path for baseline refresh coordination
_BASELINE_LOCK_FILE = "coverage-baseline.lock"


@dataclass
class WorktreeRefreshContext:
    """Context for baseline refresh worktree lifecycle.

    Holds the worktree path, command runner, and environment for running
    commands in an isolated worktree during baseline coverage refresh.
    """

    worktree_path: Path
    runner: CommandRunnerPort
    env: dict[str, str]


@contextmanager
def baseline_worktree(
    repo_path: Path,
    timeout: float,
    lock_dir: Path,
    command_runner: CommandRunnerPort,
) -> Iterator[WorktreeRefreshContext]:
    """Create and manage a temporary worktree for baseline coverage refresh.

    Args:
        repo_path: Path to the main repository.
        timeout: Timeout in seconds for commands.
        lock_dir: Lock directory to set in environment.
        command_runner: CommandRunnerPort for executing commands.

    Yields:
        WorktreeRefreshContext with worktree path, runner, and environment.

    Raises:
        RuntimeError: If worktree creation fails.
    """
    from .worktree import (
        WorktreeConfig,
        WorktreeState,
        create_worktree,
        remove_worktree,
    )

    run_id = f"baseline-{uuid.uuid4().hex[:8]}"
    temp_dir = Path(tempfile.mkdtemp(prefix="mala-baseline-"))
    worktree_config = WorktreeConfig(
        base_dir=temp_dir,
        keep_on_failure=False,
    )

    worktree_ctx = None

    try:
        worktree_ctx = create_worktree(
            repo_path=repo_path,
            commit_sha="HEAD",
            config=worktree_config,
            run_id=run_id,
            issue_id="baseline",
            attempt=1,
            command_runner=command_runner,
        )

        if worktree_ctx.state == WorktreeState.FAILED:
            raise RuntimeError(
                f"Baseline worktree creation failed: {worktree_ctx.error}"
            )

        worktree_path = worktree_ctx.path

        # Build environment
        env = {
            **os.environ,
            "AGENT_ID": f"baseline-{run_id}",
            "LOCK_DIR": str(lock_dir),
        }

        yield WorktreeRefreshContext(
            worktree_path=worktree_path,
            runner=command_runner,
            env=env,
        )
    finally:
        # Clean up temp worktree
        if worktree_ctx is not None:
            remove_worktree(
                worktree_ctx, validation_passed=True, command_runner=command_runner
            )
        # Clean up temp directory
        shutil.rmtree(temp_dir, ignore_errors=True)


@dataclass
class BaselineRefreshResult:
    """Result of a baseline coverage refresh operation.

    Attributes:
        percent: The baseline coverage percentage if successful.
        success: Whether the refresh succeeded.
        error: Error message if refresh failed.
    """

    percent: float | None
    success: bool
    error: str | None = None

    @staticmethod
    def ok(percent: float) -> BaselineRefreshResult:
        """Create a successful result."""
        return BaselineRefreshResult(percent=percent, success=True)

    @staticmethod
    def fail(error: str) -> BaselineRefreshResult:
        """Create a failed result."""
        return BaselineRefreshResult(percent=None, success=False, error=error)


class BaselineCoverageService:
    """Service for refreshing baseline coverage in an isolated worktree.

    This service handles:
    - File-based locking to prevent concurrent refreshes
    - Temporary worktree creation at HEAD
    - Running coverage command to generate baseline
    - Copying the coverage report back to the main repo

    Usage:
        config = YamlCoverageConfig(command="uv run pytest --cov", ...)
        service = BaselineCoverageService(repo_path, coverage_config=config)
        result = service.refresh_if_stale(spec)
        if result.success:
            baseline_percent = result.percent

    Note:
        If coverage_config is None or coverage_config.command is None,
        baseline refresh is unavailable and refresh_if_stale will return
        a failure result.
    """

    def __init__(
        self,
        repo_path: Path,
        env_config: EnvConfigPort,
        command_runner: CommandRunnerPort,
        lock_manager: LockManagerPort,
        coverage_config: YamlCoverageConfig | None = None,
        step_timeout_seconds: float | None = None,
    ):
        """Initialize the baseline coverage service.

        Args:
            repo_path: Path to the repository.
            env_config: Environment configuration for paths (lock_dir, etc.).
            command_runner: CommandRunnerPort for running commands.
            lock_manager: LockManagerPort for file locking.
            coverage_config: Coverage configuration from mala.yaml. Required for
                baseline refresh - if None or if command is None, refresh is unavailable.
            step_timeout_seconds: Optional fallback timeout for commands (used if
                coverage_config.timeout is None).
        """
        self.repo_path = repo_path.resolve()
        self.coverage_config = coverage_config
        self.step_timeout_seconds = step_timeout_seconds
        self.env_config = env_config
        self.command_runner = command_runner
        self.lock_manager = lock_manager

    def refresh_if_stale(
        self,
        spec: ValidationSpec,
    ) -> BaselineRefreshResult:
        """Refresh the baseline coverage if stale or missing.

        Uses file locking with double-check pattern to prevent concurrent
        agents from clobbering each other's baseline refresh.

        Args:
            spec: Validation spec with pytest command and coverage config.

        Returns:
            BaselineRefreshResult with the baseline percentage or error.
            Returns failure if coverage_config is None or has no command.
        """
        lock_mgr = self.lock_manager

        # Check if baseline refresh is available
        if self.coverage_config is None:
            return BaselineRefreshResult.fail(
                "Baseline refresh unavailable: no coverage configuration"
            )
        if self.coverage_config.command is None:
            return BaselineRefreshResult.fail(
                "Baseline refresh unavailable: no coverage command configured"
            )

        # Determine baseline report path from config
        coverage_file = Path(self.coverage_config.file)
        if coverage_file.is_absolute():
            baseline_path = coverage_file
        else:
            baseline_path = self.repo_path / coverage_file

        # Check if baseline is fresh (no refresh needed)
        if not is_baseline_stale(
            baseline_path, self.repo_path, command_runner=self.command_runner
        ):
            try:
                baseline = get_baseline_coverage(baseline_path)
                if baseline is not None:
                    return BaselineRefreshResult.ok(baseline)
            except ValueError:
                # Malformed baseline - need to refresh
                pass

        # Baseline is stale or missing - try to acquire lock for refresh
        run_id = f"baseline-{uuid.uuid4().hex[:8]}"
        agent_id = f"baseline-refresh-{run_id}"
        repo_namespace = str(self.repo_path)

        # Try to acquire lock (non-blocking first)
        if not lock_mgr.try_lock(_BASELINE_LOCK_FILE, agent_id, repo_namespace):
            # Another agent is refreshing - wait for them
            if not lock_mgr.wait_for_lock(
                _BASELINE_LOCK_FILE,
                agent_id,
                repo_namespace,
                timeout_seconds=300.0,  # 5 min max wait
                poll_interval_ms=1000,
            ):
                return BaselineRefreshResult.fail(
                    "Timeout waiting for baseline refresh lock"
                )

        # Lock acquired - double-check if still stale (another agent may have refreshed)
        try:
            if not is_baseline_stale(
                baseline_path, self.repo_path, command_runner=self.command_runner
            ):
                try:
                    baseline = get_baseline_coverage(baseline_path)
                    if baseline is not None:
                        return BaselineRefreshResult.ok(baseline)
                except ValueError:
                    pass  # Still need to refresh

            # Still stale - run refresh in temp worktree
            return self._run_refresh(spec, baseline_path)
        finally:
            # Release lock through the abstraction
            lock_mgr.release_lock(_BASELINE_LOCK_FILE, agent_id, repo_namespace)

    def _run_coverage_with_fallback(
        self,
        runner: CommandRunnerPort,
        coverage_cmd: list[str],
        coverage_file: Path,
        worktree_path: Path,
        env: dict[str, str],
        timeout: float,
    ) -> Path | str:
        """Run coverage command and fallback to combine if XML not generated.

        Args:
            runner: Command runner for executing coverage commands.
            coverage_cmd: The rewritten coverage command to run.
            coverage_file: Path to expected coverage XML file (relative to worktree).
            worktree_path: Path to the worktree directory.
            env: Environment variables for command execution.
            timeout: Timeout in seconds for each command.

        Returns:
            Path to coverage XML on success, or error string on failure.
        """
        # Run coverage command - we ignore the exit code because tests may fail
        # but still generate a valid coverage.xml baseline
        coverage_result = runner.run(
            coverage_cmd, env=env, cwd=worktree_path, timeout=timeout
        )

        worktree_coverage = worktree_path / coverage_file
        if worktree_coverage.exists():
            return worktree_coverage

        # Fallback: combine coverage data if coverage command didn't emit XML
        coverage_data = [
            path
            for path in worktree_path.glob(".coverage*")
            if path.is_file() and not path.name.endswith(".xml")
        ]

        combine_result = None
        xml_result = None

        if coverage_data:
            coverage_base = _infer_coverage_base_command(coverage_cmd)

            combine_result = runner.run(
                [*coverage_base, "combine"],
                env=env,
                cwd=worktree_path,
                timeout=timeout,
            )
            if combine_result.returncode == 0:
                xml_result = runner.run(
                    [*coverage_base, "xml", "-o", str(worktree_coverage)],
                    env=env,
                    cwd=worktree_path,
                    timeout=timeout,
                )

        if worktree_coverage.exists():
            return worktree_coverage

        # Build detailed error message
        details: list[str] = []
        if coverage_result.timed_out:
            details.append("coverage command timed out")
        elif coverage_result.returncode != 0:
            details.append(f"coverage command exited {coverage_result.returncode}")
        cmd_tail = coverage_result.stderr_tail() or coverage_result.stdout_tail()
        if cmd_tail:
            details.append(f"command output: {cmd_tail}")
        if combine_result is not None and combine_result.returncode != 0:
            combine_tail = combine_result.stderr_tail() or combine_result.stdout_tail()
            if combine_tail:
                details.append(f"coverage combine failed: {combine_tail}")
        if xml_result is not None and xml_result.returncode != 0:
            xml_tail = xml_result.stderr_tail() or xml_result.stdout_tail()
            if xml_tail:
                details.append(f"coverage xml failed: {xml_tail}")

        detail_msg = f" ({'; '.join(details)})" if details else ""
        return f"No {coverage_file} generated during baseline refresh" + detail_msg

    def _run_refresh(
        self,
        spec: ValidationSpec,
        baseline_path: Path,
    ) -> BaselineRefreshResult:
        """Run coverage command in temp worktree to refresh baseline coverage.

        Args:
            spec: Validation spec (used for worktree context only).
            baseline_path: Where to write the baseline coverage.xml.

        Returns:
            BaselineRefreshResult with the new baseline percentage or error.

        Note:
            Uses self.coverage_config.command for running coverage. This method
            assumes coverage_config and coverage_config.command are validated
            as non-None by the caller (refresh_if_stale).
        """
        # coverage_config.command is validated non-None in refresh_if_stale
        assert self.coverage_config is not None
        assert self.coverage_config.command is not None
        coverage_command = self.coverage_config.command

        # Determine timeout: prefer coverage_config.timeout, then step_timeout_seconds, then default
        timeout = float(
            self.coverage_config.timeout or self.step_timeout_seconds or 300.0
        )

        # Determine lock directory
        lock_dir = self.env_config.lock_dir

        try:
            with baseline_worktree(
                repo_path=self.repo_path,
                timeout=timeout,
                lock_dir=lock_dir,
                command_runner=self.command_runner,
            ) as ctx:
                worktree_path = ctx.worktree_path
                runner = ctx.runner
                env = ctx.env

                # Run uv sync first to install dependencies
                sync_result = runner.run(
                    ["uv", "sync", "--all-extras"],
                    env=env,
                    cwd=worktree_path,
                    timeout=timeout,
                )
                if sync_result.returncode != 0:
                    return BaselineRefreshResult.fail(
                        f"uv sync failed during baseline refresh: {sync_result.stderr}"
                    )

                # Rewrite coverage command for baseline refresh:
                # - Strip xdist flags for deterministic coverage
                # - Remove fail-under threshold
                # - Normalize marker expression (no e2e)
                # - Set output path for XML coverage
                coverage_file = Path(self.coverage_config.file)
                new_coverage_cmd = rewrite_coverage_command(
                    coverage_command, str(coverage_file)
                )

                # Run coverage with fallback to combine
                result = self._run_coverage_with_fallback(
                    runner, new_coverage_cmd, coverage_file, worktree_path, env, timeout
                )

                if isinstance(result, str):
                    return BaselineRefreshResult.fail(result)

                worktree_coverage = result

                # Atomic rename to main repo
                temp_coverage = baseline_path.with_suffix(".xml.tmp")
                shutil.copy2(worktree_coverage, temp_coverage)
                os.rename(temp_coverage, baseline_path)

                # Parse and return the coverage percentage
                try:
                    baseline = get_baseline_coverage(baseline_path)
                    if baseline is None:
                        return BaselineRefreshResult.fail(
                            f"Baseline {coverage_file} exists but has no coverage data"
                        )
                    return BaselineRefreshResult.ok(baseline)
                except ValueError as e:
                    return BaselineRefreshResult.fail(
                        f"Failed to parse baseline coverage: {e}"
                    )
        except RuntimeError as e:
            # Worktree creation failed
            return BaselineRefreshResult.fail(str(e))

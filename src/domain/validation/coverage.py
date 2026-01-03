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
import shlex
import shutil
import tempfile
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from src.infra.tools.command_runner import run_command

from .config import YamlCoverageConfig  # noqa: TC001 - used at runtime

if TYPE_CHECKING:
    from src.core.protocols import EnvConfigPort

    from .spec import ValidationSpec


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


def is_baseline_stale(report_path: Path, repo_path: Path) -> bool:
    """Check if the coverage baseline file is stale and needs refresh.

    A baseline is considered stale if:
    - The baseline file doesn't exist
    - The repo has uncommitted changes (dirty working tree)
    - The baseline file's mtime is older than the last commit time
    - Git commands fail (non-git repo or git errors)

    Args:
        report_path: Path to the coverage.xml baseline file.
        repo_path: Path to the git repository root.

    Returns:
        True if baseline is stale or doesn't exist, False if baseline is fresh.
    """
    # Missing baseline is considered stale
    if not report_path.exists():
        return True

    try:
        # Check for dirty working tree
        dirty_result = run_command(
            ["git", "status", "--porcelain"],
            cwd=repo_path,
        )
        if not dirty_result.ok:
            # Git command failed - treat as stale
            return True
        if dirty_result.stdout.strip():
            # Has uncommitted changes
            return True

        # Get last commit timestamp (Unix epoch seconds)
        commit_time_result = run_command(
            ["git", "log", "-1", "--format=%ct"],
            cwd=repo_path,
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
        coverage_config: YamlCoverageConfig | None = None,
        step_timeout_seconds: float | None = None,
        env_config: EnvConfigPort | None = None,
    ):
        """Initialize the baseline coverage service.

        Args:
            repo_path: Path to the repository.
            coverage_config: Coverage configuration from mala.yaml. Required for
                baseline refresh - if None or if command is None, refresh is unavailable.
            step_timeout_seconds: Optional fallback timeout for commands (used if
                coverage_config.timeout is None).
            env_config: Environment configuration for paths (lock_dir, etc.).
        """
        self.repo_path = repo_path.resolve()
        self.coverage_config = coverage_config
        self.step_timeout_seconds = step_timeout_seconds
        self.env_config = env_config

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
        from src.infra.tools.locking import lock_path, try_lock, wait_for_lock

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
        if not is_baseline_stale(baseline_path, self.repo_path):
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
        if not try_lock(_BASELINE_LOCK_FILE, agent_id, repo_namespace):
            # Another agent is refreshing - wait for them
            if not wait_for_lock(
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
            if not is_baseline_stale(baseline_path, self.repo_path):
                try:
                    baseline = get_baseline_coverage(baseline_path)
                    if baseline is not None:
                        return BaselineRefreshResult.ok(baseline)
                except ValueError:
                    pass  # Still need to refresh

            # Still stale - run refresh in temp worktree
            return self._run_refresh(spec, baseline_path)
        finally:
            # Release lock by removing lock file
            lock_file = lock_path(_BASELINE_LOCK_FILE, repo_namespace)
            lock_file.unlink(missing_ok=True)

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
        from src.infra.tools.command_runner import CommandRunner
        from .worktree import (
            WorktreeConfig,
            WorktreeState,
            create_worktree,
            remove_worktree,
        )

        # coverage_config.command is validated non-None in refresh_if_stale
        assert self.coverage_config is not None
        assert self.coverage_config.command is not None
        coverage_command = self.coverage_config.command

        # Create temp worktree at HEAD
        run_id = f"baseline-{uuid.uuid4().hex[:8]}"
        temp_dir = Path(tempfile.mkdtemp(prefix="mala-baseline-"))
        worktree_config = WorktreeConfig(
            base_dir=temp_dir,
            keep_on_failure=False,
        )

        worktree_ctx = None
        try:
            worktree_ctx = create_worktree(
                repo_path=self.repo_path,
                commit_sha="HEAD",
                config=worktree_config,
                run_id=run_id,
                issue_id="baseline",
                attempt=1,
            )

            if worktree_ctx.state == WorktreeState.FAILED:
                return BaselineRefreshResult.fail(
                    f"Baseline worktree creation failed: {worktree_ctx.error}"
                )

            worktree_path = worktree_ctx.path

            # Build environment
            env = {
                **os.environ,
                "AGENT_ID": f"baseline-{run_id}",
            }
            if self.env_config is not None:
                env["LOCK_DIR"] = str(self.env_config.lock_dir)
            else:
                # Fallback for legacy callers without env_config
                from src.infra.tools.env import get_lock_dir

                env["LOCK_DIR"] = str(get_lock_dir())

            # Determine timeout: prefer coverage_config.timeout, then step_timeout_seconds, then default
            timeout = float(
                self.coverage_config.timeout or self.step_timeout_seconds or 300.0
            )
            runner = CommandRunner(cwd=worktree_path, timeout_seconds=timeout)

            # Run uv sync first to install dependencies
            sync_result = runner.run(["uv", "sync", "--all-extras"], env=env)
            if sync_result.returncode != 0:
                return BaselineRefreshResult.fail(
                    f"uv sync failed during baseline refresh: {sync_result.stderr}"
                )

            # Convert coverage command to list for manipulation
            coverage_cmd = shlex.split(coverage_command)

            # Replace any existing --cov-fail-under with 0
            # This ensures we capture baseline even if it's below pyproject.toml threshold
            # Also replace any existing -m marker with "unit or integration" to
            # avoid end-to-end tests during baseline refresh.
            # Remove xdist flags to avoid coverage merge flakiness in baseline refresh.
            new_coverage_cmd = []
            marker_expr: str | None = None
            skip_next = False
            for arg in coverage_cmd:
                if skip_next:
                    skip_next = False
                    continue
                # Strip xdist flags for deterministic coverage generation
                if arg in {"-n", "--numprocesses", "--dist"}:
                    skip_next = True
                    continue
                if arg.startswith("-n=") or arg.startswith("--numprocesses="):
                    continue
                if arg.startswith("--dist="):
                    continue
                if arg.startswith("--cov-fail-under="):
                    continue
                if arg == "--cov-fail-under":
                    skip_next = True
                    continue
                # Strip any existing -m marker (we'll re-add a safe marker at the end)
                if arg.startswith("-m="):
                    marker_expr = arg.split("=", 1)[1]
                    continue
                if arg == "-m":
                    skip_next = True
                    marker_expr = None  # will be captured via skip_next branch
                    continue
                new_coverage_cmd.append(arg)

            # If we skipped "-m <expr>", capture the expression from the next arg
            if skip_next:
                # Defensive: skip_next should always be paired with an arg
                skip_next = False

            # If we stripped a marker via "-m", it was the immediate next arg
            if marker_expr is None:
                for i, arg in enumerate(coverage_cmd[:-1]):
                    if arg == "-m":
                        marker_expr = coverage_cmd[i + 1]
                        break

            # Normalize marker expression: never include e2e in baseline refresh
            marker_expr = (marker_expr or "unit or integration").strip()
            if "e2e" in marker_expr:
                marker_expr = "unit or integration"

            # Ensure XML coverage output is written to the configured path.
            # Strip any existing --cov-report=xml or --cov-report=xml:<path> arguments
            # to avoid path mismatches where output goes to a different location than
            # coverage_config.file specifies.
            coverage_file = self.coverage_config.file
            new_coverage_cmd = [
                arg
                for arg in new_coverage_cmd
                if arg != "--cov-report=xml" and not arg.startswith("--cov-report=xml:")
            ]
            new_coverage_cmd.append(f"--cov-report=xml:{coverage_file}")

            new_coverage_cmd.append("--cov-fail-under=0")
            new_coverage_cmd.extend(["-m", marker_expr])

            # Run coverage command - we ignore the exit code because tests may fail
            # but still generate a valid coverage.xml baseline
            coverage_result = runner.run(new_coverage_cmd, env=env)

            # Check for coverage file in worktree (use configured file path)
            coverage_file = Path(self.coverage_config.file)
            worktree_coverage = worktree_path / coverage_file
            if not worktree_coverage.exists():
                # Fallback: combine coverage data if coverage command didn't emit XML
                coverage_data = [
                    path
                    for path in worktree_path.glob(".coverage*")
                    if path.is_file() and not path.name.endswith(".xml")
                ]

                combine_result = None
                xml_result = None

                if coverage_data:
                    # Use the same invocation style as the coverage command when possible.
                    if (
                        len(coverage_cmd) >= 3
                        and coverage_cmd[0] == "uv"
                        and coverage_cmd[1] == "run"
                    ):
                        coverage_base = ["uv", "run", "coverage"]
                    elif (
                        len(coverage_cmd) >= 3
                        and coverage_cmd[1] == "-m"
                        and coverage_cmd[2] == "pytest"
                    ):
                        coverage_base = [coverage_cmd[0], "-m", "coverage"]
                    else:
                        coverage_base = ["coverage"]

                    combine_result = runner.run(
                        [*coverage_base, "combine"],
                        env=env,
                    )
                    if combine_result.returncode == 0:
                        xml_result = runner.run(
                            [*coverage_base, "xml", "-o", str(worktree_coverage)],
                            env=env,
                        )

                if not worktree_coverage.exists():
                    details: list[str] = []
                    if coverage_result.timed_out:
                        details.append("coverage command timed out")
                    elif coverage_result.returncode != 0:
                        details.append(
                            f"coverage command exited {coverage_result.returncode}"
                        )
                    cmd_tail = (
                        coverage_result.stderr_tail() or coverage_result.stdout_tail()
                    )
                    if cmd_tail:
                        details.append(f"command output: {cmd_tail}")
                    if combine_result is not None and combine_result.returncode != 0:
                        combine_tail = (
                            combine_result.stderr_tail() or combine_result.stdout_tail()
                        )
                        if combine_tail:
                            details.append(f"coverage combine failed: {combine_tail}")
                    if xml_result is not None and xml_result.returncode != 0:
                        xml_tail = xml_result.stderr_tail() or xml_result.stdout_tail()
                        if xml_tail:
                            details.append(f"coverage xml failed: {xml_tail}")

                    detail_msg = f" ({'; '.join(details)})" if details else ""
                    return BaselineRefreshResult.fail(
                        f"No {coverage_file} generated during baseline refresh"
                        + detail_msg
                    )

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

        finally:
            # Clean up temp worktree
            if worktree_ctx is not None:
                remove_worktree(worktree_ctx, validation_passed=True)
            # Clean up temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)

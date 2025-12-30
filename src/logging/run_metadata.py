"""Run metadata tracking for mala orchestrator runs.

Captures orchestrator configuration, issue results, and pointers to Claude logs.
Replaces the duplicate JSONL logging with structured run metadata.
"""

import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Literal

from ..models import (
    IssueResolution,
    ResolutionOutcome,
    ValidationArtifacts,
)
from ..tools.env import get_lock_dir, get_repo_runs_dir


@dataclass
class QualityGateResult:
    """Quality gate check result for an issue."""

    passed: bool
    evidence: dict[str, bool] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    """Result of validation execution for observability.

    Attributes:
        passed: Whether all validations passed.
        commands_run: List of command names that were executed.
        commands_failed: List of command names that failed.
        artifacts: Validation artifacts (logs, worktree, coverage report).
        coverage_percent: Coverage percentage if measured (None if not run).
        e2e_passed: Whether E2E tests passed (None if not run).
    """

    passed: bool
    commands_run: list[str] = field(default_factory=list)
    commands_failed: list[str] = field(default_factory=list)
    artifacts: ValidationArtifacts | None = None
    coverage_percent: float | None = None
    e2e_passed: bool | None = None


@dataclass
class IssueRun:
    """Result of running an agent on a single issue."""

    issue_id: str
    agent_id: str
    status: Literal["success", "failed", "timeout"]
    duration_seconds: float
    session_id: str | None = None  # Claude SDK session ID
    log_path: str | None = None  # Path to Claude's log file
    quality_gate: QualityGateResult | None = None
    error: str | None = None
    # Retry tracking (recorded even if defaulted)
    gate_attempts: int = 0
    review_attempts: int = 0
    # Validation results and resolution (from mala-e0i)
    validation: ValidationResult | None = None
    resolution: IssueResolution | None = None
    # Cerberus review session log path (verbose mode only)
    review_log_path: str | None = None


@dataclass
class RunConfig:
    """Orchestrator run configuration."""

    max_agents: int | None
    timeout_minutes: int | None
    max_issues: int | None
    epic_id: str | None
    only_ids: list[str] | None
    braintrust_enabled: bool
    # Retry/review config (optional for backward compatibility)
    max_gate_retries: int | None = None
    max_review_retries: int | None = None
    review_enabled: bool | None = None
    # CLI args for debugging/auditing (optional for backward compatibility)
    cli_args: dict[str, object] | None = None


class RunMetadata:
    """Tracks metadata for a single orchestrator run.

    Creates a JSON file at ~/.config/mala/runs/{run_id}.json containing:
    - Run configuration
    - Per-issue results with Claude log path pointers
    - Quality gate outcomes
    - Validation results and artifacts
    - Timing and error information
    """

    def __init__(
        self,
        repo_path: Path,
        config: RunConfig,
        version: str,
    ):
        self.run_id = str(uuid.uuid4())
        self.started_at = datetime.now(UTC)
        self.completed_at: datetime | None = None
        self.repo_path = repo_path
        self.config = config
        self.version = version
        self.issues: dict[str, IssueRun] = {}
        # Run-level validation results (from mala-e0i)
        self.run_validation: ValidationResult | None = None

    def record_issue(self, issue: IssueRun) -> None:
        """Record the result of an issue run."""
        self.issues[issue.issue_id] = issue

    def record_run_validation(self, result: ValidationResult) -> None:
        """Record run-level validation results.

        Args:
            result: The validation result for the entire run.
        """
        self.run_validation = result

    def _serialize_validation_artifacts(
        self, artifacts: ValidationArtifacts | None
    ) -> dict[str, Any] | None:
        """Serialize ValidationArtifacts to a JSON-compatible dict."""
        if artifacts is None:
            return None
        return {
            "log_dir": str(artifacts.log_dir),
            "worktree_path": str(artifacts.worktree_path)
            if artifacts.worktree_path
            else None,
            "worktree_state": artifacts.worktree_state,
            "coverage_report": str(artifacts.coverage_report)
            if artifacts.coverage_report
            else None,
            "e2e_fixture_path": str(artifacts.e2e_fixture_path)
            if artifacts.e2e_fixture_path
            else None,
        }

    def _serialize_validation_result(
        self, result: ValidationResult | None
    ) -> dict[str, Any] | None:
        """Serialize ValidationResult to a JSON-compatible dict."""
        if result is None:
            return None
        return {
            "passed": result.passed,
            "commands_run": result.commands_run,
            "commands_failed": result.commands_failed,
            "artifacts": self._serialize_validation_artifacts(result.artifacts),
            "coverage_percent": result.coverage_percent,
            "e2e_passed": result.e2e_passed,
        }

    def _serialize_issue_resolution(
        self, resolution: IssueResolution | None
    ) -> dict[str, Any] | None:
        """Serialize IssueResolution to a JSON-compatible dict."""
        if resolution is None:
            return None
        return {
            "outcome": resolution.outcome.value,
            "rationale": resolution.rationale,
        }

    def _to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "version": self.version,
            "repo_path": str(self.repo_path),
            "config": asdict(self.config),
            "issues": {
                issue_id: {
                    **asdict(issue),
                    "quality_gate": asdict(issue.quality_gate)
                    if issue.quality_gate
                    else None,
                    "validation": self._serialize_validation_result(issue.validation),
                    "resolution": self._serialize_issue_resolution(issue.resolution),
                }
                for issue_id, issue in self.issues.items()
            },
            "run_validation": self._serialize_validation_result(self.run_validation),
        }

    @staticmethod
    def _deserialize_validation_artifacts(
        data: dict[str, Any] | None,
    ) -> ValidationArtifacts | None:
        """Deserialize ValidationArtifacts from a dict."""
        if data is None:
            return None
        return ValidationArtifacts(
            log_dir=Path(data["log_dir"]),
            worktree_path=Path(data["worktree_path"])
            if data.get("worktree_path")
            else None,
            worktree_state=data.get("worktree_state"),
            coverage_report=Path(data["coverage_report"])
            if data.get("coverage_report")
            else None,
            e2e_fixture_path=Path(data["e2e_fixture_path"])
            if data.get("e2e_fixture_path")
            else None,
        )

    @staticmethod
    def _deserialize_validation_result(
        data: dict[str, Any] | None,
    ) -> ValidationResult | None:
        """Deserialize ValidationResult from a dict."""
        if data is None:
            return None
        return ValidationResult(
            passed=data["passed"],
            commands_run=data.get("commands_run", []),
            commands_failed=data.get("commands_failed", []),
            artifacts=RunMetadata._deserialize_validation_artifacts(
                data.get("artifacts")
            ),
            coverage_percent=data.get("coverage_percent"),
            e2e_passed=data.get("e2e_passed"),
        )

    @staticmethod
    def _deserialize_issue_resolution(
        data: dict[str, Any] | None,
    ) -> IssueResolution | None:
        """Deserialize IssueResolution from a dict."""
        if data is None:
            return None
        return IssueResolution(
            outcome=ResolutionOutcome(data["outcome"]),
            rationale=data["rationale"],
        )

    @classmethod
    def load(cls, path: Path) -> "RunMetadata":
        """Load run metadata from a JSON file.

        Args:
            path: Path to the JSON file.

        Returns:
            Loaded RunMetadata instance.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            json.JSONDecodeError: If the file is invalid JSON.
        """
        with open(path) as f:
            data = json.load(f)

        # Reconstruct config
        config_data = data["config"]
        config = RunConfig(
            max_agents=config_data.get("max_agents"),
            timeout_minutes=config_data.get("timeout_minutes"),
            max_issues=config_data.get("max_issues"),
            epic_id=config_data.get("epic_id"),
            only_ids=config_data.get("only_ids"),
            braintrust_enabled=config_data.get("braintrust_enabled", False),
            max_gate_retries=config_data.get("max_gate_retries"),
            max_review_retries=config_data.get("max_review_retries"),
            review_enabled=config_data.get("review_enabled"),
            cli_args=config_data.get("cli_args"),
        )

        # Create instance
        metadata = cls.__new__(cls)
        metadata.run_id = data["run_id"]
        metadata.started_at = datetime.fromisoformat(data["started_at"])
        metadata.completed_at = (
            datetime.fromisoformat(data["completed_at"])
            if data.get("completed_at")
            else None
        )
        metadata.repo_path = Path(data["repo_path"])
        metadata.config = config
        metadata.version = data["version"]

        # Reconstruct issues
        metadata.issues = {}
        for issue_id, issue_data in data.get("issues", {}).items():
            quality_gate = None
            if issue_data.get("quality_gate"):
                qg_data = issue_data["quality_gate"]
                quality_gate = QualityGateResult(
                    passed=qg_data["passed"],
                    evidence=qg_data.get("evidence", {}),
                    failure_reasons=qg_data.get("failure_reasons", []),
                )

            # Deserialize new fields
            validation = cls._deserialize_validation_result(
                issue_data.get("validation")
            )
            resolution = cls._deserialize_issue_resolution(issue_data.get("resolution"))

            issue = IssueRun(
                issue_id=issue_data["issue_id"],
                agent_id=issue_data["agent_id"],
                status=issue_data["status"],
                duration_seconds=issue_data["duration_seconds"],
                session_id=issue_data.get("session_id"),
                log_path=issue_data.get("log_path"),
                quality_gate=quality_gate,
                error=issue_data.get("error"),
                gate_attempts=issue_data.get("gate_attempts", 0),
                review_attempts=issue_data.get("review_attempts", 0),
                validation=validation,
                resolution=resolution,
                review_log_path=issue_data.get("review_log_path"),
            )
            metadata.issues[issue_id] = issue

        # Load run-level validation
        metadata.run_validation = cls._deserialize_validation_result(
            data.get("run_validation")
        )

        return metadata

    def save(self) -> Path:
        """Save run metadata to JSON file.

        Saves to a repo-specific subdirectory with timestamp-based filename
        for easier sorting: {runs_dir}/{repo-safe-name}/{timestamp}_{short-uuid}.json

        Returns:
            Path to the saved metadata file.
        """
        self.completed_at = datetime.now(UTC)
        # Use repo-specific subdirectory
        runs_dir = get_repo_runs_dir(self.repo_path)
        runs_dir.mkdir(parents=True, exist_ok=True)

        # Use timestamp + short UUID for filename
        timestamp = self.started_at.strftime("%Y-%m-%dT%H-%M-%S")
        short_id = self.run_id[:8]
        path = runs_dir / f"{timestamp}_{short_id}.json"

        with open(path, "w") as f:
            json.dump(self._to_dict(), f, indent=2)
        return path


# --- Running Instance Tracking ---


@dataclass
class RunningInstance:
    """Information about a currently running mala instance."""

    run_id: str
    repo_path: Path
    started_at: datetime
    pid: int
    max_agents: int | None = None
    issues_in_progress: int = 0


def _get_marker_path(run_id: str) -> Path:
    """Get the path to a run marker file.

    Args:
        run_id: The run ID.

    Returns:
        Path to the marker file.
    """
    return get_lock_dir() / f"run-{run_id}.marker"


def write_run_marker(
    run_id: str,
    repo_path: Path,
    max_agents: int | None = None,
) -> Path:
    """Write a run marker file to indicate a running instance.

    Creates a marker file in the lock directory that records the run's
    repo path, start time, and PID. Used by status command to detect
    running instances.

    Args:
        run_id: The unique run ID.
        repo_path: Path to the repository being processed.
        max_agents: Maximum number of concurrent agents (optional).

    Returns:
        Path to the created marker file.
    """
    lock_dir = get_lock_dir()
    lock_dir.mkdir(parents=True, exist_ok=True)

    marker_path = _get_marker_path(run_id)
    data = {
        "run_id": run_id,
        "repo_path": str(repo_path.resolve()),
        "started_at": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "max_agents": max_agents,
    }

    with open(marker_path, "w") as f:
        json.dump(data, f)

    return marker_path


def remove_run_marker(run_id: str) -> bool:
    """Remove a run marker file.

    Called when a run completes (successfully or not).

    Args:
        run_id: The run ID whose marker should be removed.

    Returns:
        True if the marker was removed, False if it didn't exist.
    """
    marker_path = _get_marker_path(run_id)
    if marker_path.exists():
        marker_path.unlink()
        return True
    return False


def get_running_instances() -> list[RunningInstance]:
    """Get all currently running mala instances.

    Reads all run marker files from the lock directory and returns
    information about each running instance. Stale markers (where the
    PID is no longer running) are automatically cleaned up.

    Returns:
        List of RunningInstance objects for all active runs.
    """
    lock_dir = get_lock_dir()
    if not lock_dir.exists():
        return []

    instances: list[RunningInstance] = []
    stale_markers: list[Path] = []

    for marker_path in lock_dir.glob("run-*.marker"):
        try:
            with open(marker_path) as f:
                data = json.load(f)

            pid = data.get("pid")
            # Check if the process is still running
            if pid and not _is_process_running(pid):
                stale_markers.append(marker_path)
                continue

            instance = RunningInstance(
                run_id=data["run_id"],
                repo_path=Path(data["repo_path"]),
                started_at=datetime.fromisoformat(data["started_at"]),
                pid=pid or 0,
                max_agents=data.get("max_agents"),
            )
            instances.append(instance)
        except (json.JSONDecodeError, KeyError, OSError):
            # Corrupted or unreadable marker - treat as stale
            stale_markers.append(marker_path)

    # Clean up stale markers
    for marker in stale_markers:
        try:
            marker.unlink()
        except OSError:
            pass

    return instances


def get_running_instances_for_dir(directory: Path) -> list[RunningInstance]:
    """Get running mala instances for a specific directory.

    Filters running instances to only those whose repo_path matches
    the given directory (resolved to absolute path).

    Args:
        directory: The directory to filter by.

    Returns:
        List of RunningInstance objects running in the specified directory.
    """
    resolved_dir = directory.resolve()
    return [
        instance
        for instance in get_running_instances()
        if instance.repo_path.resolve() == resolved_dir
    ]


def _is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is still running.

    Args:
        pid: The process ID to check.

    Returns:
        True if the process is running, False otherwise.
    """
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False

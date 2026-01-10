"""Run metadata tracking for mala orchestrator runs.

Captures orchestrator configuration, issue results, and pointers to Claude logs.
Replaces the duplicate JSONL logging with structured run metadata.
"""

import json
import logging
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Literal

from src.core.models import (
    IssueResolution,
    ResolutionOutcome,
    ValidationArtifacts,
)
from src.infra.tools.env import get_lock_dir, get_repo_runs_dir

# Type aliases for dependency injection in tests

ProcessChecker = Callable[[int], bool]
CorruptFileCallback = Callable[[Path, Exception | None], None]


def configure_debug_logging(
    repo_path: Path, run_id: str, *, runs_dir: Path | None = None
) -> Path | None:
    """Configure Python logging to write debug logs to a file.

    Creates a debug log file alongside run metadata at:
    ~/.mala/runs/{repo}/{timestamp}_{run_id}.debug.log

    All loggers in the 'src' namespace will write DEBUG+ messages to this file.

    This function is best-effort: if the log directory cannot be created or
    the log file cannot be opened (e.g., read-only filesystem, permission
    denied), it returns None and the run continues without debug logging.

    Set MALA_DISABLE_DEBUG_LOG=1 to disable debug logging entirely.

    Args:
        repo_path: Repository path for log directory.
        run_id: Run ID (UUID) for filename.
        runs_dir: Optional custom runs directory. If None, uses default from
            get_repo_runs_dir().

    Returns:
        Path to the debug log file, or None if logging could not be configured
        or is disabled via environment variable.
    """
    # Allow opt-out via environment variable
    if os.environ.get("MALA_DISABLE_DEBUG_LOG") == "1":
        return None

    try:
        effective_runs_dir = (
            runs_dir if runs_dir is not None else get_repo_runs_dir(repo_path)
        )
        effective_runs_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
        short_id = run_id[:8]
        log_path = effective_runs_dir / f"{timestamp}_{short_id}.debug.log"

        # Create file handler for debug logs
        handler = logging.FileHandler(log_path)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        # Tag the handler so we can identify it later
        handler.set_name(f"mala_debug_{run_id}")

        # Add handler to root logger for 'src' namespace
        src_logger = logging.getLogger("src")
        src_logger.setLevel(logging.DEBUG)

        # Remove any previous mala debug handlers to avoid duplicates/leaks
        for existing in src_logger.handlers[:]:
            if getattr(existing, "name", "").startswith("mala_debug_"):
                existing.close()
                src_logger.removeHandler(existing)

        src_logger.addHandler(handler)

        return log_path
    except OSError:
        # Best-effort: if we can't create the log file, continue without it
        # This handles read-only filesystems, permission denied, disk full, etc.
        return None


def cleanup_debug_logging(run_id: str) -> bool:
    """Clean up debug logging handler for a completed run.

    Removes and closes the FileHandler associated with the given run_id
    to prevent file handle leaks.

    Args:
        run_id: Run ID (UUID) whose handler should be cleaned up.

    Returns:
        True if a handler was found and cleaned up, False otherwise.
    """
    src_logger = logging.getLogger("src")
    handler_name = f"mala_debug_{run_id}"

    for handler in src_logger.handlers[:]:
        if getattr(handler, "name", "") == handler_name:
            handler.close()
            src_logger.removeHandler(handler)
            return True

    return False


@dataclass
class EvidenceCheckResult:
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
    evidence_check: EvidenceCheckResult | None = None
    error: str | None = None
    # Retry tracking (recorded even if defaulted)
    gate_attempts: int = 0
    review_attempts: int = 0
    # Validation results and resolution (from mala-e0i)
    validation: ValidationResult | None = None
    resolution: IssueResolution | None = None
    # Cerberus review session log path (verbose mode only)
    review_log_path: str | None = None
    # Baseline timestamp used for commit freshness checks
    baseline_timestamp: int | None = None
    # Review issues from last review (for resume with feedback)
    last_review_issues: list[dict[str, Any]] | None = None


@dataclass
class RunConfig:
    """Orchestrator run configuration."""

    max_agents: int | None
    timeout_minutes: int | None
    max_issues: int | None
    epic_id: str | None
    only_ids: list[str] | None
    # Retry/review config (optional for backward compatibility)
    max_gate_retries: int | None = None
    max_review_retries: int | None = None
    review_enabled: bool | None = None
    # CLI args for debugging/auditing (optional for backward compatibility)
    cli_args: dict[str, object] | None = None
    # Orphans-only filter (optional for backward compatibility)
    orphans_only: bool = False


class RunMetadata:
    """Tracks metadata for a single orchestrator run.

    Creates a JSON file at ~/.config/mala/runs/{run_id}.json containing:
    - Run configuration
    - Per-session results with Claude log path pointers
    - Quality gate outcomes
    - Validation results and artifacts
    - Timing and error information
    """

    def __init__(
        self,
        repo_path: Path,
        config: RunConfig,
        version: str,
        runs_dir: Path | None = None,
    ):
        self.run_id = str(uuid.uuid4())
        self.started_at = datetime.now(UTC)
        self.completed_at: datetime | None = None
        self.repo_path = repo_path
        self.config = config
        self.version = version
        self._runs_dir = runs_dir
        self.issues: dict[str, IssueRun] = {}
        # Global validation results (from mala-e0i)
        self.run_validation: ValidationResult | None = None
        # Baseline tracking for cumulative reviews
        self.run_start_commit: str | None = None
        self.last_cumulative_review_commits: dict[str, str] = {}
        # Key format: "run_end" or "epic_completion:<epic_id>"
        # Configure debug logging for this run (always enabled)
        self.debug_log_path: Path | None = configure_debug_logging(
            repo_path, self.run_id, runs_dir=runs_dir
        )

    def record_issue(self, issue: IssueRun) -> None:
        """Record the result of an issue run."""
        self.issues[issue.issue_id] = issue

    def record_run_validation(self, result: ValidationResult) -> None:
        """Record global validation results.

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
                    "evidence_check": asdict(issue.evidence_check)
                    if issue.evidence_check
                    else None,
                    "validation": self._serialize_validation_result(issue.validation),
                    "resolution": self._serialize_issue_resolution(issue.resolution),
                }
                for issue_id, issue in self.issues.items()
            },
            "run_validation": self._serialize_validation_result(self.run_validation),
            "debug_log_path": str(self.debug_log_path) if self.debug_log_path else None,
            "run_start_commit": self.run_start_commit,
            "last_cumulative_review_commits": self.last_cumulative_review_commits,
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
            max_gate_retries=config_data.get("max_gate_retries"),
            max_review_retries=config_data.get("max_review_retries"),
            review_enabled=config_data.get("review_enabled"),
            cli_args=config_data.get("cli_args"),
            orphans_only=config_data.get("orphans_only", False),
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
        metadata._runs_dir = None  # Loaded instances use repo_path for save()

        # Reconstruct issues
        metadata.issues = {}
        for issue_id, issue_data in data.get("issues", {}).items():
            evidence_check = None
            if issue_data.get("evidence_check"):
                qg_data = issue_data["evidence_check"]
                evidence_check = EvidenceCheckResult(
                    passed=qg_data["passed"],
                    evidence=qg_data.get("evidence", {}),
                    failure_reasons=qg_data.get("failure_reasons", []),
                )

            # Deserialize new fields
            validation = cls._deserialize_validation_result(
                issue_data.get("validation")
            )
            resolution = cls._deserialize_issue_resolution(issue_data.get("resolution"))

            # Deserialize last_review_issues (list of dicts or None)
            raw_last_review = issue_data.get("last_review_issues")
            last_review_issues = (
                raw_last_review if isinstance(raw_last_review, list) else None
            )

            issue = IssueRun(
                issue_id=issue_data["issue_id"],
                agent_id=issue_data["agent_id"],
                status=issue_data["status"],
                duration_seconds=issue_data["duration_seconds"],
                session_id=issue_data.get("session_id"),
                log_path=issue_data.get("log_path"),
                evidence_check=evidence_check,
                error=issue_data.get("error"),
                gate_attempts=issue_data.get("gate_attempts", 0),
                review_attempts=issue_data.get("review_attempts", 0),
                validation=validation,
                resolution=resolution,
                review_log_path=issue_data.get("review_log_path"),
                baseline_timestamp=(
                    issue_data.get("baseline_timestamp")
                    if isinstance(issue_data.get("baseline_timestamp"), int)
                    else None
                ),
                last_review_issues=last_review_issues,
            )
            metadata.issues[issue_id] = issue

        # Load global validation
        metadata.run_validation = cls._deserialize_validation_result(
            data.get("run_validation")
        )

        # Restore debug_log_path (don't reconfigure logging on load)
        debug_log_path = data.get("debug_log_path")
        metadata.debug_log_path = Path(debug_log_path) if debug_log_path else None

        # Load cumulative review baseline fields (with backward-compat defaults)
        metadata.run_start_commit = data.get("run_start_commit")
        metadata.last_cumulative_review_commits = dict(
            data.get("last_cumulative_review_commits", {})
        )

        return metadata

    def cleanup(self) -> None:
        """Clean up resources associated with this run.

        This method is idempotent and safe to call multiple times.
        It cleans up the debug logging handler to prevent file handle leaks.

        Should be called in a finally block to ensure cleanup happens even
        if the run crashes or is aborted before save() is called.
        """
        if self.debug_log_path is not None:
            cleanup_debug_logging(self.run_id)

    def save(self) -> Path:
        """Save run metadata to JSON file.

        Saves to a repo-specific subdirectory with timestamp-based filename
        for easier sorting: {runs_dir}/{repo-safe-name}/{timestamp}_{short-uuid}.json

        Also cleans up the debug logging handler to prevent file handle leaks.

        Returns:
            Path to the saved metadata file.
        """
        self.completed_at = datetime.now(UTC)

        # Clean up debug logging handler before saving (idempotent)
        self.cleanup()

        # Use repo-specific subdirectory (or custom runs_dir if provided)
        runs_dir = (
            self._runs_dir
            if self._runs_dir is not None
            else get_repo_runs_dir(self.repo_path)
        )
        runs_dir.mkdir(parents=True, exist_ok=True)

        # Use timestamp + short UUID for filename
        timestamp = self.started_at.strftime("%Y-%m-%dT%H-%M-%S")
        short_id = self.run_id[:8]
        path = runs_dir / f"{timestamp}_{short_id}.json"

        with open(path, "w") as f:
            json.dump(self._to_dict(), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
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


def _get_marker_path(run_id: str, lock_dir: Path | None = None) -> Path:
    """Get the path to a run marker file.

    Args:
        run_id: The run ID.
        lock_dir: Override lock directory (for testing). If None, uses default.

    Returns:
        Path to the marker file.
    """
    effective_lock_dir = lock_dir if lock_dir is not None else get_lock_dir()
    return effective_lock_dir / f"run-{run_id}.marker"


def write_run_marker(
    run_id: str,
    repo_path: Path,
    max_agents: int | None = None,
    *,
    lock_dir: Path | None = None,
) -> Path:
    """Write a run marker file to indicate a running instance.

    Creates a marker file in the lock directory that records the run's
    repo path, start time, and PID. Used by status command to detect
    running instances.

    Args:
        run_id: The unique run ID.
        repo_path: Path to the repository being processed.
        max_agents: Maximum number of concurrent agents (optional).
        lock_dir: Override lock directory (for testing). If None, uses default.

    Returns:
        Path to the created marker file.
    """
    effective_lock_dir = lock_dir if lock_dir is not None else get_lock_dir()
    effective_lock_dir.mkdir(parents=True, exist_ok=True)

    marker_path = _get_marker_path(run_id, lock_dir=effective_lock_dir)
    data = {
        "run_id": run_id,
        "repo_path": str(repo_path.resolve()),
        "started_at": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "max_agents": max_agents,
    }

    with open(marker_path, "w") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())

    return marker_path


def remove_run_marker(run_id: str, *, lock_dir: Path | None = None) -> bool:
    """Remove a run marker file.

    Called when a run completes (successfully or not).

    Args:
        run_id: The run ID whose marker should be removed.
        lock_dir: Override lock directory (for testing). If None, uses default.

    Returns:
        True if the marker was removed, False if it didn't exist.
    """
    marker_path = _get_marker_path(run_id, lock_dir=lock_dir)
    if marker_path.exists():
        marker_path.unlink()
        return True
    return False


def get_running_instances(
    *,
    lock_dir: Path | None = None,
    is_process_running: ProcessChecker | None = None,
) -> list[RunningInstance]:
    """Get all currently running mala instances.

    Reads all run marker files from the lock directory and returns
    information about each running instance. Stale markers (where the
    PID is no longer running) are automatically cleaned up.

    Args:
        lock_dir: Override lock directory (for testing). If None, uses default.
        is_process_running: Override process checker (for testing). If None,
            uses _is_process_running.

    Returns:
        List of RunningInstance objects for all active runs.
    """
    effective_lock_dir = lock_dir if lock_dir is not None else get_lock_dir()
    checker = (
        is_process_running if is_process_running is not None else _is_process_running
    )

    if not effective_lock_dir.exists():
        return []

    instances: list[RunningInstance] = []
    stale_markers: list[Path] = []

    for marker_path in effective_lock_dir.glob("run-*.marker"):
        try:
            with open(marker_path) as f:
                data = json.load(f)

            pid = data.get("pid")
            # Check if the process is still running
            if pid and not checker(pid):
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


def get_running_instances_for_dir(
    directory: Path,
    *,
    lock_dir: Path | None = None,
    is_process_running: ProcessChecker | None = None,
) -> list[RunningInstance]:
    """Get running mala instances for a specific directory.

    Filters running instances to only those whose repo_path matches
    the given directory (resolved to absolute path).

    Args:
        directory: The directory to filter by.
        lock_dir: Override lock directory (for testing). If None, uses default.
        is_process_running: Override process checker (for testing). If None,
            uses _is_process_running.

    Returns:
        List of RunningInstance objects running in the specified directory.
    """
    resolved_dir = directory.resolve()
    return [
        instance
        for instance in get_running_instances(
            lock_dir=lock_dir, is_process_running=is_process_running
        )
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


_logger = logging.getLogger(__name__)


# --- Session Discovery Functions (shared by CLI and orchestrator) ---


# Required keys for valid run metadata (used for validation in load functions)
_REQUIRED_KEYS = {"run_id", "started_at", "issues"}


def _validate_run_data(data: object) -> bool:
    """Check if run metadata is a dict with required keys and valid values.

    Args:
        data: Parsed JSON data (any JSON value).

    Returns:
        True if data is a dict with all required keys and valid values.
    """
    if not isinstance(data, dict):
        return False
    d = dict(data)  # type: dict[str, Any]
    if not _REQUIRED_KEYS.issubset(d.keys()):
        return False
    # Validate run_id is a non-null string
    if not isinstance(d.get("run_id"), str):
        return False
    # Validate started_at is a non-null string
    if not isinstance(d.get("started_at"), str):
        return False
    # Validate issues is a dict (or None which we treat as empty)
    issues = d.get("issues")
    if issues is not None and not isinstance(issues, dict):
        return False
    return True


def _parse_run_file(
    path: Path, on_corrupt: CorruptFileCallback | None = None
) -> dict[str, Any] | None:
    """Parse a run metadata JSON file.

    Args:
        path: Path to JSON file.
        on_corrupt: Optional callback invoked when a file is corrupt or invalid.
            Receives (path, exception) where exception is None for validation
            failures and the actual exception for parse/IO errors.

    Returns:
        Parsed dict if valid, None if corrupt or missing required keys.
        Logs a warning for corrupt files (unless on_corrupt is provided).
    """
    try:
        with path.open() as f:
            data = json.load(f)
        if not _validate_run_data(data):
            if on_corrupt is not None:
                on_corrupt(path, None)
            return None
        # Normalize issues to empty dict if None
        if data.get("issues") is None:
            data["issues"] = {}
        return data
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        if on_corrupt is not None:
            on_corrupt(path, e)
        else:
            _logger.warning("Skipping corrupt file %s: %s", path, e)
        return None


def _extract_timestamp_prefix(filename: str) -> str:
    """Extract the timestamp prefix from a run filename.

    Filenames have format: {timestamp}_{short_id}.json
    where timestamp is %Y-%m-%dT%H-%M-%S (second resolution).

    Args:
        filename: The filename (not full path).

    Returns:
        The timestamp prefix (everything before the first underscore),
        or empty string if no underscore found.
    """
    underscore_pos = filename.find("_")
    if underscore_pos == -1:
        return ""
    return filename[:underscore_pos]


def discover_run_files(repo_path: Path | None = None) -> list[Path]:
    """Discover run metadata JSON files.

    This is the canonical implementation for run file discovery, used by
    both CLI (logs commands) and orchestrator (session resume).

    Args:
        repo_path: Repository path to search in. If None, uses current directory.

    Returns:
        List of JSON file paths sorted by filename descending (newest first).
    """
    effective_repo = repo_path if repo_path is not None else Path.cwd()
    repo_runs_dir = get_repo_runs_dir(effective_repo)
    if not repo_runs_dir.exists():
        return []
    files = list(repo_runs_dir.glob("*.json"))
    # Sort by filename descending (timestamps in filenames give newest first)
    return sorted(files, key=lambda p: p.name, reverse=True)


def load_runs(
    files: list[Path],
    limit: int | None = None,
    on_corrupt: CorruptFileCallback | None = None,
) -> list[tuple[dict[str, Any], Path]]:
    """Load valid run metadata from files.

    Files are assumed to be pre-sorted by filename descending (newest first),
    allowing early termination once `limit` valid runs are collected.

    To preserve correctness when multiple runs share the same second-resolution
    timestamp, this function continues scanning files with the same timestamp
    prefix as the last-collected file, then returns all candidates for final
    sorting and slicing by the caller.

    Args:
        files: List of JSON file paths (pre-sorted newest first).
        limit: Maximum number of runs to collect. None means collect all.
        on_corrupt: Optional callback invoked when a file is corrupt or invalid.
            Receives (path, exception) where exception is None for validation
            failures and the actual exception for parse/IO errors. If not provided,
            corrupt files are logged via _logger.warning.

    Returns:
        List of (run_data, path) tuples. Caller must apply final sort and
        slice to limit for correctness.
    """
    # Guard against invalid limit values
    if limit is not None and limit <= 0:
        return []

    runs: list[tuple[dict[str, Any], Path]] = []
    boundary_prefix: str | None = None

    for path in files:
        # If we've reached limit, only continue for files with same timestamp prefix
        if limit is not None and len(runs) >= limit:
            if boundary_prefix is None:
                # Extract timestamp prefix from the last-collected run's file
                boundary_prefix = _extract_timestamp_prefix(runs[-1][1].name)

            current_prefix = _extract_timestamp_prefix(path.name)
            if current_prefix != boundary_prefix:
                # Different timestamp prefix, safe to stop
                break

        data = _parse_run_file(path, on_corrupt=on_corrupt)
        if data is not None:
            runs.append((data, path))

    return runs


def _get_repo_path_from_run(run: dict[str, Any], metadata_path: Path) -> str | None:
    """Get repo path from run metadata, preferring stored value.

    Falls back to the encoded directory name if repo_path is not in metadata.

    Args:
        run: Run metadata dict.
        metadata_path: Path to the metadata file (for fallback).

    Returns:
        Repo path string or None.
    """
    stored_path = run.get("repo_path")
    if isinstance(stored_path, str):
        return stored_path
    parent_name = metadata_path.parent.name
    return parent_name if parent_name else None


@dataclass
class SessionInfo:
    """Information about a session for a specific issue.

    Returned by find_sessions_for_issue() with all relevant metadata.
    """

    run_id: str
    session_id: str | None
    issue_id: str
    run_started_at: str
    started_at_ts: float  # Epoch timestamp for sorting
    status: str | None
    log_path: str | None
    metadata_path: Path
    repo_path: str | None
    baseline_timestamp: int | None = None
    last_review_issues: list[dict[str, Any]] | None = None


def extract_session_from_run(
    data: dict[str, Any], path: Path, issue_id: str
) -> SessionInfo | None:
    """Extract session info for a specific issue from run metadata.

    This is the canonical implementation for session extraction, used by
    both find_sessions_for_issue and CLI's _find_sessions_all_repos.

    Args:
        data: Parsed run metadata dict.
        path: Path to the metadata file.
        issue_id: The issue ID to extract.

    Returns:
        SessionInfo if the issue exists in the run, None otherwise.
    """
    issues = data.get("issues", {})
    if not isinstance(issues, dict):
        return None

    issue_data = issues.get(issue_id)
    if not isinstance(issue_data, dict):
        return None

    started_at = data.get("started_at") or ""
    # Validate session_id is a string (or None)
    raw_session_id = issue_data.get("session_id")
    session_id = raw_session_id if isinstance(raw_session_id, str) else None

    # Extract last_review_issues (list of dicts or None)
    # Filter out non-dict items to guard against corrupted metadata
    raw_review_issues = issue_data.get("last_review_issues")
    if isinstance(raw_review_issues, list):
        last_review_issues = [
            item for item in raw_review_issues if isinstance(item, dict)
        ] or None  # Return None if all items were filtered out
    else:
        last_review_issues = None

    return SessionInfo(
        run_id=data.get("run_id") or "",
        session_id=session_id,
        issue_id=issue_id,
        run_started_at=started_at,
        started_at_ts=parse_timestamp(started_at),
        status=issue_data.get("status"),
        log_path=issue_data.get("log_path"),
        metadata_path=path,
        repo_path=_get_repo_path_from_run(data, path),
        baseline_timestamp=(
            issue_data.get("baseline_timestamp")
            if isinstance(issue_data.get("baseline_timestamp"), int)
            else None
        ),
        last_review_issues=last_review_issues,
    )


def find_sessions_for_issue(
    repo_path: Path | None,
    issue_id: str,
    on_corrupt: CorruptFileCallback | None = None,
) -> list[SessionInfo]:
    """Find all sessions for a specific issue.

    This is the canonical implementation for session lookup, used by
    both CLI (logs sessions) and orchestrator (session resume).

    Args:
        repo_path: Repository path. If None, uses current directory.
        issue_id: The issue ID to filter by (exact match).
        on_corrupt: Optional callback invoked when a file is corrupt or invalid.
            Receives (path, exception) where exception is None for validation
            failures and the actual exception for parse/IO errors.

    Returns:
        List of SessionInfo sorted by run_started_at descending, then run_id
        ascending for ties.
    """
    files = discover_run_files(repo_path)
    sessions: list[SessionInfo] = []

    for path in files:
        data = _parse_run_file(path, on_corrupt=on_corrupt)
        if data is None:
            continue

        session = extract_session_from_run(data, path, issue_id)
        if session is not None:
            sessions.append(session)

    # Sort by started_at descending, then run_id ascending for ties
    return sorted(
        sessions,
        key=lambda s: (-s.started_at_ts, s.run_id),
    )


def lookup_prior_session_info(repo_path: Path, issue_id: str) -> SessionInfo | None:
    """Look up the most recent session info from prior runs on this issue.

    Scans run metadata files in the repo's runs directory, finds entries
    for the given issue, and returns the SessionInfo from the most recent
    run (sorted by started_at timestamp descending, with run_id as tiebreaker).

    Files are sorted by filename descending before scanning (leveraging the
    timestamp prefix in filenames for efficiency). Includes early-exit
    optimization: stops scanning when files are guaranteed to be older than
    the best match found.

    Args:
        repo_path: Repository path for finding run metadata.
        issue_id: The issue ID to look up.

    Returns:
        SessionInfo from the most recent run on this issue, or None if not found.
    """
    runs_dir = get_repo_runs_dir(repo_path)
    if not runs_dir.exists():
        return None

    # Sort files by filename descending (newest first based on timestamp prefix)
    json_files = sorted(runs_dir.glob("*.json"), key=lambda p: p.name, reverse=True)

    # Track best match: (started_at_ts, run_id, session_info)
    best: tuple[float, str, SessionInfo] | None = None

    for json_path in json_files:
        # Early-exit: if we have a match and current file's timestamp prefix
        # is older than our best, we can stop (files are sorted newest-first)
        if best is not None:
            current_prefix = _extract_timestamp_prefix(json_path.name)
            if current_prefix:
                try:
                    file_ts = datetime.strptime(
                        current_prefix, "%Y-%m-%dT%H-%M-%S"
                    ).replace(tzinfo=UTC)
                    # Compare at second granularity
                    if file_ts.timestamp() < int(best[0]):
                        break  # All remaining files are older
                except ValueError:
                    pass  # Non-standard filename, continue scanning

        data = _parse_run_file(json_path)
        if data is None:
            continue

        session_info = extract_session_from_run(data, json_path, issue_id)
        if session_info is None or not session_info.session_id:
            continue

        # Parse started_at for sorting
        started_at = data.get("started_at") or ""
        run_id = data.get("run_id") or ""
        timestamp = parse_timestamp(started_at)

        # Update best if this is newer, or same timestamp with smaller run_id
        if (
            best is None
            or timestamp > best[0]
            or (timestamp == best[0] and run_id < best[1])
        ):
            best = (timestamp, run_id, session_info)

    return best[2] if best else None


def lookup_prior_session(repo_path: Path, issue_id: str) -> str | None:
    """Look up the session ID from a prior run on this issue."""
    info = lookup_prior_session_info(repo_path, issue_id)
    return info.session_id if info is not None else None


def parse_timestamp(ts: str) -> float:
    """Parse ISO timestamp to epoch float for sorting.

    This is the canonical implementation for timestamp parsing across
    the codebase. Used by both run_metadata.py and cli/logs.py.

    Args:
        ts: ISO format timestamp string (with Z or +00:00 suffix).

    Returns:
        Epoch timestamp as float, or 0.0 if parsing fails.
    """
    try:
        # Handle both Z suffix and +00:00
        ts = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(ts)
        return dt.timestamp()
    except (ValueError, TypeError, AttributeError):
        return 0.0

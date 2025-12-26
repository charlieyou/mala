"""Run metadata tracking for mala orchestrator runs.

Captures orchestrator configuration, issue results, and pointers to Claude logs.
Replaces the duplicate JSONL logging with structured run metadata.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ..tools.env import RUNS_DIR


@dataclass
class QualityGateResult:
    """Quality gate check result for an issue."""

    passed: bool
    evidence: dict[str, bool] = field(default_factory=dict)
    failure_reasons: list[str] = field(default_factory=list)


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
    codex_review: bool | None = None


class RunMetadata:
    """Tracks metadata for a single orchestrator run.

    Creates a JSON file at ~/.config/mala/runs/{run_id}.json containing:
    - Run configuration
    - Per-issue results with Claude log path pointers
    - Quality gate outcomes
    - Timing and error information
    """

    def __init__(
        self,
        repo_path: Path,
        config: RunConfig,
        version: str,
    ):
        self.run_id = str(uuid.uuid4())
        self.started_at = datetime.now(timezone.utc)
        self.completed_at: datetime | None = None
        self.repo_path = repo_path
        self.config = config
        self.version = version
        self.issues: dict[str, IssueRun] = {}

    def record_issue(self, issue: IssueRun) -> None:
        """Record the result of an issue run."""
        self.issues[issue.issue_id] = issue

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
                }
                for issue_id, issue in self.issues.items()
            },
        }

    def save(self) -> Path:
        """Save run metadata to JSON file.

        Returns:
            Path to the saved metadata file.
        """
        self.completed_at = datetime.now(timezone.utc)
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        path = RUNS_DIR / f"{self.run_id}.json"
        with open(path, "w") as f:
            json.dump(self._to_dict(), f, indent=2)
        return path

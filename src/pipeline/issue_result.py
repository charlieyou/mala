"""Issue result dataclass for MalaOrchestrator.

This module contains the IssueResult dataclass which holds the result
of a single issue implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

    from src.core.models import IssueResolution
    from src.core.protocols.review import ReviewIssueProtocol
    from src.core.session_end_result import SessionEndResult


@dataclass
class IssueResult:
    """Result from a single issue implementation."""

    issue_id: str
    agent_id: str
    success: bool
    summary: str
    duration_seconds: float = 0.0
    session_id: str | None = None  # Claude SDK session ID
    gate_attempts: int = 1  # Number of gate retry attempts
    review_attempts: int = 0  # Number of Codex review attempts
    resolution: IssueResolution | None = None  # Resolution outcome if using markers
    low_priority_review_issues: list[ReviewIssueProtocol] | None = None  # P2/P3 issues
    session_log_path: Path | None = None  # Path to session log file
    review_log_path: str | None = None  # Path to Cerberus review session log
    baseline_timestamp: int | None = None  # Commit freshness baseline for this run
    last_review_issues: list[dict[str, Any]] | None = None  # Review issues for resume
    is_epic: bool = (
        False  # True if this is an epic issue (for periodic trigger filtering)
    )
    base_sha: str | None = None  # Git SHA at issue session start (before agent writes)
    session_end_result: SessionEndResult | None = None  # Session end validation result

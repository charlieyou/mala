"""Issue result dataclass for MalaOrchestrator.

This module contains the IssueResult dataclass which holds the result
of a single issue implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.models import IssueResolution
    from src.core.protocols import ReviewIssueProtocol


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

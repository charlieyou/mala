"""SessionEndResult and SessionEndRetryState dataclasses.

These dataclasses hold session_end execution state and results per spec R5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from datetime import datetime

    from src.infra.tools.command_runner import CommandResult


@dataclass
class CodeReviewResult:
    """Result of session_end code review.

    Represents whether code review ran, its outcome, and any findings.
    """

    ran: bool
    passed: bool | None  # None if ran=False
    findings: list[dict[str, object]] = field(default_factory=list)


@dataclass
class SessionEndRetryState:
    """State for session_end remediation retries.

    Tracks retry progress when failure_mode is remediate.

    Attributes:
        attempt: Current attempt number (1-indexed, 1 = initial run).
        max_retries: Maximum number of retry cycles after initial failure.
        log_offset: Byte offset in log file for resuming on crash.
        previous_commit_hash: Last known good commit before remediation.
    """

    attempt: int = 1
    max_retries: int = 0
    log_offset: int = 0
    previous_commit_hash: str | None = None


@dataclass
class SessionEndResult:
    """Result of session_end trigger execution.

    Holds the outcome of per-issue session_end validation including
    command results and optional code review.

    Per spec R5, this is passed to review for evidence but review
    does not auto-fail based on status.

    Attributes:
        status: Outcome of session_end execution.
        started_at: When session_end started (null only when skipped).
        finished_at: When session_end completed (null only when skipped).
        commands: Command execution results (empty if skipped or partial discarded).
        code_review_result: Code review outcome if configured.
        reason: Explanation for non-pass status (e.g., "gate_failed", "not_configured").
    """

    status: Literal["pass", "fail", "timeout", "interrupted", "skipped"]
    started_at: datetime | None = None
    finished_at: datetime | None = None
    commands: list[CommandResult] = field(default_factory=list)
    code_review_result: CodeReviewResult | None = None
    reason: str | None = None

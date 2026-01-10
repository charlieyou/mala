"""SessionEndResult and SessionEndRetryState dataclasses.

These dataclasses hold session_end execution state and results per spec R5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from datetime import datetime


@dataclass
class CommandOutcome:
    """Spec-compliant command execution outcome.

    Per spec R5, commands in SessionEndResult use this summarized schema:
    {ref, passed, duration_seconds, error_message}

    Attributes:
        ref: Command reference name from config (e.g., "test", "lint").
        passed: Whether the command succeeded (returncode == 0).
        duration_seconds: How long the command took.
        error_message: Error output when failed (stderr or timeout message), None if passed.
    """

    ref: str
    passed: bool
    duration_seconds: float
    error_message: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert to a JSON-serializable dict."""
        return {
            "ref": self.ref,
            "passed": self.passed,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
        }


@dataclass
class CodeReviewResult:
    """Result of session_end code review.

    Represents whether code review ran, its outcome, and any findings.
    """

    ran: bool
    passed: bool | None  # None if ran=False
    findings: list[dict[str, object]] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Convert to a JSON-serializable dict."""
        return {
            "ran": self.ran,
            "passed": self.passed,
            "findings": self.findings,
        }


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

    def to_dict(self) -> dict[str, object]:
        """Convert to a JSON-serializable dict."""
        return {
            "attempt": self.attempt,
            "max_retries": self.max_retries,
            "log_offset": self.log_offset,
            "previous_commit_hash": self.previous_commit_hash,
        }


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
    commands: list[CommandOutcome] = field(default_factory=list)
    code_review_result: CodeReviewResult | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Convert to a JSON-serializable dict.

        Converts datetime fields to ISO format strings and nested objects
        to their dict representations. Commands use spec R5 schema:
        {ref, passed, duration_seconds, error_message}.
        """
        return {
            "status": self.status,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "commands": [cmd.to_dict() for cmd in self.commands],
            "code_review_result": (
                self.code_review_result.to_dict() if self.code_review_result else None
            ),
            "reason": self.reason,
        }

"""Shared domain-agnostic dataclasses for mala.

This module provides shared types that are used across multiple domains
(logging, validation, orchestrator) to avoid circular dependencies.

Types:
- ResolutionOutcome: Enum for issue resolution outcomes
- IssueResolution: Records how an issue was resolved
- ValidationArtifacts: Record of validation outputs for observability
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path


class ResolutionOutcome(Enum):
    """Outcome of issue resolution."""

    SUCCESS = "success"
    NO_CHANGE = "no_change"
    OBSOLETE = "obsolete"
    ALREADY_COMPLETE = "already_complete"


@dataclass(frozen=True)
class IssueResolution:
    """Records how an issue was resolved.

    Attributes:
        outcome: The resolution outcome (success, no_change, obsolete).
        rationale: Explanation for the resolution.
    """

    outcome: ResolutionOutcome
    rationale: str


@dataclass
class ValidationArtifacts:
    """Record of validation outputs for observability.

    Attributes:
        log_dir: Directory containing validation logs.
        worktree_path: Path to the worktree (if any).
        worktree_state: State of the worktree ("kept" or "removed").
        coverage_report: Path to coverage report.
        e2e_fixture_path: Path to E2E fixture directory.
    """

    log_dir: Path
    worktree_path: Path | None = None
    worktree_state: Literal["kept", "removed"] | None = None
    coverage_report: Path | None = None
    e2e_fixture_path: Path | None = None

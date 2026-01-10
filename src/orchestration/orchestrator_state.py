"""Mutable state for a single orchestration run.

This module contains the OrchestratorState dataclass which encapsulates
per-run mutable state extracted from MalaOrchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from src.pipeline.issue_result import IssueResult


@dataclass
class OrchestratorState:
    """Encapsulates mutable state for a single orchestration run.

    Created at the start of run() and passed to methods that need state.
    Pure data container - no behavior, no locks.

    Attributes:
        agent_ids: Maps issue_id to agent_id for active/completed agents.
        completed: List of IssueResult objects from completed issues.
        active_session_log_paths: Maps issue_id to session log path for deadlock handling.
        deadlock_cleaned_agents: Tracks agent IDs cleaned during deadlock resolution.
        deadlock_victim_issues: Tracks issue IDs killed due to deadlock resolution.
        run_start_commit: HEAD commit captured at run start for baseline tracking.
        issue_base_shas: Maps issue_id to base_sha (HEAD at issue session start).
    """

    agent_ids: dict[str, str] = field(default_factory=dict)
    completed: list[IssueResult] = field(default_factory=list)
    active_session_log_paths: dict[str, Path] = field(default_factory=dict)
    deadlock_cleaned_agents: set[str] = field(default_factory=set)
    deadlock_victim_issues: set[str] = field(default_factory=set)
    run_start_commit: str | None = None
    issue_base_shas: dict[str, str] = field(default_factory=dict)

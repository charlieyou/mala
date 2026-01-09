"""CumulativeReviewRunner: Orchestrates cumulative code review execution.

This module handles cumulative code review that runs at trigger points
(epic_completion, run_end) to review accumulated changes since a baseline.

Key responsibilities:
- Determine baseline commit based on trigger type and config
- Generate diff between baseline and HEAD
- Execute code review via ReviewRunner
- Return structured results for orchestrator integration

Design principles:
- Protocol-based dependencies for testability
- Explicit input/output types for clarity
- Separation from trigger execution logic
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import asyncio
    from pathlib import Path

    from src.domain.validation.config import CodeReviewConfig, TriggerType
    from src.infra.clients.beads_client import BeadsClient
    from src.infra.git_utils import DiffStat
    from src.infra.io.log_output.run_metadata import RunMetadata
    from src.pipeline.review_runner import ReviewRunner

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReviewFinding:
    """A single finding from cumulative review.

    Attributes:
        file: Path to the file with the finding.
        line_start: Starting line number.
        line_end: Ending line number.
        priority: Finding priority (P0-P3).
        title: Short description of the finding.
        body: Detailed explanation.
        reviewer: Name of the reviewer that found this.
    """

    file: str
    line_start: int
    line_end: int
    priority: int
    title: str
    body: str
    reviewer: str


@dataclass(frozen=True)
class DiffResult:
    """Result of diff generation between commits.

    Attributes:
        baseline_commit: The base commit SHA.
        head_commit: The target commit SHA.
        stat: Statistics about the diff.
        content: The actual diff content (may be truncated for large diffs).
        truncated: Whether the diff was truncated due to size.
    """

    baseline_commit: str
    head_commit: str
    stat: DiffStat
    content: str
    truncated: bool


@dataclass(frozen=True)
class CumulativeReviewResult:
    """Result of cumulative code review execution.

    Attributes:
        status: Overall result - "success", "skipped", or "failed".
        findings: Tuple of review findings (empty if skipped/failed).
        new_baseline_commit: Commit to use as next baseline, or None if skipped.
        skip_reason: Explanation if status is "skipped".
    """

    status: Literal["success", "skipped", "failed"]
    findings: tuple[ReviewFinding, ...]
    new_baseline_commit: str | None
    skip_reason: str | None = None


class CumulativeReviewRunner:
    """Orchestrates cumulative code review execution.

    This class handles the full review workflow for cumulative reviews:
    1. Determine baseline commit from trigger type and config
    2. Generate diff between baseline and HEAD
    3. Execute code review via ReviewRunner
    4. Return structured results for orchestrator integration

    Usage:
        runner = CumulativeReviewRunner(
            review_runner=review_runner,
            git_utils=git_utils,
            beads_client=beads_client,
            logger=logger,
        )
        result = await runner.run_review(
            trigger_type=TriggerType.EPIC_COMPLETION,
            config=code_review_config,
            run_metadata=run_metadata,
            repo_path=repo_path,
            interrupt_event=interrupt_event,
            epic_id="epic-123",
        )
    """

    LARGE_DIFF_THRESHOLD = 5000  # lines

    def __init__(
        self,
        review_runner: ReviewRunner,
        beads_client: BeadsClient,
        logger: logging.Logger,
    ) -> None:
        """Initialize CumulativeReviewRunner.

        Args:
            review_runner: ReviewRunner for executing code reviews.
            beads_client: Client for beads issue operations.
            logger: Logger instance for this runner.
        """
        self._review_runner = review_runner
        self._beads_client = beads_client
        self._logger = logger

    async def run_review(
        self,
        trigger_type: TriggerType,
        config: CodeReviewConfig,
        run_metadata: RunMetadata,
        repo_path: Path,
        interrupt_event: asyncio.Event,
        *,
        issue_id: str | None = None,
        epic_id: str | None = None,
    ) -> CumulativeReviewResult:
        """Run cumulative code review for accumulated changes.

        Args:
            trigger_type: The type of trigger firing (epic_completion, run_end).
            config: Code review configuration from trigger.
            run_metadata: Current run metadata with baseline tracking.
            repo_path: Path to the git repository.
            interrupt_event: Event to check for SIGINT interruption.
            issue_id: Optional issue ID for context.
            epic_id: Optional epic ID (required for epic_completion triggers).

        Returns:
            CumulativeReviewResult with status, findings, and new baseline.

        Raises:
            NotImplementedError: This method is a skeleton pending T007-T008.
        """
        raise NotImplementedError(
            "CumulativeReviewRunner.run_review not yet implemented"
        )

    def _get_baseline_commit(
        self,
        trigger_type: TriggerType,
        config: CodeReviewConfig,
        run_metadata: RunMetadata,
        *,
        epic_id: str | None = None,
    ) -> str | None:
        """Determine the baseline commit for cumulative review.

        Logic depends on config.baseline setting:
        - "since_run_start": Use run_metadata.run_start_commit
        - "since_last_review": Use last_cumulative_review_commits[key]

        The key format is:
        - "run_end" for TriggerType.RUN_END
        - "epic_completion:<epic_id>" for TriggerType.EPIC_COMPLETION

        Args:
            trigger_type: The type of trigger firing.
            config: Code review configuration.
            run_metadata: Current run metadata.
            epic_id: Epic ID for epic_completion triggers.

        Returns:
            Baseline commit SHA, or None if no baseline available.

        Raises:
            NotImplementedError: This method is a skeleton pending T007.
        """
        raise NotImplementedError()

    def _generate_diff(
        self,
        baseline_commit: str,
        head_commit: str,
        repo_path: Path,
    ) -> DiffResult:
        """Generate diff between baseline and HEAD commits.

        Args:
            baseline_commit: The base commit SHA.
            head_commit: The target commit SHA (typically HEAD).
            repo_path: Path to the git repository.

        Returns:
            DiffResult with statistics and content.

        Raises:
            NotImplementedError: This method is a skeleton pending T008.
        """
        raise NotImplementedError()

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

    from src.core.protocols import IssueProvider, ReviewRunnerProtocol
    from src.domain.validation.config import CodeReviewConfig, TriggerType
    from src.infra.git_utils import DiffStat, GitUtils
    from src.infra.io.log_output.run_metadata import RunMetadata


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
class BaselineResult:
    """Result of baseline commit determination.

    Attributes:
        commit: The baseline commit SHA, or None if baseline cannot be determined.
        skip_reason: Explanation if commit is None (e.g., shallow clone).
    """

    commit: str | None
    skip_reason: str | None = None


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
        review_runner: ReviewRunnerProtocol,
        git_utils: GitUtils,
        beads_client: IssueProvider,
        logger: logging.Logger,
    ) -> None:
        """Initialize CumulativeReviewRunner.

        Args:
            review_runner: ReviewRunnerProtocol for executing code reviews.
            git_utils: Git operations provider for diff generation.
            beads_client: IssueProvider for beads issue operations.
            logger: Logger instance for this runner.
        """
        self._review_runner = review_runner
        self._git_utils = git_utils
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

    async def _get_baseline_commit(
        self,
        trigger_type: TriggerType,
        config: CodeReviewConfig,
        run_metadata: RunMetadata,
        *,
        issue_id: str | None = None,
        epic_id: str | None = None,
    ) -> BaselineResult:
        """Determine the baseline commit for cumulative review.

        Logic depends on trigger type and config.baseline setting:

        For session_end:
        - Calls get_baseline_for_issue() to find parent of first issue commit

        For epic_completion / run_end with "since_run_start":
        - Use run_metadata.run_start_commit
        - Fallback if None: log warning, return current HEAD

        For epic_completion / run_end with "since_last_review":
        - Look up run_metadata.last_cumulative_review_commits[key]
        - Key format: "run_end" or "epic_completion:<epic_id>"
        - Fallback if not found: use since_run_start behavior

        Reachability check:
        - After determining baseline, verifies commit is reachable locally
        - In shallow clones, baseline may not exist locally
        - Returns skip_reason="baseline_not_reachable" if unreachable

        Args:
            trigger_type: The type of trigger firing.
            config: Code review configuration.
            run_metadata: Current run metadata.
            issue_id: Issue ID for session_end triggers.
            epic_id: Epic ID for epic_completion triggers.

        Returns:
            BaselineResult with commit SHA and optional skip_reason.
        """
        from src.domain.validation.config import TriggerType as TT

        baseline: str | None = None

        # session_end: Use issue-specific baseline from git history
        if trigger_type == TT.SESSION_END:
            if not issue_id:
                self._logger.warning(
                    "session_end trigger without issue_id, cannot determine baseline"
                )
                return BaselineResult(
                    commit=None, skip_reason="session_end trigger missing issue_id"
                )
            baseline = await self._git_utils.get_baseline_for_issue(issue_id)
            if baseline is None:
                self._logger.debug(
                    "No commits found for issue %s, skipping review", issue_id
                )
                return BaselineResult(
                    commit=None,
                    skip_reason=f"no commits found for issue {issue_id}",
                )
        else:
            # epic_completion / run_end: Use baseline mode from config
            baseline_mode = config.baseline or "since_run_start"

            if baseline_mode == "since_last_review":
                # Build lookup key
                if trigger_type == TT.EPIC_COMPLETION:
                    if not epic_id:
                        self._logger.warning(
                            "epic_completion trigger without epic_id, "
                            "falling back to since_run_start"
                        )
                    else:
                        key = f"epic_completion:{epic_id}"
                        baseline = run_metadata.last_cumulative_review_commits.get(key)
                        if baseline:
                            self._logger.debug(
                                "Using last review baseline for %s: %s", key, baseline
                            )
                        else:
                            self._logger.debug(
                                "No previous review for %s, "
                                "falling back to since_run_start",
                                key,
                            )
                elif trigger_type == TT.RUN_END:
                    key = "run_end"
                    baseline = run_metadata.last_cumulative_review_commits.get(key)
                    if baseline:
                        self._logger.debug(
                            "Using last review baseline for %s: %s", key, baseline
                        )
                    else:
                        self._logger.debug(
                            "No previous review for run_end, "
                            "falling back to since_run_start"
                        )
                # Fall through to since_run_start behavior if baseline not set

            # since_run_start (or fallback): Use run_start_commit
            if baseline is None and run_metadata.run_start_commit:
                baseline = run_metadata.run_start_commit

            # Fallback: Capture current HEAD (resume case without run_start_commit)
            if baseline is None:
                self._logger.warning(
                    "run_start_commit not set (possible resume), "
                    "capturing HEAD as baseline"
                )
                head = await self._git_utils.get_head_commit()
                baseline = head if head else None

        # Handle case where baseline is still None
        if baseline is None:
            return BaselineResult(
                commit=None, skip_reason="could not determine baseline commit"
            )

        # Check reachability (important for shallow clones)
        if not await self._git_utils.is_commit_reachable(baseline):
            self._logger.warning(
                "Baseline commit %s is not reachable (shallow clone?), skipping review",
                baseline,
            )
            return BaselineResult(
                commit=None,
                skip_reason=f"baseline commit {baseline} not reachable (shallow clone)",
            )

        return BaselineResult(commit=baseline)

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

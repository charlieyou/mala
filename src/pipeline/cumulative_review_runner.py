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

import hashlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from src.domain.validation.config import FailureMode

if TYPE_CHECKING:
    import asyncio
    from pathlib import Path

    from src.core.protocols.issue import IssueProvider
    from src.core.protocols.review import ReviewRunnerProtocol
    from src.domain.validation.config import CodeReviewConfig, TriggerType
    from src.infra.git_utils import GitUtils
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


def _get_finding_fingerprint(finding: ReviewFinding) -> str:
    """Generate a unique fingerprint for a finding.

    Uses file path, line range, and title to create a stable hash.
    Returns a hex hash to ensure safe tag matching.
    """
    content = f"{finding.file}:{finding.line_start}:{finding.line_end}:{finding.title}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


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
        baseline_override: str | None = None,
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
            baseline_override: Optional explicit baseline commit SHA. If provided,
                skips automatic baseline detection and uses this value directly.
                Used by session_end to pass base_sha from IssueResult per R11.

        Returns:
            CumulativeReviewResult with status, findings, and new baseline.
        """
        from src.core.models import ReviewInput
        from src.domain.validation.config import TriggerType as TT

        # 1. Determine baseline (use override if provided)
        if baseline_override is not None:
            # Use explicit baseline from caller (e.g., base_sha from IssueResult)
            baseline_result = BaselineResult(commit=baseline_override, skip_reason=None)
        else:
            baseline_result = await self._get_baseline_commit(
                trigger_type, config, run_metadata, issue_id=issue_id, epic_id=epic_id
            )
        if baseline_result.commit is None:
            self._logger.info(
                "Skipping cumulative review: %s", baseline_result.skip_reason
            )
            return CumulativeReviewResult(
                status="skipped",
                findings=(),
                new_baseline_commit=None,
                skip_reason=baseline_result.skip_reason,
            )

        baseline = baseline_result.commit

        # 2. Get current HEAD first (to avoid race condition with diff stat)
        head_commit = await self._git_utils.get_head_commit()

        # 3. Get diff stat between baseline and HEAD
        diff_stat = await self._git_utils.get_diff_stat(baseline, head_commit)

        # 4. Empty diff check
        if diff_stat.total_lines == 0:
            self._logger.info("No changes since baseline %s, skipping review", baseline)
            return CumulativeReviewResult(
                status="skipped",
                findings=(),
                new_baseline_commit=None,
                skip_reason="empty_diff",
            )

        # 5. Large diff warning
        if diff_stat.total_lines > self.LARGE_DIFF_THRESHOLD:
            self._logger.warning(
                "Large diff (%d lines, %d files) - proceeding with review",
                diff_stat.total_lines,
                len(diff_stat.files_changed),
            )

        # 6. Get diff content for the review
        diff_content = await self._git_utils.get_diff_content(baseline, head_commit)

        # 7. Execute review via ReviewRunner with diff content
        review_session_id = None
        run_id = getattr(run_metadata, "run_id", None)
        if isinstance(run_id, str) and run_id:
            review_session_id = f"cumulative-{trigger_type.value}-{run_id}"

        review_input = ReviewInput(
            issue_id=issue_id or f"cumulative-{trigger_type.value}",
            repo_path=repo_path,
            commit_shas=[baseline, head_commit],
            issue_description=f"Cumulative review for {trigger_type.value}",
            diff_content=diff_content,
            claude_session_id=review_session_id,
        )

        try:
            review_output = await self._review_runner.run_review(
                review_input, interrupt_event
            )
        except Exception as e:
            self._logger.error("Review execution failed: %s", e)
            # execution_error - do NOT update baseline
            # Respect failure_mode: CONTINUE returns skipped, others return failed
            if config.failure_mode == FailureMode.CONTINUE:
                return CumulativeReviewResult(
                    status="skipped",
                    findings=(),
                    new_baseline_commit=None,
                    skip_reason=f"execution_error: {e}",
                )
            return CumulativeReviewResult(
                status="failed",
                findings=(),
                new_baseline_commit=None,
                skip_reason=f"execution_error: {e}",
            )

        # 8. Extract findings from review result
        findings: list[ReviewFinding] = []
        for issue in review_output.result.issues:
            findings.append(
                ReviewFinding(
                    file=getattr(issue, "file", "unknown"),
                    line_start=getattr(issue, "line_start", 0),
                    line_end=getattr(issue, "line_end", 0),
                    priority=getattr(issue, "priority", 2),
                    title=getattr(issue, "title", "Review finding"),
                    body=getattr(issue, "body", ""),
                    reviewer="cumulative_review",
                )
            )

        # 9. Create beads issues for findings with dedup and metadata
        # Skip if track_review_issues is disabled
        if not config.track_review_issues:
            self._logger.debug(
                "Skipping beads issue creation: track_review_issues=False"
            )
        for finding in findings:
            if not config.track_review_issues:
                continue
            fingerprint = _get_finding_fingerprint(finding)
            fingerprint_tag = f"fp:{fingerprint}"

            try:
                # Dedup: Check if finding already exists
                existing_id = await self._beads_client.find_issue_by_tag_async(
                    fingerprint_tag
                )
                if existing_id:
                    self._logger.debug(
                        "Skipping duplicate finding: %s (exists as %s)",
                        finding.title,
                        existing_id,
                    )
                    continue

                # Build location string for description
                if finding.line_start == finding.line_end or finding.line_end == 0:
                    location = (
                        f"{finding.file}:{finding.line_start}" if finding.file else ""
                    )
                else:
                    location = (
                        f"{finding.file}:{finding.line_start}-{finding.line_end}"
                        if finding.file
                        else ""
                    )

                # Build rich description with metadata
                desc_parts = [
                    "## Review Finding",
                    "",
                    f"**Priority:** P{finding.priority}",
                    f"**Trigger:** {trigger_type.value}",
                    f"**Baseline:** {baseline}..{head_commit}",
                ]
                if epic_id:
                    desc_parts.append(f"**Epic:** {epic_id}")
                if location:
                    desc_parts.append(f"**Location:** {location}")
                if issue_id:
                    desc_parts.append(f"**Source Issue:** {issue_id}")
                desc_parts.extend(
                    ["", "---", "", finding.body if finding.body else finding.title]
                )
                description = "\n".join(desc_parts)

                # Build tags with dedup fingerprint and metadata
                tags = [
                    "auto_generated",
                    "review_finding",
                    "cumulative-review",
                    f"trigger:{trigger_type.value}",
                    fingerprint_tag,
                ]
                if epic_id:
                    tags.append(f"epic:{epic_id}")

                await self._beads_client.create_issue_async(
                    title=finding.title,
                    description=description,
                    priority=f"P{finding.priority}",
                    tags=tags,
                    parent_id=epic_id,
                )
            except Exception as e:
                self._logger.warning("Failed to create beads issue: %s", e)

        # 10. Update baseline on completion (success or completed_with_findings)
        # Key format: "run_end" or "epic_completion:<epic_id>"
        if trigger_type == TT.EPIC_COMPLETION and epic_id:
            key = f"epic_completion:{epic_id}"
        elif trigger_type == TT.RUN_END:
            key = "run_end"
        else:
            key = trigger_type.value

        run_metadata.last_cumulative_review_commits[key] = head_commit
        self._logger.debug("Updated baseline for %s to %s", key, head_commit)

        # Both passed and failed reviews are "success" status - review completed
        # "failed" status is reserved for execution errors
        status: Literal["success", "skipped", "failed"] = "success"

        return CumulativeReviewResult(
            status=status,
            findings=tuple(findings),
            new_baseline_commit=head_commit,
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

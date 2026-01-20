"""IssueFinalizer: Finalization pipeline stage for issue results.

Extracted from MalaOrchestrator to separate finalization logic from orchestration.
This module handles:
- Recording issue runs to metadata
- Cleaning up session paths
- Emitting completion events
- Closing issues and triggering epic checks

The IssueFinalizer receives explicit inputs and returns explicit outputs,
making it testable without orchestrator dependencies.

Design principles:
- Callback-based dependencies for orchestrator-owned operations
- Explicit input/output types for clarity
- Reuses existing gate_metadata helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.infra.io.log_output.console import truncate_text
from src.infra.io.log_output.run_metadata import IssueRun
from src.pipeline.gate_metadata import (
    GateMetadata,
    build_gate_metadata,
    build_gate_metadata_from_logs,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from src.core.protocols.review import ReviewIssueProtocol
    from src.core.protocols.validation import GateChecker, GateResultProtocol
    from src.domain.evidence_check import GateResult
    from src.domain.validation.spec import ValidationSpec
    from src.infra.io.log_output.run_metadata import RunMetadata
    from src.pipeline.issue_result import IssueResult


@dataclass
class IssueFinalizeConfig:
    """Configuration for IssueFinalizer behavior.

    Attributes:
        track_review_issues: Whether to create tracking issues for P2/P3 findings.
    """

    track_review_issues: bool = False


@dataclass
class IssueFinalizeInput:
    """Input for issue finalization.

    Bundles all the data needed to finalize a single issue result.

    Attributes:
        issue_id: The issue being finalized.
        result: The issue result to finalize.
        run_metadata: The run metadata to record to.
        log_path: Path to the session log (if any).
        stored_gate_result: Stored gate result (if available).
        review_log_path: Path to the review log (if any).
    """

    issue_id: str
    result: IssueResult
    run_metadata: RunMetadata
    log_path: Path | None = None
    stored_gate_result: GateResult | GateResultProtocol | None = None
    review_log_path: str | None = None


@dataclass
class IssueFinalizeOutput:
    """Output from issue finalization.

    Attributes:
        success: Whether finalization completed successfully.
        closed: Whether the issue was closed.
        gate_metadata: The extracted gate metadata.
    """

    success: bool
    closed: bool
    gate_metadata: GateMetadata


@dataclass
class IssueFinalizeCallbacks:
    """Callbacks for orchestrator-owned operations during finalization.

    These callbacks allow the finalizer to trigger orchestrator operations
    without taking dependencies on orchestrator internals.

    Attributes:
        close_issue: Close an issue in beads. Returns True if closed successfully.
        mark_needs_followup: Mark an issue as needing followup in beads.
        on_issue_closed: Emit issue closed event.
        on_issue_completed: Emit issue completed event.
        trigger_epic_closure: Trigger epic closure check.
        create_tracking_issues: Create tracking issues for P2/P3 review findings.
    """

    close_issue: Callable[[str], Awaitable[bool]]
    mark_needs_followup: Callable[[str, str, Path | None], Awaitable[bool]]
    on_issue_closed: Callable[[str, str], None]
    on_issue_completed: Callable[[str, str, bool, float, str], None]
    trigger_epic_closure: Callable[[str, RunMetadata], Awaitable[None]]
    create_tracking_issues: Callable[[str, list[ReviewIssueProtocol]], Awaitable[None]]


@dataclass
class IssueFinalizer:
    """Issue finalization pipeline stage.

    This class encapsulates the finalization logic that was previously
    inline in MalaOrchestrator._finalize_issue_result. It receives
    callbacks for orchestrator-owned operations.

    The IssueFinalizer is responsible for:
    - Building gate metadata from stored results or logs
    - Closing issues and emitting events
    - Triggering epic closure checks
    - Creating tracking issues for review findings
    - Recording issue runs to metadata
    - Emitting completion events

    Attributes:
        config: Finalization configuration.
        callbacks: Callbacks for orchestrator-owned operations.
        evidence_check: Quality gate checker for fallback metadata extraction.
        per_session_spec: Validation spec for fallback metadata extraction.
    """

    config: IssueFinalizeConfig
    callbacks: IssueFinalizeCallbacks
    evidence_check: GateChecker | None = None
    per_session_spec: ValidationSpec | None = None

    async def finalize(self, input: IssueFinalizeInput) -> IssueFinalizeOutput:
        """Finalize an issue result.

        Records the result, updates metadata, and emits logs.
        Uses stored gate result to derive metadata, avoiding duplicate
        validation parsing.

        Args:
            input: The finalization input containing issue data.

        Returns:
            IssueFinalizeOutput with finalization results.
        """
        result = input.result
        issue_id = input.issue_id

        # Build gate metadata from stored result or logs
        gate_metadata = self._build_gate_metadata(input)

        # Close issue in beads on success
        # Failed issues are excluded via failed_issues set and may be retried
        closed = False
        if result.success:
            closed = await self.callbacks.close_issue(issue_id)
            if closed:
                self.callbacks.on_issue_closed(issue_id, issue_id)
                await self.callbacks.trigger_epic_closure(issue_id, input.run_metadata)

            # Create tracking issues for P2/P3 review findings (if enabled)
            if self.config.track_review_issues and result.low_priority_review_issues:
                await self.callbacks.create_tracking_issues(
                    issue_id,
                    result.low_priority_review_issues,
                )

        # Record to run metadata
        self._record_issue_run(input, gate_metadata)

        # Emit completion event and handle failure tracking
        await self._emit_completion(input)

        return IssueFinalizeOutput(
            success=True,
            closed=closed,
            gate_metadata=gate_metadata,
        )

    def _build_gate_metadata(self, input: IssueFinalizeInput) -> GateMetadata:
        """Build gate metadata from stored result or logs.

        Args:
            input: The finalization input.

        Returns:
            GateMetadata extracted from results or logs.
        """
        stored_gate_result = input.stored_gate_result
        log_path = input.log_path
        result = input.result

        if stored_gate_result is not None:
            return build_gate_metadata(stored_gate_result, result.success)
        elif (
            not result.success
            and log_path
            and log_path.exists()
            and self.evidence_check is not None
            and self.per_session_spec is not None
        ):
            return build_gate_metadata_from_logs(
                log_path,
                result.summary,
                result.success,
                self.evidence_check,
                self.per_session_spec,
            )
        else:
            return GateMetadata()

    def _record_issue_run(
        self,
        input: IssueFinalizeInput,
        gate_metadata: GateMetadata,
    ) -> None:
        """Record an issue run to the run metadata.

        Args:
            input: The finalization input.
            gate_metadata: Extracted gate metadata.
        """
        result = input.result
        log_path = input.log_path

        issue_run = IssueRun(
            issue_id=result.issue_id,
            agent_id=result.agent_id,
            status="success" if result.success else "failed",
            duration_seconds=result.duration_seconds,
            session_id=result.session_id,
            log_path=str(log_path) if log_path else None,
            evidence_check=gate_metadata.evidence_check_result,
            error=result.summary if not result.success else None,
            gate_attempts=result.gate_attempts,
            review_attempts=result.review_attempts,
            validation=gate_metadata.validation_result,
            resolution=result.resolution,
            review_log_path=input.review_log_path,
            baseline_timestamp=result.baseline_timestamp,
            last_review_issues=result.last_review_issues,
        )
        input.run_metadata.record_issue(issue_run)

    async def _emit_completion(self, input: IssueFinalizeInput) -> None:
        """Emit completion event and handle failure tracking.

        Args:
            input: The finalization input.
        """
        result = input.result
        issue_id = input.issue_id
        log_path = input.log_path

        self.callbacks.on_issue_completed(
            issue_id,
            issue_id,
            result.success,
            result.duration_seconds,
            truncate_text(result.summary, 50) if result.success else result.summary,
        )
        if not result.success:
            await self.callbacks.mark_needs_followup(issue_id, result.summary, log_path)

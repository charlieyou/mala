"""IssueFinalizer: Finalization pipeline stage for issue results.

Extracted from MalaOrchestrator to separate finalization logic from orchestration.
This module handles:
- Recording issue runs to metadata
- Cleaning up session paths
- Emitting completion events
- Closing issues and notifying the orchestrator about successful closes

The IssueFinalizer receives explicit inputs and returns explicit outputs,
making it testable without orchestrator dependencies.

Design principles:
- Port-based dependencies for issue/event/lifecycle operations
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
    from collections.abc import Callable
    from pathlib import Path

    from src.core.protocols.review import ReviewIssueProtocol
    from src.core.protocols.events import MalaEventSink
    from src.core.protocols.issue import IssueProvider
    from src.core.protocols.issue_lifecycle_port import IssueLifecyclePort
    from src.core.protocols.validation import GateChecker, GateResultProtocol
    from src.domain.evidence_check import GateResult
    from src.domain.validation.spec import ValidationSpec
    from src.pipeline.epic_verification_coordinator import EpicVerificationCoordinator
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
class IssueFinalizer:
    """Issue finalization pipeline stage.

    This class encapsulates the finalization logic that was previously
    inline in MalaOrchestrator._finalize_issue_result. It receives
    protocol ports for issue tracking, events, and lifecycle state.

    The IssueFinalizer is responsible for:
    - Building gate metadata from stored results or logs
    - Closing issues and emitting events
    - Notifying its owner immediately after successful issue closure
    - Creating tracking issues for review findings
    - Recording issue runs to metadata
    - Emitting completion events

    Attributes:
        config: Finalization configuration.
        issue_provider: Issue tracking port.
        event_sink: Event emission port.
        issue_lifecycle: Issue lifecycle state/effect port.
        epic_verification_coordinator: Coordinator for parent epic closure checks.
        on_issue_closed_successfully: Optional callback invoked after close_async
            succeeds and any review-tracking children are created, before later
            metadata/completion bookkeeping that may raise.
        evidence_check: Quality gate checker for fallback metadata extraction.
        per_session_spec: Validation spec for fallback metadata extraction.
    """

    config: IssueFinalizeConfig
    issue_provider: IssueProvider
    event_sink: MalaEventSink
    issue_lifecycle: IssueLifecyclePort
    epic_verification_coordinator: EpicVerificationCoordinator
    on_issue_closed_successfully: Callable[[str, RunMetadata], None] | None = None
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

        # Close issue in beads on success. Epic verification is intentionally
        # scheduled by the orchestrator after this fast finalization path returns;
        # awaiting verifier/remediation work here would block the issue coordinator
        # from finalizing other completed agent tasks.
        # Failed issues are excluded via failed_issues set and may be retried
        closed = False
        if result.success:
            closed = await self.issue_provider.close_async(issue_id)
            if closed:
                self.event_sink.on_issue_closed(issue_id, issue_id)

            # Create tracking issues for P2/P3 review findings (if enabled)
            if self.config.track_review_issues and result.low_priority_review_issues:
                await self._create_tracking_issues(
                    issue_id, result.low_priority_review_issues
                )

            if closed and self.on_issue_closed_successfully is not None:
                self.on_issue_closed_successfully(issue_id, input.run_metadata)

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

        self.event_sink.on_issue_completed(
            issue_id,
            issue_id,
            result.success,
            result.duration_seconds,
            truncate_text(result.summary, 50) if result.success else result.summary,
        )
        if not result.success:
            await self.issue_provider.mark_needs_followup_async(
                issue_id, result.summary, log_path
            )

    async def _create_tracking_issues(
        self,
        issue_id: str,
        review_issues: list[ReviewIssueProtocol],
    ) -> None:
        """Create tracking issues for non-blocking review findings."""
        from src.pipeline.review_tracking import create_review_tracking_issues

        parent_epic_id = await self.issue_provider.get_parent_epic_async(issue_id)
        await create_review_tracking_issues(
            self.issue_provider,
            self.event_sink,
            issue_id,
            review_issues,
            parent_epic_id=parent_epic_id,
        )

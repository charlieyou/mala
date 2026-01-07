"""Canonical FakeGateChecker for testing.

Provides an in-memory implementation of the GateChecker protocol that enables
behavior-based testing without filesystem or git dependencies.

Observable state:
- check_with_resolution_calls: list of call arguments
- check_no_progress_calls: list of call arguments
- get_log_end_offset_calls: list of call arguments
- check_commit_exists_calls: list of call arguments
- parse_validation_evidence_calls: list of call arguments
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from src.domain.evidence_check import CommitResult, GateResult, ValidationEvidence

if TYPE_CHECKING:
    from pathlib import Path

    from src.domain.validation.spec import ValidationSpec


@dataclass
class FakeGateChecker:
    """In-memory fake for GateChecker protocol.

    Allows tests to configure gate behavior without filesystem or git.
    Tracks all method calls for verification.

    Attributes:
        gate_result: Pre-configured result for check_with_resolution().
        no_progress_result: Pre-configured result for check_no_progress().
        log_end_offset: Pre-configured result for get_log_end_offset().
        commit_result: Pre-configured result for check_commit_exists().
        validation_evidence: Pre-configured result for parse_validation_evidence_with_spec().
        check_with_resolution_calls: List of call argument dicts.
        check_no_progress_calls: List of call argument dicts.
        get_log_end_offset_calls: List of call argument dicts.
        check_commit_exists_calls: List of call argument dicts.
        parse_validation_evidence_calls: List of call argument dicts.
    """

    # Pre-configured results
    gate_result: GateResult = field(
        default_factory=lambda: GateResult(passed=True, failure_reasons=[])
    )
    no_progress_result: bool = False
    log_end_offset: int = 1000
    commit_result: CommitResult = field(
        default_factory=lambda: CommitResult(exists=True, commit_hash="abc123")
    )
    validation_evidence: ValidationEvidence = field(default_factory=ValidationEvidence)

    # Call tracking
    check_with_resolution_calls: list[dict] = field(default_factory=list)
    check_no_progress_calls: list[dict] = field(default_factory=list)
    get_log_end_offset_calls: list[dict] = field(default_factory=list)
    check_commit_exists_calls: list[dict] = field(default_factory=list)
    parse_validation_evidence_calls: list[dict] = field(default_factory=list)

    # Alias for backward compatibility
    @property
    def no_progress_calls(self) -> list[dict]:
        """Alias for check_no_progress_calls (backward compatibility)."""
        return self.check_no_progress_calls

    def check_with_resolution(
        self,
        issue_id: str,
        log_path: Path,
        baseline_timestamp: int | None = None,
        log_offset: int = 0,
        spec: ValidationSpec | None = None,
    ) -> GateResult:
        """Record call and return pre-configured gate result."""
        self.check_with_resolution_calls.append(
            {
                "issue_id": issue_id,
                "log_path": log_path,
                "baseline_timestamp": baseline_timestamp,
                "log_offset": log_offset,
                "spec": spec,
            }
        )
        return self.gate_result

    def get_log_end_offset(self, log_path: Path, start_offset: int = 0) -> int:
        """Record call and return pre-configured offset."""
        self.get_log_end_offset_calls.append(
            {"log_path": log_path, "start_offset": start_offset}
        )
        return self.log_end_offset

    def check_no_progress(
        self,
        log_path: Path,
        log_offset: int,
        previous_commit_hash: str | None,
        current_commit_hash: str | None,
        spec: ValidationSpec | None = None,
        check_validation_evidence: bool = True,
    ) -> bool:
        """Record call and return pre-configured no_progress result."""
        self.check_no_progress_calls.append(
            {
                "log_path": log_path,
                "log_offset": log_offset,
                "previous_commit_hash": previous_commit_hash,
                "current_commit_hash": current_commit_hash,
                "spec": spec,
                "check_validation_evidence": check_validation_evidence,
            }
        )
        return self.no_progress_result

    def parse_validation_evidence_with_spec(
        self, log_path: Path, spec: ValidationSpec, offset: int = 0
    ) -> ValidationEvidence:
        """Record call and return pre-configured validation evidence."""
        self.parse_validation_evidence_calls.append(
            {"log_path": log_path, "spec": spec, "offset": offset}
        )
        return self.validation_evidence

    def check_commit_exists(
        self, issue_id: str, baseline_timestamp: int | None = None
    ) -> CommitResult:
        """Record call and return pre-configured commit result."""
        self.check_commit_exists_calls.append(
            {"issue_id": issue_id, "baseline_timestamp": baseline_timestamp}
        )
        return self.commit_result

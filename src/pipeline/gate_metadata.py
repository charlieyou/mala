"""Gate metadata extraction for MalaOrchestrator.

This module contains the GateMetadata dataclass and helper functions for
extracting quality gate results for run finalization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from src.domain.evidence_check import EVIDENCE_CHECK_IGNORED_KINDS
from src.domain.validation.spec import CommandKind
from src.infra.io.log_output.run_metadata import (
    EvidenceCheckResult,
    ValidationResult as MetaValidationResult,
)

if TYPE_CHECKING:
    from pathlib import Path

    from src.core.protocols.validation import (
        GateChecker,
        GateResultProtocol,
        ValidationSpecProtocol,
    )
    from src.domain.evidence_check import GateResult
    from src.domain.validation.spec import ValidationSpec


@dataclass
class GateMetadata:
    """Metadata extracted from quality gate results for finalization.

    This dataclass holds the processed results from a quality gate check,
    separating the extraction logic from the finalization flow.
    """

    evidence_check_result: EvidenceCheckResult | None = None
    validation_result: MetaValidationResult | None = None


def build_gate_metadata(
    gate_result: GateResult | GateResultProtocol | None,
    passed: bool,
) -> GateMetadata:
    """Build GateMetadata from a stored gate result.

    Extracts evidence from the stored gate result without re-running validation.
    This is the primary path used when gate results are available from
    _run_evidence_check_sync.

    Args:
        gate_result: The stored gate result (may be None if no gate ran).
        passed: Whether the overall run passed (affects evidence_check_result.passed).

    Returns:
        GateMetadata with extracted quality gate and validation results.
    """
    if gate_result is None:
        return GateMetadata()

    evidence = gate_result.validation_evidence
    commit_hash = gate_result.commit_hash

    # Build evidence dict from stored evidence
    evidence_dict: dict[str, bool] = {}
    if evidence is not None:
        evidence_dict = {
            c.kind.value: c.seen
            for c in evidence.commands.values()
            if c.kind != CommandKind.CUSTOM
        }
    evidence_dict["commit_found"] = commit_hash is not None

    evidence_check_result = EvidenceCheckResult(
        passed=passed if passed else gate_result.passed,
        evidence=evidence_dict,
        failure_reasons=[] if passed else list(gate_result.failure_reasons),
    )

    validation_result: MetaValidationResult | None = None
    if evidence is not None:
        # Deduplicate: multiple commands may share a kind, and multiple kinds may
        # map to the same observed command (e.g., LINT+FORMAT for a single ruff run).
        commands_run = list(
            dict.fromkeys(
                c.kind.value
                for c in evidence.commands.values()
                if c.seen and c.kind != CommandKind.CUSTOM
            )
        )
        # Match legacy parser semantics: failed CUSTOM commands are surfaced
        # via the strict/advisory custom-command path, not via commands_failed.
        commands_failed = list(
            dict.fromkeys(
                c.observed_command or c.name
                for c in evidence.commands.values()
                if c.status == "failed"
                and c.kind not in EVIDENCE_CHECK_IGNORED_KINDS
                and c.kind != CommandKind.CUSTOM
            )
        )
        validation_result = MetaValidationResult(
            passed=passed if passed else gate_result.passed,
            commands_run=commands_run,
            commands_failed=commands_failed,
        )

    return GateMetadata(
        evidence_check_result=evidence_check_result,
        validation_result=validation_result,
    )


def build_gate_metadata_from_logs(
    log_path: Path,
    result_summary: str,
    result_success: bool,
    evidence_check: GateChecker,
    per_session_spec: ValidationSpec | ValidationSpecProtocol | None,
) -> GateMetadata:
    """Build GateMetadata by parsing logs directly (fallback path).

    This is a fallback path used when no stored gate result is available.
    It parses the log file directly to extract validation evidence.

    Args:
        log_path: Path to the session log file.
        result_summary: Summary from the issue result (for extracting failure reasons).
        result_success: Whether the run succeeded (determines passed status).
        evidence_check: The GateChecker instance for parsing.
        per_session_spec: ValidationSpec for parsing evidence (if None, returns empty).

    Returns:
        GateMetadata with extracted results, or empty if spec is None.
    """
    if per_session_spec is None:
        return GateMetadata()

    evidence = evidence_check.parse_validation_evidence_with_spec(
        log_path, cast("ValidationSpecProtocol", per_session_spec)
    )

    # Extract failure reasons from result summary
    failure_reasons: list[str] = []
    if "Quality gate failed:" in result_summary:
        reasons_part = result_summary.replace("Quality gate failed: ", "")
        failure_reasons = [r.strip() for r in reasons_part.split(";")]

    evidence_dict: dict[str, bool] = {
        c.kind.value: c.seen
        for c in evidence.commands.values()
        if c.kind != CommandKind.CUSTOM
    }

    # Check commit exists (we don't have stored result, so check now)
    # Note: We don't call check_commit_exists here because it's expensive
    # and this is a fallback path - just mark as unknown
    evidence_dict["commit_found"] = False

    evidence_check_result = EvidenceCheckResult(
        passed=result_success,
        evidence=evidence_dict,
        failure_reasons=failure_reasons,
    )

    # Deduplicate and exclude CUSTOM kinds to match build_gate_metadata semantics.
    # Failed CUSTOM commands are surfaced via the strict/advisory path, not here.
    commands_run = list(
        dict.fromkeys(
            c.kind.value
            for c in evidence.commands.values()
            if c.seen and c.kind != CommandKind.CUSTOM
        )
    )
    commands_failed = list(
        dict.fromkeys(
            c.observed_command or c.name
            for c in evidence.commands.values()
            if c.status == "failed"
            and c.kind not in EVIDENCE_CHECK_IGNORED_KINDS
            and c.kind != CommandKind.CUSTOM
        )
    )
    validation_result = MetaValidationResult(
        passed=result_success,
        commands_run=commands_run,
        commands_failed=commands_failed,
    )

    return GateMetadata(
        evidence_check_result=evidence_check_result,
        validation_result=validation_result,
    )

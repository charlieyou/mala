"""Backward-compatibility shim for src.quality_gate.

This module re-exports all public symbols from src.domain.quality_gate.
New code should import directly from src.domain.quality_gate.
"""

from src.domain.quality_gate import (
    QUALITY_GATE_IGNORED_COMMANDS,
    QUALITY_GATE_IGNORED_KINDS,
    CommitResult,
    GateResult,
    JsonlEntry,
    QualityGate,
    ValidationEvidence,
    check_evidence_against_spec,
    get_required_evidence_kinds,
)

# Re-export types that tests import from here (originally re-exported from validation.spec)
from src.validation.spec import CommandKind

# Re-export internal imports for test compatibility (tests patch these)
from src.tools.command_runner import run_command

__all__ = [
    "QUALITY_GATE_IGNORED_COMMANDS",
    "QUALITY_GATE_IGNORED_KINDS",
    "CommandKind",
    "CommitResult",
    "GateResult",
    "JsonlEntry",
    "QualityGate",
    "ValidationEvidence",
    "check_evidence_against_spec",
    "get_required_evidence_kinds",
    "run_command",  # For test backward compatibility
]

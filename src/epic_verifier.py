"""Backward-compatibility shim for src.epic_verifier.

This module re-exports all public symbols from src.infra.epic_verifier.
New code should import directly from src.infra.epic_verifier.
"""

from src.infra.epic_verifier import (
    EPIC_VERIFY_LOCK_TIMEOUT_SECONDS,
    LOW_CONFIDENCE_THRESHOLD,
    SPEC_PATH_PATTERNS,
    ClaudeEpicVerificationModel,
    EpicVerifier,
    _compute_criterion_hash,
    _extract_json_from_code_blocks,
    extract_spec_paths,
)

__all__ = [
    "EPIC_VERIFY_LOCK_TIMEOUT_SECONDS",
    "LOW_CONFIDENCE_THRESHOLD",
    "SPEC_PATH_PATTERNS",
    "ClaudeEpicVerificationModel",
    "EpicVerifier",
    "_compute_criterion_hash",
    "_extract_json_from_code_blocks",
    "extract_spec_paths",
]

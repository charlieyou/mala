"""Backward-compatibility shim for src.validation.spec.

This module re-exports all public symbols from src.domain.validation.spec.
New code should import directly from src.domain.validation.spec.
"""

# Re-export everything from the domain module
from src.domain.validation.spec import (
    CODE_EXTENSIONS,
    CODE_FILES,
    CODE_PATHS,
    DOC_EXTENSIONS,
    CommandKind,
    CoverageConfig,
    E2EConfig,
    IssueResolution,
    RepoType,
    ResolutionOutcome,
    ValidationArtifacts,
    ValidationCommand,
    ValidationContext,
    ValidationScope,
    ValidationSpec,
    build_validation_spec,
    classify_change,
    detect_repo_type,
)

__all__ = [
    "CODE_EXTENSIONS",
    "CODE_FILES",
    "CODE_PATHS",
    "DOC_EXTENSIONS",
    "CommandKind",
    "CoverageConfig",
    "E2EConfig",
    "IssueResolution",
    "RepoType",
    "ResolutionOutcome",
    "ValidationArtifacts",
    "ValidationCommand",
    "ValidationContext",
    "ValidationScope",
    "ValidationSpec",
    "build_validation_spec",
    "classify_change",
    "detect_repo_type",
]

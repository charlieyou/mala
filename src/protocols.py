"""Backward-compatibility shim for src.protocols.

This module re-exports all public types from src.core.protocols.
New code should import directly from src.core.protocols.
"""

from src.core.protocols import (
    CodeReviewer,
    CommitResultProtocol,
    EpicVerdictProtocol,
    EpicVerificationModel,
    GateChecker,
    GateResultProtocol,
    IssueProvider,
    IssueResolutionProtocol,
    JsonlEntryProtocol,
    LogProvider,
    ReviewIssueProtocol,
    ReviewResultProtocol,
    UnmetCriterionProtocol,
    ValidationEvidenceProtocol,
    ValidationSpecProtocol,
)

__all__ = [
    "CodeReviewer",
    "CommitResultProtocol",
    "EpicVerdictProtocol",
    "EpicVerificationModel",
    "GateChecker",
    "GateResultProtocol",
    "IssueProvider",
    "IssueResolutionProtocol",
    "JsonlEntryProtocol",
    "LogProvider",
    "ReviewIssueProtocol",
    "ReviewResultProtocol",
    "UnmetCriterionProtocol",
    "ValidationEvidenceProtocol",
    "ValidationSpecProtocol",
]

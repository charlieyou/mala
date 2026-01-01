"""Backward-compatibility shim for src.cerberus_review.

This module re-exports all public symbols from src.infra.clients.cerberus_review.
New code should import directly from src.infra.clients.cerberus_review.
"""

from src.infra.clients.cerberus_review import (
    DefaultReviewer,
    ReviewIssue,
    ReviewResult,
    _to_relative_path,
    format_review_issues,
    map_exit_code_to_result,
    parse_cerberus_json,
)

__all__ = [
    "DefaultReviewer",
    "ReviewIssue",
    "ReviewResult",
    "_to_relative_path",
    "format_review_issues",
    "map_exit_code_to_result",
    "parse_cerberus_json",
]

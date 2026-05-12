"""External service clients for mala.

This package contains clients for external services:
- anthropic_client: Shared Anthropic client factory
- beads_client: BeadsClient wrapper for bd CLI
- cerberus_review: Cerberus review-gate adapter
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.infra.clients.anthropic_client import create_anthropic_client
from src.infra.clients.beads_client import BeadsClient
from src.infra.clients.cerberus_output_parser import (
    ReviewIssue,
    ReviewResult,
    map_exit_code_to_result,
    parse_cerberus_json,
)

if TYPE_CHECKING:
    from src.infra.clients.cerberus_review import DefaultReviewer

__all__ = [
    "BeadsClient",
    "DefaultReviewer",
    "ReviewIssue",
    "ReviewResult",
    "create_anthropic_client",
    "format_review_issues",
    "map_exit_code_to_result",
    "parse_cerberus_json",
]


def __getattr__(name: str) -> object:
    if name in {"DefaultReviewer", "format_review_issues"}:
        from src.infra.clients.cerberus_review import (
            DefaultReviewer,
            format_review_issues,
        )

        return {
            "DefaultReviewer": DefaultReviewer,
            "format_review_issues": format_review_issues,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

"""External service clients for mala.

This package contains clients for external services:
- anthropic_client: Shared Anthropic client factory
- beads_client: BeadsClient wrapper for bd CLI
- braintrust_integration: Braintrust tracing integration
- cerberus_review: Cerberus review-gate adapter
"""

from src.infra.clients.anthropic_client import create_anthropic_client
from src.infra.clients.beads_client import BeadsClient
from src.infra.clients.braintrust_integration import (
    BraintrustProvider,
    BraintrustSpan,
    TracedAgentExecution,
    flush_braintrust,
    is_braintrust_enabled,
)
from src.infra.clients.cerberus_review import (
    DefaultReviewer,
    ReviewIssue,
    ReviewResult,
    format_review_issues,
    map_exit_code_to_result,
    parse_cerberus_json,
)

__all__ = [
    "BeadsClient",
    "BraintrustProvider",
    "BraintrustSpan",
    "DefaultReviewer",
    "ReviewIssue",
    "ReviewResult",
    "TracedAgentExecution",
    "create_anthropic_client",
    "flush_braintrust",
    "format_review_issues",
    "is_braintrust_enabled",
    "map_exit_code_to_result",
    "parse_cerberus_json",
]

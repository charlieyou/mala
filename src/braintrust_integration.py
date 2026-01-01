"""Backward-compatibility shim for src.braintrust_integration.

This module re-exports all public symbols from src.infra.clients.braintrust_integration.
New code should import directly from src.infra.clients.braintrust_integration.
"""

from src.infra.clients.braintrust_integration import (
    BRAINTRUST_AVAILABLE,
    BraintrustProvider,
    BraintrustSpan,
    TracedAgentExecution,
    flush,
    flush_braintrust,
    is_braintrust_enabled,
    start_span,
)

__all__ = [
    "BRAINTRUST_AVAILABLE",
    "BraintrustProvider",
    "BraintrustSpan",
    "TracedAgentExecution",
    "flush",
    "flush_braintrust",
    "is_braintrust_enabled",
    "start_span",
]

"""Backward-compatibility shim for src.anthropic_client.

This module re-exports all public symbols from src.infra.clients.anthropic_client.
New code should import directly from src.infra.clients.anthropic_client.
"""

from src.infra.clients.anthropic_client import create_anthropic_client

__all__ = ["create_anthropic_client"]

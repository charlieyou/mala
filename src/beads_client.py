"""Backward-compatibility shim for src.beads_client.

This module re-exports all public symbols from src.infra.clients.beads_client.
New code should import directly from src.infra.clients.beads_client.
"""

from src.infra.clients.beads_client import BeadsClient

__all__ = ["BeadsClient"]

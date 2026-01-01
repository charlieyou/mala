"""Backward-compatibility shim for src.validation.e2e.

This module re-exports all public symbols from src.domain.validation.e2e.
New code should import directly from src.domain.validation.e2e.
"""

from src.domain.validation.e2e import *  # noqa: F403

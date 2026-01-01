"""Backward-compatibility shim for src.validation.spec.

This module re-exports all public symbols from src.domain.validation.spec.
New code should import directly from src.domain.validation.spec.
"""

from src.domain.validation.spec import *  # noqa: F403

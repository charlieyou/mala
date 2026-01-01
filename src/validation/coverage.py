"""Backward-compatibility shim for src.validation.coverage.

This module re-exports all public symbols from src.domain.validation.coverage.
New code should import directly from src.domain.validation.coverage.
"""

from src.domain.validation.coverage import *  # noqa: F403

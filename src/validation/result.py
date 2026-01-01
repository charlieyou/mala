"""Backward-compatibility shim for src.validation.result.

This module re-exports all public symbols from src.domain.validation.result.
New code should import directly from src.domain.validation.result.
"""

from src.domain.validation.result import *  # noqa: F403

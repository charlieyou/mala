"""Backward-compatibility shim for src.validation.lint_cache.

This module re-exports all public symbols from src.domain.validation.lint_cache.
New code should import directly from src.domain.validation.lint_cache.
"""

from src.domain.validation.lint_cache import *  # noqa: F403

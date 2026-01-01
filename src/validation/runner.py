"""Backward-compatibility shim for src.validation.runner.

This module re-exports all public symbols from src.domain.validation.runner.
New code should import directly from src.domain.validation.runner.
"""

from src.domain.validation.runner import *  # noqa: F403

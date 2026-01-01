"""Backward-compatibility shim for src.validation.spec_executor.

This module re-exports all public symbols from src.domain.validation.spec_executor.
New code should import directly from src.domain.validation.spec_executor.
"""

from src.domain.validation.spec_executor import *  # noqa: F403

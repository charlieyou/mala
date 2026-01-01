"""Backward-compatibility shim for src.validation.spec_result_builder.

This module re-exports all public symbols from src.domain.validation.spec_result_builder.
New code should import directly from src.domain.validation.spec_result_builder.
"""

from src.domain.validation.spec_result_builder import *  # noqa: F403

"""Backward-compatibility shim for src.validation.spec_workspace.

This module re-exports all public symbols from src.domain.validation.spec_workspace.
New code should import directly from src.domain.validation.spec_workspace.
"""

from src.domain.validation.spec_workspace import *  # noqa: F403

"""Backward-compatibility shim for src.validation.command_runner.

This module re-exports all public symbols from src.domain.validation.command_runner.
New code should import directly from src.domain.validation.command_runner.
"""

from src.domain.validation.command_runner import *  # noqa: F403

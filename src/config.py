"""Backward-compatibility shim for src.config.

This module re-exports all public symbols from src.infra.io.config.
New code should import directly from src.infra.io.config.
"""

from src.infra.io.config import (
    ConfigurationError,
    MalaConfig,
    _find_cerberus_bin_path,
)

__all__ = ["ConfigurationError", "MalaConfig", "_find_cerberus_bin_path"]

"""Backward-compatibility shim for src.session_log_parser.

This module re-exports all public symbols from src.infra.io.session_log_parser.
New code should import directly from src.infra.io.session_log_parser.
"""

from src.infra.io.session_log_parser import (
    FileSystemLogProvider,
    JsonlEntry,
    SessionLogParser,
)

__all__ = ["FileSystemLogProvider", "JsonlEntry", "SessionLogParser"]

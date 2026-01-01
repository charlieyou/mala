"""Backward-compatibility shim for src.issue_manager.

This module re-exports all public symbols from src.infra.issue_manager.
New code should import directly from src.infra.issue_manager.
"""

from src.infra.issue_manager import IssueManager

__all__ = ["IssueManager"]

"""Backward-compatibility shim for src.git_utils.

This module re-exports all public symbols from src.infra.git_utils.
New code should import directly from src.infra.git_utils.
"""

from src.infra.git_utils import (
    DEFAULT_GIT_TIMEOUT,
    get_baseline_for_issue,
    get_git_branch_async,
    get_git_commit_async,
    get_issue_commits_async,
)

# Also expose CommandRunner and run_command_async for tests that monkeypatch them
# Must use same source as infra module for patches to work
from src.infra.tools.command_runner import CommandRunner, run_command_async

__all__ = [
    "DEFAULT_GIT_TIMEOUT",
    "CommandRunner",
    "get_baseline_for_issue",
    "get_git_branch_async",
    "get_git_commit_async",
    "get_issue_commits_async",
    "run_command_async",
]

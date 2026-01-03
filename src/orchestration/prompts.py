"""Prompt loading utilities for MalaOrchestrator.

This module handles loading and caching of prompt templates used by the orchestrator.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING

from src.infra.tools.env import SCRIPTS_DIR, get_lock_dir

if TYPE_CHECKING:
    from src.domain.prompts import PromptValidationCommands

# Prompt file paths
_PROMPT_DIR = Path(__file__).parent.parent / "prompts"
PROMPT_FILE = _PROMPT_DIR / "implementer_prompt.md"
REVIEW_FOLLOWUP_FILE = _PROMPT_DIR / "review_followup.md"
FIXER_PROMPT_FILE = _PROMPT_DIR / "fixer.md"


@functools.cache
def get_implementer_prompt() -> str:
    """Load implementer prompt template (cached on first use)."""
    return PROMPT_FILE.read_text()


@functools.cache
def get_review_followup_prompt() -> str:
    """Load review follow-up prompt (cached on first use)."""
    return REVIEW_FOLLOWUP_FILE.read_text()


@functools.cache
def get_fixer_prompt() -> str:
    """Load fixer prompt (cached on first use)."""
    return FIXER_PROMPT_FILE.read_text()


def format_implementer_prompt(
    issue_id: str,
    repo_path: Path,
    agent_id: str,
    validation_commands: PromptValidationCommands,
) -> str:
    """Format the implementer prompt with runtime values.

    Args:
        issue_id: The issue ID being implemented.
        repo_path: Path to the repository.
        agent_id: The agent ID for this session.
        validation_commands: Validation commands for the prompt.

    Returns:
        Formatted prompt string.
    """
    return get_implementer_prompt().format(
        issue_id=issue_id,
        repo_path=repo_path,
        lock_dir=get_lock_dir(),
        scripts_dir=SCRIPTS_DIR,
        agent_id=agent_id,
        lint_command=validation_commands.lint,
        format_command=validation_commands.format,
        typecheck_command=validation_commands.typecheck,
        test_command=validation_commands.test,
    )

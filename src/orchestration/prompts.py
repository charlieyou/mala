"""Prompt loading utilities for MalaOrchestrator.

This module handles loading and caching of prompt templates used by the orchestrator.
"""

from __future__ import annotations

import functools
from pathlib import Path

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

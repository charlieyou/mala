"""Shared prompt loading utilities.

This module centralizes prompt file loading to avoid duplication across modules.
"""

from __future__ import annotations

import functools
from pathlib import Path


# Prompt directory relative to this module
_PROMPT_DIR = Path(__file__).parent / "prompts"

# File path constants
GATE_FOLLOWUP_FILE = _PROMPT_DIR / "gate_followup.md"


@functools.cache
def get_gate_followup_prompt() -> str:
    """Load gate follow-up prompt (cached on first use)."""
    return GATE_FOLLOWUP_FILE.read_text()

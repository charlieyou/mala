"""Backward-compatibility shim for src.prompts.

This module re-exports public symbols from src.domain.prompts.
New code should import directly from src.domain.prompts.

NOTE: The path constants are defined here rather than re-exported from
src.domain.prompts because the domain module has an incorrect path
(points to src/domain/prompts/ instead of src/prompts/). This shim
provides the correct paths until the domain module is fixed.
"""

from __future__ import annotations

import functools
from pathlib import Path

# Prompt directory at src/prompts (markdown templates)
_PROMPT_DIR = Path(__file__).parent / "prompts"

# File path constants
GATE_FOLLOWUP_FILE = _PROMPT_DIR / "gate_followup.md"


@functools.cache
def get_gate_followup_prompt() -> str:
    """Load gate follow-up prompt (cached on first use)."""
    return GATE_FOLLOWUP_FILE.read_text()


__all__ = ["GATE_FOLLOWUP_FILE", "get_gate_followup_prompt"]

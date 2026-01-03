"""Shared prompt loading utilities.

This module centralizes prompt file loading to avoid duplication across modules.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.validation.config import PromptValidationCommands


# Prompt directory - points to src/prompts/ where prompt files live
_PROMPT_DIR = Path(__file__).parent.parent / "prompts"

# File path constants
GATE_FOLLOWUP_FILE = _PROMPT_DIR / "gate_followup.md"


@functools.cache
def get_gate_followup_prompt() -> str:
    """Load gate follow-up prompt (cached on first use)."""
    return GATE_FOLLOWUP_FILE.read_text()


def get_default_validation_commands() -> PromptValidationCommands:
    """Return default Python/uv validation commands with cache isolation.

    These defaults are used when no mala.yaml configuration is found.
    Commands include cache isolation flags for parallel agent runs,
    using $AGENT_ID environment variable (set in the agent environment).

    Returns:
        PromptValidationCommands with default Python/uv toolchain commands.
    """
    from src.domain.validation.config import PromptValidationCommands

    return PromptValidationCommands(
        lint="RUFF_CACHE_DIR=/tmp/ruff-${AGENT_ID:-default} uvx ruff check .",
        format="RUFF_CACHE_DIR=/tmp/ruff-${AGENT_ID:-default} uvx ruff format --check .",
        typecheck="uvx ty check",
        test="uv run pytest -o cache_dir=/tmp/pytest-${AGENT_ID:-default}",
    )


def build_prompt_validation_commands(repo_path: Path) -> PromptValidationCommands:
    """Build PromptValidationCommands for a repository.

    Loads the mala.yaml configuration, merges with preset if specified,
    and returns the validation commands formatted for prompt templates.

    Args:
        repo_path: Path to the repository root directory.

    Returns:
        PromptValidationCommands with command strings for prompt templates.
        Returns default Python/uv commands if no config is found.
    """
    from src.domain.validation.config import PromptValidationCommands
    from src.domain.validation.config_loader import ConfigMissingError, load_config
    from src.domain.validation.config_merger import merge_configs
    from src.domain.validation.preset_registry import PresetRegistry

    try:
        user_config = load_config(repo_path)
    except ConfigMissingError:
        # No config file - return defaults
        return get_default_validation_commands()

    # Load and merge preset if specified
    if user_config.preset is not None:
        registry = PresetRegistry()
        preset_config = registry.get(user_config.preset)
        merged_config = merge_configs(preset_config, user_config)
    else:
        merged_config = user_config

    return PromptValidationCommands.from_validation_config(merged_config)

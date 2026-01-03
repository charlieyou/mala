"""Shared prompt loading utilities.

This module centralizes prompt file loading to avoid duplication across modules.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.validation.config import PromptValidationCommands


@dataclass(frozen=True)
class PromptProvider:
    """Data class holding all loaded prompt templates.

    This is a pure data object constructed at startup boundary.
    All fields are immutable string contents of prompt files.
    """

    implementer_prompt: str
    review_followup_prompt: str
    gate_followup_prompt: str
    fixer_prompt: str
    idle_resume_prompt: str


def load_prompts(prompt_dir: Path) -> PromptProvider:
    """Load all prompt templates from disk.

    Args:
        prompt_dir: Directory containing prompt template files.

    Returns:
        PromptProvider with all loaded prompt templates.

    Raises:
        FileNotFoundError: If any required prompt file is missing.
    """
    return PromptProvider(
        implementer_prompt=(prompt_dir / "implementer_prompt.md").read_text(),
        review_followup_prompt=(prompt_dir / "review_followup.md").read_text(),
        gate_followup_prompt=(prompt_dir / "gate_followup.md").read_text(),
        fixer_prompt=(prompt_dir / "fixer.md").read_text(),
        idle_resume_prompt=(prompt_dir / "idle_resume.md").read_text(),
    )


# Prompt directory - points to src/prompts/ where prompt files live
_PROMPT_DIR = Path(__file__).parent.parent / "prompts"

# File path constants
GATE_FOLLOWUP_FILE = _PROMPT_DIR / "gate_followup.md"
PROMPT_FILE = _PROMPT_DIR / "implementer_prompt.md"
REVIEW_FOLLOWUP_FILE = _PROMPT_DIR / "review_followup.md"
FIXER_PROMPT_FILE = _PROMPT_DIR / "fixer.md"
IDLE_RESUME_PROMPT_FILE = _PROMPT_DIR / "idle_resume.md"


@functools.cache
def get_gate_followup_prompt() -> str:
    """Load gate follow-up prompt (cached on first use)."""
    return GATE_FOLLOWUP_FILE.read_text()


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


@functools.cache
def get_idle_resume_prompt() -> str:
    """Load idle resume prompt (cached on first use)."""
    return IDLE_RESUME_PROMPT_FILE.read_text()


def format_implementer_prompt(
    issue_id: str,
    repo_path: Path,
    agent_id: str,
    validation_commands: PromptValidationCommands,
    lock_dir: Path,
    scripts_dir: Path,
) -> str:
    """Format the implementer prompt with runtime values.

    Args:
        issue_id: The issue ID being implemented.
        repo_path: Path to the repository.
        agent_id: The agent ID for this session.
        validation_commands: Validation commands for the prompt.
        lock_dir: Directory for lock files (from infra layer).
        scripts_dir: Directory containing helper scripts (from infra layer).

    Returns:
        Formatted prompt string.
    """
    return get_implementer_prompt().format(
        issue_id=issue_id,
        repo_path=repo_path,
        lock_dir=lock_dir,
        scripts_dir=scripts_dir,
        agent_id=agent_id,
        lint_command=validation_commands.lint,
        format_command=validation_commands.format,
        typecheck_command=validation_commands.typecheck,
        test_command=validation_commands.test,
    )


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
        format="RUFF_CACHE_DIR=/tmp/ruff-${AGENT_ID:-default} uvx ruff format .",
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

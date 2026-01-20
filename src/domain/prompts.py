"""Shared prompt loading utilities.

This module centralizes prompt file loading to avoid duplication across modules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.domain.validation.config import PromptValidationCommands, ValidationConfig


def build_custom_commands_section(
    custom_commands: tuple[tuple[str, str, int, bool], ...],
) -> str:
    """Build the custom commands section for the implementer prompt.

    Each custom command is wrapped with markers for status tracking.
    Strict commands (allow_fail=False) exit with the command's status.
    Advisory commands (allow_fail=True) always exit 0.

    Args:
        custom_commands: Tuple of (name, command, timeout, allow_fail) tuples.

    Returns:
        Formatted custom commands section, or empty string if no custom commands.
    """
    if not custom_commands:
        return ""

    lines = ["# Custom validation commands (run after typecheck, before test)"]

    for name, command, timeout, allow_fail in custom_commands:
        # Build marker-wrapped command pattern
        # Uses subshell to isolate exit and prevent terminating parent shell
        # Wrap command in bash -c to handle shell metacharacters (;, &&, ||, pipes, env vars)
        # Escape single quotes in command by replacing ' with '\''
        # Note: command names are already validated by CUSTOM_COMMAND_NAME_PATTERN
        # to only contain [a-zA-Z_][a-zA-Z0-9_]* so no escaping needed for names
        escaped_command = command.replace("'", "'\\''")
        if allow_fail:
            # Advisory: always exit 0 from subshell, note failure but don't block
            wrapper = (
                f"(echo '[custom:{name}:start]'; "
                f"__status=0; timeout {timeout} bash -c '{escaped_command}' || __status=$?; "
                f"if [ $__status -eq 0 ]; then echo '[custom:{name}:pass]'; "
                f"elif [ $__status -eq 124 ]; then echo '[custom:{name}:timeout]'; "
                f'else echo "[custom:{name}:fail exit=$__status]"; fi; '
                f"exit 0)  # advisory (allow_fail=true)"
            )
            lines.append(f"# {name} (advisory - failures don't block)")
        else:
            # Strict: exit from subshell with command's status
            wrapper = (
                f"(echo '[custom:{name}:start]'; "
                f"__status=0; timeout {timeout} bash -c '{escaped_command}' || __status=$?; "
                f"if [ $__status -eq 0 ]; then echo '[custom:{name}:pass]'; "
                f"elif [ $__status -eq 124 ]; then echo '[custom:{name}:timeout]'; "
                f'else echo "[custom:{name}:fail exit=$__status]"; fi; '
                f"exit $__status)"
            )
            lines.append(f"# {name}")

        lines.append(wrapper)

    return "\n".join(lines)


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
    checkpoint_request_prompt: str
    continuation_prompt: str
    review_agent_prompt: str = ""


def load_prompts(prompt_dir: Path) -> PromptProvider:
    """Load all prompt templates from disk.

    Args:
        prompt_dir: Directory containing prompt template files.

    Returns:
        PromptProvider with all loaded prompt templates.

    Raises:
        FileNotFoundError: If any required prompt file is missing.
    """
    # Load optional review_agent.md (backwards compat: empty string if missing)
    review_agent_path = prompt_dir / "review_agent.md"
    review_agent_prompt = (
        review_agent_path.read_text() if review_agent_path.exists() else ""
    )

    return PromptProvider(
        implementer_prompt=(prompt_dir / "implementer_prompt.md").read_text(),
        review_followup_prompt=(prompt_dir / "review_followup.md").read_text(),
        gate_followup_prompt=(prompt_dir / "gate_followup.md").read_text(),
        fixer_prompt=(prompt_dir / "fixer.md").read_text(),
        idle_resume_prompt=(prompt_dir / "idle_resume.md").read_text(),
        checkpoint_request_prompt=(prompt_dir / "checkpoint_request.md").read_text(),
        continuation_prompt=(prompt_dir / "continuation.md").read_text(),
        review_agent_prompt=review_agent_prompt,
    )


def format_implementer_prompt(
    implementer_prompt: str,
    issue_id: str,
    repo_path: Path,
    agent_id: str,
    validation_commands: PromptValidationCommands,
    lock_dir: Path,
    issue_description: str | None,
) -> str:
    """Format the implementer prompt with runtime values.

    Args:
        implementer_prompt: The raw implementer prompt template.
        issue_id: The issue ID being implemented.
        repo_path: Path to the repository.
        agent_id: The agent ID for this session.
        validation_commands: Validation commands for the prompt.
        lock_dir: Directory for lock files (from infra layer).
        issue_description: Description of the issue being implemented.

    Returns:
        Formatted prompt string.
    """
    # Build custom_commands_section from custom_commands tuple
    custom_commands_section = build_custom_commands_section(
        validation_commands.custom_commands
    )

    return implementer_prompt.format(
        issue_id=issue_id,
        repo_path=repo_path,
        lock_dir=lock_dir,
        agent_id=agent_id,
        lint_command=validation_commands.lint,
        format_command=validation_commands.format,
        typecheck_command=validation_commands.typecheck,
        custom_commands_section=custom_commands_section,
        test_command=validation_commands.test,
        issue_description=(issue_description or "No description available").replace(
            "{", "{{"
        ).replace("}", "}}"),
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
        custom_commands=(),
    )


def _default_prompt_dir() -> Path:
    """Return the default prompts directory."""
    return Path(__file__).parent.parent / "prompts"


def load_prompt(name: str, prompt_dir: Path | None = None) -> str:
    """Load a single prompt template by name.

    Args:
        name: Name of the prompt (without .md extension).
        prompt_dir: Directory containing prompt files. Defaults to src/prompts.

    Returns:
        The prompt template content.

    Raises:
        FileNotFoundError: If the prompt file doesn't exist.
    """
    if prompt_dir is None:
        prompt_dir = _default_prompt_dir()
    return (prompt_dir / f"{name}.md").read_text()


def extract_checkpoint(text: str) -> str:
    """Extract checkpoint block from agent response text.

    Looks for content between <checkpoint> and </checkpoint> tags.
    For nested tags, returns the outermost checkpoint content.
    Returns full text as fallback if no tags found (stripping code block wrappers).

    Args:
        text: Raw agent response text.

    Returns:
        Extracted checkpoint content, or full text if no tags found.
    """
    # Find first opening tag
    start_match = re.search(r"<checkpoint>", text)
    if not start_match:
        # Fallback: strip code block wrappers and return
        stripped = re.sub(r"\A\s*```\w*[ \t]*\n?", "", text)
        stripped = re.sub(r"\n?[ \t]*```[ \t]*\Z", "", stripped)
        return stripped

    # Track nesting depth to find matching closing tag
    start_pos = start_match.end()
    depth = 1
    pos = start_pos

    while depth > 0 and pos < len(text):
        next_open = text.find("<checkpoint>", pos)
        next_close = text.find("</checkpoint>", pos)

        if next_close == -1:
            # No more closing tags, return from start to end
            break

        if next_open != -1 and next_open < next_close:
            # Found nested opening tag
            depth += 1
            pos = next_open + len("<checkpoint>")
        else:
            # Found closing tag
            depth -= 1
            if depth == 0:
                return text[start_pos:next_close]
            pos = next_close + len("</checkpoint>")

    # No proper closing found, return from start to end
    return text[start_pos:]


def build_continuation_prompt(continuation_template: str, checkpoint_text: str) -> str:
    """Build a continuation prompt with checkpoint context.

    Args:
        continuation_template: The continuation prompt template from PromptProvider.
        checkpoint_text: The checkpoint block from the previous session.

    Returns:
        Formatted continuation prompt with checkpoint embedded.
    """
    # Use str.replace instead of str.format to avoid KeyError if checkpoint
    # contains curly braces (e.g., JSON or code snippets)
    return continuation_template.replace("{checkpoint}", checkpoint_text)


def build_prompt_validation_commands(
    repo_path: Path,
    *,
    validation_config: ValidationConfig | None = None,
    config_missing: bool = False,
) -> PromptValidationCommands:
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

    if config_missing:
        return get_default_validation_commands()

    if validation_config is not None:
        user_config = validation_config
    else:
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

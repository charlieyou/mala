"""Shared prompt loading utilities.

This module centralizes prompt file loading to avoid duplication across modules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from src.domain.validation.spec import (
    CommandKind,
    ValidationCommand,
)
from src.domain.validation_wrapper import build_canonical_wrapper

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.domain.validation.config_types import (
        PromptValidationCommands,
        ValidationConfig,
    )


def build_custom_commands_section(
    custom_commands: tuple[tuple[str, str, int, bool], ...],
    *,
    issue_id: str,
    validation_log_dir: Path,
) -> str:
    """Build the custom commands section using canonical wrappers.

    Each custom command is wrapped with the canonical Bash wrapper from
    `build_canonical_wrapper`, which emits one `MALA_EVIDENCE` summary line
    per command. Strict commands (`allow_fail=False`) propagate the
    command's status; advisory commands (`allow_fail=True`) exit 0 from the
    subshell so they do not block the pipeline.

    Args:
        custom_commands: Tuple of (name, command, timeout, allow_fail) tuples.
        issue_id: The issue ID used to compose log file names.
        validation_log_dir: Directory for validation command output logs.

    Returns:
        One fenced canonical wrapper snippet per command, joined by blank
        lines, or the empty string when there are no custom commands.
    """
    if not custom_commands:
        return ""

    wrappers = [
        "```bash\n"
        + build_canonical_wrapper(
            ValidationCommand(
                name=name,
                command=command,
                kind=CommandKind.CUSTOM,
                timeout=timeout,
                allow_fail=allow_fail,
            ),
            issue_id=issue_id,
            validation_log_dir=validation_log_dir,
        )
        + "\n```"
        for name, command, timeout, allow_fail in custom_commands
    ]
    return "\n\n".join(wrappers)


def build_gate_followup_wrappers(
    missing_commands: Sequence[ValidationCommand],
    *,
    issue_id: str,
    validation_log_dir: Path,
) -> str:
    """Emit one canonical wrapper per missing command for gate_followup.md.

    Args:
        missing_commands: Validation commands the gate flagged as missing or
            invalid evidence. Order is preserved in the rendered output.
        issue_id: The issue ID used to compose log file names.
        validation_log_dir: Directory for validation command output logs.

    Returns:
        Fenced canonical wrapper snippets joined by blank lines, or the empty
        string when no commands are missing.
    """
    if not missing_commands:
        return ""

    wrappers = [
        "```bash\n"
        + build_canonical_wrapper(
            cmd,
            issue_id=issue_id,
            validation_log_dir=validation_log_dir,
        )
        + "\n```"
        for cmd in missing_commands
    ]
    return "\n\n".join(wrappers)


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
    await_resume_prompt: str
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
        await_resume_prompt=(prompt_dir / "await_resume.md").read_text(),
        review_agent_prompt=review_agent_prompt,
    )


# Mapping of built-in command attributes on PromptValidationCommands to their
# evidence name, kind, and timeout-field name. Used to render canonical
# wrappers that honour configured per-command timeouts (`commands.<kind>.timeout`
# from mala.yaml). The timeout-field fallback to 120s is on the dataclass.
_BUILTIN_COMMANDS: tuple[tuple[str, str, CommandKind, str], ...] = (
    ("format", "format", CommandKind.FORMAT, "format_timeout"),
    ("lint", "lint", CommandKind.LINT, "lint_timeout"),
    ("typecheck", "typecheck", CommandKind.TYPECHECK, "typecheck_timeout"),
    ("test", "test", CommandKind.TEST, "test_timeout"),
)


def _builtin_command_records(
    validation_commands: PromptValidationCommands,
) -> dict[str, ValidationCommand]:
    """Build ValidationCommand records for the four built-in commands.

    Returns a dict keyed by evidence name (`format`, `lint`, `typecheck`,
    `test`) so callers can render each built-in's canonical wrapper into the
    matching prompt placeholder. Configured per-command timeouts from
    `mala.yaml` (e.g., `commands.test.timeout: 600`) are threaded through
    so the wrapper's `timeout` invocation matches the configured budget.
    """
    return {
        name: ValidationCommand(
            name=name,
            command=getattr(validation_commands, attr),
            kind=kind,
            timeout=getattr(validation_commands, timeout_attr),
            allow_fail=False,
        )
        for attr, name, kind, timeout_attr in _BUILTIN_COMMANDS
    }


def build_validation_wrapper_placeholders(
    validation_commands: PromptValidationCommands,
    *,
    issue_id: str,
    validation_log_dir: Path,
) -> dict[str, str]:
    """Build prompt placeholders for all validation command wrappers.

    Built-in command placeholders (`format_command`, `lint_command`,
    `typecheck_command`, `test_command`) and `custom_commands_section` all use
    the same canonical wrapper builder.
    """
    builtins = _builtin_command_records(validation_commands)
    wrappers = {
        f"{name}_command": build_canonical_wrapper(
            cmd,
            issue_id=issue_id,
            validation_log_dir=validation_log_dir,
        )
        for name, cmd in builtins.items()
    }
    wrappers["custom_commands_section"] = build_custom_commands_section(
        validation_commands.custom_commands,
        issue_id=issue_id,
        validation_log_dir=validation_log_dir,
    )
    return wrappers


def format_implementer_prompt(
    implementer_prompt: str,
    issue_id: str,
    repo_path: Path,
    agent_id: str,
    validation_commands: PromptValidationCommands,
    lock_dir: Path,
    validation_log_dir: Path,
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
        validation_log_dir: Directory for validation command output logs.
        issue_description: Description of the issue being implemented.

    Returns:
        Formatted prompt string.
    """
    wrappers = build_validation_wrapper_placeholders(
        validation_commands,
        issue_id=issue_id,
        validation_log_dir=validation_log_dir,
    )

    return implementer_prompt.format(
        issue_id=issue_id,
        repo_path=repo_path,
        lock_dir=lock_dir,
        validation_log_dir=validation_log_dir,
        agent_id=agent_id,
        issue_description=(issue_description or "No description available")
        .replace("{", "{{")
        .replace("}", "}}"),
        **wrappers,
    )


def format_gate_followup_prompt(
    gate_followup_template: str,
    *,
    attempt: int,
    max_attempts: int,
    failure_reasons: str,
    issue_id: str,
    missing_commands: Sequence[ValidationCommand],
    validation_log_dir: Path,
) -> str:
    """Format the gate follow-up prompt with canonical wrappers.

    Args:
        gate_followup_template: Raw gate_followup.md template.
        attempt: Current retry attempt number.
        max_attempts: Maximum retry attempts.
        failure_reasons: Pre-formatted failure-reason text.
        issue_id: The issue ID being implemented.
        missing_commands: Validation commands flagged as missing or invalid
            evidence. One canonical wrapper is rendered per command.
        validation_log_dir: Directory for validation command output logs.

    Returns:
        Formatted gate-followup prompt with `{missing_command_wrappers}`
        substituted by canonical wrappers.
    """
    missing_command_wrappers = build_gate_followup_wrappers(
        missing_commands,
        issue_id=issue_id,
        validation_log_dir=validation_log_dir,
    )
    return gate_followup_template.format(
        attempt=attempt,
        max_attempts=max_attempts,
        failure_reasons=failure_reasons,
        issue_id=issue_id,
        missing_command_wrappers=missing_command_wrappers,
        validation_log_dir=validation_log_dir,
    )


def format_await_resume_prompt(
    await_resume_template: str,
    *,
    issue_id: str,
    output_file: str,
    status: str,
    summary: str,
) -> str:
    """Format the await-resume prompt shown after a backgrounded task finishes.

    Substitutes the ``{issue_id}``, ``{output_file}``, ``{status}`` and
    ``{summary}`` placeholders. Values are inserted verbatim by a single
    ``str.format`` call, which does not re-parse substituted values, so literal
    braces in ``status``/``summary`` (e.g. JSON) are preserved as-is and need no
    escaping.

    Args:
        await_resume_template: Raw ``await_resume.md`` template.
        issue_id: The issue ID being implemented.
        output_file: Path to the backgrounded task's captured output file.
        status: Completion status reported by the SDK (e.g. ``completed``,
            ``failed``, ``stopped``).
        summary: Free-form summary from the SDK, typically including the exit
            code of the backgrounded process.

    Returns:
        Formatted await-resume prompt string.
    """

    return await_resume_template.format(
        issue_id=issue_id,
        output_file=output_file,
        status=status,
        summary=summary,
    )


def commands_for_gate_followup(
    validation_commands: PromptValidationCommands,
) -> list[ValidationCommand]:
    """Return ValidationCommand records for every configured command.

    Used by the gate-followup renderer when it must rerun the full validation
    suite (no per-command evidence map is available yet). Order matches the
    implementer prompt: format → lint → typecheck → custom → test.
    """
    builtins = _builtin_command_records(validation_commands)
    ordered: list[ValidationCommand] = [
        builtins["format"],
        builtins["lint"],
        builtins["typecheck"],
    ]
    for name, command, timeout, allow_fail in validation_commands.custom_commands:
        ordered.append(
            ValidationCommand(
                name=name,
                command=command,
                kind=CommandKind.CUSTOM,
                timeout=timeout,
                allow_fail=allow_fail,
            )
        )
    ordered.append(builtins["test"])
    return ordered


def get_default_validation_commands() -> PromptValidationCommands:
    """Return default Python/uv validation commands with cache isolation.

    These defaults are used when no mala.yaml configuration is found.
    Commands include cache isolation flags for parallel agent runs,
    using $AGENT_ID environment variable (set in the agent environment).

    Returns:
        PromptValidationCommands with default Python/uv toolchain commands.
    """
    from src.domain.validation.config_types import PromptValidationCommands

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
    config_path: Path | None = None,
) -> PromptValidationCommands:
    """Build PromptValidationCommands for a repository.

    Loads the Mala project configuration, merges with preset if specified,
    and returns the validation commands formatted for prompt templates.

    Args:
        repo_path: Path to the repository root directory.
        validation_config: Pre-loaded ValidationConfig to avoid re-reading.
        config_missing: Set True to skip config loading and return defaults.
        config_path: Optional explicit config file path. Relative paths are
            resolved relative to the current working directory when loading is
            needed. Ignored when ``validation_config`` or ``config_missing`` is
            supplied.

    Returns:
        PromptValidationCommands with command strings for prompt templates.
        Returns default Python/uv commands if no config is found.
    """
    from src.domain.validation.config_types import PromptValidationCommands
    from src.domain.validation.config_loader import ConfigMissingError, load_config
    from src.domain.validation.config_merger import merge_configs
    from src.domain.validation.preset_registry import PresetRegistry

    if config_missing:
        return get_default_validation_commands()

    if validation_config is not None:
        user_config = validation_config
    else:
        try:
            user_config = load_config(repo_path, config_path=config_path)
        except ConfigMissingError:
            if config_path is not None:
                raise
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

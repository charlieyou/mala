"""Extract human-readable tool names from shell commands.

This module provides a utility function to extract the primary tool name from
complex shell commands for use in quality gate messaging. For example:
- "npx eslint ." → "eslint"
- "uvx ruff check ." → "ruff"
- "uv run pytest" → "pytest"
"""

from __future__ import annotations

import logging
import shlex

logger = logging.getLogger(__name__)

# Wrappers where we skip the wrapper and use the next positional argument
# Note: pnpm and yarn are NOT here because they have compound command handling
_SINGLE_TOKEN_WRAPPERS: frozenset[str] = frozenset({"npx", "bunx", "uvx", "pipx"})

# Multi-token wrapper sequences where we skip the entire sequence
# Format: (first_token, second_token) → skip both tokens
_MULTI_TOKEN_WRAPPERS: frozenset[tuple[str, str]] = frozenset(
    {
        ("python", "-m"),
        ("python3", "-m"),
        ("uv", "run"),
        ("poetry", "run"),
        ("pipx", "run"),
    }
)

# Compound commands where we include the subcommand in the tool name
# Format: base_command → set of subcommands that form compound tools
_COMPOUND_COMMANDS: dict[str, frozenset[str]] = {
    "go": frozenset({"test", "build", "vet", "fmt", "mod", "generate"}),
    "cargo": frozenset({"clippy", "test", "build", "check", "fmt", "bench"}),
    "npm": frozenset({"test", "run"}),
    "pnpm": frozenset({"test", "run"}),
    "yarn": frozenset({"test", "run"}),
}

# Shell built-ins to skip (these appear before a real command)
_SHELL_BUILTINS: frozenset[str] = frozenset({"export", "set", "source", "."})

# Commands that are typically setup/shell commands, not actual tools
# When these appear in a multi-segment command (with && or ;), prefer later segments
_SETUP_COMMANDS: frozenset[str] = frozenset({"cd", "echo", "install"})


def _is_env_assignment(token: str) -> bool:
    """Check if a token is an environment variable assignment.

    Args:
        token: Token to check.

    Returns:
        True if token looks like VAR=value assignment.
    """
    # Must have = and the part before = must be a valid identifier
    if "=" not in token:
        return False
    name, _, _ = token.partition("=")
    # Empty name or starts with digit or has special chars isn't a valid identifier
    if not name or name[0].isdigit():
        return False
    return all(c.isalnum() or c == "_" for c in name)


def _strip_path_prefix(cmd: str) -> str:
    """Strip path prefixes from command names.

    Args:
        cmd: Command that might have a path prefix.

    Returns:
        Just the base command name.
    """
    # Split on / and take the last part
    if "/" in cmd:
        return cmd.rsplit("/", 1)[-1]
    return cmd


def _parse_command(command: str) -> list[str]:
    """Parse a command string into tokens.

    Uses shlex for proper shell parsing, falls back to whitespace
    splitting if shlex fails.

    Args:
        command: Shell command string.

    Returns:
        List of parsed tokens.
    """
    try:
        return shlex.split(command)
    except ValueError:
        # Fallback to simple whitespace splitting for malformed commands
        logger.warning(
            "shlex.split failed for command, using whitespace split: %r", command
        )
        return command.split()


def _extract_from_tokens(tokens: list[str]) -> str:
    """Extract tool name from a list of parsed tokens.

    Args:
        tokens: Parsed command tokens.

    Returns:
        Extracted tool name.
    """
    if not tokens:
        return ""

    # Skip leading env var assignments
    idx = 0
    while idx < len(tokens) and _is_env_assignment(tokens[idx]):
        idx += 1

    if idx >= len(tokens):
        # Only env assignments, return empty
        return ""

    first = _strip_path_prefix(tokens[idx])

    # Skip shell built-ins and their arguments
    while first in _SHELL_BUILTINS:
        idx += 1
        # Skip any arguments to the built-in (e.g., FOO=bar after export)
        while idx < len(tokens) and _is_env_assignment(tokens[idx]):
            idx += 1
        if idx >= len(tokens):
            return ""  # Only built-ins, return empty
        first = _strip_path_prefix(tokens[idx])

    # Check for multi-token wrapper sequences
    if idx + 1 < len(tokens):
        second = tokens[idx + 1]
        if (first, second) in _MULTI_TOKEN_WRAPPERS:
            idx += 2
            if idx >= len(tokens):
                # Wrapper without command, return wrapper name
                logger.warning("Wrapper %s %s without following command", first, second)
                return first
            first = _strip_path_prefix(tokens[idx])

    # Check for compound commands BEFORE single-token wrappers
    # This allows npm, pnpm, yarn to be recognized as compound commands
    if first in _COMPOUND_COMMANDS:
        if idx + 1 < len(tokens):
            next_token = tokens[idx + 1]
            # npm/pnpm/yarn run has special handling: npm run lint → npm run:lint
            if first in ("npm", "pnpm", "yarn") and next_token == "run":
                if idx + 2 < len(tokens):
                    script_name = tokens[idx + 2]
                    return f"{first} run:{script_name}"
                return f"{first} run"
            # Other compound commands: go test → go test
            if next_token in _COMPOUND_COMMANDS[first]:
                return f"{first} {next_token}"

    # Check for single-token wrappers
    if first in _SINGLE_TOKEN_WRAPPERS:
        idx += 1
        # After wrapper, skip any leading flags to find the actual tool
        while idx < len(tokens) and tokens[idx].startswith("-"):
            idx += 1
        if idx >= len(tokens):
            logger.warning("Wrapper %s without following command", first)
            return first
        first = _strip_path_prefix(tokens[idx])

    return first


def _is_meaningful_tool(tool_name: str) -> bool:
    """Check if a tool name is meaningful for reporting.

    Some commands like 'cd', 'echo', 'npm install' are setup commands
    rather than the actual tool being run.

    Args:
        tool_name: Extracted tool name.

    Returns:
        True if the tool name is meaningful.
    """
    if not tool_name:
        return False
    # Get the base command (first word)
    base = tool_name.split()[0]
    # Skip common setup commands
    if base in _SETUP_COMMANDS:
        return False
    # npm without compound (i.e., just "npm") or "npm install" is setup
    if base == "npm":
        if tool_name == "npm" or "install" in tool_name:
            return False
    return True


def extract_tool_name(command: str) -> str:
    """Extract human-readable tool name from a shell command.

    This function extracts the primary tool name from complex shell commands
    for use in quality gate messaging and logging.

    Algorithm:
    1. Handle shell operators (&&, ||, |, ;) by trying each segment
    2. Parse via shlex.split; fallback to whitespace if parsing fails
    3. Skip env var assignments (tokens with = before command)
    4. Skip shell built-ins (export, set, cd)
    5. Strip path prefixes (/usr/bin/eslint → eslint)
    6. Apply wrapper rules:
       - Single-token: npx, bunx, uvx, pipx → skip wrapper, use next positional
       - Multi-token: python -m, uv run, poetry run → skip sequence
       - Compound: go test, cargo clippy, npm test → include subcommand
       - Script: npm run lint → npm run:lint
    7. Fallback: return first token

    Args:
        command: Shell command string.

    Returns:
        Human-readable tool name. For empty/malformed commands, returns
        the best-effort result (possibly empty string) with warning logged.

    Examples:
        >>> extract_tool_name("npx eslint .")
        'eslint'
        >>> extract_tool_name("uvx ruff check .")
        'ruff'
        >>> extract_tool_name("uv run pytest")
        'pytest'
        >>> extract_tool_name("go test ./...")
        'go test'
        >>> extract_tool_name("cargo clippy")
        'cargo clippy'
        >>> extract_tool_name("npm run lint")
        'npm run:lint'
    """
    if not command or not command.strip():
        logger.warning("Empty command provided to extract_tool_name")
        return ""

    command = command.strip()

    # Handle shell operators by trying each segment
    # Split on common operators and process each segment
    # Try in order: &&, ||, ;, | (pipe last since it's most common for chaining)
    for operator in ("&&", "||", ";", "|"):
        if operator in command:
            segments = command.split(operator)
            # Try to find a meaningful tool from any segment
            for segment in segments:
                segment = segment.strip()
                if segment:
                    result = _extract_from_tokens(_parse_command(segment))
                    if result and _is_meaningful_tool(result):
                        return result
            # If no meaningful tool found, try first segment as fallback
            for segment in segments:
                segment = segment.strip()
                if segment:
                    result = _extract_from_tokens(_parse_command(segment))
                    if result:
                        return result

    # No operators or operators didn't yield a result - parse directly
    tokens = _parse_command(command)
    result = _extract_from_tokens(tokens)

    if not result:
        logger.warning("Could not extract tool name from command: %r", command)

    return result

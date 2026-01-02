"""Validation gating: determines whether validation should trigger based on changed files.

This module provides logic to filter validation triggers based on patterns
from mala.yaml configuration:
- `code_patterns`: patterns for files that trigger validation
- `config_files`: patterns for files that trigger validation and invalidate
  lint/format/typecheck cache
- `setup_files`: patterns for files that trigger validation and invalidate
  setup cache (uv.lock, etc.)

Special cases:
- Empty `code_patterns` matches all files (validation always triggers)
- Changes to `mala.yaml` always trigger validation
- Changes to `config_files` or `setup_files` trigger cache invalidation
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .code_pattern_matcher import filter_matching_files


# Config file name that always triggers validation when changed
MALA_CONFIG_FILE = "mala.yaml"


class ValidationSpecLike(Protocol):
    """Protocol for objects with validation spec attributes.

    This allows duck-typing for testing without requiring the full ValidationSpec.
    """

    code_patterns: list[str]
    config_files: list[str]
    setup_files: list[str]


def should_trigger_validation(
    changed_files: list[str],
    spec: ValidationSpecLike,
) -> bool:
    """Determine if validation should run based on changed files.

    Validation should trigger when:
    1. `mala.yaml` is in the changed files (always triggers)
    2. `code_patterns` is empty (matches all files)
    3. Any changed file matches a pattern in `code_patterns`, `config_files`,
       or `setup_files`

    Args:
        changed_files: List of file paths that were modified.
        spec: ValidationSpec containing code_patterns to match against.

    Returns:
        True if validation should run, False to skip validation.
    """
    if not changed_files:
        # No changes = no validation needed
        return False

    # Special case: mala.yaml changes always trigger validation
    if any(_is_mala_config(f) for f in changed_files):
        return True

    # Empty patterns means match all files
    if not spec.code_patterns:
        return True

    # Check if any changed file matches code/config/setup patterns
    patterns = [*spec.code_patterns, *spec.config_files, *spec.setup_files]
    if not patterns:
        return True
    matching_files = filter_matching_files(changed_files, patterns)
    return len(matching_files) > 0


def get_matching_code_files(
    changed_files: list[str],
    spec: ValidationSpecLike,
) -> list[str]:
    """Get the subset of changed files that match code patterns.

    This is useful for understanding which files triggered validation.

    Args:
        changed_files: List of file paths that were modified.
        spec: ValidationSpec containing code_patterns to match against.

    Returns:
        List of files that match code patterns (or all files if patterns empty).
    """
    if not spec.code_patterns:
        return list(changed_files)
    return filter_matching_files(changed_files, spec.code_patterns)


def should_invalidate_lint_cache(
    changed_files: list[str],
    spec: ValidationSpecLike,
) -> bool:
    """Determine if lint/format/typecheck cache should be invalidated.

    Cache should be invalidated when:
    - mala.yaml is changed (always)
    - Any changed file matches `config_files` patterns

    Args:
        changed_files: List of file paths that were modified.
        spec: ValidationSpec containing config_files patterns.

    Returns:
        True if lint cache should be invalidated.
    """
    if not changed_files:
        return False

    # mala.yaml changes always invalidate lint cache
    if any(_is_mala_config(f) for f in changed_files):
        return True

    # Check config_files patterns (if any defined)
    if not spec.config_files:
        return False

    matching_files = filter_matching_files(changed_files, spec.config_files)
    return len(matching_files) > 0


def should_invalidate_setup_cache(
    changed_files: list[str],
    spec: ValidationSpecLike,
) -> bool:
    """Determine if setup cache should be invalidated.

    Setup cache should be invalidated when any changed file matches
    `setup_files` patterns (e.g., uv.lock, requirements.txt, etc.).

    Args:
        changed_files: List of file paths that were modified.
        spec: ValidationSpec containing setup_files patterns.

    Returns:
        True if setup cache should be invalidated.
    """
    if not changed_files or not spec.setup_files:
        return False

    matching_files = filter_matching_files(changed_files, spec.setup_files)
    return len(matching_files) > 0


def get_config_files_changed(
    changed_files: list[str],
    spec: ValidationSpecLike,
) -> list[str]:
    """Get config files that were changed (for logging/debugging).

    Args:
        changed_files: List of file paths that were modified.
        spec: ValidationSpec containing config_files patterns.

    Returns:
        List of changed config files.
    """
    if not spec.config_files:
        return []
    result = filter_matching_files(changed_files, spec.config_files)
    # Also include mala.yaml if changed
    if any(_is_mala_config(f) for f in changed_files):
        if MALA_CONFIG_FILE not in result:
            result = [MALA_CONFIG_FILE, *result]
    return result


def get_setup_files_changed(
    changed_files: list[str],
    spec: ValidationSpecLike,
) -> list[str]:
    """Get setup files that were changed (for logging/debugging).

    Args:
        changed_files: List of file paths that were modified.
        spec: ValidationSpec containing setup_files patterns.

    Returns:
        List of changed setup files.
    """
    if not spec.setup_files:
        return []
    return filter_matching_files(changed_files, spec.setup_files)


def _is_mala_config(file_path: str) -> bool:
    """Check if a file path is the mala config file.

    Handles both absolute and relative paths by checking the basename.

    Args:
        file_path: File path to check.

    Returns:
        True if this is the mala.yaml config file.
    """
    # Check basename to handle paths like /repo/mala.yaml or mala.yaml
    return Path(file_path).name == MALA_CONFIG_FILE

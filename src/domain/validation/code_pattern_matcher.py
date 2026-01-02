"""Code pattern matcher for file path filtering.

This module provides utilities to match file paths against glob patterns
for validation gating. It supports:
- `*` matches any non-slash characters
- `**` matches anything including `/` (zero or more directory levels)
- Filename-only patterns (no `/`) match against basename
- Path patterns (contain `/`) match against full relative path
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Convert a glob pattern to a compiled regex pattern.

    Supports:
    - `*` matches any character except `/`
    - `**` matches anything including `/`

    If the pattern is invalid (e.g., contains unbalanced brackets),
    treat it as a literal string and log a warning.

    Args:
        pattern: Glob pattern to convert.

    Returns:
        Compiled regex pattern.
    """
    try:
        # Build regex by processing the pattern character by character
        regex_parts: list[str] = []
        i = 0
        n = len(pattern)

        while i < n:
            char = pattern[i]

            # Check for **/ or ** at end (matches zero or more directory segments)
            if char == "*" and i + 1 < n and pattern[i + 1] == "*":
                i += 2
                if i < n and pattern[i] == "/":
                    # **/ matches zero or more complete directory segments
                    # Either nothing (zero segments) or anything ending with /
                    regex_parts.append("(?:.*/)?")
                    i += 1
                else:
                    # ** at end or not followed by / - matches anything
                    regex_parts.append(".*")
            elif char == "*":
                # Single * matches any character except /
                regex_parts.append("[^/]*")
                i += 1
            elif char == "?":
                # ? matches any single character except /
                regex_parts.append("[^/]")
                i += 1
            elif char in ".^$+{}|()[]":
                # Escape regex special characters
                regex_parts.append("\\" + char)
                i += 1
            elif char == "\\":
                # Escape next character
                if i + 1 < n:
                    regex_parts.append("\\" + pattern[i + 1])
                    i += 2
                else:
                    regex_parts.append("\\\\")
                    i += 1
            else:
                regex_parts.append(char)
                i += 1

        regex_str = "^" + "".join(regex_parts) + "$"
        return re.compile(regex_str)
    except re.error as e:
        # Invalid pattern - treat as literal string
        logger.warning("Invalid glob pattern '%s', treating as literal: %s", pattern, e)
        return re.compile("^" + re.escape(pattern) + "$")


def matches_pattern(path: str, pattern: str) -> bool:
    """Check if a path matches a glob pattern.

    Matching rules:
    - Filename-only patterns (no `/`): match against os.path.basename(path)
    - Path patterns (contain `/`): match against full relative path

    Args:
        path: File path to check.
        pattern: Glob pattern to match against.

    Returns:
        True if path matches pattern, False otherwise.
    """
    # Normalize path separators
    path = path.replace("\\", "/")
    pattern = pattern.replace("\\", "/")

    # Determine if this is a filename-only pattern or a path pattern
    if "/" in pattern:
        # Path pattern - match against full path
        # Handle patterns starting with **/ which should match any path
        target = path.lstrip("/")
    else:
        # Filename-only pattern - match against basename
        target = os.path.basename(path)

    regex = glob_to_regex(pattern)
    return regex.match(target) is not None


def filter_matching_files(files: list[str], patterns: list[str]) -> list[str]:
    """Filter files that match any of the given patterns.

    Args:
        files: List of file paths to filter.
        patterns: List of glob patterns. Empty list matches everything.

    Returns:
        List of files that match at least one pattern.
    """
    if not patterns:
        # Empty patterns list matches everything
        return list(files)

    return [f for f in files if any(matches_pattern(f, p) for p in patterns)]

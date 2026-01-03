"""Pure helper functions for rewriting coverage command arguments.

These functions extract and transform pytest/coverage command arguments
without side effects, making them easy to test in isolation.
"""

from __future__ import annotations

import shlex


def strip_xdist_flags(args: list[str]) -> list[str]:
    """Remove pytest-xdist parallel execution flags.

    Strips -n, --numprocesses, and --dist flags (both standalone and =value forms)
    to ensure deterministic coverage generation.

    Args:
        args: Command arguments list.

    Returns:
        New list with xdist flags removed.
    """
    result = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        # Flags that take a separate value
        if arg in {"-n", "--numprocesses", "--dist"}:
            skip_next = True
            continue
        # Flags with value attached via =
        if (
            arg.startswith("-n=")
            or arg.startswith("--numprocesses=")
            or arg.startswith("--dist=")
        ):
            continue
        result.append(arg)
    return result


def strip_cov_fail_under(args: list[str]) -> list[str]:
    """Remove --cov-fail-under arguments.

    Args:
        args: Command arguments list.

    Returns:
        New list with --cov-fail-under removed.
    """
    result = []
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg.startswith("--cov-fail-under="):
            continue
        if arg == "--cov-fail-under":
            skip_next = True
            continue
        result.append(arg)
    return result


def extract_marker_expr(args: list[str]) -> tuple[list[str], str | None]:
    """Extract -m marker expression from arguments.

    Args:
        args: Command arguments list.

    Returns:
        Tuple of (args with -m removed, extracted marker expression or None).
    """
    result = []
    marker_expr: str | None = None
    skip_next = False
    for arg in args:
        if skip_next:
            # This is the value after -m
            marker_expr = arg
            skip_next = False
            continue
        if arg.startswith("-m="):
            marker_expr = arg.split("=", 1)[1]
            continue
        if arg == "-m":
            skip_next = True
            continue
        result.append(arg)
    return result, marker_expr


def normalize_marker_expr(expr: str | None) -> str:
    """Normalize marker expression for baseline coverage.

    Defaults to "unit or integration" and filters out e2e markers.

    Args:
        expr: Original marker expression or None.

    Returns:
        Normalized marker expression.
    """
    marker = (expr or "unit or integration").strip()
    if "e2e" in marker:
        marker = "unit or integration"
    return marker


def strip_cov_report_xml(args: list[str]) -> list[str]:
    """Remove --cov-report=xml arguments.

    Strips both --cov-report=xml and --cov-report=xml:<path> forms.

    Args:
        args: Command arguments list.

    Returns:
        New list with xml report arguments removed.
    """
    return [
        arg
        for arg in args
        if arg != "--cov-report=xml" and not arg.startswith("--cov-report=xml:")
    ]


def rewrite_coverage_command(cmd: str, coverage_file: str) -> list[str]:
    """Rewrite coverage command for baseline refresh.

    Applies all transformations:
    - Strips xdist flags
    - Strips --cov-fail-under
    - Extracts and normalizes marker expression
    - Strips existing xml report paths
    - Adds new xml report path and --cov-fail-under=0

    Args:
        cmd: Original coverage command string.
        coverage_file: Path for XML coverage output.

    Returns:
        Transformed command as list of arguments.
    """
    args = shlex.split(cmd)
    args = strip_xdist_flags(args)
    args = strip_cov_fail_under(args)
    args, marker_expr = extract_marker_expr(args)
    marker_expr = normalize_marker_expr(marker_expr)
    args = strip_cov_report_xml(args)
    args.append(f"--cov-report=xml:{coverage_file}")
    args.append("--cov-fail-under=0")
    args.extend(["-m", marker_expr])
    return args

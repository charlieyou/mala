"""E2E test configuration - Claude auth and credential setup."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.nodes import Item


def pytest_configure(config: pytest.Config) -> None:
    """Copy OAuth credentials to test config dir for SDK e2e tests."""
    test_claude_dir = Path("/tmp/mala-test-claude")
    test_claude_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CLAUDE_CONFIG_DIR"] = str(test_claude_dir)

    real_credentials = Path.home() / ".claude" / ".credentials.json"
    if real_credentials.exists():
        shutil.copy2(real_credentials, test_claude_dir / ".credentials.json")


def pytest_collection_modifyitems(config: pytest.Config, items: list[Item]) -> None:
    """Skip flaky_sdk tests unless explicitly requested with --include-flaky-sdk.

    This allows `pytest -m e2e` to run without hanging on flaky SDK tests
    while still allowing explicit inclusion via marker expression like
    `pytest -m 'e2e and flaky_sdk'` or the --include-flaky-sdk flag.
    """
    # Check if flaky_sdk is explicitly included in the marker expression
    marker_expr = config.getoption("-m", default="")
    if "flaky_sdk" in marker_expr and "not flaky_sdk" not in marker_expr:
        # User explicitly wants flaky_sdk tests, don't skip
        return

    # Check for custom flag (could add via addopts if needed)
    if config.getoption("--include-flaky-sdk", default=False):
        return

    skip_flaky = pytest.mark.skip(
        reason="flaky_sdk tests skipped by default (use -m 'e2e and flaky_sdk' to run)"
    )
    for item in items:
        if "flaky_sdk" in [marker.name for marker in item.iter_markers()]:
            item.add_marker(skip_flaky)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add --include-flaky-sdk option."""
    parser.addoption(
        "--include-flaky-sdk",
        action="store_true",
        default=False,
        help="Include flaky_sdk tests that are skipped by default",
    )

"""E2E test configuration - Claude auth and credential setup."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest  # noqa: TC002 - needed at runtime for pytest_configure

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


def pytest_collection_modifyitems(items: list[Item]) -> None:
    """Add reruns to flaky_sdk tests."""
    for item in items:
        if item.get_closest_marker("flaky_sdk"):
            item.add_marker(pytest.mark.flaky(reruns=2, reruns_delay=1))

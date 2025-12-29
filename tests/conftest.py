"""Pytest configuration for mala tests."""

import os
import shutil
from pathlib import Path

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Configure test environment before collection.

    Sets up environment variables to:
    - Disable Braintrust tracing
    - Redirect run metadata to /tmp to avoid polluting ~/.config/mala/runs/
    - Redirect Claude SDK logs to /tmp to avoid polluting ~/.claude/projects/
    - Copy OAuth credentials to test config dir for SDK integration tests
    """
    # Remove BRAINTRUST_API_KEY to disable Braintrust tracing
    os.environ.pop("BRAINTRUST_API_KEY", None)

    # Redirect run metadata to /tmp to avoid polluting user config
    os.environ["MALA_RUNS_DIR"] = "/tmp/mala-test-runs"

    # Redirect Claude SDK logs to /tmp to avoid polluting user Claude config
    test_claude_dir = Path("/tmp/mala-test-claude")
    test_claude_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CLAUDE_CONFIG_DIR"] = str(test_claude_dir)

    # Copy OAuth credentials to test config dir so SDK integration tests work
    real_credentials = Path.home() / ".claude" / ".credentials.json"
    if real_credentials.exists():
        shutil.copy2(real_credentials, test_claude_dir / ".credentials.json")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Default unmarked tests to unit category."""
    for item in items:
        if any(marker in item.keywords for marker in ("unit", "integration", "e2e")):
            continue
        item.add_marker(pytest.mark.unit)

"""Pytest configuration for mala tests."""

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Configure test environment before collection.

    Sets up environment variables to:
    - Disable Braintrust tracing
    - Redirect run metadata to /tmp to avoid polluting ~/.config/mala/runs/
    - Redirect Claude SDK logs to /tmp to avoid polluting ~/.claude/projects/
    """
    # Remove BRAINTRUST_API_KEY to disable Braintrust tracing
    os.environ.pop("BRAINTRUST_API_KEY", None)

    # Redirect run metadata to /tmp to avoid polluting user config
    os.environ["MALA_RUNS_DIR"] = "/tmp/mala-test-runs"

    # Redirect Claude SDK logs to /tmp to avoid polluting user Claude config
    os.environ["CLAUDE_CONFIG_DIR"] = "/tmp/mala-test-claude"

"""Pytest configuration for mala tests."""

import os


def pytest_configure(config):
    """Disable Braintrust during test runs by removing the API key."""
    # Remove BRAINTRUST_API_KEY to disable Braintrust tracing
    os.environ.pop("BRAINTRUST_API_KEY", None)

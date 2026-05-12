"""E2E smoke checks for the Cerberus v2 binary."""

from __future__ import annotations

import shutil
import subprocess

import pytest


pytestmark = pytest.mark.skipif(
    shutil.which("cerberus") is None,
    reason="cerberus binary not found in PATH",
)


@pytest.mark.e2e
def test_cerberus_binary_available() -> None:
    """The v2 cerberus binary is available for real e2e runs."""
    result = subprocess.run(
        ["cerberus", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_amp_smoke_workflow_overrides_default_pytest_addopts() -> None:
    workflow = Path(".github/workflows/amp-smoke.yml").read_text()

    assert (
        'uv run pytest -o addopts="" -m e2e tests/e2e/test_amp_real_cli.py' in workflow
    )

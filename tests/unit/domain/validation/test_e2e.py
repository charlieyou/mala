"""Unit tests for E2E validation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

from src.domain.validation.e2e import _disable_fixture_e2e_config


def test_disable_fixture_e2e_config_inserts_null(tmp_path: Path) -> None:
    config_path = tmp_path / "mala.yaml"
    config_path.write_text(
        'preset: python-uv\nvalidation_triggers:\n  run_end:\n    failure_mode: continue\n    commands:\n      - ref: test\n        command: "pytest"\n'
    )

    _disable_fixture_e2e_config(tmp_path)

    updated = config_path.read_text()
    assert "commands:" in updated
    assert "e2e: null" in updated

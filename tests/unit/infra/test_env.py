"""Tests for infra environment path helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.infra.tools.env import (
    encode_repo_path,
    get_repo_validation_log_dir,
    get_validation_log_dir,
)


@pytest.mark.unit
def test_validation_log_dir_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MALA_VALIDATION_LOG_DIR", raising=False)

    assert get_validation_log_dir() == Path("/tmp/mala-validation-logs")


@pytest.mark.unit
def test_validation_log_dir_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MALA_VALIDATION_LOG_DIR", "/tmp/custom-validation-logs")

    assert get_validation_log_dir() == Path("/tmp/custom-validation-logs")


@pytest.mark.unit
def test_repo_validation_log_dir_is_repo_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MALA_VALIDATION_LOG_DIR", "/tmp/custom-validation-logs")
    repo_path = tmp_path / "repo"

    assert get_repo_validation_log_dir(repo_path) == (
        Path("/tmp/custom-validation-logs") / encode_repo_path(repo_path)
    )

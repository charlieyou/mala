"""Tests for infra environment path helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.infra.tools.env import (
    encode_repo_path,
    get_claude_log_path,
    get_repo_validation_log_dir,
)


@pytest.mark.unit
def test_repo_validation_log_dir_is_repo_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MALA_VALIDATION_LOG_DIR", "/tmp/custom-validation-logs")
    repo_path = tmp_path / "repo"

    assert get_repo_validation_log_dir(repo_path) == (
        Path("/tmp/custom-validation-logs") / encode_repo_path(repo_path)
    )


@pytest.mark.unit
def test_repo_validation_log_dir_preserves_hidden_directory_dots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MALA_VALIDATION_LOG_DIR", "/tmp/custom-validation-logs")
    repo_path = Path("/Users/charlieyou/code/rwa/rwa_db/.claude/worktrees/shadow_write-infra")

    assert get_repo_validation_log_dir(repo_path) == Path(
        "/tmp/custom-validation-logs/"
        "-Users-charlieyou-code-rwa-rwa-db-.claude-worktrees-shadow-write-infra"
    )


@pytest.mark.unit
def test_claude_log_path_matches_hidden_directory_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/tmp/claude-config")
    repo_path = Path("/Users/charlieyou/code/rwa/rwa-db/.claude/worktrees/shadow-write-infra")

    assert (
        get_claude_log_path(repo_path, "session-123")
        == Path(
            "/tmp/claude-config/projects/"
            "-Users-charlieyou-code-rwa-rwa-db--claude-worktrees-shadow-write-infra/"
            "session-123.jsonl"
        )
    )

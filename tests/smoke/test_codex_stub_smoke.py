"""Smoke test: ``mala run --coder codex`` reaches the CodexAgentProvider.

End-to-end evidence for AC #1 + AC #14 of
plans/2026-05-07-codex-provider-plan.md: selecting ``coder=codex`` via the
CLI must reach
:class:`src.infra.clients.codex_provider.CodexAgentProvider` and, when
prerequisites are missing, surface a structured fail-closed error
(:class:`CodexNotInstalledError` or :class:`CodexHookNotActiveError`) so
users see a clear failure instead of a half-built session pipeline.

The unit/integration tests for AC #1 mock ``_create_orchestrator`` and assert
on factory wiring; this smoke test invokes the real ``mala`` binary as a
subprocess so the assertion covers the full CLI path.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


_SUBPROCESS_TIMEOUT_SECONDS = 60.0


def _require_cli(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        pytest.skip(f"{name} CLI not found in PATH")
    return path


def _run_checked(command: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"Command failed: {' '.join(command)}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def _prepare_ready_issue(repo_path: Path) -> None:
    """Create a minimal real repo with one ready beads issue."""
    _run_checked(["git", "init"], cwd=repo_path)
    (repo_path / "README.md").write_text("# smoke\n", encoding="utf-8")
    _run_checked(["git", "add", "README.md"], cwd=repo_path)
    _run_checked(
        [
            "git",
            "-c",
            "user.name=Mala Smoke",
            "-c",
            "user.email=mala-smoke@example.com",
            "commit",
            "-m",
            "Initial",
        ],
        cwd=repo_path,
    )
    _run_checked(["br", "init"], cwd=repo_path)
    _run_checked(["br", "create", "Codex smoke issue", "-p", "1"], cwd=repo_path)


def test_run_coder_codex_fails_closed_on_missing_prerequisites(tmp_path: Path) -> None:
    """``mala run --coder codex`` exits non-zero with a structured error.

    The fixture creates a real ready beads issue so ``mala run`` must
    construct the Codex provider, then forces ``CODEX_BINARY`` to a missing
    path while seeding a minimal auth file. That keeps the smoke assertion
    on the real CLI/provider fail-closed path instead of accidentally
    passing because the temp repo had no work or because auth was missing
    before the live app-server spawn.
    """
    mala = _require_cli("mala")
    _require_cli("br")
    _prepare_ready_issue(tmp_path)

    env = dict(os.environ)
    missing_codex = tmp_path / "missing-codex-binary"
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        '{"OPENAI_API_KEY":"sk-smoke-test"}\n', encoding="utf-8"
    )
    env["CODEX_BINARY"] = str(missing_codex)
    env["CODEX_HOME"] = str(codex_home)
    for auth_var in ("OPENAI_API_KEY", "CODEX_API_KEY", "CODEX_ACCESS_TOKEN"):
        env.pop(auth_var, None)

    completed = subprocess.run(
        [mala, "run", str(tmp_path), "--coder", "codex"],
        env=env,
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
        check=False,
    )

    combined = (completed.stdout or "") + (completed.stderr or "")

    assert completed.returncode != 0, (
        f"Expected non-zero exit when --coder codex misses prerequisites; "
        f"got {completed.returncode}. Output:\n{combined}"
    )
    assert (
        "CodexNotInstalledError" in combined or "CodexHookNotActiveError" in combined
    ), (
        "Structured fail-closed error class missing from CLI output; the "
        f"CLI may have swallowed the exception. Output:\n{combined}"
    )
    assert "missing-codex-binary" in combined and "Codex binary not found" in combined, (
        "CLI output did not mention the forced missing CODEX_BINARY path; "
        f"the smoke test may not have reached the intended branch. Output:\n{combined}"
    )

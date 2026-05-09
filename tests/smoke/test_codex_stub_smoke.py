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

import shutil
import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


_SUBPROCESS_TIMEOUT_SECONDS = 60.0


def _require_mala_cli() -> None:
    if shutil.which("mala") is None:
        pytest.skip("mala CLI not found in PATH")


def test_run_coder_codex_fails_closed_on_missing_prerequisites(tmp_path: Path) -> None:
    """``mala run --coder codex`` exits non-zero with a structured error.

    ``install_prerequisites`` (invoked once during ``create_orchestrator``
    before any session spawns) raises :class:`CodexNotInstalledError`
    when ``codex_app_server`` is not importable, or
    :class:`CodexHookNotActiveError` when the bundled hook cannot be
    proven active. The CLI does not catch either, so the class name
    surfaces in stderr/stdout and the process exits with a non-zero
    code. Either error is acceptable here — the smoke test asserts the
    fail-closed posture, not the precise reason (which depends on the
    local environment's Codex install state).
    """
    _require_mala_cli()

    completed = subprocess.run(
        ["mala", "run", str(tmp_path), "--coder", "codex"],
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

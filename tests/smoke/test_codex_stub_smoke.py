"""Smoke test: ``mala run --coder codex`` reaches the CodexAgentProvider stub.

End-to-end evidence for AC #1 of plans/2026-05-07-codex-provider-plan.md
(Phase B): selecting ``coder=codex`` via the CLI must reach
:class:`src.infra.clients.codex_provider.CodexAgentProvider` and surface its
"not yet implemented" error so users see a clear failure instead of a
half-built session pipeline.

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


def test_run_coder_codex_raises_not_implemented_stub(tmp_path: Path) -> None:
    """``mala run --coder codex`` exits non-zero with the stub error.

    The Phase B stub raises :class:`CodexNotImplementedError` from
    ``install_prerequisites``, which is invoked once during
    ``create_orchestrator`` before any session spawns. The CLI does not
    catch this exception, so it surfaces in stderr/stdout and the process
    exits with code 1.
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
        f"Expected non-zero exit when --coder codex hits the stub; got "
        f"{completed.returncode}. Output:\n{combined}"
    )
    assert "CodexNotImplementedError" in combined, (
        "Stub error class missing from CLI output; the CLI may have "
        f"swallowed the exception. Output:\n{combined}"
    )
    assert "not yet implemented" in combined.lower() or "Phases C-I" in combined, (
        "Stub message missing from CLI output; the provider may not have "
        f"been reached. Output:\n{combined}"
    )

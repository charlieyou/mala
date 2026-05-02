"""Run the ``mala-safety.ts`` shell-write classifier unit tests under Bun.

These tests pin AC-9a parser correctness against the adversarial inputs
external review flagged in bd-mala-eymhx.20: ``/dev/fd/`` traversal,
``$(...)`` / backtick command substitution, ``>|`` clobber-override,
mixed-quoting concatenation, ``)`` separator handling, and
command-position anchoring of ``tee``/``mv``/``cp``/``dd``/etc.

Gating
------
Skipped when ``bun`` is not on PATH. The plugin runs under Bun in
production (Amp's PLUGINS=all loader uses Bun for ``.ts`` plugins), so
running the same tests with the same runtime is the highest-fidelity
parity check available without spawning real Amp.

The :file:`tests/integration/test_amp_lock_enforcement.py` module
exercises the same plugin end-to-end via a real ``amp --execute``
process; this module isolates the parser itself so regressions surface
without LLM-compliance flake.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

_BUN_REASON = (
    "bun not on PATH; mala-safety.ts shell-write classifier unit tests "
    "require Bun (the same runtime Amp's PLUGINS=all loader uses)."
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(shutil.which("bun") is None, reason=_BUN_REASON),
]


_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "amp"
_TEST_FILE = _PLUGIN_DIR / "mala-safety.test.ts"
_PLUGIN_SOURCE = _PLUGIN_DIR / "mala-safety.ts"


def test_plugin_directory_layout() -> None:
    """Pre-flight: assert the test wrapper points at a real Bun test file.

    A pure subprocess invocation gives no clue when the path is wrong
    (Bun reports "no tests found" with exit 1, which is identical to a
    real failure). Asserting the file exists turns a typo here into a
    meaningful fail.
    """
    assert _PLUGIN_SOURCE.exists(), f"plugin source missing: {_PLUGIN_SOURCE}"
    assert _TEST_FILE.exists(), f"plugin test file missing: {_TEST_FILE}"


def test_mala_safety_shell_classifier() -> None:
    """Run the bun test suite for ``mala-safety.test.ts``.

    Failure here means one of the adversarial classifier scenarios
    regressed. The bun test output includes the exact failing assertion
    and the file:line of the source-of-truth case.
    """
    result = subprocess.run(
        ["bun", "test", _TEST_FILE.name],
        cwd=str(_PLUGIN_DIR),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(
            "bun test failed for mala-safety.test.ts "
            f"(exit {result.returncode}).\n"
            f"--- bun stdout ---\n{result.stdout}\n"
            f"--- bun stderr ---\n{result.stderr}"
        )

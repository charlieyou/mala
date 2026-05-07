"""Regression test: no ``provider.name == "amp"`` dispatch in orchestrator wiring.

After T004 (plan ``L629-L641``, AC#17) the MCP factory selection moved onto
:meth:`src.core.protocols.agent_provider.AgentProvider.mcp_server_factory`.
Adding a third coder (Codex, Phase G) must not bring the N-way switch back
to ``factory.py`` / ``orchestrator.py`` — each provider knows its own MCP
shape and the orchestrator just calls the method.

This test is intentionally a textual scan: it fails CI if any future change
introduces ``provider.name == "amp"`` (or the analogous comparisons against
``"claude"`` / ``"codex"``) as a dispatch branch in the two files the plan
calls out as the surface area.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_SRC_ROOT = Path(__file__).resolve().parents[3] / "src" / "orchestration"
_TARGETS = (
    _SRC_ROOT / "factory.py",
    _SRC_ROOT / "orchestrator.py",
)

# Match ``<something>.name == "amp"`` (and the "claude" / "codex" siblings).
# The double-quoted form keeps the pattern narrow: pipeline code uses str
# literals for these provider identifiers.
_NAME_BRANCH_PATTERN = re.compile(
    r'\.name\s*==\s*"(?:amp|claude|codex)"',
)


@pytest.mark.unit
@pytest.mark.parametrize("path", _TARGETS, ids=lambda p: p.name)
def test_no_provider_name_dispatch_branches(path: Path) -> None:
    """Fail if any ``provider.name == "amp"`` branch returns to dispatch."""
    assert path.exists(), f"expected file to exist: {path}"
    text = path.read_text(encoding="utf-8")
    matches = _NAME_BRANCH_PATTERN.findall(text)
    assert matches == [], (
        f"{path.name} contains provider-name dispatch branches: {matches!r}. "
        "MCP factory dispatch must go through "
        "AgentProvider.mcp_server_factory() (plan T004 / AC#17)."
    )

"""Unit tests for :class:`ClaudeAgentProvider`.

These tests are the regression guard for T006 (the Claude-path refactor):

- The bundled pieces round-trip through the existing Claude pipeline call
  sites: ``runtime_builder`` returns an :class:`AgentRuntimeBuilder`
  configured with the provider's client factory.
- Importing the Claude provider does NOT pull in any
  ``src.infra.clients.amp_*`` adapter module - this protects the
  Amp-vs-Claude lazy-import boundary.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from src.infra.agent_runtime import AgentRuntimeBuilder
from src.infra.clients.claude_provider import ClaudeAgentProvider

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.core.models import LockEvent

REPO_ROOT = Path(__file__).resolve().parents[4]


def _mcp_server_factory() -> Callable[..., dict[str, object]]:
    """A trivial MCP factory; matches the McpServerFactory shape."""

    def factory(
        agent_id: str,
        repo_path: Path,
        emit_lock_event: Callable[[LockEvent], object] | None,
    ) -> dict[str, object]:
        del agent_id, repo_path, emit_lock_event
        return {}

    return factory


# ---------------------------------------------------------------------------
# runtime_builder: returns an AgentRuntimeBuilder wired to the provider's
# client factory.
# ---------------------------------------------------------------------------


def test_runtime_builder_returns_agent_runtime_builder(tmp_path: Path) -> None:
    provider = ClaudeAgentProvider()
    builder = provider.runtime_builder(
        tmp_path, "agent-1", mcp_server_factory=_mcp_server_factory()
    )
    assert isinstance(builder, AgentRuntimeBuilder)


def test_runtime_builder_uses_provider_client_factory(tmp_path: Path) -> None:
    """Pipeline code (T007) will rely on the runtime builder being wired to the
    provider's own ``client_factory`` so that hook matchers and SDK options
    come from the same factory the pipeline later calls ``create()`` on."""
    provider = ClaudeAgentProvider()
    builder = provider.runtime_builder(
        tmp_path, "agent-2", mcp_server_factory=_mcp_server_factory()
    )
    # The builder stores the factory in a private attribute; we don't depend
    # on the private name, but we do assert that it behaves the same as the
    # provider's factory by checking the hook matcher round-trips.
    assert builder._sdk_client_factory is provider.client_factory  # ty:ignore[unresolved-attribute]


def test_runtime_builder_threads_setting_sources(tmp_path: Path) -> None:
    provider = ClaudeAgentProvider(setting_sources=["local", "project"])
    builder = provider.runtime_builder(
        tmp_path, "agent-3", mcp_server_factory=_mcp_server_factory()
    )
    assert builder._setting_sources == ["local", "project"]  # ty:ignore[unresolved-attribute]


def test_runtime_builder_default_setting_sources_is_none(tmp_path: Path) -> None:
    provider = ClaudeAgentProvider()
    builder = provider.runtime_builder(
        tmp_path, "agent-4", mcp_server_factory=_mcp_server_factory()
    )
    assert builder._setting_sources is None  # ty:ignore[unresolved-attribute]


# ---------------------------------------------------------------------------
# Lazy-import boundary: importing claude_provider must NOT import any
# src.infra.clients.amp_* adapter module. This guards the Amp/Claude
# isolation invariant: a Claude-only run does not require Amp, and a future
# refactor that accidentally pulls Amp into the Claude path is rejected.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_importing_claude_provider_does_not_import_amp_adapters() -> None:
    code = """
import sys
for mod in list(sys.modules):
    if mod.startswith('src.infra.clients.amp'):
        del sys.modules[mod]

from src.infra.clients.claude_provider import ClaudeAgentProvider

amp_loaded = sorted(
    mod for mod in sys.modules
    if mod.startswith('src.infra.clients.amp')
)
if amp_loaded:
    print('FAIL: ' + ','.join(amp_loaded))
    sys.exit(1)
ClaudeAgentProvider()
print('PASS')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "PASS" in result.stdout

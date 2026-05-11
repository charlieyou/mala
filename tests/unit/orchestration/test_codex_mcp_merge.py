"""Codex MCP server merge tests (Phase G3, T016).

Verifies that :class:`CodexAgentProvider`'s MCP factory merges
user-supplied ``coder_options.codex.mcp_servers`` with the bundled
``mala-locking`` launch spec — and that the bundled key is **never
overridden** by user config (plan G3).

The factory itself never reaches Codex inline: per
:class:`CodexClient.query`, MCP wiring flows through the bundled plugin's
``.mcp.json``. The merged map surfaces on
:attr:`CodexRuntime.mcp_servers` for forward-compat / inspection — these
tests assert on that surface so a future SDK that consumes inline specs
sees the resolved configuration the bundled plugin presents.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from src.infra.clients.codex_mcp_factory import (
    CODEX_BUNDLED_MCP_LAUNCHER_COMMAND,
    CODEX_BUNDLED_MCP_SERVER_NAME,
)
from src.infra.clients.codex_provider import CodexAgentProvider


pytestmark = pytest.mark.unit


def _invoke_factory(
    provider: CodexAgentProvider, *, agent_id: str = "agent-x", repo: str = "/tmp/repo"
) -> dict[str, object]:
    factory = provider.mcp_server_factory()
    return factory(agent_id, Path(repo), None)


def _bundled_spec(servers: dict[str, object]) -> dict[str, object]:
    raw = servers[CODEX_BUNDLED_MCP_SERVER_NAME]
    assert isinstance(raw, dict)
    return cast("dict[str, object]", raw)


def _spec_env(spec: dict[str, object]) -> dict[str, str]:
    raw = spec["env"]
    assert isinstance(raw, dict)
    return cast("dict[str, str]", raw)


def test_bundled_mala_locking_always_present_with_no_user_config() -> None:
    """G1: bundled ``mala-locking`` launch spec is emitted unconditionally."""
    provider = CodexAgentProvider()

    servers = _invoke_factory(provider)

    assert CODEX_BUNDLED_MCP_SERVER_NAME in servers
    bundled = _bundled_spec(servers)
    assert bundled["command"] == CODEX_BUNDLED_MCP_LAUNCHER_COMMAND


def test_bundled_spec_carries_agent_and_repo_namespace() -> None:
    """The launcher must receive ``agent_id`` + ``repo_namespace`` so its
    lock-key derivation matches the rest of the run.

    This is the cross-coder contract that lets the same ``<hash>.lock``
    files be read by both Codex's bundled plugin and any other tool
    using ``src.infra.tools.locking``.
    """
    provider = CodexAgentProvider()

    servers = _invoke_factory(provider, agent_id="agent-codex-A", repo="/work/repo")

    bundled = _bundled_spec(servers)
    assert bundled["args"] == [
        "--agent-id",
        "agent-codex-A",
        "--repo-namespace",
        "/work/repo",
    ]
    env = _spec_env(bundled)
    assert env["MALA_AGENT_ID"] == "agent-codex-A"
    assert env["MALA_REPO_NAMESPACE"] == "/work/repo"


def test_user_supplied_non_clashing_server_merges_with_bundled() -> None:
    """G3: user keys other than ``mala-locking`` pass through alongside bundled."""
    user_spec: dict[str, object] = {
        "command": "my-custom-mcp",
        "args": [],
        "env": {},
    }
    provider = CodexAgentProvider(
        mcp_servers=(("custom-server", user_spec),),
    )

    servers = _invoke_factory(provider)

    assert servers["custom-server"] == user_spec
    assert CODEX_BUNDLED_MCP_SERVER_NAME in servers


def test_user_attempt_to_override_bundled_is_replaced() -> None:
    """G3 (anti-override): a user ``mala-locking`` entry is silently replaced.

    Bundled is mandatory — letting the user redirect ``mala-locking`` at
    a different binary would bypass the safety contract the plugin's
    ``.mcp.json`` was wired against (plan G3 ``bundled is mandatory,
    never replaced``).
    """
    hostile_user_spec: dict[str, object] = {
        "command": "/tmp/evil-launcher",
        "args": [],
        "env": {},
    }
    provider = CodexAgentProvider(
        mcp_servers=((CODEX_BUNDLED_MCP_SERVER_NAME, hostile_user_spec),),
    )

    servers = _invoke_factory(provider)

    bundled = _bundled_spec(servers)
    assert bundled["command"] == CODEX_BUNDLED_MCP_LAUNCHER_COMMAND
    assert bundled["command"] != hostile_user_spec["command"]


def test_multiple_user_servers_all_present_with_bundled_intact() -> None:
    """Mixed merge: many non-clashing user servers + bundled all coexist."""
    spec_a: dict[str, object] = {"command": "tool-a", "args": [], "env": {}}
    spec_b: dict[str, object] = {"command": "tool-b", "args": [], "env": {}}
    provider = CodexAgentProvider(
        mcp_servers=(("server-a", spec_a), ("server-b", spec_b)),
    )

    servers = _invoke_factory(provider)

    assert servers["server-a"] == spec_a
    assert servers["server-b"] == spec_b
    bundled = _bundled_spec(servers)
    assert bundled["command"] == CODEX_BUNDLED_MCP_LAUNCHER_COMMAND


def test_factory_forwards_mala_lock_dir_when_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``MALA_LOCK_DIR`` is set in the orchestrator env, the bundled
    spec forwards it to the spawned launcher; when unset, the launcher
    falls through to the default. Mirrors Amp's behavior at
    ``amp_provider.py:127-129``.
    """
    monkeypatch.setenv("MALA_LOCK_DIR", "/tmp/run-locks")
    provider = CodexAgentProvider()

    servers = _invoke_factory(provider)
    env = _spec_env(_bundled_spec(servers))
    assert env["MALA_LOCK_DIR"] == "/tmp/run-locks"

    monkeypatch.delenv("MALA_LOCK_DIR", raising=False)
    servers = _invoke_factory(provider)
    env = _spec_env(_bundled_spec(servers))
    assert "MALA_LOCK_DIR" not in env

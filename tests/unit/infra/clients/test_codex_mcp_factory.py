"""Unit tests for :mod:`src.infra.clients.codex_mcp_factory` launch-spec builders.

Covers the pure ``command`` / ``args`` / ``env`` shape contract of the
helpers extracted from :mod:`src.infra.clients.codex_provider` in
Workstream B (finding #3). End-to-end provider wiring and the
``mcp_server_factory()`` callable that closes over user-supplied
servers live in :mod:`tests.unit.orchestration.test_codex_mcp_merge`;
this file pins the extracted helpers' shape directly so a future
provider rewire cannot silently drift the launch-spec payload.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from src.infra.clients.codex_mcp_factory import (
    CODEX_BUNDLED_MCP_LAUNCHER_COMMAND,
    CODEX_BUNDLED_MCP_SERVER_NAME,
    _build_bundled_codex_mcp_spec,
    _build_merged_codex_plugin_mcp_json,
    _create_codex_mcp_server_factory,
)


pytestmark = pytest.mark.unit


class TestBuildBundledCodexMcpSpec:
    def test_returns_command_args_env_shape(self) -> None:
        spec = _build_bundled_codex_mcp_spec("agent-x", Path("/work/repo"))

        assert set(spec.keys()) == {"command", "args", "env"}
        assert spec["command"] == CODEX_BUNDLED_MCP_LAUNCHER_COMMAND

    def test_args_carry_agent_id_and_repo_namespace(self) -> None:
        spec = _build_bundled_codex_mcp_spec("agent-codex-A", Path("/work/repo"))

        assert spec["args"] == [
            "--agent-id",
            "agent-codex-A",
            "--repo-namespace",
            "/work/repo",
        ]

    def test_env_contains_agent_id_and_repo_namespace(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("MALA_LOCK_DIR", raising=False)

        spec = _build_bundled_codex_mcp_spec("agent-x", Path("/work/repo"))

        env = cast("dict[str, str]", spec["env"])
        assert env["MALA_AGENT_ID"] == "agent-x"
        assert env["MALA_REPO_NAMESPACE"] == "/work/repo"
        assert "MALA_LOCK_DIR" not in env

    def test_env_forwards_mala_lock_dir_when_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MALA_LOCK_DIR", "/tmp/run-locks")

        spec = _build_bundled_codex_mcp_spec("agent-x", Path("/work/repo"))

        env = cast("dict[str, str]", spec["env"])
        assert env["MALA_LOCK_DIR"] == "/tmp/run-locks"


class TestBuildMergedCodexPluginMcpJson:
    def test_bundled_only_payload_when_no_user_servers(self) -> None:
        raw = _build_merged_codex_plugin_mcp_json()

        payload = json.loads(raw.decode("utf-8"))
        servers = payload["mcpServers"]
        assert list(servers.keys()) == [CODEX_BUNDLED_MCP_SERVER_NAME]
        bundled = servers[CODEX_BUNDLED_MCP_SERVER_NAME]
        assert bundled["command"] == CODEX_BUNDLED_MCP_LAUNCHER_COMMAND
        assert bundled["args"] == []
        assert bundled["env"] == {}
        assert "default_tools_approval_mode" not in bundled
        assert bundled["env_vars"] == [
            "MALA_AGENT_ID",
            "MALA_LOCK_DIR",
            "MALA_REPO_NAMESPACE",
        ]

    def test_user_non_clashing_servers_are_preserved(self) -> None:
        user_spec: dict[str, object] = {
            "command": "my-custom-mcp",
            "args": ["--flag"],
            "env": {"X": "1"},
        }

        raw = _build_merged_codex_plugin_mcp_json((("custom-server", user_spec),))

        payload = json.loads(raw.decode("utf-8"))
        servers = payload["mcpServers"]
        assert servers["custom-server"] == user_spec
        assert CODEX_BUNDLED_MCP_SERVER_NAME in servers

    def test_user_attempt_to_override_bundled_is_replaced(self) -> None:
        hostile: dict[str, object] = {
            "command": "/tmp/evil-launcher",
            "args": [],
            "env": {},
        }

        raw = _build_merged_codex_plugin_mcp_json(
            ((CODEX_BUNDLED_MCP_SERVER_NAME, hostile),),
        )

        payload = json.loads(raw.decode("utf-8"))
        bundled = payload["mcpServers"][CODEX_BUNDLED_MCP_SERVER_NAME]
        assert bundled["command"] == CODEX_BUNDLED_MCP_LAUNCHER_COMMAND

    def test_emits_trailing_newline_and_utf8_bytes(self) -> None:
        raw = _build_merged_codex_plugin_mcp_json()

        assert isinstance(raw, bytes)
        assert raw.endswith(b"\n")


class TestCreateCodexMcpServerFactory:
    def test_factory_emits_bundled_entry_with_no_user_config(self) -> None:
        factory = _create_codex_mcp_server_factory()

        servers = factory("agent-x", Path("/work/repo"), None)

        assert CODEX_BUNDLED_MCP_SERVER_NAME in servers
        bundled = cast("dict[str, object]", servers[CODEX_BUNDLED_MCP_SERVER_NAME])
        assert bundled["command"] == CODEX_BUNDLED_MCP_LAUNCHER_COMMAND
        assert bundled["args"] == [
            "--agent-id",
            "agent-x",
            "--repo-namespace",
            "/work/repo",
        ]

    def test_factory_passes_through_non_clashing_user_servers(self) -> None:
        user_spec: dict[str, object] = {
            "command": "tool-a",
            "args": [],
            "env": {},
        }
        factory = _create_codex_mcp_server_factory((("server-a", user_spec),))

        servers = factory("agent-x", Path("/work/repo"), None)

        assert servers["server-a"] == user_spec
        assert CODEX_BUNDLED_MCP_SERVER_NAME in servers

    def test_factory_replaces_user_attempt_to_override_bundled(self) -> None:
        hostile: dict[str, object] = {
            "command": "/tmp/evil",
            "args": [],
            "env": {},
        }
        factory = _create_codex_mcp_server_factory(
            ((CODEX_BUNDLED_MCP_SERVER_NAME, hostile),),
        )

        servers = factory("agent-x", Path("/work/repo"), None)

        bundled = cast("dict[str, object]", servers[CODEX_BUNDLED_MCP_SERVER_NAME])
        assert bundled["command"] == CODEX_BUNDLED_MCP_LAUNCHER_COMMAND

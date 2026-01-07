"""Unit tests for Claude SDK transport SIGINT isolation."""

from __future__ import annotations

import sys

import anyio
import pytest
from claude_agent_sdk.types import ClaudeAgentOptions

from src.infra.sdk_transport import ensure_sigint_isolated_cli_transport
from src.infra.tools import command_runner

unix_only = pytest.mark.skipif(sys.platform == "win32", reason="Unix-only test")


class FakeProcess:
    """Minimal async process stub for transport tests."""

    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.returncode: int | None = None
        self.stdin = None
        self.stdout = None
        self.stderr = None

    def terminate(self) -> None:
        pass

    async def wait(self) -> int:
        return 0


@unix_only
@pytest.mark.asyncio
async def test_cli_transport_registers_sigint_pgid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI transport runs in its own session and registers for SIGINT forwarding."""
    monkeypatch.setenv("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")
    command_runner._SIGINT_FORWARD_PGIDS.clear()
    ensure_sigint_isolated_cli_transport()

    captured: dict[str, object] = {}

    async def fake_open_process(*args: object, **kwargs: object) -> FakeProcess:
        captured["start_new_session"] = kwargs.get("start_new_session")
        return FakeProcess(pid=9001)

    monkeypatch.setattr(anyio, "open_process", fake_open_process)

    from claude_agent_sdk._internal.transport import subprocess_cli

    options = ClaudeAgentOptions(cli_path="/bin/echo")
    transport = subprocess_cli.SubprocessCLITransport(prompt="hi", options=options)

    await transport.connect()
    assert captured["start_new_session"] is True
    assert command_runner._SIGINT_FORWARD_PGIDS == {9001}

    await transport.close()
    assert command_runner._SIGINT_FORWARD_PGIDS == set()

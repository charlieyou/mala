"""Unit tests for Claude SDK transport SIGINT isolation."""

from __future__ import annotations

import signal
import subprocess
import sys
import types
from unittest.mock import patch

import anyio
import pytest
from claude_agent_sdk.types import ClaudeAgentOptions

from src.infra.sdk_transport import (
    ensure_codex_sigint_isolated_app_server,
    ensure_sigint_isolated_cli_transport,
)
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


@unix_only
@pytest.mark.asyncio
async def test_cli_transport_resolves_default_cli_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI transport resolves cli_path before building the subprocess command."""
    monkeypatch.setenv("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")
    command_runner._SIGINT_FORWARD_PGIDS.clear()
    ensure_sigint_isolated_cli_transport()

    captured: dict[str, object] = {}

    async def fake_open_process(*args: object, **kwargs: object) -> FakeProcess:
        captured["cmd"] = args[0]
        return FakeProcess(pid=9003)

    monkeypatch.setattr(anyio, "open_process", fake_open_process)

    from claude_agent_sdk._internal.transport import subprocess_cli

    options = ClaudeAgentOptions()
    transport = subprocess_cli.SubprocessCLITransport(prompt="hi", options=options)
    monkeypatch.setattr(transport, "_find_cli", lambda: "/bin/echo")

    await transport.connect()
    cmd = captured["cmd"]
    assert isinstance(cmd, list)
    assert cmd[0] == "/bin/echo"

    await transport.close()


@unix_only
@pytest.mark.asyncio
async def test_cli_transport_logs_sdk_subprocess_spawned_with_flow(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CLI transport logs sdk_subprocess_spawned with pid, pgid, and flow from env."""
    import logging

    monkeypatch.setenv("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")
    monkeypatch.setenv("MALA_SDK_FLOW", "epic_remediation")
    command_runner._SIGINT_FORWARD_PGIDS.clear()
    ensure_sigint_isolated_cli_transport()

    async def fake_open_process(*args: object, **kwargs: object) -> FakeProcess:
        return FakeProcess(pid=9002)

    monkeypatch.setattr(anyio, "open_process", fake_open_process)

    from claude_agent_sdk._internal.transport import subprocess_cli

    options = ClaudeAgentOptions(cli_path="/bin/echo")
    transport = subprocess_cli.SubprocessCLITransport(prompt="hi", options=options)

    with caplog.at_level(logging.INFO, logger="src.infra.sdk_transport"):
        await transport.connect()

    # Verify sdk_subprocess_spawned log was emitted with correct flow
    log_records = [r for r in caplog.records if "sdk_subprocess_spawned" in r.message]
    assert len(log_records) == 1
    assert "pid=9002" in log_records[0].message
    assert "pgid=9002" in log_records[0].message
    assert "flow=epic_remediation" in log_records[0].message

    await transport.close()


@unix_only
def test_codex_app_server_popen_isolated_and_registered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex app-server is spawned in a separate process group for SIGINT drain."""
    import codex_app_server.client as codex_client

    command_runner._SIGINT_FORWARD_PGIDS.clear()

    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 9101

    def fake_popen(*args: object, **kwargs: object) -> FakeProcess:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(codex_client, "_MALA_SIGINT_PATCHED", False, raising=False)
    monkeypatch.setattr(codex_client, "subprocess", types.SimpleNamespace(Popen=fake_popen))

    ensure_codex_sigint_isolated_app_server()

    proc = codex_client.subprocess.Popen(["codex", "app-server"])

    assert proc.pid == 9101
    assert captured["kwargs"] == {"start_new_session": True}
    assert command_runner._SIGINT_FORWARD_PGIDS == {9101}

    command_runner._SIGINT_FORWARD_PGIDS.clear()


@unix_only
def test_codex_patch_does_not_mutate_global_subprocess_popen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex Popen wrapping must be scoped to codex_app_server.client."""
    import codex_app_server.client as codex_client

    original_global_popen = subprocess.Popen
    monkeypatch.setattr(codex_client, "_MALA_SIGINT_PATCHED", False, raising=False)
    monkeypatch.setattr(codex_client, "subprocess", subprocess)

    ensure_codex_sigint_isolated_app_server()

    assert subprocess.Popen is original_global_popen
    assert codex_client.subprocess is not subprocess
    assert codex_client.subprocess.Popen is not original_global_popen


@unix_only
def test_codex_popen_registers_explicit_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If Codex SDK manages process_group, Mala forwards to that PGID."""
    import codex_app_server.client as codex_client

    command_runner._SIGINT_FORWARD_PGIDS.clear()

    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 9103

    def fake_popen(*args: object, **kwargs: object) -> FakeProcess:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(codex_client, "_MALA_SIGINT_PATCHED", False, raising=False)
    monkeypatch.setattr(codex_client, "subprocess", types.SimpleNamespace(Popen=fake_popen))

    ensure_codex_sigint_isolated_app_server()
    proc = codex_client.subprocess.Popen(["codex", "app-server"], process_group=1234)

    assert getattr(proc, "_mala_sigint_pgid") == 1234
    assert captured["kwargs"] == {"process_group": 1234}
    assert command_runner._SIGINT_FORWARD_PGIDS == {1234}

    command_runner._SIGINT_FORWARD_PGIDS.clear()


@unix_only
def test_codex_popen_process_group_zero_registers_child_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """process_group=0 means the child PID is also the process-group id."""
    import codex_app_server.client as codex_client

    command_runner._SIGINT_FORWARD_PGIDS.clear()

    class FakeProcess:
        pid = 9105

    def fake_popen(*args: object, **kwargs: object) -> FakeProcess:
        del args, kwargs
        return FakeProcess()

    monkeypatch.setattr(codex_client, "_MALA_SIGINT_PATCHED", False, raising=False)
    monkeypatch.setattr(codex_client, "subprocess", types.SimpleNamespace(Popen=fake_popen))

    ensure_codex_sigint_isolated_app_server()
    proc = codex_client.subprocess.Popen(["codex", "app-server"], process_group=0)

    assert getattr(proc, "_mala_sigint_pgid") == 9105
    assert command_runner._SIGINT_FORWARD_PGIDS == {9105}

    command_runner._SIGINT_FORWARD_PGIDS.clear()


@unix_only
def test_codex_popen_does_not_register_unknown_preexec_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If SDK owns preexec_fn grouping, Mala must not assume PID equals PGID."""
    import codex_app_server.client as codex_client

    command_runner._SIGINT_FORWARD_PGIDS.clear()

    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 9104

    def fake_preexec() -> None:
        pass

    def fake_popen(*args: object, **kwargs: object) -> FakeProcess:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(codex_client, "_MALA_SIGINT_PATCHED", False, raising=False)
    monkeypatch.setattr(codex_client, "subprocess", types.SimpleNamespace(Popen=fake_popen))

    ensure_codex_sigint_isolated_app_server()
    codex_client.subprocess.Popen(["codex", "app-server"], preexec_fn=fake_preexec)

    assert captured["kwargs"] == {"preexec_fn": fake_preexec}
    assert command_runner._SIGINT_FORWARD_PGIDS == set()


@unix_only
def test_codex_popen_preexec_with_start_new_session_is_not_inferred(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """preexec_fn makes grouping opaque even with start_new_session=True."""
    import codex_app_server.client as codex_client

    command_runner._SIGINT_FORWARD_PGIDS.clear()

    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 9106

    def fake_preexec() -> None:
        pass

    def fake_popen(*args: object, **kwargs: object) -> FakeProcess:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(codex_client, "_MALA_SIGINT_PATCHED", False, raising=False)
    monkeypatch.setattr(codex_client, "subprocess", types.SimpleNamespace(Popen=fake_popen))

    ensure_codex_sigint_isolated_app_server()
    codex_client.subprocess.Popen(
        ["codex", "app-server"],
        preexec_fn=fake_preexec,
        start_new_session=True,
    )

    assert captured["kwargs"] == {
        "preexec_fn": fake_preexec,
        "start_new_session": True,
    }
    assert command_runner._SIGINT_FORWARD_PGIDS == set()


@unix_only
def test_codex_app_server_registered_for_second_and_third_ctrl_c(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Codex receives SIGINT on 2x Ctrl-C and SIGKILL on 3x Ctrl-C."""
    import codex_app_server.client as codex_client

    command_runner._SIGINT_FORWARD_PGIDS.clear()

    class FakeProcess:
        pid = 9102

    def fake_popen(*args: object, **kwargs: object) -> FakeProcess:
        del args, kwargs
        return FakeProcess()

    monkeypatch.setattr(codex_client, "_MALA_SIGINT_PATCHED", False, raising=False)
    monkeypatch.setattr(codex_client, "subprocess", types.SimpleNamespace(Popen=fake_popen))

    ensure_codex_sigint_isolated_app_server()
    codex_client.subprocess.Popen(["codex", "app-server"])

    assert command_runner._SIGINT_FORWARD_PGIDS == {9102}

    with patch("os.killpg") as mock_killpg:
        # Stage 2: LifecycleController calls CommandRunner.forward_sigint().
        command_runner.CommandRunner.forward_sigint()
        # Stage 3: LifecycleController calls CommandRunner.kill_active_process_groups().
        command_runner.CommandRunner.kill_active_process_groups()

    assert [(call.args[0], call.args[1]) for call in mock_killpg.call_args_list] == [
        (9102, signal.SIGINT),
        (9102, signal.SIGKILL),
    ]
    assert command_runner._SIGINT_FORWARD_PGIDS == set()

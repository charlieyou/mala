"""Unit tests for :class:`CodexClient` (Phase C, T010).

The Codex SDK (``codex_app_server``) is not installed in mala's test
environment; these tests substitute a fake module via ``sys.modules``
that exposes the surface :class:`CodexClient` consumes
(``AsyncCodex``, ``TextInput``, ``AsyncThread``, ``AsyncTurnHandle``).
The fakes are minimal and reflect the contract documented in the plan
(``L540-L583``).

Coverage:

  * Lazy-import contract: importing ``codex_client`` does not import
    ``codex_app_server`` (the lazy import lives inside ``__aenter__`` /
    ``query``).
  * Lifecycle: ``async with`` enters the SDK context exactly once and
    ``disconnect`` tears it down exactly once (idempotent).
  * ``query(prompt)`` calls ``thread_start`` with the runtime's
    model/sandbox/approval_policy/mcp_servers; subsequent ``query``
    calls reuse the existing thread.
  * Resume: a runtime carrying ``resume_thread_id`` triggers
    ``thread_resume`` rather than ``thread_start`` (AC #8 partial).
  * ``receive_response`` consumes ``TurnHandle.stream()`` and emits
    coder-agnostic :class:`AgentEvent`s for the Phase C surface
    (text deltas, ``turn/completed``, ``error``); Phase D extends the
    item-level mappings.
  * ``disconnect`` calls ``TurnHandle.interrupt`` and the SDK
    ``close()`` exactly once across repeat calls (plan L732 AC-3).
  * SIGINT mid-turn cancellation reaches ``disconnect`` and invokes
    ``TurnHandle.interrupt`` + ``AsyncCodex.close`` exactly once.
  * ``session_id`` reflects the started thread's id.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from src.core.protocols.agent_event import (
    AgentResultEvent,
    AgentTextEvent,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import ModuleType
    from typing import Self

    from src.core.protocols.agent_event import AgentEventValue
    from src.infra.clients.codex_runtime import CodexRuntime


REPO_ROOT = Path(__file__).resolve().parents[5]


# ---------------------------------------------------------------------------
# Fake codex_app_server SDK
# ---------------------------------------------------------------------------


@dataclass
class _FakeTextInput:
    """Stand-in for ``codex_app_server.TextInput``."""

    prompt: str


@dataclass
class _FakeTurnHandle:
    """Stand-in for ``codex_app_server.AsyncTurnHandle``."""

    notifications: list[object] = field(default_factory=list)
    interrupt_calls: int = 0

    async def stream(self) -> AsyncIterator[object]:
        for note in list(self.notifications):
            yield note

    async def interrupt(self) -> None:
        self.interrupt_calls += 1


@dataclass
class _FakeThread:
    """Stand-in for ``codex_app_server.AsyncThread``."""

    id: str
    turn_handles: list[_FakeTurnHandle] = field(default_factory=list)
    turn_inputs: list[_FakeTextInput] = field(default_factory=list)
    turn_kwargs: list[dict[str, object]] = field(default_factory=list)

    async def turn(
        self, text_input: _FakeTextInput, **kwargs: object
    ) -> _FakeTurnHandle:
        handle = _FakeTurnHandle()
        self.turn_handles.append(handle)
        self.turn_inputs.append(text_input)
        self.turn_kwargs.append(dict(kwargs))
        return handle


@dataclass
class _FakeAppServerConfig:
    """Stand-in for ``codex_app_server.AppServerConfig``.

    The real SDK dataclass exposes ``codex_bin``, ``cwd``, ``env``, etc.
    Tests assert on these fields after :class:`CodexClient.__aenter__`
    constructs the config.
    """

    codex_bin: str | None = None
    cwd: str | None = None
    env: dict[str, str] | None = None


@dataclass
class _FakeAsyncCodex:
    """Stand-in for ``codex_app_server.AsyncCodex``.

    The real SDK constructor is ``AsyncCodex(config=AppServerConfig(...))``.
    Tracks the config, every ``thread_start`` / ``thread_resume``
    invocation, and the close lifecycle so tests can assert on
    observable behavior (the issue's plan calls for the cancellation
    path to interrupt the active turn and close ``AsyncCodex`` exactly
    once).
    """

    config: _FakeAppServerConfig | None = None
    threads_started: list[dict[str, object]] = field(default_factory=list)
    threads_resumed: list[str] = field(default_factory=list)
    enter_calls: int = 0
    exit_calls: int = 0
    close_calls: int = 0
    next_thread: _FakeThread | None = None
    resumed_thread: _FakeThread | None = None

    async def __aenter__(self) -> Self:
        self.enter_calls += 1
        return self

    async def __aexit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> None:
        del exc_type, exc_val, exc_tb
        self.exit_calls += 1

    async def close(self) -> None:
        self.close_calls += 1

    async def thread_start(self, **kwargs: object) -> _FakeThread:
        self.threads_started.append(dict(kwargs))
        if self.next_thread is None:
            self.next_thread = _FakeThread(id="thr_fake_start")
        return self.next_thread

    async def thread_resume(self, thread_id: str) -> _FakeThread:
        self.threads_resumed.append(thread_id)
        if self.resumed_thread is None:
            self.resumed_thread = _FakeThread(id=thread_id)
        return self.resumed_thread


def _install_fake_sdk(monkeypatch: pytest.MonkeyPatch) -> _FakeAsyncCodex:
    """Insert a fake ``codex_app_server`` module into ``sys.modules``.

    Returns the singleton :class:`_FakeAsyncCodex` the fake module's
    ``AsyncCodex`` factory will hand out, so tests can assert on the
    captured config / thread-start / interrupt / close calls.
    """
    import types

    fake_codex = _FakeAsyncCodex()

    def async_codex_factory(
        config: _FakeAppServerConfig | None = None, **kwargs: object
    ) -> _FakeAsyncCodex:
        if kwargs:
            raise TypeError(
                f"AsyncCodex() got unexpected keyword argument {next(iter(kwargs))!r}"
            )
        fake_codex.config = config
        return fake_codex

    fake_module: ModuleType = types.ModuleType("codex_app_server")
    fake_module.AsyncCodex = async_codex_factory  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]
    fake_module.AppServerConfig = _FakeAppServerConfig  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]
    fake_module.TextInput = _FakeTextInput  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "codex_app_server", fake_module)
    return fake_codex


def _build_runtime(
    tmp_path: Path, *, resume_thread_id: str | None = None
) -> CodexRuntime:
    """Build a minimal :class:`CodexRuntime` for a CodexClient under test."""
    from src.infra.clients.codex_runtime import CodexRuntimeBuilder

    def factory(
        agent_id: str, repo_path: Path, emit_lock_event: object
    ) -> dict[str, object]:
        del agent_id, repo_path, emit_lock_event
        return {"mala-locking": {"command": "mala-codex-mcp-locking"}}

    builder = CodexRuntimeBuilder(
        tmp_path,
        "agent-x",
        factory,
        model="gpt-5.5",
        effort="medium",
        approval_policy="never",
        sandbox="danger-full-access",
    )
    if resume_thread_id is not None:
        builder = builder.with_resume(resume_thread_id)
    return builder.build()


# ---------------------------------------------------------------------------
# Lazy-import contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_importing_codex_client_does_not_pull_codex_app_server() -> None:
    """``import src.infra.clients.codex_client`` must not load the SDK."""
    code = """
import sys
import src.infra.clients.codex_client  # noqa: F401
loaded = sorted(m for m in sys.modules if m.startswith('codex_app_server'))
if loaded:
    print('FAIL: ' + ','.join(loaded))
    sys.exit(1)
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


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_aenter_constructs_async_codex_with_runtime_env_and_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``__aenter__`` constructs ``AsyncCodex(config=AppServerConfig(...))``.

    The real SDK shape (``codex_app_server.AsyncCodex.__init__``,
    ``codex_app_server/api.py:291``) takes a single ``config`` arg with
    the per-process env / cwd overlay on
    :class:`AppServerConfig`. Without the env overlay, the bundled
    Phase E hook would see no ``MALA_AGENT_ID`` and deny every shell
    write fail-closed.
    """
    from src.infra.clients.codex_client import CodexClient

    fake_codex = _install_fake_sdk(monkeypatch)
    runtime = _build_runtime(tmp_path)

    async with CodexClient(runtime) as client:
        assert fake_codex.enter_calls == 1
        config = fake_codex.config
        assert config is not None
        assert config.cwd == str(tmp_path)
        assert config.env is not None
        env_dict = cast("dict[str, str]", config.env)
        assert env_dict["MALA_AGENT_ID"] == "agent-x"
        assert env_dict["MALA_REPO_NAMESPACE"] == str(tmp_path)
        assert client is not None
    assert fake_codex.close_calls == 1
    assert fake_codex.exit_calls == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_aenter_honors_codex_binary_env_var(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``CODEX_BINARY`` env var is plumbed through ``AppServerConfig.codex_bin``."""
    from src.infra.clients.codex_client import CodexClient

    fake_codex = _install_fake_sdk(monkeypatch)
    monkeypatch.setenv("CODEX_BINARY", "/opt/codex/bin/codex")
    runtime = _build_runtime(tmp_path)

    async with CodexClient(runtime):
        config = fake_codex.config
        assert config is not None
        assert config.codex_bin == "/opt/codex/bin/codex"


# ---------------------------------------------------------------------------
# query() — thread_start / thread_resume / reuse
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_query_starts_thread_with_runtime_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``query(prompt)`` calls ``thread_start`` with runtime params + a turn.

    Regression coverage for Phase C reviewer findings: ``cwd`` from
    :class:`CodexRuntime` must reach ``thread_start``; ``effort`` is
    per-turn (lives on :meth:`AsyncThread.turn`, not ``thread_start``);
    ``mcp_servers`` must NOT be passed to ``thread_start`` because the
    real SDK signature does not accept it (``codex_app_server/api.py:336``)
    — MCP servers ship through the bundled Codex plugin's
    ``.mcp.json`` (Phase G3).
    """
    from src.infra.clients.codex_client import CodexClient

    fake_codex = _install_fake_sdk(monkeypatch)
    runtime = _build_runtime(tmp_path)
    fake_codex.next_thread = _FakeThread(id="thr_started")

    async with CodexClient(runtime) as client:
        await client.query("hello world")
        assert len(fake_codex.threads_started) == 1
        kwargs = fake_codex.threads_started[0]
        assert kwargs["model"] == "gpt-5.5"
        assert kwargs["sandbox"] == "danger-full-access"
        assert kwargs["approval_policy"] == "never"
        assert kwargs["cwd"] == str(tmp_path)
        # SDK shape contract — these MUST NOT be on thread_start.
        assert "effort" not in kwargs
        assert "mcp_servers" not in kwargs
        assert fake_codex.threads_resumed == []
        assert fake_codex.next_thread.turn_handles  # turn was issued
        assert fake_codex.next_thread.turn_inputs[0].prompt == "hello world"
        # ``effort`` is forwarded per-turn instead.
        assert fake_codex.next_thread.turn_kwargs[0].get("effort") == "medium"
        assert client.session_id == "thr_started"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_query_reuses_existing_thread_on_subsequent_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second ``query`` only issues a new turn — does not restart the thread."""
    from src.infra.clients.codex_client import CodexClient

    fake_codex = _install_fake_sdk(monkeypatch)
    runtime = _build_runtime(tmp_path)
    fake_codex.next_thread = _FakeThread(id="thr_started")

    async with CodexClient(runtime) as client:
        await client.query("first")
        await client.query("second", session_id="thr_ignored")
        assert len(fake_codex.threads_started) == 1
        assert fake_codex.threads_resumed == []
        assert len(fake_codex.next_thread.turn_inputs) == 2
        assert fake_codex.next_thread.turn_inputs[0].prompt == "first"
        assert fake_codex.next_thread.turn_inputs[1].prompt == "second"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_query_resumes_when_runtime_carries_resume_thread_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC #8 (partial): ``runtime.resume_thread_id`` → ``thread_resume`` on next ``query``."""
    from src.infra.clients.codex_client import CodexClient

    fake_codex = _install_fake_sdk(monkeypatch)
    runtime = _build_runtime(tmp_path, resume_thread_id="thr_resume_me")

    async with CodexClient(runtime) as client:
        await client.query("hi again")
        assert fake_codex.threads_resumed == ["thr_resume_me"]
        assert fake_codex.threads_started == []
        assert client.session_id == "thr_resume_me"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_with_resume_overrides_runtime_resume_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``with_resume(thread_id)`` overrides any earlier resume token."""
    from src.infra.clients.codex_client import CodexClient

    fake_codex = _install_fake_sdk(monkeypatch)
    runtime = _build_runtime(tmp_path)

    async with CodexClient(runtime) as client:
        client.with_resume("thr_late_bound")
        await client.query("hi")
        assert fake_codex.threads_resumed == ["thr_late_bound"]
        assert fake_codex.threads_started == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_query_before_aenter_raises(tmp_path: Path) -> None:
    from src.infra.clients.codex_client import CodexClient

    runtime = _build_runtime(tmp_path)
    client = CodexClient(runtime)
    with pytest.raises(RuntimeError, match="before __aenter__"):
        await client.query("hi")


# ---------------------------------------------------------------------------
# receive_response — emits AgentEvents from TurnHandle.stream() (Phase C AC-1)
# ---------------------------------------------------------------------------


@dataclass
class _FakeNotification:
    """Stand-in for ``codex_app_server.models.Notification``.

    The real SDK ``Notification`` dataclass exposes ``method`` (string,
    e.g. ``"item/agentMessage/delta"``) and ``payload`` (typed Pydantic
    model, e.g. :class:`AgentMessageDeltaNotification`). The adapter in
    :mod:`src.infra.clients.codex_client` only reads attributes by name,
    so this minimal shape is sufficient.
    """

    method: str
    payload: object


@pytest.mark.unit
@pytest.mark.asyncio
async def test_receive_response_maps_notifications_to_agent_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase C AC-1: ``receive_response`` emits :class:`AgentEvent` values.

    Mirrors the plan's mapping table (``plans/2026-05-07-codex-provider-plan.md``
    L746-L762) for the Phase C surface: ``item/agentMessage/delta`` →
    :class:`AgentTextEvent`; ``turn/completed`` → :class:`AgentResultEvent`
    with ``is_error`` derived from ``TurnStatus``. Phase D (T011) extends
    the dispatch with the item-level mappings.
    """
    from src.infra.clients.codex_client import CodexClient

    fake_codex = _install_fake_sdk(monkeypatch)
    runtime = _build_runtime(tmp_path)
    thread = _FakeThread(id="thr_stream")
    fake_codex.next_thread = thread

    notifications: list[object] = [
        _FakeNotification(
            method="item/agentMessage/delta",
            payload=SimpleNamespace(
                delta="hi",
                item_id="item_1",
                thread_id="thr_stream",
                turn_id="turn_1",
            ),
        ),
        _FakeNotification(
            method="turn/completed",
            payload=SimpleNamespace(
                thread_id="thr_stream",
                turn=SimpleNamespace(
                    id="turn_1",
                    status=SimpleNamespace(value="completed"),
                    error=None,
                ),
            ),
        ),
    ]

    async with CodexClient(runtime) as client:
        await client.query("stream test")
        thread.turn_handles[0].notifications.extend(notifications)
        received: list[AgentEventValue] = []
        async for event in client.receive_response():
            received.append(event)

    assert len(received) == 2
    text_event = received[0]
    assert isinstance(text_event, AgentTextEvent)
    assert text_event.text == "hi"
    assert text_event.kind == "text"

    result_event = received[1]
    assert isinstance(result_event, AgentResultEvent)
    assert result_event.session_id == "thr_stream"
    assert result_event.is_error is False
    assert result_event.subtype == "completed"
    assert result_event.kind == "result"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_receive_response_maps_failed_turn_to_error_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``turn/completed`` with non-``completed`` status surfaces as an error.

    Without this, a Codex turn that ``TurnStatus.failed`` would close the
    pipeline cleanly and the orchestrator would treat the run as a
    success — masking provider failures (plan ``L761`` D6).
    """
    from src.infra.clients.codex_client import CodexClient

    fake_codex = _install_fake_sdk(monkeypatch)
    runtime = _build_runtime(tmp_path)
    thread = _FakeThread(id="thr_failed")
    fake_codex.next_thread = thread

    failed_notification = _FakeNotification(
        method="turn/completed",
        payload=SimpleNamespace(
            thread_id="thr_failed",
            turn=SimpleNamespace(
                id="turn_1",
                status=SimpleNamespace(value="failed"),
                error=SimpleNamespace(message="boom"),
            ),
        ),
    )

    async with CodexClient(runtime) as client:
        await client.query("doomed turn")
        thread.turn_handles[0].notifications.append(failed_notification)
        received: list[AgentEventValue] = []
        async for event in client.receive_response():
            received.append(event)

    assert len(received) == 1
    result_event = received[0]
    assert isinstance(result_event, AgentResultEvent)
    assert result_event.is_error is True
    assert result_event.subtype == "completed"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_receive_response_maps_error_notification_to_result_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ErrorNotification`` becomes ``AgentResultEvent(is_error=True)``.

    Plan ``L762`` D7: classify the error as a terminal failure so the
    lifecycle does not stall on the missing ``turn/completed``.
    """
    from src.infra.clients.codex_client import CodexClient

    fake_codex = _install_fake_sdk(monkeypatch)
    runtime = _build_runtime(tmp_path)
    thread = _FakeThread(id="thr_error")
    fake_codex.next_thread = thread

    error_payload = SimpleNamespace(message="429 rate limited")
    error_notification = _FakeNotification(
        method="error",
        payload=SimpleNamespace(
            error=error_payload,
            thread_id="thr_error",
            turn_id="turn_1",
            will_retry=False,
        ),
    )

    async with CodexClient(runtime) as client:
        await client.query("rate-limited turn")
        thread.turn_handles[0].notifications.append(error_notification)
        received: list[AgentEventValue] = []
        async for event in client.receive_response():
            received.append(event)

    assert len(received) == 1
    result_event = received[0]
    assert isinstance(result_event, AgentResultEvent)
    assert result_event.is_error is True
    assert result_event.subtype == "error"
    assert result_event.session_id == "thr_error"
    assert result_event.result is error_payload


@pytest.mark.unit
@pytest.mark.asyncio
async def test_receive_response_drops_unmapped_phase_d_notifications(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Item-level + reasoning notifications are dropped until Phase D lands.

    The pipeline must not crash on notification kinds the Phase C
    surface doesn't yet map (``item/started``, ``item/completed``,
    ``item/reasoning/delta``, ...). Phase D (T011) extends the dispatch.
    """
    from src.infra.clients.codex_client import CodexClient

    fake_codex = _install_fake_sdk(monkeypatch)
    runtime = _build_runtime(tmp_path)
    thread = _FakeThread(id="thr_phase_d")
    fake_codex.next_thread = thread

    notifications: list[object] = [
        _FakeNotification(
            method="item/started",
            payload=SimpleNamespace(
                item=SimpleNamespace(id="item_1", type="commandExecution"),
                thread_id="thr_phase_d",
                turn_id="turn_1",
            ),
        ),
        _FakeNotification(
            method="item/reasoning/text/delta",
            payload=SimpleNamespace(delta="thinking..."),
        ),
        _FakeNotification(
            method="item/agentMessage/delta",
            payload=SimpleNamespace(delta="hello"),
        ),
    ]

    async with CodexClient(runtime) as client:
        await client.query("phase d preview")
        thread.turn_handles[0].notifications.extend(notifications)
        received: list[AgentEventValue] = []
        async for event in client.receive_response():
            received.append(event)

    # Only the agentMessage/delta survives the Phase C dispatch.
    assert len(received) == 1
    assert isinstance(received[0], AgentTextEvent)
    assert received[0].text == "hello"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_receive_response_before_query_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.infra.clients.codex_client import CodexClient

    _install_fake_sdk(monkeypatch)
    runtime = _build_runtime(tmp_path)
    async with CodexClient(runtime) as client:
        with pytest.raises(RuntimeError, match="before query"):
            async for _ in client.receive_response():
                pass


# ---------------------------------------------------------------------------
# disconnect — interrupts active turn + closes AsyncCodex exactly once
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_disconnect_interrupts_active_turn_and_closes_async_codex_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC: SIGINT mid-turn calls interrupt + close exactly once."""
    from src.infra.clients.codex_client import CodexClient

    fake_codex = _install_fake_sdk(monkeypatch)
    runtime = _build_runtime(tmp_path)
    thread = _FakeThread(id="thr_x")
    fake_codex.next_thread = thread

    client = CodexClient(runtime)
    await client.__aenter__()
    await client.query("hi")
    turn = thread.turn_handles[0]

    # Disconnect twice: must be idempotent. interrupt() and close() should
    # each fire exactly once. The plan-backed AC-3 contract names
    # AsyncCodex.close() specifically, so we assert close() is called and
    # __aexit__ is not the teardown primitive used here.
    await client.disconnect()
    await client.disconnect()

    assert turn.interrupt_calls == 1
    assert fake_codex.close_calls == 1
    assert fake_codex.exit_calls == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_disconnect_swallows_sdk_close_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Errors from the SDK teardown must not propagate from ``disconnect``.

    The teardown path is opportunistic — it must not mask the original
    exception that drove the disconnect (e.g., ``CancelledError``).
    """
    from src.infra.clients.codex_client import CodexClient

    fake_codex = _install_fake_sdk(monkeypatch)

    async def boom_close(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated SDK close error")

    fake_codex.close = boom_close  # type: ignore[method-assign,assignment]  # ty:ignore[invalid-assignment]
    runtime = _build_runtime(tmp_path)

    client = CodexClient(runtime)
    await client.__aenter__()
    await client.query("hi")
    # Should not raise.
    await client.disconnect()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_sigint_mid_turn_cancellation_invokes_interrupt_and_close_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC-3 regression: SIGINT mid-turn → interrupt + close exactly once.

    Simulates the production cancellation chain (lifecycle controller's
    abort mode cancels the active task → ``async with CodexClient``
    unwinds via ``CancelledError`` → ``__aexit__`` → ``disconnect()``).
    The ``CancelledError`` raised inside the ``async with`` block models
    the runtime adapter's cancellation point; the CodexClient's
    ``__aexit__`` must reach the SDK's ``close()`` primitive named in
    plan L732, not just unwind the codex async-context.
    """
    import asyncio

    from src.infra.clients.codex_client import CodexClient

    fake_codex = _install_fake_sdk(monkeypatch)
    runtime = _build_runtime(tmp_path)
    thread = _FakeThread(id="thr_sigint")
    fake_codex.next_thread = thread

    with pytest.raises(asyncio.CancelledError):
        async with CodexClient(runtime) as client:
            await client.query("mid-turn prompt")
            # Cancellation is delivered while the turn is active and the
            # adapter is still inside the ``async with`` block.
            raise asyncio.CancelledError

    turn = thread.turn_handles[0]
    assert turn.interrupt_calls == 1
    assert fake_codex.close_calls == 1
    # __aexit__ on the SDK must not also fire — close() is the named
    # AC-3 teardown primitive.
    assert fake_codex.exit_calls == 0


# ---------------------------------------------------------------------------
# session_id property reflects thread id
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_session_id_is_none_until_query_starts_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.infra.clients.codex_client import CodexClient

    _install_fake_sdk(monkeypatch)
    runtime = _build_runtime(tmp_path)
    async with CodexClient(runtime) as client:
        assert client.session_id is None
        await client.query("hi")
        assert client.session_id == "thr_fake_start"

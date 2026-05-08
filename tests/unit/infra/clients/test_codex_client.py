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
  * ``receive_response`` yields raw notifications from
    ``TurnHandle.stream()`` verbatim (Phase D translates them).
  * ``disconnect`` calls ``TurnHandle.interrupt`` and the SDK
    ``__aexit__`` exactly once across repeat calls.
  * ``session_id`` reflects the started thread's id.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import ModuleType
    from typing import Self

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

    async def turn(self, text_input: _FakeTextInput) -> _FakeTurnHandle:
        handle = _FakeTurnHandle()
        self.turn_handles.append(handle)
        self.turn_inputs.append(text_input)
        return handle


@dataclass
class _FakeAsyncCodex:
    """Stand-in for ``codex_app_server.AsyncCodex``.

    Tracks every kwarg passed to the constructor, every ``thread_start``
    /  ``thread_resume`` invocation, and the close lifecycle so tests can
    assert on observable behavior (the issue's plan calls for the
    cancellation path to interrupt the active turn and close
    ``AsyncCodex`` exactly once).
    """

    init_kwargs: dict[str, object]
    threads_started: list[dict[str, object]] = field(default_factory=list)
    threads_resumed: list[str] = field(default_factory=list)
    enter_calls: int = 0
    exit_calls: int = 0
    next_thread: _FakeThread | None = None
    resumed_thread: _FakeThread | None = None
    accept_env_kwarg: bool = True

    async def __aenter__(self) -> Self:
        self.enter_calls += 1
        return self

    async def __aexit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> None:
        del exc_type, exc_val, exc_tb
        self.exit_calls += 1

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


def _install_fake_sdk(
    monkeypatch: pytest.MonkeyPatch,
    *,
    accept_env_kwarg: bool = True,
) -> _FakeAsyncCodex:
    """Insert a fake ``codex_app_server`` module into ``sys.modules``.

    Returns the singleton :class:`_FakeAsyncCodex` the fake module's
    ``AsyncCodex`` factory will hand out, so tests can assert on the
    captured kwargs / thread-start / interrupt / close calls.

    The ``accept_env_kwarg`` toggle exercises the
    ``CodexClient.__aenter__`` fallback that retries without ``env=``
    when the SDK rejects the kwarg (Phase B/C spike contingency).
    """
    import types

    fake_codex = _FakeAsyncCodex(init_kwargs={}, accept_env_kwarg=accept_env_kwarg)

    def async_codex_factory(**kwargs: object) -> _FakeAsyncCodex:
        if "env" in kwargs and not accept_env_kwarg:
            raise TypeError("AsyncCodex() got unexpected keyword argument 'env'")
        fake_codex.init_kwargs = dict(kwargs)
        return fake_codex

    fake_module: ModuleType = types.ModuleType("codex_app_server")
    fake_module.AsyncCodex = async_codex_factory  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]
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
async def test_aenter_constructs_async_codex_with_runtime_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``__aenter__`` lazy-constructs ``AsyncCodex`` with ``env=runtime.env``."""
    from src.infra.clients.codex_client import CodexClient

    fake_codex = _install_fake_sdk(monkeypatch)
    runtime = _build_runtime(tmp_path)

    async with CodexClient(runtime) as client:
        assert fake_codex.enter_calls == 1
        # ``env`` kwarg passes the runtime's per-process env dict so the
        # SDK can plumb ``MALA_*`` to its subprocess if it supports it.
        assert "env" in fake_codex.init_kwargs
        env_obj = fake_codex.init_kwargs["env"]
        assert isinstance(env_obj, dict)
        env_dict = cast("dict[str, str]", env_obj)
        assert env_dict["MALA_AGENT_ID"] == "agent-x"
        assert client is not None
    assert fake_codex.exit_calls == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_aenter_falls_back_when_sdk_rejects_env_kwarg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the SDK does not accept ``env=``, fall back to ``AsyncCodex()``.

    Phase B/C spike contingency: the per-session state-file fallback
    (Phase E) covers env propagation if the SDK rejects ``env=``.
    """
    from src.infra.clients.codex_client import CodexClient

    fake_codex = _install_fake_sdk(monkeypatch, accept_env_kwarg=False)
    runtime = _build_runtime(tmp_path)

    async with CodexClient(runtime):
        assert fake_codex.enter_calls == 1
        assert "env" not in fake_codex.init_kwargs


# ---------------------------------------------------------------------------
# query() — thread_start / thread_resume / reuse
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_query_starts_thread_with_runtime_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``query(prompt)`` calls ``thread_start`` with runtime params + a turn.

    Regression coverage for the Phase C reviewer findings: ``effort`` and
    ``cwd`` from :class:`CodexRuntime` must reach ``thread_start`` so the
    user-configured Codex effort (``MalaConfig.coder_options.codex.effort``)
    is honored and the thread runs against the orchestrated repo path
    rather than wherever the SDK app-server happened to be launched.
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
        assert kwargs["effort"] == "medium"
        assert kwargs["sandbox"] == "danger-full-access"
        assert kwargs["approval_policy"] == "never"
        assert kwargs["mcp_servers"] == {
            "mala-locking": {"command": "mala-codex-mcp-locking"}
        }
        assert kwargs["cwd"] == str(tmp_path)
        assert fake_codex.threads_resumed == []
        assert fake_codex.next_thread.turn_handles  # turn was issued
        assert fake_codex.next_thread.turn_inputs[0].prompt == "hello world"
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
# receive_response — yields raw notifications from TurnHandle.stream()
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_receive_response_yields_turn_stream_notifications(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase C: notifications pass through verbatim. Phase D wraps them."""
    from src.infra.clients.codex_client import CodexClient

    fake_codex = _install_fake_sdk(monkeypatch)
    runtime = _build_runtime(tmp_path)
    thread = _FakeThread(id="thr_stream")
    fake_codex.next_thread = thread

    sentinel_notifications: list[object] = [
        {"type": "agent_message_delta", "delta": "hi"},
        {"type": "turn_completed", "status": "completed"},
    ]

    async with CodexClient(runtime) as client:
        await client.query("stream test")
        # Inject the notifications into the turn handle the fake created.
        thread.turn_handles[0].notifications.extend(sentinel_notifications)
        received: list[object] = []
        async for note in client.receive_response():
            received.append(note)
        assert received == sentinel_notifications


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

    # Disconnect twice: must be idempotent. interrupt() and exit() should
    # each fire exactly once.
    await client.disconnect()
    await client.disconnect()

    assert turn.interrupt_calls == 1
    assert fake_codex.exit_calls == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_disconnect_swallows_sdk_exit_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Errors from the SDK teardown must not propagate from ``disconnect``.

    The teardown path is opportunistic — it must not mask the original
    exception that drove the disconnect (e.g., ``CancelledError``).
    """
    from src.infra.clients.codex_client import CodexClient

    fake_codex = _install_fake_sdk(monkeypatch)

    async def boom_exit(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated SDK teardown error")

    fake_codex.__aexit__ = boom_exit  # type: ignore[method-assign,assignment]  # ty:ignore[invalid-assignment]
    runtime = _build_runtime(tmp_path)

    client = CodexClient(runtime)
    await client.__aenter__()
    await client.query("hi")
    # Should not raise.
    await client.disconnect()


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

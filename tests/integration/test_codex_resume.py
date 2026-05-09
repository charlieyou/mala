"""Integration test: Codex resume across two turns (T017 H1).

Phase H AC #8 (resume + idle/review retries work for Codex) requires the
provider → runtime → client wiring to honor a resume token across the
``thread_resume`` boundary. This test issues two turns through the full
:class:`CodexAgentProvider` stack:

  1. The first turn starts a fresh thread via
     ``AsyncCodex.thread_start`` and surfaces an :class:`AgentTextEvent`
     plus the terminal :class:`AgentResultEvent`.
  2. ``client_factory.with_resume(runtime, thread_id)`` then produces a
     sibling :class:`CodexRuntime` carrying the first turn's thread id.
  3. The second turn is issued through a fresh :class:`CodexClient`
     bound to the resumed runtime; the SDK call MUST be
     ``thread_resume(thr_X)`` rather than a fresh ``thread_start``.
  4. The second turn's emitted text confirms the SDK-resumed thread is
     the path the events came from — i.e. cross-resume continuity is
     preserved at the provider boundary.

The fake ``codex_app_server`` substitutes a minimal
``AsyncCodex`` / ``AsyncThread`` / ``AsyncTurnHandle`` so the test does
not require the real Codex SDK on PATH. The shape mirrors
``tests/integration/test_codex_provider.py`` so reviewers can trace
behavior between the two suites.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from src.core.protocols.agent_event import (
    AgentResultEvent,
    AgentTextEvent,
)
from src.infra.clients.codex_provider import CodexAgentProvider
from src.infra.clients.codex_runtime import CodexRuntime

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from pathlib import Path
    from types import ModuleType
    from typing import Self

    from src.core.protocols.agent_event import AgentEventValue


@dataclass
class _FakeTextInput:
    prompt: str


@dataclass
class _FakeTurnHandle:
    notifications: list[object] = field(default_factory=list)

    async def stream(self) -> AsyncIterator[object]:
        for note in list(self.notifications):
            yield note

    async def interrupt(self) -> None:
        return None


@dataclass
class _FakeThread:
    """Stand-in for ``codex_app_server.AsyncThread`` with scripted turns."""

    id: str
    scripted_turns: list[list[object]] = field(default_factory=list)
    turn_inputs: list[_FakeTextInput] = field(default_factory=list)

    async def turn(
        self, text_input: _FakeTextInput, **kwargs: object
    ) -> _FakeTurnHandle:
        del kwargs
        self.turn_inputs.append(text_input)
        notes = self.scripted_turns.pop(0) if self.scripted_turns else []
        return _FakeTurnHandle(notifications=notes)


@dataclass
class _FakeAppServerConfig:
    codex_bin: str | None = None
    cwd: str | None = None
    env: dict[str, str] | None = None


@dataclass
class _FakeAsyncCodex:
    """Captures ``thread_start`` / ``thread_resume`` for cross-turn assertions."""

    config: _FakeAppServerConfig | None = None
    started_kwargs: list[dict[str, object]] = field(default_factory=list)
    resumed_ids: list[str] = field(default_factory=list)
    started_thread: _FakeThread | None = None
    resumed_thread: _FakeThread | None = None
    close_calls: int = 0

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> None:
        del exc_type, exc_val, exc_tb

    async def close(self) -> None:
        self.close_calls += 1

    async def thread_start(self, **kwargs: object) -> _FakeThread:
        self.started_kwargs.append(dict(kwargs))
        if self.started_thread is None:
            self.started_thread = _FakeThread(id="thr_started_default")
        return self.started_thread

    async def thread_resume(self, thread_id: str) -> _FakeThread:
        self.resumed_ids.append(thread_id)
        if self.resumed_thread is None:
            self.resumed_thread = _FakeThread(id=thread_id)
        return self.resumed_thread


def _install_fake_codex_app_server(
    monkeypatch: pytest.MonkeyPatch,
    *,
    started_thread: _FakeThread,
    resumed_thread: _FakeThread,
) -> _FakeAsyncCodex:
    """Insert a fake ``codex_app_server`` returning the supplied threads."""
    fake_codex = _FakeAsyncCodex(
        started_thread=started_thread,
        resumed_thread=resumed_thread,
    )

    def async_codex_factory(
        config: _FakeAppServerConfig | None = None,
    ) -> _FakeAsyncCodex:
        fake_codex.config = config
        return fake_codex

    fake_module: ModuleType = types.ModuleType("codex_app_server")
    fake_module.AsyncCodex = async_codex_factory  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]
    fake_module.AppServerConfig = _FakeAppServerConfig  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]
    fake_module.TextInput = _FakeTextInput  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "codex_app_server", fake_module)
    return fake_codex


def _empty_factory() -> Callable[..., dict[str, object]]:
    def factory(
        agent_id: str, repo_path: Path, emit_lock_event: object
    ) -> dict[str, object]:
        del agent_id, repo_path, emit_lock_event
        return {}

    return factory


@dataclass
class _FakeNotification:
    method: str
    payload: object


def _text_then_completed(*, thread_id: str, turn_id: str, text: str) -> list[object]:
    return [
        _FakeNotification(
            method="item/agentMessage/delta",
            payload=SimpleNamespace(
                delta=text,
                item_id=f"item_{turn_id}",
                thread_id=thread_id,
                turn_id=turn_id,
            ),
        ),
        _FakeNotification(
            method="turn/completed",
            payload=SimpleNamespace(
                thread_id=thread_id,
                turn=SimpleNamespace(
                    id=turn_id,
                    status=SimpleNamespace(value="completed"),
                    error=None,
                ),
            ),
        ),
    ]


@pytest.fixture(autouse=True)
def _redirect_codex_tee_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tee'd JSONL inside the test's tmp_path (parity with unit suite)."""
    monkeypatch.setenv("MALA_CODEX_SESSIONS_DIR", str(tmp_path / "codex-tee"))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resume_threads_first_turns_id_into_thread_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two turns: turn-1 uses ``thread_start``, turn-2 uses ``thread_resume``.

    ``CodexAgentProvider.client_factory.with_resume`` must produce a
    sibling runtime whose next ``CodexClient`` resumes the first turn's
    thread id rather than starting a new thread. Without this, AC #8 (resume
    + idle/review retries) regresses for Codex: an idle retry would route
    a fresh ``thread_start`` and the orchestrator would lose the prior
    turn's context.
    """
    started_thread = _FakeThread(
        id="thr_codex_resume",
        scripted_turns=[
            _text_then_completed(
                thread_id="thr_codex_resume",
                turn_id="turn_one",
                text="turn-one-says-hi",
            )
        ],
    )
    resumed_thread = _FakeThread(
        id="thr_codex_resume",
        scripted_turns=[
            _text_then_completed(
                thread_id="thr_codex_resume",
                turn_id="turn_two",
                text="turn-two-after-resume",
            )
        ],
    )
    fake_codex = _install_fake_codex_app_server(
        monkeypatch,
        started_thread=started_thread,
        resumed_thread=resumed_thread,
    )

    provider = CodexAgentProvider()
    runtime = provider.runtime_builder(
        tmp_path, "agent-resume", mcp_server_factory=_empty_factory()
    ).build()
    assert isinstance(runtime, CodexRuntime)

    # --- Turn 1: fresh thread_start ---
    first_events: list[AgentEventValue] = []
    client_one = provider.client_factory.create(runtime)
    async with client_one:
        await client_one.query("first prompt")
        async for event in client_one.receive_response():
            first_events.append(cast("AgentEventValue", event))
        first_thread_id = client_one.session_id  # ty:ignore[unresolved-attribute]

    assert fake_codex.started_kwargs and fake_codex.resumed_ids == []
    assert first_thread_id == "thr_codex_resume"
    text_one = first_events[0]
    assert isinstance(text_one, AgentTextEvent)
    assert text_one.text == "turn-one-says-hi"
    result_one = first_events[-1]
    assert isinstance(result_one, AgentResultEvent)
    assert result_one.session_id == "thr_codex_resume"
    assert result_one.is_error is False

    # --- Turn 2: provider routes the prior thread id into thread_resume ---
    resumed_runtime = provider.client_factory.with_resume(runtime, first_thread_id)
    assert isinstance(resumed_runtime, CodexRuntime)
    assert resumed_runtime.resume_thread_id == "thr_codex_resume"

    second_events: list[AgentEventValue] = []
    client_two = provider.client_factory.create(resumed_runtime)
    async with client_two:
        await client_two.query("second prompt")
        async for event in client_two.receive_response():
            second_events.append(cast("AgentEventValue", event))
        assert client_two.session_id == "thr_codex_resume"  # ty:ignore[unresolved-attribute]

    # SDK contract: turn-2 must hit thread_resume, NOT a second thread_start.
    assert fake_codex.resumed_ids == ["thr_codex_resume"]
    assert len(fake_codex.started_kwargs) == 1
    # Continuity proof: events on turn-2 came from the resumed thread, not a fresh one.
    text_two = second_events[0]
    assert isinstance(text_two, AgentTextEvent)
    assert text_two.text == "turn-two-after-resume"
    assert resumed_thread.turn_inputs[0].prompt == "second prompt"
    assert started_thread.turn_inputs[0].prompt == "first prompt"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_with_resume_none_returns_runtime_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``with_resume(runtime, None)`` is a no-op (decision #10 boundary).

    The Phase H wiring lets the orchestrator's
    :class:`IdleTimeoutRetryPolicy` thread the optional resume id from
    state.pending_session_id straight into ``with_resume`` without
    branching on coder; when no resume id is known, the call must hand
    back the original runtime so the next ``CodexClient`` falls back to
    ``thread_start`` instead of receiving a sentinel that resumes
    ``"None"``. Regression target: a typo here would silently break
    fresh-session continuation when the policy's no-tool fallback path
    is taken.
    """
    _install_fake_codex_app_server(
        monkeypatch,
        started_thread=_FakeThread(id="thr_anything"),
        resumed_thread=_FakeThread(id="thr_unused"),
    )
    provider = CodexAgentProvider()
    runtime = provider.runtime_builder(
        tmp_path, "agent-noresume", mcp_server_factory=_empty_factory()
    ).build()
    assert isinstance(runtime, CodexRuntime)
    assert runtime.resume_thread_id is None

    same = provider.client_factory.with_resume(runtime, None)
    assert same is runtime

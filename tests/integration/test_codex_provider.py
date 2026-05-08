"""Integration test: provider → runtime → client → fake notification stream (T010).

Smokes the full Phase C wiring path with a fake ``codex_app_server``
module installed via ``sys.modules``. Asserts that:

  * ``CodexAgentProvider.runtime_builder(...).build()`` returns a
    :class:`CodexRuntime` carrying the resolved options.
  * ``CodexAgentProvider.client_factory.create(runtime)`` returns a
    :class:`CodexClient` bound to that runtime.
  * ``async with CodexClient(runtime)`` enters the SDK context and
    ``query()`` issues a fake ``thread_start`` + turn whose
    notifications stream out unchanged through ``receive_response``
    (Phase D / T011 wraps the raw stream into ``AgentEvent``s; this
    test verifies the lifecycle wiring without that wrapping).

The fake SDK is intentionally minimal — the unit suite in
``tests/unit/infra/clients/test_codex_client.py`` covers fan-out
combinations; this test demonstrates that the full provider → runtime
→ client wiring composes end-to-end.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from src.infra.clients.codex_provider import CodexAgentProvider
from src.infra.clients.codex_runtime import CodexRuntime

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from pathlib import Path
    from types import ModuleType
    from typing import Self


@dataclass
class _FakeTextInput:
    prompt: str


@dataclass
class _FakeTurnHandle:
    notifications: list[object] = field(default_factory=list)
    interrupt_calls: int = 0

    async def stream(self) -> AsyncIterator[object]:
        for note in list(self.notifications):
            yield note

    async def interrupt(self) -> None:
        self.interrupt_calls += 1


@dataclass
class _FakeThread:
    id: str
    seeded_notifications: list[object] = field(default_factory=list)

    async def turn(self, text_input: _FakeTextInput) -> _FakeTurnHandle:
        del text_input
        return _FakeTurnHandle(notifications=list(self.seeded_notifications))


@dataclass
class _FakeAsyncCodex:
    init_kwargs: dict[str, object]
    started_kwargs: list[dict[str, object]] = field(default_factory=list)
    resumed_ids: list[str] = field(default_factory=list)
    fixed_thread: _FakeThread | None = None
    exit_calls: int = 0

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> None:
        del exc_type, exc_val, exc_tb
        self.exit_calls += 1

    async def thread_start(self, **kwargs: object) -> _FakeThread:
        self.started_kwargs.append(dict(kwargs))
        if self.fixed_thread is None:
            self.fixed_thread = _FakeThread(id="thr_int_fake")
        return self.fixed_thread

    async def thread_resume(self, thread_id: str) -> _FakeThread:
        self.resumed_ids.append(thread_id)
        return _FakeThread(id=thread_id)


def _install_fake_codex_app_server(
    monkeypatch: pytest.MonkeyPatch,
    *,
    seeded_notifications: list[object],
) -> _FakeAsyncCodex:
    fake_codex = _FakeAsyncCodex(init_kwargs={})
    fake_codex.fixed_thread = _FakeThread(
        id="thr_integration", seeded_notifications=seeded_notifications
    )

    def async_codex_factory(**kwargs: object) -> _FakeAsyncCodex:
        fake_codex.init_kwargs = dict(kwargs)
        return fake_codex

    fake_module: ModuleType = types.ModuleType("codex_app_server")
    fake_module.AsyncCodex = async_codex_factory  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]
    fake_module.TextInput = _FakeTextInput  # type: ignore[attr-defined]  # ty:ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "codex_app_server", fake_module)
    return fake_codex


def _empty_factory() -> Callable[..., dict[str, object]]:
    def factory(
        agent_id: str, repo_path: Path, emit_lock_event: object
    ) -> dict[str, object]:
        del agent_id, repo_path, emit_lock_event
        return {"mala-locking": {"command": "mala-codex-mcp-locking"}}

    return factory


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_runtime_client_end_to_end_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provider → runtime → client → fake notification stream."""
    seeded: list[object] = [
        {"type": "agent_message_delta", "delta": "hi"},
        {"type": "turn_completed", "status": "completed"},
    ]
    fake_codex = _install_fake_codex_app_server(
        monkeypatch, seeded_notifications=seeded
    )

    provider = CodexAgentProvider(
        model="gpt-5.5-foo",
        effort="medium",
        approval_policy="never",
        sandbox="danger-full-access",
    )

    runtime = provider.runtime_builder(
        tmp_path, "agent-x", mcp_server_factory=_empty_factory()
    ).build()
    assert isinstance(runtime, CodexRuntime)
    assert runtime.model == "gpt-5.5-foo"
    assert runtime.mcp_servers == {
        "mala-locking": {"command": "mala-codex-mcp-locking"}
    }

    client = provider.client_factory.create(runtime)
    received: list[object] = []
    async with client:
        await client.query("smoke prompt")
        async for note in client.receive_response():
            received.append(note)
        assert client.session_id == "thr_integration"  # ty:ignore[unresolved-attribute]

    # Lifecycle: thread_start fired with runtime params, no resume,
    # AsyncCodex closed exactly once on context exit.
    assert len(fake_codex.started_kwargs) == 1
    started = fake_codex.started_kwargs[0]
    assert started["model"] == "gpt-5.5-foo"
    assert started["sandbox"] == "danger-full-access"
    assert started["approval_policy"] == "never"
    assert fake_codex.resumed_ids == []
    assert received == seeded
    assert fake_codex.exit_calls == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_with_resume_runtime_resumes_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``client_factory.with_resume`` produces a runtime that resumes on next query()."""
    fake_codex = _install_fake_codex_app_server(monkeypatch, seeded_notifications=[])

    provider = CodexAgentProvider()
    runtime = provider.runtime_builder(
        tmp_path, "agent-x", mcp_server_factory=_empty_factory()
    ).build()
    resumed_runtime = provider.client_factory.with_resume(runtime, "thr_resume_me")
    assert isinstance(resumed_runtime, CodexRuntime)

    client = provider.client_factory.create(resumed_runtime)
    async with client:
        await client.query("resume prompt")
        async for _ in client.receive_response():
            pass
        assert client.session_id == "thr_resume_me"  # ty:ignore[unresolved-attribute]

    assert fake_codex.resumed_ids == ["thr_resume_me"]
    assert fake_codex.started_kwargs == []

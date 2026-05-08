"""Integration test: provider → runtime → client → AgentEvent stream (T010).

Smokes the full Phase C wiring path with a fake ``codex_app_server``
module installed via ``sys.modules``. Asserts that:

  * ``CodexAgentProvider.runtime_builder(...).build()`` returns a
    :class:`CodexRuntime` carrying the resolved options.
  * ``CodexAgentProvider.client_factory.create(runtime)`` returns a
    :class:`CodexClient` bound to that runtime.
  * ``async with CodexClient(runtime)`` enters the SDK context and
    ``query()`` issues a fake ``thread_start`` + turn whose
    notifications are mapped to :class:`AgentEvent` values through
    ``receive_response`` (Phase C AC-1, plan C3); Phase D (T011)
    extends the adapter with item-level mappings.

The fake SDK is intentionally minimal — the unit suite in
``tests/unit/infra/clients/test_codex_client.py`` covers fan-out
combinations; this test demonstrates that the full provider → runtime
→ client wiring composes end-to-end.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from src.core.protocols.agent_event import AgentResultEvent, AgentTextEvent
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
    turn_kwargs: list[dict[str, object]] = field(default_factory=list)

    async def turn(
        self, text_input: _FakeTextInput, **kwargs: object
    ) -> _FakeTurnHandle:
        del text_input
        self.turn_kwargs.append(dict(kwargs))
        return _FakeTurnHandle(notifications=list(self.seeded_notifications))


@dataclass
class _FakeAppServerConfig:
    codex_bin: str | None = None
    cwd: str | None = None
    env: dict[str, str] | None = None


@dataclass
class _FakeAsyncCodex:
    config: _FakeAppServerConfig | None = None
    started_kwargs: list[dict[str, object]] = field(default_factory=list)
    resumed_ids: list[str] = field(default_factory=list)
    fixed_thread: _FakeThread | None = None
    exit_calls: int = 0
    close_calls: int = 0

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> None:
        del exc_type, exc_val, exc_tb
        self.exit_calls += 1

    async def close(self) -> None:
        self.close_calls += 1

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
    fake_codex = _FakeAsyncCodex()
    fake_codex.fixed_thread = _FakeThread(
        id="thr_integration", seeded_notifications=seeded_notifications
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
        return {"mala-locking": {"command": "mala-codex-mcp-locking"}}

    return factory


@dataclass
class _FakeNotification:
    """Stand-in for ``codex_app_server.models.Notification`` (``method`` + ``payload``)."""

    method: str
    payload: object


@pytest.mark.integration
@pytest.mark.asyncio
async def test_provider_runtime_client_end_to_end_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provider → runtime → client → AgentEvent stream (Phase C AC-1)."""
    seeded: list[object] = [
        _FakeNotification(
            method="item/agentMessage/delta",
            payload=SimpleNamespace(
                delta="hi",
                item_id="item_1",
                thread_id="thr_integration",
                turn_id="turn_1",
            ),
        ),
        _FakeNotification(
            method="turn/completed",
            payload=SimpleNamespace(
                thread_id="thr_integration",
                turn=SimpleNamespace(
                    id="turn_1",
                    status=SimpleNamespace(value="completed"),
                    error=None,
                ),
            ),
        ),
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
    received: list[AgentEventValue] = []
    async with client:
        await client.query("smoke prompt")
        async for event in client.receive_response():
            # ``SDKClientProtocol.receive_response`` is typed as
            # ``AsyncIterator[object]`` for cross-coder generality;
            # ``CodexClient`` narrows it to ``AsyncIterator[AgentEventValue]``.
            received.append(cast("AgentEventValue", event))
        assert client.session_id == "thr_integration"  # ty:ignore[unresolved-attribute]

    # Lifecycle: thread_start fired with runtime params, no resume,
    # AsyncCodex closed exactly once on context exit. SDK-shape contract
    # (codex_app_server/api.py:336): ``thread_start`` accepts ``model``,
    # ``approval_policy``, ``sandbox``, ``cwd``, ``base_instructions``;
    # NOT ``effort`` (which is per-turn) and NOT ``mcp_servers``
    # (which ships through the bundled Codex plugin's .mcp.json in
    # Phase G3). The runtime-supplied env + cwd land on
    # ``AppServerConfig`` instead.
    assert len(fake_codex.started_kwargs) == 1
    started = fake_codex.started_kwargs[0]
    assert started["model"] == "gpt-5.5-foo"
    assert started["sandbox"] == "danger-full-access"
    assert started["approval_policy"] == "never"
    assert started["cwd"] == str(tmp_path)
    assert "effort" not in started
    assert "mcp_servers" not in started
    # ``effort`` is per-turn.
    assert fake_codex.fixed_thread is not None
    assert fake_codex.fixed_thread.turn_kwargs[0].get("effort") == "medium"
    # AppServerConfig carried the per-process env so the Phase E hook
    # can authenticate the agent.
    config = fake_codex.config
    assert config is not None
    assert config.cwd == str(tmp_path)
    assert config.env is not None
    assert config.env.get("MALA_AGENT_ID") == "agent-x"
    assert fake_codex.resumed_ids == []
    assert len(received) == 2
    text_event = received[0]
    assert isinstance(text_event, AgentTextEvent)
    assert text_event.text == "hi"
    result_event = received[1]
    assert isinstance(result_event, AgentResultEvent)
    assert result_event.session_id == "thr_integration"
    assert result_event.is_error is False
    assert result_event.subtype == "completed"
    assert fake_codex.close_calls == 1
    assert fake_codex.exit_calls == 0


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

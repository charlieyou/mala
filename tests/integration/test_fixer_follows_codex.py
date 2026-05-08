"""Integration test: fixer follows main coder when ``coder=codex`` (T017 H2).

Phase H AC #5 (when ``coder=codex``, FixerService spawns Codex) is the
parity bullet to the Amp fixer-follows test. Post Phase A
:class:`FixerService` is coder-agnostic — it consumes
:class:`AgentProvider` and never branches on coder name. This test wires
a real :class:`CodexAgentProvider` (with a fake ``codex_app_server``
substituted in) into a :class:`FixerService` and asserts:

  * ``run_fixer`` produces a :class:`CodexClient`-shaped session
    (``thread_start`` was called, not Claude-style ``query``-with-resume).
  * The fixer prompt threaded through the configured template reaches
    ``AsyncThread.turn`` verbatim.
  * The synthetic ``fixer-<uuid>`` agent id is NOT mistaken for a Codex
    thread id (``thr_*``); the SDK gets ``thread_start`` not
    ``thread_resume`` (regression coverage for the
    :class:`SDKClientProtocol` ``session_id`` parameter contract — see
    :func:`tests.unit.infra.clients.test_codex_client.test_query_session_id_is_not_treated_as_resume_thread_id`).
  * The terminal :class:`AgentResultEvent` from Codex propagates as a
    successful :class:`FixerResult`.

This protects the cross-coder wiring at the integration boundary: a
regression that re-introduced Claude-only assumptions in
``FixerService`` (e.g. ``isinstance(builder, ClaudeAgentRuntimeBuilder)``
without an ``else`` branch, or a ``getattr(client, ..., default)`` that
silently no-op'd for non-Claude coders) would surface here even before
:class:`AgentSessionRunner` is touched.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from src.infra.clients.codex_client import CodexClient
from src.infra.clients.codex_provider import CodexAgentProvider
from src.pipeline.fixer_service import (
    FailureContext,
    FixerService,
    FixerServiceConfig,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from pathlib import Path
    from types import ModuleType
    from typing import Self

    from src.core.protocols.agent_provider import AgentProvider


# ---------------------------------------------------------------------------
# Minimal fake codex_app_server SDK (mirrors test_codex_provider.py shape)
# ---------------------------------------------------------------------------


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
    id: str
    seeded_notifications: list[object] = field(default_factory=list)
    turn_inputs: list[_FakeTextInput] = field(default_factory=list)

    async def turn(
        self, text_input: _FakeTextInput, **kwargs: object
    ) -> _FakeTurnHandle:
        del kwargs
        self.turn_inputs.append(text_input)
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

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> None:
        del exc_type, exc_val, exc_tb

    async def close(self) -> None:
        return None

    async def thread_start(self, **kwargs: object) -> _FakeThread:
        self.started_kwargs.append(dict(kwargs))
        if self.fixed_thread is None:
            self.fixed_thread = _FakeThread(id="thr_codex_fixer")
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
        id="thr_codex_fixer", seeded_notifications=seeded_notifications
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


@dataclass
class _FakeNotification:
    method: str
    payload: object


def _empty_factory() -> Callable[..., dict[str, object]]:
    def factory(
        agent_id: str, repo_path: Path, emit_lock_event: object
    ) -> dict[str, object]:
        del agent_id, repo_path, emit_lock_event
        return {}

    return factory


@pytest.fixture(autouse=True)
def _redirect_codex_tee_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Quarantine tee'd JSONL inside tmp_path."""
    monkeypatch.setenv("MALA_CODEX_SESSIONS_DIR", str(tmp_path / "codex-tee"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fixer_runs_through_codex_provider_starts_fresh_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC #5: when wired to ``CodexAgentProvider``, ``run_fixer`` spawns a Codex turn.

    Asserts:
      * ``AsyncCodex.thread_start`` was called with the runtime's
        Codex-shaped kwargs (model / sandbox / approval_policy / cwd).
      * ``thread_resume`` was NOT called — the synthetic ``fixer-<uuid>``
        passed by :class:`FixerService` as ``query(session_id=...)`` is
        not a valid Codex thread id and must not be treated as a resume
        token.
      * The formatted prompt reached ``AsyncThread.turn`` verbatim, so a
        regression in ``FixerService.run_fixer`` that dropped the format
        kwargs (failed_command / validation_commands / etc.) would
        surface here.
      * :class:`FixerResult` is success — the terminal
        :class:`AgentResultEvent` from ``turn/completed`` flows through
        :meth:`FixerService._process_event`.
    """
    seeded: list[object] = [
        _FakeNotification(
            method="item/agentMessage/delta",
            payload=SimpleNamespace(
                delta="working on it",
                item_id="item_fixer_1",
                thread_id="thr_codex_fixer",
                turn_id="turn_fixer_1",
            ),
        ),
        _FakeNotification(
            method="turn/completed",
            payload=SimpleNamespace(
                thread_id="thr_codex_fixer",
                turn=SimpleNamespace(
                    id="turn_fixer_1",
                    status=SimpleNamespace(value="completed"),
                    error=None,
                ),
            ),
        ),
    ]
    fake_codex = _install_fake_codex_app_server(
        monkeypatch, seeded_notifications=seeded
    )

    provider = CodexAgentProvider()
    config = FixerServiceConfig(
        repo_path=tmp_path,
        timeout_seconds=600,
        fixer_prompt=(
            "Attempt {attempt}/{max_attempts}\n"
            "Failed command: {failed_command}\n"
            "Failure: {failure_output}\n"
            "Run validations:\n{validation_commands}"
        ),
    )
    service = FixerService(
        config=config, agent_provider=cast("AgentProvider", provider)
    )
    failure_ctx = FailureContext(
        failure_output="ruff: 1 lint violation",
        attempt=1,
        max_attempts=3,
        failed_command="ruff check .",
        validation_commands="   - `ruff check .`",
    )

    result = await service.run_fixer(failure_ctx)

    assert result.success is True
    assert result.interrupted is False

    # The fixer reached a real CodexClient — thread_start fired with Codex-shaped kwargs.
    assert len(fake_codex.started_kwargs) == 1
    started = fake_codex.started_kwargs[0]
    assert started["cwd"] == str(tmp_path)
    assert started["sandbox"] == "danger-full-access"
    assert started["approval_policy"] == "never"
    # AC #5 regression guard: the fixer's synthetic agent id is NOT a
    # Codex thread id, so resume MUST NOT be triggered.
    assert fake_codex.resumed_ids == []

    # The configured fixer prompt template reaches AsyncThread.turn verbatim.
    assert fake_codex.fixed_thread is not None
    sent_prompt = fake_codex.fixed_thread.turn_inputs[0].prompt
    assert "Attempt 1/3" in sent_prompt
    assert "Failed command: ruff check ." in sent_prompt
    assert "Failure: ruff: 1 lint violation" in sent_prompt
    assert "   - `ruff check .`" in sent_prompt


@pytest.mark.integration
@pytest.mark.asyncio
async def test_fixer_uses_codex_provider_runtime_builder_and_client_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The FixerService consults the provider for both runtime + client.

    AC #5 wiring proof: a regression that wires
    :class:`FixerService` to the wrong factory (e.g. always Claude) would
    surface as a non-:class:`CodexClient` instance escaping
    ``client_factory.create``. We verify the fixer-spawned client is a
    real :class:`CodexClient` by intercepting ``client_factory.create``.
    """
    _install_fake_codex_app_server(
        monkeypatch,
        seeded_notifications=[
            _FakeNotification(
                method="turn/completed",
                payload=SimpleNamespace(
                    thread_id="thr_codex_fixer",
                    turn=SimpleNamespace(
                        id="turn_only",
                        status=SimpleNamespace(value="completed"),
                        error=None,
                    ),
                ),
            ),
        ],
    )

    provider = CodexAgentProvider()
    spawned_clients: list[object] = []
    real_create = provider.client_factory.create

    def _capturing_create(runtime: object) -> object:
        client = real_create(runtime)
        spawned_clients.append(client)
        return client

    # Avoid mutating the cached factory's bound method via setattr on the
    # frozen-ish proxy — wrap by monkeypatching the class method instead.
    monkeypatch.setattr(
        type(provider.client_factory),
        "create",
        lambda self, runtime: _capturing_create(runtime),
        raising=True,
    )

    config = FixerServiceConfig(
        repo_path=tmp_path,
        timeout_seconds=300,
        fixer_prompt="Fix {failure_output}: {validation_commands}",
    )
    service = FixerService(
        config=config, agent_provider=cast("AgentProvider", provider)
    )
    failure_ctx = FailureContext(
        failure_output="boom",
        attempt=1,
        max_attempts=2,
        failed_command="pytest",
        validation_commands="(none)",
    )

    result = await service.run_fixer(failure_ctx)

    assert result.success is True
    assert len(spawned_clients) == 1
    assert isinstance(spawned_clients[0], CodexClient), (
        "FixerService configured with CodexAgentProvider must spawn a "
        "CodexClient via the provider's client_factory; got "
        f"{type(spawned_clients[0]).__name__}."
    )
    # Sanity: the captured client is the live, async-context-managed
    # object the fixer used. ``isinstance`` above narrows to
    # ``CodexClient`` so attribute access is type-checked.
    captured = spawned_clients[0]
    assert isinstance(captured, CodexClient)
    assert captured.session_id == "thr_codex_fixer"

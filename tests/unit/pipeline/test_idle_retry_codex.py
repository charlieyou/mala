"""Idle-timeout retry resumes a Codex thread (T017 H3).

AC #8 (resume + idle/review retries work for Codex) requires that when a
Codex turn's notification stream goes idle past the configured threshold,
:class:`IdleTimeoutRetryPolicy` resumes via ``AsyncCodex.thread_resume``
on the *same* thread id rather than starting a fresh thread. After Phase
A the retry policy is coder-agnostic — it consumes
:class:`SDKClientFactoryProtocol` and never inspects the runtime's shape
— so this test exercises the policy with the real
:class:`CodexAgentProvider.client_factory` plus a fake ``codex_app_server``
to prove no Codex-specific runtime code is needed.

The test wires the production policy:

  * First iteration: the stream processor raises
    :class:`IdleTimeoutError` after recording the client-known thread id
    (Codex thread ids arrive via :attr:`CodexClient.session_id` after
    ``thread_start``).
  * Policy disconnects, calls ``client_factory.with_resume(runtime,
    resume_id)`` to produce a sibling :class:`CodexRuntime` carrying
    ``resume_thread_id``, and creates a fresh client.
  * Second iteration: the new client's :meth:`query` reaches
    ``AsyncCodex.thread_resume`` with the captured thread id; the
    processor returns success.

The simulated stalled iterator lives inside the stream processor stub
(faster than wiring a real :class:`asyncio.wait_for` race against a fake
``AsyncCodex`` stream) — :class:`IdleTimeoutStream`'s timeout semantics
already have unit coverage in
``tests/unit/pipeline/test_message_stream_processor.py``; this test
focuses on the cross-coder *wiring* the retry path depends on.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pytest

from src.domain.lifecycle import LifecycleContext
from src.infra.clients.codex_client import CodexClient
from src.infra.clients.codex_provider import CodexAgentProvider
from src.infra.clients.codex_runtime import CodexRuntime
from src.pipeline.idle_retry_policy import (
    IdleTimeoutRetryPolicy,
    RetryConfig,
)
from src.pipeline.message_stream_processor import (
    IdleTimeoutError,
    MessageIterationResult,
    MessageIterationState,
)
from tests.fakes import FakeLintCache

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from pathlib import Path
    from types import ModuleType
    from typing import Self

    from src.core.protocols.sdk import (
        SDKClientFactoryProtocol,
        SDKClientProtocol,
    )
    from src.infra.telemetry import TelemetrySpan
    from src.pipeline.message_stream_processor import (
        IdleTimeoutStream,
        LintCacheProtocol,
    )


# ---------------------------------------------------------------------------
# Fake codex_app_server (mirrors test_codex_provider.py shape)
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
    started_thread: _FakeThread | None = None
    resumed_thread: _FakeThread | None = None

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
        if self.started_thread is None:
            self.started_thread = _FakeThread(id="thr_idle_codex")
        return self.started_thread

    async def thread_resume(self, thread_id: str) -> _FakeThread:
        self.resumed_ids.append(thread_id)
        if self.resumed_thread is None:
            self.resumed_thread = _FakeThread(id=thread_id)
        return self.resumed_thread


def _install_fake_codex_app_server(
    monkeypatch: pytest.MonkeyPatch,
) -> _FakeAsyncCodex:
    fake_codex = _FakeAsyncCodex()
    fake_codex.started_thread = _FakeThread(id="thr_idle_codex")
    fake_codex.resumed_thread = _FakeThread(id="thr_idle_codex")

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


# ---------------------------------------------------------------------------
# Capturing client factory: forwards every call to the real factory but
# records the runtimes + clients so tests can assert on the cross-resume
# wiring without monkeypatching ``_CodexClientFactory``.
# ---------------------------------------------------------------------------


class _CapturingClientFactory:
    """Wraps :class:`SDKClientFactoryProtocol` to record observable calls."""

    def __init__(self, inner: SDKClientFactoryProtocol) -> None:
        self._inner = inner
        self.create_runtimes: list[object] = []
        self.spawned_clients: list[SDKClientProtocol] = []

    def create(self, runtime: object) -> SDKClientProtocol:
        self.create_runtimes.append(runtime)
        client = self._inner.create(runtime)
        self.spawned_clients.append(client)
        return client

    def with_resume(self, runtime: object, resume: str | None) -> object:
        return self._inner.with_resume(runtime, resume)


# ---------------------------------------------------------------------------
# Stream processor that simulates a stalled iterator
# ---------------------------------------------------------------------------


class _IdleThenSuccessProcessor:
    """Raises :class:`IdleTimeoutError` once, then succeeds.

    Mirrors the production stream processor's contract: it owns the
    iteration over :class:`IdleTimeoutStream` and is the layer that
    raises :class:`IdleTimeoutError` when the wrapped stream times out.
    Substituting it here lets the test exercise the retry-policy →
    Codex-client-factory wiring deterministically without depending on
    real wall-clock timeouts.
    """

    def __init__(self, stalled_thread_id: str) -> None:
        self.stalled_thread_id = stalled_thread_id
        self.call_count = 0

    async def process_stream(
        self,
        stream: IdleTimeoutStream,
        issue_id: str,
        state: MessageIterationState,
        lifecycle_ctx: LifecycleContext,
        lint_cache: LintCacheProtocol,
        query_start: float,
        tracer: TelemetrySpan | None,
    ) -> MessageIterationResult:
        del stream, issue_id, lifecycle_ctx, lint_cache, query_start, tracer
        self.call_count += 1
        if self.call_count == 1:
            # Tool side effects already happened on the stalled turn —
            # the policy MUST resume rather than restart fresh.
            state.tool_calls_this_turn = 7
            raise IdleTimeoutError("Codex notification iterator stalled")
        return MessageIterationResult(success=True, session_id=self.stalled_thread_id)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _redirect_codex_tee_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Quarantine tee'd JSONL inside tmp_path."""
    monkeypatch.setenv("MALA_CODEX_SESSIONS_DIR", str(tmp_path / "codex-tee"))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_idle_timeout_resumes_same_codex_thread(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stalled iterator → policy resumes the captured Codex thread.

    AC #8 wiring proof: the retry policy's :meth:`with_resume` plumbing
    composes correctly with :class:`_CodexClientFactory.with_resume`,
    and the resumed runtime drives the next :class:`CodexClient` to
    issue ``thread_resume`` on the thread id captured before the idle
    error. No Codex-specific runtime code is needed — the existing
    coder-agnostic policy + ``with_resume`` plumbing is enough.
    """
    fake_codex = _install_fake_codex_app_server(monkeypatch)
    provider = CodexAgentProvider()
    runtime = provider.runtime_builder(
        tmp_path, "agent-idle", mcp_server_factory=_empty_factory()
    ).build()
    assert isinstance(runtime, CodexRuntime)
    assert runtime.resume_thread_id is None  # baseline: nothing to resume yet

    processor = _IdleThenSuccessProcessor(stalled_thread_id="thr_idle_codex")
    policy = IdleTimeoutRetryPolicy(
        sdk_client_factory=provider.client_factory,
        stream_processor_factory=lambda: processor,  # ty:ignore[invalid-argument-type]
        config=RetryConfig(
            max_idle_retries=2,
            idle_retry_backoff=(0.0, 0.0),
            idle_resume_prompt="Resume {issue_id}",
        ),
    )

    state = MessageIterationState()
    lifecycle_ctx = LifecycleContext()
    # The policy first reads from the live client (CodexClient.session_id)
    # before falling back to lifecycle_ctx; pre-seeding lifecycle_ctx
    # ensures the retry succeeds even if a future regression breaks the
    # client-side capture path.
    lifecycle_ctx.session_id = "thr_idle_codex"

    result = await policy.execute_iteration(
        query="Initial Codex prompt",
        issue_id="ISSUE-CODEX-IDLE",
        runtime=runtime,
        state=state,
        lifecycle_ctx=lifecycle_ctx,
        lint_cache=FakeLintCache(),
        idle_timeout_seconds=300.0,
    )

    assert result.success is True
    assert processor.call_count == 2

    # First turn started a fresh Codex thread; the retry called
    # thread_resume on the SAME id rather than starting a second thread.
    assert len(fake_codex.started_kwargs) == 1
    assert fake_codex.resumed_ids == ["thr_idle_codex"]
    # The retry's prompt followed the configured idle-resume template.
    assert fake_codex.resumed_thread is not None
    resumed_prompts = [t.prompt for t in fake_codex.resumed_thread.turn_inputs]
    assert resumed_prompts == ["Resume ISSUE-CODEX-IDLE"]
    # State reflects exactly one idle retry attempt.
    assert state.idle_retry_count == 1


@pytest.mark.unit
@pytest.mark.asyncio
async def test_idle_resume_runtime_is_codex_runtime_with_resume_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runtime fed to the retry's ``CodexClient`` carries ``resume_thread_id``.

    Regression target: a future change that introduced a Claude-shaped
    options dict between the policy and the Codex factory would silently
    drop the resume id (the dict-merge path in
    :class:`FakeSDKClientFactory.with_resume` is permissive). Asserting
    on the concrete :class:`CodexRuntime` field at the boundary catches
    that drift.
    """
    fake_codex = _install_fake_codex_app_server(monkeypatch)
    provider = CodexAgentProvider()
    runtime = provider.runtime_builder(
        tmp_path, "agent-idle-2", mcp_server_factory=_empty_factory()
    ).build()
    assert isinstance(runtime, CodexRuntime)

    capturing_factory = _CapturingClientFactory(provider.client_factory)

    processor = _IdleThenSuccessProcessor(stalled_thread_id="thr_idle_codex")
    policy = IdleTimeoutRetryPolicy(
        sdk_client_factory=capturing_factory,
        stream_processor_factory=lambda: processor,  # ty:ignore[invalid-argument-type]
        config=RetryConfig(
            max_idle_retries=2,
            idle_retry_backoff=(0.0, 0.0),
            idle_resume_prompt="Resume {issue_id}",
        ),
    )
    lifecycle_ctx = LifecycleContext()
    lifecycle_ctx.session_id = "thr_idle_codex"
    state = MessageIterationState()

    result = await policy.execute_iteration(
        query="initial",
        issue_id="ISSUE-CODEX-IDLE-2",
        runtime=runtime,
        state=state,
        lifecycle_ctx=lifecycle_ctx,
        lint_cache=FakeLintCache(),
        idle_timeout_seconds=300.0,
    )
    assert result.success is True

    # First spawn used the original (no-resume) runtime; second spawn
    # used a sibling CodexRuntime with the captured thread id.
    assert len(capturing_factory.create_runtimes) == 2
    initial, retry = capturing_factory.create_runtimes
    assert isinstance(initial, CodexRuntime)
    assert initial.resume_thread_id is None
    assert isinstance(retry, CodexRuntime)
    assert retry.resume_thread_id == "thr_idle_codex"
    # Resume runtime is a sibling, not the same instance — the policy
    # never mutates the input runtime in place.
    assert retry is not initial
    # And the second spawn produced a real CodexClient that hit thread_resume.
    assert fake_codex.resumed_ids == ["thr_idle_codex"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_each_idle_retry_creates_fresh_codex_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each retry attempt spawns a brand-new :class:`CodexClient`.

    The Codex SDK manages its own ``codex app-server`` subprocess per
    :class:`AsyncCodex` instance. Re-using a stalled client across an
    idle retry would mean re-using the subprocess that just hung — which
    is exactly what the policy's disconnect-then-recreate dance avoids
    on the Amp path. This test pins the same posture for Codex so a
    regression that re-used the stalled client (e.g. an in-place
    ``client.with_resume`` instead of factory + new instance) would fail
    visibly here.
    """
    _install_fake_codex_app_server(monkeypatch)
    provider = CodexAgentProvider()
    runtime = provider.runtime_builder(
        tmp_path, "agent-idle-3", mcp_server_factory=_empty_factory()
    ).build()

    capturing_factory = _CapturingClientFactory(provider.client_factory)

    processor = _IdleThenSuccessProcessor(stalled_thread_id="thr_idle_codex")
    policy = IdleTimeoutRetryPolicy(
        sdk_client_factory=capturing_factory,
        stream_processor_factory=lambda: processor,  # ty:ignore[invalid-argument-type]
        config=RetryConfig(
            max_idle_retries=2,
            idle_retry_backoff=(0.0, 0.0),
            idle_resume_prompt="Resume {issue_id}",
        ),
    )
    lifecycle_ctx = LifecycleContext()
    lifecycle_ctx.session_id = "thr_idle_codex"
    state = MessageIterationState()

    result = await policy.execute_iteration(
        query="initial",
        issue_id="ISSUE-CODEX-IDLE-3",
        runtime=runtime,
        state=state,
        lifecycle_ctx=lifecycle_ctx,
        lint_cache=FakeLintCache(),
        idle_timeout_seconds=300.0,
    )
    assert result.success is True

    assert len(capturing_factory.spawned_clients) == 2
    assert all(isinstance(c, CodexClient) for c in capturing_factory.spawned_clients)
    assert (
        capturing_factory.spawned_clients[0] is not capturing_factory.spawned_clients[1]
    )

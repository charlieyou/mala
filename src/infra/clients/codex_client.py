"""Codex SDK client (Phase C, T010).

:class:`CodexClient` conforms structurally to
:class:`src.core.protocols.sdk.SDKClientProtocol`, backed by
``codex_app_server.AsyncCodex`` + ``AsyncThread`` + ``AsyncTurnHandle``.
The Codex SDK manages its own ``codex app-server`` subprocess; mala does
not spawn ``codex`` directly the way :class:`AmpClient` spawns ``amp``.
Implication: most adapter complexity from the Amp path (process group,
SIGTERM/SIGKILL grace, stderr ring buffer, tee'd JSONL bootstrap) does
not recur — the SDK owns the subprocess lifecycle.

Lazy-import contract (plan ``L733``): importing this module must NOT
transitively pull ``codex_app_server`` so a Claude-only or Amp-only run
does not need the Codex SDK installed. The SDK is imported lazily inside
:meth:`__aenter__` and :meth:`query` (and only those entry points).

Phase C scope (plan ``L723-L740``):

  * Lifecycle: ``async with CodexClient(runtime) as client:`` constructs
    ``AsyncCodex`` lazily and enters its async context.
  * ``query(prompt, session_id)``: starts (``thread_start``) or resumes
    (``thread_resume``) a Codex thread on first call, then issues a turn
    via ``AsyncThread.turn(TextInput(prompt))`` and stashes the
    ``AsyncTurnHandle``.
  * ``receive_response()``: yields raw ``codex_app_server`` notifications
    from ``TurnHandle.stream()``. Phase D (T011) replaces this with the
    notification → :class:`AgentEvent` adapter; Phase C deliberately
    keeps the stream raw so the lifecycle wiring can land independently.
  * ``with_resume(thread_id)``: stores a resume token consumed on the
    next :meth:`query`.
  * ``disconnect()``: idempotent; calls ``AsyncTurnHandle.interrupt()``
    if a turn is active and tears down ``AsyncCodex`` exactly once.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType
    from typing import Self

    from src.infra.clients.codex_runtime import CodexRuntime

logger = logging.getLogger(__name__)


class CodexClient:
    """SDKClientProtocol implementation backed by ``codex_app_server.AsyncCodex``.

    The client owns one ``AsyncCodex`` instance for its async-context
    lifetime. ``AsyncCodex`` itself manages the ``codex app-server``
    subprocess; mala only orchestrates thread + turn lifecycle on top of
    that.
    """

    def __init__(self, runtime: CodexRuntime) -> None:
        """Bind the per-session runtime; defer SDK construction to ``__aenter__``."""
        self._runtime = runtime
        self._codex: Any = None  # ``codex_app_server.AsyncCodex`` once entered
        self._thread: Any = None  # ``AsyncThread`` once started/resumed
        self._turn: Any = None  # ``AsyncTurnHandle`` for the active turn
        self._closed: bool = False
        # ``with_resume`` overrides ``runtime.resume_thread_id`` if called
        # after the runtime was built (parity with
        # :meth:`AmpClient.with_resume`).
        self._resume_thread_id: str | None = runtime.resume_thread_id

    # ------------------------------------------------------------------
    # Properties used by the orchestrator + tests
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str | None:
        """Codex thread id (``thr_...``) once a turn has started.

        Returns ``None`` until :meth:`query` starts or resumes a thread;
        callers that need the id for evidence / log routing must wait
        for the first turn to begin.
        """
        if self._thread is None:
            return None
        thread_id = getattr(self._thread, "id", None)
        return None if thread_id is None else str(thread_id)

    @property
    def thread_id(self) -> str | None:
        """Alias for :attr:`session_id` (Codex's session is its thread)."""
        return self._session_id_or_none()

    def _session_id_or_none(self) -> str | None:
        return self.session_id

    def with_resume(self, thread_id: str) -> Self:
        """Configure the next :meth:`query` to resume ``thread_id``.

        Stores the resume token; the next :meth:`query` will pick
        ``AsyncCodex.thread_resume(thread_id)`` over ``thread_start``.
        Returns ``self`` for fluent chaining.
        """
        self._resume_thread_id = thread_id
        return self

    # ------------------------------------------------------------------
    # Async context management
    # ------------------------------------------------------------------

    async def __aenter__(self) -> Self:
        """Lazy-construct ``AsyncCodex`` and enter its async context.

        The lazy import is the lazy-import contract pivot point: any
        caller that reaches here must have ``codex_app_server``
        importable. Callers that want to test the wiring without the SDK
        installed inject a fake module via ``sys.modules`` before this
        call (see ``tests/unit/infra/clients/test_codex_client.py``).
        """
        from codex_app_server import (  # type: ignore[import-not-found]  # ty:ignore[unresolved-import]
            AsyncCodex,
        )

        # ``AsyncCodex()`` accepts an optional ``env`` overlay per the
        # plan's per-process env isolation contract; the runtime carries
        # the per-process env dict ready for that path. The exact
        # constructor argument name is confirmed by the Phase B/C spike
        # (plan ``L1317``) — for Phase C we pass the env opportunistically
        # via a kwarg the SDK will expose, and fall back to constructing
        # ``AsyncCodex()`` without it if the SDK doesn't accept it (the
        # Phase E state-file fallback covers the missing-env case).
        try:
            self._codex = AsyncCodex(env=dict(self._runtime.env))
        except TypeError:
            # SDK does not accept ``env=`` — Phase E state file is the
            # transport instead. Tests cover both shapes.
            self._codex = AsyncCodex()
        await self._codex.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # SDKClientProtocol surface
    # ------------------------------------------------------------------

    async def query(self, prompt: str, session_id: str | None = None) -> None:
        """Start (or resume) a Codex thread and issue a turn for ``prompt``.

        On the first call, picks between ``thread_start`` (fresh thread)
        and ``thread_resume`` (when :attr:`_resume_thread_id` is set,
        either via the runtime or :meth:`with_resume`). Subsequent calls
        on the same client reuse the existing thread and only issue a
        new turn — Codex threads accept multiple turns and the per-issue
        lifecycle never opens a fresh thread mid-session.

        ``session_id`` (from :class:`SDKClientProtocol`) is treated as a
        resume thread id when no prior :meth:`with_resume` /
        ``runtime.resume_thread_id`` was set, mirroring the Amp adapter's
        contract.
        """
        codex = self._codex
        if codex is None:
            raise RuntimeError(
                "CodexClient.query() called before __aenter__; the "
                "AsyncCodex context must be entered first."
            )
        if session_id is not None and self._resume_thread_id is None:
            self._resume_thread_id = session_id

        if self._thread is None:
            if self._resume_thread_id is not None:
                self._thread = await codex.thread_resume(self._resume_thread_id)
            else:
                self._thread = await codex.thread_start(
                    model=self._runtime.model,
                    effort=self._runtime.effort,
                    sandbox=self._runtime.sandbox,
                    approval_policy=self._runtime.approval_policy,
                    mcp_servers=self._runtime.mcp_servers,
                    base_instructions=self._runtime.base_instructions,
                    cwd=str(self._runtime.cwd),
                )

        from codex_app_server import (  # type: ignore[import-not-found]  # ty:ignore[unresolved-import]
            TextInput,
        )

        self._turn = await self._thread.turn(TextInput(prompt))

    async def receive_response(self) -> AsyncIterator[object]:
        """Yield raw ``codex_app_server`` notifications from the active turn.

        Phase C emits the SDK notifications verbatim; Phase D (T011)
        wraps this iterator with the notification → :class:`AgentEvent`
        adapter so :class:`MessageStreamProcessor` can consume them
        without provider-specific branches.
        """
        turn = self._turn
        if turn is None:
            raise RuntimeError(
                "CodexClient.receive_response() called before query(); no "
                "active TurnHandle to stream from."
            )
        async for notification in turn.stream():
            yield notification

    async def disconnect(self) -> None:
        """Idempotently interrupt the active turn and close ``AsyncCodex``.

        Calls ``AsyncTurnHandle.interrupt()`` and ``AsyncCodex.close()``
        at most once per client (plan ``L732``). Errors from the SDK
        teardown are logged at debug level rather than raised so a
        failing close path does not mask the original error that drove
        ``disconnect`` (e.g., ``CancelledError`` from SIGINT).
        """
        if self._closed:
            return
        self._closed = True

        turn = self._turn
        if turn is not None:
            try:
                await turn.interrupt()
            except Exception as exc:
                logger.debug(
                    "CodexClient: TurnHandle.interrupt raised on disconnect: %s",
                    exc,
                )
            self._turn = None

        codex = self._codex
        if codex is not None:
            try:
                await codex.__aexit__(None, None, None)
            except Exception as exc:
                logger.debug(
                    "CodexClient: AsyncCodex teardown raised on disconnect: %s",
                    exc,
                )
            self._codex = None

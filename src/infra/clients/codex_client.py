"""Codex SDK client (Phase C/D, T010 + T011).

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
  * ``receive_response()``: consumes ``TurnHandle.stream()`` and emits
    coder-agnostic :class:`AgentEvent`s (plan C3 / D1-D7).
    :class:`CodexEventAdapter` owns the per-turn translation state (it
    needs to dedup ``AgentMessage`` deltas vs the matching
    ``ItemCompleted`` carrying the same text), so a fresh adapter is
    constructed for each :meth:`query` turn.
  * ``with_resume(thread_id)``: stores a resume token consumed on the
    next :meth:`query`.
  * ``disconnect()``: idempotent; calls ``AsyncTurnHandle.interrupt()``
    if a turn is active and tears down ``AsyncCodex`` exactly once.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.infra.clients.codex_event_adapter import CodexEventAdapter

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from types import TracebackType
    from typing import Self

    from src.core.protocols.agent_event import AgentEventValue
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
        # The adapter holds per-turn dedup state (D1: agent message
        # delta vs completed item). A fresh instance is bound when each
        # :meth:`query` issues a new turn so state never leaks across
        # turns on the same client.
        self._event_adapter: CodexEventAdapter | None = None

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

        The real SDK shape is ``AsyncCodex(config: AppServerConfig | None)``,
        with the per-process env overlay living on
        ``AppServerConfig.env`` (see ``codex_app_server/api.py:291`` and
        ``client.py:125``). We bundle the runtime's per-process env dict
        plus the orchestrated repo cwd into the config so the spawned
        ``codex app-server`` subprocess inherits ``MALA_AGENT_ID`` /
        ``MALA_REPO_NAMESPACE`` (parity with the Amp path's posture in
        ``AmpRuntimeBuilder.build``, plan ``L815``). Without this, the
        Phase E ``mala-codex-pre-tool-use`` hook would see the missing
        ``MALA_AGENT_ID`` and deny every shell write fail-closed.

        ``CODEX_BINARY`` is honored when set so users can point at an
        external ``codex`` binary; otherwise the SDK's bundled
        ``codex_cli_bin`` resolution applies (see the spike's
        ``_resolve_codex_bin`` helper in
        ``tests/spike/test_codex_thread_read_evidence.py``).
        """
        import os

        from codex_app_server import (  # type: ignore[import-not-found]  # ty:ignore[unresolved-import]
            AppServerConfig,
            AsyncCodex,
        )

        codex_bin = os.environ.get("CODEX_BINARY") or None
        config = AppServerConfig(
            codex_bin=codex_bin,
            cwd=str(self._runtime.cwd),
            env=dict(self._runtime.env),
        )
        self._codex = AsyncCodex(config=config)
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

        ``session_id`` (from :class:`SDKClientProtocol`) is intentionally
        ignored: Codex resume tokens are wired through
        :meth:`with_resume` / ``runtime.resume_thread_id`` (e.g. via
        ``client_factory.with_resume(...)``). Callers such as
        :class:`FixerService` pass an opaque agent id (e.g.
        ``fixer-abcd1234``) here, which is not a Codex thread id
        (``thr_...``) and must not be misread as a resume token.

        SDK shape (``codex_app_server.AsyncCodex.thread_start``,
        ``codex_app_server/api.py:336``) accepts ``model``,
        ``approval_policy``, ``sandbox``, ``base_instructions``, ``cwd``,
        and a config-override JsonObject — but NOT ``effort`` or
        ``mcp_servers``. ``effort`` is a per-turn parameter on
        :meth:`AsyncThread.turn` (``api.py:610``); MCP server
        configuration ships through the bundled Codex plugin's
        ``.mcp.json`` (Phase G3 / T016), not via this call. The
        runtime's :attr:`mcp_servers` field is plumbed end-to-end for
        forward-compat with Phase G but is intentionally NOT passed
        here.
        """
        del session_id  # see docstring: not a Codex resume token
        codex = self._codex
        if codex is None:
            raise RuntimeError(
                "CodexClient.query() called before __aenter__; the "
                "AsyncCodex context must be entered first."
            )

        if self._thread is None:
            if self._resume_thread_id is not None:
                self._thread = await codex.thread_resume(self._resume_thread_id)
            else:
                self._thread = await codex.thread_start(
                    model=self._runtime.model,
                    sandbox=self._runtime.sandbox,
                    approval_policy=self._runtime.approval_policy,
                    base_instructions=self._runtime.base_instructions,
                    cwd=str(self._runtime.cwd),
                )

        from codex_app_server import (  # type: ignore[import-not-found]  # ty:ignore[unresolved-import]
            TextInput,
        )

        self._turn = await self._thread.turn(
            TextInput(prompt), effort=self._runtime.effort
        )
        # Reset per-turn dedup state. Without this a second turn on the
        # same client would mistakenly suppress the first
        # ``ItemCompleted`` that happens to reuse an item id seen on
        # the prior turn.
        self._event_adapter = CodexEventAdapter()

    async def receive_response(self) -> AsyncIterator[AgentEventValue]:
        """Yield :class:`AgentEvent`s parsed from the active turn's stream.

        Each ``codex_app_server`` ``Notification`` (``method`` + ``payload``)
        is run through :class:`CodexEventAdapter`, which implements the
        Phase D mapping table (plan ``L749-L762``):

          * ``item/agentMessage/delta`` → :class:`AgentTextEvent`,
            with the matching ``item/completed`` carrying
            ``AgentMessageThreadItem`` deduplicated against the deltas.
          * ``item/started`` carrying ``CommandExecutionThreadItem`` /
            ``FileChangeThreadItem`` / ``McpToolCallThreadItem`` →
            :class:`AgentToolUseEvent`; the matching ``item/completed``
            → paired :class:`AgentToolResultEvent` keyed by item id.
          * ``turn/completed`` → :class:`AgentResultEvent` with
            ``is_error`` derived from ``TurnStatus``.
          * ``error`` → :class:`AgentResultEvent` with ``is_error=True``
            so a mid-turn provider error surfaces as a terminal failure
            rather than an idle hang.
          * ``item/reasoning/*`` (and reasoning items via
            ``item/completed``) → no event (decision #12); reasoning is
            tee'd to disk by Phase F (T013), not surfaced to the
            pipeline.
        """
        turn = self._turn
        if turn is None:
            raise RuntimeError(
                "CodexClient.receive_response() called before query(); no "
                "active TurnHandle to stream from."
            )
        adapter = self._event_adapter
        if adapter is None:
            # Defensive: ``query()`` always installs an adapter, but if
            # a caller wires up ``receive_response`` from a fresh
            # client they should still get a usable iterator.
            adapter = CodexEventAdapter()
            self._event_adapter = adapter
        async for notification in turn.stream():
            for event in adapter.to_events(notification):
                yield event

    async def disconnect(self) -> None:
        """Idempotently interrupt the active turn and close ``AsyncCodex``.

        Calls ``AsyncTurnHandle.interrupt()`` and ``AsyncCodex.close()``
        at most once per client (plan ``L732``). The SDK's
        ``AsyncCodex.__aexit__`` delegates to ``close()`` (see
        ``codex_app_server/api.py:301``), but we invoke ``close()``
        directly so the SIGINT cancellation path explicitly satisfies
        the AC-3 contract regardless of whether disconnect is reached
        via ``async with`` unwinding or an explicit out-of-band call.

        Cancellation safety: in production the SIGINT chain delivers
        ``task.cancel()`` to whichever task is unwinding through this
        method, so the first yielding ``await`` (``turn.interrupt()``)
        raises :class:`asyncio.CancelledError` — a ``BaseException``
        that an ``except Exception`` cannot catch. ``close()`` is
        guarded with ``try/finally`` so the cancellation path still
        invokes ``AsyncCodex.close()`` exactly once before
        ``CancelledError`` propagates. Non-cancellation errors from
        the SDK teardown are logged at debug level rather than raised
        so a failing close path does not mask the original error that
        drove ``disconnect`` (e.g., ``CancelledError`` from SIGINT).
        """
        if self._closed:
            return
        self._closed = True

        turn = self._turn
        self._turn = None
        codex = self._codex
        self._codex = None

        try:
            if turn is not None:
                try:
                    await turn.interrupt()
                except Exception as exc:
                    logger.debug(
                        "CodexClient: TurnHandle.interrupt raised on disconnect: %s",
                        exc,
                    )
        finally:
            if codex is not None:
                try:
                    await codex.close()
                except Exception as exc:
                    logger.debug(
                        "CodexClient: AsyncCodex.close raised on disconnect: %s",
                        exc,
                    )

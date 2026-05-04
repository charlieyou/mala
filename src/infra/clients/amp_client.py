"""Subprocess + stream-json adapter for the Amp coder.

:class:`AmpClient` conforms structurally to
:class:`src.core.protocols.sdk.SDKClientProtocol` so the existing pipeline
(``MessageStreamProcessor`` etc.) consumes it without provider-specific
branches. It spawns ``amp --execute --stream-json`` as a child process,
parses every JSON line from stdout into the synthetic Anthropic-shaped
dataclasses defined in :mod:`src.infra.clients.amp_messages`, and tees
every event verbatim to ``~/.config/mala/amp-sessions/{thread_id}.jsonl``
so :class:`AmpLogProvider` (T010) can later parse validation evidence.

Key invariants:
  * Tee writes happen inline with stdout iteration. A crash mid-stream
    must not lose evidence already on disk.
  * The tee path starts pending (``.pending-{uuid}.jsonl``) and is
    atomically renamed once the first ``system(init)`` event captures the
    Amp thread id. On resume, the existing ``{thread_id}.jsonl`` is
    appended to (append-mode open) so cross-invocation evidence is
    preserved exactly as the plan describes.
  * If ``assistant`` / ``user`` / ``result`` events arrive without a
    preceding ``system(init)``, the pending file is preserved on disk
    and :class:`AmpStreamMissingInitError` is raised with truncated
    stderr/stdout slices for diagnostics.
  * Stderr is captured into a bounded ring buffer so unbounded growth
    cannot OOM the orchestrator on a chatty Amp run.

The adapter is clean-room: it does **not** import ``claude_agent_sdk``.
It receives its argv from :class:`src.infra.clients.amp_runtime.AmpRuntime`
(assembled in T012); ``AmpClient`` does not reassemble argv itself, only
applies a documented resume-shape modification when ``with_resume`` was
called.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import re
import signal
import sys
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from src.infra.clients.amp_messages import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping, Sequence
    from pathlib import Path
    from types import TracebackType
    from typing import Self

logger = logging.getLogger(__name__)


_STDERR_BUFFER_LIMIT = 4096
"""Cap stderr ring buffer at 4 KiB; anything older is discarded."""

_STDOUT_TAIL_LIMIT = 4096
"""Cap stdout-tail buffer at 4 KiB for diagnostic attachment to errors."""

_STREAM_JSON_READER_LIMIT = 16 * 1024 * 1024
"""Allow large single-line stream-json events such as verbose tool results."""

_DEFAULT_KILL_GRACE_SECONDS = 2.0
"""Seconds between SIGTERM and SIGKILL during ``__aexit__`` cleanup."""

_HANDOFF_EXPORT_POLL_SECONDS = 1.0
"""Seconds between child-thread export polls after a followed handoff."""

_HANDOFF_EXPORT_TIMEOUT_SECONDS = 900.0
"""Maximum time to wait for an Amp handoff child thread to complete."""

_AUTH_ERROR_MARKERS: tuple[str, ...] = (
    "unauthorized",
    "401",
    "forbidden",
)
"""Substrings on stderr that classify the run as an auth failure."""


ResumeStrategy = Literal["thread-id-flag", "threads-continue", "fallback"]
"""Which resume argv shape :meth:`AmpClient.with_resume` produces.

  * ``"threads-continue"`` (default): replaces the leading argv with
    ``[amp, threads, continue, <id>, ...]`` keeping all subsequent
    flags (including ``--execute`` and ``--stream-json``). This is the
    only shape the current Amp CLI accepts together with
    ``--stream-json``.
  * ``"thread-id-flag"``: adds (or updates) ``--thread-id <id>`` on the
    existing ``amp --execute`` argv. Retained for compatibility but no
    longer accepted by recent Amp CLI builds (returns ``unknown option
    '--thread-id'``).
  * ``"fallback"``: leaves argv untouched; logs a warning. Caller is
    responsible for preserving context via prompt accumulation.
"""


@dataclass(frozen=True)
class AmpClientOptions:
    """Spawn options for one :class:`AmpClient` session.

    Mirrors the plan's L237-L244 sketch. Bundles everything ``AmpClient``
    needs to spawn ``amp --execute --stream-json`` and tee its output:
    cwd, env, full argv (assembled by :class:`AmpRuntimeBuilder` in T012
    — ``AmpClient`` does not reassemble argv itself), tee log path,
    optional resume thread id, and extra CLI args (always includes
    ``--dangerously-allow-all`` per the plan's safety contract).
    """

    cwd: Path
    env: Mapping[str, str]
    argv: Sequence[str]
    log_path: Path
    thread_id: str | None = None
    lock_event_log_path: Path | None = None
    lock_event_callback: Callable[[object], object] | None = None
    extra_cli_args: tuple[str, ...] = ("--dangerously-allow-all",)
    resume_strategy: ResumeStrategy = "threads-continue"
    kill_grace_seconds: float = _DEFAULT_KILL_GRACE_SECONDS


class AmpStreamMissingInitError(Exception):
    """Raised when ``assistant``/``user``/``result`` arrives before ``system(init)``.

    Indicates upstream stream-json drift or an Amp bug. The pending tee
    file is preserved on disk for diagnostics; ``stderr_tail`` /
    ``stdout_tail`` are bounded to at most 4 KiB each to avoid unbounded
    memory attached to the exception.
    """

    def __init__(
        self,
        message: str,
        *,
        pending_path: Path,
        stderr_tail: str,
        stdout_tail: str,
    ) -> None:
        super().__init__(message)
        self.pending_path = pending_path
        self.stderr_tail = stderr_tail
        self.stdout_tail = stdout_tail


class AmpClientError(Exception):
    """Raised on fatal subprocess/parse errors that are *not* drift.

    Carries truncated stderr/stdout for diagnostics. Distinct from
    :class:`AmpStreamMissingInitError`, which is reserved for the
    missing-init drift case (so callers can branch on intent).
    """

    def __init__(
        self,
        message: str,
        *,
        stderr_tail: str = "",
        stdout_tail: str = "",
    ) -> None:
        super().__init__(message)
        self.stderr_tail = stderr_tail
        self.stdout_tail = stdout_tail


@dataclass
class _ClientState:
    """Internal mutable state collected during one ``AmpClient`` lifecycle."""

    proc: asyncio.subprocess.Process | None = None
    session_id: str | None = None
    init_seen: bool = False
    closed: bool = False
    stderr_buf: bytearray = field(default_factory=bytearray)
    stdout_tail: bytearray = field(default_factory=bytearray)
    stderr_task: asyncio.Task[None] | None = None
    lock_event_task: asyncio.Task[None] | None = None
    lock_event_offset: int = 0
    tee_path: Path | None = None
    tee_file: Any = None  # binary file handle; typed Any to avoid IO[...] noise
    handoff_tool_follow: dict[str, bool] = field(default_factory=dict)
    handoff_thread_id: str | None = None
    query_prompt: str | None = None


class AmpClient:
    """SDKClientProtocol implementation for ``amp --execute --stream-json``.

    Lifecycle:
      * ``async with AmpClient(options) as client:`` opens the tee file
        for append.
      * ``await client.query(prompt)`` spawns the subprocess, writes the
        prompt to stdin, and closes stdin.
      * ``async for msg in client.receive_response():`` parses stream-json
        and yields synthetic ``AssistantMessage`` / ``ResultMessage``
        instances. Tees every event line inline, before yielding, so a
        crash mid-stream does not lose evidence already on disk.
      * ``__aexit__`` terminates the subprocess (SIGTERM → grace →
        SIGKILL), cancels the stderr collector task, and flushes/closes
        the tee file.
    """

    def __init__(self, options: AmpClientOptions) -> None:
        self._options = options
        self._state = _ClientState()
        # If options.thread_id is set, the caller already configured a
        # resume target (e.g. ``AmpRuntimeBuilder.with_resume`` was used).
        # ``with_resume`` overrides this on the next ``query()``.
        self._resume_thread_id: str | None = options.thread_id

    # ------------------------------------------------------------------
    # Properties used by AmpAgentProvider / orchestrator
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str | None:
        """Amp's ``T-...`` thread id, captured from the first ``system`` event."""
        return self._state.session_id

    @property
    def thread_id(self) -> str | None:
        """Alias for :attr:`session_id` (Amp uses thread = session id)."""
        return self._state.session_id

    def get_stderr(self) -> str:
        """Return the bounded stderr ring buffer as a UTF-8-decoded string."""
        return self._state.stderr_buf.decode(errors="replace")

    def is_auth_error(self) -> bool:
        """Heuristic auth-error classification on captured stderr text."""
        text = self.get_stderr().lower()
        return any(marker.lower() in text for marker in _AUTH_ERROR_MARKERS)

    def with_resume(self, thread_id: str) -> Self:
        """Configure the next ``query()`` to continue an existing thread.

        Stores ``thread_id``; the next ``query()`` modifies argv per
        :data:`AmpClientOptions.resume_strategy`. Returns ``self`` for
        fluent chaining.
        """
        self._resume_thread_id = thread_id
        return self

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> Self:
        log_path = self._options.log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if self._options.lock_event_log_path is not None:
            self._options.lock_event_log_path.parent.mkdir(parents=True, exist_ok=True)
            if self._state.lock_event_offset == 0:
                self._options.lock_event_log_path.unlink(missing_ok=True)
                self._options.lock_event_log_path.touch()
            self._state.lock_event_task = asyncio.create_task(
                self._tail_lock_events(self._options.lock_event_log_path)
            )
        # Append-mode is correct for both pending (new file) and resume
        # (existing thread file) — the cross-invocation evidence guarantee
        # depends on this being append-only.
        self._state.tee_file = log_path.open("ab")
        self._state.tee_path = log_path
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.disconnect()

    # ------------------------------------------------------------------
    # Subprocess control
    # ------------------------------------------------------------------

    async def query(self, prompt: str, session_id: str | None = None) -> None:
        """Spawn ``amp`` and write ``prompt`` to its stdin verbatim.

        ``session_id`` (from :class:`SDKClientProtocol`) is treated as a
        resume thread id when no prior :meth:`with_resume` was called;
        :meth:`with_resume` takes precedence when both are set.
        """
        if session_id is not None and self._resume_thread_id is None:
            self._resume_thread_id = session_id
        self._state.query_prompt = prompt

        if self._state.proc is not None:
            raise AmpClientError(
                "AmpClient.query() called twice on the same client; "
                "amp --execute is one-shot — open a new client for each query."
            )

        argv = self._build_argv()

        # ``start_new_session`` puts ``amp`` in its own process group so
        # SIGTERM/SIGKILL during cleanup also reaches its children
        # (Bun/plugin/MCP server processes).
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=_STREAM_JSON_READER_LIMIT,
            cwd=str(self._options.cwd),
            env=dict(self._options.env),
            start_new_session=sys.platform != "win32",
        )
        self._state.proc = proc

        if proc.stdin is None:
            raise AmpClientError("amp subprocess opened without stdin pipe")
        try:
            proc.stdin.write(prompt.encode())
            await proc.stdin.drain()
        finally:
            try:
                proc.stdin.close()
            except (OSError, BrokenPipeError):
                # amp may have already exited (e.g. immediate auth fail).
                pass

        if proc.stderr is not None:
            self._state.stderr_task = asyncio.create_task(
                self._collect_stderr(proc.stderr)
            )

    def _build_argv(self) -> list[str]:
        """Assemble the spawn argv, applying the resume strategy if set."""
        base = list(self._options.argv)
        if self._resume_thread_id is None:
            return base
        strategy = self._options.resume_strategy
        if strategy == "thread-id-flag":
            if "--thread-id" in base:
                idx = base.index("--thread-id")
                if idx + 1 < len(base):
                    base[idx + 1] = self._resume_thread_id
                    return base
            return [*base, "--thread-id", self._resume_thread_id]
        if strategy == "threads-continue":
            if not base or base[0] != "amp":
                logger.warning(
                    "AmpClient: 'threads-continue' resume strategy expects "
                    "argv[0] == 'amp'; got %r — leaving argv unchanged.",
                    base[0] if base else None,
                )
                return base
            # Keep ``--execute`` and ``--stream-json`` on the tail: the
            # current Amp CLI requires ``--execute`` to accept
            # ``--stream-json`` even under the ``threads continue``
            # subcommand, and dropping it produced an immediate
            # subprocess exit on every gate retry.
            return ["amp", "threads", "continue", self._resume_thread_id, *base[1:]]
        # "fallback": preserve argv; caller carries context via prompt.
        logger.warning(
            "AmpClient: resume_strategy='fallback' for thread_id=%s — "
            "running fresh thread; context preservation is the caller's "
            "responsibility (prompt accumulation).",
            self._resume_thread_id,
        )
        return base

    async def disconnect(self) -> None:
        """Idempotently terminate the subprocess and close the tee file."""
        if self._state.closed:
            return
        self._state.closed = True

        proc = self._state.proc
        if proc is not None and proc.returncode is None:
            await self._terminate(proc)

        stderr_task = self._state.stderr_task
        if stderr_task is not None and not stderr_task.done():
            stderr_task.cancel()
            try:
                await stderr_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.debug("AmpClient stderr collector raised on shutdown: %s", exc)

        lock_event_task = self._state.lock_event_task
        if lock_event_task is not None and not lock_event_task.done():
            lock_event_task.cancel()
            try:
                await lock_event_task
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                logger.debug("AmpClient lock-event tailer raised on shutdown: %s", exc)
        await self._drain_configured_lock_events()

        tee_file = self._state.tee_file
        if tee_file is not None:
            try:
                tee_file.flush()
            except OSError:
                pass
            try:
                tee_file.close()
            except OSError:
                pass
            self._state.tee_file = None

    async def _terminate(self, proc: asyncio.subprocess.Process) -> None:
        """SIGTERM → grace → SIGKILL. Mirrors the existing subprocess pattern."""
        use_pgroup = sys.platform != "win32"
        try:
            if use_pgroup:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
            else:
                proc.terminate()
            try:
                await asyncio.wait_for(
                    proc.wait(), timeout=self._options.kill_grace_seconds
                )
            except TimeoutError:
                if use_pgroup:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
                elif proc.returncode is None:
                    proc.kill()
                if proc.returncode is None:
                    await proc.wait()
        except ProcessLookupError:
            # Process exited on its own between checks — fine.
            pass

    # ------------------------------------------------------------------
    # Stream parsing
    # ------------------------------------------------------------------

    async def receive_response(self) -> AsyncIterator[object]:
        """Yield synthetic messages parsed from ``amp`` stdout.

        Iterates stdout line-by-line. Every line is tee'd verbatim to the
        log path *before* the yield so a crash mid-stream cannot lose
        evidence already on disk. JSON parse errors and non-system events
        arriving before ``system(init)`` raise so the caller can fail the
        issue with diagnostics.
        """
        proc = self._state.proc
        if proc is None:
            raise AmpClientError(
                "AmpClient.receive_response called before query() — "
                "no subprocess was spawned."
            )
        stdout = proc.stdout
        if stdout is None:
            raise AmpClientError("amp subprocess opened without stdout pipe")

        try:
            while True:
                line = await stdout.readline()
                if not line:
                    break
                self._tee_line(line)
                self._record_stdout_tail(line)
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    event = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise AmpClientError(
                        f"AmpClient could not parse stream-json line: {exc}",
                        stderr_tail=self.get_stderr(),
                        stdout_tail=self._stdout_tail_text(),
                    ) from exc
                if not isinstance(event, dict):
                    logger.warning(
                        "AmpClient: stream-json line is not an object: %r",
                        type(event).__name__,
                    )
                    continue
                async for msg in self._dispatch(event):
                    yield msg
        finally:
            # Don't block the iterator on subprocess wait; __aexit__ owns
            # full cleanup. We only opportunistically reap if exit is
            # already imminent so the returncode is observable to callers.
            await self._drain_configured_lock_events()
            await self._reap_if_done()

    async def _dispatch(self, event: dict[str, Any]) -> AsyncIterator[object]:
        """Map one parsed event to zero or more synthetic messages."""
        event_type = event.get("type")
        if event_type == "system":
            self._handle_system(event)
            return
        # All non-system events require system(init) to have arrived first.
        if not self._state.init_seen:
            self._raise_missing_init()
        if event_type == "assistant":
            msg = self._build_assistant(event)
            if msg is not None:
                yield msg
        elif event_type == "user":
            msg = self._build_tool_result(event)
            if msg is not None:
                yield msg
        elif event_type == "result":
            await self._drain_configured_lock_events()
            if self._is_resume_context_limit_result(event):
                yield await self._handoff_after_resume_context_limit(event)
                return
            result_msg = self._build_result(event)
            if self._state.handoff_thread_id is not None:
                result = await self._materialize_handoff_child(
                    self._state.handoff_thread_id
                )
                self._state.session_id = self._state.handoff_thread_id
                result_msg = ResultMessage(
                    session_id=self._state.handoff_thread_id,
                    result=result if result is not None else result_msg.result,
                )
            yield result_msg
        else:
            # Unknown event types are warn-level only (plan L651): they
            # may be future Amp additions; do not crash.
            logger.warning(
                "AmpClient: unknown stream-json event type %r — skipping.",
                event_type,
            )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _handle_system(self, event: dict[str, Any]) -> None:
        """Capture session_id on first ``system`` event; finalize tee path."""
        sid = event.get("session_id")
        is_first = not self._state.init_seen
        # Treat any system event as init for missing-init detection;
        # capture session_id when it's present (Amp emits it on init).
        self._state.init_seen = True
        if sid is not None and self._state.session_id is None:
            self._state.session_id = str(sid)
        if is_first and self._state.session_id is not None:
            self._maybe_finalize_tee_path()

    def _build_assistant(self, event: dict[str, Any]) -> AssistantMessage | None:
        message = event.get("message") or {}
        content = message.get("content") or []
        blocks: list[object] = []
        for raw in content:
            if not isinstance(raw, dict):
                continue
            btype = raw.get("type")
            if btype == "text":
                blocks.append(TextBlock(text=str(raw.get("text", ""))))
            elif btype == "tool_use":
                tool_id = str(raw.get("id", ""))
                name = str(raw.get("name", ""))
                tool_input = dict(raw.get("input") or {})
                if name.lower() == "handoff" and tool_id:
                    self._state.handoff_tool_follow[tool_id] = bool(
                        tool_input.get("follow", False)
                    )
                blocks.append(
                    ToolUseBlock(
                        id=tool_id,
                        name=name,
                        input=tool_input,
                    )
                )
            elif btype in ("thinking", "redacted_thinking"):
                # MVP: thinking blocks are tee'd for diagnostics but not
                # surfaced to the pipeline. Plan L404.
                continue
            else:
                logger.warning(
                    "AmpClient: unknown assistant block type %r — skipping.",
                    btype,
                )
        if not blocks:
            return None
        return AssistantMessage(content=blocks)

    def _build_tool_result(self, event: dict[str, Any]) -> AssistantMessage | None:
        message = event.get("message") or {}
        content = message.get("content") or []
        blocks: list[object] = []
        for raw in content:
            if not isinstance(raw, dict):
                continue
            if raw.get("type") != "tool_result":
                continue
            tool_use_id = str(raw.get("tool_use_id", ""))
            handoff_thread_id = self._extract_handoff_thread_id(raw.get("content"))
            if (
                handoff_thread_id is not None
                and self._state.handoff_tool_follow.get(tool_use_id, False)
            ):
                if (
                    self._state.handoff_thread_id is not None
                    and self._state.handoff_thread_id != handoff_thread_id
                ):
                    logger.warning(
                        "AmpClient: multiple followed handoffs observed; using latest child thread %s",
                        handoff_thread_id,
                    )
                self._state.handoff_thread_id = handoff_thread_id
            blocks.append(
                ToolResultBlock(
                    tool_use_id=tool_use_id,
                    is_error=bool(raw.get("is_error", False)),
                    content=raw.get("content"),
                )
            )
        if not blocks:
            return None
        return AssistantMessage(content=blocks)

    def _build_result(self, event: dict[str, Any]) -> ResultMessage:
        sid = event.get("session_id") or self._state.session_id or ""
        # Amp emits result/subtype values: success | error_during_execution |
        # error_max_turns. Surface the explicit ``result`` field if present,
        # otherwise fall back to the subtype so the orchestrator can branch
        # on a stable string.
        result: object = event.get("result")
        if result is None:
            result = event.get("subtype")
        return ResultMessage(session_id=str(sid), result=result)

    def _is_resume_context_limit_result(self, event: dict[str, Any]) -> bool:
        """Return true for Amp's resumed-thread context-limit no-op result."""
        if self._resume_thread_id is None:
            return False
        if event.get("type") != "result":
            return False
        if event.get("subtype") != "error_during_execution":
            return False
        text_parts = [
            event.get("error"),
            event.get("result"),
            event.get("message"),
            event.get("details"),
        ]
        text = " ".join(str(part) for part in text_parts if part is not None)
        return "context limit reached" in text.lower()

    async def _handoff_after_resume_context_limit(
        self,
        event: dict[str, Any],
    ) -> ResultMessage:
        """Create and materialize a fresh Amp child thread after resume saturation."""
        del event
        parent_thread_id = self._resume_thread_id or self._state.session_id
        goal = self._state.query_prompt
        if not parent_thread_id:
            raise AmpClientError(
                "Amp resume hit context limit, but no parent thread id is available",
                stderr_tail=self.get_stderr(),
                stdout_tail=self._stdout_tail_text(),
            )
        if goal is None:
            raise AmpClientError(
                f"Amp resume hit context limit for {parent_thread_id}, but no query prompt is available for handoff",
                stderr_tail=self.get_stderr(),
                stdout_tail=self._stdout_tail_text(),
            )

        child_thread_id = await self._run_cli_handoff(parent_thread_id, goal)
        result = await self._materialize_handoff_child(child_thread_id)
        self._state.session_id = child_thread_id
        return ResultMessage(
            session_id=child_thread_id,
            result=result if result is not None else "success",
        )

    async def _run_cli_handoff(self, parent_thread_id: str, goal: str) -> str:
        """Run ``amp threads handoff`` and return the created child thread id."""
        amp_executable = self._options.argv[0] if self._options.argv else "amp"
        proc = await asyncio.create_subprocess_exec(
            amp_executable,
            "threads",
            "handoff",
            parent_thread_id,
            "--goal",
            goal,
            "--print",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._options.cwd),
            env=dict(self._options.env),
        )
        stdout, stderr = await proc.communicate()
        stdout_text = stdout.decode(errors="replace")
        stderr_text = stderr.decode(errors="replace")
        if proc.returncode != 0:
            raise AmpClientError(
                f"Amp handoff failed for parent thread {parent_thread_id} with exit code {proc.returncode}",
                stderr_tail=stderr_text[-_STDERR_BUFFER_LIMIT:],
                stdout_tail=stdout_text[-_STDOUT_TAIL_LIMIT:],
            )

        child_thread_id = self._parse_handoff_child_thread_id(stdout_text)
        if child_thread_id is None:
            raise AmpClientError(
                f"Amp handoff for parent thread {parent_thread_id} did not report a child thread id",
                stderr_tail=stderr_text[-_STDERR_BUFFER_LIMIT:],
                stdout_tail=stdout_text[-_STDOUT_TAIL_LIMIT:],
            )
        if child_thread_id == parent_thread_id:
            raise AmpClientError(
                f"Amp handoff for parent thread {parent_thread_id} returned the parent id as the child",
                stderr_tail=stderr_text[-_STDERR_BUFFER_LIMIT:],
                stdout_tail=stdout_text[-_STDOUT_TAIL_LIMIT:],
            )
        return child_thread_id

    def _parse_handoff_child_thread_id(self, stdout_text: str) -> str | None:
        """Parse an Amp thread id from ``amp threads handoff --print`` output."""
        try:
            payload = json.loads(stdout_text)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            for key in ("newThreadID", "threadID", "thread_id", "id"):
                value = payload.get(key)
                if isinstance(value, str) and value.startswith("T-"):
                    return value

        matches = re.findall(r"T-[A-Za-z0-9][A-Za-z0-9_-]*", stdout_text)
        if matches:
            return matches[-1]
        return None

    def _extract_handoff_thread_id(self, content: object) -> str | None:
        """Return Amp handoff child thread id from a handoff tool result."""
        payload = content
        if isinstance(content, str):
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                return None
        if not isinstance(payload, dict):
            return None
        payload_dict = cast("dict[str, Any]", payload)
        thread_id = payload_dict.get("newThreadID")
        if isinstance(thread_id, str) and thread_id.startswith("T-"):
            return thread_id
        return None

    async def _materialize_handoff_child(self, thread_id: str) -> object | None:
        """Export a followed handoff child into the thread-keyed JSONL log.

        Amp's parent ``--stream-json`` process reports only the handoff tool
        result. The followed child thread runs separately, so mala must fetch
        the child thread before returning its session id to the orchestrator.
        """
        deadline = time.monotonic() + _HANDOFF_EXPORT_TIMEOUT_SECONDS
        last_error: str | None = None
        while time.monotonic() < deadline:
            try:
                exported = await self._export_handoff_thread(thread_id)
                events, result = self._events_from_thread_export(exported, thread_id)
                self._write_handoff_thread_log(thread_id, events)
                if result is not None:
                    return result
            except AmpClientError as exc:
                last_error = str(exc)
            await asyncio.sleep(_HANDOFF_EXPORT_POLL_SECONDS)
        detail = f": {last_error}" if last_error else ""
        raise AmpClientError(
            f"Timed out waiting for Amp handoff child thread {thread_id}{detail}",
            stderr_tail=self.get_stderr(),
            stdout_tail=self._stdout_tail_text(),
        )

    async def _export_handoff_thread(self, thread_id: str) -> dict[str, Any]:
        """Run ``amp threads export`` for a handoff child thread."""
        amp_executable = self._options.argv[0] if self._options.argv else "amp"
        proc = await asyncio.create_subprocess_exec(
            amp_executable,
            "threads",
            "export",
            thread_id,
            "--no-ide",
            "--no-notifications",
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._options.cwd),
            env=dict(self._options.env),
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise AmpClientError(
                f"Amp handoff child export failed for {thread_id} with exit code {proc.returncode}",
                stderr_tail=stderr.decode(errors="replace")[-_STDERR_BUFFER_LIMIT:],
                stdout_tail=stdout.decode(errors="replace")[-_STDOUT_TAIL_LIMIT:],
            )
        try:
            data = json.loads(stdout.decode())
        except json.JSONDecodeError as exc:
            raise AmpClientError(
                f"Amp handoff child export for {thread_id} was not JSON: {exc}",
                stderr_tail=stderr.decode(errors="replace")[-_STDERR_BUFFER_LIMIT:],
                stdout_tail=stdout.decode(errors="replace")[-_STDOUT_TAIL_LIMIT:],
            ) from exc
        if not isinstance(data, dict):
            raise AmpClientError(f"Amp handoff child export for {thread_id} was not an object")
        return data

    def _events_from_thread_export(
        self, exported: dict[str, Any], thread_id: str
    ) -> tuple[list[dict[str, Any]], object | None]:
        """Normalize ``amp threads export`` JSON into stream-json-shaped events."""
        meta = exported.get("meta")
        agent_mode = exported.get("agentMode") or (
            meta.get("agentMode") if isinstance(meta, dict) else None
        )
        events: list[dict[str, Any]] = [
            {
                "type": "system",
                "subtype": "init",
                "cwd": str(self._options.cwd),
                "session_id": thread_id,
                "tools": [],
                "mcp_servers": [],
                "agent_mode": agent_mode,
            }
        ]
        result: object | None = None
        messages = exported.get("messages")
        if not isinstance(messages, list):
            error_result = self._export_terminal_error_result(exported)
            if error_result is not None:
                events.append(
                    {
                        "type": "result",
                        "subtype": "error_during_execution",
                        "is_error": True,
                        "result": error_result,
                        "session_id": thread_id,
                    }
                )
                return events, error_result
            return events, None

        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role not in {"assistant", "user"}:
                continue
            content = self._export_content_to_stream_blocks(
                message.get("content"), role=str(role)
            )
            if not content:
                continue
            stream_message: dict[str, Any] = {"role": role, "content": content}
            if role == "assistant":
                stream_message["type"] = "message"
                state = message.get("state")
                if isinstance(state, dict) and isinstance(state.get("stopReason"), str):
                    stream_message["stop_reason"] = state["stopReason"]
            events.append(
                {
                    "type": role,
                    "message": stream_message,
                    "parent_tool_use_id": None,
                    "session_id": thread_id,
                }
            )
            if role == "assistant" and self._export_assistant_message_is_complete(message):
                text = "".join(
                    str(block.get("text", ""))
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
                result = text or "success"

        if result is not None:
            events.append(
                {
                    "type": "result",
                    "subtype": "success",
                    "is_error": False,
                    "result": result,
                    "session_id": thread_id,
                }
            )
        else:
            error_result = self._export_terminal_error_result(exported)
            if error_result is not None:
                events.append(
                    {
                        "type": "result",
                        "subtype": "error_during_execution",
                        "is_error": True,
                        "result": error_result,
                        "session_id": thread_id,
                    }
                )
                result = error_result
        return events, result

    def _export_terminal_error_result(self, exported: dict[str, Any]) -> object | None:
        """Return an exported child-thread terminal error payload, if explicit."""
        if self._export_status_is_error(exported):
            return self._export_error_payload(exported)

        state = exported.get("state")
        if isinstance(state, dict) and self._export_status_is_error(state):
            return self._export_error_payload(state)

        result = exported.get("result")
        if isinstance(result, dict) and self._export_status_is_error(result):
            return self._export_error_payload(result)
        return None

    def _export_status_is_error(self, payload: dict[str, Any]) -> bool:
        if payload.get("is_error") is True or payload.get("isError") is True:
            return True
        for key in ("status", "type", "subtype", "stopReason", "stop_reason"):
            value = payload.get(key)
            if isinstance(value, str) and value.lower() in {
                "error",
                "failed",
                "failure",
                "error_during_execution",
            }:
                return True
        return False

    def _export_error_payload(self, payload: dict[str, Any]) -> object:
        for key in ("error", "result", "message", "details"):
            value = payload.get(key)
            if value is not None:
                return value
        return "error_during_execution"

    def _export_content_to_stream_blocks(
        self, content: object, *, role: str
    ) -> list[dict[str, Any]]:
        if not isinstance(content, list):
            return []
        blocks: list[dict[str, Any]] = []
        for raw in content:
            if not isinstance(raw, dict):
                continue
            raw_block = cast("dict[str, Any]", raw)
            block_type = raw_block.get("type")
            if block_type == "text":
                blocks.append({"type": "text", "text": str(raw_block.get("text", ""))})
            elif role == "assistant" and block_type == "tool_use":
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(raw_block.get("id", "")),
                        "name": str(raw_block.get("name", "")),
                        "input": dict(raw_block.get("input") or {}),
                    }
                )
            elif role == "user" and block_type == "tool_result":
                tool_use_id = (
                    raw_block.get("tool_use_id") or raw_block.get("toolUseID") or ""
                )
                run = raw_block.get("run")
                content_value: object = raw_block.get("content", "")
                is_error = bool(raw_block.get("is_error", False))
                if isinstance(run, dict):
                    if "result" in run:
                        content_value = run["result"]
                    elif "error" in run:
                        content_value = run["error"]
                    is_error = run.get("status") not in {None, "done", "success"}
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": str(tool_use_id),
                        "content": content_value,
                        "is_error": is_error,
                    }
                )
        return blocks

    def _export_assistant_message_is_complete(self, message: dict[str, Any]) -> bool:
        state = message.get("state")
        return (
            isinstance(state, dict)
            and state.get("type") == "complete"
            and state.get("stopReason") != "tool_use"
        )

    def _write_handoff_thread_log(
        self, thread_id: str, events: list[dict[str, Any]]
    ) -> None:
        from src.infra.clients.amp_runtime import AMP_SESSIONS_DIR

        log_path = AMP_SESSIONS_DIR / f"{thread_id}.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = log_path.with_name(f".{log_path.name}.tmp-{os.getpid()}")
        with tmp_path.open("w", encoding="utf-8") as fh:
            for event in events:
                fh.write(json.dumps(event, separators=(",", ":")))
                fh.write("\n")
        os.replace(tmp_path, log_path)

    # ------------------------------------------------------------------
    # Tee management
    # ------------------------------------------------------------------

    def _tee_line(self, line: bytes) -> None:
        """Write one stream-json line verbatim to the tee file."""
        tee_file = self._state.tee_file
        if tee_file is None:
            return
        try:
            tee_file.write(line)
            if not line.endswith(b"\n"):
                tee_file.write(b"\n")
            tee_file.flush()
        except OSError as exc:
            logger.warning("AmpClient tee write failed: %s", exc)

    def _maybe_finalize_tee_path(self) -> None:
        """Atomically rename pending tee file to ``{thread_id}.jsonl`` on init.

        Three cases:
          1. Already at the target path (resume opened it directly): no-op.
          2. Pending path, target does not exist: ``os.rename`` (atomic).
          3. Pending path, target already exists (resume w/ pending start):
             append pending bytes to target, then unlink pending. Reopen
             our handle on target so subsequent writes go to the right file.
        """
        sid = self._state.session_id
        if sid is None or self._state.tee_path is None:
            return
        target = self._state.tee_path.parent / f"{sid}.jsonl"
        if self._state.tee_path == target:
            return

        # Flush pending bytes before rename/merge.
        tee_file = self._state.tee_file
        if tee_file is not None:
            try:
                tee_file.flush()
            except OSError:
                pass

        if target.exists():
            # Merge pending → target, then unlink pending.
            try:
                with self._state.tee_path.open("rb") as src, target.open("ab") as dst:
                    while True:
                        chunk = src.read(64 * 1024)
                        if not chunk:
                            break
                        dst.write(chunk)
                if tee_file is not None:
                    try:
                        tee_file.close()
                    except OSError:
                        pass
                self._state.tee_path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(
                    "AmpClient could not merge pending tee into existing "
                    "thread file %s: %s",
                    target,
                    exc,
                )
                return
            self._state.tee_path = target
            self._state.tee_file = target.open("ab")
        else:
            if tee_file is not None:
                try:
                    tee_file.close()
                except OSError:
                    pass
            try:
                os.rename(self._state.tee_path, target)
            except OSError as exc:
                logger.warning(
                    "AmpClient could not rename pending tee %s -> %s: %s",
                    self._state.tee_path,
                    target,
                    exc,
                )
                # Reopen the original so further writes are not lost.
                self._state.tee_file = self._state.tee_path.open("ab")
                return
            self._state.tee_path = target
            self._state.tee_file = target.open("ab")

    # ------------------------------------------------------------------
    # Diagnostics + helpers
    # ------------------------------------------------------------------

    async def _collect_stderr(self, stream: asyncio.StreamReader) -> None:
        try:
            while True:
                chunk = await stream.read(1024)
                if not chunk:
                    break
                self._state.stderr_buf.extend(chunk)
                excess = len(self._state.stderr_buf) - _STDERR_BUFFER_LIMIT
                if excess > 0:
                    del self._state.stderr_buf[:excess]
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.debug("AmpClient stderr collector ended with %s", exc)

    async def _tail_lock_events(self, path: Path) -> None:
        """Tail Amp MCP lock-event JSONL and feed the deadlock monitor."""
        try:
            while True:
                await self._drain_lock_events(path)
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            raise

    async def _drain_configured_lock_events(self) -> None:
        if self._options.lock_event_log_path is not None:
            await self._drain_lock_events(self._options.lock_event_log_path)

    async def _drain_lock_events(self, path: Path) -> None:
        try:
            with path.open("r", encoding="utf-8") as fh:
                fh.seek(self._state.lock_event_offset)
                while line := fh.readline():
                    if not line.endswith("\n"):
                        break
                    self._state.lock_event_offset = fh.tell()
                    await self._handle_lock_event_line(line)
        except FileNotFoundError:
            pass

    async def _handle_lock_event_line(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            return
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            logger.warning("AmpClient: invalid lock-event JSONL line: %s", stripped[:200])
            return
        if not isinstance(data, dict):
            return
        try:
            from src.core.models import LockEvent, LockEventType

            event = LockEvent(
                event_type=LockEventType(str(data["event_type"])),
                agent_id=str(data["agent_id"]),
                lock_path=str(data["lock_path"]),
                timestamp=float(data["timestamp"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("AmpClient: malformed lock-event payload %r: %s", data, exc)
            return

        callback = self._options.lock_event_callback
        if callback is None:
            return
        try:
            result = callback(event)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception(
                "AmpClient: lock-event callback failed for agent=%s path=%s",
                event.agent_id,
                event.lock_path,
            )

    def _record_stdout_tail(self, line: bytes) -> None:
        self._state.stdout_tail.extend(line)
        excess = len(self._state.stdout_tail) - _STDOUT_TAIL_LIMIT
        if excess > 0:
            del self._state.stdout_tail[:excess]

    def _stdout_tail_text(self) -> str:
        return self._state.stdout_tail.decode(errors="replace")

    def _raise_missing_init(self) -> None:
        """Close pending file (preserve on disk) and raise the drift error."""
        pending = self._state.tee_path or self._options.log_path
        tee_file = self._state.tee_file
        if tee_file is not None:
            try:
                tee_file.flush()
            except OSError:
                pass
            try:
                tee_file.close()
            except OSError:
                pass
            self._state.tee_file = None
        raise AmpStreamMissingInitError(
            "AmpClient received non-system event before system(init); "
            "stream-json drift or upstream Amp bug. Pending tee file "
            "preserved for diagnostics.",
            pending_path=pending,
            stderr_tail=self.get_stderr()[-_STDOUT_TAIL_LIMIT:],
            stdout_tail=self._stdout_tail_text()[-_STDOUT_TAIL_LIMIT:],
        )

    async def _reap_if_done(self) -> None:
        proc = self._state.proc
        if proc is None or proc.returncode is not None:
            return
        try:
            await asyncio.wait_for(proc.wait(), timeout=0.05)
        except TimeoutError:
            # Still running; let __aexit__ handle final cleanup.
            pass

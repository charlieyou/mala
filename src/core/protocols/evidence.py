"""Evidence provider protocol for cross-coder validation evidence.

Replaces the prior ``LogProvider`` abstraction (``src/core/protocols/log.py``,
deleted in this change). Defines the typed :class:`EvidenceProvider`
protocol that providers (Claude, Amp, and Codex in Phase F) implement
directly:

  * Claude reads JSONL session logs from
    ``~/.claude/projects/{encoded-path}/`` (per-session files; offset
    scopes to the current attempt).
  * Amp reads tee'd JSONL from ``~/.config/mala/amp-sessions/{thread_id}.jsonl``
    (per-thread, append-only across resumes).
  * Codex (Phase F) reads via ``codex_app_server.Thread.read``.

Method names follow the Phase A5 spec
(``plans/2026-05-07-codex-provider-plan.md#L527-L538``):

  * :meth:`EvidenceProvider.iter_session_events` is the per-attempt read
    used for resolution-marker parsing; offset scoping is honored.
  * :meth:`EvidenceProvider.iter_thread_evidence` is the cross-attempt
    read used by validation gates; providers whose log files span
    multiple invocations of the same thread (e.g. Amp's tee) read from
    byte 0 so cross-resume Bash ``tool_use`` evidence remains visible.

The supporting methods (:meth:`get_log_path`, :meth:`get_end_offset`, the
extractors) match the prior LogProvider surface verbatim so this rename
keeps validation behavior unchanged (issue ``mala-b18dd.5.1`` AC#10).
Phase A7 (T007, ``mala-b18dd.5.3``) extends this protocol with
:meth:`EvidenceProvider.wait_for_session_ready`, which decouples
``Effect.WAIT_FOR_LOG`` from filesystem assumptions so Codex (Phase H) can
return immediately while Claude/Amp keep polling JSONL existence.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@runtime_checkable
class JsonlEntryProtocol(Protocol):
    """Protocol for parsed JSONL log entries with byte offset tracking.

    Matches the shape of :class:`src.infra.io.session_log_parser.JsonlEntry`
    for structural typing.
    """

    data: dict[str, Any]
    """The parsed JSON object from this line."""

    entry: object | None
    """The typed LogEntry if successfully parsed, None otherwise."""

    line_len: int
    """Length of the raw line in bytes (for offset tracking)."""

    offset: int
    """Byte offset where this line started in the file."""


@runtime_checkable
class EvidenceProvider(Protocol):
    """Cross-coder evidence surface consumed by validation gates.

    Replaces ``src.core.protocols.log.LogProvider``. The canonical
    Claude implementation is :class:`FileSystemLogProvider` (reads JSONL
    from the Claude SDK's ``~/.claude/projects/{encoded-path}/``); the
    Amp implementation is :class:`AmpLogProvider` (reads the per-thread
    tee'd JSONL or a discovered native log directory). Test
    implementations can return in-memory events for isolation.

    Methods:
        get_log_path: Get the filesystem path for a session log.
        iter_session_events: Per-attempt read; offset honored.
        iter_thread_evidence: Cross-attempt evidence read for validation
            gates; offset ignored on multi-invocation log files.
        get_end_offset: Lightweight byte-offset reader for retry scoping.
        wait_for_session_ready: Async readiness gate consumed by the
            ``Effect.WAIT_FOR_LOG`` handler; providers implement the
            wait however their backend exposes session state.
        extract_bash_commands / extract_tool_results /
        extract_assistant_text_blocks / extract_tool_result_content:
        Typed extractors over a single parsed entry.
    """

    def get_log_path(self, repo_path: Path, session_id: str) -> Path:
        """Get the filesystem path for a session's log file.

        Args:
            repo_path: Path to the repository the session was run in.
            session_id: Coder session/thread id.

        Returns:
            Path to the JSONL log file. The path may or may not exist
            yet.
        """
        ...

    def iter_session_events(
        self, log_path: Path, offset: int = 0
    ) -> Iterator[JsonlEntryProtocol]:
        """Iterate parsed JSONL entries for the current attempt.

        Reads the file starting from ``offset`` and yields structured
        entries. Used by resolution-marker parsing where per-attempt
        scoping is required so a stale ``ISSUE_*`` marker emitted in an
        earlier invocation does not override the current attempt's
        decision.

        Args:
            log_path: Path to the JSONL log file.
            offset: Byte offset to start reading from (default 0).

        Yields:
            JsonlEntryProtocol objects for each successfully parsed JSON
            line.

        Note:
            - Lines that fail UTF-8 decoding are silently skipped.
            - Empty lines are silently skipped.
            - Lines that fail JSON parsing are silently skipped.
            - If the file doesn't exist, yields nothing.
        """
        ...

    def iter_thread_evidence(
        self, log_path: Path, offset: int = 0
    ) -> Iterator[JsonlEntryProtocol]:
        """Iterate entries for cross-attempt evidence-presence reads.

        Behaves like :meth:`iter_session_events` for providers whose log
        files are per-session (e.g.
        :class:`FileSystemLogProvider`): the supplied ``offset`` scopes
        the read to the current attempt window.

        Providers whose log files span multiple invocations of the same
        thread (e.g. :class:`AmpLogProvider`, where every resume appends
        to the same ``{thread_id}.jsonl``) MUST ignore ``offset`` and
        read from byte 0 so cross-invocation validation evidence (Bash
        ``tool_use`` events for lint / test / typecheck) remains
        observable to validation gates after a resume — even when the
        caller has advanced ``retry_state.log_offset`` past invocation 1.

        This method is intentionally distinct from
        :meth:`iter_session_events` because resolution-marker parsing on
        the same log file *does* need offset scoping (so a stale
        resolution from an earlier invocation does not override the
        current attempt's decision).

        Args:
            log_path: Path to the JSONL log file.
            offset: Byte offset to start reading from. Honored on
                providers with per-session log files; ignored on
                providers with per-thread (multi-invocation) log files.

        Yields:
            JsonlEntryProtocol objects for each successfully parsed JSON
            line.
        """
        ...

    async def wait_for_session_ready(
        self,
        log_path: Path,
        *,
        timeout: float,
        poll_interval: float = 0.5,
    ) -> None:
        """Block until the provider can serve session evidence.

        Consumed by :class:`AgentSessionRunner._handle_log_waiting` when
        the lifecycle emits ``Effect.WAIT_FOR_LOG``. Replaces the prior
        in-runner filesystem poll so providers whose evidence is not on
        disk (Phase H Codex, which reads via ``Thread.read``) can return
        immediately, while filesystem-backed providers preserve their
        existing polling behavior.

        Args:
            log_path: Path the runner has chosen for this session via
                :meth:`get_log_path`. Filesystem-backed providers poll
                this path for existence; providers whose readiness is
                not file-based ignore the argument.
            timeout: Maximum seconds to wait. Implementations MUST raise
                ``TimeoutError`` when this elapses without readiness.
            poll_interval: Polling cadence for filesystem-backed
                providers; ignored by providers that do not poll.

        Raises:
            TimeoutError: If ``timeout`` elapses before readiness.
        """
        ...

    def get_end_offset(self, log_path: Path, start_offset: int = 0) -> int:
        """Get the byte offset at the end of a log file.

        Lightweight method for getting the current file position. Use
        this when you only need the offset for retry scoping, not the
        parsed entries themselves.

        Args:
            log_path: Path to the JSONL log file.
            start_offset: Byte offset to start from (default 0).

        Returns:
            The byte offset at the end of the file, or ``start_offset``
            if the file doesn't exist or can't be read.
        """
        ...

    def extract_bash_commands(self, entry: JsonlEntryProtocol) -> list[tuple[str, str]]:
        """Extract Bash tool_use commands from an entry.

        Args:
            entry: A JsonlEntryProtocol from :meth:`iter_session_events`
                or :meth:`iter_thread_evidence`.

        Returns:
            List of ``(tool_id, command)`` tuples for Bash tool_use
            blocks. Returns empty list if entry is not an assistant
            message.
        """
        ...

    def extract_tool_results(self, entry: JsonlEntryProtocol) -> list[tuple[str, bool]]:
        """Extract tool_result entries from an entry.

        Args:
            entry: A JsonlEntryProtocol from :meth:`iter_session_events`
                or :meth:`iter_thread_evidence`.

        Returns:
            List of ``(tool_use_id, is_error)`` tuples for tool_result
            blocks. Returns empty list if entry is not a user message.
        """
        ...

    def extract_assistant_text_blocks(self, entry: JsonlEntryProtocol) -> list[str]:
        """Extract text content from assistant message blocks.

        Args:
            entry: A JsonlEntryProtocol from :meth:`iter_session_events`
                or :meth:`iter_thread_evidence`.

        Returns:
            List of text strings from text blocks in assistant messages.
            Returns empty list if entry is not an assistant message.
        """
        ...

    def extract_tool_result_content(
        self, entry: JsonlEntryProtocol
    ) -> list[tuple[str, str]]:
        """Extract textual content from all tool_result blocks.

        Used for marker detection in custom validation commands. Returns
        content from all tool results (not just Bash) since custom
        command markers have a specific format ``[custom:<name>:<status>]``
        that avoids false positives.

        Args:
            entry: A JsonlEntryProtocol from :meth:`iter_session_events`
                or :meth:`iter_thread_evidence`.

        Returns:
            List of ``(tool_use_id, content)`` tuples for all tool_result
            blocks. Content is normalized to string with text extracted
            from content blocks. Returns empty list if entry is not a
            user message.
        """
        ...

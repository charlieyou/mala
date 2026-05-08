"""EvidenceProvider for the Codex coder (Phase F, T013).

Aggregates validation evidence across all invocations of a Codex thread by
reading the per-thread tee'd JSONL written by :class:`CodexClient` at
``~/.config/mala/codex-sessions/{thread_id}.jsonl`` (namespaced separately
from Amp's ``amp-sessions/`` per plan ``L1168`` to prevent thread-id
collision). The tee path is per-thread and append-only: every
``AsyncCodex.thread_start`` / ``thread_resume`` for the same thread (initial
run + resumes triggered by idle timeout, gate failure, or review-issue
retries) appends to the same file, so ``CommandExecutionThreadItem`` events
from earlier invocations remain observable to validation gates regardless of
how many times the thread has been resumed.

Spike outcome (T012, see ``tests/spike/test_codex_thread_read_evidence.py``):
the F1 spike disconfirmed decision #11's native ``Thread.read(include_turns=True)``
path because Codex's app-server hardcodes ``EventPersistenceMode::Limited``
and filters ``ExecCommandEnd`` out of the rollout — leaving
``CommandExecutionThreadItem.aggregated_output = None`` on every read. T013
therefore implements the F3 tee fallback (plan ``L73-L77``) instead of the
native ``Thread.read`` adapter.

Tee event shape
---------------

Each line in ``{thread_id}.jsonl`` is a JSON-serialized Codex
``Notification`` produced by :meth:`AsyncTurnHandle.stream`:

    {"method": "<method-string>", "payload": {<payload-fields>}}

The methods that carry validation-relevant evidence are:

  * ``item/completed`` with ``payload.item.type == "commandExecution"`` —
    a finished bash turn; ``item.command`` is the executed shell line and
    ``item.aggregated_output`` is the captured stdout/stderr.
  * ``item/completed`` with ``payload.item.type == "fileChange"`` — an
    apply_patch turn; ``item.changes`` enumerates the touched files.
  * ``item/completed`` with ``payload.item.type == "mcpToolCall"`` — an
    MCP tool invocation; status drives ``is_error``.
  * ``item/completed`` with ``payload.item.type == "agentMessage"`` — a
    completed assistant message; ``item.text`` carries the final text.

The extractors here mirror :class:`SessionLogParser`'s Claude/Amp surface
so :class:`EvidenceCheck` and other gate consumers parse Codex evidence
identically without provider-specific branches (AC #18 / Phase A5).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any, cast

from src.infra.io.session_log_parser import JsonlEntry
from src.infra.tools.env import USER_CONFIG_DIR

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from src.core.protocols.evidence import JsonlEntryProtocol


logger = logging.getLogger(__name__)


CODEX_SESSIONS_DIR = USER_CONFIG_DIR / "codex-sessions"
"""Directory where mala tees Codex's notification stream, keyed by thread id.

Namespaced separately from :data:`src.infra.clients.amp_runtime.AMP_SESSIONS_DIR`
(``~/.config/mala/amp-sessions/``) per plan ``L1168``; Amp ``T-...`` and
Codex ``thr_...`` ids do not collide today, but per-coder namespacing keeps
the boundary explicit and makes future id-format changes safe.
"""


_BASH_ITEM_TYPE = "commandExecution"
_FILE_CHANGE_TYPE = "fileChange"
_MCP_TOOL_CALL_TYPE = "mcpToolCall"
_AGENT_MESSAGE_TYPE = "agentMessage"
_TOOL_RESULT_ITEM_TYPES = frozenset(
    {_BASH_ITEM_TYPE, _FILE_CHANGE_TYPE, _MCP_TOOL_CALL_TYPE}
)
"""Item types the live :class:`CodexEventAdapter` emits as tool results.

Mirrors :meth:`CodexEventAdapter._handle_item_completed`'s dispatch:
``commandExecution`` / ``fileChange`` / ``mcpToolCall`` produce
``AgentToolResultEvent``; ``agentMessage`` produces only
``AgentTextEvent`` and ``reasoning`` items are dropped per decision #12.
:meth:`CodexEvidenceProvider.extract_tool_results` therefore yields
results ONLY for the three tool types — agent messages would otherwise
pick up the ``status != "completed"`` rule and surface as spurious
errors (the SDK's ``AgentMessageThreadItem`` does not carry a status
field at all).
"""
_COMPLETED_METHOD = "item/completed"
_SUCCESS_STATUS = "completed"
"""The only status the live :class:`CodexEventAdapter` treats as success.

Mirrors :data:`src.infra.clients.codex_event_adapter._TERMINAL_COMMAND_STATUSES`
/ ``_TERMINAL_PATCH_STATUSES`` / ``_TERMINAL_MCP_STATUSES`` semantics:
``completed`` is the success terminal status; every other terminal value
the SDK emits (``failed``, ``declined``, ``cancelled``, ``error``, …) is
an error. Using a single positive case is safer than enumerating failure
strings — a future Codex SDK release adding a new failure status (e.g.
``denied``, ``timed_out``) would silently fall into ``is_error=False``
under the prior allow-listed-failures shape and let the gate pass on a
real failure.
"""


class CodexEvidenceProvider:
    """:class:`EvidenceProvider` for Codex; reads tee'd notification JSONL.

    Conforms structurally to
    :class:`src.core.protocols.evidence.EvidenceProvider`. The tee path
    convention (``~/.config/mala/codex-sessions/{thread_id}.jsonl``) is shared
    with :class:`src.infra.clients.codex_client.CodexClient` via the single
    :data:`CODEX_SESSIONS_DIR` source of truth.

    Construction:
      * ``CodexEvidenceProvider()`` — default; reads from
        :data:`CODEX_SESSIONS_DIR`.
      * ``CodexEvidenceProvider(sessions_dir=...)`` — point at an alternate
        directory (used by tests).
    """

    def __init__(self, *, sessions_dir: Path | None = None) -> None:
        """Initialize the provider.

        Args:
            sessions_dir: Override the per-thread tee directory; ``None``
                (default) uses :data:`CODEX_SESSIONS_DIR`.
        """
        self._sessions_dir = (
            sessions_dir if sessions_dir is not None else CODEX_SESSIONS_DIR
        )

    @property
    def sessions_dir(self) -> Path:
        """Return the per-thread tee directory in use."""
        return self._sessions_dir

    # ------------------------------------------------------------------
    # EvidenceProvider protocol surface
    # ------------------------------------------------------------------

    def get_log_path(self, repo_path: Path, session_id: str) -> Path:
        """Return the tee log path for ``session_id`` (Codex thread id).

        ``repo_path`` is accepted for protocol parity with
        :class:`FileSystemLogProvider` but is unused on the Codex path: the
        tee directory is per-user, not per-repo (parity with
        :class:`AmpLogProvider`).

        Args:
            repo_path: Repository the session ran in. Unused.
            session_id: Codex thread id (``thr_...``).

        Returns:
            ``{sessions_dir}/{session_id}.jsonl``. The file may not exist
            yet (e.g., before :class:`CodexClient` opens its tee handle).
        """
        del repo_path
        return self._sessions_dir / f"{session_id}.jsonl"

    def iter_session_events(
        self, log_path: Path, offset: int = 0
    ) -> Iterator[JsonlEntryProtocol]:
        """Yield parsed JSONL entries from ``log_path`` starting at ``offset``.

        Honors the caller-supplied ``offset`` to scope reads to a single
        attempt (parity with :class:`FileSystemLogProvider` and
        :class:`AmpLogProvider`); resolution-marker parsing depends on this
        so a stale ``ISSUE_*`` marker emitted in an earlier invocation does
        not override the current attempt's decision.

        For cross-invocation validation-evidence reads (AC #7 — Bash items
        from invocation 1 must remain visible after invocation 2 appends to
        the same thread file) callers must use :meth:`iter_thread_evidence`,
        which ignores offset and re-reads the full thread file.

        Tolerance contract (parity with the Amp provider):
          * Missing file → empty iterator (no raise).
          * Malformed JSON line → skip with warn-level log; do not raise.
          * Bad UTF-8 → skip silently.
        """
        if not log_path.exists():
            return

        try:
            f = open(log_path, "rb")
        except OSError as exc:
            logger.warning("CodexEvidenceProvider could not open %s: %s", log_path, exc)
            return

        with f:
            f.seek(offset)
            current_offset = offset
            for line_bytes in f:
                line_len = len(line_bytes)
                line_offset = current_offset
                current_offset += line_len

                try:
                    line = line_bytes.decode("utf-8").strip()
                except UnicodeDecodeError:
                    continue

                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(
                        "CodexEvidenceProvider: skipping malformed JSONL line at "
                        "offset %d in %s",
                        line_offset,
                        log_path,
                    )
                    continue

                if not isinstance(data, dict):
                    continue

                yield cast(
                    "JsonlEntryProtocol",
                    JsonlEntry(
                        data=data,
                        entry=None,
                        line_len=line_len,
                        offset=line_offset,
                    ),
                )

    def iter_thread_evidence(
        self, log_path: Path, offset: int = 0
    ) -> Iterator[JsonlEntryProtocol]:
        """Cross-invocation evidence read; ignores ``offset`` on Codex.

        AC #7 contract: when validation-evidence parsing replays the
        thread file after a Codex resume, ``CommandExecutionThreadItem``
        events from invocation 1 (lint / test / typecheck) MUST remain
        visible. The gate path advances ``retry_state.log_offset`` to EOF
        after each attempt, but the per-thread tee is append-only across
        every Codex turn for the same thread, so honoring that offset
        would silently drop invocation 1's evidence.

        This method ignores ``offset`` and re-reads from byte 0
        (parity with :meth:`AmpLogProvider.iter_thread_evidence`).
        """
        del offset
        return self.iter_session_events(log_path, 0)

    def get_end_offset(self, log_path: Path, start_offset: int = 0) -> int:
        """Return the byte offset at the end of ``log_path``.

        Returns ``start_offset`` when the file does not exist or cannot be
        read — matches :class:`FileSystemLogProvider` /
        :class:`AmpLogProvider` semantics.
        """
        if not log_path.exists():
            return start_offset
        try:
            with open(log_path, "rb") as f:
                f.seek(0, 2)
                return f.tell()
        except OSError:
            return start_offset

    async def wait_for_session_ready(
        self,
        repo_path: Path,
        session_id: str,
        *,
        timeout: float,
        poll_interval: float = 0.5,
    ) -> None:
        """Return immediately — Codex evidence is queryable as soon as the thread starts.

        The Codex SDK guarantees ``Thread.read`` (and, for the tee path,
        the per-thread JSONL file the client is already writing) is
        available the instant a thread is created. The session-readiness
        gate the lifecycle exposes via ``Effect.WAIT_FOR_LOG`` therefore
        has nothing to wait for on Codex; returning immediately preserves
        the protocol contract without imposing a poll cycle.

        ``timeout`` and ``poll_interval`` are accepted for protocol parity
        but unused (parity with the issue's note: "no-op for Codex —
        return immediately"). A best-effort ``asyncio.sleep(0)`` lets
        cooperative schedulers run other tasks before the lifecycle
        proceeds; without it, a tight ``await`` chain could starve the
        event loop on the no-op path.
        """
        del repo_path, session_id, timeout, poll_interval
        await asyncio.sleep(0)

    # ------------------------------------------------------------------
    # Extractors over Codex tee'd notifications
    # ------------------------------------------------------------------

    def extract_bash_commands(self, entry: JsonlEntryProtocol) -> list[tuple[str, str]]:
        """Extract Bash ``CommandExecutionThreadItem`` invocations from an entry.

        Reads ``item/completed`` notifications whose ``payload.item.type``
        is ``commandExecution`` and yields ``(item_id, command)``. The
        command is the shell line Codex executed; gate consumers match it
        against lint/test/typecheck regexes the same way they match
        Claude/Amp ``Bash`` ``tool_use`` blocks.

        Only ``item/completed`` is considered — ``item/started`` carries
        the same ``command`` but no ``aggregated_output`` yet, and the gate
        evidence parser pairs commands with their results, so reading both
        would emit duplicates.

        Returns:
            List of ``(item_id, command)`` tuples. Empty list if the entry
            is not a completed ``commandExecution`` item.
        """
        item = self._completed_item(entry, _BASH_ITEM_TYPE)
        if item is None:
            return []
        item_id = self._str_field(item, "id")
        command = self._str_field(item, "command")
        if not item_id:
            return []
        return [(item_id, command)]

    def extract_tool_results(self, entry: JsonlEntryProtocol) -> list[tuple[str, bool]]:
        """Extract ``(item_id, is_error)`` tuples for completed tool items.

        Classification rules (mirrors :class:`CodexEventAdapter` —
        :meth:`_command_completed` / :meth:`_file_change_completed` /
        :meth:`_mcp_completed`):

          * Yields a result ONLY for the three tool item types in
            :data:`_TOOL_RESULT_ITEM_TYPES` (``commandExecution`` /
            ``fileChange`` / ``mcpToolCall``). ``agentMessage`` items
            produce text, not tool results, and ``reasoning`` items are
            dropped — both match the live adapter's
            :meth:`_handle_item_completed` dispatch.
          * Any status other than ``completed`` → ``is_error=True``.
            Covers ``failed``, ``declined``, ``cancelled``, ``error`` and
            any future failure status the SDK adds. Using a single
            positive success-status check keeps the rule synchronised
            with the adapter (``is_error = status != "completed"``)
            without enumerating an open-ended failure list — a future
            SDK addition like ``timed_out`` would silently fall into
            ``is_error=False`` under an allow-listed-failures shape and
            let the gate pass on a real failure.
          * For ``commandExecution`` items with ``status=completed``:
            ``is_error`` consults the shell ``exit_code`` because Codex
            marks a command as completed whenever the shell *finished*,
            even on non-zero exits. Without the exit-code check, a failed
            ``uv run pytest`` (status=completed, exit_code=1) would
            surface as ``(tool_id, False)``,
            :meth:`EvidenceCheck.parse_validation_evidence_with_spec`
            would skip adding it to ``failed_commands``, and the gate
            would silently pass on a failed test/lint/typecheck run.
          * For ``fileChange`` / ``mcpToolCall`` items with
            ``status=completed``: the Codex SDK does not surface
            ``exit_code``, so ``status`` is the sole signal and
            ``is_error=False``.

        Returns:
            List of ``(item_id, is_error)`` tuples. Empty list if the
            entry is not an ``item/completed`` notification carrying one
            of the three tool item types.
        """
        item = self._completed_item(entry, item_type=None)
        if item is None:
            return []
        item_type = self._str_field(item, "type")
        if item_type not in _TOOL_RESULT_ITEM_TYPES:
            return []
        item_id = self._str_field(item, "id")
        if not item_id:
            return []
        status = self._str_field(item, "status").lower()
        if status != _SUCCESS_STATUS:
            return [(item_id, True)]
        if item_type == _BASH_ITEM_TYPE:
            # status="completed" means the shell ran to exit; consult
            # exit_code so a non-zero return classifies as a failure.
            exit_code = item.get("exit_code")
            return [(item_id, exit_code not in (None, 0))]
        return [(item_id, False)]

    def extract_assistant_text_blocks(self, entry: JsonlEntryProtocol) -> list[str]:
        """Extract assistant text from a completed ``agentMessage`` item.

        Reads ``item/completed`` notifications whose ``payload.item.type``
        is ``agentMessage`` and yields the final ``item.text`` string.
        Codex emits incremental ``item/agentMessage/delta`` notifications
        too, but evidence parsers operate on completed-message text only —
        partial deltas would emit unstable substrings that confuse the
        resolution-marker matcher (which scans for ``ISSUE_*`` literals).

        Returns:
            List with the message text, or empty list if the entry is not
            a completed ``agentMessage`` item.
        """
        item = self._completed_item(entry, _AGENT_MESSAGE_TYPE)
        if item is None:
            return []
        text = self._str_field(item, "text")
        if not text:
            return []
        return [text]

    def extract_tool_result_content(
        self, entry: JsonlEntryProtocol
    ) -> list[tuple[str, str]]:
        """Extract ``(item_id, content)`` tuples for marker-detection scans.

        Used by :class:`EvidenceCheck` to look for custom-validation
        markers (``[custom:<name>:<status>]``) in Bash output. Returns the
        ``aggregated_output`` of completed ``commandExecution`` items;
        other item types do not carry textual output that gate parsers
        consume so they are skipped.

        Returns:
            List of ``(item_id, content)`` tuples. Empty list if the entry
            is not a completed ``commandExecution`` item.
        """
        item = self._completed_item(entry, _BASH_ITEM_TYPE)
        if item is None:
            return []
        item_id = self._str_field(item, "id")
        if not item_id:
            return []
        content = self._str_field(item, "aggregated_output")
        return [(item_id, content)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _completed_item(
        entry: JsonlEntryProtocol, item_type: str | None
    ) -> dict[str, Any] | None:
        """Return ``payload.item`` dict for an ``item/completed`` notification.

        Filters by ``item.type == item_type`` when ``item_type`` is not
        ``None``; ``None`` returns any completed item regardless of type
        (used by :meth:`extract_tool_results`).
        """
        data = entry.data
        if data.get("method") != _COMPLETED_METHOD:
            return None
        payload = data.get("payload")
        if not isinstance(payload, dict):
            return None
        item = payload.get("item")
        if not isinstance(item, dict):
            return None
        if item_type is not None and item.get("type") != item_type:
            return None
        return cast("dict[str, Any]", item)

    @staticmethod
    def _str_field(item: dict[str, Any], key: str) -> str:
        """Return ``item[key]`` as a string, defaulting to ``""``."""
        value = item.get(key)
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

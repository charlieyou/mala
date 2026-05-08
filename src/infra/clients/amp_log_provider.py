"""EvidenceProvider for the Amp coder.

Aggregates validation evidence across all invocations of an Amp thread by
reading the per-thread tee'd stream-json JSONL written by
:class:`src.infra.clients.amp_client.AmpClient` at
``~/.config/mala/amp-sessions/{thread_id}.jsonl``. The tee path is per-thread
and append-only: every ``amp --execute --stream-json`` invocation for the same
thread (initial run + resumes triggered by idle timeout, gate failure, or
review-issue retries) appends to the same file, so Bash ``tool_use`` events
from earlier invocations remain observable to validation gates regardless of
whether Amp's resume mode emits delta-only or full-history events.

The provider conforms structurally to
:class:`src.core.protocols.evidence.EvidenceProvider` and emits the same
``JsonlEntry`` shape :class:`src.infra.io.session_log_parser.FileSystemLogProvider`
produces, so ``EvidenceCheck`` and other gate consumers parse Amp evidence
identically without provider-specific branches.

Native-log probe (plan ``L122``): if a stable JSONL log directory is present
under ``~/.config/amp/`` or ``~/.local/share/amp/``, prefer it over the tee.
The Amp native log format is undocumented (plan ``L46-L47``); the probe
fails safely (returns ``None``) and the caller falls back to the tee path.
The probe runs once at construction (via :meth:`AmpLogProvider.from_probe`)
and the result is captured in the instance, so the source choice is stable
for the lifetime of the run.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

from src.core.log_events import parse_log_entry
from src.infra.clients.amp_runtime import AMP_SESSIONS_DIR
from src.infra.io.session_log_parser import JsonlEntry, SessionLogParser

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from src.core.protocols.evidence import JsonlEntryProtocol


logger = logging.getLogger(__name__)


_NATIVE_LOG_GLOB = "T-*.jsonl"
"""Heuristic for detecting Amp's native thread-keyed JSONL log files.

Amp emits thread ids of the form ``T-<...>``. We treat the presence of one
or more files matching this pattern under ``~/.config/amp/`` (or
``~/.local/share/amp/``) as evidence of a stable native log directory.
Format is undocumented (plan ``L46-L47``); when the heuristic does not
match, the probe returns ``None`` and we fall back to the tee.
"""


def _default_native_search_paths() -> tuple[Path, ...]:
    """Default candidate directories scanned by :func:`_discover_native_log_dir`."""
    home = Path.home()
    return (home / ".config" / "amp", home / ".local" / "share" / "amp")


def _discover_native_log_dir(
    search_paths: Sequence[Path] | None = None,
) -> Path | None:
    """Probe candidate directories for Amp's native thread-keyed JSONL.

    Args:
        search_paths: Directories to scan. ``None`` uses the default
            candidates (``~/.config/amp/``, ``~/.local/share/amp/``).

    Returns:
        The first directory containing at least one file matching the
        :data:`_NATIVE_LOG_GLOB` pattern, or ``None`` if no candidate
        looks like a native log directory.

    Notes:
        Failures (missing directories, permission errors) are swallowed —
        the probe is best-effort and fails safely to ``None``.
    """
    if search_paths is None:
        search_paths = _default_native_search_paths()
    for path in search_paths:
        try:
            if not path.is_dir():
                continue
            if any(path.glob(_NATIVE_LOG_GLOB)):
                return path
        except OSError:
            continue
    return None


class AmpLogProvider:
    """EvidenceProvider for Amp; reads tee'd or native JSONL log.

    Conforms structurally to
    :class:`src.core.protocols.evidence.EvidenceProvider`. The tee path
    convention (``~/.config/mala/amp-sessions/{thread_id}.jsonl``) is shared
    with :class:`src.infra.clients.amp_client.AmpClient` via the single
    :data:`src.infra.clients.amp_runtime.AMP_SESSIONS_DIR` source of truth.

    Construction modes:
      * ``AmpLogProvider()`` — use the tee path (MVP default).
      * ``AmpLogProvider(native_dir=...)`` — prefer the supplied native
        directory; thread-keyed reads resolve to ``native_dir/{tid}.jsonl``.
      * ``AmpLogProvider.from_probe()`` — perform the native-log probe at
        construction time; falls back to the tee when the probe finds
        nothing.

    Attributes:
        native_dir: The discovered (or injected) native log directory, or
            ``None`` when the tee path is in use. Stable for the lifetime
            of the instance.
    """

    def __init__(self, *, native_dir: Path | None = None) -> None:
        """Initialize the provider.

        Args:
            native_dir: Pre-discovered native log directory; when ``None``
                (default), the tee path under
                :data:`AMP_SESSIONS_DIR` is used.
        """
        self._native_dir = native_dir
        self._parser = SessionLogParser()

    @property
    def native_dir(self) -> Path | None:
        """Return the captured native log directory, or ``None`` for tee mode."""
        return self._native_dir

    @classmethod
    def from_probe(
        cls, *, search_paths: Sequence[Path] | None = None
    ) -> AmpLogProvider:
        """Construct a provider after probing for a native log directory.

        Probes once at install time (i.e., at construction); the resulting
        source choice is captured in the returned instance and is stable
        for the lifetime of the run. Falls back to the tee when no native
        directory is discovered.

        Args:
            search_paths: Override the default candidate directories. The
                default scan covers ``~/.config/amp/`` and
                ``~/.local/share/amp/``.
        """
        return cls(native_dir=_discover_native_log_dir(search_paths))

    # ------------------------------------------------------------------
    # EvidenceProvider protocol surface
    # ------------------------------------------------------------------

    def get_log_path(self, repo_path: Path, session_id: str) -> Path:
        """Return the log path for ``session_id`` (Amp thread id).

        ``repo_path`` is accepted for protocol parity with
        :class:`FileSystemLogProvider` but is unused on the Amp path: the
        tee directory is per-user, not per-repo.

        Args:
            repo_path: Repository the session ran in. Unused.
            session_id: Amp thread id (``T-...``) returned by the
                ``system(init)`` event.

        Returns:
            ``{native_dir or AMP_SESSIONS_DIR}/{session_id}.jsonl``. The
            file may not exist yet (e.g., before the first ``system(init)``
            renames the pending tee file).
        """
        del repo_path
        base = self._native_dir if self._native_dir is not None else AMP_SESSIONS_DIR
        return base / f"{session_id}.jsonl"

    def iter_session_events(
        self, log_path: Path, offset: int = 0
    ) -> Iterator[JsonlEntryProtocol]:
        """Yield parsed JSONL entries from ``log_path`` starting at ``offset``.

        Honors the caller-supplied ``offset`` — this is the protocol
        contract (matches :class:`FileSystemLogProvider`) and is what
        resolution-marker parsing depends on: when invocation 1 emitted
        a stale ``ISSUE_*`` marker that the gate rejected, the next gate
        replay must be scoped to events after that point so the new
        invocation's decision wins.

        For cross-invocation validation-evidence reads (AC#7a — Bash
        ``tool_use`` events from invocation 1 must remain visible after
        invocation 2 appends to the same thread file) callers must use
        :meth:`iter_thread_evidence`, which ignores offset on this
        provider and re-reads the full thread file.

        Tolerance contract (plan ``L675-L676``):
          * Missing file → empty iterator (no raise).
          * Malformed JSON line → skip with warn-level log; do not raise.
          * Bad UTF-8 → skip silently (matches FileSystemLogProvider).
        """
        if not log_path.exists():
            return

        try:
            f = open(log_path, "rb")
        except OSError as exc:
            logger.warning("AmpLogProvider could not open %s: %s", log_path, exc)
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
                        "AmpLogProvider: skipping malformed JSONL line at "
                        "offset %d in %s",
                        line_offset,
                        log_path,
                    )
                    continue

                if not isinstance(data, dict):
                    continue

                entry = parse_log_entry(data)
                yield cast(
                    "JsonlEntryProtocol",
                    JsonlEntry(
                        data=data,
                        entry=entry,
                        line_len=line_len,
                        offset=line_offset,
                    ),
                )

    def iter_thread_evidence(
        self, log_path: Path, offset: int = 0
    ) -> Iterator[JsonlEntryProtocol]:
        """Cross-invocation evidence read; ignores ``offset`` on Amp.

        AC#7a contract: when validation-evidence parsing replays the
        thread file after an Amp resume, Bash ``tool_use`` events from
        invocation 1 (lint / test / typecheck) MUST remain visible —
        the gate path advances ``retry_state.log_offset`` to EOF after
        each attempt, but the per-thread tee is append-only across
        every ``amp --execute`` for the same thread, so honoring that
        offset would silently drop invocation 1's evidence.

        This method ignores ``offset`` and re-reads from byte 0. The
        caller (:meth:`EvidenceCheck.parse_validation_evidence_with_spec`)
        uses this for validation-evidence presence detection;
        resolution-marker parsing keeps using :meth:`iter_session_events`
        so the offset still scopes resolution decisions to the latest
        attempt.
        """
        del offset  # see docstring rationale.
        return self.iter_session_events(log_path, 0)

    def get_end_offset(self, log_path: Path, start_offset: int = 0) -> int:
        """Return the byte offset at the end of ``log_path``.

        Returns ``start_offset`` when the file does not exist or cannot be
        read — matches :class:`FileSystemLogProvider` semantics.
        """
        return self._parser.get_log_end_offset(log_path, start_offset)

    async def wait_for_session_ready(
        self,
        log_path: Path,
        *,
        timeout: float,
        poll_interval: float = 0.5,
    ) -> None:
        """Wait for the Amp tee'd / native JSONL to appear on disk.

        Polls ``log_path`` for existence on a fixed cadence, matching the
        prior in-runner filesystem poll. The Amp tee'd path appears once
        :class:`AmpClient` finishes the rename triggered by the
        ``system(init)`` event; a native log directory has the same
        appearance semantic, so the same poll covers both.

        Raises:
            TimeoutError: If the file does not appear within ``timeout``.
        """
        wait_elapsed = 0.0
        while not log_path.exists():
            if wait_elapsed >= timeout:
                raise TimeoutError(f"Session log missing after timeout: {log_path}")
            await asyncio.sleep(poll_interval)
            wait_elapsed += poll_interval

    def extract_bash_commands(self, entry: JsonlEntryProtocol) -> list[tuple[str, str]]:
        """Delegate to :meth:`SessionLogParser.extract_bash_commands`."""
        return self._parser.extract_bash_commands(entry)

    def extract_tool_results(self, entry: JsonlEntryProtocol) -> list[tuple[str, bool]]:
        """Delegate to :meth:`SessionLogParser.extract_tool_results`."""
        return self._parser.extract_tool_results(entry)

    def extract_assistant_text_blocks(self, entry: JsonlEntryProtocol) -> list[str]:
        """Delegate to :meth:`SessionLogParser.extract_assistant_text_blocks`."""
        return self._parser.extract_assistant_text_blocks(entry)

    def extract_tool_result_content(
        self, entry: JsonlEntryProtocol
    ) -> list[tuple[str, str]]:
        """Delegate to :meth:`SessionLogParser.extract_tool_result_content`."""
        return self._parser.extract_tool_result_content(entry)

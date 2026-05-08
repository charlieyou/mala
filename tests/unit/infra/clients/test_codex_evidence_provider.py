"""Unit tests for :class:`CodexEvidenceProvider` (Phase F, T013).

Covers the test cases enumerated in the issue's Verification section:

  * Reads tee'd JSONL and emits the same Bash-evidence shape
    :class:`FileSystemLogProvider` produces (AC #7 / AC #18 — protocol
    parity across Claude / Amp / Codex).
  * ``CommandExecutionThreadItem.aggregated_output`` is observable via
    :meth:`extract_tool_result_content` so custom-validation markers
    (``[custom:<name>:<status>]``) can be detected by the gate.
  * Tool results pair with item ids: status mapping (``completed`` →
    ``is_error=False``; ``failed``/``error``/``cancelled`` →
    ``is_error=True``).
  * ``iter_session_events`` honors ``offset`` for resolution-marker
    parsing; ``iter_thread_evidence`` ignores ``offset`` and re-reads
    from byte 0 for cross-resume validation evidence (AC #7 cross-resume
    invariant — same regression coverage as
    :class:`AmpLogProvider`'s log-stitching test).
  * Missing file → empty iterator (no raise).
  * Malformed trailing line tolerance: yield all preceding events; warn.
  * ``wait_for_session_ready`` returns immediately (Codex evidence is
    queryable as soon as the thread starts — issue notes: "no-op for
    Codex").
  * Conformance to :class:`EvidenceProvider` runtime protocol.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from src.core.protocols.evidence import EvidenceProvider
from src.infra.clients.codex_evidence_provider import (
    CODEX_SESSIONS_DIR,
    CodexEvidenceProvider,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


# ---------------------------------------------------------------------------
# Fixture builders for Codex notification-shaped JSONL
# ---------------------------------------------------------------------------


def _command_completed(
    item_id: str,
    command: str,
    aggregated_output: str = "",
    status: str = "completed",
    *,
    exit_code: int | None = 0,
) -> dict[str, object]:
    """``item/completed`` notification for a ``CommandExecutionThreadItem``.

    Codex marks a command as ``status="completed"`` whenever the shell
    finished — even on non-zero exits — so tests pin ``exit_code``
    explicitly. Default 0 (clean exit) so existing fixtures keep their
    success semantics.
    """
    item: dict[str, object] = {
        "type": "commandExecution",
        "id": item_id,
        "command": command,
        "aggregated_output": aggregated_output,
        "status": status,
    }
    if exit_code is not None:
        item["exit_code"] = exit_code
    return {
        "method": "item/completed",
        "payload": {
            "item": item,
            "thread_id": "thr_test",
            "turn_id": "turn_1",
        },
    }


def _command_started(item_id: str, command: str) -> dict[str, object]:
    """``item/started`` notification — should NOT yield bash extraction."""
    return {
        "method": "item/started",
        "payload": {
            "item": {
                "type": "commandExecution",
                "id": item_id,
                "command": command,
                "status": "in_progress",
            },
            "thread_id": "thr_test",
            "turn_id": "turn_1",
        },
    }


def _agent_message_completed(item_id: str, text: str) -> dict[str, object]:
    """``item/completed`` notification for an ``AgentMessageThreadItem``."""
    return {
        "method": "item/completed",
        "payload": {
            "item": {
                "type": "agentMessage",
                "id": item_id,
                "text": text,
            },
            "thread_id": "thr_test",
            "turn_id": "turn_1",
        },
    }


def _file_change_completed(
    item_id: str, changes: list[dict[str, object]], status: str = "completed"
) -> dict[str, object]:
    """``item/completed`` notification for a ``FileChangeThreadItem``."""
    return {
        "method": "item/completed",
        "payload": {
            "item": {
                "type": "fileChange",
                "id": item_id,
                "changes": changes,
                "status": status,
            },
            "thread_id": "thr_test",
            "turn_id": "turn_1",
        },
    }


def _agent_message_delta(item_id: str, delta: str) -> dict[str, object]:
    """``item/agentMessage/delta`` — partial token stream."""
    return {
        "method": "item/agentMessage/delta",
        "payload": {
            "delta": delta,
            "item_id": item_id,
            "thread_id": "thr_test",
            "turn_id": "turn_1",
        },
    }


def _turn_completed(thread_id: str = "thr_test") -> dict[str, object]:
    return {
        "method": "turn/completed",
        "payload": {
            "thread_id": thread_id,
            "turn": {"id": "turn_1", "status": "completed"},
        },
    }


def _write_jsonl(path: Path, events: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event))
            fh.write("\n")


def _append_jsonl(path: Path, events: Iterable[dict[str, object]]) -> None:
    """Append events as if a second Codex turn tee'd to the same file."""
    with path.open("a", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event))
            fh.write("\n")


# ---------------------------------------------------------------------------
# Protocol conformance + log path resolution
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_provider_conforms_to_evidence_provider_protocol() -> None:
    provider = CodexEvidenceProvider()
    assert isinstance(provider, EvidenceProvider)


@pytest.mark.unit
def test_default_log_path_is_codex_sessions_dir() -> None:
    """``get_log_path`` resolves to ``~/.config/mala/codex-sessions/{tid}.jsonl``."""
    provider = CodexEvidenceProvider()

    path = provider.get_log_path(Path("/tmp/repo"), "thr_abc123")

    assert path == CODEX_SESSIONS_DIR / "thr_abc123.jsonl"


@pytest.mark.unit
def test_custom_sessions_dir_is_honored(tmp_path: Path) -> None:
    """Tests can point the provider at a tmp_path-rooted sessions dir."""
    provider = CodexEvidenceProvider(sessions_dir=tmp_path)

    assert provider.sessions_dir == tmp_path
    assert provider.get_log_path(Path("/tmp/repo"), "thr_x") == tmp_path / "thr_x.jsonl"


# ---------------------------------------------------------------------------
# Bash-evidence extraction + tool-result pairing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_iter_session_events_extracts_bash_command_from_completed_item(
    tmp_path: Path,
) -> None:
    """``CommandExecutionThreadItem`` → ``(item_id, command)``."""
    log_path = tmp_path / "thr_test.jsonl"
    _write_jsonl(
        log_path,
        [
            _command_completed("item-1", "uv run pytest -q", "5 passed", "completed"),
            _turn_completed(),
        ],
    )

    provider = CodexEvidenceProvider(sessions_dir=tmp_path)
    bash_commands: list[tuple[str, str]] = []
    for entry in provider.iter_session_events(log_path):
        bash_commands.extend(provider.extract_bash_commands(entry))

    assert bash_commands == [("item-1", "uv run pytest -q")]


@pytest.mark.unit
def test_extract_bash_commands_skips_started_items(tmp_path: Path) -> None:
    """``item/started`` carries the command but no output — must not duplicate.

    The gate evidence parser pairs commands with their results via
    :meth:`extract_tool_results`; emitting both ``item/started`` and
    ``item/completed`` would surface duplicate ``(id, command)`` pairs and
    inflate the validation-evidence presence count.
    """
    log_path = tmp_path / "thr_test.jsonl"
    _write_jsonl(
        log_path,
        [
            _command_started("item-1", "uv run pytest -q"),
            _command_completed("item-1", "uv run pytest -q", "ok", "completed"),
        ],
    )

    provider = CodexEvidenceProvider(sessions_dir=tmp_path)
    bash_commands: list[tuple[str, str]] = []
    for entry in provider.iter_session_events(log_path):
        bash_commands.extend(provider.extract_bash_commands(entry))

    assert bash_commands == [("item-1", "uv run pytest -q")]


@pytest.mark.unit
def test_extract_tool_results_pairs_item_id_with_status(tmp_path: Path) -> None:
    """``status`` drives ``is_error`` for completed items."""
    log_path = tmp_path / "thr_test.jsonl"
    _write_jsonl(
        log_path,
        [
            _command_completed("item-ok", "echo ok", "ok", "completed"),
            _command_completed("item-fail", "false", "", "failed"),
            _file_change_completed("item-edit", [{"path": "/x"}], status="completed"),
        ],
    )

    provider = CodexEvidenceProvider(sessions_dir=tmp_path)
    results: list[tuple[str, bool]] = []
    for entry in provider.iter_session_events(log_path):
        results.extend(provider.extract_tool_results(entry))

    assert results == [
        ("item-ok", False),
        ("item-fail", True),
        ("item-edit", False),
    ]


@pytest.mark.unit
def test_command_with_completed_status_but_nonzero_exit_is_error(
    tmp_path: Path,
) -> None:
    """Codex command items with ``status=completed`` + ``exit_code != 0`` are errors.

    Regression for the review-2 P1: Codex marks a ``commandExecution``
    item as ``status="completed"`` whenever the shell process *finished*,
    even if the command exited non-zero. If
    :meth:`CodexEvidenceProvider.extract_tool_results` only checked
    ``status``, a failed ``uv run pytest`` (status=completed, exit_code=1)
    would surface as ``(item_id, False)``,
    :meth:`EvidenceCheck.parse_validation_evidence_with_spec` would skip
    adding it to ``failed_commands``, and the validation gate would
    silently pass on a failed test/lint/typecheck run.

    Mirrors :meth:`CodexEventAdapter._command_completed`'s exit-code rule
    (``src/infra/clients/codex_event_adapter.py:343-347``) so the live
    pipeline and the evidence-replay path classify identically.
    """
    log_path = tmp_path / "thr_test.jsonl"
    _write_jsonl(
        log_path,
        [
            _command_completed(
                "item-pytest-fail",
                "uv run pytest -q",
                "1 failed",
                "completed",
                exit_code=1,
            ),
            _command_completed(
                "item-pytest-ok",
                "uv run pytest -q",
                "1 passed",
                "completed",
                exit_code=0,
            ),
            # Missing ``exit_code`` key entirely (older SDK shape /
            # not-yet-populated): treat as success rather than crash.
            _command_completed(
                "item-no-exit-code",
                "echo missing",
                "ok",
                "completed",
                exit_code=None,
            ),
        ],
    )

    provider = CodexEvidenceProvider(sessions_dir=tmp_path)
    results: list[tuple[str, bool]] = []
    for entry in provider.iter_session_events(log_path):
        results.extend(provider.extract_tool_results(entry))

    assert results == [
        ("item-pytest-fail", True),
        ("item-pytest-ok", False),
        ("item-no-exit-code", False),
    ]


@pytest.mark.unit
def test_non_command_items_ignore_exit_code(tmp_path: Path) -> None:
    """``fileChange`` / ``mcpToolCall`` / ``agentMessage`` use status only.

    Codex's non-command items do not carry an ``exit_code`` field; the
    error signal lives in ``status``. Verify the type-discrimination
    branch in ``extract_tool_results`` does not accidentally apply the
    command-only ``exit_code`` rule to other item types — which would
    treat all ``status=completed`` non-command items as success
    (correct), but a future drift to a different default could silently
    flip them.
    """
    log_path = tmp_path / "thr_test.jsonl"
    _write_jsonl(
        log_path,
        [
            _agent_message_completed("msg-1", "hello"),
            _file_change_completed("edit-ok", [{"path": "/x"}], status="completed"),
            _file_change_completed("edit-fail", [{"path": "/x"}], status="failed"),
        ],
    )

    provider = CodexEvidenceProvider(sessions_dir=tmp_path)
    results: list[tuple[str, bool]] = []
    for entry in provider.iter_session_events(log_path):
        results.extend(provider.extract_tool_results(entry))

    assert results == [
        ("msg-1", False),
        ("edit-ok", False),
        ("edit-fail", True),
    ]


@pytest.mark.unit
def test_extract_tool_result_content_returns_aggregated_output(tmp_path: Path) -> None:
    """Custom-validation markers ride on ``aggregated_output``.

    :class:`EvidenceCheck` scans this content for
    ``[custom:<name>:<status>]`` markers; without surfacing
    ``aggregated_output`` here, custom commands would always read as
    "marker absent" and the gate would loop on a passing build.
    """
    log_path = tmp_path / "thr_test.jsonl"
    output = "running\n[custom:python_test:pass]\n"
    _write_jsonl(
        log_path,
        [
            _command_completed("item-1", "uv run pytest", output, "completed"),
        ],
    )

    provider = CodexEvidenceProvider(sessions_dir=tmp_path)
    contents: list[tuple[str, str]] = []
    for entry in provider.iter_session_events(log_path):
        contents.extend(provider.extract_tool_result_content(entry))

    assert contents == [("item-1", output)]


@pytest.mark.unit
def test_extract_assistant_text_from_agent_message_item(tmp_path: Path) -> None:
    """``AgentMessageThreadItem.text`` surfaces as assistant text."""
    log_path = tmp_path / "thr_test.jsonl"
    _write_jsonl(
        log_path,
        [
            _agent_message_completed("msg-1", "Done — ISSUE_NO_CHANGE: nothing to do"),
        ],
    )

    provider = CodexEvidenceProvider(sessions_dir=tmp_path)
    texts: list[str] = []
    for entry in provider.iter_session_events(log_path):
        texts.extend(provider.extract_assistant_text_blocks(entry))

    assert texts == ["Done — ISSUE_NO_CHANGE: nothing to do"]


@pytest.mark.unit
def test_extract_assistant_text_skips_partial_deltas(tmp_path: Path) -> None:
    """``item/agentMessage/delta`` is a partial token stream — skip it.

    Resolution-marker matching scans for whole literals like
    ``ISSUE_NO_CHANGE``; a delta that splits the literal mid-token would
    yield false negatives or, worse, false positives when a token
    boundary happens to land inside a marker substring.
    """
    log_path = tmp_path / "thr_test.jsonl"
    _write_jsonl(
        log_path,
        [
            _agent_message_delta("msg-1", "ISSUE_"),
            _agent_message_delta("msg-1", "NO_CHANGE"),
            _agent_message_completed("msg-1", "ISSUE_NO_CHANGE: nothing"),
        ],
    )

    provider = CodexEvidenceProvider(sessions_dir=tmp_path)
    texts: list[str] = []
    for entry in provider.iter_session_events(log_path):
        texts.extend(provider.extract_assistant_text_blocks(entry))

    # Only the completed message — deltas excluded.
    assert texts == ["ISSUE_NO_CHANGE: nothing"]


# ---------------------------------------------------------------------------
# Offset semantics + cross-resume invariant
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_iter_session_events_honors_offset(tmp_path: Path) -> None:
    """``iter_session_events`` skips bytes before ``offset``.

    Resolution-marker parsing uses this to scope reads to the latest
    attempt; without offset honoring, a stale ``ISSUE_*`` from invocation
    1 would override invocation 2's decision.
    """
    log_path = tmp_path / "thr_test.jsonl"
    _write_jsonl(
        log_path,
        [
            _command_completed("item-old", "echo old", "old", "completed"),
        ],
    )
    cutoff = log_path.stat().st_size
    _append_jsonl(
        log_path,
        [
            _command_completed("item-new", "echo new", "new", "completed"),
        ],
    )

    provider = CodexEvidenceProvider(sessions_dir=tmp_path)
    after_cutoff: list[tuple[str, str]] = []
    for entry in provider.iter_session_events(log_path, offset=cutoff):
        after_cutoff.extend(provider.extract_bash_commands(entry))

    assert after_cutoff == [("item-new", "echo new")]


@pytest.mark.unit
def test_iter_thread_evidence_ignores_offset_for_cross_resume(tmp_path: Path) -> None:
    """Cross-resume invariant: invocation 1's bash items remain visible.

    Regression coverage for AC #7 (validation evidence is available for
    Codex runs) and the issue's cross-resume bullet: the gate path
    advances ``retry_state.log_offset`` to EOF after each attempt, but
    the per-thread tee is append-only across every Codex turn for the
    same thread, so honoring that offset would silently drop invocation
    1's lint / test / typecheck evidence.

    Mirrors :class:`AmpLogProvider`'s ``test_iter_thread_evidence_ignores_offset``
    so the protocol contract is enforced identically across both
    multi-invocation providers.
    """
    log_path = tmp_path / "thr_test.jsonl"
    _write_jsonl(
        log_path,
        [
            _command_completed("inv1-pytest", "uv run pytest", "5 passed", "completed"),
            _command_completed("inv1-ruff", "uvx ruff check .", "ok", "completed"),
            _command_completed("inv1-ty", "uvx ty check", "ok", "completed"),
        ],
    )
    cutoff = log_path.stat().st_size
    _append_jsonl(
        log_path,
        [
            _command_completed("inv2-pytest", "uv run pytest", "ok", "completed"),
        ],
    )

    provider = CodexEvidenceProvider(sessions_dir=tmp_path)
    bash_commands: list[tuple[str, str]] = []
    for entry in provider.iter_thread_evidence(log_path, offset=cutoff):
        bash_commands.extend(provider.extract_bash_commands(entry))

    # The cross-resume read returns ALL four commands, ignoring ``offset``.
    assert [item_id for item_id, _ in bash_commands] == [
        "inv1-pytest",
        "inv1-ruff",
        "inv1-ty",
        "inv2-pytest",
    ]


@pytest.mark.unit
def test_get_end_offset_returns_eof(tmp_path: Path) -> None:
    """``get_end_offset`` returns the size of the file."""
    log_path = tmp_path / "thr_test.jsonl"
    _write_jsonl(
        log_path,
        [
            _command_completed("item-1", "echo", "", "completed"),
        ],
    )
    provider = CodexEvidenceProvider(sessions_dir=tmp_path)

    end = provider.get_end_offset(log_path)

    assert end == log_path.stat().st_size


@pytest.mark.unit
def test_get_end_offset_returns_start_offset_on_missing_file(tmp_path: Path) -> None:
    """Parity with FileSystemLogProvider: missing file → return start_offset."""
    provider = CodexEvidenceProvider(sessions_dir=tmp_path)

    assert provider.get_end_offset(tmp_path / "missing.jsonl", start_offset=42) == 42


# ---------------------------------------------------------------------------
# Tolerance contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_iter_session_events_returns_empty_when_file_missing(tmp_path: Path) -> None:
    """No file → empty iterator (no raise)."""
    provider = CodexEvidenceProvider(sessions_dir=tmp_path)

    entries = list(provider.iter_session_events(tmp_path / "missing.jsonl"))

    assert entries == []


@pytest.mark.unit
def test_iter_session_events_skips_malformed_jsonl(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Malformed trailing line tolerance: yield preceding events; warn."""
    log_path = tmp_path / "thr_test.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_command_completed("item-1", "echo ok", "ok")) + "\n")
        fh.write("{not valid json\n")
        fh.write(json.dumps(_command_completed("item-2", "echo more", "more")) + "\n")

    provider = CodexEvidenceProvider(sessions_dir=tmp_path)
    with caplog.at_level(logging.WARNING):
        bash_commands: list[tuple[str, str]] = []
        for entry in provider.iter_session_events(log_path):
            bash_commands.extend(provider.extract_bash_commands(entry))

    assert bash_commands == [("item-1", "echo ok"), ("item-2", "echo more")]
    assert any(
        "skipping malformed JSONL line" in record.getMessage()
        for record in caplog.records
    )


@pytest.mark.unit
def test_iter_session_events_skips_non_dict_top_level(tmp_path: Path) -> None:
    """JSON arrays / strings at top level are skipped silently."""
    log_path = tmp_path / "thr_test.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write("[1, 2, 3]\n")
        fh.write('"just a string"\n')
        fh.write(json.dumps(_command_completed("item-1", "echo ok", "ok")) + "\n")

    provider = CodexEvidenceProvider(sessions_dir=tmp_path)
    bash_commands: list[tuple[str, str]] = []
    for entry in provider.iter_session_events(log_path):
        bash_commands.extend(provider.extract_bash_commands(entry))

    assert bash_commands == [("item-1", "echo ok")]


@pytest.mark.unit
def test_extractors_return_empty_for_unrelated_methods(tmp_path: Path) -> None:
    """Non-``item/completed`` methods produce no extracted evidence."""
    log_path = tmp_path / "thr_test.jsonl"
    _write_jsonl(
        log_path,
        [
            _turn_completed(),
            _agent_message_delta("msg-1", "hi"),
        ],
    )

    provider = CodexEvidenceProvider(sessions_dir=tmp_path)
    for entry in provider.iter_session_events(log_path):
        assert provider.extract_bash_commands(entry) == []
        assert provider.extract_tool_results(entry) == []
        assert provider.extract_assistant_text_blocks(entry) == []
        assert provider.extract_tool_result_content(entry) == []


# ---------------------------------------------------------------------------
# wait_for_session_ready — no-op contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_wait_for_session_ready_returns_immediately(tmp_path: Path) -> None:
    """Codex evidence is queryable as soon as the thread starts.

    Issue notes: ``wait_for_session_ready`` is a no-op for Codex —
    return immediately. The provider must NOT poll the filesystem; the
    Phase H lifecycle relies on this so a Codex run does not spin a
    timeout window before the first evidence read.
    """
    provider = CodexEvidenceProvider(sessions_dir=tmp_path)

    # No file written; if the provider polled it would TimeoutError after
    # ``timeout`` seconds. With ``timeout=0`` and a no-op contract, the
    # call resolves immediately.
    await provider.wait_for_session_ready(
        Path("/tmp/repo"), "thr_x", timeout=0.0, poll_interval=0.1
    )

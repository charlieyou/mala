"""Unit tests for :class:`src.infra.clients.amp_log_provider.AmpLogProvider`.

Covers the test cases enumerated in the plan's Testing & Validation section
(``plans/2026-04-29-amp-provider-plan.md#L825-L835``) and the issue's
Verification section:

  * Reads tee'd JSONL and emits the same Bash evidence shape
    :class:`FileSystemLogProvider` produces (AC#7).
  * Native-log probe returns ``None`` when the candidates contain no
    thread-keyed JSONL — provider falls back to the tee path (AC#15).
  * When a mock native log directory is supplied, ``get_log_path``
    resolves to the native location (AC#15).
  * First-invocation tee bootstrap: after the pending file is renamed to
    ``{thread_id}.jsonl`` (the rename is owned by ``AmpClient``),
    ``get_log_path(repo_path, thread_id)`` resolves to the renamed file.
  * Resume appends to the existing thread file: events from both
    invocations are present in file order (AC#7a).
  * ``iter_events()`` reads across invocations: a fixture file containing
    events written by two simulated invocations yields events in file
    order spanning both (AC#7a).
  * Cross-resume validation evidence (regression for the log-stitching
    finding): invocation 1 logs Bash ``tool_use`` events for
    ``pytest`` / ``ruff check`` / ``ty check``; invocation 2 logs only
    new events. ``iter_events()`` still surfaces the original lint /
    test / typecheck events so the gate's evidence parser observes them.
  * Missing tee file → empty iterator (no raise).
  * Malformed trailing line tolerance: yield all preceding events; warn.
  * Orphan ``.pending-*`` file ignored by thread-keyed reads.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from src.infra.clients.amp_log_provider import (
    AMP_SESSIONS_DIR,
    AmpLogProvider,
    _discover_native_log_dir,
)
from src.infra.io.session_log_parser import FileSystemLogProvider

if TYPE_CHECKING:
    from collections.abc import Iterable


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _system_init(thread_id: str = "T-abc123") -> dict[str, object]:
    return {"type": "system", "subtype": "init", "session_id": thread_id}


def _assistant_bash(tool_id: str, command: str) -> dict[str, object]:
    return {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": "Bash",
                    "input": {"command": command},
                }
            ]
        },
    }


def _assistant_amp_bash(tool_id: str, command: str) -> dict[str, object]:
    return {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": "Bash",
                    "input": {"cmd": command},
                }
            ]
        },
    }


def _assistant_amp_shell_command(tool_id: str, command: str) -> dict[str, object]:
    return {
        "type": "assistant",
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": "shell_command",
                    "input": {"command": command},
                }
            ]
        },
    }


def _user_tool_result(
    tool_use_id: str, content: str = "ok", is_error: bool = False
) -> dict[str, object]:
    return {
        "type": "user",
        "message": {
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content,
                    "is_error": is_error,
                }
            ]
        },
    }


def _result(thread_id: str = "T-abc123") -> dict[str, object]:
    return {
        "type": "result",
        "subtype": "success",
        "session_id": thread_id,
        "result": "done",
    }


def _write_jsonl(path: Path, events: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event))
            fh.write("\n")


def _append_jsonl(path: Path, events: Iterable[dict[str, object]]) -> None:
    """Append events as if a second ``amp`` invocation tee'd to the file."""
    with path.open("a", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event))
            fh.write("\n")


@pytest.mark.unit
def test_default_log_path_is_amp_sessions_dir() -> None:
    provider = AmpLogProvider()

    path = provider.get_log_path(Path("/tmp/repo"), "T-abc123")

    assert path == AMP_SESSIONS_DIR / "T-abc123.jsonl"


# ---------------------------------------------------------------------------
# Bash-evidence shape parity with FileSystemLogProvider
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_iter_events_emits_same_bash_shape_as_filesystem_provider(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "T-thread.jsonl"
    _write_jsonl(
        log_path,
        [
            _system_init("T-thread"),
            _assistant_bash("tool-1", "uv run pytest -q"),
            _user_tool_result("tool-1", "passed", is_error=False),
            _result("T-thread"),
        ],
    )

    amp = AmpLogProvider()
    fs = FileSystemLogProvider()

    amp_bash: list[tuple[str, str]] = []
    for entry in amp.iter_events(log_path):
        amp_bash.extend(amp.extract_bash_commands(entry))

    fs_bash: list[tuple[str, str]] = []
    for entry in fs.iter_events(log_path):
        fs_bash.extend(fs.extract_bash_commands(entry))

    assert amp_bash == [("tool-1", "uv run pytest -q")]
    assert amp_bash == fs_bash


@pytest.mark.unit
def test_iter_events_extracts_amp_bash_cmd_input(tmp_path: Path) -> None:
    log_path = tmp_path / "T-amp-cmd.jsonl"
    _write_jsonl(
        log_path,
        [
            _system_init("T-amp-cmd"),
            _assistant_amp_bash("tool-1", "uvx ruff check ."),
            _assistant_amp_bash("tool-2", "uvx ty check"),
        ],
    )

    provider = AmpLogProvider()

    commands = [
        command
        for entry in provider.iter_events(log_path)
        for (_tool_id, command) in provider.extract_bash_commands(entry)
    ]

    assert commands == ["uvx ruff check .", "uvx ty check"]


@pytest.mark.unit
def test_iter_events_extracts_amp_shell_command_tool(tmp_path: Path) -> None:
    log_path = tmp_path / "T-amp-shell-command.jsonl"
    _write_jsonl(
        log_path,
        [
            _system_init("T-amp-shell-command"),
            _assistant_amp_shell_command("tool-1", "uvx ruff check ."),
            _assistant_amp_shell_command("tool-2", "uvx ty check"),
        ],
    )

    provider = AmpLogProvider()

    commands = [
        command
        for entry in provider.iter_events(log_path)
        for (_tool_id, command) in provider.extract_bash_commands(entry)
    ]

    assert commands == ["uvx ruff check .", "uvx ty check"]


@pytest.mark.unit
def test_iter_events_emits_tool_results_and_text_blocks(tmp_path: Path) -> None:
    log_path = tmp_path / "T-mix.jsonl"
    _write_jsonl(
        log_path,
        [
            _system_init("T-mix"),
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "running tests"},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Bash",
                            "input": {"command": "uvx ruff check ."},
                        },
                    ]
                },
            },
            _user_tool_result("t1", "ok", is_error=False),
        ],
    )

    provider = AmpLogProvider()
    entries = list(provider.iter_events(log_path))

    assistant_texts = [
        text for e in entries for text in provider.extract_assistant_text_blocks(e)
    ]
    tool_results = [r for e in entries for r in provider.extract_tool_results(e)]

    assert "running tests" in assistant_texts
    assert ("t1", False) in tool_results


# ---------------------------------------------------------------------------
# Native-log probe
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_probe_returns_none_when_no_native_log(tmp_path: Path) -> None:
    candidate_a = tmp_path / "config-amp"
    candidate_b = tmp_path / "share-amp"
    candidate_a.mkdir()
    candidate_b.mkdir()

    discovered = _discover_native_log_dir((candidate_a, candidate_b))

    assert discovered is None


@pytest.mark.unit
def test_probe_returns_none_when_paths_missing(tmp_path: Path) -> None:
    discovered = _discover_native_log_dir(
        (tmp_path / "missing-a", tmp_path / "missing-b")
    )
    assert discovered is None


@pytest.mark.unit
def test_probe_returns_dir_when_thread_keyed_jsonl_present(tmp_path: Path) -> None:
    native_dir = tmp_path / "config-amp"
    native_dir.mkdir()
    (native_dir / "T-discovered.jsonl").write_text(
        json.dumps(_system_init("T-discovered")) + "\n", encoding="utf-8"
    )

    discovered = _discover_native_log_dir((tmp_path / "missing", native_dir))

    assert discovered == native_dir


@pytest.mark.unit
def test_from_probe_falls_back_to_tee_when_no_native(tmp_path: Path) -> None:
    provider = AmpLogProvider.from_probe(
        search_paths=(tmp_path / "config-amp", tmp_path / "share-amp")
    )

    assert provider.native_dir is None
    assert provider.get_log_path(tmp_path, "T-x") == AMP_SESSIONS_DIR / "T-x.jsonl"


@pytest.mark.unit
def test_from_probe_prefers_native_when_discovered(tmp_path: Path) -> None:
    native_dir = tmp_path / "config-amp"
    native_dir.mkdir()
    (native_dir / "T-found.jsonl").write_text("{}\n", encoding="utf-8")

    provider = AmpLogProvider.from_probe(search_paths=(native_dir,))

    assert provider.native_dir == native_dir
    assert provider.get_log_path(tmp_path, "T-found") == native_dir / "T-found.jsonl"


@pytest.mark.unit
def test_explicit_native_dir_overrides_default(tmp_path: Path) -> None:
    native_dir = tmp_path / "elsewhere"
    native_dir.mkdir()

    provider = AmpLogProvider(native_dir=native_dir)

    assert provider.native_dir == native_dir
    assert provider.get_log_path(tmp_path, "T-z") == native_dir / "T-z.jsonl"


# ---------------------------------------------------------------------------
# Tee bootstrap + resume + cross-invocation reads
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_log_path_resolves_to_renamed_thread_file(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "amp-sessions"
    sessions_dir.mkdir()
    thread_file = sessions_dir / "T-bootstrap.jsonl"
    _write_jsonl(thread_file, [_system_init("T-bootstrap")])

    # AmpLogProvider with explicit native_dir simulates the same path
    # resolution semantics: thread-keyed file under a known directory.
    provider = AmpLogProvider(native_dir=sessions_dir)

    resolved = provider.get_log_path(tmp_path, "T-bootstrap")

    assert resolved == thread_file
    assert resolved.exists()


@pytest.mark.unit
def test_iter_events_reads_across_two_invocations(tmp_path: Path) -> None:
    log_path = tmp_path / "T-resume.jsonl"

    # Invocation 1: full lint/test/typecheck Bash evidence, then result.
    _write_jsonl(
        log_path,
        [
            _system_init("T-resume"),
            _assistant_bash("t1", "uv run pytest -q"),
            _user_tool_result("t1", "passed"),
            _assistant_bash("t2", "uvx ruff check ."),
            _user_tool_result("t2", "ok"),
            _assistant_bash("t3", "uvx ty check"),
            _user_tool_result("t3", "ok"),
            _result("T-resume"),
        ],
    )
    # Invocation 2 (resume): only new events, simulating delta-only resume.
    _append_jsonl(
        log_path,
        [
            _system_init("T-resume"),
            _assistant_bash("t4", "git commit -m 'wip'"),
            _user_tool_result("t4", "committed"),
            _result("T-resume"),
        ],
    )

    provider = AmpLogProvider(native_dir=tmp_path)

    entries = list(provider.iter_events(log_path))
    bash_commands = [
        cmd for e in entries for (_id, cmd) in provider.extract_bash_commands(e)
    ]

    # File-order across invocations is preserved.
    assert bash_commands == [
        "uv run pytest -q",
        "uvx ruff check .",
        "uvx ty check",
        "git commit -m 'wip'",
    ]


@pytest.mark.unit
def test_iter_events_honors_caller_offset_for_resolution_scoping(
    tmp_path: Path,
) -> None:
    """``iter_events`` honors caller offset (resolution-marker scoping).

    The gate's resolution-marker parser passes
    ``retry_state.log_offset`` so that a stale ``ISSUE_*`` marker
    emitted in invocation 1 (and rejected by the gate) does not
    override the latest invocation's decision on retry. This is the
    standard ``LogProvider.iter_events`` contract; AmpLogProvider
    matches FileSystemLogProvider here.

    For cross-invocation evidence-persistence reads the caller must
    use :meth:`AmpLogProvider.iter_thread_events`; that case is
    covered separately in
    ``test_iter_thread_events_ignores_caller_offset_for_evidence``.
    """
    log_path = tmp_path / "T-offset-honor.jsonl"

    invocation_1 = [
        _system_init("T-offset-honor"),
        _assistant_bash("v1", "uv run pytest -q"),
        _user_tool_result("v1", "1 passed"),
        _result("T-offset-honor"),
    ]
    _write_jsonl(log_path, invocation_1)
    end_of_invocation_1 = log_path.stat().st_size

    invocation_2 = [
        _assistant_bash("w1", "echo continuing"),
        _user_tool_result("w1", "continuing"),
        _result("T-offset-honor"),
    ]
    _append_jsonl(log_path, invocation_2)

    provider = AmpLogProvider(native_dir=tmp_path)

    entries = list(provider.iter_events(log_path, offset=end_of_invocation_1))
    commands = [
        cmd for e in entries for (_id, cmd) in provider.extract_bash_commands(e)
    ]

    assert "uv run pytest -q" not in commands
    assert "echo continuing" in commands


@pytest.mark.unit
def test_iter_thread_events_ignores_caller_offset_for_evidence(
    tmp_path: Path,
) -> None:
    """Regression for cross-resume evidence persistence (AC#7a).

    Validation-evidence parsing calls
    :meth:`LogProvider.iter_thread_events`, which on Amp ignores the
    caller-supplied offset and reads from byte 0. Without this, the
    gate retry path's advanced ``retry_state.log_offset`` would skip
    invocation 1's lint/test/typecheck Bash events when invocation 2
    appends to the same per-thread file.
    """
    log_path = tmp_path / "T-thread-evidence.jsonl"

    invocation_1 = [
        _system_init("T-thread-evidence"),
        _assistant_bash("v1", "uv run pytest -q"),
        _user_tool_result("v1", "1 passed"),
        _assistant_bash("v2", "uvx ruff check ."),
        _user_tool_result("v2", "ok"),
        _assistant_bash("v3", "uvx ty check"),
        _user_tool_result("v3", "Success"),
        _result("T-thread-evidence"),
    ]
    _write_jsonl(log_path, invocation_1)
    end_of_invocation_1 = log_path.stat().st_size

    invocation_2 = [
        _assistant_bash("w1", "echo continuing"),
        _user_tool_result("w1", "continuing"),
        _result("T-thread-evidence"),
    ]
    _append_jsonl(log_path, invocation_2)

    provider = AmpLogProvider(native_dir=tmp_path)

    entries = list(provider.iter_thread_events(log_path, offset=end_of_invocation_1))
    commands = [
        cmd for e in entries for (_id, cmd) in provider.extract_bash_commands(e)
    ]

    assert "uv run pytest -q" in commands
    assert "uvx ruff check ." in commands
    assert "uvx ty check" in commands
    assert "echo continuing" in commands


@pytest.mark.unit
def test_cross_resume_validation_evidence_persists(tmp_path: Path) -> None:
    """Regression for log-stitching finding: invocation 1 evidence persists.

    Invocation 1 logs Bash tool_use events for pytest / ruff check / ty check.
    Invocation 2 (delta-only resume) appends new events. ``iter_events`` must
    still surface the original lint / test / typecheck events so the gate's
    evidence parser observes them — i.e., a delta-only resume does not lose
    validation evidence (AC#7a).
    """
    log_path = tmp_path / "T-cross.jsonl"

    invocation_1 = [
        _system_init("T-cross"),
        _assistant_bash("a1", "uv run pytest -q"),
        _user_tool_result("a1", "1 passed"),
        _assistant_bash("a2", "uvx ruff check ."),
        _user_tool_result("a2", "All checks passed"),
        _assistant_bash("a3", "uvx ty check"),
        _user_tool_result("a3", "Success"),
        _result("T-cross"),
    ]
    _write_jsonl(log_path, invocation_1)

    invocation_2_delta_only = [
        _assistant_bash("b1", "echo continuing"),
        _user_tool_result("b1", "continuing"),
        _result("T-cross"),
    ]
    _append_jsonl(log_path, invocation_2_delta_only)

    provider = AmpLogProvider(native_dir=tmp_path)
    entries = list(provider.iter_events(log_path))
    commands = [
        cmd for e in entries for (_id, cmd) in provider.extract_bash_commands(e)
    ]

    # All three validation-evidence commands from invocation 1 still
    # surface, even though invocation 2 was delta-only.
    assert "uv run pytest -q" in commands
    assert "uvx ruff check ." in commands
    assert "uvx ty check" in commands
    # Invocation 2's new event also surfaces in file order after them.
    assert commands.index("uv run pytest -q") < commands.index("echo continuing")


# ---------------------------------------------------------------------------
# Missing files / malformed lines / orphan pending files
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_iter_events_missing_file_yields_empty(tmp_path: Path) -> None:
    provider = AmpLogProvider(native_dir=tmp_path)
    log_path = tmp_path / "T-never-written.jsonl"

    # Must not raise and must yield nothing.
    entries = list(provider.iter_events(log_path))

    assert entries == []


@pytest.mark.unit
def test_iter_events_tolerates_malformed_trailing_line(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    log_path = tmp_path / "T-corrupt.jsonl"

    # Two well-formed events followed by a truncated trailing line.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(_system_init("T-corrupt")) + "\n")
        fh.write(json.dumps(_assistant_bash("c1", "echo hello")) + "\n")
        # Truncated mid-object; missing closing brace, no newline.
        fh.write('{"type":"assistant","message":{"content":[{"type":"text"')

    provider = AmpLogProvider(native_dir=tmp_path)

    with caplog.at_level(logging.WARNING, logger="src.infra.clients.amp_log_provider"):
        entries = list(provider.iter_events(log_path))

    bash = [cmd for e in entries for (_id, cmd) in provider.extract_bash_commands(e)]
    assert bash == ["echo hello"]
    # warn-level log records the partial line.
    assert any(
        "malformed JSONL line" in record.getMessage()
        and record.levelno == logging.WARNING
        for record in caplog.records
    )


@pytest.mark.unit
def test_orphan_pending_file_does_not_interfere_with_thread_reads(
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "amp-sessions"
    sessions_dir.mkdir()

    # A leftover ``.pending-<uuid>.jsonl`` from a crashed run, plus a real
    # thread file. The provider's thread-keyed read must hit only the
    # thread file; pending-prefix files are never resolved by
    # ``get_log_path`` (filename derives from session_id).
    orphan = sessions_dir / ".pending-deadbeef.jsonl"
    _write_jsonl(orphan, [_assistant_bash("orphan", "leftover-event")])

    thread_file = sessions_dir / "T-real.jsonl"
    _write_jsonl(
        thread_file,
        [
            _system_init("T-real"),
            _assistant_bash("real-1", "uv run pytest -q"),
            _user_tool_result("real-1", "ok"),
        ],
    )

    provider = AmpLogProvider(native_dir=sessions_dir)
    resolved = provider.get_log_path(tmp_path, "T-real")
    assert resolved == thread_file

    bash = [
        cmd
        for e in provider.iter_events(resolved)
        for (_id, cmd) in provider.extract_bash_commands(e)
    ]
    # Only the real thread file's events surface.
    assert bash == ["uv run pytest -q"]


@pytest.mark.unit
def test_get_end_offset_matches_file_size(tmp_path: Path) -> None:
    log_path = tmp_path / "T-offset.jsonl"
    _write_jsonl(log_path, [_system_init("T-offset"), _result("T-offset")])

    provider = AmpLogProvider(native_dir=tmp_path)
    assert provider.get_end_offset(log_path) == log_path.stat().st_size


@pytest.mark.unit
def test_get_end_offset_missing_file_returns_start(tmp_path: Path) -> None:
    provider = AmpLogProvider(native_dir=tmp_path)
    assert provider.get_end_offset(tmp_path / "absent.jsonl", start_offset=42) == 42

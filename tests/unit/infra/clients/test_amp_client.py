"""Unit tests for :class:`src.infra.clients.amp_client.AmpClient`.

Covers the test cases from the plan's Testing & Validation section
(``plans/2026-04-29-amp-provider-plan.md#L734-L744`` and L829-L835):

  * ``system(init)`` captures session/thread id and writes tee header.
  * ``assistant`` events with ``text``/``tool_use``/``thinking`` map to
    a synthetic :class:`AssistantMessage` (thinking stripped in MVP).
  * ``user`` events with ``tool_result`` map to a synthetic
    :class:`AssistantMessage` containing :class:`ToolResultBlock` blocks.
  * ``result`` (success / error_during_execution / error_max_turns) maps
    to a synthetic :class:`ResultMessage` with the right ``session_id``
    and ``result``.
  * Stderr ring buffer is bounded; auth-error classification fires on
    ``unauthorized`` / ``401`` / ``AMP_API_KEY``.
  * Subprocess lifecycle: spawn → write prompt to stdin → terminate on
    ``__aexit__`` (SIGTERM → grace → SIGKILL).
  * Resume: ``with_resume(thread_id)`` produces argv stable across the
    two candidate shapes plus a documented fallback (parametrized).
  * Tee'd log produced exactly once; first-run pending file is renamed
    to ``{thread_id}.jsonl`` atomically; resume appends to existing.
  * Cancellation mid-stream cleans up subprocess and tee handle.
  * Malformed JSON, premature exit, missing ``result`` events handled.
  * ``AmpStreamMissingInitError`` raised when ``assistant``/``user``/
    ``result`` arrive before ``system(init)``; pending file preserved;
    error carries truncated stderr/stdout.

Tests use a Python "fake amp" script that prints canned stream-json on
stdout and optional canned stderr. This keeps the unit suite isolated
(no network, no real Amp install) while exercising the real subprocess
path.
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
from typing import TYPE_CHECKING, cast

import pytest

from src.core.protocols.sdk import SDKClientProtocol
from src.infra.clients.amp_client import (
    AmpClient,
    AmpClientError,
    AmpClientOptions,
    AmpStreamMissingInitError,
)
from src.infra.clients.amp_messages import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Sequence
    from pathlib import Path

    from src.infra.clients.amp_client import ResumeStrategy


# ---------------------------------------------------------------------------
# Fake-amp script helpers: spawn a Python subprocess that emits canned
# stream-json on stdout. Writing the script to a tmp file and running it
# under ``sys.executable`` keeps tests OS-portable.
# ---------------------------------------------------------------------------


_FAKE_AMP_TEMPLATE = """\
import json, os, sys, time

LINES = {lines!r}
STDERR = {stderr!r}
EXIT_CODE = {exit_code}
DELAY = {delay}
HANG = {hang}
ECHO_STDIN_TO_PATH = {echo_stdin_to_path!r}
RECORD_ARGV_TO = {record_argv_to!r}

if RECORD_ARGV_TO:
    with open(RECORD_ARGV_TO, "w") as fh:
        json.dump(sys.argv, fh)

if ECHO_STDIN_TO_PATH:
    data = sys.stdin.read()
    with open(ECHO_STDIN_TO_PATH, "w") as fh:
        fh.write(data)
else:
    # Drain stdin so the writer's drain() resolves.
    try:
        sys.stdin.read()
    except Exception:
        pass

if STDERR:
    sys.stderr.write(STDERR)
    sys.stderr.flush()

for line in LINES:
    sys.stdout.write(line + "\\n")
    sys.stdout.flush()
    if DELAY:
        time.sleep(DELAY)

if HANG:
    # Sleep forever; test-side termination must reap us.
    while True:
        time.sleep(1)

sys.exit(EXIT_CODE)
"""


def _write_fake_amp(
    tmp_path: Path,
    *,
    lines: Sequence[str] = (),
    stderr: str = "",
    exit_code: int = 0,
    delay: float = 0.0,
    hang: bool = False,
    echo_stdin_to: Path | None = None,
    record_argv_to: Path | None = None,
) -> Path:
    """Write a fake-amp Python script and return its path."""
    script = tmp_path / f"fake_amp_{abs(hash((tuple(lines), stderr, exit_code)))}.py"
    script.write_text(
        _FAKE_AMP_TEMPLATE.format(
            lines=list(lines),
            stderr=stderr,
            exit_code=exit_code,
            delay=delay,
            hang=hang,
            echo_stdin_to_path=str(echo_stdin_to) if echo_stdin_to else None,
            record_argv_to=str(record_argv_to) if record_argv_to else None,
        )
    )
    st = script.stat()
    script.chmod(st.st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return script


def _make_options(
    *,
    log_path: Path,
    argv: Sequence[str],
    cwd: Path,
    thread_id: str | None = None,
    resume_strategy: ResumeStrategy = "thread-id-flag",
    kill_grace_seconds: float = 0.5,
) -> AmpClientOptions:
    return AmpClientOptions(
        cwd=cwd,
        env=dict(os.environ),
        argv=tuple(argv),
        log_path=log_path,
        thread_id=thread_id,
        resume_strategy=resume_strategy,
        kill_grace_seconds=kill_grace_seconds,
    )


def _python_argv_for(script: Path) -> list[str]:
    return [sys.executable, str(script)]


# ---------------------------------------------------------------------------
# Sample stream-json events
# ---------------------------------------------------------------------------


def _system_init(session_id: str = "T-abc123") -> str:
    return json.dumps({"type": "system", "subtype": "init", "session_id": session_id})


def _assistant_text(text: str = "hi") -> str:
    return json.dumps(
        {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": text}]},
        }
    )


def _assistant_tool_use(
    tool_id: str = "tool-1",
    name: str = "Bash",
    inp: dict[str, object] | None = None,
) -> str:
    return json.dumps(
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": tool_id,
                        "name": name,
                        "input": inp or {"command": "ls"},
                    }
                ]
            },
        }
    )


def _assistant_thinking(text: str = "ponder") -> str:
    return json.dumps(
        {
            "type": "assistant",
            "message": {"content": [{"type": "thinking", "text": text}]},
        }
    )


def _user_tool_result(
    tool_use_id: str = "tool-1",
    content: object = "ok",
    is_error: bool = False,
) -> str:
    return json.dumps(
        {
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
    )


def _result(
    session_id: str = "T-abc123",
    subtype: str = "success",
    result: object = "done",
) -> str:
    return json.dumps(
        {
            "type": "result",
            "subtype": subtype,
            "session_id": session_id,
            "result": result,
        }
    )


# ---------------------------------------------------------------------------
# Helpers to drive an AmpClient end-to-end
# ---------------------------------------------------------------------------


async def _drain(client: AmpClient) -> list[object]:
    out: list[object] = []
    async for msg in client.receive_response():
        out.append(msg)
    return out


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_amp_client_conforms_to_sdk_client_protocol(tmp_path: Path) -> None:
    options = _make_options(
        log_path=tmp_path / ".pending-x.jsonl",
        argv=["true"],
        cwd=tmp_path,
    )
    client = AmpClient(options)
    assert isinstance(client, SDKClientProtocol)


# ---------------------------------------------------------------------------
# system(init) capture + tee finalize
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_system_init_captures_session_id_and_renames_pending(
    tmp_path: Path,
) -> None:
    pending = tmp_path / ".pending-abc.jsonl"
    script = _write_fake_amp(
        tmp_path,
        lines=[_system_init("T-abc123"), _result("T-abc123")],
    )
    options = _make_options(
        log_path=pending, argv=_python_argv_for(script), cwd=tmp_path
    )
    async with AmpClient(options) as client:
        await client.query("hello")
        await _drain(client)
        assert client.session_id == "T-abc123"
        assert client.thread_id == "T-abc123"

    # Pending file renamed to {thread_id}.jsonl in the same directory.
    target = tmp_path / "T-abc123.jsonl"
    assert target.exists()
    assert not pending.exists()
    body = target.read_text().splitlines()
    assert any(json.loads(line).get("type") == "system" for line in body)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_resume_appends_to_existing_thread_file(tmp_path: Path) -> None:
    """Resume opens an existing ``{thread_id}.jsonl`` in append mode."""
    target = tmp_path / "T-resume.jsonl"
    target.write_text(json.dumps({"type": "preexisting"}) + "\n")

    script = _write_fake_amp(
        tmp_path,
        lines=[_system_init("T-resume"), _result("T-resume")],
    )
    options = _make_options(
        log_path=target,
        argv=_python_argv_for(script),
        cwd=tmp_path,
        thread_id="T-resume",
    )
    async with AmpClient(options) as client:
        await client.query("again")
        await _drain(client)

    lines = target.read_text().splitlines()
    parsed = [json.loads(ln) for ln in lines]
    assert parsed[0] == {"type": "preexisting"}
    assert any(p.get("type") == "system" for p in parsed)
    assert any(p.get("type") == "result" for p in parsed)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_pending_merges_into_existing_thread_file(tmp_path: Path) -> None:
    """If a pending session lands and ``{thread_id}.jsonl`` already exists,
    pending bytes are appended into the target and pending unlinked."""
    target = tmp_path / "T-existing.jsonl"
    target.write_text("EARLIER\n")
    pending = tmp_path / ".pending-merge.jsonl"

    script = _write_fake_amp(
        tmp_path,
        lines=[_system_init("T-existing"), _result("T-existing")],
    )
    options = _make_options(
        log_path=pending, argv=_python_argv_for(script), cwd=tmp_path
    )
    async with AmpClient(options) as client:
        await client.query("hi")
        await _drain(client)

    assert not pending.exists()
    body = target.read_text()
    assert body.startswith("EARLIER\n")
    assert '"type": "system"' in body or '"type":"system"' in body
    assert '"type": "result"' in body or '"type":"result"' in body


# ---------------------------------------------------------------------------
# Event mapping: assistant text / tool_use / thinking
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_assistant_text_emits_assistant_message_with_textblock(
    tmp_path: Path,
) -> None:
    script = _write_fake_amp(
        tmp_path,
        lines=[_system_init(), _assistant_text("hello world"), _result()],
    )
    options = _make_options(
        log_path=tmp_path / ".pending-1.jsonl",
        argv=_python_argv_for(script),
        cwd=tmp_path,
    )
    async with AmpClient(options) as client:
        await client.query("p")
        msgs = await _drain(client)

    assistants = [m for m in msgs if isinstance(m, AssistantMessage)]
    assert len(assistants) == 1
    blocks = assistants[0].content
    assert len(blocks) == 1
    assert isinstance(blocks[0], TextBlock)
    assert blocks[0].text == "hello world"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_assistant_tool_use_emits_tool_use_block(tmp_path: Path) -> None:
    script = _write_fake_amp(
        tmp_path,
        lines=[
            _system_init(),
            _assistant_tool_use(tool_id="t-7", name="Bash", inp={"command": "ls -la"}),
            _result(),
        ],
    )
    options = _make_options(
        log_path=tmp_path / ".pending-2.jsonl",
        argv=_python_argv_for(script),
        cwd=tmp_path,
    )
    async with AmpClient(options) as client:
        await client.query("p")
        msgs = await _drain(client)

    assistants = [m for m in msgs if isinstance(m, AssistantMessage)]
    assert len(assistants) == 1
    blocks = assistants[0].content
    assert len(blocks) == 1
    block = blocks[0]
    assert isinstance(block, ToolUseBlock)
    assert block.id == "t-7"
    assert block.name == "Bash"
    assert block.input == {"command": "ls -la"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_assistant_thinking_block_is_stripped(tmp_path: Path) -> None:
    """Thinking blocks tee'd for diagnostics; not surfaced in MVP."""
    # An assistant event containing ONLY thinking yields no AssistantMessage.
    script = _write_fake_amp(
        tmp_path,
        lines=[_system_init(), _assistant_thinking("…"), _result()],
    )
    options = _make_options(
        log_path=tmp_path / ".pending-3.jsonl",
        argv=_python_argv_for(script),
        cwd=tmp_path,
    )
    async with AmpClient(options) as client:
        await client.query("p")
        msgs = await _drain(client)

    assistants = [m for m in msgs if isinstance(m, AssistantMessage)]
    assert assistants == []
    # ResultMessage still produced.
    assert any(isinstance(m, ResultMessage) for m in msgs)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_user_tool_result_emits_assistant_message_with_tool_result_block(
    tmp_path: Path,
) -> None:
    script = _write_fake_amp(
        tmp_path,
        lines=[
            _system_init(),
            _user_tool_result(tool_use_id="t-9", content="output", is_error=False),
            _result(),
        ],
    )
    options = _make_options(
        log_path=tmp_path / ".pending-4.jsonl",
        argv=_python_argv_for(script),
        cwd=tmp_path,
    )
    async with AmpClient(options) as client:
        await client.query("p")
        msgs = await _drain(client)

    assistants = [m for m in msgs if isinstance(m, AssistantMessage)]
    assert len(assistants) == 1
    blocks = assistants[0].content
    assert len(blocks) == 1
    block = blocks[0]
    assert isinstance(block, ToolResultBlock)
    assert block.tool_use_id == "t-9"
    assert block.content == "output"
    assert block.is_error is False


# ---------------------------------------------------------------------------
# result event mapping
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "subtype",
    ["success", "error_during_execution", "error_max_turns"],
)
async def test_result_subtypes_emit_result_message(
    tmp_path: Path, subtype: str
) -> None:
    script = _write_fake_amp(
        tmp_path,
        lines=[
            _system_init("T-zzz"),
            _result(session_id="T-zzz", subtype=subtype, result="payload"),
        ],
    )
    options = _make_options(
        log_path=tmp_path / f".pending-{subtype}.jsonl",
        argv=_python_argv_for(script),
        cwd=tmp_path,
    )
    async with AmpClient(options) as client:
        await client.query("p")
        msgs = await _drain(client)

    results = [m for m in msgs if isinstance(m, ResultMessage)]
    assert len(results) == 1
    assert results[0].session_id == "T-zzz"
    assert results[0].result == "payload"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_result_falls_back_to_subtype_when_result_field_missing(
    tmp_path: Path,
) -> None:
    """Some Amp error subtypes omit ``result``; the synthetic ResultMessage
    surfaces ``subtype`` instead so the orchestrator can branch on a stable
    string."""
    raw = json.dumps(
        {"type": "result", "subtype": "error_max_turns", "session_id": "T-x"}
    )
    script = _write_fake_amp(tmp_path, lines=[_system_init("T-x"), raw])
    options = _make_options(
        log_path=tmp_path / ".pending-em.jsonl",
        argv=_python_argv_for(script),
        cwd=tmp_path,
    )
    async with AmpClient(options) as client:
        await client.query("p")
        msgs = await _drain(client)

    results = [m for m in msgs if isinstance(m, ResultMessage)]
    assert len(results) == 1
    assert results[0].result == "error_max_turns"


# ---------------------------------------------------------------------------
# Unknown event types: warn-level, do not crash
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unknown_event_type_does_not_crash(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    unknown = json.dumps({"type": "future_event", "data": "whatever"})
    script = _write_fake_amp(
        tmp_path,
        lines=[_system_init(), unknown, _result()],
    )
    options = _make_options(
        log_path=tmp_path / ".pending-u.jsonl",
        argv=_python_argv_for(script),
        cwd=tmp_path,
    )
    import logging

    with caplog.at_level(logging.WARNING):
        async with AmpClient(options) as client:
            await client.query("p")
            msgs = await _drain(client)

    assert any(isinstance(m, ResultMessage) for m in msgs)
    assert any("future_event" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# Stderr ring buffer + auth error classification
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stderr_ring_buffer_is_bounded(tmp_path: Path) -> None:
    """Stderr buffer never exceeds the documented 4 KiB bound."""
    big = "x" * 200_000
    script = _write_fake_amp(
        tmp_path,
        lines=[_system_init(), _result()],
        stderr=big,
    )
    options = _make_options(
        log_path=tmp_path / ".pending-s.jsonl",
        argv=_python_argv_for(script),
        cwd=tmp_path,
    )
    async with AmpClient(options) as client:
        await client.query("p")
        await _drain(client)
        # Give stderr collector a chance to drain.
        await asyncio.sleep(0.05)
        text = client.get_stderr()

    assert len(text.encode()) <= 4096


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "marker",
    ["unauthorized", "401", "AMP_API_KEY", "forbidden"],
)
async def test_auth_error_classification(tmp_path: Path, marker: str) -> None:
    script = _write_fake_amp(
        tmp_path,
        lines=[_system_init(), _result()],
        stderr=f"some preamble; {marker}; trailing",
    )
    options = _make_options(
        log_path=tmp_path / f".pending-auth-{marker}.jsonl",
        argv=_python_argv_for(script),
        cwd=tmp_path,
    )
    async with AmpClient(options) as client:
        await client.query("p")
        await _drain(client)
        # Wait for stderr collector.
        for _ in range(20):
            if client.is_auth_error():
                break
            await asyncio.sleep(0.01)

    assert client.is_auth_error() is True


@pytest.mark.unit
@pytest.mark.asyncio
async def test_no_auth_error_when_stderr_clean(tmp_path: Path) -> None:
    script = _write_fake_amp(tmp_path, lines=[_system_init(), _result()])
    options = _make_options(
        log_path=tmp_path / ".pending-clean.jsonl",
        argv=_python_argv_for(script),
        cwd=tmp_path,
    )
    async with AmpClient(options) as client:
        await client.query("p")
        await _drain(client)
        await asyncio.sleep(0.05)
        assert client.is_auth_error() is False


# ---------------------------------------------------------------------------
# Subprocess lifecycle: prompt to stdin + termination on __aexit__
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_query_writes_prompt_verbatim_to_stdin(tmp_path: Path) -> None:
    stdin_capture = tmp_path / "stdin.txt"
    script = _write_fake_amp(
        tmp_path,
        lines=[_system_init(), _result()],
        echo_stdin_to=stdin_capture,
    )
    options = _make_options(
        log_path=tmp_path / ".pending-stdin.jsonl",
        argv=_python_argv_for(script),
        cwd=tmp_path,
    )
    prompt = "hello\nworld\n# very specific bytes"
    async with AmpClient(options) as client:
        await client.query(prompt)
        await _drain(client)

    assert stdin_capture.read_text() == prompt


@pytest.mark.unit
@pytest.mark.asyncio
async def test_aexit_terminates_hanging_subprocess(tmp_path: Path) -> None:
    """A hanging amp process is reaped by ``__aexit__`` (SIGTERM/SIGKILL)."""
    script = _write_fake_amp(
        tmp_path,
        # Emit init + one assistant message so ``__anext__`` returns; then
        # the script hangs forever and ``__aexit__`` must reap it.
        lines=[_system_init(), _assistant_text("hi")],
        hang=True,
    )
    options = _make_options(
        log_path=tmp_path / ".pending-hang.jsonl",
        argv=_python_argv_for(script),
        cwd=tmp_path,
        kill_grace_seconds=0.3,
    )
    client = AmpClient(options)
    async with client:
        await client.query("p")
        gen = cast("AsyncGenerator[object, None]", client.receive_response())
        # Pull the assistant message, then abort the iterator while the
        # subprocess is still hanging.
        first = await asyncio.wait_for(gen.__anext__(), timeout=2.0)
        assert isinstance(first, AssistantMessage)
        await gen.aclose()

    proc = client._state.proc
    assert proc is not None
    assert proc.returncode is not None  # reaped


# ---------------------------------------------------------------------------
# Resume argv shapes (parametrized across two candidate shapes + fallback)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_resume_thread_id_flag_appends_when_missing(tmp_path: Path) -> None:
    options = _make_options(
        log_path=tmp_path / ".pending-r1.jsonl",
        argv=["amp", "--execute", "--stream-json", "--dangerously-allow-all"],
        cwd=tmp_path,
        resume_strategy="thread-id-flag",
    )
    client = AmpClient(options).with_resume("T-resumed")
    argv = client._build_argv()
    assert "--thread-id" in argv
    idx = argv.index("--thread-id")
    assert argv[idx + 1] == "T-resumed"
    # Existing flags preserved.
    assert "--execute" in argv
    assert "--dangerously-allow-all" in argv


@pytest.mark.unit
def test_resume_thread_id_flag_replaces_existing_value(tmp_path: Path) -> None:
    options = _make_options(
        log_path=tmp_path / ".pending-r2.jsonl",
        argv=["amp", "--execute", "--stream-json", "--thread-id", "T-old"],
        cwd=tmp_path,
        resume_strategy="thread-id-flag",
    )
    client = AmpClient(options).with_resume("T-new")
    argv = client._build_argv()
    # Only one --thread-id and value is updated.
    assert argv.count("--thread-id") == 1
    idx = argv.index("--thread-id")
    assert argv[idx + 1] == "T-new"


@pytest.mark.unit
def test_resume_threads_continue_replaces_argv(tmp_path: Path) -> None:
    options = _make_options(
        log_path=tmp_path / ".pending-r3.jsonl",
        argv=[
            "amp",
            "--execute",
            "--stream-json",
            "--dangerously-allow-all",
            "--mode",
            "smart",
        ],
        cwd=tmp_path,
        resume_strategy="threads-continue",
    )
    client = AmpClient(options).with_resume("T-c")
    argv = client._build_argv()
    assert argv[:4] == ["amp", "threads", "continue", "T-c"]
    # Remaining flags preserved (excluding --execute).
    assert "--execute" not in argv
    assert "--stream-json" in argv
    assert "--dangerously-allow-all" in argv
    assert "--mode" in argv


@pytest.mark.unit
def test_resume_fallback_keeps_argv_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    options = _make_options(
        log_path=tmp_path / ".pending-r4.jsonl",
        argv=["amp", "--execute", "--stream-json"],
        cwd=tmp_path,
        resume_strategy="fallback",
    )
    client = AmpClient(options).with_resume("T-fb")
    import logging

    with caplog.at_level(logging.WARNING):
        argv = client._build_argv()
    assert argv == ["amp", "--execute", "--stream-json"]
    assert any("fallback" in rec.message for rec in caplog.records)


@pytest.mark.unit
def test_no_resume_means_argv_unchanged(tmp_path: Path) -> None:
    base = ["amp", "--execute", "--stream-json"]
    options = _make_options(
        log_path=tmp_path / ".pending-rn.jsonl",
        argv=list(base),
        cwd=tmp_path,
    )
    client = AmpClient(options)
    assert client._build_argv() == base


@pytest.mark.unit
def test_with_resume_returns_self_for_chaining(tmp_path: Path) -> None:
    options = _make_options(
        log_path=tmp_path / ".pending-rc.jsonl",
        argv=["amp"],
        cwd=tmp_path,
    )
    client = AmpClient(options)
    assert client.with_resume("T-x") is client


@pytest.mark.unit
@pytest.mark.asyncio
async def test_with_resume_argv_observed_by_subprocess(tmp_path: Path) -> None:
    """End-to-end: ``with_resume`` causes the spawned argv to include ``--thread-id``."""
    argv_record = tmp_path / "argv.json"
    script = _write_fake_amp(
        tmp_path,
        lines=[_system_init(), _result()],
        record_argv_to=argv_record,
    )
    base_argv = [
        *_python_argv_for(script),
        "--execute",
        "--stream-json",
        "--dangerously-allow-all",
    ]
    options = _make_options(
        log_path=tmp_path / ".pending-end.jsonl",
        argv=base_argv,
        cwd=tmp_path,
        resume_strategy="thread-id-flag",
    )
    async with AmpClient(options) as client:
        client.with_resume("T-e2e")
        await client.query("p")
        await _drain(client)

    spawned = json.loads(argv_record.read_text())
    assert "--thread-id" in spawned
    assert spawned[spawned.index("--thread-id") + 1] == "T-e2e"


# ---------------------------------------------------------------------------
# Tee'd log idempotence
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_tee_log_contains_each_event_exactly_once(tmp_path: Path) -> None:
    pending = tmp_path / ".pending-once.jsonl"
    script = _write_fake_amp(
        tmp_path,
        lines=[
            _system_init("T-once"),
            _assistant_text("a"),
            _user_tool_result(),
            _result("T-once"),
        ],
    )
    options = _make_options(
        log_path=pending, argv=_python_argv_for(script), cwd=tmp_path
    )
    async with AmpClient(options) as client:
        await client.query("p")
        await _drain(client)

    target = tmp_path / "T-once.jsonl"
    parsed = [json.loads(ln) for ln in target.read_text().splitlines() if ln.strip()]
    types = [p.get("type") for p in parsed]
    assert types.count("system") == 1
    assert types.count("assistant") == 1
    assert types.count("user") == 1
    assert types.count("result") == 1


# ---------------------------------------------------------------------------
# Cancellation mid-stream
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_cancellation_mid_stream_cleans_up(tmp_path: Path) -> None:
    """Aborting the receive_response generator before completion leaves no
    leaked subprocess and closes the tee handle."""
    script = _write_fake_amp(
        tmp_path,
        lines=[_system_init(), _assistant_text("partial")],
        delay=0.05,
        hang=True,
    )
    pending = tmp_path / ".pending-cx.jsonl"
    options = _make_options(
        log_path=pending,
        argv=_python_argv_for(script),
        cwd=tmp_path,
        kill_grace_seconds=0.3,
    )
    client = AmpClient(options)
    async with client:
        await client.query("p")
        gen = cast("AsyncGenerator[object, None]", client.receive_response())
        # Pull at least one message, then abort.
        first = await asyncio.wait_for(gen.__anext__(), timeout=2.0)
        assert first is not None
        await gen.aclose()

    assert client._state.tee_file is None
    proc = client._state.proc
    assert proc is not None
    assert proc.returncode is not None


# ---------------------------------------------------------------------------
# Malformed JSON / premature exit / missing result
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_malformed_json_raises_amp_client_error(tmp_path: Path) -> None:
    script = _write_fake_amp(
        tmp_path,
        lines=[_system_init(), "not-json-at-all"],
    )
    options = _make_options(
        log_path=tmp_path / ".pending-mj.jsonl",
        argv=_python_argv_for(script),
        cwd=tmp_path,
    )
    async with AmpClient(options) as client:
        await client.query("p")
        with pytest.raises(AmpClientError) as exc_info:
            await _drain(client)
    assert "stream-json" in str(exc_info.value).lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_premature_exit_with_no_result_returns_no_result_message(
    tmp_path: Path,
) -> None:
    """Process exits cleanly after init but emits no ``result``; the iterator
    completes deterministically (no hang, no crash)."""
    script = _write_fake_amp(
        tmp_path,
        lines=[_system_init("T-pe")],
        exit_code=0,
    )
    options = _make_options(
        log_path=tmp_path / ".pending-pe.jsonl",
        argv=_python_argv_for(script),
        cwd=tmp_path,
    )
    async with AmpClient(options) as client:
        await client.query("p")
        msgs = await _drain(client)

    assert all(not isinstance(m, ResultMessage) for m in msgs)
    assert client.session_id == "T-pe"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_premature_exit_nonzero_does_not_hang(tmp_path: Path) -> None:
    script = _write_fake_amp(
        tmp_path,
        lines=[_system_init("T-px")],
        exit_code=2,
    )
    options = _make_options(
        log_path=tmp_path / ".pending-px.jsonl",
        argv=_python_argv_for(script),
        cwd=tmp_path,
    )
    async with AmpClient(options) as client:
        await client.query("p")
        msgs = await asyncio.wait_for(_drain(client), timeout=5.0)

    # Iterator completes; no result message but session_id captured.
    assert all(not isinstance(m, ResultMessage) for m in msgs)


# ---------------------------------------------------------------------------
# Missing system(init) → AmpStreamMissingInitError
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_assistant_before_init_raises_missing_init_error(
    tmp_path: Path,
) -> None:
    script = _write_fake_amp(
        tmp_path,
        lines=[_assistant_text("oops"), _result()],
        stderr="x" * 50,
    )
    pending = tmp_path / ".pending-mi.jsonl"
    options = _make_options(
        log_path=pending, argv=_python_argv_for(script), cwd=tmp_path
    )
    async with AmpClient(options) as client:
        await client.query("p")
        with pytest.raises(AmpStreamMissingInitError) as exc_info:
            await _drain(client)

    err = exc_info.value
    # Pending path preserved on disk.
    assert err.pending_path == pending
    assert pending.exists()
    # Truncated stderr/stdout slices attached and bounded.
    assert isinstance(err.stderr_tail, str)
    assert isinstance(err.stdout_tail, str)
    assert len(err.stderr_tail.encode()) <= 4096
    assert len(err.stdout_tail.encode()) <= 4096


@pytest.mark.unit
@pytest.mark.asyncio
async def test_result_before_init_also_raises_missing_init_error(
    tmp_path: Path,
) -> None:
    script = _write_fake_amp(tmp_path, lines=[_result()])
    options = _make_options(
        log_path=tmp_path / ".pending-rb.jsonl",
        argv=_python_argv_for(script),
        cwd=tmp_path,
    )
    async with AmpClient(options) as client:
        await client.query("p")
        with pytest.raises(AmpStreamMissingInitError):
            await _drain(client)


# ---------------------------------------------------------------------------
# Misc invariants
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_calling_query_twice_raises(tmp_path: Path) -> None:
    """Amp ``--execute`` is one-shot; reusing a client is unsafe."""
    script = _write_fake_amp(tmp_path, lines=[_system_init(), _result()])
    options = _make_options(
        log_path=tmp_path / ".pending-twice.jsonl",
        argv=_python_argv_for(script),
        cwd=tmp_path,
    )
    async with AmpClient(options) as client:
        await client.query("first")
        with pytest.raises(AmpClientError):
            await client.query("second")
        await _drain(client)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_disconnect_is_idempotent(tmp_path: Path) -> None:
    script = _write_fake_amp(tmp_path, lines=[_system_init(), _result()])
    options = _make_options(
        log_path=tmp_path / ".pending-dc.jsonl",
        argv=_python_argv_for(script),
        cwd=tmp_path,
    )
    client = AmpClient(options)
    async with client:
        await client.query("p")
        await _drain(client)
        await client.disconnect()
        await client.disconnect()  # No-op the second time.


@pytest.mark.unit
@pytest.mark.asyncio
async def test_query_session_id_param_acts_as_resume_default(tmp_path: Path) -> None:
    """When ``with_resume`` was not called, the protocol's ``session_id``
    parameter is treated as the resume target (parity with Claude path)."""
    argv_record = tmp_path / "argv-q.json"
    script = _write_fake_amp(
        tmp_path,
        lines=[_system_init("T-from-query"), _result("T-from-query")],
        record_argv_to=argv_record,
    )
    options = _make_options(
        log_path=tmp_path / ".pending-q.jsonl",
        argv=[*_python_argv_for(script), "--execute"],
        cwd=tmp_path,
    )
    async with AmpClient(options) as client:
        await client.query("p", session_id="T-from-query")
        await _drain(client)

    spawned = json.loads(argv_record.read_text())
    assert "--thread-id" in spawned
    assert spawned[spawned.index("--thread-id") + 1] == "T-from-query"

"""E2E test for real ``amp --execute --stream-json``.

This mirrors the existing CLI-backed e2e tests: skip when the ``amp`` CLI is
not installed, otherwise run the real CLI with a tiny prompt and assert the
stream-json schema fields mala consumes.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING, Any, cast

import pytest

if TYPE_CHECKING:
    from pathlib import Path


_E2E_PROMPT = "Reply with the single word: ok\n"
"""Trivial one-line prompt so the test validates schema, not model behavior."""

_SUBPROCESS_TIMEOUT_SECONDS = 120.0
"""Generous bound for a tiny prompt; absorbs cold-start jitter."""


def _require_amp_cli() -> None:
    """Skip the real-Amp e2e test when the Amp CLI is not installed."""
    if shutil.which("amp") is None:
        pytest.skip("Amp CLI not installed")


def _run_amp_version() -> str:
    """Best-effort capture of the observed Amp version string."""
    try:
        completed = subprocess.run(
            ["amp", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"<version-unavailable: {exc!r}>"
    out = (completed.stdout or "").strip() or (completed.stderr or "").strip()
    return out or "<version-unavailable: empty output>"


class StreamJsonParseError(AssertionError):
    """Raised when stdout contains a line that is not valid stream-json."""


def _parse_stream_lines(stdout_text: str) -> list[dict[str, Any]]:
    """Parse Amp ``--stream-json`` stdout, mirroring ``AmpClient``'s contract."""
    events: list[dict[str, Any]] = []
    for raw_line in stdout_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StreamJsonParseError(
                "amp stdout contained a non-JSON line, likely upstream "
                "stream-json drift (banner/progress emitted on stdout). "
                "AmpClient raises AmpClientError on this same input at "
                f"src/infra/clients/amp_client.py:435-442. Offending "
                f"line: {line!r}; json error: {exc}"
            ) from exc
        if not isinstance(obj, dict):
            print(
                f"[amp-e2e] warn: stream-json line is not an object: {obj!r}",
                file=sys.stderr,
            )
            continue
        events.append(obj)
    return events


def _run_amp_stream_json(prompt: str) -> list[dict[str, Any]]:
    """Spawn real ``amp --execute --stream-json`` and collect parsed events."""
    completed = subprocess.run(
        ["amp", "--execute", "--stream-json"],
        input=prompt,
        check=False,
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
    )
    print(f"[amp-e2e] exit_code={completed.returncode}", file=sys.stderr)
    if completed.stderr:
        print(f"[amp-e2e] stderr:\n{completed.stderr}", file=sys.stderr)
    return _parse_stream_lines(completed.stdout or "")


def _empty_mcp_server_factory(
    agent_id: str,
    repo_path: Path,
    emit_lock_event: object | None,
) -> dict[str, object]:
    """Return no MCP servers for the minimal real-Amp e2e session."""
    del agent_id, repo_path, emit_lock_event
    return {}


def _entry_has_any_tool_use(data: dict[str, Any]) -> bool:
    """Return whether a raw Amp/Claude-shaped log entry contains a tool_use."""
    if data.get("type") != "assistant":
        return False
    message = data.get("message")
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") == "tool_use" for block in content
    )


def _amp_session_log_contains_tool_use(log_path: Path) -> bool:
    """Check whether Mala's Amp log provider can read tool evidence."""
    from src.infra.clients.amp_log_provider import AmpLogProvider

    if not log_path.exists():
        return False

    provider = AmpLogProvider(native_dir=log_path.parent)
    for entry in provider.iter_events(log_path):
        if provider.extract_bash_commands(entry):
            return True
        if provider.extract_tool_results(entry):
            return True
        if _entry_has_any_tool_use(entry.data):
            return True
    return False


def _assistant_text(messages: list[object]) -> str:
    """Collect text blocks from synthetic Amp ``AssistantMessage`` objects."""
    chunks: list[str] = []
    for message in messages:
        if type(message).__name__ != "AssistantMessage":
            continue
        for block in getattr(message, "content", []):
            text = getattr(block, "text", None)
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunks)


@pytest.fixture
def test_repo(tmp_path: Path) -> Path:
    """Create a minimal git repo with a single file change."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "commit.gpgsign", "false"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    (tmp_path / "example.py").write_text("")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    (tmp_path / "example.py").write_text(
        "def greet(name: str) -> str:\n"
        '    """Return a greeting message."""\n'
        '    return f"Hello, {name}!"\n'
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Add greet function"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )

    return tmp_path


@pytest.mark.e2e
def test_amp_real_cli_stream_json_schema() -> None:
    """Assert the stream-json schema fields mala depends on are present."""
    _require_amp_cli()

    version = _run_amp_version()
    print(f"[amp-e2e] amp --version: {version}", file=sys.stderr)

    events = _run_amp_stream_json(_E2E_PROMPT)
    assert events, "amp --execute --stream-json produced no parseable events"

    for idx, event in enumerate(events):
        assert "type" in event, f"event[{idx}] missing 'type': {event!r}"
        assert isinstance(event["type"], str), (
            f"event[{idx}].type is not a string: {event!r}"
        )

    observed_types = sorted({str(e["type"]) for e in events})
    print(f"[amp-e2e] observed event.type values: {observed_types}", file=sys.stderr)

    system_inits = [
        e for e in events if e.get("type") == "system" and e.get("subtype") == "init"
    ]
    if not system_inits:
        system_inits = [e for e in events if e.get("type") == "system"]
    assert system_inits, (
        f"no 'system' event found in stream; observed types: {observed_types}"
    )
    init = system_inits[0]
    assert "session_id" in init, f"system init event missing 'session_id': {init!r}"
    assert isinstance(init["session_id"], str) and init["session_id"], (
        f"system init 'session_id' is not a non-empty string: {init!r}"
    )
    print(
        f"[amp-e2e] session_id captured: {init['session_id']!r}",
        file=sys.stderr,
    )

    assistant_events = [e for e in events if e.get("type") == "assistant"]
    assert assistant_events, (
        f"no 'assistant' event found in stream; observed types: {observed_types}"
    )
    saw_typed_block = False
    for evt in assistant_events:
        message: Any = evt.get("message")
        assert isinstance(message, dict), (
            f"assistant event missing dict 'message': {evt!r}"
        )
        content: Any = message.get("content")
        assert isinstance(content, list), (
            f"assistant.message.content is not a list: {evt!r}"
        )
        for block_idx, block in enumerate(content):
            assert isinstance(block, dict), (
                f"assistant content[{block_idx}] is not an object: {block!r}"
            )
            block_dict = cast("dict[str, Any]", block)
            assert "type" in block_dict, (
                f"assistant content[{block_idx}] missing 'type': {block!r}"
            )
            block_type: Any = block_dict["type"]
            assert isinstance(block_type, str), (
                f"assistant content[{block_idx}].type is not a string: {block!r}"
            )
            saw_typed_block = True
    assert saw_typed_block, "no typed content blocks observed in any assistant event"


@pytest.mark.e2e
@pytest.mark.flaky_sdk
@pytest.mark.asyncio
async def test_real_amp_provider_client_flow(
    test_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test Mala's Amp provider/client/log-provider flow with real Amp.

    Mirrors the Claude Agent SDK e2e coverage at the Amp adapter boundary:
    real provider construction, real client session management, synthetic
    message conversion, and session-log evidence parsing.
    """
    _require_amp_cli()

    from src.infra.clients import amp_runtime
    from src.infra.clients.amp_client import AmpClient
    from src.infra.clients.amp_provider import AmpAgentProvider
    from src.infra.clients.amp_runtime import AmpRuntime, AmpRuntimeBuilder

    sessions_dir = tmp_path / "amp-sessions"
    monkeypatch.setattr(amp_runtime, "AMP_SESSIONS_DIR", sessions_dir)

    provider = AmpAgentProvider(mode="rush")
    builder = provider.runtime_builder(
        test_repo,
        "amp-e2e-agent",
        mcp_server_factory=_empty_mcp_server_factory,
    )
    assert isinstance(builder, AmpRuntimeBuilder)

    runtime = builder.with_mcp(servers={}).build()
    assert isinstance(runtime, AmpRuntime)

    client = provider.client_factory.create(runtime)
    assert isinstance(client, AmpClient)

    last_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=test_repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    prompt = f"""Use the shell tool to run exactly this command:
git show --stat --oneline {last_commit}

After the command completes, reply with exactly: AMP_E2E_OK
Do not edit files. Do not run any other command.
"""

    messages: list[object] = []
    async with client:
        await client.query(prompt)
        async for message in client.receive_response():
            messages.append(message)

    assert not client.is_auth_error(), (
        f"Amp CLI reported an auth error: {client.get_stderr()[-1000:]}"
    )
    assert client.session_id is not None, "AmpClient did not capture a session id"
    assert messages, "AmpClient yielded no messages"
    assert any(type(message).__name__ == "ResultMessage" for message in messages), (
        "AmpClient did not yield a result message"
    )
    assert "AMP_E2E_OK" in _assistant_text(messages), (
        "Amp did not produce the expected completion marker"
    )

    log_path = sessions_dir / f"{client.session_id}.jsonl"
    assert log_path.exists(), f"Amp session log was not written: {log_path}"
    assert _amp_session_log_contains_tool_use(log_path), (
        f"Amp session log at {log_path} should contain tool usage evidence"
    )


# Parser regression tests run in the default suite. Mark them ``unit`` so
# ``tests/conftest.py`` does not auto-mark them as e2e by path.


@pytest.mark.unit
def test_parse_stream_lines_fails_on_non_json_stdout() -> None:
    """Banner or progress lines on stdout must fail the parser."""
    drifted = (
        "Starting amp v9.9.9...\n"
        + json.dumps({"type": "system", "subtype": "init", "session_id": "abc"})
        + "\n"
        + json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "ok"}]},
            }
        )
        + "\n"
    )
    with pytest.raises(StreamJsonParseError, match="non-JSON"):
        _parse_stream_lines(drifted)


@pytest.mark.unit
def test_parse_stream_lines_accepts_clean_stream() -> None:
    """Sanity check: a well-formed stream parses without raising."""
    clean = (
        "\n"
        + json.dumps({"type": "system", "subtype": "init", "session_id": "abc"})
        + "\n"
        + "\n"
        + json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "ok"}]},
            }
        )
        + "\n"
    )
    events = _parse_stream_lines(clean)
    assert [e["type"] for e in events] == ["system", "assistant"]


@pytest.mark.unit
def test_parse_stream_lines_warns_on_non_object_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Non-dict JSON values match AmpClient's warn-and-skip behavior."""
    mixed = (
        "[1, 2, 3]\n"
        + json.dumps({"type": "system", "subtype": "init", "session_id": "abc"})
        + "\n"
    )
    events = _parse_stream_lines(mixed)
    assert len(events) == 1
    assert events[0]["type"] == "system"
    captured = capsys.readouterr()
    assert "not an object" in captured.err


@pytest.mark.unit
def test_require_amp_cli_skips_when_amp_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real-Amp e2e gate is path-only."""
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    with pytest.raises(BaseException) as exc_info:
        _require_amp_cli()
    assert "Amp CLI not installed" in str(exc_info.value)
    assert type(exc_info.value).__name__ == "Skipped"


@pytest.mark.unit
def test_require_amp_cli_succeeds_when_amp_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real-Amp e2e gate only checks that ``amp`` is on PATH."""
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/local/bin/amp")

    _require_amp_cli()

"""Unit tests for the stdio MCP locking launcher (AC-9 wiring).

Covers behavior of the standalone ``mala-amp-mcp-locking`` launcher in
:mod:`src.infra.tools.locking_mcp_stdio`. The integration test
``test_lock_mcp_via_amp.py`` exercises the same code via real Amp; this
file pins the contract independently of an Amp install so a regression
in the launcher's lock-key derivation, JSON wire shape, or argparse
fail-closed behavior surfaces in unit-only CI.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


@pytest.fixture
def lock_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    d = tmp_path / "locks"
    d.mkdir()
    monkeypatch.setenv("MALA_LOCK_DIR", str(d))
    return d


def test_lock_acquire_writes_hash_lock_file(lock_dir: Path, tmp_path: Path) -> None:
    """``lock_acquire`` must produce the same ``<hash>.lock`` shape that
    :func:`src.infra.tools.locking.try_lock` writes — the cross-language
    contract with ``plugins/amp/mala-safety.ts`` depends on this."""
    from src.infra.tools.locking import lock_path
    from src.infra.tools.locking_mcp_stdio import _handle_lock_acquire

    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "foo.py"
    target.write_text("# original\n")

    body = asyncio.run(
        _handle_lock_acquire(
            {"filepaths": [str(target)], "timeout_seconds": 0},
            agent_id="agent-A",
            repo_namespace=str(repo),
        )
    )
    payload = json.loads(body)
    assert payload["all_acquired"] is True
    assert payload["results"][0]["acquired"] is True
    assert payload["results"][0]["filepath"] == str(target)

    # Same lock key + body the Claude path's try_lock would produce.
    lp = lock_path(str(target), repo_namespace=str(repo))
    assert lp.exists(), f"Expected lock file {lp} from MCP round-trip"
    assert lp.read_text().strip() == "agent-A"


def test_lock_acquire_emits_acquired_event(
    lock_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.infra.tools.locking_mcp_stdio import _handle_lock_acquire

    event_log = tmp_path / "events.jsonl"
    monkeypatch.setenv("MALA_LOCK_EVENT_LOG", str(event_log))
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "foo.py"

    asyncio.run(
        _handle_lock_acquire(
            {"filepaths": [str(target)], "timeout_seconds": 0},
            agent_id="agent-A",
            repo_namespace=str(repo),
        )
    )

    events = [json.loads(line) for line in event_log.read_text().splitlines()]
    assert events[0]["event_type"] == "acquired"
    assert events[0]["agent_id"] == "agent-A"
    assert events[0]["lock_path"] == str(target.resolve())


def test_lock_acquire_does_not_emit_acquired_for_reentrant_lock(
    lock_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.infra.tools.locking import try_lock
    from src.infra.tools.locking_mcp_stdio import _handle_lock_acquire

    event_log = tmp_path / "events.jsonl"
    monkeypatch.setenv("MALA_LOCK_EVENT_LOG", str(event_log))
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "foo.py"
    assert try_lock(str(target), agent_id="agent-A", repo_namespace=str(repo))

    asyncio.run(
        _handle_lock_acquire(
            {"filepaths": [str(target)], "timeout_seconds": 0},
            agent_id="agent-A",
            repo_namespace=str(repo),
        )
    )

    assert not event_log.exists()


def test_lock_acquire_nonblocking_reports_holder_when_blocked(
    lock_dir: Path, tmp_path: Path
) -> None:
    """timeout_seconds=0 must return immediately with ``acquired=False``
    and the current holder, mirroring locking_mcp.py's behavior."""
    from src.infra.tools.locking import try_lock
    from src.infra.tools.locking_mcp_stdio import _handle_lock_acquire

    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "foo.py"

    # Pre-acquire as agent-B so agent-A's call is blocked.
    assert try_lock(str(target), agent_id="agent-B", repo_namespace=str(repo))

    body = asyncio.run(
        _handle_lock_acquire(
            {"filepaths": [str(target)], "timeout_seconds": 0},
            agent_id="agent-A",
            repo_namespace=str(repo),
        )
    )
    payload = json.loads(body)
    assert payload["all_acquired"] is False
    result = payload["results"][0]
    assert result["acquired"] is False
    assert result["holder"] == "agent-B"


def test_blocking_lock_acquire_emits_waiting_event(
    lock_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.infra.tools.locking import try_lock
    from src.infra.tools.locking_mcp_stdio import _handle_lock_acquire

    event_log = tmp_path / "events.jsonl"
    monkeypatch.setenv("MALA_LOCK_EVENT_LOG", str(event_log))
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "foo.py"
    assert try_lock(str(target), agent_id="agent-B", repo_namespace=str(repo))

    asyncio.run(
        _handle_lock_acquire(
            {"filepaths": [str(target)], "timeout_seconds": 0.01},
            agent_id="agent-A",
            repo_namespace=str(repo),
        )
    )

    events = [json.loads(line) for line in event_log.read_text().splitlines()]
    assert [event["event_type"] for event in events] == ["waiting"]
    assert events[0]["agent_id"] == "agent-A"
    assert events[0]["lock_path"] == str(target.resolve())


def test_lock_acquire_validates_empty_filepaths(lock_dir: Path) -> None:
    from src.infra.tools.locking_mcp_stdio import _handle_lock_acquire

    body = asyncio.run(
        _handle_lock_acquire({"filepaths": []}, agent_id="agent-A", repo_namespace=None)
    )
    payload = json.loads(body)
    assert "error" in payload
    assert "non-empty" in payload["error"]


def test_lock_release_releases_owned_lock(lock_dir: Path, tmp_path: Path) -> None:
    from src.infra.tools.locking import lock_path, try_lock
    from src.infra.tools.locking_mcp_stdio import _handle_lock_release

    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "foo.py"
    assert try_lock(str(target), agent_id="agent-A", repo_namespace=str(repo))
    lp = lock_path(str(target), repo_namespace=str(repo))
    assert lp.exists()

    body = asyncio.run(
        _handle_lock_release(
            {"filepaths": [str(target)]},
            agent_id="agent-A",
            repo_namespace=str(repo),
        )
    )
    payload = json.loads(body)
    assert payload["count"] == 1
    assert payload["released"] == [str(target)]
    assert not lp.exists()


def test_lock_release_emits_event_for_idempotent_release(
    lock_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.infra.tools.locking_mcp_stdio import _handle_lock_release

    event_log = tmp_path / "events.jsonl"
    monkeypatch.setenv("MALA_LOCK_EVENT_LOG", str(event_log))
    repo = tmp_path / "repo"
    repo.mkdir()
    target = repo / "foo.py"

    body = asyncio.run(
        _handle_lock_release(
            {"filepaths": [str(target)]},
            agent_id="agent-A",
            repo_namespace=str(repo),
        )
    )
    payload = json.loads(body)
    assert payload["count"] == 1
    assert payload["released"] == [str(target)]

    events = [json.loads(line) for line in event_log.read_text().splitlines()]
    assert events[0]["event_type"] == "released"
    assert events[0]["agent_id"] == "agent-A"
    assert events[0]["lock_path"] == str(target.resolve())


def test_lock_release_rejects_both_params() -> None:
    from src.infra.tools.locking_mcp_stdio import _handle_lock_release

    body = asyncio.run(
        _handle_lock_release(
            {"filepaths": ["x.py"], "all": True},
            agent_id="agent-A",
            repo_namespace=None,
        )
    )
    payload = json.loads(body)
    assert "error" in payload
    assert "both" in payload["error"].lower()


def test_lock_release_rejects_no_params() -> None:
    from src.infra.tools.locking_mcp_stdio import _handle_lock_release

    body = asyncio.run(
        _handle_lock_release({}, agent_id="agent-A", repo_namespace=None)
    )
    payload = json.loads(body)
    assert "error" in payload


def test_main_fails_closed_without_agent_id(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without ``--agent-id`` (CLI or env), the launcher must exit 2.

    A launcher that started with no owner would write lock files whose
    body equals the empty string — silently breaking the plugin's
    holder-equality check. Fail-closed at startup is the safer default.
    """
    monkeypatch.delenv("MALA_AGENT_ID", raising=False)
    from src.infra.tools.locking_mcp_stdio import main

    rc = main([])
    assert rc == 2
    captured = capsys.readouterr()
    assert "agent-id" in captured.err.lower()


def test_main_accepts_agent_id_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """``MALA_AGENT_ID`` must be honored as a CLI fallback so the
    factory's ``env`` field is sufficient even if Amp drops process
    args between MCP-config parse and spawn."""
    monkeypatch.setenv("MALA_AGENT_ID", "agent-from-env")
    from src.infra.tools.locking_mcp_stdio import _parse_args

    ns = _parse_args([])
    assert ns.agent_id == "agent-from-env"

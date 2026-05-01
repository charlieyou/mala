"""Real-Amp ``lock_acquire`` MCP round-trip + plugin lock-ownership E2E.

Companion to ``test_amp_lock_enforcement.py``. While that file exercises
each of the seven plugin lock-ownership scenarios in isolation against a
**Python-written** lock fixture, this file exercises the full pipeline:

  1. Spawn real ``amp --execute --stream-json --mcp-config <stdio_spec>``.
  2. Drive the LLM to call the ``lock_acquire`` MCP tool for a target file.
  3. Drive the LLM to call ``edit_file`` for the same path — the plugin
     observes a real lock written by the locking_mcp server (which is the
     same Python writer used elsewhere) and **allows** the write.
  4. Drive a second ``edit_file`` against an unlocked sibling — the
     plugin **rejects** with the no-lock message.

This is the only test in the suite that exercises both the plugin lock
reader AND the MCP lock writer in one Amp process; it owns the
end-to-end leg of AC#9a.

Gating
------
Identical ``amp``-on-PATH gate to ``test_amp_lock_enforcement.py``.
Additionally, the test requires a real **stdio** locking MCP launcher
reachable via ``amp --mcp-config``. The MVP locking server
(``src/infra/tools/locking_mcp.py``) is built on the Claude Agent SDK's
in-process server and has no standalone stdio entry point yet — plan
L702 tracks that work as part of the real-Amp lock-enforcement effort.
Until a stdio launcher exists, this test skips with a clear reference to
the missing piece. The skip ensures the test remains a regression guard
the moment the launcher lands.

Why a placeholder factory is not enough
---------------------------------------
``src.orchestration.orchestration_wiring.create_amp_mcp_server_factory``
returns a stdio spec pointing at the placeholder command
``mala-amp-mcp-locking``. That command is not on disk anywhere; spawning
``amp`` against the placeholder would surface "command not found" errors
mid-MCP startup and abort before any tool call. Skipping with a precise
reason is preferable to a noisy false-failure.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import dataclasses
from pathlib import Path
from typing import Any

import pytest

from src.infra.clients.amp_plugin_installer import AmpPluginInstaller


# ---------------------------------------------------------------------------
# Module-level gating
# ---------------------------------------------------------------------------

_AMP_BINARY_REASON = (
    "amp binary not on PATH; coder=amp requires the official binary install. "
    "This real-Amp integration module is skipped unless local prerequisites "
    "are present."
)


def _has_stdio_locking_launcher() -> bool:
    """True if a stdio MCP launcher for ``locking_mcp`` is available.

    The MVP locking server has no standalone stdio entry point yet (plan
    L702). When the launcher lands, it should appear on ``PATH`` under
    a stable name; this probe is intentionally conservative — any of the
    candidate names below resolves the gate. Until then, this returns
    ``False`` and the tests in this module skip cleanly.
    """
    candidate_names = (
        "mala-amp-mcp-locking",
        "mala-locking-mcp",
        "mala-mcp-locking",
    )
    for name in candidate_names:
        if shutil.which(name) is not None:
            return True
    return False


_STDIO_LAUNCHER_REASON = (
    "stdio MCP launcher for locking_mcp not on PATH. The MVP locking server "
    "(src/infra/tools/locking_mcp.py) is in-process via the Claude SDK's "
    "create_sdk_mcp_server; no standalone stdio entry point exists yet "
    "(plan L702). Skip until the launcher lands so this test becomes a "
    "regression guard automatically when it does."
)

pytestmark = [
    pytest.mark.skipif(shutil.which("amp") is None, reason=_AMP_BINARY_REASON),
    pytest.mark.skipif(
        not _has_stdio_locking_launcher(),
        reason=_STDIO_LAUNCHER_REASON,
    ),
]


_AMP_TIMEOUT_SECONDS = 240.0
"""Generous bound: this test makes multiple LLM tool calls in one
session (``lock_acquire`` + two ``edit_file`` calls), each a round-trip,
so the wall time budget is roughly twice the single-tool tests."""

_AGENT_ID = "agent-mcp-A"


# ---------------------------------------------------------------------------
# Test environment
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AmpMcpEnv:
    home_dir: Path
    plugin_dir: Path
    lock_dir: Path
    repo_path: Path


def _copy_amp_login_state(home: Path) -> None:
    """Copy the current Amp CLI login state into a synthetic ``HOME``.

    The tests redirect ``HOME`` so Amp loads the test-installed plugin from a
    temp directory. Current Amp stores login state under ``~/.local/share/amp``;
    without copying that state, real Amp prompts for login before plugins load.
    """
    real_home = Path.home()

    real_settings = real_home / ".config" / "amp" / "settings.json"
    if real_settings.exists():
        target_settings = home / ".config" / "amp" / "settings.json"
        target_settings.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(real_settings, target_settings)

    real_data_dir = real_home / ".local" / "share" / "amp"
    target_data_dir = home / ".local" / "share" / "amp"
    target_data_dir.mkdir(parents=True, exist_ok=True)
    for name in ("session.json", "secrets.json", "device-id.json"):
        source = real_data_dir / name
        if source.exists():
            shutil.copy2(source, target_data_dir / name)


@pytest.fixture
def amp_mcp_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AmpMcpEnv:
    """Fresh per-test plugin install + lock dir + repo namespace.

    Same shape as ``test_amp_lock_enforcement.py``'s fixture but extracted
    here to keep the two modules independently runnable; no shared
    fixtures tie the test files together.
    """
    home = tmp_path / "home"
    _copy_amp_login_state(home)
    plugin_dir = home / ".config" / "amp" / "plugins"
    AmpPluginInstaller().install(target_dir=plugin_dir)

    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()

    repo = tmp_path / "repo"
    repo.mkdir()

    monkeypatch.setenv("MALA_LOCK_DIR", str(lock_dir))

    return AmpMcpEnv(
        home_dir=home,
        plugin_dir=plugin_dir,
        lock_dir=lock_dir,
        repo_path=repo,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_stdio_launcher() -> str:
    """Return the stdio launcher command name to wire into --mcp-config.

    Matches :func:`_has_stdio_locking_launcher`'s probe; assumes the
    skipif gate already verified availability.
    """
    for name in ("mala-amp-mcp-locking", "mala-locking-mcp", "mala-mcp-locking"):
        if shutil.which(name) is not None:
            return name
    raise RuntimeError(
        "stdio MCP launcher for locking_mcp disappeared after skipif gate; "
        "this should not happen — file a bug."
    )


def _build_mcp_config(env: AmpMcpEnv) -> str:
    """Construct the ``--mcp-config`` JSON payload for Amp.

    Uses the same direct server-map shape Amp expects on the CLI and that
    :class:`src.infra.clients.amp_runtime.AmpRuntimeBuilder` builds in
    production.

    The launcher is invoked with the agent id and repo namespace it
    needs to compute lock keys identically to the in-test Python
    fixtures.
    """
    launcher = _resolve_stdio_launcher()
    payload = {
        "mala-locking": {
            "command": launcher,
            "args": [
                "--agent-id",
                _AGENT_ID,
                "--repo-namespace",
                str(env.repo_path),
            ],
            "env": {
                "MALA_LOCK_DIR": str(env.lock_dir),
                "MALA_AGENT_ID": _AGENT_ID,
                "MALA_REPO_NAMESPACE": str(env.repo_path),
            },
        }
    }
    return json.dumps(payload)


@dataclasses.dataclass
class AmpResult:
    returncode: int
    events: list[dict[str, Any]]
    stderr_text: str

    @property
    def sentinel_loaded(self) -> bool:
        return any(
            marker in self.stderr_text
            for marker in (
                '"mala_plugin":"loaded"',
                '"mala_plugin": "loaded"',
                '\\"mala_plugin\\":\\"loaded\\"',
                '\\"mala_plugin\\": \\"loaded\\"',
            )
        )


def _build_amp_env(env: AmpMcpEnv) -> dict[str, str]:
    spawned: dict[str, str] = {**os.environ}
    spawned["HOME"] = str(env.home_dir)
    spawned["XDG_CONFIG_HOME"] = str(env.home_dir / ".config")
    spawned["PLUGINS"] = "all"
    spawned["MALA_AGENT_ID"] = _AGENT_ID
    spawned["MALA_LOCK_DIR"] = str(env.lock_dir)
    spawned["MALA_REPO_NAMESPACE"] = str(env.repo_path)
    return spawned


def _run_amp_with_mcp(
    prompt: str,
    spawned_env: dict[str, str],
    cwd: Path,
    mcp_config_json: str,
    *,
    timeout: float = _AMP_TIMEOUT_SECONDS,
) -> AmpResult:
    log_path = cwd.parent / "amp-mcp-cli.log"
    try:
        log_path.unlink(missing_ok=True)
    except OSError:
        pass
    argv = [
        "amp",
        "--execute",
        "--stream-json",
        "--dangerously-allow-all",
        "--log-level",
        "debug",
        "--log-file",
        str(log_path),
        "--mcp-config",
        mcp_config_json,
        "--mode",
        "rush",
    ]
    completed = subprocess.run(
        argv,
        input=prompt,
        env=spawned_env,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    print(f"[amp-mcp] exit_code={completed.returncode}", file=sys.stderr)
    if completed.stderr:
        print(
            f"[amp-mcp] stderr (truncated):\n{completed.stderr[:2048]}", file=sys.stderr
        )
    try:
        plugin_log_text = log_path.read_text(errors="replace")
    except OSError:
        plugin_log_text = ""

    events: list[dict[str, Any]] = []
    for raw_line in (completed.stdout or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return AmpResult(
        returncode=completed.returncode,
        events=events,
        stderr_text="\n".join(
            part for part in (completed.stderr or "", plugin_log_text) if part
        ),
    )


def _tool_uses(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ev in events:
        if ev.get("type") != "assistant":
            continue
        message = ev.get("message")
        if not isinstance(message, dict):
            continue
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                out.append(block)
    return out


def _tool_result_text(events: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for ev in events:
        if ev.get("type") != "user":
            continue
        message = ev.get("message")
        if not isinstance(message, dict):
            continue
        for block in message.get("content") or []:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            body = block.get("content")
            if isinstance(body, str):
                parts.append(body)
            elif isinstance(body, list):
                for sub in body:
                    if isinstance(sub, dict):
                        text = sub.get("text") or sub.get("output") or ""
                        if isinstance(text, str) and text:
                            parts.append(text)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# E2E test
# ---------------------------------------------------------------------------


def test_lock_acquire_then_edit_allowed_and_unlocked_edit_rejected(
    amp_mcp_env: AmpMcpEnv,
) -> None:
    """End-to-end: real Amp + real stdio locking_mcp + real plugin.

    Drives one ``amp`` invocation through three tool calls:

      1. ``lock_acquire`` for ``locked.py`` (MCP server writes the
         ``<hash>.lock`` fixture using the production Python locking
         code path — same writer, same format as
         ``test_amp_lock_enforcement.py``).
      2. ``edit_file`` against ``locked.py`` — plugin sees the lock and
         **allows** the write.
      3. ``edit_file`` against ``unlocked.py`` (no preceding
         ``lock_acquire``) — plugin **rejects** with the no-lock message.

    Asserts the plugin sentinel marker fires (proves the safety plugin
    loaded under PLUGINS=all). Asserts the MCP server's
    ``lock_acquire`` round-trip succeeded by checking the produced
    lock-file directly on disk.

    LLM-compliance caveat applies: when the model fails to make all three
    tool calls in order, the test skips with a precise reason rather than
    asserting against an inconclusive transcript. This trades sensitivity
    for false-positive avoidance — better than a flaky test that misleads
    the next reader.
    """
    locked_target = amp_mcp_env.repo_path / "locked.py"
    locked_target.write_text("# original\n")

    unlocked_target = amp_mcp_env.repo_path / "unlocked.py"
    unlocked_target.write_text("# original\n")

    prompt = (
        "You will perform exactly three tool calls in order. Do not run any "
        "other tools (no Bash, no read_file, no grep). Reply only with the "
        "single word done after the third call returns.\n"
        "\n"
        "Step 1. Call the lock_acquire MCP tool from the mala-locking server "
        f'with filepaths=["{locked_target}"] and timeout_seconds=0.\n'
        f"Step 2. Call the edit_file tool to overwrite {locked_target} with "
        "the single line: locked-edit\n"
        f"Step 3. Call the edit_file tool to overwrite {unlocked_target} with "
        "the single line: unlocked-edit\n"
    )

    mcp_config_json = _build_mcp_config(amp_mcp_env)
    spawned_env = _build_amp_env(amp_mcp_env)
    result = _run_amp_with_mcp(
        prompt,
        spawned_env,
        amp_mcp_env.repo_path,
        mcp_config_json,
    )

    assert result.sentinel_loaded, (
        "plugin sentinel marker not seen on stderr — PLUGINS=all loading or "
        "the bundled mala-safety.ts plugin did not run. Without the plugin, "
        "downstream allow/reject observations are meaningless. stderr (4 KiB):\n"
        f"{result.stderr_text[:4096]}"
    )

    uses = _tool_uses(result.events)
    used_names = [u.get("name") for u in uses]
    if "lock_acquire" not in used_names:
        pytest.skip(
            "amp did not call the lock_acquire MCP tool; LLM did not comply "
            "with the directive prompt. Cannot verify the MCP round-trip. "
            f"Observed tool_use names: {used_names!r}"
        )
    edit_uses = [u for u in uses if u.get("name") == "edit_file"]
    if len(edit_uses) < 2:
        pytest.skip(
            "amp did not produce two edit_file tool_use blocks (one for "
            "locked target, one for unlocked target); LLM did not comply "
            "with the directive prompt. Cannot distinguish allow from "
            f"reject. Observed edit_file calls: {len(edit_uses)}"
        )

    # Cross-language contract check #1: after lock_acquire, a real
    # <hash>.lock file should exist in MALA_LOCK_DIR. The MCP server
    # writes via the same Python try_lock used in
    # test_amp_lock_enforcement.py, so the on-disk file is the bridge
    # between the two modules' coverage.
    lock_files = list(amp_mcp_env.lock_dir.glob("*.lock"))
    assert lock_files, (
        f"lock_acquire MCP round-trip did not produce a <hash>.lock file in "
        f"{amp_mcp_env.lock_dir}; MCP server may have failed silently. "
        f"Observed events: {len(result.events)}, stderr 4 KiB:\n"
        f"{result.stderr_text[:4096]}"
    )

    text = _tool_result_text(result.events)
    # Both substrings must coexist in tool_results: the rejection for the
    # unlocked target is what proves the plugin is gating; the absence of
    # a rejection for the locked target proves the MCP-written lock was
    # honored. We use distinct path substrings to disambiguate.
    assert str(unlocked_target) in text and "is not locked" in text, (
        "expected the no-lock rejection for the unlocked target. The plugin "
        "should reject edit_file against unlocked.py while allowing it "
        f"against locked.py. tool_result content:\n{text[:2048]}"
    )
    # Sanity: the locked target should not draw ANY plugin lock rejection.
    # Both rejection markers are checked because the failure modes are
    # distinct and load-bearing:
    #
    #   * "is not locked" → MCP server's lock-key derivation diverged from
    #     the plugin's (or the <hash>.lock filename hashing drifted), so
    #     the plugin sees no lock for the path the MCP just wrote.
    #   * "is locked by"  → MCP server wrote the lock body with a holder
    #     other than this Amp agent's MALA_AGENT_ID. The MCP server is
    #     supposed to record the same agent_id the plugin reads from env;
    #     a divergence here would let an MCP-written lock pass the plugin's
    #     existence check while still failing the holder-equality check —
    #     a subtler regression than the no-lock case but equally fatal.
    _LOCKED_REJECTION_MARKERS = ("is not locked", "is locked by")
    locked_segments = [
        seg
        for seg in text.split("\n")
        if str(locked_target) in seg
        and any(marker in seg for marker in _LOCKED_REJECTION_MARKERS)
    ]
    assert not locked_segments, (
        "unexpected lock-rejection for the locked target — the MCP-written "
        "lock was not honored by the same Amp agent. Either the MCP server's "
        "lock-key derivation diverged from the plugin's, the <hash>.lock "
        "filename hashing drifted, or the MCP server recorded a different "
        f"agent_id than MALA_AGENT_ID. Bad segments: {locked_segments!r}"
    )

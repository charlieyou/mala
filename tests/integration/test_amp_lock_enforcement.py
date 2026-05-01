"""End-to-end integration tests for ``mala-safety.ts`` lock-ownership (T014).

Owns AC#9a: an Amp file-write is allowed only when the current Amp agent
holds the lock for that file in this repo namespace, with parity to the
Claude path's :func:`src.infra.hooks.locking.make_lock_enforcement_hook`.

The tests drive the **bundled** ``plugins/amp/mala-safety.ts`` plugin via
**real Amp** (which provides the Bun runtime transitively) against a real
``MALA_LOCK_DIR`` populated by the **real Python** :func:`try_lock`. The
plugin's TypeScript reader and the Python writer therefore share the same
on-disk fixture: any drift in lock-key derivation, filename hashing, or
lock-file body format manifests as a failure here. **Lock files are never
hand-constructed in this module** — the format-drift gate depends on the
fixture coming from production code.

Gating
------
This file is gated on real-Amp integration preconditions:

  * ``amp`` binary on ``PATH`` (the binary install is the only supported
    install path for ``coder=amp`` — npm-installed Amp is rejected at
    runtime by the plugin self-test, see :class:`AmpPluginNotActiveError`).

This is a CI-time concern; locally these tests skip cleanly so
``uv run pytest`` stays green without an Amp install.

Scenarios (plan ``L777-L804``)
------------------------------
  1. **Allow**: agent A holds lock for ``foo.py``; A's ``edit_file`` is
     allowed.
  2. **No-lock**: no lock; ``edit_file`` rejected with the no-lock message.
  3. **Wrong-agent**: agent B holds the lock; A's ``edit_file`` rejected.
  4. **Wrong-namespace**: lock written under a different
     ``MALA_REPO_NAMESPACE`` is invisible to the plugin; rejected.
  5. **Format-drift resilience**: the lock fixture is written by real
     Python ``try_lock``; the plugin parses it and allows the write. If
     the Python format changes, this test fails first — the cross-language
     contract is test-enforced.
  6. **Path canonicalization parity**: lock acquired via relative path
     under the same ``MALA_REPO_NAMESPACE`` matches a tool.call carrying
     the absolute path of the same file.

LLM-compliance caveat
---------------------
These tests rely on the real model emitting an actual ``edit_file``
``tool_use`` for the prompt. When the model declines or routes through
a different tool (e.g. ``Bash`` heredoc), the test cannot verify the
plugin's allow/reject decision and skips with an explicit reason rather
than asserting a false positive. The sentinel-marker assertion (plugin
loaded under ``PLUGINS=all``) still runs in every case, so plugin-load
regressions surface even when LLM compliance is patchy.
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
from src.infra.tools.locking import lock_path, try_lock


# ---------------------------------------------------------------------------
# Module-level gating (``amp`` on PATH)
# ---------------------------------------------------------------------------

_AMP_BINARY_REASON = (
    "amp binary not on PATH; coder=amp requires the official binary install. "
    "This real-Amp integration module is skipped unless local prerequisites "
    "are present."
)

pytestmark = [
    pytest.mark.skipif(shutil.which("amp") is None, reason=_AMP_BINARY_REASON),
]


_AMP_TIMEOUT_SECONDS = 180.0
"""Generous bound for one ``amp --execute --stream-json`` invocation. Real
Amp + plugin load + one LLM tool call typically completes in <60s; the
wider window absorbs cold-start jitter on CI."""

_DEFAULT_AGENT_ID = "agent-A"
_OTHER_AGENT_ID = "agent-B"


# ---------------------------------------------------------------------------
# Test environment dataclass + fixture
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AmpLockEnv:
    """Per-test scratch environment.

    Attributes:
        home_dir: Synthetic ``$HOME`` exported into the spawned ``amp``
            process so its plugin loader resolves
            ``$HOME/.config/amp/plugins/mala-safety.ts``. Real Amp reads
            its plugin directory via the OS-level home, so the test must
            redirect ``HOME`` (and ``XDG_CONFIG_HOME`` for safety) rather
            than monkey-patching :meth:`pathlib.Path.home`.
        plugin_dir: Resolved ``$HOME/.config/amp/plugins/`` containing the
            installed ``mala-safety.ts``.
        lock_dir: Fresh ``MALA_LOCK_DIR`` under ``tmp_path``.
        repo_path: Synthetic repo namespace (cwd + ``MALA_REPO_NAMESPACE``).
    """

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
def amp_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AmpLockEnv:
    """Fresh per-test plugin install + lock dir + repo namespace.

    The bundled ``plugins/amp/mala-safety.ts`` is installed via real
    :class:`AmpPluginInstaller` so the on-disk bytes (and version hash)
    match what production code installs. ``MALA_LOCK_DIR`` is exported into
    the parent test process's env so subsequent calls to
    :func:`src.infra.tools.locking.try_lock` write into the redirected
    directory; the same env var is propagated to the spawned ``amp`` so
    the plugin reads from the same location.
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

    return AmpLockEnv(
        home_dir=home,
        plugin_dir=plugin_dir,
        lock_dir=lock_dir,
        repo_path=repo,
    )


# ---------------------------------------------------------------------------
# Amp spawn helpers
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class AmpResult:
    """Captured output of one ``amp --execute --stream-json`` invocation."""

    returncode: int
    events: list[dict[str, Any]]
    stderr_text: str

    @property
    def sentinel_loaded(self) -> bool:
        """True if the plugin's ``session.start`` sentinel marker was seen.

        The plugin emits ``{"mala_plugin": "loaded", "version": "<hash>"}``
        at every ``session.start`` (see
        ``plugins/amp/mala-safety.ts::emitSentinelMarker``). Current Amp
        records plugin stderr in the CLI debug log, while older versions
        forwarded it to process stderr. Presence of this string in either
        source proves PLUGINS=all + Bun runtime + plugin load all worked.
        """
        return any(
            marker in self.stderr_text
            for marker in (
                '"mala_plugin":"loaded"',
                '"mala_plugin": "loaded"',
                '\\"mala_plugin\\":\\"loaded\\"',
                '\\"mala_plugin\\": \\"loaded\\"',
            )
        )


def _build_amp_env(
    env: AmpLockEnv,
    *,
    agent_id: str = _DEFAULT_AGENT_ID,
    lock_dir: Path | None = None,
    repo_namespace: str | None = None,
    drop: tuple[str, ...] = (),
) -> dict[str, str]:
    """Compose the env dict passed to ``subprocess.run(env=...)``.

    Inherits the parent's env, then overlays the redirect to the synthetic
    ``$HOME`` plus the plugin's required ``MALA_*`` lock-ownership vars.
    ``drop`` removes specific keys after the overlay so the env-missing
    scenario can simulate a misconfigured runtime.
    """
    spawned: dict[str, str] = {**os.environ}
    spawned["HOME"] = str(env.home_dir)
    spawned["XDG_CONFIG_HOME"] = str(env.home_dir / ".config")
    spawned["PLUGINS"] = "all"
    spawned["MALA_AGENT_ID"] = agent_id
    spawned["MALA_LOCK_DIR"] = str(lock_dir or env.lock_dir)
    spawned["MALA_REPO_NAMESPACE"] = repo_namespace or str(env.repo_path)
    for key in drop:
        spawned.pop(key, None)
    return spawned


def _run_amp(
    prompt: str,
    spawned_env: dict[str, str],
    cwd: Path,
    *,
    timeout: float = _AMP_TIMEOUT_SECONDS,
) -> AmpResult:
    """Spawn ``amp --execute --stream-json --dangerously-allow-all --mode rush``.

    ``--mode rush`` (Haiku) is used to keep test cost and latency low; the
    plugin's lock-ownership decision is independent of the model's
    reasoning depth. Output is captured fully (text mode) and stdout lines
    are parsed line-by-line into :class:`dict` events; non-JSON or
    non-object lines are skipped (matching the Amp client's tolerance).
    """
    log_path = cwd.parent / "amp-cli.log"
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
    print(f"[amp-lock] exit_code={completed.returncode}", file=sys.stderr)
    if completed.stderr:
        print(
            f"[amp-lock] stderr (truncated):\n{completed.stderr[:2048]}",
            file=sys.stderr,
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
    """Extract every ``tool_use`` block emitted by the assistant."""
    out: list[dict[str, Any]] = []
    for ev in events:
        if ev.get("type") != "assistant":
            continue
        message = ev.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                out.append(block)
    return out


def _tool_result_text(events: list[dict[str, Any]]) -> str:
    """Concatenate every ``tool_result`` body text in stream order.

    The plugin's ``reject-and-continue`` decision surfaces in stream-json
    as a synthesized ``tool_result`` block whose content carries the
    rejection message. We flatten every result block (regardless of
    ``is_error``) into a single string so substring assertions can match
    the plugin's exact rejection wording without depending on the precise
    representation Amp uses for ``content`` (string vs. list-of-blocks).
    """
    parts: list[str] = []
    for ev in events:
        if ev.get("type") != "user":
            continue
        message = ev.get("message")
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            body = block.get("content")
            if isinstance(body, str):
                parts.append(body)
            elif isinstance(body, list):
                for sub in body:
                    if not isinstance(sub, dict):
                        continue
                    text = sub.get("text") or sub.get("output") or ""
                    if isinstance(text, str) and text:
                        parts.append(text)
    return "\n".join(parts)


def _file_write_attempted(events: list[dict[str, Any]]) -> bool:
    """True if the assistant emitted any file-write ``tool_use`` block.

    The plugin gates ``edit_file`` / ``create_file`` / ``undo_edit`` /
    ``apply_patch``; any of those proves the LLM routed the prompt through
    a tool the plugin actually inspects. If none fired, the test cannot
    distinguish "plugin allowed" from "model never tried" and must skip.
    """
    file_write_tools = {"edit_file", "create_file", "undo_edit", "apply_patch"}
    return any(u.get("name") in file_write_tools for u in _tool_uses(events))


def _assert_plugin_loaded(result: AmpResult) -> None:
    """Hard assertion: plugin sentinel was seen on stderr or CLI log."""
    assert result.sentinel_loaded, (
        "plugin sentinel marker not seen on stderr — PLUGINS=all loading or "
        "the bundled mala-safety.ts plugin did not run. Without the plugin, "
        "downstream allow/reject observations are meaningless. stderr/log (4 KiB):\n"
        f"{result.stderr_text[:4096]}"
    )


def _skip_if_no_file_write(events: list[dict[str, Any]], scenario: str) -> None:
    """Skip when the LLM never attempted a file-write tool call.

    LLM-compliance is the documented uncertainty (module docstring); when
    the model routed the prompt through a non-file-write tool (e.g. plain
    Bash echo) or refused outright, this scenario cannot verify the
    plugin's allow/reject decision. Skipping is preferred to a
    false-positive pass.
    """
    if not _file_write_attempted(events):
        pytest.skip(
            f"[{scenario}] amp produced no edit_file/create_file/undo_edit/"
            "apply_patch tool_use; LLM did not comply with the directive. "
            "Cannot verify plugin allow/reject decision."
        )


def _edit_prompt(target: Path, body: str = "edited") -> str:
    """Directive prompt: invoke ``edit_file`` exactly once on ``target``.

    The wording is intentionally specific so a rush-mode model has the
    minimum information to call ``edit_file`` directly without preamble.
    Phrased as a tool-call instruction (path, contents, terminate) to
    minimize the probability of routing through Bash/heredoc.

    The path-shape label (``absolute path`` vs ``relative path``) is
    derived from ``target`` so the path-canonicalization-parity test can
    feed a genuinely relative ``Path`` and have the prompt instruct the
    model to keep the relative form. Hard-coding ``absolute path`` would
    nudge the model to convert relative inputs to absolute and defeat
    the parity coverage.
    """
    if target.is_absolute():
        path_label = "absolute path"
        path_note = ""
    else:
        path_label = "relative path"
        path_note = (
            " Pass exactly the relative string above to edit_file's path "
            "argument; do not convert it to an absolute path."
        )
    return (
        f"Use the edit_file tool to overwrite the file at the {path_label} "
        f"{target} so that its entire contents become the single line:\n"
        f"{body}\n"
        f"Make exactly one edit_file tool call.{path_note} Do not run any "
        "other tools (no Bash, no read_file, no grep). After the tool call "
        "returns, reply with the single word done and stop."
    )


def _create_prompt(target: Path, body: str = "edited") -> str:
    """Directive prompt: invoke ``create_file`` exactly once at ``target``.

    Used by the non-existent-target canonicalization parity test, where
    the file does not exist at lock time **or** at the plugin's
    ``tool.call`` check (the plugin's ``resolveWithParents`` branch must
    fire on both Python and TS sides). ``edit_file`` would either fail
    or create the file ahead of the plugin's check on some Amp versions;
    ``create_file`` is the documented surface for new files.
    """
    return (
        f"Use the create_file tool to create a new file at the absolute path "
        f"{target} containing the single line:\n"
        f"{body}\n"
        "Make exactly one create_file tool call. Do not run any other tools "
        "(no Bash, no read_file, no grep, no edit_file). After the tool call "
        "returns, reply with the single word done and stop."
    )


# ---------------------------------------------------------------------------
# Helpers for lock-fixture management (always via real Python try_lock)
# ---------------------------------------------------------------------------


def _create_python_lock(
    target_path: str,
    agent_id: str,
    repo_namespace: str,
) -> Path:
    """Create a real ``<hash>.lock`` fixture via the production writer.

    Returns the resolved lock-file path. Asserts both the boolean and the
    on-disk file as a pre-condition for the cross-language contract test.
    """
    assert try_lock(target_path, agent_id, repo_namespace), (
        f"try_lock failed unexpectedly for {target_path!r} as {agent_id!r}"
    )
    lp = lock_path(target_path, repo_namespace)
    assert lp.exists(), f"expected lock file at {lp} after try_lock returned True"
    return lp


# ---------------------------------------------------------------------------
# Scenario tests
# ---------------------------------------------------------------------------


def test_allow_when_agent_holds_lock(amp_env: AmpLockEnv) -> None:
    """Allow case (plan L777-L779).

    Agent A holds the lock for ``foo.py``; the same agent's ``edit_file``
    is permitted. The plugin should not emit any rejection text and the
    file should be modifiable.
    """
    target = amp_env.repo_path / "foo.py"
    target.write_text("# original\n")

    _create_python_lock(
        str(target),
        _DEFAULT_AGENT_ID,
        str(amp_env.repo_path),
    )

    spawned_env = _build_amp_env(amp_env, agent_id=_DEFAULT_AGENT_ID)
    result = _run_amp(_edit_prompt(target), spawned_env, amp_env.repo_path)

    _assert_plugin_loaded(result)
    _skip_if_no_file_write(result.events, "allow")

    text = _tool_result_text(result.events)
    assert "is not locked" not in text, (
        f"unexpected no-lock rejection while agent A holds the lock: {text[:1024]}"
    )
    assert "is locked by" not in text, (
        f"unexpected wrong-agent rejection while agent A holds the lock: {text[:1024]}"
    )
    assert "lock-ownership env missing" not in text, (
        f"unexpected fail-closed rejection while env is fully populated: {text[:1024]}"
    )


def test_reject_when_no_lock(amp_env: AmpLockEnv) -> None:
    """No-lock case (plan L780-L782).

    No lock file exists; the plugin rejects with the no-lock message
    pointing the agent at ``lock_acquire``.
    """
    target = amp_env.repo_path / "no_lock.py"
    target.write_text("# original\n")

    spawned_env = _build_amp_env(amp_env, agent_id=_DEFAULT_AGENT_ID)
    result = _run_amp(_edit_prompt(target), spawned_env, amp_env.repo_path)

    _assert_plugin_loaded(result)
    _skip_if_no_file_write(result.events, "no-lock")

    text = _tool_result_text(result.events)
    assert "is not locked" in text, (
        "expected the plugin's no-lock rejection text "
        "('File ... is not locked. Use lock_acquire ...') "
        f"in tool_result content; got:\n{text[:2048]}"
    )


def test_reject_wrong_agent(amp_env: AmpLockEnv) -> None:
    """Wrong-agent case (plan L783-L784).

    Agent B holds the lock; agent A's ``edit_file`` is rejected with the
    "locked by <holder>" message.
    """
    target = amp_env.repo_path / "shared.py"
    target.write_text("# original\n")

    _create_python_lock(
        str(target),
        _OTHER_AGENT_ID,
        str(amp_env.repo_path),
    )

    spawned_env = _build_amp_env(amp_env, agent_id=_DEFAULT_AGENT_ID)
    result = _run_amp(_edit_prompt(target), spawned_env, amp_env.repo_path)

    _assert_plugin_loaded(result)
    _skip_if_no_file_write(result.events, "wrong-agent")

    text = _tool_result_text(result.events)
    assert "is locked by" in text and _OTHER_AGENT_ID in text, (
        "expected the plugin's wrong-agent rejection text "
        f"('File ... is locked by {_OTHER_AGENT_ID}.') in tool_result; got:\n"
        f"{text[:2048]}"
    )


def test_reject_wrong_namespace(amp_env: AmpLockEnv) -> None:
    """Wrong-namespace case (plan L785-L787).

    A lock exists for the same path under a different
    ``MALA_REPO_NAMESPACE``. The plugin (running with this run's
    namespace) does not see a matching ``<hash>.lock`` because the
    lock-key derivation includes the namespace; rejects with the no-lock
    message.

    This guards against a regression where the plugin and Python writer
    diverged on namespace handling — same file, different namespace, must
    yield distinct lock-file paths on both sides.
    """
    target = amp_env.repo_path / "ns_test.py"
    target.write_text("# original\n")

    other_ns = amp_env.repo_path.parent / "other-namespace"
    other_ns.mkdir()
    _create_python_lock(
        str(target),
        _DEFAULT_AGENT_ID,
        str(other_ns),
    )

    # Plugin runs with the test's namespace, NOT other_ns.
    spawned_env = _build_amp_env(
        amp_env,
        agent_id=_DEFAULT_AGENT_ID,
        repo_namespace=str(amp_env.repo_path),
    )
    result = _run_amp(_edit_prompt(target), spawned_env, amp_env.repo_path)

    _assert_plugin_loaded(result)
    _skip_if_no_file_write(result.events, "wrong-namespace")

    text = _tool_result_text(result.events)
    assert "is not locked" in text, (
        "expected the no-lock rejection: lock under a different namespace must "
        "not be visible to the plugin running with this run's namespace. "
        f"tool_result content:\n{text[:2048]}"
    )


def test_format_drift_resilience(amp_env: AmpLockEnv) -> None:
    """Cross-language format contract (plan L792-L796).

    The lock fixture is written by **real Python** :func:`try_lock`. The
    plugin parses it via its own canonicalize+SHA-256 helper. If either
    side drifts on:

      * filename hashing (``sha256(<namespace>:<canonical>)[:16] + .lock``)
      * key derivation (``<repo_namespace>:<canonical_path>``)
      * body format (``<agent_id>\\n``)

    the plugin will not match the fixture and the allow path will not
    fire. This test pins the contract.
    """
    target = amp_env.repo_path / "drift.py"
    target.write_text("# original\n")

    lock_file = _create_python_lock(
        str(target),
        _DEFAULT_AGENT_ID,
        str(amp_env.repo_path),
    )
    # Reassert format contract on the fixture itself for diagnostic clarity.
    body = lock_file.read_text()
    assert body == f"{_DEFAULT_AGENT_ID}\n", (
        "real Python try_lock must produce body=<agent_id>\\n exactly; the "
        "plugin trims and compares the first line. If this assertion fails, "
        "the contract has changed and plugins/amp/mala-safety.ts must be "
        f"updated in the same commit. Got: {body!r}"
    )
    assert lock_file.suffix == ".lock", (
        f"unexpected lock filename shape: {lock_file.name}"
    )

    spawned_env = _build_amp_env(amp_env, agent_id=_DEFAULT_AGENT_ID)
    result = _run_amp(_edit_prompt(target), spawned_env, amp_env.repo_path)

    _assert_plugin_loaded(result)
    _skip_if_no_file_write(result.events, "format-drift")

    text = _tool_result_text(result.events)
    # The fixture is correct; the plugin must NOT produce any of the
    # rejection messages. If any of these strings appear, format drift is
    # the most likely cause.
    assert "is not locked" not in text, (
        "format drift detected: the plugin reported 'not locked' but the "
        "fixture was written by the real Python try_lock for the same agent "
        f"and namespace. tool_result:\n{text[:2048]}"
    )
    assert "is locked by" not in text, (
        "format drift detected: the plugin reported a different holder than "
        "what the Python try_lock wrote. The body format "
        f"(<agent_id>\\n) is the cross-language contract. tool_result:\n{text[:2048]}"
    )


def test_path_canonicalization_parity(amp_env: AmpLockEnv) -> None:
    """Path canonicalization parity (plan L797-L801).

    A lock acquired with a relative path under the same
    ``MALA_REPO_NAMESPACE`` must match a tool.call carrying the absolute
    path for the same file. Mirrors
    :func:`src.infra.tools.locking._canonicalize_path` semantics.

    The Python writer and the TS reader both normalize via:

      * If absolute & exists → ``realpath()``.
      * If relative under namespace & exists → ``realpath(namespace/relpath)``.

    The inverse case (tool.call carrying a relative path) is not covered here
    because current Amp rejects relative ``edit_file`` paths before plugin
    hooks can inspect them.
    """
    rel_name = "canon_test.py"
    abs_path = amp_env.repo_path / rel_name
    abs_path.write_text("# original\n")

    _create_python_lock(
        rel_name,
        _DEFAULT_AGENT_ID,
        str(amp_env.repo_path),
    )

    spawned_env = _build_amp_env(amp_env, agent_id=_DEFAULT_AGENT_ID)
    result = _run_amp(_edit_prompt(abs_path), spawned_env, amp_env.repo_path)

    _assert_plugin_loaded(result)
    _skip_if_no_file_write(
        result.events,
        "path-canonicalization-parity[lock=relative,tool=absolute]",
    )

    text = _tool_result_text(result.events)
    assert "is not locked" not in text, (
        "canonicalization parity broke: lock_via=relative, tool_via=absolute. "
        "The plugin and Python writer disagreed on canonical path; the "
        f"<hash>.lock filenames diverged. tool_result:\n{text[:2048]}"
    )
    assert "is locked by" not in text, (
        "canonicalization parity broke (wrong-agent reported): "
        f"lock_via=relative, tool_via=absolute. tool_result:\n{text[:2048]}"
    )


@pytest.mark.parametrize(
    ("lock_via", "tool_via"),
    [
        ("symlinked", "realpath"),
        ("realpath", "symlinked"),
    ],
    ids=["lock=symlinked,tool=realpath", "lock=realpath,tool=symlinked"],
)
def test_symlinked_parent_canonicalization_parity(
    amp_env: AmpLockEnv, lock_via: str, tool_via: str
) -> None:
    """Symlinked-parent canonicalization parity (plan L800-L801).

    Setup: ``repo_path/real/x.py`` exists; ``repo_path/link`` is a
    directory symlink pointing to ``repo_path/real``. The lock is taken
    via one path shape (through the symlink, or via the realpath); the
    Amp ``tool.call`` carries the other. Both sides must canonicalize to
    the same realpath — the cross-language parity surface is
    Python's :func:`Path.resolve` / :func:`os.path.realpath` vs the TS
    plugin's :func:`fs.realpathSync` (called from
    ``mala-safety.ts::tryRealpath``).

    A divergence here would be a regression in symlink handling on
    either side: e.g., the TS plugin not realpath'ing through symlinked
    intermediate dirs would yield ``link/x.py`` as canonical and miss
    the lock written under ``real/x.py``.
    """
    real_dir = amp_env.repo_path / "real"
    real_dir.mkdir()
    real_target = real_dir / "x.py"
    real_target.write_text("# original\n")

    link_dir = amp_env.repo_path / "link"
    os.symlink(real_dir, link_dir, target_is_directory=True)
    sym_target = link_dir / "x.py"

    lock_path_arg = str(sym_target) if lock_via == "symlinked" else str(real_target)
    _create_python_lock(lock_path_arg, _DEFAULT_AGENT_ID, str(amp_env.repo_path))

    tool_target = sym_target if tool_via == "symlinked" else real_target

    spawned_env = _build_amp_env(amp_env, agent_id=_DEFAULT_AGENT_ID)
    result = _run_amp(_edit_prompt(tool_target), spawned_env, amp_env.repo_path)

    _assert_plugin_loaded(result)
    _skip_if_no_file_write(
        result.events,
        f"symlinked-parent-parity[lock={lock_via},tool={tool_via}]",
    )

    text = _tool_result_text(result.events)
    assert "is not locked" not in text, (
        f"symlinked-parent canonicalization parity broke: "
        f"lock_via={lock_via}, tool_via={tool_via}. The plugin's "
        "tryRealpath(symlink) diverged from Python's Path.resolve(). "
        f"tool_result:\n{text[:2048]}"
    )
    assert "is locked by" not in text, (
        f"symlinked-parent parity broke (wrong-agent reported): "
        f"lock_via={lock_via}, tool_via={tool_via}. tool_result:\n{text[:2048]}"
    )


def test_nonexistent_target_canonicalization_parity(amp_env: AmpLockEnv) -> None:
    """Non-existent-target canonicalization parity (plan L800-L801).

    The file does not exist at lock time **or** at the plugin's
    ``tool.call`` check; both sides must take the parents-walk branch
    (``_resolve_with_parents`` in Python; ``resolveWithParents`` in the
    TS plugin) and produce the same canonical path. ``create_file`` is
    used so the tool fires before the file exists; the plugin's hook
    runs first and must compute the same SHA-256 hash as the lock
    fixture.

    Pre-existing parent ``repo_path`` is the existing ancestor both
    walks reach; missing components ``newdir/missing.py`` are appended
    in the same order on both sides.
    """
    target = amp_env.repo_path / "newdir" / "missing.py"
    assert not target.exists(), "test setup invariant: file must not exist"
    assert not target.parent.exists(), (
        "test setup invariant: parent dir must not exist either, so the "
        "parents-walk on both sides walks past it up to repo_path"
    )

    _create_python_lock(str(target), _DEFAULT_AGENT_ID, str(amp_env.repo_path))

    spawned_env = _build_amp_env(amp_env, agent_id=_DEFAULT_AGENT_ID)
    result = _run_amp(_create_prompt(target), spawned_env, amp_env.repo_path)

    _assert_plugin_loaded(result)
    # File-write predicate accepts create_file (in FILE_WRITE_TOOLS).
    _skip_if_no_file_write(result.events, "nonexistent-target-parity")

    text = _tool_result_text(result.events)
    assert "is not locked" not in text, (
        "non-existent-target canonicalization parity broke: the plugin's "
        "resolveWithParents diverged from Python's _resolve_with_parents. "
        "Both should walk up from missing.py → newdir → repo_path "
        "(existing) and append missing components in the same order. "
        f"tool_result:\n{text[:2048]}"
    )
    assert "is locked by" not in text, (
        "non-existent-target parity broke (wrong-agent reported); "
        f"tool_result:\n{text[:2048]}"
    )

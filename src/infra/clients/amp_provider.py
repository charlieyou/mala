"""Amp :class:`AgentProvider` implementation (T013).

Replaces the T007 stub with the real provider. Binds:
  * :class:`AmpClient` (T008) via the lazy ``client_factory``
  * :class:`AmpRuntimeBuilder` (T012) via :meth:`runtime_builder`
  * :class:`AmpLogProvider` (T011) via the lazy ``evidence_provider``
  * :class:`AmpPluginInstaller` (T010) inside :meth:`install_prerequisites`

:meth:`install_prerequisites` is the safety-critical fail-closed gate
documented at ``plans/2026-04-29-amp-provider-plan.md#L168-L171,L271-L321``.
On every Amp run it:

  1. Warns once (per run) when ``MALA_DISALLOWED_TOOLS`` is set, since the
     env var is a no-op under ``coder=amp`` in MVP (plan ``L690``).
  2. Installs ``mala-safety.ts`` to ``~/.config/amp/plugins/`` via
     :class:`AmpPluginInstaller`.
  3. Runs the runtime plugin-load self-test using ``runtime_builder(repo_path,
     "amp-selftest", mcp_server_factory=...)`` so the same env / ``PLUGINS=all``
     / ``--mcp-config`` real sessions use is exercised. The self-test passes
     when the plugin's ``session.start`` sentinel marker arrives on stderr
     or in Amp's debug log within a bounded timeout AND the version hash
     matches the installed plugin's hash; the runtime terminates ``amp``
     immediately on detection before any LLM call to bound self-test
     latency to plugin-load time.
  4. Caches the result in-memory keyed on ``(amp_version, plugin_hash)``
     for the duration of the run; not persisted across runs.

If any of the failure signatures fires (npm-installed Amp, Bun missing,
``PLUGINS=all`` not honored, version mismatch, amp binary missing, sentinel
marker absent), an :class:`AmpPluginNotActiveError` is raised with a
structured reason and a pointer to the binary-install docs. The orchestrator
refuses to spawn any issue agent in that case (AC#16, AC#17).

Lazy-import contract: importing this module does NOT pull in
``claude_agent_sdk`` or any Amp-specific subprocess machinery. The Amp
client / log provider / plugin installer are imported inside method bodies
so a Claude-only run is not penalized by Amp adapters being on disk.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from enum import StrEnum
from typing import IO, TYPE_CHECKING, Any, Literal, cast

from src.infra.clients.amp_runtime import AmpRuntimeBuilder

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.core.protocols.agent_provider import CoderRuntimeBuilder
    from src.core.protocols.evidence import EvidenceProvider
    from src.core.protocols.lifecycle import DeadlockMonitorProtocol
    from src.core.protocols.sdk import (
        McpServerFactory,
        SDKClientFactoryProtocol,
        SDKClientProtocol,
    )
    from src.infra.clients.amp_runtime import AmpMode, AmpRuntime

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Amp MCP server factory (provider-owned)
# ---------------------------------------------------------------------------


def _create_amp_mcp_server_factory() -> McpServerFactory:
    """Amp-shaped MCP server factory returning **stdio launch specs**.

    The Amp ``--mcp-config`` payload is plain JSON (see
    :class:`AmpRuntimeBuilder.build`); it cannot consume the in-process
    Claude SDK server objects produced by the Claude provider's factory
    (they contain a ``Server`` instance and fail ``json.dumps``).

    Wires Amp at the same ``locking_mcp`` server the Claude path uses by
    pointing at the ``mala-amp-mcp-locking`` console script (registered
    via ``[project.scripts]``; entry point lives at
    :mod:`src.infra.tools.locking_mcp_stdio`). The launcher exposes
    ``lock_acquire`` / ``lock_release`` over stdio JSON-RPC and writes
    ``<hash>.lock`` files via the same Python primitives
    (:func:`src.infra.tools.locking.try_lock`) that
    :mod:`src.infra.tools.locking_mcp` uses, so the plugin's lock-key
    derivation in ``plugins/amp/mala-safety.ts`` sees identical fixtures
    regardless of which path produced them — closing AC#9 (plan
    ``L702``).

    ``MALA_AGENT_ID`` and ``MALA_REPO_NAMESPACE`` are also passed via
    args **and** env so the launcher works whether Amp inherits parent
    env into spawned MCP children or not. ``MALA_LOCK_DIR`` is forwarded
    from the orchestrator's environment when set (it is what the Claude
    path uses to choose the lock directory); when unset, the launcher
    falls back to the default in :func:`src.infra.tools.env.get_lock_dir`,
    matching Claude-side behavior.

    The ``emit_lock_event`` parameter is interpreted by
    :class:`AmpRuntimeBuilder`: when present it injects ``MALA_LOCK_EVENT_LOG``
    into this stdio spec, and ``AmpClient`` tails that JSONL side channel back
    into the orchestrator's deadlock monitor.
    """

    def factory(
        agent_id: str,
        repo_path: Path,
        emit_lock_event: Callable | None,
    ) -> dict[str, object]:
        del emit_lock_event
        env: dict[str, str] = {
            "MALA_AGENT_ID": agent_id,
            "MALA_REPO_NAMESPACE": str(repo_path),
        }
        # Forward the orchestrator's lock dir when one is configured so
        # the spawned launcher writes lock files where the rest of the
        # run reads them. Falling through to the env.py default when the
        # var is unset matches the Claude path's behavior.
        lock_dir = os.environ.get("MALA_LOCK_DIR")
        if lock_dir:
            env["MALA_LOCK_DIR"] = lock_dir

        return {
            "mala-locking": {
                "command": "mala-amp-mcp-locking",
                "args": [
                    "--agent-id",
                    agent_id,
                    "--repo-namespace",
                    str(repo_path),
                ],
                "env": env,
            }
        }

    return factory


# ---------------------------------------------------------------------------
# Public exception + reason enum
# ---------------------------------------------------------------------------


class AmpPluginNotActiveReason(StrEnum):
    """Why the Amp plugin self-test failed (structured form for callers).

    The string values are stable identifiers the orchestrator / telemetry
    can branch on without parsing prose error messages.
    """

    PLUGIN_MARKER_MISSING = "plugin_marker_missing"
    VERSION_MISMATCH = "version_mismatch"
    BUN_UNAVAILABLE = "bun_unavailable"
    PLUGINS_ALL_UNSET = "plugins_all_unset"
    AMP_BINARY_MISSING = "amp_binary_missing"
    NPM_INSTALL_FINGERPRINT = "npm_install_fingerprint"


_BINARY_INSTALL_DOCS_URL = "https://ampcode.com/manual"


class AmpPluginNotActiveError(RuntimeError):
    """Fail-closed error raised when the Amp safety plugin is not active.

    Carries a structured :attr:`reason` so the orchestrator can branch on
    failure type and present a tailored remediation. The message always
    points at the binary-install docs so end users have a single landing
    page for the fix.
    """

    def __init__(
        self,
        message: str,
        *,
        reason: AmpPluginNotActiveReason,
        docs_url: str = _BINARY_INSTALL_DOCS_URL,
    ) -> None:
        super().__init__(f"{message} See {docs_url} for the binary install path.")
        self.reason = reason
        self.docs_url = docs_url


# ---------------------------------------------------------------------------
# Self-test internals
# ---------------------------------------------------------------------------


_SELFTEST_TIMEOUT_SECONDS = 10.0
"""Bound on plugin-load time. Plan ``L822`` uses 10s as the example."""

_SELFTEST_PROMPT = (
    "/* mala plugin self-test - terminate immediately on session.start */\n"
)
"""Prompt fed to ``amp --execute`` stdin. The plugin self-test only depends
on ``session.start`` firing (which happens before any LLM call), so any
non-empty input that does not trigger an early validation error suffices."""

_SELFTEST_AGENT_ID = "amp-selftest"
"""Per plan ``L292``: synthetic agent id for the self-test runtime; it never
reaches a real beads issue."""

_TERMINATE_GRACE_SECONDS = 1.0
"""SIGTERM → SIGKILL grace window for the self-test subprocess."""


# Stderr fingerprints used to classify a failed self-test.
_BUN_FINGERPRINTS: tuple[str, ...] = (
    "bun: command not found",
    "bun: not found",
    "no such file or directory: bun",
    "bun runtime",
)
_NPM_INSTALL_FINGERPRINTS: tuple[str, ...] = (
    "@sourcegraph/amp",
    "npm install -g",
    "node_modules/@sourcegraph/amp",
    "npm-installed",
)
_PLUGINS_ALL_FINGERPRINTS: tuple[str, ...] = (
    "plugins=all",
    "plugins env not set",
    "plugin loading is disabled",
)


def _classify_no_marker(stderr_text: str) -> AmpPluginNotActiveReason:
    """Map captured stderr to the most likely failure reason.

    Order matters: NPM-install fingerprints are the most actionable
    diagnostic (the docs fix is "switch to the binary install"), so they
    take priority over generic plugin-missing.
    """
    lowered = stderr_text.lower()
    if any(p.lower() in lowered for p in _NPM_INSTALL_FINGERPRINTS):
        return AmpPluginNotActiveReason.NPM_INSTALL_FINGERPRINT
    if any(p.lower() in lowered for p in _BUN_FINGERPRINTS):
        return AmpPluginNotActiveReason.BUN_UNAVAILABLE
    if any(p.lower() in lowered for p in _PLUGINS_ALL_FINGERPRINTS):
        return AmpPluginNotActiveReason.PLUGINS_ALL_UNSET
    return AmpPluginNotActiveReason.PLUGIN_MARKER_MISSING


def _extract_sentinel_marker(text: str) -> dict[str, Any] | None:
    """Return the first ``mala_plugin=loaded`` marker found in ``text``.

    Current Amp writes plugin stderr into its structured debug log as a JSON
    line with a nested ``data`` string. Older versions forwarded the plugin's
    raw JSON line directly to process stderr. Support both shapes.
    """
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("mala_plugin") == "loaded":
            return obj
        if not isinstance(obj, dict):
            continue
        data = obj.get("data")
        if not isinstance(data, str):
            continue
        try:
            nested = json.loads(data.strip())
        except json.JSONDecodeError:
            continue
        if isinstance(nested, dict) and nested.get("mala_plugin") == "loaded":
            return nested
    return None


def _read_text_if_exists(path: Path) -> str:
    """Best-effort text read for optional diagnostics files."""
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def _stderr_reader(
    stream: IO[bytes],
    state: dict[str, Any],
    done: threading.Event,
) -> None:
    """Read stderr line-by-line; record the first sentinel marker found.

    Captures every line text into ``state['lines']`` and stops as soon as
    a JSON object with ``mala_plugin == "loaded"`` is observed (recorded
    in ``state['marker']``). On EOF, ``done`` is set unconditionally so
    the main thread does not block past process exit.
    """
    try:
        while True:
            raw = stream.readline()
            if not raw:
                return
            try:
                text = raw.decode("utf-8", errors="replace")
            except Exception:  # defensive
                text = repr(raw)
            state["lines"].append(text)
            stripped = text.strip()
            if not stripped:
                continue
            marker = _extract_sentinel_marker(stripped)
            if marker is not None:
                state["marker"] = marker
                return
    finally:
        done.set()


def _terminate_process(proc: subprocess.Popen[bytes]) -> None:
    """SIGTERM → grace → SIGKILL the self-test subprocess group.

    The subprocess is spawned with ``start_new_session=True`` (where
    supported) so the whole process tree (e.g., bash → sleep) is in a
    fresh session whose group leader is ``proc.pid``. We signal the
    whole group; otherwise an orphan child holding stderr open would
    keep our reader thread blocked past the bounded timeout.
    """
    if proc.poll() is not None:
        return

    use_pgroup = sys.platform != "win32"
    try:
        if use_pgroup:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        else:
            proc.terminate()
    except OSError:
        return

    try:
        proc.wait(timeout=_TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            if use_pgroup:
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
        except (ProcessLookupError, PermissionError, OSError):
            pass
        try:
            proc.wait(timeout=_TERMINATE_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            pass


def _run_selftest_subprocess(
    runtime: AmpRuntime,
    expected_hash: str,
    *,
    timeout_seconds: float | None = None,
) -> None:
    """Spawn the self-test subprocess; raise on any fail-closed signature.

    Implements the plan's L302-L320 contract:

    1. ``shutil.which("amp")`` / ``Popen`` resolves the binary against the
       runtime's PATH; missing binary is the first signal.
    2. The plugin emits ``{"mala_plugin": "loaded", "version": "<hash>"}``
       at ``session.start`` (before any LLM call). Current Amp routes plugin
       stderr into the CLI debug log, while older builds forwarded it to
       process stderr. We block until either marker source appears or the
       bounded timeout elapses.
    3. We terminate ``amp`` as soon as the marker is seen so latency is
       capped at plugin-load time and no LLM cost is incurred per run.
    4. If the marker arrives but the version hash does not match the
       installed plugin's hash, raise ``VERSION_MISMATCH``.
    5. If no marker arrives within the timeout, classify stderr and raise
       the most likely cause (npm install / Bun missing / PLUGINS=all
       unset / generic missing).

    The default timeout reads :data:`_SELFTEST_TIMEOUT_SECONDS` from the
    module **at call time**, not at function-definition time, so a test
    that does ``monkeypatch.setattr(amp_provider, "_SELFTEST_TIMEOUT_SECONDS",
    1.0)`` actually shrinks the bound for that test.
    """
    if timeout_seconds is None:
        timeout_seconds = _SELFTEST_TIMEOUT_SECONDS
    env = dict(runtime.env)
    path = env.get("PATH", "")
    # Up-front binary check produces the cleanest error for the missing-
    # binary case (no spawn-time exception trace mixed in).
    if shutil.which("amp", path=path) is None:
        raise AmpPluginNotActiveError(
            "amp binary not found on PATH; coder=amp requires the official "
            "binary install (the npm package is not supported because plugins "
            "do not load there).",
            reason=AmpPluginNotActiveReason.AMP_BINARY_MISSING,
        )

    plugin_log_path = runtime.log_path.with_suffix(".plugin-selftest.log")
    try:
        plugin_log_path.parent.mkdir(parents=True, exist_ok=True)
        plugin_log_path.unlink(missing_ok=True)
    except OSError:
        pass
    selftest_argv = [
        *runtime.argv,
        "--log-level",
        "debug",
        "--log-file",
        str(plugin_log_path),
    ]

    try:
        proc = subprocess.Popen(
            selftest_argv,
            cwd=str(runtime.cwd),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            # Put amp + its children (bun runtime, plugin process, MCP
            # servers) into a fresh session/process-group so SIGTERM /
            # SIGKILL during cleanup reaches the whole tree. Without this,
            # an orphan child holding stderr open would block the reader
            # thread past the bounded self-test timeout (plan L309-L312).
            start_new_session=sys.platform != "win32",
        )
    except FileNotFoundError as exc:
        raise AmpPluginNotActiveError(
            f"amp binary not spawnable on PATH: {exc}",
            reason=AmpPluginNotActiveReason.AMP_BINARY_MISSING,
        ) from exc
    except OSError as exc:
        raise AmpPluginNotActiveError(
            f"Could not spawn amp self-test: {exc}",
            reason=AmpPluginNotActiveReason.AMP_BINARY_MISSING,
        ) from exc

    state: dict[str, Any] = {"lines": [], "marker": None}
    done = threading.Event()
    reader_thread = threading.Thread(
        target=_stderr_reader,
        args=(proc.stderr, state, done),
        daemon=True,
        name="amp-selftest-stderr",
    )
    reader_thread.start()

    try:
        # Send a no-op prompt and close stdin so amp can begin its session.
        if proc.stdin is not None:
            try:
                proc.stdin.write(_SELFTEST_PROMPT.encode("utf-8"))
                proc.stdin.close()
            except (OSError, BrokenPipeError):
                # amp may already have exited (e.g. immediate auth failure).
                pass

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if state["marker"] is not None:
                break
            marker = _extract_sentinel_marker(_read_text_if_exists(plugin_log_path))
            if marker is not None:
                state["marker"] = marker
                done.set()
                break
            if proc.poll() is not None:
                done.wait(timeout=0.05)
                marker = _extract_sentinel_marker(_read_text_if_exists(plugin_log_path))
                if marker is not None:
                    state["marker"] = marker
                    done.set()
                break
            time.sleep(0.05)
    finally:
        _terminate_process(proc)
        # The reader thread captured every line via readline(); after
        # SIGTERM/SIGKILL hits the whole process group, all stderr writers
        # close their FDs and readline() returns b"" so the reader exits.
        # Bound the join because a misbehaving stderr writer (or a kernel
        # quirk) shouldn't be able to keep the orchestrator past the
        # self-test deadline.
        reader_thread.join(timeout=_TERMINATE_GRACE_SECONDS)
        if proc.stderr is not None:
            try:
                proc.stderr.close()
            except OSError:
                pass
        if proc.stdout is not None:
            try:
                proc.stdout.close()
            except OSError:
                pass

    stderr_text = "".join(state["lines"])
    plugin_log_text = _read_text_if_exists(plugin_log_path)
    if plugin_log_text:
        stderr_text = f"{stderr_text}\n{plugin_log_text}"
    marker = state["marker"]

    if marker is not None:
        version = str(marker.get("version", ""))
        if version != expected_hash:
            raise AmpPluginNotActiveError(
                "Amp safety plugin loaded but its version hash does not "
                f"match the installed file (expected {expected_hash!r}, "
                f"got {version!r}). The on-disk plugin is stale — rerun "
                "to refresh, or remove ~/.config/amp/plugins/mala-safety.ts "
                "and try again.",
                reason=AmpPluginNotActiveReason.VERSION_MISMATCH,
            )
        return  # success

    reason = _classify_no_marker(stderr_text)
    if reason is AmpPluginNotActiveReason.NPM_INSTALL_FINGERPRINT:
        msg = (
            "Amp safety plugin did not load and the install looks like the "
            "npm package (@sourcegraph/amp). The npm install path does not "
            "honor plugins; install Amp via the official binary path instead."
        )
    elif reason is AmpPluginNotActiveReason.BUN_UNAVAILABLE:
        msg = (
            "Amp safety plugin did not load because the Bun runtime is "
            "missing or unusable. Bun is provided by the official Amp binary "
            "install; verify the install."
        )
    elif reason is AmpPluginNotActiveReason.PLUGINS_ALL_UNSET:
        msg = (
            "Amp safety plugin did not load because PLUGINS=all was not "
            "honored by the amp invocation. mala always sets it; this points "
            "at an Amp install where plugin loading is disabled."
        )
    else:
        msg = (
            "Amp safety plugin did not emit its session.start sentinel marker "
            "within the self-test timeout. Plugin appears not to be loaded; "
            "the orchestrator refuses to run under --dangerously-allow-all "
            "without the safety plugin active."
        )
    raise AmpPluginNotActiveError(msg, reason=reason)


# ---------------------------------------------------------------------------
# MALA_DISALLOWED_TOOLS warn-once
# ---------------------------------------------------------------------------


_MALA_DISALLOWED_WARNING = (
    "MALA_DISALLOWED_TOOLS is set but has no effect under coder=amp in MVP. "
    "Amp does not expose an SDK-level disallowed-tools surface; the bundled "
    "mala-safety.ts plugin enforces dangerous-command + lock-ownership "
    "checks instead. Tracked as a follow-up."
)


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


class _AmpClientFactory:
    """Slim :class:`SDKClientFactoryProtocol` implementation for the Amp coder.

    Implements just the cross-coder surface (``create`` / ``with_resume``).
    Claude-only knobs (``create_options`` / ``create_hook_matcher``) do
    not appear on the protocol, so the Amp factory has no
    ``NotImplementedError`` walls — a misrouted Claude code path now
    fails at type-check time rather than deep inside the subprocess
    pipeline.
    """

    def create(self, runtime: object) -> SDKClientProtocol:
        """Return a new :class:`AmpClient` for ``runtime``.

        ``runtime`` must be an :class:`AmpRuntime` produced by
        :meth:`AmpAgentProvider.runtime_builder().build()`. The factory
        privately unpacks the runtime's argv/env/log_path into an
        :class:`AmpClientOptions`; the pipeline never inspects the
        runtime shape (plan A3).

        The lazy import keeps :mod:`amp_client` (and its asyncio +
        signal imports) out of the Claude-only path.
        """
        from src.infra.clients.amp_client import AmpClient, AmpClientOptions
        from src.infra.clients.amp_runtime import AmpRuntime

        if not isinstance(runtime, AmpRuntime):
            raise TypeError(
                "AmpAgentProvider.client_factory.create(runtime) requires an "
                "AmpRuntime; got "
                f"{type(runtime).__name__}. Build it via "
                "AmpAgentProvider.runtime_builder(...).build()."
            )
        options = AmpClientOptions(
            cwd=runtime.cwd,
            env=runtime.env,
            argv=runtime.argv,
            log_path=runtime.log_path,
            thread_id=runtime.resume_thread_id,
            lock_event_log_path=runtime.lock_event_log_path,
            lock_event_callback=runtime.lock_event_callback,
        )
        return cast("SDKClientProtocol", AmpClient(options))

    def with_resume(self, runtime: object, resume: str | None) -> object:
        """Return a copy of ``runtime`` continuing Amp thread ``resume``.

        Updates :attr:`AmpRuntime.resume_thread_id`; the next
        :meth:`create` materializes an :class:`AmpClientOptions` whose
        ``thread_id`` carries the resume token, and the configured
        resume strategy (``amp threads continue <id>`` by default) is
        applied at spawn time.
        """
        from src.infra.clients.amp_runtime import AmpRuntime

        if not isinstance(runtime, AmpRuntime):
            raise TypeError(
                "AmpAgentProvider.client_factory.with_resume(runtime, resume) "
                "requires AmpRuntime; got "
                f"{type(runtime).__name__}."
            )
        if resume is None:
            return runtime
        return replace(runtime, resume_thread_id=resume)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class AmpAgentProvider:
    """Real :class:`AgentProvider` for the Amp coder backend.

    Conforms structurally to
    :class:`src.core.protocols.agent_provider.AgentProvider`. Components
    are lazy-instantiated so importing this module does not pull in
    :mod:`amp_client` / :mod:`amp_log_provider` until they are first
    used (the Claude-only path never touches them).

    Attributes:
        name: Provider identifier (always ``"amp"``).
        client_factory: :class:`SDKClientFactoryProtocol` returning
            :class:`AmpClient` instances.
        evidence_provider: :class:`EvidenceProvider` reading the per-thread
            tee log.
    """

    name: Literal["amp"] = "amp"

    def __init__(self, *, mode: AmpMode = "deep", effort: str | None = None) -> None:
        """Initialize the provider.

        Args:
            mode: Amp execution mode (``"smart" | "rush" | "deep"``).
                Sourced from ``MalaConfig.coder_options.amp.mode`` after
                CLI > env > yaml > default precedence resolution.
            effort: Optional Mala-level reasoning effort. Forwarded to the
                Amp CLI as ``--effort <value>`` for smart and deep modes.
                Rush mode ignores it; the runtime builder logs an info message
                when effort is set against rush so the silent drop is
                observable in logs.
        """
        self._mode: AmpMode = mode
        self._effort: str | None = effort
        self._client_factory_cached: _AmpClientFactory | None = None
        self._evidence_provider_cached: object | None = None

        # In-memory self-test cache keyed on (amp_version, plugin_hash).
        # Per plan L121: not persisted across runs. Per plan L823: invoked
        # exactly once per run.
        self._selftest_cache_key: tuple[str, str] | None = None
        self._disallowed_warned: bool = False

    # ------------------------------------------------------------------
    # AgentProvider protocol surface
    # ------------------------------------------------------------------

    @property
    def client_factory(self) -> SDKClientFactoryProtocol:
        """Lazily-instantiated Amp client factory.

        Constructed on first access so importing :mod:`amp_provider`
        does not pull in the subprocess + asyncio machinery from
        :mod:`amp_client`.
        """
        if self._client_factory_cached is None:
            self._client_factory_cached = _AmpClientFactory()
        return cast("SDKClientFactoryProtocol", self._client_factory_cached)

    @property
    def evidence_provider(self) -> EvidenceProvider:
        """Lazily-instantiated Amp evidence provider.

        Constructed on first access using
        :meth:`AmpLogProvider.from_probe` so the native-log probe runs
        once per run (per plan ``L122``).
        """
        if self._evidence_provider_cached is None:
            from src.infra.clients.amp_log_provider import AmpLogProvider

            self._evidence_provider_cached = AmpLogProvider.from_probe()
        return cast("EvidenceProvider", self._evidence_provider_cached)

    def runtime_builder(
        self,
        repo_path: Path,
        agent_id: str,
        *,
        mcp_server_factory: McpServerFactory,
        deadlock_monitor: DeadlockMonitorProtocol | None = None,
    ) -> CoderRuntimeBuilder:
        """Construct an :class:`AmpRuntimeBuilder` for one Amp session."""
        return AmpRuntimeBuilder(
            repo_path,
            agent_id,
            mcp_server_factory,
            mode=self._mode,
            effort=self._effort,
            deadlock_monitor=deadlock_monitor,
        )

    def mcp_server_factory(self) -> McpServerFactory:
        """Return the Amp-shaped MCP server factory (stdio launch specs)."""
        return _create_amp_mcp_server_factory()

    def install_prerequisites(
        self,
        repo_path: Path,
        *,
        mcp_server_factory: McpServerFactory,
    ) -> None:
        """Install ``mala-safety.ts`` and run the plugin-load self-test.

        Idempotent within a run: the result is cached on
        ``(amp_version, plugin_hash)``; subsequent calls within the same
        run short-circuit without respawning ``amp`` and without
        re-emitting the ``MALA_DISALLOWED_TOOLS`` warning.

        Raises:
            AmpPluginNotActiveError: When the safety plugin is not active
                at runtime (see :class:`AmpPluginNotActiveReason` for the
                structured failure modes).
        """
        # 1. MALA_DISALLOWED_TOOLS warn-once (plan L690). Bound to
        # install_prerequisites, which fires once per run before the first
        # session, so the warning is emitted at most once per run.
        if not self._disallowed_warned and os.environ.get("MALA_DISALLOWED_TOOLS"):
            logger.warning(_MALA_DISALLOWED_WARNING)
            self._disallowed_warned = True

        # 2. Install the plugin file (idempotent on its own).
        from src.infra.clients.amp_plugin_installer import AmpPluginInstaller

        installer = AmpPluginInstaller()
        install_result = installer.install()
        plugin_hash = install_result.plugin_hash

        # 3. Cache key: (amp_version, plugin_hash). Plan L121 keys on both,
        # but plan L109 also calls out that automatic version detection is
        # only adopted when "a one-line `amp --version` parse is trivially
        # reliable" — `amp --version` against arbitrary installs is not
        # reliable enough today (some amp builds keep the process alive
        # past version output and stall ``subprocess.run`` past its
        # timeout when children inherit stderr). We therefore key on a
        # placeholder version and the plugin_hash; any future move to a
        # real version probe is additive.
        cache_key = ("unknown", plugin_hash)
        if self._selftest_cache_key == cache_key:
            return

        # 4. Build the self-test runtime via the same runtime_builder real
        # sessions use, so PLUGINS=all + MALA_* env + the --mcp-config
        # payload all hit the same code path (plan L302-L312).
        builder = self.runtime_builder(
            repo_path,
            _SELFTEST_AGENT_ID,
            mcp_server_factory=mcp_server_factory,
        )
        runtime = cast("AmpRuntime", builder.build())

        # 5. Spawn amp + read sentinel marker; raises AmpPluginNotActiveError
        # on any fail-closed signature.
        _run_selftest_subprocess(runtime, plugin_hash)

        # 6. Cache success for the rest of the run.
        self._selftest_cache_key = cache_key

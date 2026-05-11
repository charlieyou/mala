"""Codex selftest helpers — extracted from :mod:`src.infra.clients.codex_provider`.

Workstream B / finding #3: ``codex_provider`` becomes a facade over
focused helper modules (MCP factory, plugin installer, selftest).
This module owns the selftest surface — the pure hash/identity helpers
that back the trusted-hash computation, *plus* the live + structural
probes the provider's ``install_prerequisites`` delegates to.

The probes raise :class:`CodexHookNotActiveError` (defined in
:mod:`codex_provider`) on fail-closed signatures; that import lives
inside each probe body so this module can be imported by
``codex_provider`` itself without a circular import. The provider
remains the single source of truth for the public error types
``CodexHookNotActiveError`` / ``CodexHookNotActiveReason`` so callers
that catch them by name continue to import from ``codex_provider``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

from src.core.constants import DEFAULT_CODEX_MODEL
from src.infra.clients.codex_sdk_compat import (
    import_codex_app_server,
    import_codex_generated_v2_all,
    resolve_codex_bin_for_app_server,
    start_thread_compat,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine


_T = TypeVar("_T")


def _normalized_hook_identity_value(event: str = "pre_tool_use") -> object:
    """Build the ``NormalizedHookIdentity`` payload Codex hashes for ``current_hash``.

    Source-of-truth: ``codex-rs/hooks/src/engine/discovery.rs::command_hook_hash``
    serializes the struct to TOML, then ``codex-rs/config/src/fingerprint.rs::version_for_toml``
    canonicalizes the JSON and returns ``"sha256:<hex>"``. The struct is
    ``{event_name, **MatcherGroup}`` with ``MatcherGroup = {matcher, hooks: [HookHandlerConfig]}``.
    Our hook is a single command handler with no matcher; Codex
    normalizes the timeout to 600 (``unwrap_or(600).max(1)``,
    ``discovery.rs:409``) and emits ``async`` / ``status_message`` /
    ``timeout`` per the ``HookHandlerConfig::Command`` serde shape
    (``hook_config.rs:123-135``: rename ``timeout_sec`` → ``timeout``,
    ``status_message`` → ``statusMessage``). ``Option::None`` fields are
    not present in the TOML round-trip used by the hash routine.
    """
    return {
        "event_name": event,
        "hooks": [
            {
                "type": "command",
                "command": "mala-codex-pre-tool-use",
                "timeout": 600,
                "async": False,
            }
        ],
    }


def _canonical_json(value: object) -> object:
    """Canonical-JSON normalization: dict keys sorted, lists left in order.

    Mirrors ``codex-rs/config/src/fingerprint.rs::canonical_json``.
    """
    if isinstance(value, dict):
        return {k: _canonical_json(value[k]) for k in sorted(value.keys())}
    if isinstance(value, list):
        return [_canonical_json(item) for item in value]
    return value


def _compute_normalized_hook_hash(event: str = "pre_tool_use") -> str:
    """Compute the ``sha256:<hex>`` ``current_hash`` Codex assigns the bundled hook.

    The Rust path serializes the struct to TOML, canonicalizes it, and
    JSON-encodes it before hashing. We approximate that by serializing
    the equivalent structure as canonical JSON (``json.dumps`` with
    ``sort_keys=True``, separators tightly packed). The TOML→JSON
    round-trip in the Rust pipeline happens to land on the same JSON
    shape Python's ``json.dumps`` produces for plain types (string,
    int, bool, list, dict), so the hex digest matches in practice for
    our minimal command-hook payload.
    """
    canonical = _canonical_json(_normalized_hook_identity_value(event))
    serialized = json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(serialized).hexdigest()}"


def _build_module_hash_probe_code(modules: tuple[str, ...]) -> str:
    """Emit the probe Python code that hashes the on-PATH interpreter's
    view of every module in ``modules``.

    The probe code resolves each module via the same
    ``importlib.util.find_spec`` path the hook itself uses at startup,
    reads its source bytes, and folds them into a length-prefixed
    deterministic SHA-256. ``NOMODULE:<name>`` is printed and the probe
    exits non-zero when a module cannot be located. Output on success
    is a single hex digest line.
    """
    return (
        "import importlib.util, hashlib, sys\n"
        f"modules = {modules!r}\n"
        "h = hashlib.sha256()\n"
        "for name in modules:\n"
        "    spec = importlib.util.find_spec(name)\n"
        "    if spec is None or spec.origin is None:\n"
        "        print('NOMODULE:' + name); sys.exit(2)\n"
        "    with open(spec.origin, 'rb') as fp:\n"
        "        data = fp.read()\n"
        "    h.update(name.encode('utf-8'))\n"
        "    h.update(b'\\0')\n"
        "    h.update(len(data).to_bytes(8, 'big'))\n"
        "    h.update(data)\n"
        "print(h.hexdigest())\n"
    )


def _run_coro_sync(coro_factory: Callable[[], Coroutine[Any, Any, _T]]) -> _T:
    """Run an async Codex SDK probe from sync prerequisite code.

    ``install_prerequisites`` is intentionally synchronous because it is
    used by the CLI/provider setup path. Some tests and embedding callers
    invoke that sync API from an existing event loop, where ``asyncio.run``
    is illegal. In that case, run the SDK coroutine on a short-lived helper
    thread with its own event loop and propagate the original result or
    exception back to the caller.
    """
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        has_running_loop = False
    else:
        has_running_loop = True

    if not has_running_loop:
        return asyncio.run(coro_factory())

    outcome: dict[str, object] = {}

    def _runner() -> None:
        try:
            outcome["result"] = asyncio.run(coro_factory())
        except BaseException as exc:
            outcome["error"] = exc

    thread = threading.Thread(target=_runner, name="mala-codex-sdk-probe")
    thread.start()
    thread.join()
    error = outcome.get("error")
    if error is not None:
        raise cast("BaseException", error)
    return cast("_T", outcome.get("result"))


# ---------------------------------------------------------------------------
# Default selftest probe (Phase E5 structural verification + real-Codex spawn)
# ---------------------------------------------------------------------------


_HOOK_COMMAND_NAME = "mala-codex-pre-tool-use"
"""Hook command literal we expect to find inside the installed plugin's
``hooks.json``. Matches the ``[project.scripts]`` entry in
``pyproject.toml`` and the bundled ``hooks.json`` registration."""


_HOOK_PROBE_TIMEOUT_SECONDS = 10.0
"""Timeout for the on-PATH hook subprocess invocation in the default
probe. The hook is a pure JSON-in/JSON-out script that returns
immediately; any wall-clock time past this bound implies it's hung
(stale install, broken interpreter, etc.) and the run should fail
closed."""


_HOOK_SELFTEST_MARKER_ENV = "MALA_CODEX_HOOK_SELFTEST_MARKER"
"""Env var the live-hook step sets so the hook writes a presence /
version marker to a known temp path. Mirrored verbatim from
:data:`src.infra.hooks.codex.selftest_policy.SELFTEST_MARKER_ENV` —
import-linter forbids ``src.infra.clients`` from importing
``src.infra.hooks`` directly, so the literal is duplicated and the
equality is pinned by a unit test."""


_HOOK_SELFTEST_MARKER_DIR_ENV = "MALA_CODEX_HOOK_SELFTEST_MARKER_DIR"
"""Env var pointing the hook at a directory for per-event markers.

Mirrors :data:`src.infra.hooks.codex.selftest_policy.SELFTEST_MARKER_DIR_ENV`.
Used by the live PreToolUse-driven dispatch probe so SessionStart
and PreToolUse hook firing produce distinct per-event marker files
(``SessionStart.json`` vs ``PreToolUse.json``) — Codex tracks the
two trust states independently
(``[hooks.state.<...>:session_start:0:0]`` vs
``[hooks.state.<...>:pre_tool_use:0:0]``), so both must fire to
prove the per-handler trust evaluation is correct for both events
real PreToolUse turns rely on (review-4 P1)."""


_HOOK_SELFTEST_TARGET_EVENT_ENV = "MALA_CODEX_HOOK_SELFTEST_TARGET_EVENT"
"""Env var naming the event whose hook should ``decision=block`` /
``continue=false`` so the live probe's turn aborts. Mirrors
:data:`src.infra.hooks.codex.selftest_policy.SELFTEST_TARGET_EVENT_ENV`."""


_HOOK_IDENTITY_MODULES: tuple[str, ...] = (
    # The PreToolUse hook entry-point (top-level decide / main).
    "src.infra.hooks.codex_pre_tool_use",
    # Hook imports pure shell-tokenization / command-substitution /
    # redirection helpers from this module
    # (``codex_pre_tool_use.py:32-43``); a stale install with a
    # byte-identical entry-point but a divergent ``shell_parser.py``
    # would parse shell commands differently and gate writes via a
    # different heuristic.
    "src.infra.hooks.codex.shell_parser",
    # Hook imports the write-target extractors (``_extract_shell_write_
    # paths`` / ``_apply_patch_paths`` / ``_command_basename_str`` /
    # ``_find_git_subcommand_info``) from this module
    # (``codex_pre_tool_use.py:39-45``); a stale install with a
    # byte-identical entry-point but a divergent ``write_targets.py``
    # would extract a different set of write paths from the same shell
    # command or apply_patch payload and gate the lock-check on those
    # different paths.
    "src.infra.hooks.codex.write_targets",
    # Hook imports the lock-policy decision functions
    # (``_MSG_ENV_MISSING`` / ``_gate_write_targets``) from this module;
    # a stale install with a byte-identical entry-point but a divergent
    # ``lock_policy.py`` would make a different lock-ownership decision
    # on the same write target (different deny-reason wording, different
    # shell-expansion metachar handling, different cwd-resolution
    # behavior), silently changing every shell/file-edit gate outcome.
    "src.infra.hooks.codex.lock_policy",
    # Hook imports the destructive-command classification helpers
    # (``_msg_dangerous`` / ``_detect_dangerous_pattern_recursive``)
    # from this module after the T_B11 split; a stale install with a
    # byte-identical entry-point but a divergent
    # ``codex/dangerous_commands.py`` would silently no-op the
    # destructive-git / atomic-add-commit / fork-bomb / ``rm -rf /``
    # gates while emitting an identical selftest digest, so the
    # module bytes must be folded into the hook identity hash.
    # Distinct from ``src.infra.hooks.dangerous_commands`` (the shared
    # patterns + Claude-side hook) below — both modules are
    # safety-critical and tracked independently.
    "src.infra.hooks.codex.dangerous_commands",
    # Hook imports the selftest marker / live-probe helpers
    # (``_emit_selftest_marker_if_requested`` /
    # ``_handle_session_start_event`` / ``_handle_pre_tool_use_selftest``
    # / ``_selftest_mode_active`` / ``_selftest_target_event_matches``)
    # from this module after the T_B12 split. A stale install with a
    # byte-identical entry-point but a divergent
    # ``codex/selftest_policy.py`` could emit a marker with the wrong
    # version digest, skip the SessionStart abort, or fail to deny the
    # PreToolUse selftest probe — every one of those breaks the
    # provider's live-hook proof contract. The module also owns
    # :data:`SELFTEST_IDENTITY_MODULES`, so a divergence here changes
    # which modules' bytes the marker hashes over; folding the file's
    # bytes into the identity hash binds the tuple's authoritative
    # definition to the digest the provider verifies.
    "src.infra.hooks.codex.selftest_policy",
    # Hook imports BASH_TOOL_NAMES / DANGEROUS_PATTERNS /
    # DESTRUCTIVE_GIT_PATTERNS from this module
    # (``codex_pre_tool_use.py:32-36``); a stale install whose
    # ``codex_pre_tool_use.py`` matches ours but whose
    # ``dangerous_commands.py`` differs would invoke different
    # deny-path logic on shell commands.
    "src.infra.hooks.dangerous_commands",
    # ``dangerous_commands`` reads ``MALA_DISALLOWED_TOOLS`` from
    # this module (``dangerous_commands.py:12``); the disallowed-tool
    # list is part of the hook's enforcement surface.
    "src.infra.tool_config",
    # Hook lazily imports ``get_lock_holder`` from this module
    # (``codex_pre_tool_use.py:332``); the lock-key derivation /
    # canonicalization logic lives here and gates every file-edit /
    # shell-write decision.
    "src.infra.tools.locking",
    # ``locking`` reads the lock directory via this module
    # (``locking.py:14``); a different ``MALA_LOCK_DIR`` resolver
    # would change which on-disk locks the hook consults.
    "src.infra.tools.env",
)
"""Allowlist of mala-bundled modules whose source bytes form the
identity of the hook the on-PATH ``mala-codex-pre-tool-use`` would
execute. Hashing the entry-point alone is insufficient — the hook
imports enforcement data and lock-key logic from the modules above,
and a stale install with a matching entry-point but a different
``dangerous_commands.py`` (or any other module here) would still
take different deny / lock-ownership paths at runtime. A combined
hash over all modules below is the smallest set of bytes that
fingerprints the hook's safety-critical behavior."""


_MODULE_HASH_PROBE_CODE = _build_module_hash_probe_code(_HOOK_IDENTITY_MODULES)
"""Probe code emitted to the on-PATH hook's interpreter.

Inherits the hook's interpreter via the script's shebang so it sees
exactly the ``sys.path`` / site-packages the hook sees — a different
mala install on PATH resolves a different set of module files and a
different combined hash."""


_DEFAULT_PLUGIN_MARKETPLACE = "local"
"""Marketplace component of the Codex ``PluginId.as_key()`` value.

Hardcoded to the same literal as
:attr:`src.infra.clients.codex_plugin_installer.PLUGIN_MARKETPLACE` so
the two stay in lockstep without forcing the installer module to be
imported at module-load time (the lazy-import contract requires the
installer stay off the cold path until
:meth:`CodexAgentProvider.install_prerequisites` is invoked). A
regression test
(``tests/unit/infra/clients/test_codex_provider.py::test_default_marketplace_matches_installer_constant``)
pins the two literals together so a drift fails CI.

Codex's ``PluginId.as_key()`` returns
``<plugin_name>@<marketplace_name>``
(``codex-rs/plugin/src/plugin_id.rs:45``); using the literal ``local``
matches Codex's ``DEFAULT_PLUGIN_VERSION`` / local-marketplace
convention.
"""


def _compute_combined_module_hash(modules: tuple[str, ...]) -> str:
    """Compute the same combined SHA-256 the probe emits, but in-process.

    Used by :func:`_default_selftest_probe` to derive *our* expected
    hash; the on-PATH probe must match this value byte-for-byte.
    Mirrors the probe code's iteration order and length-prefix scheme
    exactly so both sides land on the same digest.
    """
    import hashlib as _hashlib
    import importlib.util as _importlib_util

    # Lazy import keeps the module-load circle from forming: the
    # provider imports this module, so the error types it defines are
    # only available after this module finishes loading.
    from src.infra.clients.codex_provider import (
        CodexHookNotActiveError,
        CodexHookNotActiveReason,
    )

    h = _hashlib.sha256()
    for name in modules:
        spec = _importlib_util.find_spec(name)
        if spec is None or spec.origin is None:
            raise CodexHookNotActiveError(
                f"Cannot locate this process's {name} module — repo invariant broken.",
                reason=CodexHookNotActiveReason.SCRIPT_MISSING,
            )
        path = Path(spec.origin)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise CodexHookNotActiveError(
                f"Cannot read embedded hook-dependency module at {path}: {exc}.",
                reason=CodexHookNotActiveReason.SCRIPT_MISSING,
            ) from exc
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return h.hexdigest()


_LIVE_DISPATCH_PROBE_PROMPT = (
    "Use the shell tool to run exactly: echo mala-selftest\n"
    "Do not use any other tool, do not read or edit files, and do not "
    "ask any questions. Issue the shell call immediately."
)
"""Prompt the live PreToolUse-driven probe sends to nudge a tool call.

Codex's PreToolUse hook only fires on real tool calls, which require
an LLM round trip. The prompt is short, targets a harmless ``echo``,
and explicitly forbids file edits so the model is unlikely to attempt
anything besides the requested shell call (which the bundled hook
will deny in selftest mode anyway). Cost: one model round trip
(~$0.001 on gpt-5.5) per ``install_prerequisites`` call, cached per
selftest run via ``_selftest_cache``."""


def _live_codex_hook_dispatch_probe(
    env_overlay: dict[str, str],
    repo_path: Path,
    expected_marker_hash: str,
) -> None:
    """Drive a real Codex turn that fires BOTH SessionStart AND PreToolUse.

    Plan AC-5 / E5 (review-4 P1 follow-up): each hook event has its
    own ``[hooks.state."<plugin_id>:...:<event>:0:0"]`` trust block
    in Codex; trusting ``...:session_start:0:0`` does NOT imply
    ``...:pre_tool_use:0:0`` is also trusted. A SessionStart-only
    probe lets the selftest pass while real ``apply_patch`` /
    ``bash`` PreToolUse turns silently skip the safety hook under
    ``danger-full-access``. This probe closes that gap by driving
    a real LLM-backed turn and asserting the bundled
    ``mala-codex-pre-tool-use`` script ran for *both* events:

      1. ``SessionStart`` fires before the LLM call (returns
         ``continue=true`` in selftest mode so the turn proceeds —
         hook still emits a per-event marker file).
      2. The LLM emits a ``shell`` tool call for ``echo
         mala-selftest`` (one round trip, ~$0.001 on gpt-5.5;
         bounded to one shot at the first tool call by the prompt
         and by interrupting the turn from the client side as soon
         as the marker appears).
      3. ``PreToolUse`` fires for the tool call, writes its own
         per-event marker, returns ``decision=block`` +
         ``continue=false`` so the call is denied (no shell command
         executed) and Codex aborts the turn.
      4. The provider polls the per-event marker dir for both
         ``SessionStart.json`` and ``PreToolUse.json`` and verifies
         each marker's ``version`` field matches
         ``expected_marker_hash``.

    Failure modes:

      * SDK / spawn / RPC failure → ``CODEX_BINARY_MISSING``.
      * Either marker absent within bounded timeout →
        ``HOOK_MARKER_MISSING`` (Codex's per-handler trust
        evaluation skipped the hook for that event).
      * Marker present but ``version`` mismatch → ``VERSION_MISMATCH``
        (Codex invoked the hook, but the on-PATH script's identity
        drifted from this process).

    Atomic marker writes + per-event marker files together protect
    against the review-4 P2 TOCTOU race where a partial read of
    a single shared marker file would parse as malformed JSON.

    Cost: one bounded LLM turn per ``install_prerequisites`` call.
    Cached via ``_selftest_cache`` so a long-running orchestrator
    pays at most once per startup.
    """
    import asyncio
    import json as _json
    import tempfile
    import time as _time

    from src.infra.clients.codex_provider import (
        CodexHookNotActiveError,
        CodexHookNotActiveReason,
    )

    expected_event_files = ("SessionStart.json", "PreToolUse.json")

    async def _drive_turn(marker_dir: Path) -> dict[str, str]:
        codex_app_server = import_codex_app_server()
        AppServerConfig = codex_app_server.AppServerConfig
        AsyncCodex = codex_app_server.AsyncCodex
        TextInput = codex_app_server.TextInput

        codex_bin = resolve_codex_bin_for_app_server()
        merged_env = dict(os.environ)
        merged_env.update(env_overlay)
        # Wire the marker dir + target event so the bundled hook can
        # write per-event markers (review-4 P1) and so the PreToolUse
        # firing returns decision=block to deny the requested shell
        # call. Codex inherits this env into the hook subprocess via
        # ``AppServerConfig.env``.
        merged_env[_HOOK_SELFTEST_MARKER_DIR_ENV] = str(marker_dir)
        merged_env[_HOOK_SELFTEST_TARGET_EVENT_ENV] = "PreToolUse"
        config = AppServerConfig(
            codex_bin=codex_bin,
            cwd=str(repo_path),
            env=merged_env,
        )
        sandbox_value = cast("Any", "read-only")
        approval_policy_value = cast("Any", "never")

        async with AsyncCodex(config=config) as codex:
            thread = await start_thread_compat(
                codex,
                {
                    "model": DEFAULT_CODEX_MODEL,
                    "sandbox": "read-only",
                    "approvalPolicy": "never",
                    "cwd": str(repo_path),
                },
                typed_kwargs={
                    "model": DEFAULT_CODEX_MODEL,
                    "sandbox": sandbox_value,
                    "approval_policy": approval_policy_value,
                    "cwd": str(repo_path),
                },
            )
            turn = await thread.turn(TextInput(_LIVE_DISPATCH_PROBE_PROMPT))
            turn_id = getattr(turn, "id", None)

            async def _interrupt_turn() -> None:
                await turn.interrupt()

            deadline = _time.monotonic() + _HOOK_PROBE_TIMEOUT_SECONDS
            collected: dict[str, str] = {}
            interrupted = False
            while _time.monotonic() < deadline:
                for filename in expected_event_files:
                    marker_path = marker_dir / filename
                    if filename in collected or not marker_path.is_file():
                        continue
                    try:
                        text = marker_path.read_text(encoding="utf-8")
                    except OSError:
                        continue
                    if not text.strip():
                        continue
                    try:
                        _json.loads(text)
                    except _json.JSONDecodeError:
                        # Mid-write — retry next tick. Atomic writes
                        # in the hook should make this rare; the
                        # retry is a belt-and-suspenders bound on
                        # the review-4 P2 TOCTOU race.
                        continue
                    collected[filename] = text
                if len(collected) == len(expected_event_files):
                    break
                # As soon as PreToolUse has fired we no longer need
                # the LLM-backed turn to keep running; interrupt it
                # client-side to bound LLM cost.
                if (
                    not interrupted
                    and "PreToolUse.json" in collected
                    and turn_id is not None
                ):
                    try:
                        await _interrupt_turn()
                    except Exception:
                        # Best-effort interrupt: any failure here just
                        # leaves the turn running until the bound
                        # timeout in ``_drive_with_timeout``. Selftest
                        # success doesn't depend on the interrupt
                        # succeeding — it depends on the per-event
                        # markers being present, which they already are.
                        pass
                    interrupted = True
                await asyncio.sleep(0.1)
            if len(collected) < len(expected_event_files):
                missing = [
                    filename
                    for filename in expected_event_files
                    if filename not in collected
                ]
                raise CodexHookNotActiveError(
                    "Codex did not fire all bundled hook events within "
                    f"{_HOOK_PROBE_TIMEOUT_SECONDS:.0f}s during the live "
                    f"selftest probe; missing event markers: "
                    f"{', '.join(missing)}. The plugin tree is on disk "
                    "and plugin/list reports it loaded, but the "
                    "per-handler [hooks.state.<key>] trust evaluation "
                    "skipped at least one event. Real PreToolUse turns "
                    "would skip the safety hook for the same reason.",
                    reason=CodexHookNotActiveReason.HOOK_MARKER_MISSING,
                )
            return collected

    async def _drive_with_timeout(marker_dir: Path) -> dict[str, str]:
        return await asyncio.wait_for(
            _drive_turn(marker_dir),
            timeout=_HOOK_PROBE_TIMEOUT_SECONDS * 3.0,
        )

    with tempfile.TemporaryDirectory(prefix="mala-codex-dispatch-selftest-") as tmpdir:
        marker_dir = Path(tmpdir)
        try:
            collected = _run_coro_sync(lambda: _drive_with_timeout(marker_dir))
        except TimeoutError as exc:
            raise CodexHookNotActiveError(
                f"Codex hook-dispatch selftest did not complete within "
                f"{_HOOK_PROBE_TIMEOUT_SECONDS * 3.0:.0f}s. A hung Codex "
                "thread or LLM stall would block real turns the same "
                "way; failing closed.",
                reason=CodexHookNotActiveReason.CODEX_BINARY_MISSING,
            ) from exc
        except CodexHookNotActiveError:
            raise
        except Exception as exc:
            raise CodexHookNotActiveError(
                "Codex app-server failed during the live hook-dispatch "
                f"selftest probe: {exc!r}. Codex's hook dispatch could "
                "not be exercised end-to-end even though the SDK / "
                "runtime / auth / plugin-list probes succeeded.",
                reason=CodexHookNotActiveReason.CODEX_BINARY_MISSING,
            ) from exc

    for filename, marker_text in collected.items():
        try:
            marker_data = _json.loads(marker_text)
        except _json.JSONDecodeError as exc:
            raise CodexHookNotActiveError(
                f"Hook wrote malformed marker for {filename!r} during "
                f"the live probe: {exc}. Text: {marker_text[:256]!r}.",
                reason=CodexHookNotActiveReason.HOOK_MARKER_MISSING,
            ) from exc
        marker_version = (
            marker_data.get("version") if isinstance(marker_data, dict) else None
        )
        if marker_version != expected_marker_hash:
            raise CodexHookNotActiveError(
                f"Hook fired for {filename!r} through Codex but emitted "
                f"a marker version {marker_version!r} that does not "
                f"match this process's expected hash "
                f"{expected_marker_hash!r}. The on-PATH hook script's "
                "identity diverged from this mala install's modules.",
                reason=CodexHookNotActiveReason.VERSION_MISMATCH,
            )


def _live_codex_plugin_list_probe(
    env_overlay: dict[str, str],
    repo_path: Path,
    *,
    marketplace: str = _DEFAULT_PLUGIN_MARKETPLACE,
) -> None:
    """Spawn the real Codex app-server and verify the bundled plugin loads.

    Plan AC-5 / E5: the prior structural / identity / direct-spawn
    checks all bypass Codex's own plugin loader — a broken
    ``[hooks.state.<key>]`` block, a feature-flag toggle Codex is
    re-evaluating differently, an unexpected schema change in
    ``plugin.json`` parsing, or any other Codex-side loader mismatch
    would leave the on-disk tree looking right while real turns silently
    skip the safety hook under ``danger-full-access``. This probe
    closes that gap by issuing an actual ``plugin/list`` JSON-RPC
    request to a live ``codex app-server`` subprocess and asserting
    Codex itself reports the bundled ``mala-safety`` plugin as
    ``installed=True`` and ``enabled=True``.

    The ``plugin/list`` response goes through:

      * Plugin discovery (``PluginsManager.list_marketplaces_for_config``
        walks the ``$CODEX_HOME/plugins/cache`` tree the installer
        wrote into).
      * The ``Feature::Plugins`` and workspace-codex-plugins gates
        (early-return ``empty_response()`` paths in
        ``plugin_list_response`` short-circuit when either is off).
      * The ``[plugins.<id>] enabled`` config flag (surfaced as
        ``PluginSummary.enabled``).

    Failure modes mapped to existing
    :class:`CodexHookNotActiveReason` values:

      * SDK fails to spawn / initialize → ``CODEX_BINARY_MISSING``
        (the SDK + binary checks earlier in ``install_prerequisites``
        already run; this branch catches a regression where they pass
        but the spawned app-server still cannot start).
      * ``plugin/list`` returns no marketplaces / our plugin missing →
        ``PLUGIN_DISABLED`` (Codex did not discover the plugin tree).
      * Plugin found but ``installed=False`` → ``PLUGIN_DISABLED``
        (Codex sees the plugin tree but considers the install
        incomplete / broken).
      * Plugin found but ``enabled=False`` → ``PLUGIN_DISABLED``
        (Codex's user config shows the plugin disabled even though
        we wrote ``enabled = true``; the trusted_hash + state writes
        in ``_write_codex_plugin_config`` rely on the same TOML
        splice and would surface here).

    Tests inject a fake by monkeypatching
    ``_live_codex_plugin_list_probe`` to a noop or a controlled
    failure-driver — spawning a real app-server in unit tests would
    require a live Codex binary which the in-repo fakes do not
    provide.
    """
    import asyncio

    from src.infra.clients.codex_plugin_installer import _plugin_id
    from src.infra.clients.codex_provider import (
        CodexHookNotActiveError,
        CodexHookNotActiveReason,
    )

    expected_plugin_id = _plugin_id(marketplace)

    async def _query() -> tuple[list[object], list[object]]:
        # Lazy import keeps the lazy-import contract on the success path of
        # `install_prerequisites`: the SDK is only imported when the live
        # probe actually runs (which only happens after the upstream
        # SDK importability + binary + auth checks already passed).
        codex_app_server = import_codex_app_server()
        codex_v2_all = import_codex_generated_v2_all()
        AppServerConfig = codex_app_server.AppServerConfig
        AsyncCodex = codex_app_server.AsyncCodex
        PluginListResponse = codex_v2_all.PluginListResponse

        codex_bin = resolve_codex_bin_for_app_server()
        merged_env = dict(os.environ)
        merged_env.update(env_overlay)
        config = AppServerConfig(
            codex_bin=codex_bin,
            cwd=str(repo_path),
            env=merged_env,
        )
        async with AsyncCodex(config=config) as codex:
            # The SDK's typed wrappers omit ``plugin/list``, so go through
            # the underlying client's generic ``request`` method. Single
            # ``cwds`` entry pinned to the orchestrated repo so Codex
            # walks the right repo-marketplace search root.
            # Use the underlying client's generic ``request`` since the
            # pinned SDK version omits a typed ``plugin/list`` wrapper.
            response = await codex._client.request(
                "plugin/list",
                {"cwds": [str(repo_path)]},
                response_model=PluginListResponse,
            )
        marketplaces: list[object] = list(response.marketplaces or [])
        load_errors: list[object] = list(response.marketplace_load_errors or [])
        return marketplaces, load_errors

    async def _query_with_timeout() -> tuple[list[object], list[object]]:
        return await asyncio.wait_for(_query(), timeout=_HOOK_PROBE_TIMEOUT_SECONDS)

    try:
        marketplaces, load_errors = _run_coro_sync(_query_with_timeout)
    except TimeoutError as exc:
        raise CodexHookNotActiveError(
            f"Codex app-server did not respond to plugin/list within "
            f"{_HOOK_PROBE_TIMEOUT_SECONDS:.0f}s during the live selftest "
            "probe. A hung app-server / RPC stall would block the "
            "orchestrator indefinitely; failing closed instead.",
            reason=CodexHookNotActiveReason.CODEX_BINARY_MISSING,
        ) from exc
    except CodexHookNotActiveError:
        raise
    except Exception as exc:
        # Catch-all for SDK / RPC / spawn failures: any exception out of
        # the async ``plugin/list`` request means Codex's app-server could not
        # respond, which is a fail-closed signal regardless of the
        # underlying cause.
        raise CodexHookNotActiveError(
            f"Codex app-server failed to respond to plugin/list during the "
            f"live selftest probe: {exc!r}. Codex's plugin loader could not "
            "be reached even though the SDK / runtime / auth probes "
            "succeeded; the safety hook would not fire on real turns.",
            reason=CodexHookNotActiveReason.CODEX_BINARY_MISSING,
        ) from exc

    if load_errors:
        # Marketplace load errors mean Codex saw the plugin tree but
        # rejected its layout. The on-disk install looks right but
        # Codex's loader hit a parse/permission/missing-file failure.
        rendered = ", ".join(repr(err) for err in load_errors[:3])
        raise CodexHookNotActiveError(
            "Codex's plugin/list returned marketplace_load_errors during "
            f"the live selftest probe: {rendered}. The on-disk plugin tree "
            "is corrupt from Codex's loader's perspective; rerun mala to "
            "reinstall.",
            reason=CodexHookNotActiveReason.PLUGIN_DISABLED,
        )

    summaries: list[object] = []
    for marketplace_entry in marketplaces:
        plugins_attr = getattr(marketplace_entry, "plugins", None)
        if plugins_attr:
            summaries.extend(plugins_attr)
    matching = [
        summary
        for summary in summaries
        if str(getattr(summary, "id", "")) == expected_plugin_id
    ]
    if not matching:
        raise CodexHookNotActiveError(
            f"Codex's plugin/list does not include the bundled "
            f"{expected_plugin_id!r} plugin during the live selftest "
            f"probe (saw {len(summaries)} plugin summaries). Codex's "
            "plugin discovery does not see our install — the install "
            "target diverges from Codex's PluginStore search path, or "
            "the plugin manifest is malformed at Codex's layer.",
            reason=CodexHookNotActiveReason.PLUGIN_DISABLED,
        )
    matched = matching[0]
    if not getattr(matched, "installed", False):
        raise CodexHookNotActiveError(
            f"Codex's plugin/list reports {expected_plugin_id!r} as "
            f"NOT installed during the live selftest probe (got "
            f"installed={getattr(matched, 'installed', None)!r}). "
            "Codex sees the marketplace entry but considers the install "
            "broken; rerun mala to reinstall.",
            reason=CodexHookNotActiveReason.PLUGIN_DISABLED,
        )
    if not getattr(matched, "enabled", False):
        raise CodexHookNotActiveError(
            f"Codex's plugin/list reports {expected_plugin_id!r} as "
            f"NOT enabled during the live selftest probe (got "
            f"enabled={getattr(matched, 'enabled', None)!r}). The "
            f'`[plugins."{expected_plugin_id}"] enabled = true` write '
            "did not take effect, or a higher-priority config layer "
            "disabled the plugin.",
            reason=CodexHookNotActiveReason.PLUGIN_DISABLED,
        )


def _default_selftest_probe(
    repo_path: Path,
    env_overlay: dict[str, str],
    expected_hash: str,
) -> None:
    """Default selftest probe — structural + module-identity verification.

    Fail-closed verification of the on-disk plugin tree at the install
    target Codex reads, *plus* a cryptographic identity check of the
    Python module the on-PATH hook script actually loads. Catches every
    install-time and PATH-resolution failure mode the up-front
    structural checks (SDK importable, codex runtime resolvable, hook
    script on PATH) leave open:

      * Plugin manifest at
        ``$CODEX_HOME/plugins/cache/<marketplace>/<plugin>/<version>/.codex-plugin/plugin.json``
        exists, parses as JSON, and points at ``./.codex-plugin/hooks.json``.
        Failure → :class:`CodexHookNotActiveError(PLUGIN_DISABLED)`.
      * Installed hooks.json exists, parses, and declares a PreToolUse
        command handler whose ``command`` is ``mala-codex-pre-tool-use``.
        Failure → :class:`CodexHookNotActiveError(HOOK_MARKER_MISSING)`.
      * Hook-module identity: the combined SHA-256 over every
        safety-critical mala-bundled module the hook depends on
        (entry-point ``src.infra.hooks.codex_pre_tool_use`` plus its
        first-party imports — see :data:`_HOOK_IDENTITY_MODULES`)
        the on-PATH hook binary's interpreter resolves must equal the
        combined SHA-256 of *our* embedded modules. A divergence means
        the on-PATH binary is from a different mala install whose
        Python resolves a different set of module files — Codex would
        invoke that other version of the hook + dependencies, and its
        deny-path / dangerous-command / lock-ownership logic could
        differ from ours. Surfaces
        :class:`CodexHookNotActiveError(VERSION_MISMATCH)`. A
        crashing / hung / missing-shebang / unimportable-dependency
        probe surfaces
        :class:`CodexHookNotActiveError(HOOK_MARKER_MISSING)`.

    The combined-hash check closes the gap a benign-input behavioral
    check left open: a stale install whose entry-point bytes match
    ours but whose ``dangerous_commands.py`` (or
    ``tools/locking.py``, ``tool_config.py``, ``tools/env.py``)
    differs would invoke different deny / lock-ownership logic at
    runtime. Hashing the entry-point alone misses that case; hashing
    every safety-critical dependency catches *any* logic difference,
    including ones not exercised by a single sentinel input.

      * Live-hook invocation: after the structural + module-identity
        checks pass, the on-PATH hook script is spawned with a synthetic
        Codex-shape PreToolUse payload on stdin and
        :data:`SELFTEST_MARKER_ENV` set to a temp file path. The hook
        emits ``{"mala_codex_hook": "loaded", "version": "<sha256>"}``
        before processing the payload (see
        :func:`src.infra.hooks.codex.selftest_policy._emit_selftest_marker_if_requested`).
        The probe asserts the marker file is present, parses, and its
        ``version`` field equals our combined identity hash. Failure
        modes:

          * Script crashes / times out / writes no marker →
            :class:`CodexHookNotActiveError(HOOK_MARKER_MISSING)`.
          * Marker present but its ``version`` does not match →
            :class:`CodexHookNotActiveError(VERSION_MISMATCH)`.

        This closes the gap pure module-identity hashing left open
        (plan AC-5 / E5): an importable module whose ``main()`` raises
        before emitting the marker, a packaging shape that prevents
        the script from spawning, or an interpreter where
        :func:`os.environ.get` / file I/O is broken — none of which
        the static hash probe would catch — all fail closed here.

    Tests inject custom probes to drive each
    :class:`CodexHookNotActiveReason` directly, including the runtime
    reasons the live-hook step surfaces.
    """
    del expected_hash  # superseded by the combined module-source hash below
    import json as _json
    import shutil as _shutil
    import subprocess as _subprocess
    import tempfile

    from src.infra.clients.codex_plugin_installer import (
        PLUGIN_DIRNAME,
        _HOOK_EVENTS,
        _resolve_codex_home,
        plugin_root_dir,
    )
    from src.infra.clients.codex_provider import (
        CodexHookNotActiveError,
        CodexHookNotActiveReason,
    )

    codex_home_str = env_overlay.get("CODEX_HOME")
    codex_home = Path(codex_home_str) if codex_home_str else _resolve_codex_home()
    # Plugin tree must be under Codex's PluginStore cache root
    # (``codex-rs/core-plugins/src/store.rs::PluginStore``); the manifest
    # lives in ``.codex-plugin/`` inside that cache root.
    plugin_dir = plugin_root_dir(codex_home) / PLUGIN_DIRNAME
    manifest_path = plugin_dir / "plugin.json"
    hooks_json_path = plugin_dir / "hooks.json"

    # 1. plugin.json exists, parses, and links to the right hooks file.
    try:
        manifest_text = manifest_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CodexHookNotActiveError(
            f"Codex plugin manifest missing at {manifest_path}: {exc}",
            reason=CodexHookNotActiveReason.PLUGIN_DISABLED,
        ) from exc
    try:
        manifest = _json.loads(manifest_text)
    except _json.JSONDecodeError as exc:
        raise CodexHookNotActiveError(
            f"Codex plugin manifest at {manifest_path} is not valid JSON: {exc}",
            reason=CodexHookNotActiveReason.PLUGIN_DISABLED,
        ) from exc
    hooks_field = manifest.get("hooks") if isinstance(manifest, dict) else None
    if hooks_field != "./.codex-plugin/hooks.json":
        raise CodexHookNotActiveError(
            f"Codex plugin manifest at {manifest_path} does not point at the "
            f"bundled hooks.json (got {hooks_field!r}, expected "
            "'./.codex-plugin/hooks.json'). The on-disk plugin tree is corrupt; "
            "rerun mala to reinstall.",
            reason=CodexHookNotActiveReason.PLUGIN_DISABLED,
        )

    # 2. hooks.json exists, parses, and declares the expected command hook.
    try:
        hooks_text = hooks_json_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CodexHookNotActiveError(
            f"Codex plugin hooks.json missing at {hooks_json_path}: {exc}",
            reason=CodexHookNotActiveReason.HOOK_MARKER_MISSING,
        ) from exc
    try:
        hooks_payload = _json.loads(hooks_text)
    except _json.JSONDecodeError as exc:
        raise CodexHookNotActiveError(
            f"Codex plugin hooks.json at {hooks_json_path} is not valid JSON: {exc}",
            reason=CodexHookNotActiveReason.HOOK_MARKER_MISSING,
        ) from exc
    for codex_event_name, _ in _HOOK_EVENTS:
        if not _hooks_json_declares_event_command(
            hooks_payload,
            event=codex_event_name,
            expected_command=_HOOK_COMMAND_NAME,
        ):
            raise CodexHookNotActiveError(
                f"Codex plugin hooks.json at {hooks_json_path} does not "
                f"declare a {codex_event_name} command handler for "
                f"{_HOOK_COMMAND_NAME!r}; the safety hook + selftest "
                "dispatch probe would not fire.",
                reason=CodexHookNotActiveReason.HOOK_MARKER_MISSING,
            )

    # 3. Hook-module identity check — the SHA-256 of the
    # ``src.infra.hooks.codex_pre_tool_use`` source the on-PATH hook's
    # interpreter resolves must equal the SHA-256 of our embedded
    # module's source. Catches stale-install-on-PATH cases that
    # ``shutil.which`` (file presence) and a single benign-input
    # behavioral check (same allow-JSON for the safe path) cannot
    # distinguish.
    hook_path = _shutil.which(_HOOK_COMMAND_NAME)
    if hook_path is None:
        raise CodexHookNotActiveError(
            f"{_HOOK_COMMAND_NAME!r} not on PATH at probe time.",
            reason=CodexHookNotActiveReason.SCRIPT_MISSING,
        )

    # Compute our expected combined hash over every safety-critical
    # module the hook depends on (entry-point + dangerous_commands +
    # tool_config + locking + env). Hashing the entry-point alone
    # would miss the case where a stale install has the same
    # ``codex_pre_tool_use.py`` but different deny patterns in
    # ``dangerous_commands.py`` — Codex would still execute the wrong
    # deny logic on shell traffic.
    our_hash = _compute_combined_module_hash(_HOOK_IDENTITY_MODULES)

    # Resolve the interpreter the on-PATH hook uses by parsing its
    # shebang line. The hook script is a ``[project.scripts]``-style
    # console wrapper whose first line embeds the absolute path to
    # the venv's Python. That interpreter's site-packages determine
    # which ``src.infra.hooks.codex_pre_tool_use`` (and its imports)
    # Codex actually loads at runtime.
    hook_interpreter = _resolve_hook_interpreter(hook_path)

    try:
        probe_result = _subprocess.run(
            [hook_interpreter, "-c", _MODULE_HASH_PROBE_CODE],
            capture_output=True,
            text=True,
            timeout=_HOOK_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except _subprocess.TimeoutExpired as exc:
        raise CodexHookNotActiveError(
            f"Hook interpreter {hook_interpreter!r} did not respond "
            f"within {_HOOK_PROBE_TIMEOUT_SECONDS:.0f}s during the "
            "module-identity probe.",
            reason=CodexHookNotActiveReason.HOOK_MARKER_MISSING,
        ) from exc
    except OSError as exc:
        raise CodexHookNotActiveError(
            f"Cannot spawn hook interpreter {hook_interpreter!r}: {exc}.",
            reason=CodexHookNotActiveReason.HOOK_MARKER_MISSING,
        ) from exc

    # The probe emits ``NOMODULE:<name>`` and ``sys.exit(2)`` when
    # ``find_spec`` returns None for one of the safety-critical
    # modules. Surface that specific diagnostic BEFORE the generic
    # non-zero-exit branch so the user sees the offending module
    # name rather than a bare ``exited 2; stderr: ''`` message
    # (regression: review-20 P2). All other non-zero exits fall
    # through to the generic branch.
    their_hash = probe_result.stdout.strip()
    if their_hash.startswith("NOMODULE"):
        raise CodexHookNotActiveError(
            f"Hook interpreter {hook_interpreter!r} cannot import "
            f"a hook-dependency module ({their_hash}) — Codex would "
            "crash when invoking the hook.",
            reason=CodexHookNotActiveReason.HOOK_MARKER_MISSING,
        )

    if probe_result.returncode != 0:
        raise CodexHookNotActiveError(
            f"Hook interpreter {hook_interpreter!r} exited "
            f"{probe_result.returncode} during module-identity probe; "
            f"stderr: {probe_result.stderr[:512]!r}.",
            reason=CodexHookNotActiveReason.HOOK_MARKER_MISSING,
        )
    if their_hash != our_hash:
        raise CodexHookNotActiveError(
            f"On-PATH {_HOOK_COMMAND_NAME!r} ({hook_path}) resolves a "
            "different set of hook-identity modules (combined "
            f"sha256={their_hash!r}) than this process's embedded "
            f"modules (combined sha256={our_hash!r}, modules: "
            f"{_HOOK_IDENTITY_MODULES}). The hook script on PATH is "
            "from a different mala install or a different version; "
            "remove the stale install or run mala from the "
            "environment that owns the hook script.",
            reason=CodexHookNotActiveReason.VERSION_MISMATCH,
        )

    # 4. Live-hook invocation — drive the on-PATH ``mala-codex-pre-tool-use``
    # entry-point end-to-end with a Codex-shape PreToolUse payload and a
    # selftest marker env var. The hook emits a JSON marker before
    # processing the payload (see
    # ``src.infra.hooks.codex.selftest_policy._emit_selftest_marker_if_requested``);
    # the marker proves ``main()`` actually executed and the hook's own
    # identity-hash computation produced the same digest the
    # module-identity probe above produced. Plan AC-5 / E5: the prior
    # structural + identity checks proved the BYTES on PATH are correct,
    # but did not prove the script can be invoked end-to-end (e.g. an
    # interpreter where ``main()`` raises before emitting any output, a
    # shebang the kernel rejects at exec time, or a packaging shape
    # that drops the entry-point). Without this step the selftest could
    # cache success while real Codex turns fail at the first PreToolUse.
    selftest_payload = _json.dumps({"tool_name": "noop", "tool_input": {}})
    with tempfile.TemporaryDirectory(prefix="mala-codex-hook-selftest-") as marker_dir:
        marker_path = Path(marker_dir) / "marker.json"
        # Inherit the current process env so the hook's interpreter can
        # find its site-packages, then overlay the marker env var. The
        # ``noop`` tool name routes through the hook's
        # neither-shell-nor-file-edit branch so the lock-env vars
        # (``MALA_AGENT_ID`` etc.) are not consulted; we do not need to
        # scrub them and a real run's env is the most representative
        # test surface.
        live_env = dict(os.environ)
        live_env[_HOOK_SELFTEST_MARKER_ENV] = str(marker_path)
        try:
            live_result = _subprocess.run(
                [hook_path],
                input=selftest_payload,
                capture_output=True,
                text=True,
                timeout=_HOOK_PROBE_TIMEOUT_SECONDS,
                check=False,
                env=live_env,
            )
        except _subprocess.TimeoutExpired as exc:
            raise CodexHookNotActiveError(
                f"On-PATH {_HOOK_COMMAND_NAME!r} ({hook_path}) did not "
                f"respond within {_HOOK_PROBE_TIMEOUT_SECONDS:.0f}s "
                "during the live selftest invocation; Codex would "
                "block on the same timeout at PreToolUse.",
                reason=CodexHookNotActiveReason.HOOK_MARKER_MISSING,
            ) from exc
        except OSError as exc:
            raise CodexHookNotActiveError(
                f"Cannot spawn on-PATH {_HOOK_COMMAND_NAME!r} "
                f"({hook_path}) during the live selftest invocation: "
                f"{exc}.",
                reason=CodexHookNotActiveReason.HOOK_MARKER_MISSING,
            ) from exc

        if live_result.returncode != 0:
            raise CodexHookNotActiveError(
                f"On-PATH {_HOOK_COMMAND_NAME!r} ({hook_path}) exited "
                f"{live_result.returncode} during the live selftest "
                f"invocation; stderr: {live_result.stderr[:512]!r}.",
                reason=CodexHookNotActiveReason.HOOK_MARKER_MISSING,
            )

        try:
            marker_text = marker_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CodexHookNotActiveError(
                f"On-PATH {_HOOK_COMMAND_NAME!r} ({hook_path}) did not "
                f"write the live selftest marker at {marker_path}: "
                f"{exc}. The hook entry-point did not run end-to-end; "
                "Codex would not be able to invoke the hook on PreToolUse.",
                reason=CodexHookNotActiveReason.HOOK_MARKER_MISSING,
            ) from exc

        try:
            marker_data = _json.loads(marker_text)
        except _json.JSONDecodeError as exc:
            raise CodexHookNotActiveError(
                f"On-PATH {_HOOK_COMMAND_NAME!r} ({hook_path}) wrote a "
                f"malformed live selftest marker: {exc}. Marker text: "
                f"{marker_text[:512]!r}.",
                reason=CodexHookNotActiveReason.HOOK_MARKER_MISSING,
            ) from exc

        marker_version = (
            marker_data.get("version") if isinstance(marker_data, dict) else None
        )
        if marker_version != our_hash:
            raise CodexHookNotActiveError(
                f"On-PATH {_HOOK_COMMAND_NAME!r} ({hook_path}) emitted a "
                f"live selftest marker whose version {marker_version!r} "
                f"does not match this process's expected hash "
                f"{our_hash!r}. The hook ran but computed a divergent "
                "identity — its embedded module list or hash logic "
                "differs from this mala install.",
                reason=CodexHookNotActiveReason.VERSION_MISMATCH,
            )

    # 5. Live Codex plugin/list probe — verify Codex's own plugin loader
    # (``Feature::Plugins`` gate, workspace-plugins gate, plugin
    # discovery, ``[plugins."<id>"] enabled``) considers the bundled
    # plugin loaded. Cheap RPC; no LLM cost.
    _live_codex_plugin_list_probe(env_overlay, repo_path)

    # 6. Live Codex hook-dispatch probe — drive a one-shot Codex turn
    # that fires the bundled ``SessionStart`` hook through Codex's full
    # dispatch pipeline (``Feature::PluginHooks`` /
    # ``Feature::CodexHooks`` gates, per-handler
    # ``[hooks.state."<key>"]`` trust evaluation,
    # ``hook_runtime::run_pending_session_start_hooks``). The hook
    # writes the same selftest marker the direct-spawn step writes,
    # then returns ``continue=false`` so the turn aborts BEFORE any
    # LLM call. Plan AC-5 / E5 (review-2 follow-up): without this step
    # a wrong ``hook_state_key`` or stale ``trusted_hash`` could pass
    # plugin/list while Codex skips the safety hook on real PreToolUse
    # turns under ``danger-full-access``.
    _live_codex_hook_dispatch_probe(env_overlay, repo_path, our_hash)


def _resolve_hook_interpreter(hook_path: str) -> str:
    """Parse the on-PATH hook script's shebang to find its interpreter.

    Reads the first line of the executable. ``[project.scripts]``
    console wrappers (the shape pip / uv emit) start with
    ``#!<absolute python path>`` so the kernel knows which interpreter
    to spawn; that path is the source-of-truth for which Python
    resolves ``src.infra.hooks.codex_pre_tool_use`` at runtime.

    Raises :class:`CodexHookNotActiveError(HOOK_MARKER_MISSING)` when
    the file is unreadable or has no shebang (a native binary, a
    shell wrapper without ``#!``, etc. — none of these are valid
    hook-script shapes our packaging produces).
    """
    from src.infra.clients.codex_provider import (
        CodexHookNotActiveError,
        CodexHookNotActiveReason,
    )

    try:
        with open(hook_path, "rb") as fp:
            first_line = fp.readline()
    except OSError as exc:
        raise CodexHookNotActiveError(
            f"Cannot read on-PATH {_HOOK_COMMAND_NAME!r} at {hook_path}: {exc}.",
            reason=CodexHookNotActiveReason.HOOK_MARKER_MISSING,
        ) from exc
    if not first_line.startswith(b"#!"):
        raise CodexHookNotActiveError(
            f"On-PATH {_HOOK_COMMAND_NAME!r} ({hook_path}) has no "
            "shebang; cannot identify the Python interpreter that "
            "would resolve the hook module.",
            reason=CodexHookNotActiveReason.HOOK_MARKER_MISSING,
        )
    shebang = first_line[2:].decode("utf-8", errors="replace").strip()
    if not shebang:
        raise CodexHookNotActiveError(
            f"On-PATH {_HOOK_COMMAND_NAME!r} ({hook_path}) has an empty shebang line.",
            reason=CodexHookNotActiveReason.HOOK_MARKER_MISSING,
        )
    # Honour the typical ``#!/usr/bin/env python3`` shape: take the
    # first whitespace-delimited token as the interpreter, drop trailing
    # ``-m`` / ``-S`` / etc. flags by ignoring the rest.
    interpreter = shebang.split()[0]
    if interpreter.endswith("/env") or interpreter == "env":
        # ``#!/usr/bin/env python3 [-S]`` — second token is the program name.
        rest = shebang.split()[1:]
        if not rest:
            raise CodexHookNotActiveError(
                f"On-PATH {_HOOK_COMMAND_NAME!r} ({hook_path}) uses "
                "``env`` shebang without a program name.",
                reason=CodexHookNotActiveReason.HOOK_MARKER_MISSING,
            )
        program = rest[0]
        # Resolve via PATH so the same lookup the kernel/env uses applies.
        resolved = shutil.which(program)
        if resolved is None:
            raise CodexHookNotActiveError(
                f"``env``-style shebang in {hook_path} references "
                f"{program!r} which is not on PATH.",
                reason=CodexHookNotActiveReason.HOOK_MARKER_MISSING,
            )
        return resolved
    return interpreter


def _hooks_json_declares_event_command(
    payload: object,
    *,
    event: str,
    expected_command: str,
) -> bool:
    """Return True iff ``payload`` declares ``expected_command`` for ``event``."""
    payload_dict = (
        cast("dict[str, object]", payload) if isinstance(payload, dict) else None
    )
    if payload_dict is None:
        return False
    hooks_obj = payload_dict.get("hooks")
    hooks_dict = (
        cast("dict[str, object]", hooks_obj) if isinstance(hooks_obj, dict) else None
    )
    if hooks_dict is None:
        return False
    matchers = hooks_dict.get(event)
    if not isinstance(matchers, list):
        return False
    for group_obj in matchers:
        if not isinstance(group_obj, dict):
            continue
        group = cast("dict[str, object]", group_obj)
        handlers_obj = group.get("hooks")
        if not isinstance(handlers_obj, list):
            continue
        for handler_obj in handlers_obj:
            if not isinstance(handler_obj, dict):
                continue
            handler = cast("dict[str, object]", handler_obj)
            if (
                handler.get("type") == "command"
                and handler.get("command") == expected_command
            ):
                return True
    return False


def _hooks_json_declares_pre_tool_use_command(
    payload: object, expected_command: str
) -> bool:
    """Return True iff ``payload`` declares ``expected_command`` for PreToolUse.

    Retained for parity with existing tests; the live selftest probe
    additionally requires SessionStart to be declared (so the
    SessionStart-driven dispatch probe in step 6 can fire). Use
    :func:`_hooks_json_declares_event_command` directly when checking
    SessionStart.
    """
    return _hooks_json_declares_event_command(
        payload, event="PreToolUse", expected_command=expected_command
    )

"""Codex :class:`AgentProvider` (Phase C T010, Phase F T013, Phase E T015).

Wires :class:`CodexAgentProvider` to the real :class:`CodexClient` /
:class:`CodexRuntime` / :class:`CodexRuntimeBuilder` shipped in Phase C
(T010), to the real :class:`CodexEvidenceProvider` shipped in Phase F
(T013), and to the bundled ``mala-safety`` plugin install + selftest
shipped in Phase E (T015). ``install_prerequisites`` now performs the
full safety-critical fail-closed gate — SDK / binary checks, plugin
install, trusted-hash auto-write, hook self-test — paralleling the
Amp provider's posture.

Lazy-import contract (plan ``L733``): importing this module does NOT
transitively import ``codex_app_server``. The SDK is referenced only
inside runtime/probe paths such as :class:`CodexClient.__aenter__`,
:meth:`CodexClient.query`, and the live Codex selftests;
:class:`CodexClient` itself is imported lazily by
:meth:`_CodexClientFactory.create` so module-load remains SDK-free.
:class:`CodexEvidenceProvider` does not pull the SDK either — it reads
tee'd JSONL only — so eager construction is safe. The plugin installer
imports stay lazy too so a Claude/Amp-only run does not pay for the
Codex install path on import.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
import threading
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeVar, cast

from src.core.constants import (
    DEFAULT_CODEX_APPROVAL_POLICY,
    DEFAULT_CODEX_EFFORT,
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_SANDBOX,
)
from src.infra.clients._plugin_provider_base import PluginInstallError
from src.infra.clients.codex_evidence_provider import CodexEvidenceProvider
from src.infra.clients.codex_mcp_factory import (
    _build_merged_codex_plugin_mcp_json,
    _create_codex_mcp_server_factory,
)
from src.infra.clients.codex_sdk_compat import (
    codex_runtime_resolvable,
    import_codex_app_server,
    import_codex_generated_v2_all,
    resolve_codex_bin_for_app_server,
    start_thread_compat,
)
from src.infra.clients.codex_selftest import (
    _build_module_hash_probe_code,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine

    from src.core.protocols.agent_provider import CoderRuntimeBuilder
    from src.core.protocols.evidence import EvidenceProvider
    from src.core.protocols.lifecycle import DeadlockMonitorProtocol
    from src.core.protocols.sdk import (
        McpServerFactory,
        SDKClientFactoryProtocol,
        SDKClientProtocol,
    )


_T = TypeVar("_T")


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
# Public exception + reason enum (Phase E5, T015)
# ---------------------------------------------------------------------------


class CodexNotInstalledError(RuntimeError):
    """Raised when Codex itself is not usable — SDK, runtime, or auth missing.

    Plan AC #14 (decision #8): ``coder=codex`` runs fail closed with an
    actionable message before any Codex turn is attempted. Mirrors
    :class:`AmpPluginNotActiveError`'s posture without conflating
    "Codex not installed" with "plugin/hook not loaded" (the latter is
    :class:`CodexHookNotActiveError`).
    """


class CodexHookNotActiveReason(StrEnum):
    """Why the Codex safety-hook selftest failed (structured form for callers).

    The string values are stable identifiers the orchestrator / telemetry
    can branch on without parsing prose error messages. Mirrors
    :class:`AmpPluginNotActiveReason` shape for cross-coder dashboards.
    """

    HOOK_MARKER_MISSING = "hook_marker_missing"
    VERSION_MISMATCH = "version_mismatch"
    SCRIPT_MISSING = "script_missing"
    PLUGIN_DISABLED = "plugin_disabled"
    TRUSTED_HASH_MISMATCH = "trusted_hash_mismatch"
    CODEX_BINARY_MISSING = "codex_binary_missing"


_HOOK_DOCS_URL = (
    "https://github.com/charlieyou/mala/blob/main/docs/cli-reference.md"
    "#codex-prerequisites"
)


class CodexHookNotActiveError(PluginInstallError):
    """Fail-closed error raised when the Codex safety hook is not active.

    Carries a structured :attr:`reason` so the orchestrator can branch on
    failure type and present a tailored remediation. The message always
    points at the docs URL so users have a single landing page for the
    fix; the docs subsection enumerates each reason and the matching
    one-time prerequisite step (relevant when auto-trust is unavailable —
    plan E6 fallback).
    """

    def __init__(
        self,
        message: str,
        *,
        reason: CodexHookNotActiveReason,
        docs_url: str = _HOOK_DOCS_URL,
    ) -> None:
        super().__init__(f"{message} See {docs_url} for the prerequisite steps.")
        self.reason = reason
        self.docs_url = docs_url


# Selftest probe contract: a callable that exercises a one-shot Codex turn
# and verifies the bundled hook ran with the expected version hash. Returns
# ``None`` on success; raises :class:`CodexHookNotActiveError` on any
# fail-closed signature. The provider's default probe stays minimal in
# this issue (T015) — Phase I/T020 layers SDK / runtime / auth checks on
# top, and the real-Codex e2e gate (I3) drives the spawn path. Tests
# inject custom probes to drive each :class:`CodexHookNotActiveReason`.


# ---------------------------------------------------------------------------
# Real client factory (Phase C)
# ---------------------------------------------------------------------------


class _CodexClientFactory:
    """:class:`SDKClientFactoryProtocol` for the Codex coder.

    ``create(runtime)`` constructs a :class:`CodexClient` bound to the
    provided :class:`CodexRuntime` (lazy import preserves the SDK
    isolation contract). ``with_resume(runtime, resume)`` returns a
    sibling runtime with ``resume_thread_id`` populated so the next
    :meth:`CodexClient.query` picks ``AsyncCodex.thread_resume`` over
    ``thread_start``.
    """

    def create(self, runtime: object) -> SDKClientProtocol:
        # Lazy import keeps ``codex_provider`` free of the
        # ``codex_client`` module's module-load-time cost; under
        # Claude/Amp-only runs the factory is constructed but
        # ``create`` is never reached.
        from src.infra.clients.codex_client import CodexClient
        from src.infra.clients.codex_runtime import CodexRuntime

        if not isinstance(runtime, CodexRuntime):
            raise TypeError(
                "CodexAgentProvider.client_factory.create(runtime) requires "
                "a CodexRuntime; got "
                f"{type(runtime).__name__}."
            )
        return cast("SDKClientProtocol", CodexClient(runtime))

    def with_resume(self, runtime: object, resume: str | None) -> object:
        from dataclasses import replace

        from src.infra.clients.codex_runtime import CodexRuntime

        if not isinstance(runtime, CodexRuntime):
            raise TypeError(
                "CodexAgentProvider.client_factory.with_resume(runtime, resume) "
                "requires a CodexRuntime; got "
                f"{type(runtime).__name__}."
            )
        if resume is None:
            return runtime
        return replace(runtime, resume_thread_id=resume)


# ---------------------------------------------------------------------------
# Codex runtime resolution (Phase E5)
# ---------------------------------------------------------------------------


def _codex_runtime_resolvable() -> bool:
    """Return True iff Codex can find a runtime at session-start.

    Kept as a provider-local wrapper because tests and the e2e gate import
    it directly. The underlying helper is also used by the SDK client/probes,
    so the gate and the real ``AppServerConfig`` code path stay aligned.
    """
    return codex_runtime_resolvable()


_CODEX_AUTH_ENV_VARS: tuple[str, ...] = (
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
    "CODEX_ACCESS_TOKEN",
)
"""Env vars Codex's auth manager treats as a valid credential source
(``codex-rs/login/src/auth/manager.rs:465-489``). A non-empty value in
any of these short-circuits the on-disk ``auth.json`` lookup, so an
auth probe that ignores them would falsely fail when a CI environment
injects ``OPENAI_API_KEY`` directly."""

_CODEX_CONFIG_AUTH_SEED_KEYS: tuple[str, ...] = (
    "chatgpt_base_url",
    "cli_auth_credentials_store",
    "forced_chatgpt_workspace_id",
    "forced_login_method",
)
"""User ``config.toml`` top-level keys allowed into Mala's isolated home.

Mala-launched Codex workers need enough user config for authentication
to keep working (notably keyring-backed ChatGPT credentials), but they
must not inherit user plugin/hook/MCP/tooling config. Inheriting those
tables lets interactive local Codex extensions (for example stop hooks
that wait for a review gate) run inside unattended Mala workers and can
block completion after the model has already emitted its final marker.

Keep this allowlist intentionally small and auth-scoped. Mala supplies
its own per-worker model/sandbox/approval/cwd values through the SDK and
writes its own plugin/hook trust state into the isolated config later.
"""


def _codex_auth_present() -> bool:
    """Return True iff Codex has a usable credential at session-start.

    Mirrors the credential discovery order Codex itself walks
    (``codex-rs/login/src/auth/manager.rs``): a non-empty
    ``OPENAI_API_KEY`` / ``CODEX_API_KEY`` / ``CODEX_ACCESS_TOKEN`` env
    var is accepted directly, otherwise the on-disk
    ``$CODEX_HOME/auth.json`` written by ``codex login`` is the source
    of truth (``codex-rs/login/src/auth/auth_tests.rs:57-114``).
    When ``cli_auth_credentials_store = "keyring"`` is configured in
    ``$CODEX_HOME/config.toml``, Codex's keyring backend deletes
    ``auth.json`` after a successful save, so neither env vars nor the
    file may be present even though Codex itself has a valid stored
    credential — the probe defers to Codex's auth manager in that case
    by treating the keyring opt-in as authenticated.
    Returning False fails closed via
    :class:`CodexNotInstalledError` so unattended ``coder=codex`` runs
    do not silently spawn ``codex app-server`` only to crash on the
    first turn.

    The probe is non-invasive: it does not import the Codex SDK,
    spawn a subprocess, or trigger an interactive ``Sign in with
    ChatGPT`` flow. That matters because ``install_prerequisites`` is
    called on the orchestrator's hot path before any thread starts;
    side effects there would race concurrent agents.
    """
    from src.infra.clients.codex_plugin_installer import _resolve_codex_home

    for var in _CODEX_AUTH_ENV_VARS:
        if (os.environ.get(var) or "").strip():
            return True
    codex_home = _resolve_codex_home()
    if (codex_home / "auth.json").is_file():
        return True
    return _codex_uses_keyring_credentials_store(codex_home)


def _codex_uses_keyring_credentials_store(codex_home: Path) -> bool:
    """Return True iff ``config.toml`` opts into the keyring backend.

    Codex's ``cli_auth_credentials_store = "keyring"`` knob switches
    credential storage from ``auth.json`` to the OS keyring; on
    successful keyring save the Rust auth manager removes the on-disk
    file, so a fully authenticated user can have neither the file nor
    any auth env var. Detecting the opt-in here lets the probe defer
    the actual credential read to Codex itself (which the probe cannot
    perform without importing the SDK or invoking the CLI) instead of
    incorrectly raising :class:`CodexNotInstalledError`.

    Failures to read or parse ``config.toml`` are swallowed and treated
    as "no keyring opt-in" so a malformed config file falls back to the
    file/env probe rather than masking a genuinely missing credential.
    """
    from src.infra.clients.codex_plugin_installer import _HOOK_CONFIG_FILENAME

    config_path = codex_home / _HOOK_CONFIG_FILENAME
    if not config_path.is_file():
        return False
    import tomllib

    try:
        raw = config_path.read_bytes()
        parsed = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return False
    value = parsed.get("cli_auth_credentials_store")
    return isinstance(value, str) and value.strip().lower() == "keyring"


def _render_toml_seed_scalar(value: object) -> str | None:
    """Render a small TOML scalar for the isolated Codex config seed.

    The auth seed allowlist currently contains string-valued Codex
    settings, but accepting primitive scalars keeps the helper tolerant
    if Codex adds a boolean/int auth knob later. Complex tables and
    arrays are deliberately skipped: copying tables is exactly how user
    ``[plugins]`` / ``[hooks]`` / ``[mcp_servers]`` config leaked into
    unattended workers.
    """
    if isinstance(value, str):
        import json as _json

        return _json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    return None


def _build_isolated_codex_config_seed(user_config: Path) -> str:
    """Return the sanitized ``config.toml`` seed for Mala's ``CODEX_HOME``.

    The prior implementation copied the user's whole ``config.toml`` into
    the temporary home and then appended Mala's plugin/trust entries. That
    preserved unrelated user plugins and hook trust state; when a local
    plugin registered a long-running ``Stop`` hook, Mala's noninteractive
    Codex worker could hang after emitting a final result because Codex
    waited for that user hook to finish. This seed copies only the
    top-level auth-scoped scalars in
    :data:`_CODEX_CONFIG_AUTH_SEED_KEYS`; everything else is recreated by
    Mala or left to Codex defaults.

    Read/parse failures are best-effort and return an empty seed, matching
    the existing isolation behavior where auth was probed against the
    user's real home before the isolated home was allocated.
    """
    import tomllib

    try:
        parsed = tomllib.loads(user_config.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return ""

    lines: list[str] = []
    for key in _CODEX_CONFIG_AUTH_SEED_KEYS:
        rendered = _render_toml_seed_scalar(parsed.get(key))
        if rendered is not None:
            lines.append(f"{key} = {rendered}\n")
    if not lines:
        return ""
    return (
        "# Generated by mala for isolated Codex workers. Only auth-related "
        "user config is copied.\n" + "".join(lines)
    )


# ---------------------------------------------------------------------------
# Trusted-hash auto-trust helpers (Phase E6, decision #16)
# ---------------------------------------------------------------------------
#
# The TOML rewriting helpers (``_resolve_codex_home``, ``_plugin_id``,
# ``_hook_state_key``, ``_write_codex_plugin_config``,
# ``_ensure_key_in_section``, ``_rewrite_toml_block``) and the supporting
# constants live in :mod:`src.infra.clients.codex_plugin_installer`. The
# provider delegates to that module via lazy imports inside the call
# sites so importing :mod:`codex_provider` does not pull the installer
# on the cold path (see
# ``test_importing_codex_provider_does_not_pull_codex_app_server``).


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
:data:`src.infra.hooks.codex_pre_tool_use.SELFTEST_MARKER_ENV` —
import-linter forbids ``src.infra.clients`` from importing
``src.infra.hooks`` directly, so the literal is duplicated and the
equality is pinned by a unit test."""


_HOOK_SELFTEST_MARKER_DIR_ENV = "MALA_CODEX_HOOK_SELFTEST_MARKER_DIR"
"""Env var pointing the hook at a directory for per-event markers.

Mirrors :data:`src.infra.hooks.codex_pre_tool_use.SELFTEST_MARKER_DIR_ENV`.
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
:data:`src.infra.hooks.codex_pre_tool_use.SELFTEST_TARGET_EVENT_ENV`."""


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


def _compute_combined_module_hash(modules: tuple[str, ...]) -> str:
    """Compute the same combined SHA-256 the probe emits, but in-process.

    Used by :func:`_default_selftest_probe` to derive *our* expected
    hash; the on-PATH probe must match this value byte-for-byte.
    Mirrors the probe code's iteration order and length-prefix scheme
    exactly so both sides land on the same digest.
    """
    import hashlib as _hashlib
    import importlib.util as _importlib_util

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
selftest run via ``_selftest_cache_key``."""


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
    Cached via ``_selftest_cache_key`` so a long-running orchestrator
    pays at most once per startup.
    """
    import asyncio
    import json as _json
    import tempfile
    import time as _time

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
        :func:`src.infra.hooks.codex_pre_tool_use._emit_selftest_marker_if_requested`).
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

    from src.infra.clients.codex_plugin_installer import (
        PLUGIN_DIRNAME,
        _HOOK_EVENTS,
        _resolve_codex_home,
        plugin_root_dir,
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
    # ``src.infra.hooks.codex_pre_tool_use._emit_selftest_marker_if_requested``);
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


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class CodexAgentProvider:
    """:class:`AgentProvider` for the Codex coder backend (T010, T013, T015).

    Conforms structurally to
    :class:`src.core.protocols.agent_provider.AgentProvider`. After T010
    ``client_factory`` and ``runtime_builder`` are real; after T013
    ``evidence_provider`` is real; after T015 (this issue)
    ``install_prerequisites`` runs the bundled-plugin install + hook
    self-test and is fail-closed on every documented signature.

    Attributes:
        name: Provider identifier (always ``"codex"``).
        client_factory: :class:`SDKClientFactoryProtocol` whose
            ``create`` returns a :class:`CodexClient` bound to the
            given :class:`CodexRuntime`.
        evidence_provider: :class:`CodexEvidenceProvider` reading the
            per-thread tee'd notification stream.
    """

    name: Literal["codex"] = "codex"

    def __init__(
        self,
        *,
        model: str = DEFAULT_CODEX_MODEL,
        effort: str | None = DEFAULT_CODEX_EFFORT,
        approval_policy: Literal[
            "never", "on-request", "on-failure", "untrusted"
        ] = DEFAULT_CODEX_APPROVAL_POLICY,
        sandbox: Literal[
            "read-only", "workspace-write", "danger-full-access"
        ] = DEFAULT_CODEX_SANDBOX,
        mcp_servers: tuple[tuple[str, object], ...] = (),
        selftest_probe: Callable[[Path, dict[str, str], str], None] | None = None,
    ) -> None:
        """Initialize the provider.

        Individual ``model`` / ``effort`` / ``approval_policy`` / ``sandbox``
        kwargs let callers construct the provider without importing
        ``src.infra.io.config``, preserving the import-linter contract that
        ``src.infra.clients`` does not depend on ``src.infra.io``. The
        :func:`src.orchestration.factory._create_agent_provider` branch
        unpacks ``MalaConfig.coder_options.codex`` into these kwargs so
        the resolved options reach the provider unchanged.

        ``selftest_probe`` is the runtime hook used by
        :meth:`install_prerequisites` to verify the bundled safety hook
        is loaded. The default probe statically validates the on-disk
        plugin tree (manifest path links + hooks.json declarations);
        tests inject custom probes to drive each
        :class:`CodexHookNotActiveReason` from
        :class:`CodexHookNotActiveError`, including the runtime-only
        reasons (HOOK_MARKER_MISSING / VERSION_MISMATCH /
        PLUGIN_DISABLED / TRUSTED_HASH_MISMATCH).
        """
        self._model: str = model
        self._effort: str | None = effort
        self._approval_policy: Literal[
            "never", "on-request", "on-failure", "untrusted"
        ] = approval_policy
        self._sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = (
            sandbox
        )
        self._mcp_servers: tuple[tuple[str, object], ...] = mcp_servers
        self._selftest_probe: Callable[[Path, dict[str, str], str], None] = (
            selftest_probe if selftest_probe is not None else _default_selftest_probe
        )
        self._client_factory_cached: _CodexClientFactory | None = None
        self._evidence_provider_cached: CodexEvidenceProvider | None = None
        # In-memory selftest cache keyed on the installed plugin hash. Plan
        # E5 invokes selftest at most once per run; the key matches the
        # Amp provider's ``(coder_version, plugin_hash)`` pattern keyed
        # with a placeholder coder version.
        self._selftest_cache_key: tuple[str, str] | None = None
        # Per-provider isolated ``CODEX_HOME`` populated by
        # :meth:`install_prerequisites`. Codex's plugin tree and hook
        # trust state are global to a ``CODEX_HOME``; writing the
        # bundled ``mala-safety`` plugin into the user's real
        # ``~/.codex`` makes normal, non-mala Codex sessions discover
        # the safety hook too. Routing every mala Codex invocation
        # through a provider-private CODEX_HOME keeps the plugin scoped
        # to the Codex app-server subprocesses mala launches, while
        # also preserving the previous user-MCP race fix (each
        # invocation owns its merged ``.mcp.json`` and ``config.toml``).
        # ``None`` until :meth:`install_prerequisites` allocates it.
        self._isolated_codex_home: tempfile.TemporaryDirectory[str] | None = None

    # ------------------------------------------------------------------
    # AgentProvider protocol surface
    # ------------------------------------------------------------------

    @property
    def client_factory(self) -> SDKClientFactoryProtocol:
        if self._client_factory_cached is None:
            self._client_factory_cached = _CodexClientFactory()
        return cast("SDKClientFactoryProtocol", self._client_factory_cached)

    @property
    def evidence_provider(self) -> EvidenceProvider:
        if self._evidence_provider_cached is None:
            self._evidence_provider_cached = CodexEvidenceProvider()
        return cast("EvidenceProvider", self._evidence_provider_cached)

    @property
    def model(self) -> str:
        """Resolved Codex model carried by this provider."""
        return self._model

    @property
    def effort(self) -> str | None:
        """Resolved Codex reasoning effort."""
        return self._effort

    @property
    def approval_policy(
        self,
    ) -> Literal["never", "on-request", "on-failure", "untrusted"]:
        """Resolved Codex approval policy."""
        return self._approval_policy

    @property
    def sandbox(
        self,
    ) -> Literal["read-only", "workspace-write", "danger-full-access"]:
        """Resolved Codex sandbox mode."""
        return self._sandbox

    def runtime_builder(
        self,
        repo_path: Path,
        agent_id: str,
        *,
        mcp_server_factory: McpServerFactory,
        deadlock_monitor: DeadlockMonitorProtocol | None = None,
    ) -> CoderRuntimeBuilder:
        """Construct a per-session :class:`CodexRuntimeBuilder`.

        Threads the resolved Codex options
        (``MalaConfig.coder_options.codex.*``) and the injected
        ``mcp_server_factory`` into the builder. ``deadlock_monitor``
        is accepted for protocol parity (plan A6); Phase C does not
        wire it because Codex's lock-event surface ships with the
        Phase E hook + Phase G MCP, which arrive after T010.
        """
        del deadlock_monitor
        # Lazy import so module-load of ``codex_provider`` does not
        # transitively reach ``src.infra.hooks`` via the runtime's
        # ``LintCache`` carrier (the import-linter contract is OK with
        # the codex_runtime exception, but keeping this lazy mirrors
        # the Amp path's posture and keeps the ``coder=claude`` /
        # ``coder=amp`` cold-path unchanged unless Codex is selected.
        from src.infra.clients.codex_runtime import CodexRuntimeBuilder

        builder = CodexRuntimeBuilder(
            repo_path,
            agent_id,
            mcp_server_factory,
            model=self._model,
            effort=self._effort,
            approval_policy=self._approval_policy,
            sandbox=self._sandbox,
        )
        # Thread the per-provider isolated ``CODEX_HOME`` into the
        # runtime's per-process env so the spawned ``codex app-server``
        # reads the mala-scoped plugin tree from the isolated location
        # instead of the user's normal Codex home. Without this overlay
        # the installation would be isolated on disk but Codex would
        # still inherit the user's ``CODEX_HOME`` at runtime.
        if self._isolated_codex_home is not None:
            builder.with_env(
                extra={"CODEX_HOME": str(Path(self._isolated_codex_home.name))}
            )
        return cast("CoderRuntimeBuilder", builder)

    def mcp_server_factory(self) -> McpServerFactory:
        """Return the Codex-shaped MCP server factory (Phase G3, T016).

        The factory closes over ``self._mcp_servers`` (user-supplied
        ``coder_options.codex.mcp_servers``) and merges them with the
        bundled ``mala-locking`` launch spec; the bundled key is always
        last-wins so user config cannot override it (plan G3).
        """
        return _create_codex_mcp_server_factory(self._mcp_servers)

    def _ensure_isolated_codex_home(self, user_codex_home: Path) -> Path:
        """Lazily allocate this provider's private ``CODEX_HOME`` and
        seed it with the user's auth credential.

        The plugin tree + ``config.toml`` Codex reads at
        ``thread_start`` live under ``$CODEX_HOME``. Installing Mala's
        safety plugin into the user's real Codex home makes it visible
        to ordinary Codex CLI sessions, where ``MALA_AGENT_ID`` /
        ``MALA_LOCK_DIR`` are absent and the hook fails closed on
        edits. Allocating a per-provider ``TemporaryDirectory`` whose
        path is fed to the spawned Codex via :meth:`runtime_builder`
        scopes the plugin and trust entries to this mala run only.

        The same isolation also removes the user-MCP race the prior
        design fixed only for ``coder_options.codex.mcp_servers``: a
        concurrent mala invocation with a different MCP config cannot
        overwrite this invocation's merged ``.mcp.json`` because no
        other process can name this directory.

        Auth.json is symlinked from the user's real ``$CODEX_HOME`` so
        Codex's auth manager finds the credential at the same logical
        path it would otherwise read directly. When ``OPENAI_API_KEY`` /
        ``CODEX_API_KEY`` / ``CODEX_ACCESS_TOKEN`` is set the env-var
        path covers auth and the symlink is unnecessary, but we
        attempt it regardless so a refresh-token rotation by Codex
        (which writes back to ``auth.json``) lands in the user's real
        file rather than the temp dir.

        ``config.toml`` is **sanitized** (not symlinked or copied
        wholesale) from the user's real ``$CODEX_HOME``. Only
        auth-scoped top-level scalars such as
        ``cli_auth_credentials_store = "keyring"`` carry into the
        isolated home. Without this seed, a user who logged in via
        Codex's keyring backend has
        neither ``auth.json`` (the keyring backend deletes it after a
        successful save) nor the keyring opt-in inside the isolated
        home — the spawned Codex falls back to the default
        ``auth_json`` credential store, finds no file, and crashes at
        ``thread_start`` instead of using the stored keyring
        credential. Sanitizing (rather than symlinking or copying) is
        required because :func:`_write_codex_plugin_config` writes to
        this file later and because third-party user plugin/hook tables
        must not enter Mala's unattended worker config.

        The :class:`tempfile.TemporaryDirectory` handle is held on the
        provider so the directory survives for the orchestrator's
        lifetime; cleanup happens when the provider is garbage-collected
        or the process exits, mirroring the lifecycle of a single mala
        invocation.
        """
        from src.infra.clients.codex_plugin_installer import _HOOK_CONFIG_FILENAME

        if self._isolated_codex_home is None:
            self._isolated_codex_home = tempfile.TemporaryDirectory(
                prefix="mala-codex-home-"
            )
            isolated = Path(self._isolated_codex_home.name)
            user_auth = user_codex_home / "auth.json"
            target_auth = isolated / "auth.json"
            if user_auth.is_file():
                try:
                    target_auth.symlink_to(user_auth)
                except OSError:
                    # Filesystems without symlink support (rare on POSIX,
                    # possible on Windows-mounted volumes) fall back to
                    # a copy. The copy will diverge if Codex refreshes
                    # the user's auth.json mid-run; that is the trade-off
                    # for those filesystems and matches Amp's posture.
                    try:
                        shutil.copy2(user_auth, target_auth)
                    except OSError:
                        # Best-effort: if even copy fails, leave the
                        # isolated home auth-less and let the spawned
                        # Codex surface the auth error itself. The
                        # provider's auth probe ran before this point
                        # against the user's real home, so the failure
                        # here would only manifest at thread_start.
                        pass
            user_config = user_codex_home / _HOOK_CONFIG_FILENAME
            if user_config.is_file():
                try:
                    # Write a sanitized config seed rather than copying
                    # the user's full config.toml. That preserves the
                    # auth knobs Codex needs (notably keyring-backed
                    # credentials) while stripping user plugins, hooks,
                    # MCP servers, shell env policies, and other
                    # interactive local extensions that can hang Mala's
                    # noninteractive workers. ``write_text`` also keeps
                    # the isolated copy writable even when the user's
                    # real config is managed read-only by dotfile tools.
                    config_seed = _build_isolated_codex_config_seed(user_config)
                    if config_seed:
                        (isolated / _HOOK_CONFIG_FILENAME).write_text(
                            config_seed, encoding="utf-8"
                        )
                except OSError:
                    # Best-effort: a missing seed here means
                    # ``_write_codex_plugin_config`` writes a
                    # plugin-only file and a keyring user loses the
                    # opt-in. The auth probe ran against the user's
                    # real home before this point so the failure mode
                    # surfaces only at thread_start; matches the
                    # auth.json copy fallback above.
                    pass
        return Path(self._isolated_codex_home.name)

    def install_prerequisites(
        self,
        repo_path: Path,
        *,
        mcp_server_factory: McpServerFactory,
    ) -> None:
        """Install the bundled Codex plugin and run the hook self-test (T015).

        Idempotent within a run: the result is cached on the installed
        plugin hash; subsequent calls within the same run short-circuit
        without re-running the install or the selftest.

        Steps (plan ``L778-L924``):

          1. Verify ``codex_app_server`` SDK is importable (raise
             :class:`CodexNotInstalledError` if absent).
          2. Verify the ``codex`` binary is on ``PATH`` (raise
             :class:`CodexHookNotActiveError(CODEX_BINARY_MISSING)`).
          3. Verify Codex auth state is present — either an
             ``OPENAI_API_KEY`` / ``CODEX_API_KEY`` /
             ``CODEX_ACCESS_TOKEN`` env var is set, or
             ``$CODEX_HOME/auth.json`` exists (raise
             :class:`CodexNotInstalledError` pointing at ``codex login``
             otherwise). Plan I1 / decision #8: missing auth fails
             closed before any ``codex app-server`` spawn so unattended
             runs never trip an interactive ``Sign in with ChatGPT``
             prompt.
          4. Run :class:`CodexPluginInstaller` to copy the bundled plugin
             tree into a per-run isolated Codex home at
             ``<isolated CODEX_HOME>/plugins/cache/<marketplace>/<plugin>/<version>/.codex-plugin/``
             (per ``codex-rs/core-plugins/src/store.rs::PluginStore.plugin_root``).
          5. Verify the ``mala-codex-pre-tool-use`` console script is on
             ``PATH`` (raise
             :class:`CodexHookNotActiveError(SCRIPT_MISSING)`).
          6. Auto-write the five Codex-config preconditions to the
             isolated ``CODEX_HOME/config.toml`` (decision #16):
             ``[features] plugins = true`` to override an opt-out
             setting that would short-circuit plugin loading entirely
             (``plugins_for_config_with_force_reload`` early-returns
             when ``plugins_enabled`` is false);
             ``[features] plugin_hooks = true`` so Codex's
             ``catalog_processor`` actually loads plugin-bundled hooks
             (this feature ships ``default_enabled = false``);
             ``[features] hooks = true`` so Codex's global hook
             execution gate cannot leave the loaded/trusted safety hook
             dormant;
             ``[plugins."<plugin_id>"] enabled = true`` so Codex's
             ``configured_plugins_from_stack`` enumerates the plugin;
             and
             ``[hooks.state."<plugin_id>:.codex-plugin/hooks.json:pre_tool_use:0:0"]``
             with ``enabled = true`` + ``trusted_hash`` so the hook is
             marked Trusted. I/O failures raise
             :class:`CodexHookNotActiveError(TRUSTED_HASH_MISMATCH)`.
          7. Run the selftest probe (default: structural verification
             of the installed plugin tree at the install target); it
             raises :class:`CodexHookNotActiveError` on
             ``HOOK_MARKER_MISSING`` / ``VERSION_MISMATCH`` /
             ``PLUGIN_DISABLED`` / ``TRUSTED_HASH_MISMATCH``. Tests
             inject custom probes to drive the runtime-only reasons.

        Args:
            repo_path: Working directory the selftest probe should use
                (mirrors the runtime_builder shape).
            mcp_server_factory: Codex-shaped MCP factory threaded
                through the selftest so the probe can build a runtime
                that matches a real session's MCP wiring.

        Raises:
            CodexNotInstalledError: When the Codex SDK is not
                importable or when no Codex credential is detectable.
            CodexHookNotActiveError: When any structural or runtime
                fail-closed signature fires.
        """
        del (
            mcp_server_factory
        )  # threaded for protocol parity; selftest probe doesn't need it directly

        # 1. SDK importability — module-spec lookup avoids actually
        # importing ``codex_app_server`` so the lazy-import contract
        # (plan L733) is preserved on the success path. ``find_spec``
        # itself can import parent packages but does not load the
        # module body.
        try:
            sdk_spec = importlib.util.find_spec("codex_app_server")
        except (ImportError, ValueError):
            sdk_spec = None
        if sdk_spec is None:
            raise CodexNotInstalledError(
                "codex_app_server SDK is not importable. Install it with "
                "`uv add openai-codex-app-server-sdk` and rerun mala. "
                "See the docs for the full Codex prerequisites."
            )

        # 2. Codex runtime resolvable. Either an external ``codex`` binary
        # on PATH (passed explicitly as ``AppServerConfig.codex_bin``), or
        # the SDK-bundled ``codex_cli_bin`` runtime package (which
        # ``codex_app_server.client.resolve_codex_bin`` falls back to when
        # ``AppServerConfig.codex_bin`` is None). Honors ``CODEX_BINARY`` so
        # an explicit operator override skips both lookups.
        if not _codex_runtime_resolvable():
            raise CodexHookNotActiveError(
                "Codex runtime not resolvable: no `codex` binary on PATH, "
                "no `CODEX_BINARY` override, and the bundled "
                "`codex_cli_bin` runtime package is not importable. "
                "Install `openai-codex-cli-bin` (typically as a dependency "
                "of `openai-codex-app-server-sdk`) or place a `codex` "
                "binary on PATH.",
                reason=CodexHookNotActiveReason.CODEX_BINARY_MISSING,
            )

        # 3. Codex auth state. Without a credential the spawned
        # ``codex app-server`` would either fail the first turn with an
        # opaque RPC error or — worse, on a fresh machine — pop the
        # interactive ``Sign in with ChatGPT`` flow. Both are
        # unacceptable on the unattended orchestrator path; fail closed
        # here with the install-and-login command per decision #8 / AC
        # #14.
        if not _codex_auth_present():
            from src.infra.clients.codex_plugin_installer import (
                _resolve_codex_home as _resolve_codex_home_for_msg,
            )

            raise CodexNotInstalledError(
                "Codex auth missing: none of `OPENAI_API_KEY`, "
                "`CODEX_API_KEY`, or `CODEX_ACCESS_TOKEN` is set, "
                f"`{_resolve_codex_home_for_msg() / 'auth.json'}` does not exist, "
                'and `cli_auth_credentials_store = "keyring"` is not '
                "configured in `config.toml`. "
                "Run `codex login` (Sign in with ChatGPT) or set "
                "`OPENAI_API_KEY` to an OpenAI API key, then rerun mala. "
                "See https://developers.openai.com/codex/auth for the "
                "full auth setup."
            )

        # 4. Run the plugin installer (idempotent on its own).
        from src.infra.clients.codex_plugin_installer import (
            PLUGIN_DIRNAME,
            CodexPluginInstallError,
            CodexPluginInstaller,
            _resolve_codex_home,
            _write_codex_plugin_config,
            plugin_root_dir,
        )

        # Choose the active ``CODEX_HOME`` for the rest of install.
        # Always allocate a provider-private isolated home so the
        # bundled plugin + hook trust entries are visible only to Codex
        # app-server subprocesses launched by this mala provider, never
        # to the user's ordinary Codex CLI sessions. The isolated path
        # is also threaded into runtime env via :meth:`runtime_builder`.
        user_codex_home = _resolve_codex_home()
        codex_home = self._ensure_isolated_codex_home(user_codex_home)
        install_target = plugin_root_dir(codex_home) / PLUGIN_DIRNAME

        # Build the merged ``.mcp.json`` payload (Phase G3 / AC-3) and
        # hand it to the installer as ``mcp_json_override`` so the
        # installer is the sole writer of the plugin tree. Routing the
        # merge through the installer keeps idempotency intact (a rerun
        # with the same merged bytes short-circuits at
        # ``action="skipped"``) and — combined with the per-provider
        # isolated ``CODEX_HOME`` above — closes the cross-process race
        # the prior post-install rewrite (and the prior single-shared-
        # CODEX_HOME design) opened: without isolation, Process B's
        # installer would silently revert Process A's merged file to a
        # bundled-only payload because B's own user config builds a
        # different override; with isolation, B writes to its own
        # ``CODEX_HOME`` and A's plugin tree is untouched. The same
        # isolation now applies to the default bundled-only path, so a
        # mala run never mutates the user's normal Codex plugin state
        # (plan ``L954``: bundled is mandatory, never replaced;
        # non-conflicting user keys pass through).
        mcp_json_override = _build_merged_codex_plugin_mcp_json(self._mcp_servers)
        try:
            install_result = CodexPluginInstaller(
                mcp_json_override=mcp_json_override
            ).install(target_dir=install_target)
        except CodexPluginInstallError as exc:
            raise CodexHookNotActiveError(
                f"Codex plugin install failed: {exc}",
                reason=CodexHookNotActiveReason.PLUGIN_DISABLED,
            ) from exc

        plugin_hash = install_result.plugin_hash

        # 5. Hook script must be on PATH so Codex can launch it via
        # ``"command": "mala-codex-pre-tool-use"`` from hooks.json.
        if shutil.which("mala-codex-pre-tool-use") is None:
            raise CodexHookNotActiveError(
                "mala-codex-pre-tool-use console script not on PATH. "
                "Ensure mala is installed via `uv tool install mala-agent` "
                "or `uv sync` so the project.scripts entry point is exposed.",
                reason=CodexHookNotActiveReason.SCRIPT_MISSING,
            )

        # 6. Auto-write the five config-side preconditions to
        # ``<isolated codex_home>/config.toml`` (decision #16). Codex
        # requires ALL FIVE for the hook to fire: ``[features] plugins = true``
        # to override an opt-out user config that early-returns from
        # ``plugins_for_config_with_force_reload``;
        # ``[features] plugin_hooks = true`` to enable plugin-bundled
        # hook loading (default-off in upstream Codex);
        # ``[features] hooks = true`` to enable global hook execution;
        # ``[plugins."<id>"] enabled = true`` to surface the plugin via
        # ``configured_plugins_from_stack``; and
        # ``[hooks.state."<id>:..."]`` with the matching ``trusted_hash``
        # to mark the hook Trusted. ``codex_home`` here is the *active*
        # home (isolated when user MCP is configured, user's real home
        # otherwise) so the writes land in the same directory the
        # spawned Codex will read.
        _write_codex_plugin_config(codex_home=codex_home)

        # 7. Cache key + selftest. The key matches the Amp provider's
        # ``(coder_version, plugin_hash)`` shape; ``"unknown"`` carries
        # the placeholder until a reliable ``codex --version`` parse is
        # added (parity with the Amp path's posture).
        cache_key = ("unknown", plugin_hash)
        if self._selftest_cache_key == cache_key:
            return

        env_overlay: dict[str, str] = {
            "CODEX_HOME": str(codex_home),
        }
        # The probe contract raises CodexHookNotActiveError on any
        # fail-closed signature; default probe is a no-op. Tests inject
        # custom probes that simulate each Reason.
        self._selftest_probe(repo_path, env_overlay, plugin_hash)

        self._selftest_cache_key = cache_key

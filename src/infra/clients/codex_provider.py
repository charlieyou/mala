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
inside :class:`CodexClient.__aenter__` / :meth:`CodexClient.query` (and
even those uses are guarded by ``try/except TypeError`` for backward
compatibility); :class:`CodexClient` itself is imported lazily by
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
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from src.core.constants import (
    DEFAULT_CODEX_APPROVAL_POLICY,
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_SANDBOX,
)
from src.infra.clients.codex_evidence_provider import CodexEvidenceProvider

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.core.protocols.agent_provider import CoderRuntimeBuilder
    from src.core.protocols.evidence import EvidenceProvider
    from src.core.protocols.lifecycle import DeadlockMonitorProtocol
    from src.core.protocols.sdk import (
        McpServerFactory,
        SDKClientFactoryProtocol,
        SDKClientProtocol,
    )

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


class CodexHookNotActiveError(RuntimeError):
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
# MCP server factory (provider-owned, Phase B placeholder; Phase G3 wires it)
# ---------------------------------------------------------------------------


def _create_codex_mcp_server_factory() -> McpServerFactory:
    """Stub Codex-shaped MCP factory.

    Phase G wires the bundled ``mala-locking`` launcher inside the Codex
    plugin (`plans/2026-05-07-codex-provider-plan.md` G1-G3); for Phase C
    the factory still returns an empty map so :meth:`AgentProvider.mcp_server_factory`
    has a callable to hand back. The runtime carries this dict through
    to ``AsyncCodex.thread_start(mcp_servers=...)`` unchanged so Phase G
    is a one-line factory swap.
    """

    def factory(
        agent_id: str,
        repo_path: Path,
        emit_lock_event: object,
    ) -> dict[str, object]:
        del agent_id, repo_path, emit_lock_event
        return {}

    return cast("McpServerFactory", factory)


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
    """Return True iff Codex can find a ``codex`` binary at session-start.

    Mirrors the resolution path :func:`codex_app_server.client.resolve_codex_bin`
    walks (``codex/sdk/python/src/codex_app_server/client.py:107-117``):

      1. ``CODEX_BINARY`` env (operator override) — if set, accept and let
         the SDK validate the path.
      2. ``codex`` on ``PATH`` — accept if found.
      3. SDK-bundled ``codex_cli_bin`` package — accept if importable.

    Returning True when only the bundled runtime is present matches the
    SDK's ``AppServerConfig(codex_bin=None)`` default that
    :class:`CodexClient.__aenter__` relies on
    (``src/infra/clients/codex_client.py:226-228``). Returning False
    raises :class:`CodexHookNotActiveError(CODEX_BINARY_MISSING)`.
    """
    if os.environ.get("CODEX_BINARY"):
        return True
    if shutil.which("codex") is not None:
        return True
    try:
        spec = importlib.util.find_spec("codex_cli_bin")
    except (ImportError, ValueError):
        spec = None
    return spec is not None


# ---------------------------------------------------------------------------
# Trusted-hash auto-trust helpers (Phase E6, decision #16)
# ---------------------------------------------------------------------------


_HOOK_CONFIG_FILENAME = "config.toml"
"""Codex's user config file relative to ``CODEX_HOME``. Hook trust state
lives in ``[hooks.state."<key>"]`` tables here per
``codex-rs/core/tests/common/hooks.rs::trusted_config_layer_stack``: that
test fixture is the source-of-truth for how Codex marks discovered
plugin hooks as trusted. The schema's flat ``[state."<key>"]`` form (in
a standalone ``hooks.toml``) deserializes the same data, but the
canonical install path Codex reads on startup is ``config.toml``.
"""

_PLUGIN_SOURCE_RELATIVE_PATH = ".codex-plugin/hooks.json"
"""Codex's hook-key uses the plugin-relative path of the hooks file
(``codex-rs/hooks/src/declarations.rs:35`` ``plugin_hook_key_source``)
combined with ``<plugin_id>``. Our hooks.json is bundled at
``mala-safety/.codex-plugin/hooks.json``; the relative path Codex
records is therefore ``.codex-plugin/hooks.json``.
"""


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


def _resolve_codex_home() -> Path:
    """Return the active Codex home directory honoring ``CODEX_HOME``."""
    env_value = os.environ.get("CODEX_HOME")
    if env_value:
        return Path(env_value)
    return Path("~/.codex").expanduser()


def _plugin_id(marketplace: str = _DEFAULT_PLUGIN_MARKETPLACE) -> str:
    """Return the Codex ``PluginId.as_key()`` value for our plugin."""
    return f"mala-safety@{marketplace}"


def _hook_state_key(*, marketplace: str = _DEFAULT_PLUGIN_MARKETPLACE) -> str:
    """Compute the Codex hook-state key for the bundled PreToolUse entry.

    Mirrors ``codex-rs/hooks/src/declarations.rs::plugin_hook_key_source`` +
    ``codex-rs/hooks/src/lib.rs::hook_key``: for plugin hooks the key
    is ``<plugin_id>:<source_relative_path>:<event>:<group>:<handler>``.
    Our hooks.json declares exactly one PreToolUse matcher with one
    command handler, so the indices are ``0:0``.
    """
    return f"{_plugin_id(marketplace)}:{_PLUGIN_SOURCE_RELATIVE_PATH}:pre_tool_use:0:0"


def _normalized_hook_identity_value() -> object:
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
        "event_name": "pre_tool_use",
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


def _compute_normalized_hook_hash() -> str:
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
    import hashlib
    import json as _json

    canonical = _canonical_json(_normalized_hook_identity_value())
    serialized = _json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(serialized).hexdigest()}"


def _write_codex_plugin_config(
    *,
    codex_home: Path,
    marketplace: str = _DEFAULT_PLUGIN_MARKETPLACE,
) -> None:
    """Fail-closed write of the three config-side preconditions to ``config.toml``.

    Codex requires THREE config-side preconditions for the bundled
    PreToolUse hook to fire:

      * The ``plugin_hooks`` feature flag is enabled:
        ``[features] plugin_hooks = true`` (per
        ``codex-rs/features/src/lib.rs:957`` — ``Feature::PluginHooks``
        ships ``default_enabled = false`` and gates plugin-bundled hook
        loading inside ``catalog_processor`` /
        ``manager.plugins_for_config``). Without it, Codex caches the
        plugin tree but registers zero hooks.
      * The plugin is enabled in user config: ``[plugins."<id>"] enabled = true``
        (per ``codex-rs/core-plugins/src/manager.rs::configured_plugins_from_stack``
        — only ``[plugins."<key>"]`` entries are surfaced as "configured
        plugins" and only those whose ``enabled`` is True are loaded).
      * The hook is marked trusted: ``[hooks.state."<id>:<rel>:pre_tool_use:0:0"]``
        with ``enabled = true`` and the matching ``trusted_hash`` (per
        ``codex-rs/hooks/src/engine/discovery.rs::hook_trust_status``).

    Without all three entries, Codex would discover the cached plugin
    tree (``$CODEX_HOME/plugins/cache/<marketplace>/<plugin>/<version>/``)
    yet still skip the hook on PreToolUse — the orchestrator would
    proceed under ``danger-full-access`` / ``approval_policy=never``
    without the safety gate. Auto-trust is therefore the safety-critical
    bridge between "plugin tree on disk" and "hook actually loaded"
    (decision #16). I/O failures here raise
    :class:`CodexHookNotActiveError(TRUSTED_HASH_MISMATCH)` so
    :meth:`install_prerequisites` aborts the run; the documented
    one-time interactive trust fallback (plan E6) covers users whose
    ``CODEX_HOME`` is read-only.

    The "already correct" short-circuit (when all three blocks already
    match the desired content) returns without writing. The
    ``[features]`` block is updated in-place (preserving any other
    feature flags the user already set) rather than replaced wholesale.
    """
    plugin_id = _plugin_id(marketplace)
    plugin_section = f'[plugins."{plugin_id}"]'
    plugin_block = f"{plugin_section}\nenabled = true\n"

    state_section = f'[hooks.state."{_hook_state_key(marketplace=marketplace)}"]'
    trusted_hash_value = _compute_normalized_hook_hash()
    state_block = (
        f'{state_section}\nenabled = true\ntrusted_hash = "{trusted_hash_value}"\n'
    )

    target = codex_home / _HOOK_CONFIG_FILENAME

    try:
        codex_home.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise CodexHookNotActiveError(
            f"Cannot create Codex home directory {codex_home} to write the "
            f"plugin/feature config entries ({exc}). Auto-trust is "
            "the bridge that lets Codex load the bundled safety hook; "
            "without it Codex marks the hook untrusted and never invokes it.",
            reason=CodexHookNotActiveReason.TRUSTED_HASH_MISMATCH,
        ) from exc

    try:
        existing = target.read_text(encoding="utf-8") if target.exists() else ""
    except OSError as exc:
        raise CodexHookNotActiveError(
            f"Cannot read Codex config file {target} to update the "
            f"plugin/feature config entries ({exc}).",
            reason=CodexHookNotActiveReason.TRUSTED_HASH_MISMATCH,
        ) from exc

    # 1. Ensure ``plugin_hooks = true`` inside the existing ``[features]``
    # block (preserving any other feature flags the user already set).
    rewritten = _ensure_key_in_section(
        existing, section_header="[features]", key="plugin_hooks", value="true"
    )
    # 2 + 3. Replace the per-plugin and per-hook state blocks.
    rewritten = _rewrite_toml_block(
        rewritten, section_header=plugin_section, new_block=plugin_block
    )
    rewritten = _rewrite_toml_block(
        rewritten, section_header=state_section, new_block=state_block
    )
    if rewritten == existing:
        return  # all three already correct

    try:
        target.write_text(rewritten, encoding="utf-8")
    except OSError as exc:
        raise CodexHookNotActiveError(
            f"Cannot write Codex plugin/hook config to {target} ({exc}). "
            "The bundled safety hook would remain disabled or untrusted; "
            "refusing to run under danger-full-access without the safety gate.",
            reason=CodexHookNotActiveReason.TRUSTED_HASH_MISMATCH,
        ) from exc


def _ensure_key_in_section(
    existing: str, *, section_header: str, key: str, value: str
) -> str:
    """Ensure ``<key> = <value>`` lives inside ``[<section_header>]``.

    Pure-text rewrite suitable for flat key=value blocks like
    ``[features]`` (the ``Feature`` registry is keyed by snake-case
    string and serialized as bool / int / string scalars per
    ``codex-rs/features/src/lib.rs``; nested table syntax is not used
    inside this section). Behavior:

      * If ``[section_header]`` does not exist: append it with the new
        key=value line.
      * If ``[section_header]`` exists and contains ``<key> = ...``:
        rewrite that line to ``<key> = <value>`` (preserving every
        other key inside the section).
      * If ``[section_header]`` exists but lacks the key: append the
        new line at the end of the section (just before the next
        top-level ``[...]`` header).

    Returns the existing text unchanged if the block already has
    ``<key> = <value>``.
    """
    new_line = f"{key} = {value}\n"
    lines = existing.splitlines(keepends=True)
    if not lines:
        return f"{section_header}\n{new_line}"

    section_start = -1
    for idx, line in enumerate(lines):
        if line.strip() == section_header:
            section_start = idx
            break

    if section_start < 0:
        out = list(lines)
        if out and not out[-1].endswith("\n"):
            out[-1] = out[-1] + "\n"
        out.append("\n")
        out.append(f"{section_header}\n")
        out.append(new_line)
        return "".join(out)

    # Find the end of the section (next top-level header or EOF).
    section_end = len(lines)
    for idx in range(section_start + 1, len(lines)):
        stripped = lines[idx].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            section_end = idx
            break

    # Look for an existing ``<key> = ...`` line inside the section.
    key_pattern_prefix = f"{key} ="
    for idx in range(section_start + 1, section_end):
        stripped = lines[idx].strip()
        if stripped.startswith(key_pattern_prefix) or stripped.startswith(f"{key}="):
            if stripped == f"{key} = {value}":
                return existing  # already correct
            # Preserve indentation if any.
            indent_len = len(lines[idx]) - len(lines[idx].lstrip())
            lines[idx] = lines[idx][:indent_len] + new_line
            return "".join(lines)

    # Key missing inside the section — splice it just before section_end.
    insert_at = section_end
    # Skip trailing blank lines so the new key sits adjacent to other
    # keys rather than after a blank gap.
    while insert_at > section_start + 1 and lines[insert_at - 1].strip() == "":
        insert_at -= 1
    lines.insert(insert_at, new_line)
    return "".join(lines)


def _rewrite_toml_block(existing: str, *, section_header: str, new_block: str) -> str:
    """Return ``existing`` with ``section_header``'s block replaced by ``new_block``.

    Pure-text rewrite (no TOML round-trip) so the function can run
    without a third-party TOML writer. The replacement is bounded to
    one ``[<section>]`` table at a time: every line from the matching
    header up to the next top-level ``[...]`` header (or EOF) is
    dropped and ``new_block`` is spliced in.

    Used for both ``[plugins."<id>"]`` and ``[hooks.state."<key>"]``
    blocks; the contract is identical (a single bracketed header
    followed by simple ``key = value`` lines, terminated by the next
    top-level table header).
    """
    lines = existing.splitlines(keepends=True)
    if not lines:
        return new_block.rstrip("\n") + "\n"

    out: list[str] = []
    i = 0
    found = False
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped == section_header and not found:
            found = True
            out.append(new_block if new_block.endswith("\n") else new_block + "\n")
            i += 1
            while i < len(lines):
                next_stripped = lines[i].strip()
                if next_stripped.startswith("[") and next_stripped.endswith("]"):
                    break
                i += 1
            continue
        out.append(lines[i])
        i += 1
    if not found:
        if out and not out[-1].endswith("\n"):
            out[-1] = out[-1] + "\n"
        out.append("\n" if out else "")
        out.append(new_block if new_block.endswith("\n") else new_block + "\n")
    return "".join(out)


# ---------------------------------------------------------------------------
# Default selftest probe (Phase E5 structural verification + real-Codex spawn)
# ---------------------------------------------------------------------------


_HOOK_COMMAND_NAME = "mala-codex-pre-tool-use"
"""Hook command literal we expect to find inside the installed plugin's
``hooks.json``. Matches the ``[project.scripts]`` entry in
``pyproject.toml`` and the bundled ``hooks.json`` registration."""


def _default_selftest_probe(
    repo_path: Path,
    env_overlay: dict[str, str],
    expected_hash: str,
) -> None:
    """Default selftest probe — structural verification of the installed plugin.

    Fail-closed verification of the on-disk plugin tree at the install
    target Codex reads. Catches the install-corruption / wrong-path /
    wrong-command failure modes that the up-front structural checks
    (SDK importable, codex on PATH, hook script on PATH) leave open:

      * Plugin manifest at
        ``$CODEX_HOME/plugins/cache/<marketplace>/<plugin>/<version>/.codex-plugin/plugin.json``
        exists, parses as JSON, and points at ``./.codex-plugin/hooks.json``.
        Failure → :class:`CodexHookNotActiveError(PLUGIN_DISABLED)`.
      * Installed hooks.json exists, parses, and declares a PreToolUse
        command handler whose ``command`` is ``mala-codex-pre-tool-use``.
        Failure → :class:`CodexHookNotActiveError(HOOK_MARKER_MISSING)`.

    Plan E5 also calls for a one-shot Codex turn that triggers the
    bundled hook and verifies it ran with the expected version hash.
    That live-spawn verification depends on a Codex binary + auth + a
    sentinel-emitting hook (``MALA_HOOK_SELFTEST_MARKER`` would be a
    T014 follow-up since the hook script is out of scope for this
    issue) and is therefore tracked as a follow-up real-Codex e2e gate.
    The structural verification here is the defense-in-depth baseline:
    it catches every install-time signature reliably without paying
    real-binary cost on every run.

    Tests inject custom probes to drive each
    :class:`CodexHookNotActiveReason` directly, including the runtime
    reasons (HOOK_MARKER_MISSING / VERSION_MISMATCH / PLUGIN_DISABLED /
    TRUSTED_HASH_MISMATCH) that a future live-spawn probe would surface.
    """
    del repo_path, expected_hash
    import json as _json

    from src.infra.clients.codex_plugin_installer import (
        PLUGIN_DIRNAME,
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
    if not _hooks_json_declares_pre_tool_use_command(hooks_payload, _HOOK_COMMAND_NAME):
        raise CodexHookNotActiveError(
            f"Codex plugin hooks.json at {hooks_json_path} does not declare a "
            f"PreToolUse command handler for {_HOOK_COMMAND_NAME!r}; the "
            "safety hook would not fire on tool use.",
            reason=CodexHookNotActiveReason.HOOK_MARKER_MISSING,
        )


def _hooks_json_declares_pre_tool_use_command(
    payload: object, expected_command: str
) -> bool:
    """Return True iff ``payload`` is a HooksFile JSON declaring our hook."""
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
    pre_tool_use = hooks_dict.get("PreToolUse")
    if not isinstance(pre_tool_use, list):
        return False
    for group_obj in pre_tool_use:
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
        effort: str | None = None,
        approval_policy: Literal[
            "never", "on-request", "on-failure", "untrusted"
        ] = DEFAULT_CODEX_APPROVAL_POLICY,
        sandbox: Literal[
            "read-only", "workspace-write", "danger-full-access"
        ] = DEFAULT_CODEX_SANDBOX,
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
        """Resolved Codex reasoning effort (``None`` = SDK default)."""
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
        # ``coder=amp`` cold-path identical to the current default).
        from src.infra.clients.codex_runtime import CodexRuntimeBuilder

        return cast(
            "CoderRuntimeBuilder",
            CodexRuntimeBuilder(
                repo_path,
                agent_id,
                mcp_server_factory,
                model=self._model,
                effort=self._effort,
                approval_policy=self._approval_policy,
                sandbox=self._sandbox,
            ),
        )

    def mcp_server_factory(self) -> McpServerFactory:
        """Return the Codex-shaped MCP server factory (Phase G3 stub)."""
        return _create_codex_mcp_server_factory()

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
          3. Run :class:`CodexPluginInstaller` to copy the bundled plugin
             tree into Codex's PluginStore cache at
             ``$CODEX_HOME/plugins/cache/<marketplace>/<plugin>/<version>/.codex-plugin/``
             (per ``codex-rs/core-plugins/src/store.rs::PluginStore.plugin_root``).
          4. Verify the ``mala-codex-pre-tool-use`` console script is on
             ``PATH`` (raise
             :class:`CodexHookNotActiveError(SCRIPT_MISSING)`).
          5. Auto-write the three Codex-config preconditions to
             ``$CODEX_HOME/config.toml`` (decision #16):
             ``[features] plugin_hooks = true`` so Codex's
             ``catalog_processor`` actually loads plugin-bundled hooks
             (the feature ships ``default_enabled = false``);
             ``[plugins."<plugin_id>"] enabled = true`` so Codex's
             ``configured_plugins_from_stack`` enumerates the plugin;
             and
             ``[hooks.state."<plugin_id>:.codex-plugin/hooks.json:pre_tool_use:0:0"]``
             with ``enabled = true`` + ``trusted_hash`` so the hook is
             marked Trusted. I/O failures raise
             :class:`CodexHookNotActiveError(TRUSTED_HASH_MISMATCH)`.
          6. Run the selftest probe (default: structural verification
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
            CodexNotInstalledError: When the Codex SDK is not importable.
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
        # on PATH, or the SDK-bundled ``codex_cli_bin`` runtime package
        # (which ``codex_app_server.client.resolve_codex_bin`` falls back
        # to when ``AppServerConfig.codex_bin`` is None — the same path
        # ``CodexClient.__aenter__`` exercises). Honors ``CODEX_BINARY``
        # so an explicit operator override skips both lookups.
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

        # 3. Run the plugin installer (idempotent on its own).
        from src.infra.clients.codex_plugin_installer import (
            CodexPluginInstallError,
            CodexPluginInstaller,
        )

        try:
            install_result = CodexPluginInstaller().install()
        except CodexPluginInstallError as exc:
            raise CodexHookNotActiveError(
                f"Codex plugin install failed: {exc}",
                reason=CodexHookNotActiveReason.PLUGIN_DISABLED,
            ) from exc

        plugin_hash = install_result.plugin_hash

        # 4. Hook script must be on PATH so Codex can launch it via
        # ``"command": "mala-codex-pre-tool-use"`` from hooks.json.
        if shutil.which("mala-codex-pre-tool-use") is None:
            raise CodexHookNotActiveError(
                "mala-codex-pre-tool-use console script not on PATH. "
                "Ensure mala is installed via `uv tool install mala-agent` "
                "or `uv sync` so the project.scripts entry point is exposed.",
                reason=CodexHookNotActiveReason.SCRIPT_MISSING,
            )

        # 5. Auto-write the three config-side preconditions to
        # config.toml (decision #16). Codex requires ALL THREE for the
        # hook to fire: ``[features] plugin_hooks = true`` to enable
        # plugin-bundled hook loading (default-off in upstream Codex),
        # ``[plugins."<id>"] enabled = true`` to surface the plugin via
        # ``configured_plugins_from_stack``, and ``[hooks.state."<id>:..."]``
        # with the matching ``trusted_hash`` to mark the hook Trusted.
        codex_home = _resolve_codex_home()
        _write_codex_plugin_config(codex_home=codex_home)

        # 6. Cache key + selftest. The key matches the Amp provider's
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

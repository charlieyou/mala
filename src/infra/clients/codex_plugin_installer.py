"""Idempotent installer for the bundled ``mala-safety`` Codex plugin (T015).

Copies the source-tree at ``plugins/codex/mala-safety/.codex-plugin/`` into
Codex's PluginStore cache at
``$CODEX_HOME/plugins/cache/<marketplace>/<plugin>/<version>/.codex-plugin/``
(default ``~/.codex/plugins/cache/local/mala-safety/local/.codex-plugin/``)
so Codex's loader discovers the plugin on every run via its standard
discovery path
(``codex-rs/core-plugins/src/store.rs::PluginStore.active_plugin_root``).
Production mala Codex runs pass an explicit target under a provider-private
temporary ``CODEX_HOME``; the ambient ``~/.codex`` default is kept for the
low-level installer contract and tests, not for normal mala orchestration.
The installer is concurrent-safe (per-file write-temp-then-rename) and
idempotent (SHA-256 content comparison per file).

Layout shipped (fixed, regardless of the Phase E spike outcome — the
installer is a straight copy with no path translation):

  plugins/codex/mala-safety/.codex-plugin/
    plugin.json   - Codex plugin manifest
    hooks.json    - PreToolUse command-hook registration
    .mcp.json     - mala-locking MCP launcher (T016 finalizes wire shape)

The on-disk install location alone is necessary but not sufficient for
the safety hook to fire: Codex also requires a
``[plugins."<plugin>@<marketplace>"] enabled = true`` entry in
``$CODEX_HOME/config.toml`` (``configured_plugins_from_stack``), plus a
matching ``[hooks.state."<key>"] enabled = true, trusted_hash = ...``
entry to flip the hook from Untrusted to Trusted
(``hook_trust_status``). Both auto-trust writes are owned by
``CodexAgentProvider.install_prerequisites()``; the installer here only
guarantees the bytes-on-disk side of the contract.

The 16-hex-char digest exposed via :meth:`installed_plugin_hash` is
useful for diagnostics / logging — callers compose Codex's trusted_hash
themselves from the normalized hook identity (the provider's
``_compute_normalized_hook_hash``), not from the plugin file bytes.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PLUGIN_NAME = "mala-safety"
"""Codex plugin manifest name. Must match ``plugin.json``'s ``name``."""

PLUGIN_DIRNAME = ".codex-plugin"
"""Per ``plugin-json-spec.md``: the manifest lives in ``.codex-plugin/`` of the plugin root."""

PLUGIN_FILENAMES: tuple[str, ...] = ("plugin.json", "hooks.json", ".mcp.json")
"""Files shipped inside the plugin's ``.codex-plugin/`` directory.

Order is deterministic so :meth:`CodexPluginInstaller.installed_plugin_hash`
returns a stable hash across runs.
"""

PLUGIN_MARKETPLACE = "local"
"""Marketplace component of the Codex ``PluginId.as_key()``. Codex's
``DEFAULT_PLUGIN_VERSION`` constant is ``"local"`` (per
``codex-rs/core-plugins/src/store.rs:14``); the same string is also the
canonical marketplace name for plugins installed without a remote
catalog. Using ``local`` for both segments keeps the installed cache
path and the user-config plugin key consistent with Codex's
local-plugin convention."""

PLUGIN_VERSION = "local"
"""Plugin version directory under
``$CODEX_HOME/plugins/cache/<marketplace>/<plugin>/<version>/``.

``codex-rs/core-plugins/src/store.rs:91`` ``active_plugin_root``
selects the active version via ``active_plugin_version`` which prefers
the literal ``"local"`` over any other value when present, then falls
back to the lexicographically-largest version directory. Installing
under ``local`` therefore keeps the active version stable across
mala upgrades — there is no version-bump churn that would drop
trusted-hash state in user config."""

PLUGIN_ID_KEY = f"{PLUGIN_NAME}@{PLUGIN_MARKETPLACE}"
"""``PluginId.as_key()`` value Codex uses for ``[plugins."<key>"]`` and
``[hooks.state."<key>:..."]`` entries (per
``codex-rs/plugin/src/plugin_id.rs:45``)."""

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

_HOOK_EVENTS: tuple[tuple[str, str], ...] = (
    ("PreToolUse", "pre_tool_use"),
    ("SessionStart", "session_start"),
)
"""Hook events the bundled plugin's ``hooks.json`` registers, paired
with the Codex internal snake_case identifier used in
:func:`_hook_state_key` / ``_normalized_hook_identity_value``.

Both entries point at the same ``mala-codex-pre-tool-use`` command;
PreToolUse is the safety gate (lock / dangerous-cmd / disallowed-tool
/ shell-write enforcement), and SessionStart is exercised by the
live-Codex selftest probe so the on-disk install is exercised through
Codex's actual plugin discovery + feature gates + per-handler trust
state + dispatch machinery without an LLM call (decision: SessionStart
hook returns ``continue=false`` in selftest mode, aborting the turn
before any model call). The two entries must stay in lockstep with
``plugins/codex/mala-safety/.codex-plugin/hooks.json``."""

DEFAULT_CODEX_HOME = Path("~/.codex").expanduser()
"""Codex configuration directory (per ``codex_utils_home_dir::find_codex_home``).

Honors ``CODEX_HOME`` env var when set; otherwise defaults to ``~/.codex``.
:func:`_resolve_codex_home` is the runtime form that reads ``CODEX_HOME``
on every call so a test or user-override takes effect.
"""

_MARKETPLACE_MANIFEST_RELATIVE_PATH = ".agents/plugins/marketplace.json"
"""Home-scoped marketplace manifest path Codex scans via ``plugin/list``.

Codex 0.130 no longer surfaces arbitrary cache entries in ``plugin/list``.
The cache only answers "is this configured plugin installed?"; the visible
plugin catalog comes from marketplace manifests discovered from configured
marketplace roots. Mala writes a minimal marketplace into the isolated
``CODEX_HOME`` and points ``[marketplaces."local"].source`` at that home.
"""


def marketplace_manifest_path(codex_home: Path) -> Path:
    """Return the bundled local marketplace manifest path for ``codex_home``."""
    return codex_home / _MARKETPLACE_MANIFEST_RELATIVE_PATH


def _resolve_codex_home() -> Path:
    """Return the active Codex home directory.

    Honors ``CODEX_HOME`` (the env var Codex itself reads) so tests and
    user-overrides target the same directory Codex consults.
    """
    env_value = os.environ.get("CODEX_HOME")
    if env_value:
        return Path(env_value)
    return DEFAULT_CODEX_HOME


def plugin_root_dir(codex_home: Path | None = None) -> Path:
    """Return the plugin root Codex's ``PluginStore.active_plugin_root`` resolves to.

    Layout (per ``codex-rs/core-plugins/src/store.rs::PluginStore``):

      ``<codex_home>/plugins/cache/<marketplace>/<plugin>/<version>/``

    The ``.codex-plugin/`` manifest directory lives one level below
    this; see :func:`default_plugin_target_dir`.
    """
    home = codex_home if codex_home is not None else _resolve_codex_home()
    return (
        home / "plugins" / "cache" / PLUGIN_MARKETPLACE / PLUGIN_NAME / PLUGIN_VERSION
    )


def _marketplace_source_path(marketplace: str = PLUGIN_MARKETPLACE) -> str:
    """Return the marketplace-local source path for the installed plugin root."""
    return f"./plugins/cache/{marketplace}/{PLUGIN_NAME}/{PLUGIN_VERSION}"


def default_plugin_target_dir() -> Path:
    """Resolve the default Codex plugin manifest directory at call time.

    Lands at
    ``$CODEX_HOME/plugins/cache/<marketplace>/<plugin>/<version>/.codex-plugin/``
    (per ``codex-rs/core-plugins/src/store.rs::PluginStore.plugin_root``)
    so Codex's ``active_plugin_root`` discovery path
    (``store.rs:91``) and ``configured_plugins_from_stack``
    enumeration (``manager.rs:1976``) both find the bundled plugin.
    Honors ``CODEX_HOME`` so tests pointing the env var at ``tmp_path``
    install into the redirected Codex home.

    Note: the install location alone is not sufficient for Codex to
    load the plugin's hooks — the user must also have
    ``[plugins."<plugin>@<marketplace>"] enabled = true`` in
    ``$CODEX_HOME/config.toml``. That config-write is the
    :class:`CodexAgentProvider.install_prerequisites`'s responsibility
    (it runs after this installer).
    """
    return plugin_root_dir() / PLUGIN_DIRNAME


_VERSION_MARKER_HEX_CHARS = 16
"""Length of the hex prefix returned by :meth:`installed_plugin_hash`.

Mirrors the Amp plugin's ``computeOwnVersionHash`` (16-char SHA-256
prefix) so logs / debug output stay consistent across coders.
"""

_TEMP_PREFIX = ".mala-safety."
_TEMP_SUFFIX = ".tmp"

_WHEEL_DATA_DIRNAME = "_codex_plugin_data"
"""Sibling directory inside the installed wheel that holds the bundled
plugin tree. Populated by ``[tool.hatch.build.targets.wheel.force-include]``
in ``pyproject.toml`` (mirroring the Amp plugin's wheel-data layout).
The source-checkout layout does not have this directory; the resolver
falls back to the repository's ``plugins/codex/mala-safety/.codex-plugin``
location.
"""


InstallAction = Literal["wrote", "skipped", "replaced"]


@dataclass(frozen=True)
class InstallResult:
    """Outcome of one :meth:`CodexPluginInstaller.install` call.

    Attributes:
        target_dir: Directory the plugin tree was copied to.
        plugin_hash: 16-hex-char SHA-256 prefix of the combined source
            bytes; suitable as the ``trusted_hash`` value Codex's
            hook-state file expects.
        action: ``"wrote"`` (no prior install), ``"skipped"`` (existing
            install already matches), ``"replaced"`` (existing install
            was stale).
    """

    target_dir: Path
    plugin_hash: str
    action: InstallAction


class CodexPluginInstallError(RuntimeError):
    """Raised when the installer cannot guarantee the installed plugin tree."""


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bundled_source_dir(module_file: Path | None = None) -> Path:
    """Return the bundled plugin-tree path for the active install layout.

    Wheel install: hatch's ``force-include`` ships the plugin at
    ``<this-module-dir>/_codex_plugin_data/`` so the files are inside
    the package payload. The resolver checks this location first and
    returns it when the manifest exists.

    Source checkout / editable install: the files live at their original
    repository path
    ``<repo-root>/plugins/codex/mala-safety/.codex-plugin/``. The
    resolver falls back to that location when the wheel-data sibling
    does not exist.

    ``module_file`` is exposed for unit tests so both branches can be
    exercised against a synthetic install layout under ``tmp_path``.
    """
    here = (module_file if module_file is not None else Path(__file__)).resolve()
    wheel_data = here.parent / _WHEEL_DATA_DIRNAME
    if (wheel_data / "plugin.json").is_file():
        return wheel_data
    return here.parents[3] / "plugins" / "codex" / PLUGIN_NAME / PLUGIN_DIRNAME


def _read_source_files(source_dir: Path) -> dict[str, bytes]:
    """Read the bundled source tree into ``{filename: bytes}``.

    Raises :class:`CodexPluginInstallError` with an actionable message
    if any of the required files is missing.
    """
    payload: dict[str, bytes] = {}
    for name in PLUGIN_FILENAMES:
        path = source_dir / name
        try:
            payload[name] = path.read_bytes()
        except FileNotFoundError as exc:
            raise CodexPluginInstallError(
                f"Bundled Codex plugin file missing at {path}; "
                "the mala source tree appears incomplete."
            ) from exc
        except OSError as exc:
            raise CodexPluginInstallError(
                f"Cannot read bundled Codex plugin file at {path}: {exc}"
            ) from exc
    return payload


def _combined_hash(payload: dict[str, bytes]) -> str:
    """Deterministic SHA-256 over the full plugin tree payload.

    Uses a length-prefixed, name-sorted format so the digest is stable
    regardless of dict iteration order and unaffected by content that
    happens to look like a delimiter.
    """
    h = hashlib.sha256()
    for name in PLUGIN_FILENAMES:
        body = payload[name]
        name_bytes = name.encode("utf-8")
        h.update(len(name_bytes).to_bytes(4, "big"))
        h.update(name_bytes)
        h.update(len(body).to_bytes(8, "big"))
        h.update(body)
    return h.hexdigest()


def _plugin_id(marketplace: str = PLUGIN_MARKETPLACE) -> str:
    """Return the Codex ``PluginId.as_key()`` value for our plugin."""
    return f"{PLUGIN_NAME}@{marketplace}"


def _hook_state_key(
    event: str = "pre_tool_use",
    *,
    marketplace: str = PLUGIN_MARKETPLACE,
) -> str:
    """Compute the Codex hook-state key for a bundled hook entry.

    Mirrors ``codex-rs/hooks/src/declarations.rs::plugin_hook_key_source`` +
    ``codex-rs/hooks/src/lib.rs::hook_key``: for plugin hooks the key
    is ``<plugin_id>:<source_relative_path>:<event>:<group>:<handler>``.
    Our hooks.json declares exactly one matcher group per event with
    one command handler, so the indices are ``0:0`` for both
    ``pre_tool_use`` and ``session_start``.
    """
    return f"{_plugin_id(marketplace)}:{_PLUGIN_SOURCE_RELATIVE_PATH}:{event}:0:0"


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
    # Tolerate any whitespace between the key and ``=`` (TOML allows
    # value alignment, e.g. ``plugins    = false``); reject non-key
    # prefixes (e.g. ``plugins_extra = ...`` must not match ``plugins``).
    for idx in range(section_start + 1, section_end):
        stripped = lines[idx].strip()
        if not stripped.startswith(key):
            continue
        remainder = stripped[len(key) :].lstrip()
        if not remainder.startswith("="):
            continue  # ``<key>_extra = ...`` and similar — not our key.
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
    # Guarantee the previous line ends with a newline before splicing the
    # new key in. Without this, a config.toml that ends mid-section
    # without a trailing ``\n`` (e.g. user-edited file) would have the
    # new ``key = value`` concatenated onto the previous line, producing
    # malformed TOML that crashes Codex on startup. Regression: gemini
    # P1 on review-17.
    if insert_at > 0 and not lines[insert_at - 1].endswith("\n"):
        lines[insert_at - 1] = lines[insert_at - 1] + "\n"
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


def _write_codex_plugin_config(
    *,
    codex_home: Path,
    marketplace: str = PLUGIN_MARKETPLACE,
) -> None:
    """Fail-closed write of the config-side preconditions to ``config.toml``.

    Codex requires six config-side preconditions for the bundled
    PreToolUse hook to fire:

      * The global ``plugins`` feature flag is enabled:
        ``[features] plugins = true`` (per
        ``codex-rs/features/src/lib.rs:951`` — ``Feature::Plugins``
        ships ``default_enabled = true`` but a user can explicitly
        set ``[features] plugins = false`` and Codex's
        ``PluginsManager.plugins_for_config_with_force_reload``
        early-returns ``PluginLoadOutcome::default()`` when
        ``plugins_enabled`` is false, so we always pin it to ``true``).
      * The ``plugin_hooks`` feature flag is enabled:
        ``[features] plugin_hooks = true`` (per
        ``codex-rs/features/src/lib.rs:957`` — ``Feature::PluginHooks``
        ships ``default_enabled = false`` and gates plugin-bundled hook
        loading inside ``catalog_processor`` /
        ``manager.plugins_for_config``). Without it, Codex caches the
        plugin tree but registers zero hooks.
      * The global ``hooks`` feature flag is enabled:
        ``[features] hooks = true`` (per Codex's ``Feature::CodexHooks``
        gate on hook execution). A user-level opt-out leaves hooks
        registered/trusted but not executed, so we always pin it to
        ``true`` before trusting the unattended safety install.
      * The plugin is enabled in the active config:
        ``[plugins."<id>"] enabled = true``
        (per ``codex-rs/core-plugins/src/manager.rs::configured_plugins_from_stack``
        — only ``[plugins."<key>"]`` entries are surfaced as "configured
        plugins" and only those whose ``enabled`` is True are loaded).
      * The plugin's marketplace is registered:
        ``[marketplaces."<marketplace>"] source_type = "local"`` and
        ``source = "<isolated CODEX_HOME>"``. Codex's ``plugin/list``
        enumerates configured marketplace roots, then overlays cache
        and enabled state; an isolated Mala ``CODEX_HOME`` that has the
        cached plugin tree but no marketplace manifest reports zero
        plugin summaries.
      * The hook is marked trusted: ``[hooks.state."<id>:<rel>:pre_tool_use:0:0"]``
        with ``enabled = true`` and the matching ``trusted_hash`` (per
        ``codex-rs/hooks/src/engine/discovery.rs::hook_trust_status``).

    Without all six entries, Codex would discover the cached plugin
    tree (``$CODEX_HOME/plugins/cache/<marketplace>/<plugin>/<version>/``)
    yet still skip the hook on PreToolUse — the orchestrator would
    proceed under ``danger-full-access`` / ``approval_policy=never``
    without the safety gate. Auto-trust is therefore the safety-critical
    bridge between "plugin tree on disk" and "hook actually loaded"
    (decision #16). I/O failures here raise
    :class:`CodexHookNotActiveError(TRUSTED_HASH_MISMATCH)` so
    :meth:`install_prerequisites` aborts the run. The provider calls
    this helper with a per-run isolated ``CODEX_HOME`` so the writes
    do not mutate the user's normal Codex config.

    The "already correct" short-circuit (when all entries already
    match the desired content) returns without writing. The
    ``[features]`` block is updated in-place (preserving any other
    feature flags the user already set) rather than replaced wholesale.
    """
    # Lazy import so this module stays off the cold path of
    # :mod:`src.infra.clients.codex_provider` (the provider's lazy-import
    # contract is asserted by
    # ``test_importing_codex_provider_does_not_pull_codex_app_server``).
    # When :meth:`CodexAgentProvider.install_prerequisites` runs, it
    # lazy-imports this module; the provider's error class is already
    # defined at that point, so the back-import here resolves cleanly.
    from src.infra.clients.codex_provider import (
        CodexHookNotActiveError,
        CodexHookNotActiveReason,
    )
    from src.infra.clients.codex_selftest import _compute_normalized_hook_hash

    plugin_id = _plugin_id(marketplace)
    plugin_section = f'[plugins."{plugin_id}"]'
    plugin_block = f"{plugin_section}\nenabled = true\n"
    marketplace_section = f'[marketplaces."{marketplace}"]'
    marketplace_source_value = json.dumps(str(codex_home), ensure_ascii=False)

    # Each hook handler in the bundled ``hooks.json`` (PreToolUse +
    # SessionStart) gets its own ``[hooks.state."<key>"]`` block with
    # the per-event trusted_hash. Without writing the SessionStart
    # state Codex would mark the SessionStart handler untrusted and
    # the live-Codex selftest probe would never observe the hook
    # firing — the safety hook would still load (because PreToolUse
    # has its own state block) but the cross-Codex validation that
    # the install actually wired up correctly would silently break.
    state_blocks: list[tuple[str, str]] = []
    for _, snake_event in _HOOK_EVENTS:
        section = (
            f'[hooks.state."{_hook_state_key(snake_event, marketplace=marketplace)}"]'
        )
        trusted_hash_value = _compute_normalized_hook_hash(snake_event)
        block = f'{section}\nenabled = true\ntrusted_hash = "{trusted_hash_value}"\n'
        state_blocks.append((section, block))

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

    # 1a. Ensure ``plugins = true`` inside the existing ``[features]``
    # block — Feature::Plugins defaults to true but a user can opt
    # out, in which case ``plugins_for_config_with_force_reload``
    # early-returns and our hook never loads.
    rewritten = _ensure_key_in_section(
        existing, section_header="[features]", key="plugins", value="true"
    )
    # 1b. Ensure ``plugin_hooks = true`` inside the same ``[features]``
    # block — Feature::PluginHooks defaults to false and gates plugin
    # hook loading. Both keys live in the same flat table; the
    # in-place rewriter preserves any other feature flags the user
    # already set.
    rewritten = _ensure_key_in_section(
        rewritten, section_header="[features]", key="plugin_hooks", value="true"
    )
    # 1c. Ensure ``hooks = true`` inside the same ``[features]`` block —
    # Codex's global hook execution gate can be disabled independently
    # of plugin discovery and plugin-provided hook loading. If it stays
    # false, the hook can be installed and trusted but never invoked.
    rewritten = _ensure_key_in_section(
        rewritten, section_header="[features]", key="hooks", value="true"
    )
    # 2. Ensure the local marketplace exists so Codex's plugin/list
    # enumerates the marketplace manifest inside this isolated CODEX_HOME.
    rewritten = _ensure_key_in_section(
        rewritten,
        section_header=marketplace_section,
        key="source_type",
        value='"local"',
    )
    rewritten = _ensure_key_in_section(
        rewritten,
        section_header=marketplace_section,
        key="source",
        value=marketplace_source_value,
    )
    # 3 + 4. Replace the per-plugin and per-hook state blocks.
    rewritten = _rewrite_toml_block(
        rewritten, section_header=plugin_section, new_block=plugin_block
    )
    for state_section, state_block in state_blocks:
        rewritten = _rewrite_toml_block(
            rewritten, section_header=state_section, new_block=state_block
        )
    if rewritten == existing:
        return  # all config entries already correct

    try:
        target.write_text(rewritten, encoding="utf-8")
    except OSError as exc:
        raise CodexHookNotActiveError(
            f"Cannot write Codex plugin/hook config to {target} ({exc}). "
            "The bundled safety hook would remain disabled or untrusted; "
            "refusing to run under danger-full-access without the safety gate.",
            reason=CodexHookNotActiveReason.TRUSTED_HASH_MISMATCH,
        ) from exc


def _write_codex_plugin_marketplace_manifest(
    *,
    codex_home: Path,
    marketplace: str = PLUGIN_MARKETPLACE,
) -> None:
    """Write the isolated marketplace manifest that makes ``plugin/list`` see Mala.

    The installed plugin cache at ``plugins/cache/<marketplace>/<plugin>/<version>``
    is necessary for Codex to mark ``mala-safety@local`` as installed, but it is
    not itself the catalog that ``plugin/list`` displays. Codex discovers
    catalog entries from marketplace manifests. This helper writes a minimal
    local marketplace under the isolated ``CODEX_HOME`` and
    :func:`_write_codex_plugin_config` points ``[marketplaces."<marketplace>"]``
    at that same home.
    """
    from src.infra.clients.codex_provider import (
        CodexHookNotActiveError,
        CodexHookNotActiveReason,
    )

    manifest_path = marketplace_manifest_path(codex_home)
    payload = {
        "name": marketplace,
        "plugins": [
            {
                "name": PLUGIN_NAME,
                "source": {
                    "source": "local",
                    "path": _marketplace_source_path(marketplace),
                },
            }
        ],
    }

    try:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        raise CodexHookNotActiveError(
            f"Cannot write Codex local marketplace manifest to {manifest_path} "
            f"({exc}). Codex's plugin/list would not discover the bundled "
            "mala-safety plugin in the isolated CODEX_HOME.",
            reason=CodexHookNotActiveReason.PLUGIN_DISABLED,
        ) from exc


def _read_existing_payload(target_dir: Path) -> dict[str, bytes] | None:
    """Read the on-disk plugin tree at ``target_dir`` if all files exist.

    Returns ``None`` when any required file is missing or unreadable so
    the caller treats the install as a fresh write rather than an
    idempotent skip.
    """
    out: dict[str, bytes] = {}
    for name in PLUGIN_FILENAMES:
        path = target_dir / name
        try:
            out[name] = path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError:
            return None
    return out


class CodexPluginInstaller:
    """Idempotent, concurrent-safe installer for the ``mala-safety`` plugin.

    The installer is constructed once per orchestrator run and invoked
    by :meth:`CodexAgentProvider.install_prerequisites` (T015). The
    default ``source_dir`` resolves the bundled plugin tree inside the
    mala source tree; tests pass a custom directory to drive variants.

    ``mcp_json_override`` lets the caller substitute the bundled
    ``.mcp.json`` bytes with a merged user+bundled payload (Phase G3 /
    AC-3). The installer remains the **sole writer** of the plugin
    tree's files: routing the merged content through the same
    atomic-write + hash-check pipeline keeps idempotency intact (a
    rerun with the same merged bytes short-circuits at
    ``action="skipped"``) and prevents the cross-process race a
    separate post-install rewrite would introduce — without the
    override, Process A's render of the merged file would be silently
    overwritten by Process B's installer reverting to the bundled-only
    static file (``coder_options.codex.mcp_servers`` would not become
    effective at runtime).
    """

    def __init__(
        self,
        source_dir: Path | None = None,
        *,
        mcp_json_override: bytes | None = None,
    ) -> None:
        self._source_dir: Path = (
            source_dir if source_dir is not None else _bundled_source_dir()
        )
        self._mcp_json_override: bytes | None = mcp_json_override

    @property
    def source_dir(self) -> Path:
        """Path to the bundled plugin tree inside the mala source tree."""
        return self._source_dir

    def installed_plugin_hash(self) -> str:
        """Return the 16-hex-char trusted-hash for the effective plugin tree.

        Computed from the **effective** source bytes (the bundled tree
        with any ``mcp_json_override`` substituted in); a successful
        :meth:`install` guarantees the on-disk bytes match. The value
        is suitable as the ``trusted_hash`` Codex's hook-state file
        expects (Phase E6, decision #16).
        """
        payload = self._effective_source_payload()
        return _combined_hash(payload)[:_VERSION_MARKER_HEX_CHARS]

    def _effective_source_payload(self) -> dict[str, bytes]:
        """Return the source payload, applying any ``mcp_json_override``.

        Without an override this is the bundled tree as-is; with an
        override the ``.mcp.json`` entry is replaced so the rest of the
        installer (hash check, atomic write, post-rename verify) treats
        the merged bytes as the source of truth — there is no second
        writer that could drift the on-disk file off the recorded hash.
        """
        payload = _read_source_files(self._source_dir)
        if self._mcp_json_override is not None:
            payload[".mcp.json"] = self._mcp_json_override
        return payload

    def install(self, target_dir: Path | None = None) -> InstallResult:
        """Install the plugin tree into ``target_dir``, idempotently.

        Behavior:

          * ``mkdir -p target_dir`` (the user's Codex home may not have
            a plugins directory yet).
          * If every existing file at ``target_dir`` already matches
            the effective source bytes (bundled tree + any
            ``mcp_json_override``), return ``action="skipped"`` without
            writing.
          * Otherwise, write each effective-source file via
            temp-then-rename (atomic on POSIX) and verify the final
            hash matches.

        Concurrent invocations are safe per file (each writer takes its
        own ``NamedTemporaryFile`` and ``os.replace`` is atomic); last
        writer wins, but every writer wrote identical bytes from the
        same effective source, so the post-rename tree always matches
        the recorded hash.

        Args:
            target_dir: Directory to install into. Defaults to
                ``$CODEX_HOME/plugins/mala-safety/.codex-plugin/``
                (Codex's standard plugin location).

        Returns:
            :class:`InstallResult` describing the action taken plus the
            16-hex-char trusted-hash.

        Raises:
            CodexPluginInstallError: If the target directory cannot be
                created or written to, the bundled source is missing,
                or a post-rename hash does not match the bundled hash.
        """
        target_dir = (
            target_dir if target_dir is not None else default_plugin_target_dir()
        )

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CodexPluginInstallError(
                f"Cannot create or access Codex plugin directory {target_dir}: {exc}"
            ) from exc

        source_payload = self._effective_source_payload()
        source_hash = _combined_hash(source_payload)
        version_hash = source_hash[:_VERSION_MARKER_HEX_CHARS]

        existing_payload = _read_existing_payload(target_dir)
        if (
            existing_payload is not None
            and _combined_hash(existing_payload) == source_hash
        ):
            return InstallResult(
                target_dir=target_dir,
                plugin_hash=version_hash,
                action="skipped",
            )

        action: InstallAction = "replaced" if existing_payload is not None else "wrote"
        for name in PLUGIN_FILENAMES:
            self._atomic_write(target_dir, target_dir / name, source_payload[name])

        installed = _read_existing_payload(target_dir)
        if installed is None or _combined_hash(installed) != source_hash:
            raise CodexPluginInstallError(
                "Installed Codex plugin hash mismatch after rename. "
                "Concurrent writer or filesystem corruption suspected."
            )

        return InstallResult(
            target_dir=target_dir,
            plugin_hash=version_hash,
            action=action,
        )

    @staticmethod
    def _atomic_write(target_dir: Path, target_path: Path, payload: bytes) -> None:
        # Track tmp_path immediately after NamedTemporaryFile creates the file
        # on disk, BEFORE write/flush/fsync run. If any of those I/O calls
        # raises (ENOSPC, EIO, EDQUOT) the finally block must still find the
        # temp path so it can unlink the partial file. Assigning after fsync
        # would leak ``.mala-safety.*.tmp`` files into the Codex plugin
        # directory on every failure (parity with the Amp installer fix).
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target_dir,
                prefix=_TEMP_PREFIX,
                suffix=_TEMP_SUFFIX,
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(payload)
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_path, target_path)
            tmp_path = None
        except OSError as exc:
            raise CodexPluginInstallError(
                f"Cannot install Codex plugin file at {target_path}: {exc}"
            ) from exc
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except OSError as cleanup_exc:
                    if cleanup_exc.errno != errno.ENOENT:
                        # Best-effort cleanup; do not mask the original error.
                        pass

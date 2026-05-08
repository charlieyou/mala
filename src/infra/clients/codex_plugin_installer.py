"""Idempotent installer for the bundled ``mala-safety`` Codex plugin (T015).

Copies the source-tree at ``plugins/codex/mala-safety/.codex-plugin/`` into
Codex's PluginStore cache at
``$CODEX_HOME/plugins/cache/<marketplace>/<plugin>/<version>/.codex-plugin/``
(default ``~/.codex/plugins/cache/local/mala-safety/local/.codex-plugin/``)
so Codex's loader discovers the plugin on every run via its standard
discovery path
(``codex-rs/core-plugins/src/store.rs::PluginStore.active_plugin_root``).
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

DEFAULT_CODEX_HOME = Path("~/.codex").expanduser()
"""Codex configuration directory (per ``codex_utils_home_dir::find_codex_home``).

Honors ``CODEX_HOME`` env var when set; otherwise defaults to ``~/.codex``.
:func:`_resolve_codex_home` is the runtime form that reads ``CODEX_HOME``
on every call so a test or user-override takes effect.
"""


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
    """

    def __init__(self, source_dir: Path | None = None) -> None:
        self._source_dir: Path = (
            source_dir if source_dir is not None else _bundled_source_dir()
        )

    @property
    def source_dir(self) -> Path:
        """Path to the bundled plugin tree inside the mala source tree."""
        return self._source_dir

    def installed_plugin_hash(self) -> str:
        """Return the 16-hex-char trusted-hash for the bundled plugin tree.

        Computed from the **bundled** source bytes (not the on-disk
        copy); a successful :meth:`install` guarantees the on-disk
        bytes match. The value is suitable as the ``trusted_hash``
        Codex's hook-state file expects (Phase E6, decision #16).
        """
        payload = _read_source_files(self._source_dir)
        return _combined_hash(payload)[:_VERSION_MARKER_HEX_CHARS]

    def install(self, target_dir: Path | None = None) -> InstallResult:
        """Install the plugin tree into ``target_dir``, idempotently.

        Behavior:

          * ``mkdir -p target_dir`` (the user's Codex home may not have
            a plugins directory yet).
          * If every existing file at ``target_dir`` already matches
            the bundled bytes, return ``action="skipped"`` without
            writing.
          * Otherwise, write each bundled file via temp-then-rename
            (atomic on POSIX) and verify the final hash matches.

        Concurrent invocations are safe per file (each writer takes its
        own ``NamedTemporaryFile`` and ``os.replace`` is atomic); last
        writer wins, but every writer wrote identical bytes from the
        same bundled source, so the post-rename tree always matches the
        bundled hash.

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

        source_payload = _read_source_files(self._source_dir)
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

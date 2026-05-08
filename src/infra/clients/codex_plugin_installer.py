"""Idempotent installer for the bundled ``mala-safety`` Codex plugin (T015).

Copies the source-tree at ``plugins/codex/mala-safety/.codex-plugin/`` into
the user's Codex plugin directory (default
``~/.codex/plugins/mala-safety/.codex-plugin/``) so that Codex discovers
the plugin on every run. The installer is concurrent-safe (per-file
write-temp-then-rename) and idempotent (SHA-256 content comparison
per file).

Layout shipped (fixed, regardless of the Phase E spike outcome — the
installer is a straight copy with no path translation):

  plugins/codex/mala-safety/.codex-plugin/
    plugin.json   - Codex plugin manifest
    hooks.json    - PreToolUse command-hook registration
    .mcp.json     - mala-locking MCP launcher (T016 finalizes wire shape)

Companion ``CodexAgentProvider.install_prerequisites()`` (Phase E5)
spawns a Codex one-shot turn against the installed plugin and verifies
the bundled hook actually runs; the on-disk content guarantee here is
necessary but not sufficient (parity with ``AmpPluginInstaller`` →
``_run_selftest_subprocess`` split, which is the reference pattern).

Trusted-hash auto-trust (Phase E6, decision #16) is also computed here:
``installed_plugin_hash`` returns a 16-hex-char SHA-256 prefix of the
bundled source bytes (the manifest + hook config + mcp config combined).
The provider writes that value into Codex's hook-state file
(``~/.codex/hooks.toml`` via ``HookStateToml.trusted_hash`` per
``codex-rs/config/src/hook_config.rs``) so Codex loads the hook without
an interactive trust prompt; if the spike outcome is that interactive
trust is mandatory, the documented one-time prompt fallback applies.
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


def default_plugin_target_dir() -> Path:
    """Resolve the default Codex plugin directory at call time.

    Honors ``CODEX_HOME`` so tests pointing the env var at ``tmp_path``
    install into the redirected Codex home. The plan's open question on
    the exact discovery path (``~/.codex/plugins/`` vs. another
    location) is captured here; the installer is a straight copy with
    no path translation, so a different default — once the spike
    confirms it — is a one-line change.
    """
    return _resolve_codex_home() / "plugins" / PLUGIN_NAME / PLUGIN_DIRNAME


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

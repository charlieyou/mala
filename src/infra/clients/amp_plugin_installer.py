"""Idempotent installer for the bundled ``mala-safety.ts`` Amp plugin.

Copies ``plugins/amp/mala-safety.ts`` from the mala source distribution to
``~/.config/amp/plugins/`` so that ``amp --execute`` (with ``PLUGINS=all``)
loads it on every Amp coder run. The installer is concurrent-safe via
``write-temp-then-rename`` and idempotent via SHA-256 content comparison.

This task (T010) only guarantees the file is *present* on disk with the
expected bytes. The runtime fail-closed gate (T013) is what proves the
plugin actually loaded under ``PLUGINS=all`` — content-hash on disk alone
is insufficient under ``--dangerously-allow-all`` per
``plans/2026-04-29-amp-provider-plan.md#L171``.

The plugin's own version marker (emitted on ``session.start``) is a
SHA-256 prefix of its source bytes; :meth:`AmpPluginInstaller.installed_plugin_hash`
returns that same 16-char prefix so T013 can compare the marker payload
against the just-installed file.
"""

from __future__ import annotations

import errno
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

PLUGIN_FILENAME = "mala-safety.ts"
"""Bundled plugin filename. Must match the file shipped in ``plugins/amp/``
and the filename Amp loads from ``~/.config/amp/plugins/``."""

DEFAULT_PLUGIN_DIR = Path("~/.config/amp/plugins").expanduser()
"""Standard Amp global plugin directory (per Amp's plugin API)."""

_VERSION_MARKER_HEX_CHARS = 16
"""Length of the hex prefix returned by :meth:`installed_plugin_hash`. Must
match ``plugins/amp/mala-safety.ts::computeOwnVersionHash`` (which slices
the SHA-256 to 16 hex chars before emitting the marker)."""

_TEMP_PREFIX = ".mala-safety."
_TEMP_SUFFIX = ".ts.tmp"


InstallAction = Literal["wrote", "skipped", "replaced"]


@dataclass(frozen=True)
class InstallResult:
    """Outcome of one :meth:`AmpPluginInstaller.install` call.

    Consumed by T013's self-test wiring; ``plugin_hash`` is the 16-hex-char
    version prefix expected to appear in the plugin's ``session.start``
    sentinel marker.
    """

    target_path: Path
    plugin_hash: str
    action: InstallAction


class AmpPluginInstallError(RuntimeError):
    """Raised when the installer cannot guarantee the installed plugin file."""


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bundled_source_path() -> Path:
    return Path(__file__).resolve().parents[3] / "plugins" / "amp" / PLUGIN_FILENAME


class AmpPluginInstaller:
    """Idempotent, concurrent-safe installer for ``mala-safety.ts``.

    The installer is constructed once per orchestrator run and invoked by
    ``AmpAgentProvider.install_prerequisites()`` (T013). The default
    ``source_path`` resolves the bundled plugin file inside the mala
    source tree; tests pass a custom path to drive variants.
    """

    def __init__(self, source_path: Path | None = None) -> None:
        self._source_path: Path = (
            source_path if source_path is not None else _bundled_source_path()
        )

    @property
    def source_path(self) -> Path:
        """Path to the bundled plugin file inside the mala source tree."""
        return self._source_path

    def installed_plugin_hash(self) -> str:
        """Return the 16-hex-char version prefix the plugin emits at runtime.

        Computed from the **bundled** source bytes (not the on-disk copy);
        a successful :meth:`install` guarantees the on-disk bytes match.
        T013's runtime self-test compares this to the ``version`` field of
        the plugin's ``session.start`` sentinel marker.
        """
        return _sha256_hex(self._read_source_bytes())[:_VERSION_MARKER_HEX_CHARS]

    def install(self, target_dir: Path | None = None) -> InstallResult:
        """Install ``mala-safety.ts`` into ``target_dir``, idempotently.

        Behavior:
          * ``mkdir -p target_dir`` (first-time Amp installs may not have it).
          * If an existing copy at ``target_dir/mala-safety.ts`` already has
            the bundled SHA-256, return ``action="skipped"`` without writing.
          * Otherwise, write the bundled bytes to a unique temp file in
            ``target_dir`` and atomically ``os.replace`` it onto the target,
            then re-read and verify the SHA-256.

        Concurrent invocations are safe: every caller writes its own temp
        file (``tempfile.NamedTemporaryFile`` guarantees uniqueness inside
        ``target_dir``) and ``os.replace`` is atomic on POSIX. Last writer
        wins, but every writer wrote identical bytes from the same bundled
        source, so the post-rename file always matches the bundled hash.

        Args:
            target_dir: Directory to install into. Defaults to
                ``~/.config/amp/plugins`` (Amp's standard plugin directory).

        Returns:
            :class:`InstallResult` describing the action taken plus the
            16-hex-char version hash.

        Raises:
            AmpPluginInstallError: If the target directory cannot be created
                or written to, the bundled source is missing, or the
                post-rename hash does not match the bundled hash.
        """
        target_dir = target_dir if target_dir is not None else DEFAULT_PLUGIN_DIR
        target_path = target_dir / PLUGIN_FILENAME

        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AmpPluginInstallError(
                f"Cannot create or access Amp plugin directory {target_dir}: {exc}"
            ) from exc

        source_bytes = self._read_source_bytes()
        source_hash = _sha256_hex(source_bytes)
        version_hash = source_hash[:_VERSION_MARKER_HEX_CHARS]

        existing_hash = _read_hash_if_exists(target_path)
        if existing_hash == source_hash:
            return InstallResult(
                target_path=target_path,
                plugin_hash=version_hash,
                action="skipped",
            )

        action: InstallAction = "replaced" if existing_hash is not None else "wrote"
        self._atomic_write(target_dir, target_path, source_bytes)

        try:
            installed_bytes = target_path.read_bytes()
        except OSError as exc:
            raise AmpPluginInstallError(
                f"Cannot read installed plugin at {target_path} for hash verification: {exc}"
            ) from exc
        installed_hash = _sha256_hex(installed_bytes)
        if installed_hash != source_hash:
            raise AmpPluginInstallError(
                "Installed plugin hash mismatch after rename: "
                f"expected {source_hash}, got {installed_hash}. "
                "Concurrent writer or filesystem corruption suspected."
            )

        return InstallResult(
            target_path=target_path,
            plugin_hash=version_hash,
            action=action,
        )

    def _read_source_bytes(self) -> bytes:
        try:
            return self._source_path.read_bytes()
        except FileNotFoundError as exc:
            raise AmpPluginInstallError(
                f"Bundled Amp plugin missing at {self._source_path}; "
                "the mala source tree appears incomplete."
            ) from exc
        except OSError as exc:
            raise AmpPluginInstallError(
                f"Cannot read bundled plugin at {self._source_path}: {exc}"
            ) from exc

    @staticmethod
    def _atomic_write(target_dir: Path, target_path: Path, payload: bytes) -> None:
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=target_dir,
                prefix=_TEMP_PREFIX,
                suffix=_TEMP_SUFFIX,
                delete=False,
            ) as tmp:
                tmp.write(payload)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, target_path)
            tmp_path = None
        except OSError as exc:
            raise AmpPluginInstallError(
                f"Cannot install Amp plugin at {target_path}: {exc}"
            ) from exc
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except OSError as cleanup_exc:
                    if cleanup_exc.errno != errno.ENOENT:
                        # Best-effort cleanup; do not mask the original error.
                        pass


def _read_hash_if_exists(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError:
        return None
    return _sha256_hex(data)

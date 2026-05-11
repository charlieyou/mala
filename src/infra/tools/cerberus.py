"""Cerberus plugin discovery utilities.

Provides functions for locating cerberus plugin binaries from Claude's
installed plugins.
"""

from __future__ import annotations

import json
from pathlib import Path


def _is_v1_version(version: object) -> bool:
    """Return True when a plugin version is in the supported 1.x line."""
    version_text = str(version).strip()
    if version_text.startswith("v"):
        version_text = version_text[1:]
    return version_text == "1" or version_text.startswith("1.")


def _read_package_version(plugin_root: Path) -> object | None:
    """Read a plugin version from package.json when present."""
    package_file = plugin_root / "package.json"
    if not package_file.exists():
        return None

    try:
        package_data = json.loads(package_file.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    if not isinstance(package_data, dict):
        return None
    return package_data.get("version")


def _is_v1_install_path(install_path: Path) -> bool:
    """Return True when the install path can be identified as Cerberus 1.x."""
    package_version = _read_package_version(install_path)
    if package_version is not None:
        return _is_v1_version(package_version)
    return _is_v1_version(install_path.name)


def find_cerberus_bin_path(claude_config_dir: Path) -> Path | None:
    """Find the cerberus plugin bin directory from Claude's installed plugins.

    Looks up the cerberus plugin installation path from Claude's
    installed_plugins.json (v2 schema) and returns the path to its
    bin/ directory. Falls back to known plugin locations if metadata is missing.

    Args:
        claude_config_dir: Path to Claude config directory (typically ~/.claude).

    Returns:
        Path to cerberus bin directory, or None if not found.
    """
    plugins_root = claude_config_dir / "plugins"
    plugins_file = plugins_root / "installed_plugins.json"

    def _iter_plugin_entries(data: object) -> list[tuple[str, object]]:
        if isinstance(data, dict):
            plugins = dict.get(data, "plugins")
            if isinstance(plugins, dict):
                return [(str(key), installs) for key, installs in plugins.items()]
        return []

    if plugins_file.exists():
        try:
            data = json.loads(plugins_file.read_text())
            # Look for cerberus plugin (key format: "cerberus@cerberus" or similar)
            for key, installs in _iter_plugin_entries(data):
                if "cerberus" in str(key).lower() and isinstance(installs, list):
                    for install in installs:
                        if not isinstance(install, dict):
                            continue
                        install_path = dict.get(install, "installPath")
                        version = dict.get(install, "version")
                        if version is not None:
                            is_v1_install = _is_v1_version(version)
                        elif install_path:
                            is_v1_install = _is_v1_install_path(Path(install_path))
                        else:
                            is_v1_install = False

                        if install_path and is_v1_install:
                            bin_path = Path(install_path) / "bin"
                            if bin_path.exists():
                                return bin_path
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # Fallback to known locations if installed_plugins.json is missing or stale.
    marketplace_bin = plugins_root / "marketplaces" / "cerberus" / "bin"
    if marketplace_bin.exists() and _is_v1_install_path(marketplace_bin.parent):
        return marketplace_bin

    cache_root = plugins_root / "cache" / "cerberus" / "cerberus"
    if cache_root.exists():
        candidates = sorted(
            (path for path in cache_root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            if not _is_v1_install_path(candidate):
                continue
            bin_path = candidate / "bin"
            if bin_path.exists():
                return bin_path

    return None

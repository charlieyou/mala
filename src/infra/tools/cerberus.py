"""Cerberus plugin discovery utilities.

Provides functions for locating cerberus plugin binaries from Claude's
installed plugins.
"""

from __future__ import annotations

import json
from pathlib import Path


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
                return list(plugins.items())
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
                        if install_path:
                            bin_path = Path(install_path) / "bin"
                            if bin_path.exists():
                                return bin_path
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

    # Fallback to known locations if installed_plugins.json is missing or stale.
    marketplace_bin = plugins_root / "marketplaces" / "cerberus" / "bin"
    if marketplace_bin.exists():
        return marketplace_bin

    cache_root = plugins_root / "cache" / "cerberus" / "cerberus"
    if cache_root.exists():
        candidates = sorted(
            (path for path in cache_root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            bin_path = candidate / "bin"
            if bin_path.exists():
                return bin_path

    return None

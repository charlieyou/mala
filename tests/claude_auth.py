"""Shared helpers for checking Claude Code CLI authentication in tests."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def is_claude_cli_available() -> bool:
    """Return True if the Claude Code CLI is installed and callable."""
    try:
        result = subprocess.run(
            ["claude", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _credentials_path() -> Path | None:
    """Return the credentials path Claude CLI will use, if any."""
    test_config_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if test_config_dir:
        return Path(test_config_dir) / ".credentials.json"
    return Path.home() / ".claude" / ".credentials.json"


def _extract_expiry_ms(payload: dict[str, Any]) -> float | None:
    """Extract an OAuth expiry timestamp (ms since epoch) if present."""
    candidates: list[dict[str, Any]] = [payload]
    nested = payload.get("claudeAiOauth")
    if isinstance(nested, dict):
        candidates.insert(0, nested)

    for candidate in candidates:
        for key in ("expiresAt", "expires_at", "expires", "expiry", "expiration"):
            value = candidate.get(key)
            if value is None:
                continue
            try:
                expiry = float(value)
            except (TypeError, ValueError):
                continue
            # Heuristic: seconds vs milliseconds.
            if expiry < 1e12:
                expiry *= 1000.0
            return expiry
    return None


def has_valid_oauth_credentials() -> bool:
    """Return True if OAuth credentials exist and are not expired."""
    creds_path = _credentials_path()
    if creds_path is None or not creds_path.exists():
        return False

    try:
        payload = json.loads(creds_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False

    expiry_ms = _extract_expiry_ms(payload)
    if expiry_ms is None:
        # If expiry isn't present, assume credentials are valid.
        return True

    # Require at least 60s of validity to avoid mid-test expiry.
    if expiry_ms > (time.time() * 1000.0 + 60_000.0):
        return True

    # Expired access tokens can often be refreshed by the CLI.
    oauth = payload.get("claudeAiOauth")
    if isinstance(oauth, dict) and oauth.get("refreshToken"):
        return True

    return False

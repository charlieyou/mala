"""Environment configuration and loading for mala.

Centralizes config paths and dotenv loading. Import this module early
to ensure environment variables are set before Braintrust setup.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# User config directory (stores .env, runs, etc.)
USER_CONFIG_DIR = Path.home() / ".config" / "mala"

# Run metadata directory
RUNS_DIR = USER_CONFIG_DIR / "runs"

# Lock directory for multi-agent coordination
# Can be overridden via MALA_LOCK_DIR environment variable
LOCK_DIR = Path(os.environ.get("MALA_LOCK_DIR", "/tmp/mala-locks"))

# Lock scripts directory (relative to this file: src/tools/env.py -> src/scripts/)
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def load_user_env() -> None:
    """Load environment from user config directory.

    Loads ${USER_CONFIG_DIR}/.env (typically ~/.config/mala/.env).
    Call this early for Braintrust API key setup before SDK imports.
    """
    load_dotenv(dotenv_path=USER_CONFIG_DIR / ".env")


def load_env(repo_path: Path | None = None) -> None:
    """Load environment from user config and optionally repo.

    NOTE: The repo_path parameter is for TESTING ONLY. Production code should
    only use load_user_env() to load from ~/.config/mala/.env.

    Args:
        repo_path: Optional repository path. If provided, loads <repo_path>/.env
                   with override=True. FOR TESTING ONLY.
    """
    load_user_env()
    if repo_path is not None:
        load_dotenv(dotenv_path=repo_path / ".env", override=True)


def encode_repo_path(repo_path: Path) -> str:
    """Encode repo path to match Claude SDK project directory naming.

    Claude SDK stores session logs in ~/.claude/projects/{encoded-path}/.
    The encoding replaces path separators with hyphens and prefixes with hyphen.

    Example: /home/cyou/mala -> -home-cyou-mala

    Args:
        repo_path: Repository path to encode.

    Returns:
        Encoded path string suitable for Claude projects directory.
    """
    resolved = repo_path.resolve()
    # Skip root and join parts with hyphens, prefix with hyphen
    return "-" + "-".join(resolved.parts[1:])


def get_claude_log_path(repo_path: Path, session_id: str) -> Path:
    """Get path to Claude SDK's session log file.

    Claude SDK writes session logs to:
    ~/.claude/projects/{encoded-repo-path}/{session_id}.jsonl

    Args:
        repo_path: Repository path the session was run in.
        session_id: Claude SDK session ID (UUID from ResultMessage).

    Returns:
        Path to the JSONL log file.
    """
    encoded = encode_repo_path(repo_path)
    return Path.home() / ".claude" / "projects" / encoded / f"{session_id}.jsonl"

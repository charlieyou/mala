"""Environment configuration and loading for mala.

Centralizes config paths and dotenv loading. Import this module early
to ensure environment variables are set before Braintrust setup.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# User config directory (stores .env, logs, etc.)
USER_CONFIG_DIR = Path.home() / ".config" / "mala"

# JSONL log directory
JSONL_LOG_DIR = USER_CONFIG_DIR / "logs"

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

"""CLI support module for re-exporting infra utilities.

This module is part of the orchestration layer and provides CLI-safe access
to infrastructure utilities. CLI imports from this module instead of directly
from infra layer modules.

This maintains the architectural boundary: CLI -> orchestration -> infra.
"""

from __future__ import annotations

# Environment configuration (from src.infra.tools.env)
from src.infra.tools.env import (
    USER_CONFIG_DIR,
    get_runs_dir,
    load_user_env,
)

# BeadsClient (from src.beads_client)
from src.beads_client import BeadsClient

# Locking utilities (from src.infra.tools.locking)
from src.infra.tools.locking import get_lock_dir

# Run metadata
from src.infra.io.log_output.run_metadata import (
    get_running_instances,
    get_running_instances_for_dir,
)

# Console utilities
from src.infra.io.log_output.console import (
    Colors,
    log,
    set_verbose,
)

__all__ = [
    "USER_CONFIG_DIR",
    "BeadsClient",
    "Colors",
    "get_lock_dir",
    "get_running_instances",
    "get_running_instances_for_dir",
    "get_runs_dir",
    "load_user_env",
    "log",
    "set_verbose",
]

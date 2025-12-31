"""CLI support module for re-exporting infra utilities.

This module is part of the orchestration layer and provides CLI-safe access
to infrastructure utilities. CLI imports from this module instead of directly
from infra layer modules (src.tools, src.log_output).

This maintains the architectural boundary: CLI -> orchestration -> infra.
"""

from __future__ import annotations

# Environment configuration (from src.tools.env)
from .tools.env import (
    USER_CONFIG_DIR,
    SCRIPTS_DIR,
    get_runs_dir,
    load_user_env,
)

# Locking utilities (from src.tools.locking)
from .tools.locking import get_lock_dir

# Run metadata (from src.log_output.run_metadata)
from .log_output.run_metadata import (
    get_running_instances,
    get_running_instances_for_dir,
)

# Console utilities (from src.log_output.console)
from .log_output.console import (
    Colors,
    log,
    set_verbose,
)

__all__ = [
    "SCRIPTS_DIR",
    "USER_CONFIG_DIR",
    "Colors",
    "get_lock_dir",
    "get_running_instances",
    "get_running_instances_for_dir",
    "get_runs_dir",
    "load_user_env",
    "log",
    "set_verbose",
]

"""Backward-compatibility shim for src.log_output package.

This module re-exports all public symbols from src.infra.io.log_output.
New code should import directly from src.infra.io.log_output.

Sub-module imports like `from src.log_output.console import ...` are supported
via __path__ manipulation.
"""

import sys

from src.infra.io.log_output import *  # noqa: F403

# Enable sub-module imports: from src.log_output.console import ...
from src.infra.io import log_output as _new_log_output

sys.modules[__name__].__path__ = _new_log_output.__path__  # type: ignore[attr-defined]

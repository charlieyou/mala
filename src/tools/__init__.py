"""Backward-compatibility shim for src.tools package.

This module re-exports all public symbols from src.infra.tools.
New code should import directly from src.infra.tools.

Sub-module imports like `from src.tools.locking import ...` are supported
via __path__ manipulation.
"""

import sys

from src.infra.tools import *  # noqa: F403

# Enable sub-module imports: from src.tools.locking import ...
from src.infra import tools as _new_tools

sys.modules[__name__].__path__ = _new_tools.__path__  # type: ignore[attr-defined]

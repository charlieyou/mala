"""Backward-compatibility shim for src.hooks package.

This module re-exports all public symbols from src.infra.hooks.
New code should import directly from src.infra.hooks.

Sub-module imports like `from src.hooks.locking import ...` are supported
via __path__ manipulation.
"""

import sys

from src.infra.hooks import *  # noqa: F403

# Enable sub-module imports: from src.hooks.locking import ...
from src.infra import hooks as _new_hooks

sys.modules[__name__].__path__ = _new_hooks.__path__  # type: ignore[attr-defined]

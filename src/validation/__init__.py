"""Backward-compatibility shim for src.validation package.

This module re-exports all public symbols from src.domain.validation.
New code should import directly from src.domain.validation.

Sub-module imports like `from src.validation.result import ...` are supported
via __path__ manipulation.
"""

import sys

from src.domain.validation import *  # noqa: F403

# Enable sub-module imports: from src.validation.spec import ...
from src.domain import validation as _new_validation

sys.modules[__name__].__path__ = _new_validation.__path__  # type: ignore[attr-defined]

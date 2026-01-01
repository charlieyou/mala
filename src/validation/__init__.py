"""Backward-compatibility shim for src.validation package.

This module re-exports all public symbols from src.domain.validation.
New code should import directly from src.domain.validation.

Sub-module imports like `from src.validation.spec import ...` are supported
via module aliasing in sys.modules.
"""

import sys

from src.domain.validation import *  # noqa: F403

# Enable sub-module imports by aliasing domain modules
# This ensures that imports like `from src.validation.spec import CommandKind`
# return the exact same objects as `from src.domain.validation.spec import CommandKind`
from src.domain import validation as _new_validation

# Alias submodules so that src.validation.X and src.domain.validation.X
# return the same module object (important for enum identity)
_submodules = [
    "coverage",
    "e2e",
    "helpers",
    "lint_cache",
    "result",
    "runner",
    "spec",
    "spec_executor",
    "spec_result_builder",
    "spec_runner",
    "spec_workspace",
    "worktree",
]

for _name in _submodules:
    _domain_module = getattr(_new_validation, _name, None)
    if _domain_module is not None:
        sys.modules[f"src.validation.{_name}"] = _domain_module

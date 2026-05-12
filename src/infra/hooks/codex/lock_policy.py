"""Pure lock-policy decision logic for the Codex pre-tool-use hook.

Extracted from ``src.infra.hooks.codex_pre_tool_use`` (plan B.3 sub-commit
T_B10). The hook delegates to this module for the lock-ownership
decision that gates every file-edit / shell-write path. The module
returns the deny *reason* (string) rather than the hook's full deny
dict so the policy stays independent of the hook's output schema; the
hook wraps the reason into its ``permissionDecision`` shape.

Branches covered:

* **unresolved sentinel** — the parseable-but-unresolved write target
  (plan L846) always denies because locking the sentinel string would
  be a trivial bypass.
* **substitution placeholder** — tokens carrying the
  ``__MALA_CMDSUB_LITERAL__`` / ``__MALA_ANSI_C_LITERAL__`` markers
  always deny because bash will substitute a different real path at
  runtime.
* **shell expansion metachar** — glob / brace / parameter / tilde
  characters always deny for the same runtime-divergence reason.
* **lock holder lookup** — ``get_lock_holder`` returns ``None`` (no
  active lock) → deny; returns a holder different from ``agent_id``
  → deny with wrong-owner message; returns ``agent_id`` → allow.

The :func:`_resolve_against_cwd` helper resolves relative targets
against the payload ``cwd`` so the lock-key derivation path matches
the path the shell would actually write to.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

from .shell_parser import _ANSI_C_PLACEHOLDER, _CMDSUB_PLACEHOLDER
from .write_targets import _UNRESOLVED_SENTINEL

if TYPE_CHECKING:
    from collections.abc import Iterable


__all__ = [
    "_MSG_ENV_MISSING",
    "_SHELL_EXPANSION_RE",
    "_check_lock",
    "_gate_write_targets",
    "_msg_no_lock",
    "_msg_wrong_owner",
    "_resolve_against_cwd",
]


# Deny-message templates (plan L854-L860). These strings are byte-identical
# across the apply_patch branch and the bash write-path branch so reviewers
# can grep for the exact wording.
def _msg_no_lock(rel: str, me: str) -> str:
    return f"path '{rel}' has no active lock for agent '{me}'"


def _msg_wrong_owner(rel: str, other: str, me: str) -> str:
    return f"path '{rel}' is locked by agent '{other}'; current agent is '{me}'"


_MSG_ENV_MISSING = (
    "MALA_AGENT_ID/MALA_LOCK_DIR/MALA_REPO_NAMESPACE missing; refusing to evaluate lock"
)


# Shell glob, brace-expansion, and parameter-expansion metacharacters.
# When any of these appear in an extracted write-target, the actual
# files bash will operate on cannot be statically resolved — and the
# agent could acquire a lock on the literal pattern (``$cwd/*.py`` or
# ``~/foo`` or ``$VAR``) while bash expands to a different file at
# runtime. Treat such targets as unresolved (always-deny).
#
# Includes: glob (``*``, ``?``, ``[``, ``]``), brace expansion (``{``,
# ``}``), parameter/command substitution (``$``), and tilde expansion
# (``~``). Backticks (`` ` ``) are already extracted as
# command-substitution bodies before the lock-check; any backtick
# reaching this gate has been preserved inside single quotes (literal)
# and is not a write surface.
_SHELL_EXPANSION_RE = re.compile(r"[*?\[\]{}$~]")
_SHELL_EXPANSION_EXCEPT_DOLLAR_RE = re.compile(r"[*?\[\]{}~]")


def _resolve_against_cwd(target: str, cwd: str | None) -> str:
    """Resolve a relative ``target`` against the payload ``cwd``.

    Shell commands (and patch headers) write paths relative to the
    process cwd, not to the repo namespace root. Without this resolution,
    a hook input with ``cwd=/repo/pkg`` and ``command='touch generated.py'``
    would lock-check ``/repo/generated.py`` (namespace-root) and let an
    unowned write to ``/repo/pkg/generated.py`` slip past whenever the
    agent happens to hold the root-level lock.
    """
    if not target or target == _UNRESOLVED_SENTINEL:
        return target
    p = Path(target)
    if p.is_absolute():
        return target
    if not cwd:
        return target
    return str(Path(cwd) / target)


def _check_lock(
    target: str,
    agent_id: str,
    lock_dir: str,
    repo_namespace: str,
    cwd: str | None,
    *,
    allow_literal_dollar: bool = False,
) -> str | None:
    """Look up the lock holder for ``target`` and return the deny reason if blocked.

    Returns ``None`` when the agent holds the lock (i.e. the write is
    permitted), otherwise a deny-reason string the caller wraps into a
    hook ``permissionDecision`` response.

    Relative ``target`` paths are resolved against the payload ``cwd``
    before lock-key derivation so the lock-check path matches the path
    the shell would actually write to. The unresolved-target sentinel
    always denies (plan L846: deny on parseable-but-unresolved write
    expressions). Targets containing shell glob, brace, parameter, or
    tilde expansion metacharacters also always deny — bash expands
    them at runtime, and locking the literal pattern would be a
    trivial bypass. Tokens that contain a command-substitution
    placeholder (the ``__MALA_CMDSUB_LITERAL__`` / ``__MALA_ANSI_C_
    LITERAL__`` markers from preprocessing) also deny because the
    real path is whatever bash produces from the substitution.
    """
    if target == _UNRESOLVED_SENTINEL:
        return _msg_no_lock(target, agent_id)
    if _CMDSUB_PLACEHOLDER in target or _ANSI_C_PLACEHOLDER in target:
        return _msg_no_lock(target, agent_id)
    expansion_re = (
        _SHELL_EXPANSION_EXCEPT_DOLLAR_RE
        if allow_literal_dollar
        else _SHELL_EXPANSION_RE
    )
    if expansion_re.search(target):
        return _msg_no_lock(target, agent_id)
    # The locking module reads ``MALA_LOCK_DIR`` from ``os.environ`` at
    # call time. Reaffirm it here so the lock-key derivation always
    # matches the value the hook resolved from the env, even if a
    # caller later mutates the env or imports a stale resolver.
    os.environ["MALA_LOCK_DIR"] = lock_dir
    # Imported lazily so the module-level import does not pin the lock dir
    # before the env is resolved.
    from ...tools.locking import get_lock_holder

    resolved = _resolve_against_cwd(target, cwd)
    holder = get_lock_holder(resolved, repo_namespace=repo_namespace)
    # Deny messages keep the original (caller-visible) target string for
    # readability, even when the lock-key was computed from the cwd-
    # resolved absolute path.
    if holder is None:
        return _msg_no_lock(target, agent_id)
    if holder != agent_id:
        return _msg_wrong_owner(target, holder, agent_id)
    return None


def _gate_write_targets(
    targets: Iterable[str],
    *,
    agent_id: str,
    lock_dir: str,
    repo_namespace: str,
    cwd: str | None,
    allow_literal_dollar: bool = False,
) -> str | None:
    """Run the lock check on each target; return the first deny reason."""
    for target in targets:
        reason = _check_lock(
            target,
            agent_id,
            lock_dir,
            repo_namespace,
            cwd,
            allow_literal_dollar=allow_literal_dollar,
        )
        if reason is not None:
            return reason
    return None

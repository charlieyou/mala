"""Pure destructive-command classification for the Codex pre-tool-use hook.

Extracted from ``src.infra.hooks.codex_pre_tool_use`` (plan B.3 sub-commit
T_B11). The hook delegates to this module for the dangerous-pattern /
destructive-git classification that gates every shell-tool call. The
module returns the matched pattern label (string) rather than the hook's
full deny dict so the policy stays independent of the hook's output
schema; the hook wraps the pattern label into its ``permissionDecision``
shape via :func:`_msg_dangerous`.

Coverage mirrors :func:`src.infra.hooks.dangerous_commands.block_dangerous_commands`
so AC #9 ("dangerous shell commands are denied") holds for the Codex
sandbox=danger-full-access path:

* **substring scan** — every pattern in ``DANGEROUS_PATTERNS`` and
  ``DESTRUCTIVE_GIT_PATTERNS`` is matched as a literal substring.
* **token-walk** — ``shlex``-tokenized commands are scanned for
  destructive ``git`` subcommands even when global options
  (``git -C dir stash``) shift the subcommand position past the
  substring matcher's window.
* **atomic add+commit gate** — a ``git commit`` with no preceding
  ``git add`` in the same command is denied.
* **substitution recursion** — backtick / ``$()`` bodies are inspected
  for hidden destructive git invocations (``echo `git -C /tmp stash```
  runs the inner stash for real when bash evaluates the substitution).

The end-to-end byte-identical contract for ``codex_pre_tool_use`` is
preserved by the golden corpus harness; this module's focused unit tests
pin every classification rule so a regression here is caught without
rerunning the whole hook.
"""

from __future__ import annotations

import shlex

from ..dangerous_commands import (
    DANGEROUS_PATTERNS,
    DESTRUCTIVE_GIT_PATTERNS,
)
from .shell_parser import (
    _extract_command_substitutions,
    _normalize_separators,
    _strip_shell_comments,
)
from .write_targets import (
    _command_basename_str,
    _find_git_subcommand_info,
)


__all__ = [
    "_detect_dangerous_pattern",
    "_detect_dangerous_pattern_recursive",
    "_detect_destructive_git_in_subtokens",
    "_msg_dangerous",
]


def _msg_dangerous(pattern: str) -> str:
    return f"command matches dangerous pattern: {pattern}"


def _detect_destructive_git_in_subtokens(toks: list[str]) -> str | None:
    """Detect a destructive git invocation starting at ``toks[0] == git``.

    Used by ``_detect_dangerous_pattern`` to catch destructive git
    subcommands when global options shift the subcommand position.
    Without tokenization, ``git -C dir stash`` does not contain the
    literal substring ``git stash`` and the substring-only detector
    misses the destructive call.
    """
    if not toks or _command_basename_str(toks[0]) != "git":
        return None
    sub_idx, _ = _find_git_subcommand_info(toks)
    if sub_idx is None:
        return None
    sub = toks[sub_idx]
    rest = toks[sub_idx + 1 :]
    rest_set = set(rest)

    if sub == "stash":
        return "git stash"
    if sub == "rebase":
        return "git rebase"
    if sub == "restore":
        return "git restore"
    if sub == "reset":
        if "--hard" in rest_set:
            return "git reset --hard"
        if "--mixed" in rest_set:
            return "git reset --mixed"
        if "--soft" in rest_set:
            return "git reset --soft"
        if rest and rest[0] == "HEAD":
            return "git reset HEAD"
    if sub == "checkout":
        if "--" in rest_set:
            return "git checkout --"
        if "-f" in rest_set or "--force" in rest_set:
            return "git checkout -f"
    if sub == "clean":
        for tok in rest:
            if tok.startswith("-") and not tok.startswith("--") and "f" in tok[1:]:
                return "git clean -f"
    if sub == "branch":
        if "-D" in rest_set:
            return "git branch -D"
        # ``git branch -d -f`` / ``--delete --force`` (and bundled
        # short forms ``-df`` / ``-fd``) is the explicit alternative
        # to ``-D`` and is equally destructive.
        has_delete = "-d" in rest_set or "--delete" in rest_set
        has_force = "-f" in rest_set or "--force" in rest_set
        if has_delete and has_force:
            return "git branch -d -f"
        for tok in rest:
            if (
                tok.startswith("-")
                and not tok.startswith("--")
                and len(tok) >= 2
                and "d" in tok[1:]
                and "f" in tok[1:]
            ):
                return "git branch -d -f"
    if sub == "commit":
        if "--amend" in rest_set:
            return "git commit --amend"
        # ``git commit -a`` / ``--all`` / bundled short flags
        # containing ``a`` (mirrors block_dangerous_commands and the
        # original substring check in _detect_dangerous_pattern).
        if "--all" in rest_set:
            return "git commit -a/--all"
        for tok in rest:
            if (
                tok.startswith("-")
                and not tok.startswith("--")
                and len(tok) >= 2
                and "a" in tok[1:]
            ):
                return "git commit -a/--all"
    if sub == "push":
        # ``--force-with-lease`` is the documented safer alternative
        # and is allowed; only deny on the bare ``--force`` / ``-f``
        # forms.
        if "--force-with-lease" not in rest_set:
            if "--force" in rest_set or "-f" in rest_set:
                return "git push --force"
            for tok in rest:
                if (
                    tok.startswith("-")
                    and not tok.startswith("--")
                    and len(tok) >= 2
                    and "f" in tok[1:]
                ):
                    return "git push --force"
    if sub in {"merge", "cherry-pick"}:
        if "--abort" in rest_set:
            return f"git {sub} --abort"
    if sub == "worktree":
        if rest and rest[0] == "remove":
            return "git worktree remove"
    if sub == "submodule":
        if rest and rest[0] == "deinit" and "-f" in rest_set:
            return "git submodule deinit -f"

    return None


def _detect_dangerous_pattern(command: str) -> str | None:
    """Mirror :func:`block_dangerous_commands` for parity with the Claude hook.

    Returns the matched pattern label (for ``_msg_dangerous``) or ``None``.
    Coverage must include every gate the Claude path applies, so AC #9
    ("dangerous shell commands are denied") holds for Codex too: dangerous
    patterns, destructive git patterns (with global-option awareness),
    ``git commit -a``/``--all``, atomic-add-commit requirement, and
    ``git push --force``.
    """
    for pattern in DANGEROUS_PATTERNS:
        if pattern in command:
            return pattern
    for pattern in DESTRUCTIVE_GIT_PATTERNS:
        if pattern in command:
            return pattern

    # Tokenization-based git check: catches destructive subcommands
    # (including commit -a, push --force, etc.) even when global
    # options shift the subcommand position past the substring
    # matcher's window. ``comments=True`` matches bash's treatment of
    # ``#`` as comment marker so ``cmd # "`` doesn't ValueError out.
    #
    # ``_normalize_separators`` is applied first for two reasons:
    #
    # 1. Control operators (``;``/``&&``/``||``/``|``/``&``/``\n``)
    #    need surrounding whitespace before ``shlex.split`` will treat
    #    them as separators. Without normalization, ``git add f;git
    #    commit -a`` tokenizes as ``['git', 'add', 'f;git', 'commit',
    #    '-a']`` — the second ``git`` is glued to the ``;`` and the
    #    token-walk misses the destructive commit.
    # 2. ANSI-C quoting (``$'...'``) is replaced with a placeholder
    #    so an escaped single-quote like ``$'\\''`` does not raise
    #    ValueError, which would silently bypass every token-based
    #    destructive-git check below.
    try:
        toks = shlex.split(_normalize_separators(command), posix=True, comments=True)
    except ValueError:
        toks = []
    saw_git_add = False
    for i, tok in enumerate(toks):
        if _command_basename_str(tok) == "git":
            pattern = _detect_destructive_git_in_subtokens(toks[i:])
            if pattern is not None:
                return pattern
            # Track whether ``git add`` precedes any ``git commit``
            # for the atomic-add-commit check below.
            sub_idx, _ = _find_git_subcommand_info(toks[i:])
            if sub_idx is not None:
                sub = toks[i + sub_idx]
                if sub == "add":
                    saw_git_add = True
                elif sub == "commit" and not saw_git_add:
                    return "git commit without prior git add"

    return None


def _detect_dangerous_pattern_recursive(command: str) -> str | None:
    """Run dangerous-pattern detection on ``command`` AND every nested
    substitution body (backticks, ``$()``).

    A destructive git invocation hidden inside a backtick body — e.g.
    ``echo `git -C /tmp stash``` — runs for real when bash evaluates
    the substitution, so the dangerous-cmd gate must inspect the
    body too. Without this recursion the outer token-walk only sees
    the placeholder ``__MALA_CMDSUB_LITERAL__`` and the destructive
    git command silently slips through.
    """
    seen: set[str] = set()
    # Strip comments at every level so commented-out destructive
    # commands (``# git -C /tmp stash``) are not flagged.
    pending = [_strip_shell_comments(command)]
    while pending:
        body = pending.pop()
        if body in seen:
            continue
        seen.add(body)
        pattern = _detect_dangerous_pattern(body)
        if pattern is not None:
            return pattern
        try:
            _, sub_bodies = _extract_command_substitutions(_strip_shell_comments(body))
        except Exception:
            continue
        pending.extend(sub_bodies)
    return None

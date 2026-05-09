"""``mala-codex-pre-tool-use`` hook entry-point (Phase E, T014).

Reads a ``pre-tool-use.command.input.schema.json`` payload from stdin,
enforces lock-ownership / dangerous-command / disallowed-tool / shell
write-path policies, and writes the Codex hook result JSON to stdout.

This is the single safety gate when Codex runs with
``sandbox=danger-full-access`` + ``approval_policy=never``. The
shell-command write-path heuristic closes a gap that an ``apply_patch``-
only lock-enforcement would leave open: without it, a turn could legally
``printf X > file``, ``sed -i ...``, ``cp src dst``, etc. and silently
mutate unlocked or wrong-owner paths.

Residual obfuscation (e.g., base64-decoded ``python -c "$(echo ... | base64 -d)"``,
indirection through env vars, dynamically-built filenames) is acknowledged
as a defense-in-depth limitation; tighter enforcement is follow-up work.

See ``plans/2026-05-07-codex-provider-plan.md`` (L778-L924) for the full
hook contract; the heuristic table at L842-L852 governs the bash branch.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .dangerous_commands import (
    BASH_TOOL_NAMES,
    DANGEROUS_PATTERNS,
    DESTRUCTIVE_GIT_PATTERNS,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


__all__ = [
    "SELFTEST_MARKER_DIR_ENV",
    "SELFTEST_MARKER_ENV",
    "SELFTEST_TARGET_EVENT_ENV",
    "decide",
    "main",
]


# Codex shell tool name candidates. ``bash`` is the public Codex tool name;
# ``local-shell`` is the internal SDK name retained for compatibility with
# the Phase D spike findings (plan L831 TBD).
SHELL_TOOL_NAMES = frozenset({"bash", "local-shell", "shell"} | set(BASH_TOOL_NAMES))

# Codex file-edit tool name candidates (plan L832 TBD); ``apply_patch`` is
# the documented Codex name and is matched case-insensitively elsewhere
# upstream, but the hook keeps to the canonical lower-case form.
FILE_EDIT_TOOL_NAMES = frozenset({"apply_patch"})

# Shell separators that end a "simple command" segment for heuristic
# parsing. Includes pipeline operators (``|``/``&&``/``||``), control
# operators (``;``/``&``/``\n``), and shell grouping delimiters
# (``(``/``)``/``{``/``}``) so a write inside a subshell or brace group
# is still seen by the per-segment extractor (``(touch unowned)`` and
# ``{ touch unowned; }`` both run their inner commands).
SHELL_SEPARATORS = frozenset(
    {";", "&&", "||", "|", "|&", "&", "\n", "(", ")", "{", "}"}
)

# Redirection operators that consume the next token as a write target.
REDIR_OPERATORS = frozenset(
    {">", ">>", "&>", "&>>", "2>", "2>>", "1>", "1>>", ">|", ">&"}
)

# File-mutation utilities whose destination args we extract.
# Maps cmd → strategy identifier (see _extract_utility_targets).
UTILITY_STRATEGIES: dict[str, str] = {
    # last positional arg is the destination
    "cp": "last_positional",
    "mv": "last_positional",
    "install": "last_positional",
    "ln": "last_positional",
    # all positional args are destinations
    "touch": "all_positional",
    "rm": "all_positional",
    "mkdir": "all_positional",
    "rmdir": "all_positional",
    "chmod": "skip_first_positional",  # first positional is the mode
    "chown": "skip_first_positional",  # first positional is the owner
    # special: of=<path>
    "dd": "dd",
}

# Git subcommands whose path arguments we treat as writes.
GIT_WRITE_SUBCOMMANDS = frozenset({"checkout", "restore", "apply", "stash"})

# Sentinel that always denies (used when a write expression is parseable
# but the literal target cannot be statically extracted, per plan L846).
_UNRESOLVED_SENTINEL = "<unresolved-write-target>"

# Placeholder emitted in place of a bash ANSI-C-quoted string (``$'...'``)
# so shlex.split treats the entire literal as a single token. The content
# of an ANSI-C string is never a command word or write target, so the
# heuristic does not need its actual value.
_ANSI_C_PLACEHOLDER = "__MALA_ANSI_C_LITERAL__"

# Placeholder emitted in place of a command-substitution body (either
# legacy ```...``` or modern ``$(...)``). Bodies are extracted and
# processed as their own segments via ``_extract_command_substitutions``
# before the main heuristic runs.
_CMDSUB_PLACEHOLDER = "__MALA_CMDSUB_LITERAL__"

# Shell reserved words that introduce a compound-command body but are
# never themselves a command word. ``_split_segments`` produces segments
# headed by these tokens (``[..., 'then', 'touch', 'unowned']``); we
# strip them so the inner extractors see the actual command word.
_SHELL_RESERVED_WORDS = frozenset(
    {
        "if",
        "then",
        "else",
        "elif",
        "fi",
        "for",
        "while",
        "do",
        "done",
        "until",
        "case",
        "esac",
        "select",
        "in",
        "function",
        "!",
        # ``coproc cmd`` runs cmd in a co-process; the inner cmd is
        # what actually executes (and writes). Strip ``coproc`` so the
        # extractor sees the real cmd word.
        "coproc",
        # ``time`` is intentionally NOT a reserved word here — it's
        # handled as an execution-prefix wrapper instead so its options
        # (``time -p cmd``) and path-qualified form (``/usr/bin/time
        # cmd``) are correctly unwrapped to the inner command.
    }
)


# Deny-message templates (plan L854-L860). These strings are byte-identical
# across the apply_patch branch and the bash write-path branch so reviewers
# can grep for the exact wording.
def _msg_no_lock(rel: str, me: str) -> str:
    return f"path '{rel}' has no active lock for agent '{me}'"


def _msg_wrong_owner(rel: str, other: str, me: str) -> str:
    return f"path '{rel}' is locked by agent '{other}'; current agent is '{me}'"


def _msg_dangerous(pattern: str) -> str:
    return f"command matches dangerous pattern: {pattern}"


def _msg_disallowed(name: str) -> str:
    return f"tool '{name}' is in MALA_DISALLOWED_TOOLS"


_MSG_ENV_MISSING = (
    "MALA_AGENT_ID/MALA_LOCK_DIR/MALA_REPO_NAMESPACE missing; refusing to evaluate lock"
)


def _allow() -> dict[str, Any]:
    """Build an ``allow`` hook result."""
    return {
        "decision": "approve",
        "hookSpecificOutput": {
            "permissionDecision": "allow",
            "permissionDecisionReason": "",
        },
    }


def _deny(reason: str) -> dict[str, Any]:
    """Build a ``deny`` hook result."""
    return {
        "decision": "block",
        "hookSpecificOutput": {
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        },
    }


def _load_state_file(session_id: str) -> dict[str, str]:
    """Read ``~/.config/mala/agent-state/{session_id}.env`` if present.

    Returns an empty dict if the file is missing or unreadable. Lines must
    be ``KEY=VALUE``; malformed lines are skipped silently.
    """
    if not session_id:
        return {}
    home = os.environ.get("HOME") or str(Path.home())
    state_path = Path(home) / ".config" / "mala" / "agent-state" / f"{session_id}.env"
    if not state_path.is_file():
        return {}
    out: dict[str, str] = {}
    try:
        for raw in state_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            out[key.strip()] = value.strip()
    except OSError:
        return {}
    return out


def _resolve_env(session_id: str) -> dict[str, str]:
    """Resolve MALA_* values from os.environ, falling back to state file.

    The lock-ownership identity bundle (``MALA_AGENT_ID``,
    ``MALA_LOCK_DIR``, ``MALA_REPO_NAMESPACE``) is sourced **atomically**,
    keyed off ``MALA_AGENT_ID`` per plan L820: if ``MALA_AGENT_ID`` is
    present in ``os.environ``, the hook uses env values for the bundle
    (the env-injection path); otherwise it falls back to the per-session
    state file as a single source for the bundle.

    Per-key merging across the identity bundle is unsafe. Identity
    (``MALA_AGENT_ID``) and lock-store coordinates (``MALA_LOCK_DIR`` /
    ``MALA_REPO_NAMESPACE``) must come from the same emitter; a stale
    ``MALA_AGENT_ID`` leaked from the parent process combined with the
    current session's ``MALA_LOCK_DIR``/``MALA_REPO_NAMESPACE`` from the
    state file would cause this Codex agent to be evaluated as the
    wrong identity against the correct lock store, allowing a write to
    a path the leaked agent happens to own.

    ``MALA_DISALLOWED_TOOLS`` is independent operator policy (not
    identity-bound) and retains per-key precedence: env wins over the
    state file when set, even when explicitly empty (so an operator can
    clear a stale state-file value with ``MALA_DISALLOWED_TOOLS=""``).
    """
    state = _load_state_file(session_id)
    out: dict[str, str] = {}
    bundle = ("MALA_AGENT_ID", "MALA_LOCK_DIR", "MALA_REPO_NAMESPACE")
    if "MALA_AGENT_ID" in os.environ:
        for key in bundle:
            env_val = os.environ.get(key)
            if env_val is not None:
                out[key] = env_val
    else:
        for key in bundle:
            if key in state:
                out[key] = state[key]
    disallowed_env = os.environ.get("MALA_DISALLOWED_TOOLS")
    if disallowed_env is not None:
        out["MALA_DISALLOWED_TOOLS"] = disallowed_env
    elif "MALA_DISALLOWED_TOOLS" in state:
        out["MALA_DISALLOWED_TOOLS"] = state["MALA_DISALLOWED_TOOLS"]
    return out


def _disallowed_tools(env: dict[str, str]) -> set[str]:
    raw = env.get("MALA_DISALLOWED_TOOLS", "")
    return {t.strip() for t in raw.split(",") if t.strip()}


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


def _check_lock(
    target: str,
    agent_id: str,
    lock_dir: str,
    repo_namespace: str,
    cwd: str | None,
) -> dict[str, Any] | None:
    """Look up the lock holder for ``target`` and return a deny dict if blocked.

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
        return _deny(_msg_no_lock(target, agent_id))
    if _CMDSUB_PLACEHOLDER in target or _ANSI_C_PLACEHOLDER in target:
        return _deny(_msg_no_lock(target, agent_id))
    if _SHELL_EXPANSION_RE.search(target):
        return _deny(_msg_no_lock(target, agent_id))
    # Ensure the locking module reads from the configured lock dir even
    # when MALA_LOCK_DIR was sourced from the state-file fallback.
    os.environ["MALA_LOCK_DIR"] = lock_dir
    # Imported lazily so the module-level import does not pin the lock dir
    # before the env is resolved.
    from ..tools.locking import get_lock_holder

    resolved = _resolve_against_cwd(target, cwd)
    holder = get_lock_holder(resolved, repo_namespace=repo_namespace)
    # Deny messages keep the original (caller-visible) target string for
    # readability, even when the lock-key was computed from the cwd-
    # resolved absolute path.
    if holder is None:
        return _deny(_msg_no_lock(target, agent_id))
    if holder != agent_id:
        return _deny(_msg_wrong_owner(target, holder, agent_id))
    return None


# ---------------------------------------------------------------------------
# Heuristic write-path extraction
# ---------------------------------------------------------------------------


def _split_segments(tokens: list[str]) -> list[list[str]]:
    """Split a token stream at shell separators into simple-command segments."""
    segments: list[list[str]] = []
    current: list[str] = []
    for tok in tokens:
        if tok in SHELL_SEPARATORS:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(tok)
    if current:
        segments.append(current)
    return segments


def _strip_env_assignments(tokens: list[str]) -> list[str]:
    """Drop leading ``KEY=VALUE`` env-assignment tokens before the command word."""
    i = 0
    while i < len(tokens) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[i]):
        i += 1
    return tokens[i:]


# POSIX execution wrappers that prefix the real command. Without
# unwrapping, ``env touch unowned`` / ``sudo cp src dst`` /
# ``command rm -rf foo`` are seen by the extractors as commands
# named ``env`` / ``sudo`` / ``command`` and the inner mutating
# utility is silently allowed.
_EXECUTION_PREFIXES = frozenset(
    {
        "env",
        "command",
        "exec",
        "sudo",
        "doas",
        "nice",
        "nohup",
        "timeout",
        "stdbuf",
        "ionice",
        "chrt",
        "taskset",
        "setsid",
        "unshare",
        # ``time`` is the GNU/BSD timing utility (``/usr/bin/time``) and
        # also a bash reserved word; treating it as an execution wrapper
        # handles both invocation styles (``time cmd`` and
        # ``time -p cmd`` / ``/usr/bin/time -v cmd``).
        "time",
    }
)

# Per-wrapper short-flag letters that consume the next token as a value.
# Used by ``_unwrap_execution_prefix`` so ``sudo -u root touch f``
# (where ``-u`` takes ``root`` as its value) does not treat ``root`` as
# the inner command.
_WRAPPER_VALUE_LETTERS: dict[str, frozenset[str]] = {
    "env": frozenset({"u", "S", "C"}),
    # NOTE: ``sudo -A`` (askpass), ``-b`` (background), ``-E`` (preserve-
    # env), ``-H`` (set-home), ``-i`` (login), ``-l`` (list), ``-n``
    # (non-interactive), ``-P`` (preserve-groups), ``-S`` (stdin), ``-s``
    # (shell), ``-V`` (version), ``-v`` (validate), ``-K``/``-k`` are
    # all BOOLEAN and intentionally NOT listed here. Listing them as
    # value letters would consume the inner cmd as the flag value.
    "sudo": frozenset({"u", "g", "h", "p", "C", "D", "T", "U", "r", "t"}),
    # ``doas`` short value-flags: ``-a STYLE`` (auth-style),
    # ``-C path`` (config), ``-u user`` (user). Without ``a`` here,
    # ``doas -a auth touch unowned`` would treat ``-a`` as boolean and
    # consume ``touch`` as the inner command's name.
    "doas": frozenset({"u", "C", "a"}),
    "exec": frozenset({"a"}),
    "nice": frozenset({"n"}),
    "stdbuf": frozenset({"i", "o", "e"}),
    "ionice": frozenset({"c", "n", "p"}),
    "chrt": frozenset({"p"}),
    "taskset": frozenset({"p", "c"}),
    "timeout": frozenset({"s", "k"}),
    # NOTE: ``unshare`` namespace flags (``-i``/``-m``/``-n``/``-p``/
    # ``-u``/``-U``/``-r``/``-T``/``-C``/``-l``/``-c``) are BOOLEAN —
    # they enable a namespace, they don't take a value. Only
    # ``-S/--setuid``, ``-G/--setgid``, ``-w/--wd``, ``-R/--root`` are
    # value-taking shorts.
    "unshare": frozenset({"S", "G", "w", "R"}),
    # ``time -p`` (POSIX format) and ``time -v`` (verbose, GNU) are
    # boolean flags. ``time -o file -a cmd`` is value-taking.
    "time": frozenset({"o", "f"}),
}

# Per-wrapper LONG-form flags that consume the next token as a value.
# Default (long flags not in this table) is BOOLEAN — they do NOT
# consume the next token. Without this table, ``sudo --preserve-env
# touch f`` would consume ``touch`` as the value of ``--preserve-env``
# and the inner mutating command would never be seen by the heuristic
# (a P1 bypass surfaced by the eighth review).
_WRAPPER_VALUE_LONG_FLAGS: dict[str, frozenset[str]] = {
    "env": frozenset({"--unset", "--chdir", "--split-string"}),
    "sudo": frozenset(
        {
            "--user",
            "--group",
            "--host",
            "--prompt",
            "--auth-type",
            "--chdir",
            "--type",
            "--role",
            "--login-class",
            "--other-user",
            "--close-from",
        }
    ),
    "doas": frozenset({"--user"}),
    "exec": frozenset({"--name"}),
    "nice": frozenset({"--adjustment"}),
    "stdbuf": frozenset({"--input", "--output", "--error"}),
    "ionice": frozenset({"--class", "--classdata", "--pid"}),
    "chrt": frozenset({"--pid"}),
    "taskset": frozenset({"--pid", "--cpu-list"}),
    "timeout": frozenset({"--signal", "--kill-after"}),
    "unshare": frozenset(
        {
            "--map-user",
            "--map-group",
            "--map-users",
            "--map-groups",
            # NOTE: ``--user``/``--ipc``/``--mount``/``--net``/``--pid``/
            # ``--time``/``--cgroup``/``--uts`` are BOOLEAN namespace
            # toggles in unshare and intentionally NOT listed here.
            "--setuid",
            "--setgid",
            "--propagation",
            "--root",
            "--wd",
        }
    ),
    "time": frozenset({"--output", "--format"}),
}

# Per-wrapper short flag letters that change the effective working
# directory. ``env -C dir cmd`` (coreutils 8.32+) runs ``cmd`` with
# cwd=``dir``; ``sudo -D dir cmd`` does the same on systems with
# sudo 1.9.3+. The heuristic must therefore treat these wrappers as a
# cd for purposes of relative-path resolution.
_WRAPPER_CWD_CHANGE_LETTERS: dict[str, frozenset[str]] = {
    "env": frozenset({"C"}),
    "sudo": frozenset({"D"}),
}

# Per-wrapper long-form flags that change the effective working dir.
_WRAPPER_CWD_CHANGE_LONG_FLAGS: dict[str, frozenset[str]] = {
    "env": frozenset({"--chdir"}),
    "sudo": frozenset({"--chdir"}),
}

# Patterns for the leading-positional argument of each wrapper. Used by
# ``_unwrap_execution_prefix`` to decide whether to drop the next token
# as the wrapper's mandatory positional. Without these patterns, a
# command like ``taskset --cpu-list 0,2 touch f`` (where the mask was
# supplied via the flag) would have ``touch`` dropped as if it were the
# mask, leaving the heuristic with just ``f`` and the inner write
# silently allowed.
_TIMEOUT_DURATION_RE = re.compile(r"^[0-9]+(\.[0-9]+)?[smhdSMHD]?$|^infinity$|^inf$")
_PRIORITY_RE = re.compile(r"^[0-9]+$")
_TASKSET_MASK_RE = re.compile(
    r"^(0[xX][0-9a-fA-F]+|[0-9]+(,[0-9]+)*|[0-9]+-[0-9]+(,[0-9]+(-[0-9]+)?)*)$"
)


# Per-wrapper short flag letters whose VALUE is itself a write target.
# When the wrapper is unwrapped, these values are added to the segment's
# write-target list so the lock-check covers wrapper-side writes too.
# Example: ``time -o /tmp/timing.log cmd`` writes to ``/tmp/timing.log``.
_WRAPPER_OUTPUT_LETTERS: dict[str, frozenset[str]] = {
    "time": frozenset({"o"}),
}

# Per-wrapper long-form flags whose value is a write target.
_WRAPPER_OUTPUT_LONG_FLAGS: dict[str, frozenset[str]] = {
    "time": frozenset({"--output"}),
}

# Per-wrapper short flag letters whose VALUE is a shell command body.
# ``env -S 'touch unowned'`` (split-string mode, coreutils 8.30+) parses
# the value as argv and runs it; ignoring the value would silently
# allow an inner ``touch unowned`` to bypass the gate. The captured
# body is processed recursively via ``_extract_shell_write_paths`` so
# its write targets are gated like any other segment.
_WRAPPER_SHELL_BODY_LETTERS: dict[str, frozenset[str]] = {
    "env": frozenset({"S"}),
}

# Per-wrapper long-form flags whose value is a shell command body.
_WRAPPER_SHELL_BODY_LONG_FLAGS: dict[str, frozenset[str]] = {
    "env": frozenset({"--split-string"}),
}


def _command_basename_str(token: str) -> str:
    """Return the basename of a single token (``/usr/bin/env`` → ``env``)."""
    return os.path.basename(token) if token else token


def _skip_wrapper_flag_args(
    tokens: list[str], wrapper: str
) -> tuple[list[str], bool, list[str], list[str]]:
    """Skip leading flag tokens (and their values) for ``wrapper``.

    Returns ``(remaining_tokens, cwd_changed, wrapper_write_targets,
    wrapper_shell_bodies)``:

    - ``cwd_changed`` is True if any consumed flag was a cwd-changing
      wrapper option (``env -C dir``, ``sudo --chdir=dir``).
    - ``wrapper_write_targets`` collects paths the wrapper itself
      writes via flag values (``time -o /tmp/log cmd`` writes
      ``/tmp/log``); the caller adds these to the segment write list
      so they are lock-checked alongside the inner command's writes.
    - ``wrapper_shell_bodies`` collects shell-command strings supplied
      via wrapper flag values (``env -S 'touch unowned'`` /
      ``env --split-string='touch unowned'``); the caller processes
      each body via ``_extract_shell_write_paths`` so its inner write
      targets are gated. Without this capture, ``-S``'s value would
      be silently consumed and the inner ``touch`` would slip past.

    Per-wrapper short and long value-flag tables ensure
    ``sudo -u root cmd`` consumes ``root`` as the value of ``-u`` and
    ``sudo --preserve-env cmd`` does NOT consume ``cmd`` (since
    ``--preserve-env`` is boolean).
    """
    value_letters = _WRAPPER_VALUE_LETTERS.get(wrapper, frozenset())
    long_value_flags = _WRAPPER_VALUE_LONG_FLAGS.get(wrapper, frozenset())
    cwd_letters = _WRAPPER_CWD_CHANGE_LETTERS.get(wrapper, frozenset())
    cwd_long_flags = _WRAPPER_CWD_CHANGE_LONG_FLAGS.get(wrapper, frozenset())
    output_letters = _WRAPPER_OUTPUT_LETTERS.get(wrapper, frozenset())
    output_long_flags = _WRAPPER_OUTPUT_LONG_FLAGS.get(wrapper, frozenset())
    body_letters = _WRAPPER_SHELL_BODY_LETTERS.get(wrapper, frozenset())
    body_long_flags = _WRAPPER_SHELL_BODY_LONG_FLAGS.get(wrapper, frozenset())
    cwd_changed = False
    wrapper_writes: list[str] = []
    wrapper_bodies: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("-") or tok == "-":
            break
        if tok == "--":
            # End-of-options marker; the next token is the cmd.
            i += 1
            break
        # Long form ``--name=value``: token includes its value.
        if tok.startswith("--") and "=" in tok:
            name, value = tok.split("=", 1)
            if name in cwd_long_flags:
                cwd_changed = True
            if name in output_long_flags and value:
                wrapper_writes.append(value)
            if name in body_long_flags and value:
                wrapper_bodies.append(value)
            i += 1
            continue
        # Long form. Only consume the next token when this flag is in
        # the wrapper's value-long-flag table; otherwise treat as boolean.
        if tok.startswith("--"):
            if tok in cwd_long_flags:
                cwd_changed = True
            if tok in output_long_flags and i + 1 < len(tokens):
                wrapper_writes.append(tokens[i + 1])
            if tok in body_long_flags and i + 1 < len(tokens):
                wrapper_bodies.append(tokens[i + 1])
            if tok in long_value_flags and i + 1 < len(tokens):
                i += 2
                continue
            i += 1
            continue
        # Short bundle. Look for any value-taking letter.
        body = tok[1:]
        consumed_value = False
        for j, ch in enumerate(body):
            if ch in cwd_letters:
                cwd_changed = True
            if ch in output_letters:
                # Capture the value as a wrapper-side write target.
                if j == len(body) - 1:
                    if i + 1 < len(tokens):
                        wrapper_writes.append(tokens[i + 1])
                else:
                    attached = body[j + 1 :]
                    if attached:
                        wrapper_writes.append(attached)
            if ch in body_letters:
                # Capture the value as a shell-command body to process.
                if j == len(body) - 1:
                    if i + 1 < len(tokens):
                        wrapper_bodies.append(tokens[i + 1])
                else:
                    attached = body[j + 1 :]
                    if attached:
                        wrapper_bodies.append(attached)
            if ch in value_letters:
                if j == len(body) - 1:
                    # Bundle ends with value-letter: next token is
                    # the value (if not itself a flag).
                    if i + 1 < len(tokens):
                        consumed_value = True
                # Else: rest of bundle is the attached value, no
                # next-token consumption.
                break
        i += 2 if consumed_value else 1
    return tokens[i:], cwd_changed, wrapper_writes, wrapper_bodies


def _unwrap_execution_prefix(
    tokens: list[str],
) -> tuple[list[str], bool, list[str], list[str]]:
    """Strip leading execution-wrapper prefixes recursively.

    Returns ``(stripped_tokens, cwd_changed, wrapper_writes,
    wrapper_shell_bodies)``: ``cwd_changed`` is True if any wrapper
    option changed cwd (``env -C``, ``sudo --chdir``);
    ``wrapper_writes`` collects write targets supplied via wrapper
    flags (``time -o file``); ``wrapper_shell_bodies`` collects shell
    command strings supplied via wrapper flags (``env -S 'cmd'``)
    that the caller processes recursively.

    Walks past ``env``/``sudo``/``exec``/``command``/``nohup``/``timeout``/
    etc., their flag args, and (for ``env``-style wrappers) leading
    ``KEY=VALUE`` env-assignments, until the first token is the real
    command. Loops to handle nested wrappers (``sudo env touch f``).
    """
    cwd_changed = False
    wrapper_writes: list[str] = []
    wrapper_bodies: list[str] = []
    iterations = 0
    while tokens and iterations < 10:
        cmd = _command_basename_str(tokens[0])
        if cmd not in _EXECUTION_PREFIXES:
            break
        new_tokens = tokens[1:]
        new_tokens, sub_cwd, sub_writes, sub_bodies = _skip_wrapper_flag_args(
            new_tokens, cmd
        )
        if sub_cwd:
            cwd_changed = True
        wrapper_writes.extend(sub_writes)
        wrapper_bodies.extend(sub_bodies)
        # Wrappers that allow ``KEY=VALUE`` before the inner cmd.
        # ``time``, like the bash reserved-word ``time``, accepts
        # ``KEY=VALUE`` env-assignments before the inner command. We
        # added ``time`` here (not as a reserved word) so its options
        # and path-qualified form are unwrapped; without env-assign
        # stripping, ``time VAR=1 touch f`` leaves ``VAR=1`` as the
        # cmd basename and the inner ``touch`` is silently allowed.
        if cmd in {"env", "sudo", "doas", "time"}:
            new_tokens = _strip_env_assignments(new_tokens)
        # Wrappers that take a leading positional (DURATION/PRIORITY/
        # MASK) before the inner cmd:
        #   timeout DURATION cmd
        #   chrt PRIORITY cmd
        #   taskset MASK cmd
        # Drop only when the next token actually looks like the
        # expected positional. ``taskset --cpu-list 0,2 cmd`` already
        # consumed the mask via the flag; the next token is the cmd
        # itself and must NOT be dropped.
        if (
            cmd == "timeout"
            and new_tokens
            and _TIMEOUT_DURATION_RE.match(new_tokens[0])
        ):
            new_tokens = new_tokens[1:]
        elif cmd == "chrt" and new_tokens and _PRIORITY_RE.match(new_tokens[0]):
            new_tokens = new_tokens[1:]
        elif cmd == "taskset" and new_tokens and _TASKSET_MASK_RE.match(new_tokens[0]):
            new_tokens = new_tokens[1:]
        if not new_tokens:
            break
        tokens = new_tokens
        iterations += 1
    return tokens, cwd_changed, wrapper_writes, wrapper_bodies


# Builtins / commands that change the shell's effective working directory
# in a way the heuristic cannot statically resolve. ``pushd``/``popd``
# both shift cwd (push to a stack / pop from it). ``builtin cd ...``
# explicitly invokes the cd builtin even when ``cd`` is shadowed.
_CD_BUILTINS = frozenset({"cd", "pushd", "popd"})


def _is_cd_segment(tokens: list[str]) -> bool:
    """True if a segment changes cwd (``cd``/``pushd``/``popd``/``builtin cd``)."""
    if not tokens:
        return False
    cmd = _command_basename(tokens)
    if cmd in _CD_BUILTINS:
        return True
    # ``builtin cd <dir>`` and ``builtin pushd ...`` — first token is
    # ``builtin``, second is the cd-family builtin.
    if cmd == "builtin" and len(tokens) >= 2:
        next_cmd = _command_basename_str(tokens[1])
        if next_cmd in _CD_BUILTINS:
            return True
    return False


def _has_brace_expansion(tok: str) -> bool:
    """True if ``tok`` contains a bash brace expansion ``{...,...}`` or
    range ``{n..m}``.

    The check is purely syntactic — any token with a ``{...}`` block
    that contains a comma or ``..`` triggers detection. Bash expands
    such tokens BEFORE command execution, so a single shlex token
    ``{touch,unowned}`` actually invokes ``touch unowned`` at runtime.
    Without this check, the hook never sees the inner command and
    silently allows the unowned write.

    ``${var}`` parameter expansion is NOT a brace expansion and is
    excluded (a leading ``$`` immediately before ``{`` distinguishes
    them; ``$`` is also caught separately by ``_SHELL_EXPANSION_RE``
    in ``_check_lock``).
    """
    n = len(tok)
    i = 0
    while i < n:
        if tok[i] == "{" and (i == 0 or tok[i - 1] != "$"):
            # Find matching ``}``; check if the body contains ``,`` or ``..``.
            depth = 1
            j = i + 1
            has_comma = False
            has_range = False
            while j < n and depth > 0:
                if tok[j] == "{":
                    depth += 1
                elif tok[j] == "}":
                    depth -= 1
                    if depth == 0:
                        break
                elif tok[j] == ",":
                    has_comma = True
                elif tok[j] == "." and j + 1 < n and tok[j + 1] == ".":
                    has_range = True
                    j += 1
                j += 1
            if depth == 0 and (has_comma or has_range):
                return True
        i += 1
    return False


def _strip_segment_prefix(tokens: list[str]) -> list[str]:
    """Strip leading reserved words and env-assignments until stable.

    Calling ``_strip_env_assignments`` once before ``_strip_leading_reserved``
    misses the case where a reserved word HIDES an env assignment, e.g.
    ``then VAR=val touch f``: ``then`` is at the head so env-strip is a
    no-op; ``then`` is then removed by reserved-strip; but ``VAR=val``
    is now at the head and never re-stripped. The result is a segment
    whose "command word" is ``VAR=val``, which no extractor matches —
    the inner ``touch`` is silently allowed. Looping until both passes
    are no-ops handles this and any other interleaving (``! VAR=val
    cmd``, ``time VAR=val cmd``).
    """
    while True:
        before = len(tokens)
        tokens = _strip_env_assignments(tokens)
        tokens = _strip_leading_reserved(tokens)
        if len(tokens) == before:
            break
    return tokens


def _strip_leading_reserved(tokens: list[str]) -> list[str]:
    """Drop leading shell reserved words (``if``/``then``/``do``/...).

    Compound commands (``if ...; then touch f; fi``) leave the actual
    command word behind a non-mutating reserved word in the resulting
    segment. Without stripping, the utility extractor sees ``then`` as
    the basename and emits no targets, silently allowing the unowned
    write that follows.
    """
    while tokens and tokens[0] in _SHELL_RESERVED_WORDS:
        tokens = tokens[1:]
    return tokens


def _strip_shell_comments(command: str) -> str:
    """Remove unquoted ``#``-comments from ``command``.

    Comments run from a word-starting ``#`` (preceded by whitespace,
    a separator, or start-of-input) up to the next newline. The
    newline itself is preserved so it can later be rewritten to ``;``
    by ``_normalize_separators`` and split into a fresh segment.

    Used as a pre-pass before ``_extract_command_substitutions`` so a
    substitution inside a comment (``# $(touch unowned)\\necho ok``)
    is NOT extracted as if bash would execute it. Also used by the
    dangerous-cmd recursive detector for the same reason.
    """
    out: list[str] = []
    i = 0
    n = len(command)
    quote: str | None = None
    while i < n:
        c = command[i]
        # Backslash escape outside single quotes consumes the next
        # char verbatim — including a ``#`` so ``\\#`` is literal.
        if c == "\\" and i + 1 < n and quote != "'":
            out.append(c)
            out.append(command[i + 1])
            i += 2
            continue
        if quote is not None:
            out.append(c)
            if c == quote:
                quote = None
            i += 1
            continue
        # ``$'...'`` ANSI-C: skip the entire block and emit a
        # placeholder so the quote tracker stays in sync. The previous
        # implementation tracked ANSI-C as a regular single-quoted
        # region, which desynced on inputs like ``$'\\''`` (it treated
        # the escaped ``\\'`` as the closer and the third ``'`` as a
        # fresh opener). The body is never a comment and never a write
        # target, so a placeholder is sufficient.
        if c == "$" and quote is None and i + 1 < n and command[i + 1] == "'":
            j = i + 2
            while j < n:
                if command[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if command[j] == "'":
                    j += 1
                    break
                j += 1
            out.append(_ANSI_C_PLACEHOLDER)
            i = j
            continue
        if c in ('"', "'"):
            out.append(c)
            quote = c
            i += 1
            continue
        # Word-starting ``#`` outside quotes: skip to next newline.
        if c == "#" and (
            i == 0
            or command[i - 1].isspace()
            or command[i - 1] in (";", "&", "|", "(", ")")
        ):
            while i < n and command[i] != "\n":
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _extract_command_substitutions(command: str) -> tuple[str, list[str]]:
    """Replace ```...``` and ``$(...)`` with a placeholder, returning bodies.

    Returns ``(cleaned, bodies)`` — ``cleaned`` is the input string with
    every command-substitution block replaced by ``_CMDSUB_PLACEHOLDER``,
    and ``bodies`` is the list of extracted body strings. Both legacy
    backticks and modern ``$()`` substitutions are pulled out so the
    inner command is processed by the recursive ``_extract_shell_write_paths``
    call.

    Inside single quotes (``'...'``) substitutions are preserved verbatim
    — bash treats them literally there. Inside double quotes both forms
    still trigger command substitution and ARE extracted.

    ``$((arith))`` (arithmetic expansion) is detected by a leading
    ``$((`` and skipped intact so its inner ``(`` / ``)`` don't confuse
    the cmd-substitution body parser.

    Without this preprocessing, ``echo `touch unowned``` and
    ``echo "$(touch unowned)"`` would leave the inner ``touch`` invisible
    to every extractor — both are standard shell-execution forms, not
    residual obfuscation gaps.
    """
    out: list[str] = []
    bodies: list[str] = []
    i = 0
    n = len(command)
    quote: str | None = None
    while i < n:
        c = command[i]
        if c == "\\" and i + 1 < n and quote != "'":
            out.append(c)
            out.append(command[i + 1])
            i += 2
            continue
        # ANSI-C quoting (``$'...'``) outside other quoted regions: skip
        # the entire block and emit ``_ANSI_C_PLACEHOLDER`` so the
        # quote/sub tracker stays in sync. Without this, an escaped
        # single quote (``$'\\''``) desyncs the parser — it treats the
        # escaped ``\\'`` as the closer and the third ``'`` as a fresh
        # opener, so any backtick / ``$()`` substitution that follows is
        # seen as "inside single quotes" and never extracted into
        # ``bodies``. Concrete bypass before this fix:
        # ``echo $'\\'' `touch unowned.py``` left the inner ``touch``
        # invisible to every extractor and the hook returned allow.
        # (Inside ``"..."`` bash treats ``$'...'`` literally, so the
        # ANSI-C skip is gated on ``quote is None``.)
        if c == "$" and quote is None and i + 1 < n and command[i + 1] == "'":
            j = i + 2
            while j < n:
                if command[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if command[j] == "'":
                    j += 1
                    break
                j += 1
            out.append(_ANSI_C_PLACEHOLDER)
            i = j
            continue
        # Legacy backtick command substitution.
        if c == "`" and quote != "'":
            j = i + 1
            body_chars: list[str] = []
            while j < n:
                if command[j] == "\\" and j + 1 < n and command[j + 1] == "`":
                    # Escaped inner backtick (`\``). bash treats this
                    # as a literal backtick in the OUTER body but as a
                    # nested-substitution delimiter when the body is
                    # re-parsed. Unescape here so the recursive call
                    # to ``_extract_command_substitutions`` on the
                    # extracted body sees real backticks and pulls
                    # the nested ``touch f`` out as its own segment.
                    body_chars.append("`")
                    j += 2
                    continue
                if command[j] == "\\" and j + 1 < n:
                    body_chars.append(command[j])
                    body_chars.append(command[j + 1])
                    j += 2
                    continue
                if command[j] == "`":
                    break
                body_chars.append(command[j])
                j += 1
            if j < n:
                bodies.append("".join(body_chars))
                out.append(_CMDSUB_PLACEHOLDER)
                i = j + 1
                continue
            # Unmatched: fall through.
        # ``$(...)`` command substitution / ``$((...))`` arithmetic.
        if c == "$" and quote != "'" and i + 1 < n and command[i + 1] == "(":
            if i + 2 < n and command[i + 2] == "(":
                # ``$(( arith ))`` — arithmetic expansion. The OUTER
                # value is numeric so the surrounding token is not a
                # write surface, but bash STILL evaluates command
                # substitutions and backticks inside the arithmetic
                # body before computing the result. ``echo $(( $(touch
                # unowned) + 1 ))`` runs ``touch unowned`` for real;
                # passing the entire ``$((...))`` block through
                # verbatim would silently allow it.
                depth = 2
                j = i + 3
                inner_quote: str | None = None
                while j < n and depth > 0:
                    cj = command[j]
                    if cj == "\\" and j + 1 < n and inner_quote != "'":
                        j += 2
                        continue
                    if inner_quote is not None:
                        if cj == inner_quote:
                            inner_quote = None
                        j += 1
                        continue
                    if cj in ('"', "'"):
                        inner_quote = cj
                        j += 1
                        continue
                    if cj == "(":
                        depth += 1
                    elif cj == ")":
                        depth -= 1
                    j += 1
                if depth == 0:
                    out.append(command[i:j])
                    # Recursively pull any cmd substitutions out of the
                    # arithmetic body so their writes are gated.
                    arith_body = command[i + 3 : j - 2]
                    _, inner_bodies = _extract_command_substitutions(arith_body)
                    bodies.extend(inner_bodies)
                    i = j
                    continue
                # Unmatched: fall through.
            else:
                # ``$( cmd )`` — find matching close paren, respecting
                # nesting and quote state inside the body.
                depth = 1
                j = i + 2
                inner_quote = None
                while j < n and depth > 0:
                    cj = command[j]
                    if cj == "\\" and j + 1 < n and inner_quote != "'":
                        j += 2
                        continue
                    if inner_quote is not None:
                        if cj == inner_quote:
                            inner_quote = None
                        j += 1
                        continue
                    if cj in ('"', "'"):
                        inner_quote = cj
                        j += 1
                        continue
                    if cj == "(":
                        depth += 1
                    elif cj == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                if depth == 0:
                    bodies.append(command[i + 2 : j])
                    out.append(_CMDSUB_PLACEHOLDER)
                    i = j + 1
                    continue
                # Unmatched: fall through.
        if quote is not None:
            out.append(c)
            if c == quote:
                quote = None
            i += 1
            continue
        if c in ('"', "'"):
            out.append(c)
            quote = c
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out), bodies


_DEV_NULL_LIKE = frozenset({"/dev/null", "/dev/stderr", "/dev/stdout", "/dev/tty"})
# File-descriptor duplications: leading-``&`` form (``&1`` / ``&-``).
# These appear as the *post-prefix* tail when the operator regex consumed
# only a leading-``&``-less prefix (e.g. ``2>`` from ``2>&1``); the residue
# ``&1`` is an fd reference, not a filesystem path.
_FD_REF_RE = re.compile(r"^&[0-9\-]+$")
# Pure fd-numeric / close-fd target. Used after a ``>&`` / ``n>&`` operator
# to distinguish ``>&1`` (fd dup, skip) from ``>&file`` (write, lock-check).
_FD_REF_TARGET_RE = re.compile(r"^[0-9]+$|^-$")
# Whole-token redirection operators. Accepted fd prefixes:
#   - numeric (``1``/``2``/``3``/...) — POSIX
#   - ``&`` — combined stdout+stderr (``&>``, ``&>>``)
#   - ``{varname}`` — bash 4.1+ variable-allocated fd (``{fd}>file``);
#     bash auto-assigns a free fd to ``$varname`` and applies the redirect.
#
# Recognised operator forms:
#   - ``>``/``>>``/``n>``/``n>>`` — write
#   - ``&>``/``&>>``/``>&``/``n>&`` — combined / legacy combined
#   - ``>|`` / ``n>|`` — force-clobber write
#   - ``<>`` / ``n<>`` — POSIX read-write open (creates file if missing,
#     so it IS a write surface)
# ``<`` alone (read-only), ``<<`` (here-doc), and ``<<<`` (here-string)
# are deliberately excluded — they don't open a file for writing.
_REDIR_FD_PREFIX = r"(?:[0-9]+|&|\{[A-Za-z_][A-Za-z0-9_]*\})"
_REDIR_TAIL = r"(?:<>|>{1,2}[&|]?)"
_REDIR_OP_TOKEN_RE = re.compile(rf"^{_REDIR_FD_PREFIX}?{_REDIR_TAIL}$")
# Prefix regex for combined-token forms like ``>file``, ``2>file``,
# ``&>file``, ``>&file``, ``3>>file``, ``{fd}>file``, ``<>file``,
# ``3<>file``. Greedy on the optional trailing ``&`` so ``2>&1`` matches
# the prefix ``2>&`` (target ``1`` → fd dup).
_COMBINED_REDIR_PREFIX_RE = re.compile(rf"^{_REDIR_FD_PREFIX}?{_REDIR_TAIL}")


def _drop_redirections(tokens: list[str]) -> list[str]:
    """Return ``tokens`` with redirect operators and their targets removed.

    Used by non-redirect segment extractors so a redirect operator+target
    that appears in the middle of an argument list (between the command
    word and the trailing positional, or after it) does not become a
    positional that the utility/oneliner extractors mistakenly treat as
    a write surface.

    Without this, ``cp owned dst >out`` left ``>`` and ``out`` in cp's
    positional list and ``last_positional`` returned ``out`` instead
    of ``dst``, so the real cp destination was never lock-checked.
    Combined with the ``&>`` split fix in ``_normalize_separators``,
    ``cp owned unowned&>out`` now lock-checks the real ``unowned``
    instead of leaking to a redirect target the agent legitimately
    holds.
    """
    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        tok = tokens[i]
        if _REDIR_OP_TOKEN_RE.match(tok):
            # Operator + its target word are both removed; ``cmd > out``
            # becomes ``cmd``. fd-dup target words (``2>&1``) are also
            # removed — fine, they aren't write surfaces.
            i += 2
            continue
        out.append(tok)
        i += 1
    return out


def _extract_redirection_targets(tokens: list[str]) -> list[str]:
    """Find ``>``/``>>``/etc. targets in a token stream.

    Operates on tokens AFTER ``_normalize_separators`` has padded every
    unquoted redirect operator with surrounding whitespace; the
    extractor therefore only needs to recognise WHOLE-token operators.
    A ``>file`` token that survives shlex unchanged was either inside
    quotes (``echo ">file"``) or escaped (``grep \\>5``) — both are
    legitimate positional arguments, not redirects, so we no longer
    apply the combined-prefix fallback (which used to false-positive
    deny on those quoted args).

    Recognised forms — POSIX shells accept ANY numeric fd prefix, not just
    ``1``/``2``, and the legacy ``>&file`` operator is equivalent to
    ``> file 2> file`` (write both stdout and stderr to ``file``):

    - ``>file`` / ``>>file`` / ``n>file`` / ``n>>file`` — write to file
    - ``&>file`` / ``&>>file`` — write stdout+stderr (modern syntax)
    - ``>&file`` / ``n>&file`` — write stdout+stderr (legacy syntax)
    - ``>&n`` / ``n>&m`` / ``>&-`` — fd duplication / close (NOT writes)
    - ``<>file`` / ``n<>file`` — POSIX read-write open (creates file)

    Fd duplications and close-fd are skipped so common idioms like
    ``cmd >out.log 2>&1`` don't false-deny when ``out.log`` is locked.
    """
    out: list[str] = []
    for i, tok in enumerate(tokens):
        # Whole-token operator: next token is the redirect target.
        if _REDIR_OP_TOKEN_RE.match(tok) and i + 1 < len(tokens):
            target = tokens[i + 1]
            # ``>&`` / ``n>&`` followed by a numeric/dash word is fd dup.
            if tok.endswith("&") and _FD_REF_TARGET_RE.match(target):
                continue
            if target in _DEV_NULL_LIKE:
                continue
            out.append(target)
            continue
        # The combined-prefix fallback (``>file`` as a single token) is
        # intentionally NOT applied here — see the docstring above. Any
        # ``>file``-shaped token reaching this point is quoted/escaped
        # and is not a real redirect surface.
    return out


def _command_basename(tokens: list[str]) -> str:
    """Return the basename of ``tokens[0]`` (or ``""`` for empty input).

    Path-qualified invocations (``/usr/bin/touch``, ``./bin/sed``, etc.)
    must dispatch to the same extractor as the bare command name; without
    basename normalisation, ``/usr/bin/touch unowned.py`` bypasses every
    utility/in-place/oneliner/git/patch extractor and the write slips
    past AC #19.
    """
    if not tokens:
        return ""
    return os.path.basename(tokens[0])


def _extract_tee_targets(tokens: list[str]) -> list[str]:
    if _command_basename(tokens) != "tee":
        return []
    out: list[str] = []
    options_ended = False
    for tok in tokens[1:]:
        if not options_ended and tok == "--":
            options_ended = True
            continue
        if options_ended:
            # After ``--``, every token is an operand even if it
            # starts with ``-``. ``tee -- -owned.py`` writes to the
            # literal file ``-owned.py``.
            out.append(tok)
            continue
        if tok.startswith("-"):
            continue
        out.append(tok)
    return out


def _has_inplace_flag(tokens: list[str], inplace_short: str = "i") -> bool:
    """True if a sed/perl token signals in-place edit (``-i``/``--in-place``)."""
    for tok in tokens[1:]:
        if not tok.startswith("-"):
            continue
        # Long-form: ``--in-place`` and ``--in-place=.bak``. Must be
        # checked before the ``--`` short-circuit so it is reachable.
        if tok == "--in-place" or tok.startswith("--in-place="):
            return True
        if tok.startswith("--"):
            continue
        body = tok.lstrip("-")
        if inplace_short in body:
            return True
    return False


# Short-flag letters whose presence in a sed/perl flag bundle means the
# script body is consumed by the flag (either as the next token, when the
# letter is the last one in the bundle, or attached, when it is followed
# by more characters in the same token). The letter set differs per tool.
_SED_VALUE_LETTERS = frozenset({"e", "f"})
_PERL_VALUE_LETTERS = frozenset({"e", "f", "I", "M", "P"})
# Long-form sed flags that take a script value (with ``--name=value`` or
# ``--name value`` syntax).
_SED_LONG_VALUE_FLAGS = frozenset({"--expression", "--file", "--regexp"})
# Perl rarely uses long-form value flags in practice; keep the set narrow.
_PERL_LONG_VALUE_FLAGS: frozenset[str] = frozenset()


def _extract_inplace_targets(tokens: list[str]) -> list[str]:
    """Trailing positional file args after a ``-i`` flag for sed/perl.

    Honours every documented form of the script flag so a misclassified
    flag does not leak the file target as the "inline script":

    - bare short ``-e SCRIPT`` (next token is the value)
    - attached short ``-eSCRIPT`` / ``-iesSCRIPT`` (script embedded in the
      same token, after the value-taking letter)
    - long form ``--expression=SCRIPT`` (value embedded after ``=``)
    - long form ``--expression SCRIPT`` (next token is the value)

    When any value-taking flag is detected, every positional is treated
    as a file target. Otherwise the first positional is the inline script
    and the rest are files (e.g., ``sed -i 's/x/y/' f``).
    """
    if not tokens:
        return []
    cmd = _command_basename(tokens)
    if cmd not in {"sed", "perl"}:
        return []
    if not _has_inplace_flag(tokens):
        return []
    short_value_letters = _SED_VALUE_LETTERS if cmd == "sed" else _PERL_VALUE_LETTERS
    long_value_flags = _SED_LONG_VALUE_FLAGS if cmd == "sed" else _PERL_LONG_VALUE_FLAGS

    positionals: list[str] = []
    skip_next = False
    has_value_flag = False
    options_ended = False
    for idx, tok in enumerate(tokens[1:], start=1):
        if skip_next:
            skip_next = False
            continue
        # ``--`` end-of-options marker: every later token is a positional
        # file argument, even if it starts with ``-``.
        if not options_ended and tok == "--":
            options_ended = True
            continue
        if options_ended:
            positionals.append(tok)
            continue
        if tok.startswith("--"):
            name = tok.split("=", 1)[0]
            if name in long_value_flags:
                has_value_flag = True
                if "=" not in tok:
                    skip_next = True
            continue
        if tok.startswith("-") and len(tok) >= 2:
            body = tok[1:]
            # BSD sed (macOS) requires ``-i SUFFIX`` as a separate
            # token, with an EMPTY suffix written ``-i ''``. shlex
            # leaves the empty as a positional. Without this skip,
            # ``sed -i '' 's/a/b/' file`` collected positionals
            # ``['', 's/a/b/', 'file']``; the script-positional logic
            # dropped only the empty, returning ``s/a/b/`` plus
            # ``file`` as write targets — a false-deny that broke
            # routine macOS sed usage even when the agent held a
            # lock on the real file. Detect ``-i`` (last/only letter
            # in the bundle) followed by an empty token and treat
            # the empty as the BSD suffix value. Trade-off: GNU
            # ``sed -i '' file`` (empty script no-op rewrite of
            # ``file``) is now under-detected, which is acceptable
            # — that idiom is degenerate and rarely used while
            # ``sed -i '' SCRIPT file`` is the macOS dev default.
            if (
                cmd == "sed"
                and body.endswith("i")
                and idx + 1 < len(tokens)
                and tokens[idx + 1] == ""
            ):
                skip_next = True
                continue
            for j, ch in enumerate(body):
                if ch in short_value_letters:
                    has_value_flag = True
                    if j == len(body) - 1:
                        # Letter is the last char in the bundle; the
                        # script value is the next token.
                        skip_next = True
                    # Else: value is embedded later in the same token.
                    break
            continue
        positionals.append(tok)
    if has_value_flag:
        return positionals
    return positionals[1:] if len(positionals) > 1 else []


# Awk flags that consume the program (so subsequent positionals are files).
_AWK_PROGRAM_LETTERS = frozenset({"f"})
_AWK_PROGRAM_LONG_FLAGS = frozenset({"--source", "--file"})
# Awk flags that take some other value (variable assignment, FS) and must
# not absorb a positional as if it were the program. Tracked separately so
# attached forms like ``-vFOO=bar`` don't trigger ``has_program_flag``.
_AWK_VALUE_LETTERS = frozenset({"v", "F"})


def _extract_awk_inplace_targets(tokens: list[str]) -> list[str]:
    """``awk -i inplace`` / ``gawk -i inplace`` file args.

    The awk program may be supplied as the first inline positional, via
    short ``-f script.awk`` / attached ``-fscript.awk``, or via long form
    ``--source=...`` / ``--file=...``. When any program-supplying flag is
    detected, every positional is a file target. Other value-taking
    flags (``-v`` assignment, ``-F`` field separator) consume a value but
    do not displace the inline program.
    """
    if not tokens:
        return []
    cmd = _command_basename(tokens)
    if cmd not in {"awk", "gawk"}:
        return []
    found_inplace = False
    has_program_flag = False
    skip_next = False
    positionals: list[str] = []
    options_ended = False
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if skip_next:
            skip_next = False
            i += 1
            continue
        # ``--`` end-of-options marker: every later token is a positional.
        if not options_ended and tok == "--":
            options_ended = True
            i += 1
            continue
        if options_ended:
            positionals.append(tok)
            i += 1
            continue
        # ``-i inplace`` (separate tokens) — gawk's documented form.
        if tok == "-i" and i + 1 < len(tokens) and tokens[i + 1] == "inplace":
            found_inplace = True
            i += 2
            continue
        # ``-iinplace`` (attached value) — gawk also accepts this. Without
        # this case, an agent can run ``awk -iinplace '{print}' file`` and
        # the heuristic returns no targets, silently allowing the in-place
        # mutation.
        if tok == "-iinplace":
            found_inplace = True
            i += 1
            continue
        # ``--include inplace`` (separate) and ``--include=inplace``
        # (attached) are gawk's long-form equivalents of ``-i inplace``.
        # Must be checked BEFORE the generic ``--`` long-flag handling
        # below so the in-place activation is not silently dropped.
        if tok == "--include" and i + 1 < len(tokens) and tokens[i + 1] == "inplace":
            found_inplace = True
            i += 2
            continue
        if tok == "--include=inplace":
            found_inplace = True
            i += 1
            continue
        # Other ``--include <lib>`` / ``--include=<lib>`` (gawk include
        # of a non-inplace library): consume the value but do NOT mark
        # in-place. Without explicit handling, the value would be left
        # in the positional list and miscategorised as a file.
        if tok == "--include" and i + 1 < len(tokens):
            i += 2
            continue
        if tok.startswith("--include="):
            i += 1
            continue
        if tok.startswith("--"):
            name = tok.split("=", 1)[0]
            if name in _AWK_PROGRAM_LONG_FLAGS:
                has_program_flag = True
                if "=" not in tok:
                    skip_next = True
            i += 1
            continue
        if tok.startswith("-") and len(tok) >= 2:
            body = tok[1:]
            for j, ch in enumerate(body):
                if ch in _AWK_PROGRAM_LETTERS:
                    has_program_flag = True
                    if j == len(body) - 1:
                        skip_next = True
                    break
                if ch in _AWK_VALUE_LETTERS:
                    if j == len(body) - 1:
                        skip_next = True
                    break
            i += 1
            continue
        positionals.append(tok)
        i += 1
    if not found_inplace:
        return []
    if has_program_flag:
        return positionals
    return positionals[1:] if len(positionals) > 1 else []


# GNU coreutils flags that change argument-position semantics. Without
# tracking these, ``cp -t dir src`` lock-checks ``src`` (the source) and
# allows an unowned write to ``dir``; ``chmod --reference=R target``
# discards ``target`` as the "mode positional" and skips the lock-check
# entirely.
_TARGET_DIR_LONG_FLAGS = frozenset({"--target-directory"})
_REFERENCE_LONG_FLAGS = frozenset({"--reference"})

# chmod symbolic-mode pattern: ``[ugoa]*([-+=][rwxXstugo]*)+(,...)*``.
# Matches mode operands like ``-w``, ``+x``, ``=rw``, ``u-r``,
# ``ugo+w``, ``u+rwx,g-x`` — distinguishing them from chmod flags
# (``-R``, ``-v``, ``-c``, ``-f``) which don't contain mode letters
# after the operator. Without this, ``chmod -w unowned.py`` is
# misclassified: ``-w`` is skipped as a flag and the only positional
# is then dropped by the ``skip_first_positional`` strategy, silently
# allowing the write.
_CHMOD_SYMBOLIC_MODE_RE = re.compile(
    r"^[ugoa]*[+\-=][rwxXstugo]*"
    r"(?:[+\-=][rwxXstugo]*)*"
    r"(?:,[ugoa]*[+\-=][rwxXstugo]*(?:[+\-=][rwxXstugo]*)*)*$"
)

# Per-utility short value-taking flags (excluding ``t``). When one of
# these letters appears in a flag bundle, it consumes the rest of the
# bundle as its argument. Without this, ``cp -St dir src dest`` would
# match ``t`` in the bundle and extract ``dir`` as the destination —
# but cp actually parses ``-S t dir src dest`` (suffix=``t``, dest=
# ``dest``), so the heuristic would lock-check the wrong path.
_UTILITY_VALUE_LETTERS: dict[str, frozenset[str]] = {
    "cp": frozenset({"S"}),
    "mv": frozenset({"S"}),
    "ln": frozenset({"S"}),
    "install": frozenset({"S", "m", "o", "g"}),
}

# Per-utility short flag letters that consume the following positional
# token as a value (``mkdir -m 755 dir``, ``touch -d "time" file``,
# ``touch -r ref file``). Without this table, the value-token (``755``,
# ``"time"``, ``ref``) is added to ``positionals`` and lock-checked,
# producing false-positive denials for legitimate commands.
#
# ``cp``/``mv``/``ln``/``install`` have their own separate handling
# above (target-directory, suffix); the letters listed here are the
# remaining value-flags that ``_extract_utility_targets`` must skip.
_UTILITY_FLAG_VALUE_LETTERS: dict[str, frozenset[str]] = {
    "touch": frozenset({"d", "r", "t"}),
    "mkdir": frozenset({"m", "Z"}),
    "chmod": frozenset(),
    "chown": frozenset(),
    "cp": frozenset({"S"}),
    "mv": frozenset({"S"}),
    "ln": frozenset({"S"}),
    "install": frozenset({"S", "m", "o", "g"}),
    "rm": frozenset(),
    "rmdir": frozenset(),
    "dd": frozenset(),
}

# Per-utility LONG-form flags that consume the next token as a value.
# Without this, ``touch --date "2026-01-01" file`` leaves the date
# literal in the positional list and the all_positional strategy
# lock-checks it as a write target (false-positive deny).
_UTILITY_FLAG_VALUE_LONG_FLAGS: dict[str, frozenset[str]] = {
    "touch": frozenset({"--date", "--reference", "--time"}),
    "mkdir": frozenset({"--mode", "--context"}),
    "chmod": frozenset({"--reference"}),
    "chown": frozenset({"--reference", "--from"}),
    "cp": frozenset({"--target-directory", "--suffix", "--reference", "--context"}),
    "mv": frozenset({"--target-directory", "--suffix", "--context"}),
    "ln": frozenset({"--target-directory", "--suffix"}),
    "install": frozenset(
        {
            "--target-directory",
            "--suffix",
            "--mode",
            "--owner",
            "--group",
            "--strip-program",
            "--context",
            # NOTE: ``--backup`` takes an OPTIONAL argument. Only the
            # ``--backup=CONTROL`` form has a value; ``--backup`` alone
            # uses the default control. Listing it here would
            # incorrectly swallow the next token (e.g. ``install -d
            # --backup dir1 dir2`` would consume ``dir1`` as the
            # backup-control value, losing dir1 from the lock-check).
        }
    ),
    "rm": frozenset(),
    "rmdir": frozenset(),
    "dd": frozenset(),
}


def _scan_short_bundle_for_target_dir(
    body: str, cmd: str
) -> tuple[str, str | None] | None:
    """Scan a short-flag bundle for the ``-t`` (target-directory) flag.

    Returns:
        ``("next_token", None)`` if the bundle ends with ``t`` (next
        argv token is the destination), ``("attached", value)`` if ``t``
        is followed by attached chars (those chars are the destination),
        or ``None`` if no ``-t`` is present (or it is consumed by a
        value-taking flag earlier in the bundle).
    """
    value_letters = _UTILITY_VALUE_LETTERS.get(cmd, frozenset())
    j = 0
    while j < len(body):
        ch = body[j]
        if ch == "t":
            if j == len(body) - 1:
                return ("next_token", None)
            return ("attached", body[j + 1 :])
        if ch in value_letters:
            # Earlier value-taking flag absorbs the rest of the bundle
            # (including any ``t`` that appears after it). The ``t`` is
            # part of that flag's value, not a separate ``-t``.
            return None
        j += 1
    return None


def _extract_utility_targets(tokens: list[str]) -> list[str]:
    """File-mutation utilities (cp/mv/rm/...): extract destination args.

    Honours GNU coreutils flags that override the positional-argument
    layout:

    - ``cp/mv/install -t DIR src``, ``cp -tDIR src``,
      ``cp --target-directory=DIR src`` — the destination directory is
      supplied via the flag, not as the trailing positional. Without
      this, the heuristic would lock-check ``src`` and miss ``DIR``.
    - ``chmod --reference=R target`` / ``chown --reference=R target`` —
      the mode/owner is supplied via the flag, so ``target`` is the
      first (and only) positional file. Without this, the
      ``skip_first_positional`` strategy would discard the file as the
      "mode positional" and the lock-check would never run.
    """
    if not tokens:
        return []
    cmd = _command_basename(tokens)
    strategy = UTILITY_STRATEGIES.get(cmd)
    if strategy is None:
        return []

    if strategy == "dd":
        out: list[str] = []
        for tok in tokens[1:]:
            if tok.startswith("of="):
                target = tok[len("of=") :]
                if target:
                    out.append(target)
        return out

    explicit_targets: list[str] = []
    has_reference_flag = False
    positionals: list[str] = []
    args = tokens[1:]
    options_ended = False
    i = 0
    while i < len(args):
        tok = args[i]
        # ``--`` end-of-options marker: every following token is a
        # positional, even if it starts with ``-``. Without this,
        # ``touch -- -owned.py`` skips ``-owned.py`` as a flag and the
        # write surface is silently allowed.
        if not options_ended and tok == "--":
            options_ended = True
            i += 1
            continue
        if options_ended:
            positionals.append(tok)
            i += 1
            continue
        # cp/mv/install/ln: detect ``-t`` anywhere in a short flag bundle
        # (``-t DIR``, ``-tDIR``, ``-fvt DIR``, ``-fvtDIR``) plus long
        # forms. Without bundle-aware detection, ``cp -fvt dir src``
        # misclassifies ``src`` as the destination.
        if cmd in {"cp", "mv", "install", "ln"}:
            if tok in _TARGET_DIR_LONG_FLAGS:
                if i + 1 < len(args):
                    explicit_targets.append(args[i + 1])
                    i += 2
                    continue
            if tok.startswith("--target-directory="):
                explicit_targets.append(tok[len("--target-directory=") :])
                i += 1
                continue
            if tok.startswith("-") and not tok.startswith("--") and len(tok) >= 2:
                body = tok[1:]
                scan = _scan_short_bundle_for_target_dir(body, cmd)
                if scan is not None:
                    kind, attached = scan
                    if kind == "next_token":
                        if i + 1 < len(args):
                            explicit_targets.append(args[i + 1])
                            i += 2
                            continue
                        i += 1
                        continue
                    if kind == "attached" and attached is not None:
                        explicit_targets.append(attached)
                        i += 1
                        continue
        # chmod/chown: ``--reference=FILE`` (or ``--reference FILE``) replaces
        # the leading mode/owner positional.
        if cmd in {"chmod", "chown"}:
            if tok in _REFERENCE_LONG_FLAGS:
                has_reference_flag = True
                i += 2 if i + 1 < len(args) else 1
                continue
            if tok.startswith("--reference="):
                has_reference_flag = True
                i += 1
                continue
        # chmod symbolic-mode operands start with ``-``/``+``/``=`` and
        # MUST be treated as the mode positional, not as flags. Without
        # this branch ``chmod -w unowned.py`` skips ``-w`` as a flag
        # and the remaining single positional is dropped by the
        # ``skip_first_positional`` strategy — silently allowing the
        # write. The pattern matches ``-w``, ``+x``, ``=rw``, ``u-r``,
        # ``ugo+w``, ``u+rwx,g-x`` etc. — distinguishing from real
        # chmod flags like ``-R`` (recursive), ``-v`` (verbose), ``-c``
        # (changes), ``-f`` (silent) which don't contain mode letters.
        if cmd == "chmod" and _CHMOD_SYMBOLIC_MODE_RE.match(tok):
            positionals.append(tok)
            i += 1
            continue
        if tok.startswith("-"):
            # Long form ``--name=value``: token includes its value.
            if tok.startswith("--") and "=" in tok:
                i += 1
                continue
            # Long form ``--name VALUE``: consume the value when the
            # flag is in the per-utility long-value table. Without
            # this, ``touch --date "2026-01-01" file`` leaves the date
            # in the positional list and ``all_positional`` lock-
            # checks it (false-positive deny).
            if tok.startswith("--"):
                long_value_flags = _UTILITY_FLAG_VALUE_LONG_FLAGS.get(cmd, frozenset())
                if tok in long_value_flags and i + 1 < len(args):
                    i += 2
                    continue
                i += 1
                continue
            # Short bundle whose last letter consumes the next token as
            # a value (``mkdir -m 755 dir``, ``touch -d "time" file``).
            # Without this skip, ``755`` / ``"time"`` would be added to
            # ``positionals`` and the ``all_positional`` strategy would
            # lock-check them — guaranteeing a false-positive deny.
            if len(tok) >= 2:
                value_letters = _UTILITY_FLAG_VALUE_LETTERS.get(cmd, frozenset())
                body = tok[1:]
                consumes_next = False
                for j, ch in enumerate(body):
                    if ch in value_letters:
                        if j == len(body) - 1:
                            # Bundle ends with value-letter: next token
                            # is the value (if any).
                            if i + 1 < len(args):
                                consumes_next = True
                        # Else: rest of bundle is the attached value.
                        break
                i += 2 if consumes_next else 1
                continue
            i += 1
            continue
        positionals.append(tok)
        i += 1

    # ``-t``/``--target-directory`` overrides every other source: the
    # destinations are the only writes the command will perform.
    if explicit_targets:
        return explicit_targets

    # ``install -d <dirs>...`` is the directory-creation mode and behaves
    # like ``mkdir -p``: every positional is a directory to create. The
    # default ``last_positional`` strategy would only lock-check the
    # last dir, letting an agent slip an unowned dir past the gate by
    # placing an owned one at the end.
    if cmd == "install" and _install_directory_mode_flag(args):
        return positionals
    # ``ln target`` (single positional) creates a link in the current
    # directory named ``./basename(target)`` — NOT to ``target``. The
    # default ``last_positional`` strategy would lock-check the source
    # path; an agent owning the source could create an unchecked link
    # in cwd.
    if cmd == "ln" and len(positionals) == 1:
        return [os.path.basename(positionals[0])]

    if strategy == "last_positional":
        return positionals[-1:] if len(positionals) >= 1 else []
    if strategy == "all_positional":
        return positionals
    if strategy == "skip_first_positional":
        if has_reference_flag:
            # Mode/owner came from ``--reference``; every positional is a file.
            return positionals
        return positionals[1:] if len(positionals) > 1 else []

    return []


def _install_directory_mode_flag(args: list[str]) -> bool:
    """True if ``install`` args include ``-d`` / ``--directory``.

    Honors the ``--`` end-of-options marker so a literal ``-d`` operand
    after ``--`` is not misread as the flag.
    """
    options_ended = False
    for tok in args:
        if not options_ended and tok == "--":
            options_ended = True
            continue
        if options_ended:
            continue
        if tok == "--directory":
            return True
        if tok.startswith("--directory="):
            return True
        if (
            tok.startswith("-")
            and not tok.startswith("--")
            and len(tok) >= 2
            and "d" in tok[1:]
        ):
            return True
    return False


# Pattern matcher for embedded write-open expressions in language one-liners.
# Captures the literal path between matching quotes immediately after the
# call name. ``\1`` enforces matching open/close quote chars.
_PYTHON_OPEN_RE = re.compile(
    r"""open\s*\(\s*(['"])([^'"]+)\1\s*,\s*['"]([waxr+]*[wax+][waxr+]*)['"]\s*[,\)]"""
)
_PYTHON_PATH_WRITE_RE = re.compile(
    r"""Path\s*\(\s*(['"])([^'"]+)\1\s*\)\s*\.\s*write_(?:text|bytes)"""
)
_NODE_FS_WRITE_RE = re.compile(
    # Boundary-aware bare-call match: ``writeFile``-family calls
    # whether bound as ``fs.writeFile`` (CommonJS), ``writeFile`` (ESM
    # ``import { writeFile } from 'node:fs'``), or via an arbitrary
    # alias (``const fsp = require('fs/promises'); fsp.writeFile``).
    # The leading ``(?:^|[^a-zA-Z0-9_$])`` allows the call to be
    # preceded by ``.``, ``)``, whitespace, or ``;`` — i.e., any
    # context that ends a JS identifier — without matching against
    # an unrelated identifier whose suffix is ``writeFile`` (e.g.,
    # ``myWriteFile``, ``customWriteFile``).
    r"""(?:^|[^a-zA-Z0-9_$])"""
    r"""(?:writeFile|writeFileSync|appendFile|appendFileSync|createWriteStream)"""
    r"""\s*\(\s*(['"])([^'"]+)\1"""
)
# ``require('fs').writeFileSync('path','x')`` — kept as a defense-in-
# depth match alongside ``_NODE_FS_WRITE_RE``. The bare-call regex
# above already catches this form (matched on the ``).writeFile`` part);
# this regex makes the require-chain intent explicit and covers
# promise-based variants ``require('fs/promises')`` and
# ``require('node:fs/promises')``.
_NODE_REQUIRE_FS_WRITE_RE = re.compile(
    r"""require\s*\(\s*['"](?:node:)?fs(?:/promises)?['"]\s*\)\s*\.\s*"""
    r"""(?:writeFile|writeFileSync|appendFile|appendFileSync|createWriteStream)"""
    r"""\s*\(\s*(['"])([^'"]+)\1"""
)
# ``fs.openSync('p', 'w')`` / ``openSync('p', 'wx')`` — file open with a
# write-mode flag. Read-only ``openSync('p', 'r')`` is NOT a write
# surface and is intentionally not matched. The flag char class
# requires at least one of ``[wax+]`` so ``'r'`` alone fails to match
# while any of ``w``/``wx``/``w+``/``a``/``ax``/``a+``/``r+``/``as``/
# ``as+``/``wx+``/``ax+`` (all write-capable, per Node's fs flag
# table) matches. Bare-call form covers ESM
# ``import { openSync } from 'node:fs'`` bindings.
_NODE_OPENSYNC_WRITE_RE = re.compile(
    r"""(?:^|[^a-zA-Z0-9_$])openSync\s*\(\s*"""
    r"""(['"])([^'"]+)\1\s*,\s*"""
    r"""(['"])[waxrs+]*[wax+][waxrs+]*\3"""
)
_RUBY_WRITE_RE = re.compile(
    r"""(?:File\.(?:write|open)|IO\.write)\s*\(\s*(['"])([^'"]+)\1"""
)
# Perl 3-arg ``open`` with explicit write-mode (``'>'`` / ``'>>'``).
# Captures the path literal in groups (1, 2). Handles both
# parenthesized ``open(my $fh, '>', '/path')`` and bareword
# ``open my $fh, '>', '/path'`` forms — bareword is idiomatic in
# modern Perl and the previous regex required ``open(`` so a
# bareword call bypassed both the literal extractor and the
# (substring) hint check.
#
# ``\s*`` inside the mode quotes mirrors the hint regex's
# whitespace tolerance: Perl strips leading/trailing whitespace
# from the mode string, so ``open(my $fh, ' > ', '/path')`` opens
# for write. Without the inner ``\s*`` the literal extractor
# missed space-padded modes and an agent legitimately writing to
# an *allowed* file fell through to the hint and was incorrectly
# denied as an unresolved write.
_PERL_OPEN_WRITE_RE = re.compile(
    r"""\bopen\b\s*\(?\s*[^,]+?,\s*"""  # open + optional ( + filehandle + ,
    r"""['"]\s*[>]+\s*['"]\s*,\s*"""  # mode literal containing `>` only
    r"""(['"])([^'"]+)\1"""  # path literal
)
# Hint regex: ``open`` call whose *mode argument* (the token after the
# filehandle and its trailing comma) is a write-mode marker — a
# ``>``-prefixed quoted string or ``q(>...)``/``qq(>...)`` quote-like
# operator. Triggers the unresolved-sentinel deny when the literal
# regex above fails (dynamic path, 2-arg ``open FH, '>file'``
# mode-in-path form, or quote-like operator).
#
# Anchoring to the mode slot (rather than scanning everywhere before
# the next terminator) avoids a false positive on read-mode opens
# whose statement happens to contain an unrelated ``>``-prefixed
# token, e.g. ``open my $fh, '<', $path or die '>'`` — the unrelated
# ``'>'`` after ``die`` previously triggered the hint and forced a
# deny on a read-only command.
#
# ``\s*`` after the bracket/quote handles Perl's leading-whitespace
# tolerance in the mode string — ``open(my $fh, " > ", $p)`` and
# 2-arg ``open FH, " >file"`` both have whitespace between the
# opening quote and ``>``, and Perl silently strips it before
# evaluating the mode. Without ``\s*`` the hint missed those forms.
#
# The filehandle class ``[^,;\n]+?`` excludes statement terminators
# (``;`` / newline) so an ``open`` token with no trailing comma in
# its own statement (1-arg ``open FH;`` or method-call ``$x->open();``)
# cannot greedily span into a *later* statement's comma+``>`` token.
# Without this bound the hint would falsely deny commands like
# ``perl -e "open FH; print STDERR, '>'"``.
_PERL_OPEN_WRITE_HINT_RE = re.compile(
    r"""\bopen\b\s*\(?\s*[^,;\n]+?,\s*"""  # open + optional ( + filehandle + ,
    r"""(?:qq?\s*[\(\[\{]|['"])\s*[>]"""  # mode arg quote/q-like with `>`
)

# Marks that suggest a write call whose target literal we cannot resolve
# statically — used to trigger the unresolved-sentinel deny per plan L846.
_PYTHON_WRITE_HINTS = (
    "open(",
    ".write_text",
    ".write_bytes",
    ".writelines",
)
_NODE_WRITE_HINTS = (
    # Write-call substrings only — module-import substrings are
    # deliberately NOT in this list. Hint-only on imports caused
    # false-deny on read-only commands like ``node -e "const
    # fs=require('fs'); fs.readFileSync('/repo/file')"`` — the
    # ``require('fs')`` import substring triggered sentinel-deny
    # despite no write surface.
    #
    # The write-call substrings cover the ``fs.``/aliased/ESM-bare
    # forms. The literal-target regex ``_NODE_FS_WRITE_RE`` already
    # uses a boundary class so ``fsp.writeFile``, bare
    # ``writeFile``, and ``fs.writeFile`` all match for literal
    # targets; this list is the dynamic-target fallback. Matching
    # on the bare function name (``writeFile`` / ``appendFile`` /
    # ``createWriteStream``) makes the hint binding-agnostic.
    "writeFile",
    "appendFile",
    "createWriteStream",
)
# Regex hint for ``openSync`` calls with a write-mode flag where the
# literal target is not statically extractable. Without this hint, an
# agent could write ``fs.openSync(dyn, 'w')`` (dynamic target) and
# bypass the gate. Read-only ``openSync(p, 'r')`` is NOT matched —
# the flag char class requires at least one ``[wax+]``.
_NODE_OPENSYNC_WRITE_HINT_RE = re.compile(
    r"""(?:^|[^a-zA-Z0-9_$])openSync\s*\([^)]*?"""
    r"""(['"])[waxrs+]*[wax+][waxrs+]*\1"""
)
_RUBY_WRITE_HINTS = (
    "File.write",
    "File.open",
    "IO.write",
)


# Language one-liner: cmd → script-flag mapping.
_ONELINER_SCRIPT_FLAGS: dict[str, str] = {
    "bash": "-c",
    "sh": "-c",
    "python": "-c",
    "python3": "-c",
    "node": "-e",
    "nodejs": "-e",
    "ruby": "-e",
    "perl": "-e",
}


# Per-interpreter short flag letters that take a separate-token VALUE
# (other than the script flag itself). Used by ``_find_oneliner_body``
# so a value token is not misread as the script flag.
# Example: ``python -W default -c 'open(...)'`` — ``-W`` consumes
# ``default`` as its value, NOT as the script flag.
_INTERPRETER_VALUE_LETTERS: dict[str, frozenset[str]] = {
    "python": frozenset({"W", "X", "Q"}),
    "python3": frozenset({"W", "X", "Q"}),
    "perl": frozenset({"I", "M", "F", "S", "x"}),
    "node": frozenset(),
    "nodejs": frozenset(),
    "ruby": frozenset(),
    "bash": frozenset({"O"}),
    "sh": frozenset(),
}

# Per-interpreter long-form value-taking flags (consume next token).
_INTERPRETER_VALUE_LONG_FLAGS: dict[str, frozenset[str]] = {
    "python": frozenset({"--check-hash-based-pycs"}),
    "python3": frozenset({"--check-hash-based-pycs"}),
    "node": frozenset({"--require", "--import", "--input-type"}),
    "nodejs": frozenset({"--require", "--import", "--input-type"}),
    "ruby": frozenset({"--encoding"}),
    "perl": frozenset(),
    "bash": frozenset(),
    "sh": frozenset(),
}


def _find_oneliner_body(tokens: list[str], script_flag: str) -> str | None:
    """Locate the script body for ``cmd ... -c BODY`` even with leading flags
    or attached body.

    Recognises three forms:

    - separate-token: ``-c BODY`` (next token is the body)
    - attached short: ``-cBODY`` (body glued to the flag in the same token)
    - leading interpreter flags before ``-c``/``-e`` (``python -u -c ...``)

    Skips OVER value-taking option values per
    ``_INTERPRETER_VALUE_LETTERS`` / ``_INTERPRETER_VALUE_LONG_FLAGS``
    so ``python -W -c -c "open('f','w')"`` does NOT match the first
    ``-c`` (which is ``-W``'s value) — without this skip the heuristic
    would treat the second ``-c`` as the script body and miss the
    actual write call entirely.
    """
    if not tokens or len(tokens) < 2:
        return None
    is_short_flag = len(script_flag) == 2 and not script_flag.startswith("--")
    cmd = _command_basename(tokens)
    short_value_letters = _INTERPRETER_VALUE_LETTERS.get(cmd, frozenset())
    long_value_flags = _INTERPRETER_VALUE_LONG_FLAGS.get(cmd, frozenset())
    skip_next = False
    for i in range(1, len(tokens)):
        tok = tokens[i]
        if skip_next:
            skip_next = False
            continue
        if tok == script_flag:
            if i + 1 < len(tokens):
                return tokens[i + 1]
            return None
        # Attached short-flag form (``-c<body>`` / ``-e<body>``). Restrict
        # to single-dash short flags so ``--check`` (which starts with
        # ``-c``) is NOT matched against ``-c``.
        if (
            is_short_flag
            and tok.startswith(script_flag)
            and len(tok) > len(script_flag)
            and not tok.startswith("--")
        ):
            return tok[len(script_flag) :]
        # Skip the value of an interpreter value-flag so we don't
        # misidentify it as the script flag.
        if tok.startswith("--") and "=" not in tok and tok in long_value_flags:
            skip_next = True
            continue
        if tok.startswith("-") and not tok.startswith("--") and len(tok) >= 2:
            body = tok[1:]
            for j, ch in enumerate(body):
                if ch in short_value_letters:
                    if j == len(body) - 1:
                        # Bundle ends with value-letter: next token is value.
                        skip_next = True
                    # Else: attached value, no next-token consumption.
                    break
    return None


def _extract_oneliner_targets(tokens: list[str]) -> list[str]:
    """Language one-liner ``-c``/``-e`` write-path detection.

    Recursively re-applies the shell heuristic to ``bash -c``/``sh -c``
    bodies; for ``python``/``node``/``ruby``/``perl`` bodies, runs a
    best-effort literal scan for ``open(<lit>, 'w')`` and friends. If a
    write hint is present but no literal can be extracted, returns the
    unresolved-sentinel so the caller denies conservatively (plan L846).

    The script flag is located by scanning ``tokens[1:]`` rather than
    pinning it to ``tokens[1]`` so leading interpreter flags (``python -u
    -c ...``, ``bash -x -c ...``) do not bypass the heuristic.
    """
    if not tokens:
        return []
    cmd = _command_basename(tokens)
    script_flag = _ONELINER_SCRIPT_FLAGS.get(cmd)
    if script_flag is None:
        return []
    body = _find_oneliner_body(tokens, script_flag)
    if body is None:
        return []

    if cmd in {"bash", "sh"}:
        return _extract_shell_write_paths(body)

    out: list[str] = []
    if cmd in {"python", "python3"}:
        out.extend(m.group(2) for m in _PYTHON_OPEN_RE.finditer(body))
        out.extend(m.group(2) for m in _PYTHON_PATH_WRITE_RE.finditer(body))
        if not out and any(hint in body for hint in _PYTHON_WRITE_HINTS):
            out.append(_UNRESOLVED_SENTINEL)
        return out
    if cmd in {"node", "nodejs"}:
        out.extend(m.group(2) for m in _NODE_FS_WRITE_RE.finditer(body))
        out.extend(m.group(2) for m in _NODE_REQUIRE_FS_WRITE_RE.finditer(body))
        out.extend(m.group(2) for m in _NODE_OPENSYNC_WRITE_RE.finditer(body))
        if not out and (
            any(hint in body for hint in _NODE_WRITE_HINTS)
            or _NODE_OPENSYNC_WRITE_HINT_RE.search(body) is not None
        ):
            out.append(_UNRESOLVED_SENTINEL)
        return out
    if cmd == "ruby":
        out.extend(m.group(2) for m in _RUBY_WRITE_RE.finditer(body))
        if not out and any(hint in body for hint in _RUBY_WRITE_HINTS):
            out.append(_UNRESOLVED_SENTINEL)
        return out
    if cmd == "perl":
        out.extend(m.group(2) for m in _PERL_OPEN_WRITE_RE.finditer(body))
        if not out and _PERL_OPEN_WRITE_HINT_RE.search(body) is not None:
            out.append(_UNRESOLVED_SENTINEL)
        return out

    return []


# Git global options that take a separate-token value. Long forms
# (``--git-dir=...``, etc.) are handled by ``=`` detection in
# ``_find_git_subcommand_index`` and don't need explicit listing here.
_GIT_GLOBAL_VALUE_FLAGS = frozenset(
    {
        "-C",
        "-c",
        "--git-dir",
        "--work-tree",
        "--namespace",
        "--super-prefix",
        "--exec-path",
        "--config-env",
    }
)


def _compose_git_chdir(prev: str | None, value: str) -> str | None:
    """Apply a single ``git -C <value>`` to the previously-composed path.

    Per ``git --help`` / ``man git`` ("Run as if git was started in
    <path>"): when multiple ``-C`` options are given, each subsequent
    non-absolute path is interpreted relative to the preceding one; an
    absolute path discards the previous value; an empty path is a
    no-op.

    Without composition, ``git -C pkg -C sub checkout file.py``
    overwrote ``chdir_value`` with ``sub`` and the heuristic resolved
    pathspecs against ``$cwd/sub/`` while git actually wrote into
    ``$cwd/pkg/sub/``. An agent holding only ``$cwd/sub/file.py``
    could mutate the unowned nested file under ``$cwd/pkg/sub/``.
    """
    if value == "":
        return prev
    if os.path.isabs(value):
        return value
    if prev is None:
        return value
    return os.path.join(prev, value)


def _find_git_subcommand_info(
    tokens: list[str],
) -> tuple[int | None, str | None]:
    """Return ``(subcommand_index, effective_write_base)`` for a git
    invocation.

    Skips global options before the subcommand. The returned
    ``effective_write_base`` is the directory pathspecs are resolved
    against to derive the actual write target on disk:

    * ``-C <dir>`` accumulates into the cwd (per ``git``: each
      subsequent non-absolute ``-C`` is interpreted relative to the
      preceding one; absolute resets; empty no-op — see
      ``_compose_git_chdir``).
    * ``--work-tree=<wt>`` (and the separate-token form) sets the
      working tree. When ``--work-tree`` is set, write paths resolve
      against ``work-tree`` (relative to the post-``-C`` cwd if
      relative; absolute if absolute). The last ``--work-tree`` value
      wins — git itself does not compose repeated ``--work-tree``.
    * ``--git-dir`` is INTENTIONALLY ignored. ``--git-dir`` specifies
      the location of the ``.git`` repository database, NOT the
      working tree. Treating it as a path-resolution base let an
      agent run ``git --git-dir=/tmp/junk checkout f`` and lock
      ``/tmp/junk/f`` (a junk path) while git actually wrote ``$cwd/f``.

    Examples:

    * ``git -C pkg checkout file.py`` → base ``pkg`` → writes ``pkg/file.py``
    * ``git -C pkg -C sub checkout f`` → base ``pkg/sub``
    * ``git -C pkg --work-tree=wt checkout f`` → base ``pkg/wt`` (work-tree
      ``wt`` resolved relative to post-``-C`` cwd ``pkg``); pre-fix this
      overwrote ``chdir_value`` with ``wt`` and let an agent locking
      ``wt/f`` overwrite the unowned ``pkg/wt/f``
    * ``git --git-dir=/tmp/junk checkout f`` → base ``None`` (writes ``$cwd/f``)
    """
    chdir_value: str | None = None
    work_tree_value: str | None = None
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("-"):
            return i, _git_effective_base(chdir_value, work_tree_value)
        # Long form with attached value: ``--work-tree=path`` /
        # ``--git-dir=path``.
        if tok.startswith("--") and "=" in tok:
            name, value = tok.split("=", 1)
            if name == "--work-tree" and value:
                work_tree_value = value
            # ``--git-dir`` intentionally ignored — see docstring.
            i += 1
            continue
        if tok in _GIT_GLOBAL_VALUE_FLAGS:
            if i + 1 < len(tokens):
                value = tokens[i + 1]
                if tok == "-C":
                    chdir_value = _compose_git_chdir(chdir_value, value)
                elif tok == "--work-tree":
                    work_tree_value = value
                # ``--git-dir`` intentionally ignored — see docstring.
                i += 2
                continue
            i += 1
            continue
        # Other short/long flags that take no value (``-h``, ``--help``,
        # ``--version``, ``--no-pager``, ``--paginate``, ``--bare``,
        # ``--literal-pathspecs``, ``--glob-pathspecs``, etc.).
        i += 1
    return None, _git_effective_base(chdir_value, work_tree_value)


def _git_effective_base(
    chdir_value: str | None, work_tree_value: str | None
) -> str | None:
    """Compose the effective path-resolution base from ``-C`` and ``--work-tree``.

    Per ``git`` semantics, ``-C`` changes the effective cwd, then
    ``--work-tree`` (if set) sets the working tree relative to that
    new cwd. Write paths resolve against the working tree if set,
    otherwise the cwd.
    """
    if work_tree_value is None:
        return chdir_value
    if os.path.isabs(work_tree_value):
        return work_tree_value
    if chdir_value is None:
        return work_tree_value
    return os.path.join(chdir_value, work_tree_value)


def _find_git_subcommand_index(tokens: list[str]) -> int | None:
    """Return the index of the first non-global-option token after ``git``.

    Backwards-compatible wrapper around ``_find_git_subcommand_info``;
    callers that don't care about the ``-C`` value can keep using this.
    """
    idx, _ = _find_git_subcommand_info(tokens)
    return idx


def _extract_git_targets(tokens: list[str]) -> list[str]:
    """git checkout/restore/apply/stash apply/pop write-path extraction.

    Honours git global options that come before the subcommand
    (``git -C /repo checkout f``, ``git --work-tree=. apply p``) so the
    subcommand-position heuristic doesn't get fooled into matching a
    flag against ``GIT_WRITE_SUBCOMMANDS``.

    When ``-C dir`` (or ``--work-tree=dir``) is present, relative
    pathspecs are resolved against that directory by joining ``dir/``
    onto each emitted write target. Without this, ``git -C pkg
    checkout file.py`` lock-checks ``$cwd/file.py`` while git actually
    writes ``$cwd/pkg/file.py`` — an agent holding the parent-level
    file lock would silently bypass the gate.
    """
    if not tokens or _command_basename(tokens) != "git" or len(tokens) < 2:
        return []
    sub_idx, chdir_value = _find_git_subcommand_info(tokens)
    if sub_idx is None:
        return []
    sub = tokens[sub_idx]
    if sub not in GIT_WRITE_SUBCOMMANDS:
        return []
    rest = tokens[sub_idx + 1 :]

    def resolve(target: str) -> str:
        """Prefix ``chdir_value/`` to relative ``target`` paths."""
        if chdir_value and target != _UNRESOLVED_SENTINEL and not os.path.isabs(target):
            return os.path.join(chdir_value, target)
        return target

    if sub in {"checkout", "restore"}:
        # All positional path args after `--` (or after subcommand for
        # paths-only forms). Conservative: collect everything that does not
        # start with `-` and is not a known revision name.
        out: list[str] = []
        seen_dash_dash = False
        for tok in rest:
            if tok == "--":
                seen_dash_dash = True
                continue
            if tok.startswith("-"):
                continue
            if seen_dash_dash:
                out.append(resolve(tok))
                continue
            # Heuristic: skip the first positional (often a ref) for
            # `git checkout HEAD file`. We can't reliably distinguish
            # paths from refs without a working tree, so be conservative:
            # treat *all* positionals as potential writes.
            out.append(resolve(tok))
        return out

    if sub == "apply":
        # `git apply patch` writes paths from the diff body. We cannot
        # parse the patch here without reading it; return the unresolved
        # sentinel so the call denies unless the agent has the patch's
        # parent directory locked.
        return [_UNRESOLVED_SENTINEL]

    if sub == "stash":
        if rest and rest[0] in {"apply", "pop"}:
            return [_UNRESOLVED_SENTINEL]
        return []

    return []


def _extract_patch_targets(tokens: list[str]) -> list[str]:
    """``patch < file`` / ``patch -p1 < file`` — diff body writes."""
    if not tokens or _command_basename(tokens) != "patch":
        return []
    return [_UNRESOLVED_SENTINEL]


_SEGMENT_EXTRACTORS = (
    _extract_redirection_targets,
    _extract_tee_targets,
    _extract_inplace_targets,
    _extract_awk_inplace_targets,
    _extract_utility_targets,
    _extract_oneliner_targets,
    _extract_git_targets,
    _extract_patch_targets,
)


def _strip_fd_prefix_from_output(out: list[str]) -> str:
    """Strip a contiguous fd prefix (digits, ``&``, or ``{varname}``) from
    the end of ``out`` IFF it is preceded by whitespace (or starts ``out``).

    Used by ``_normalize_separators`` when emitting a redirect operator:
    the fd prefix that should belong to the operator was already
    appended char-by-char as part of the surrounding word. We need to
    pop it off so the emitted operator (with surrounding spaces) carries
    its prefix in a single token (``2>``/``&>``/``{fd}>``) for shlex.

    The whitespace boundary is critical: bash only treats a numeric
    prefix as an fd selector when it stands as a separate word (or at
    the start of the redirect). Attached to a word it is part of the
    filename — ``cp src -t unowned_dir123>out`` writes to
    ``unowned_dir123``, NOT to fd 123. Without the boundary check the
    heuristic stripped ``123`` and lock-checked ``unowned_dir`` and
    ``out`` separately while bash silently created the unowned
    ``unowned_dir123``.

    Returns ``""`` if the trailing chars don't form a valid fd prefix
    or are attached to a non-whitespace previous char.
    """
    if not out:
        return ""
    last = out[-1]
    # Handle multi-char appends (placeholders, escape pairs): they are
    # never an fd prefix.
    if len(last) != 1:
        return ""

    # Locate the start index of the candidate prefix span. We don't
    # mutate ``out`` until we've validated the whitespace boundary.
    if last == "&":
        start = len(out) - 1
    elif last == "}":
        # ``{varname}`` form. Walk back through varname chars to ``{``.
        j = len(out) - 2
        while j >= 0:
            ch = out[j]
            if len(ch) != 1:
                return ""
            if ch == "{":
                break
            if not (ch.isalnum() or ch == "_"):
                return ""
            j -= 1
        else:
            return ""
        if j < 0 or out[j] != "{":
            return ""
        # Body must start with letter or ``_`` (varname rule).
        if j + 1 >= len(out) - 1:
            return ""
        first = out[j + 1]
        if not (len(first) == 1 and (first.isalpha() or first == "_")):
            return ""
        start = j
    elif last.isdigit():
        j = len(out) - 1
        while j >= 0 and len(out[j]) == 1 and out[j].isdigit():
            j -= 1
        start = j + 1
    else:
        return ""

    # Whitespace boundary: the prefix must stand alone (start of out
    # or preceded by a whitespace char). Otherwise it's part of a word.
    if start > 0:
        prev = out[start - 1]
        if not (len(prev) == 1 and prev.isspace()):
            return ""

    prefix = "".join(out[start:])
    del out[start:]
    return prefix


def _normalize_separators(command: str) -> str:
    """Insert spaces around shell control operators outside quoted regions.

    ``shlex.split`` does not split on ``;``/``&&``/``||``/``|``/``&``/
    newlines/``(``/``)`` unless the operators are already surrounded by
    whitespace, so an agent-written ``true;touch unowned.py`` or
    ``touch out&`` or ``(touch unowned.py)`` previously tokenized as a
    single attached token — no extractor saw the inner command and the
    write slipped past AC #19. This helper normalises the command string
    so the downstream shlex split + segment-extractor pipeline sees the
    pipeline boundaries.

    Newlines (``\\n``) are converted to ``;`` so they survive
    ``shlex.split`` (which would otherwise eat them as whitespace and
    let ``true\\ntouch unowned`` collapse into a single segment).

    Quote tracking is required so a literal ``;``/``|``/``&`` inside a
    single- or double-quoted argument (e.g., ``echo 'a;b' > out``) is
    not treated as a control operator.

    Backslash handling is required so an agent cannot hide control
    operators by escaping a quote outside an actual quoted region.
    Without it, ``echo \\";touch f`` would look quoted to the parser,
    leave ``;`` un-split, and silently allow the unowned write — even
    though bash treats ``\\"`` as a literal ``"`` and runs the second
    command at top level.

    ANSI-C quoting (``$'...'``) supports backslash escapes inside what
    looks like a single-quoted region. Without recognising ``$'``, the
    parser miscounts quote toggles for inputs like ``$'\\''`` (an escaped
    single-quote) and incorrectly treats subsequent control operators
    as quoted, hiding them from the heuristic.

    Bare ``&`` is treated as a backgrounding operator (and split)
    EXCEPT when it follows ``>`` or ``digit+>`` (forming the fd
    redirect operators ``>&``/``n>&``). This catches ``touch out&``
    while still leaving ``2>&1`` intact for the redirection extractor.

    ``(`` / ``)`` (subshell delimiters) are split outside quotes so
    ``(touch unowned)`` becomes a segment whose first token is
    ``touch``. ``{`` / ``}`` are left to the existing whitespace-based
    tokenisation (bash already requires whitespace around brace-group
    delimiters), and ``_split_segments`` treats them as separators.
    """
    out: list[str] = []
    i = 0
    n = len(command)
    quote: str | None = None
    while i < n:
        c = command[i]
        # Backslash escape outside quotes / inside double quotes:
        # ``\<x>`` is a literal pair. Inside regular single quotes the
        # backslash is literal.
        if c == "\\" and i + 1 < n:
            if quote is None or quote == '"':
                out.append(c)
                out.append(command[i + 1])
                i += 2
                continue
        if quote is not None:
            out.append(c)
            if c == quote:
                quote = None
            i += 1
            continue
        # ANSI-C quoting: ``$'...'`` is a bash single-quoted region in
        # which ``\<x>`` is an escape. shlex.split has no ANSI-C support,
        # so a body containing ``\'`` (escaped single quote) tricks
        # shlex's plain single-quote parser into thinking the escaped
        # quote *closes* the region and the next ``'`` *opens* a new
        # one — which then never closes (ValueError) or, worse, swallows
        # subsequent control operators.
        #
        # We don't need the literal content of the ANSI-C string for
        # write-path detection (it's a positional arg, not a write
        # target). Skip the whole block and emit a placeholder identifier
        # so shlex treats it as a single non-quote token.
        if c == "$" and i + 1 < n and command[i + 1] == "'":
            j = i + 2  # past ``$'``
            while j < n:
                if command[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if command[j] == "'":
                    j += 1
                    break
                j += 1
            out.append(_ANSI_C_PLACEHOLDER)
            i = j
            continue
        if c in ('"', "'"):
            out.append(c)
            quote = c
            i += 1
            continue
        # ``#`` comment: from ``#`` (when starting a word — i.e.,
        # preceded by whitespace, separator, or start-of-input) up to
        # the next newline. Must be handled BEFORE the newline-to-``;``
        # rewrite below: otherwise ``# foo\ntouch f`` becomes
        # ``# foo ; touch f``, and shlex's comment handling (which
        # only terminates at a newline) treats the entire remainder as
        # comment — silently dropping the second command.
        if c == "#" and (
            i == 0
            or command[i - 1].isspace()
            or command[i - 1] in (";", "&", "|", "(", ")")
        ):
            while i < n and command[i] != "\n":
                i += 1
            continue
        # Newlines are top-level command separators in shells; rewrite to
        # ``;`` so ``shlex.split`` does not collapse them into whitespace.
        if c == "\n":
            out.append(" ; ")
            i += 1
            continue
        if i + 1 < n:
            two = command[i : i + 2]
            if two in ("&&", "||", "|&", ";;"):
                out.append(" ")
                out.append(two)
                out.append(" ")
                i += 2
                continue
        if c == ";":
            out.append(" ")
            out.append(c)
            out.append(" ")
            i += 1
            continue
        if c == "|":
            # Preserve ``>|`` (force-clobber redirect): if the previous
            # output char is ``>`` we must NOT split the ``|`` away from
            # it. Splitting breaks the operator into ``>`` and ``|``,
            # the redirection regex no longer recognises the operator,
            # and the destination silently becomes a separate token
            # that no extractor can lock-check. Multi-char ``||``/``|&``
            # are already handled by the lookahead above.
            #
            # The ``>`` must be unescaped — the unescaped ``>|``
            # operator is normally consumed by the ``>`` handler
            # above, so any ``>`` left in ``out`` for the ``|``
            # handler to see was preserved by the escape branch
            # (``\>``). In that case the ``>`` is literal text and
            # the ``|`` is a real pipeline operator that MUST split.
            if out and out[-1] == ">" and not (len(out) >= 2 and out[-2] == "\\"):
                out.append(c)
                i += 1
                continue
            out.append(" ")
            out.append(c)
            out.append(" ")
            i += 1
            continue
        if c == "&":
            # ``&`` followed by ``>`` is the start of the ``&>`` redirect
            # operator. Pad the ``&`` with a leading space so it is
            # detached from any preceding word (``&`` is a bash
            # metacharacter and always terminates a word, even without
            # whitespace), then append it so the upcoming ``>`` handler
            # picks it up via ``_strip_fd_prefix_from_output`` and emits
            # the full operator together. Without the leading pad,
            # ``cp owned unowned&>out`` left ``unowned&`` glued in
            # ``out``; ``_strip_fd_prefix_from_output`` then refused to
            # strip the ``&`` (no whitespace boundary), so the cp
            # destination was lock-checked as the literal ``unowned&``
            # — an agent locking that junk path would be allowed to
            # write while bash actually wrote to ``unowned``.
            if i + 1 < n and command[i + 1] == ">":
                out.append(" ")
                out.append(c)
                i += 1
                continue
            # ``&`` is the trailing half of an fd redirect ONLY when it
            # IMMEDIATELY follows an UNESCAPED ``>``. ``\>&`` (escaped
            # ``>``) is literal-``>`` followed by backgrounding ``&``;
            # treating it as a redirect would split off ``&`` as part
            # of the operator and the real backgrounding op is lost.
            if (
                i > 0
                and command[i - 1] == ">"
                and not (i >= 2 and command[i - 2] == "\\")
            ):
                out.append(c)
                i += 1
                continue
            out.append(" ")
            out.append(c)
            out.append(" ")
            i += 1
            continue
        if c in ("(", ")"):
            # Subshell delimiters are control operators in bash and
            # always start/end a command segment.
            out.append(" ")
            out.append(c)
            out.append(" ")
            i += 1
            continue
        if c == ">":
            # Pad redirect operators so ``shlex.split`` (which does not
            # tokenise on ``>``) splits them off from adjacent words.
            # ``echo hi>file`` tokenises as the single token
            # ``hi>file``; without padding, the redirection extractor's
            # combined-prefix regex never matches and the unowned write
            # to ``file`` slips past the gate.
            #
            # The fd prefix (digits, ``&``, or ``{varname}``) is part of
            # the operator and was already appended to ``out``; strip it
            # back, build the full operator (including any trailing
            # ``>`` / ``&`` / ``|``), and emit it space-padded.
            prefix = _strip_fd_prefix_from_output(out)
            op = ">"
            next_i = i + 1
            if next_i < n and command[next_i] == ">":
                op += ">"
                next_i += 1
            if next_i < n and command[next_i] in "&|":
                op += command[next_i]
                next_i += 1
            out.append(" ")
            if prefix:
                out.append(prefix)
            out.append(op)
            out.append(" ")
            i = next_i
            continue
        if c == "<" and i + 1 < n and command[i + 1] == ">":
            # ``<>`` (POSIX read-write open) is a write surface — bash
            # creates the file if missing. Pad so the extractor sees it
            # as a separate token.
            prefix = _strip_fd_prefix_from_output(out)
            out.append(" ")
            if prefix:
                out.append(prefix)
            out.append("<>")
            out.append(" ")
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _command_contains_cd(cleaned: str) -> bool:
    """Cheap pre-scan: does ``cleaned`` contain a ``cd <dir>`` segment?

    Used to propagate the cd-cwd-change effect into nested
    command-substitution bodies. Without this propagation, a recursive
    call on a backtick / ``$()`` body would not know the parent
    pipeline already ran a ``cd``, and a relative write inside the
    body would be lock-checked against the original payload ``cwd``.
    """
    try:
        normalized = _normalize_separators(cleaned)
        tokens = shlex.split(normalized, posix=True, comments=True)
    except ValueError:
        return False
    for seg in _split_segments(tokens):
        stripped = _strip_env_assignments(seg)
        stripped = _strip_leading_reserved(stripped)
        stripped, wrapper_cwd_changed, _, _ = _unwrap_execution_prefix(stripped)
        if wrapper_cwd_changed:
            return True
        if _is_cd_segment(stripped):
            return True
    return False


def _extract_shell_write_paths(
    command: str, *, outer_cd_seen: bool = False
) -> list[str]:
    """Apply the heuristic table to ``command`` and return all write targets.

    Best-effort: shlex parse errors return an empty list (the dangerous-
    command and disallowed-tool checks still run). Callers decide whether
    to deny based on the targets returned.

    ``outer_cd_seen`` is used by recursive calls on command-substitution
    bodies. If the parent pipeline ran a ``cd`` before the substitution,
    the body's relative writes inherit that cwd change; the body
    extractor runs with ``cd_seen`` pre-set so its relative targets are
    treated as unresolved.

    Pre-processing:

    1. Command substitutions (```...``` and ``$(...)``) are extracted
       and processed recursively. ``echo `touch unowned``` and
       ``echo "$(touch unowned)"`` would otherwise leave the inner
       ``touch`` invisible to every extractor.
    2. The remainder is run through ``_normalize_separators`` to expose
       control operators that ``shlex.split`` would otherwise glue to
       adjacent tokens.

    Per-segment processing:

    1. Leading env-assignments, reserved words (``then``/``do``/...),
       and execution-wrapper prefixes (``env``/``sudo``/``exec``/...)
       are stripped/unwrapped so the extractor sees the real command
       word.
    2. ``cd <dir>`` segments mark all subsequent segments as
       cwd-changed; relative write targets after a ``cd`` are emitted
       as ``_UNRESOLVED_SENTINEL`` so the lock-check fails closed —
       the heuristic cannot reliably re-resolve relative paths once
       the shell's cwd has shifted.
    """
    # Strip ``#`` comments BEFORE substitution extraction so a
    # commented-out substitution (``# $(touch unowned)\\necho ok``)
    # is NOT extracted and processed as if bash would execute it.
    command = _strip_shell_comments(command)
    cleaned, sub_bodies = _extract_command_substitutions(command)

    # If the outer pipeline already changed cwd, OR the cleaned outer
    # command itself runs a cd, propagate that into the recursive
    # processing of substitution bodies.
    bodies_cd_seen = outer_cd_seen or _command_contains_cd(cleaned)

    out: list[str] = []
    for body in sub_bodies:
        out.extend(_extract_shell_write_paths(body, outer_cd_seen=bodies_cd_seen))

    normalized = _normalize_separators(cleaned)
    try:
        # ``comments=True`` matches bash's handling of ``#`` as
        # comment-marker so an agent cannot bypass the gate by
        # appending ``# "`` (which would otherwise raise ValueError on
        # an unclosed quote and let the hook silently allow). On
        # ValueError (genuinely malformed input) fail closed by
        # appending the unresolved sentinel.
        tokens = shlex.split(normalized, posix=True, comments=True)
    except ValueError:
        out.append(_UNRESOLVED_SENTINEL)
        return out

    cd_seen = outer_cd_seen
    for segment_raw in _split_segments(tokens):
        # Strip env-assignments and reserved words in a loop so a
        # reserved word followed by a ``KEY=VALUE`` token (``then
        # VAR=val touch f``) is fully unwrapped — without the loop the
        # ``VAR=val`` ends up as the segment's command word and no
        # extractor matches.
        segment = _strip_segment_prefix(segment_raw)
        segment, wrapper_cwd_changed, wrapper_writes, wrapper_bodies = (
            _unwrap_execution_prefix(segment)
        )
        if wrapper_cwd_changed:
            # ``env -C dir cmd`` / ``sudo --chdir dir cmd``: the wrapper
            # itself changed cwd before invoking the inner command, so
            # the inner command's relative writes inherit the change.
            cd_seen = True
        # Redirection scans the full segment regardless of cmd word.
        seg_targets: list[str] = list(_extract_redirection_targets(segment))
        # Wrapper-side writes (``time -o file cmd`` writes ``file``).
        seg_targets.extend(wrapper_writes)
        # Wrapper-supplied shell bodies (``env -S 'touch f'``): each is
        # processed recursively and its inner write targets join the
        # segment's gate. Pass through ``cd_seen`` so the body's
        # relative writes respect any prior ``cd``.
        for body in wrapper_bodies:
            seg_targets.extend(_extract_shell_write_paths(body, outer_cd_seen=cd_seen))
        if not segment:
            if seg_targets:
                _append_targets_with_cd(out, seg_targets, cd_seen)
            continue
        # Brace expansion in ANY token of the segment is unresolvable.
        # ``{touch,unowned}`` is the canonical bypass: bash expands to
        # ``touch unowned`` but shlex sees one token, no extractor
        # matches, and the heuristic emits no targets. Detecting brace
        # expansion at segment level catches both ``{cmd,args...}``
        # (command-position) and ``cmd {a,b}`` (arg-position; the
        # ``_check_lock`` glob check would also catch the arg form,
        # but emitting the unresolved sentinel here makes the
        # behaviour uniform for command-position braces too).
        if any(_has_brace_expansion(tok) for tok in segment):
            seg_targets.append(_UNRESOLVED_SENTINEL)
            _append_targets_with_cd(out, seg_targets, cd_seen)
            continue
        # cwd-changing builtins / ``pushd`` / ``popd`` invalidate
        # relative-path resolution for following segments.
        if _is_cd_segment(segment):
            cd_seen = True
            if seg_targets:
                _append_targets_with_cd(out, seg_targets, cd_seen)
            continue
        # Strip redirect operators+targets before passing to the
        # non-redirect extractors. ``_extract_redirection_targets``
        # already collected the redirect surfaces above; the non-
        # redirect extractors should not see those tokens or they
        # leak into utility/oneliner positional lists. ``cp owned
        # unowned&>out`` would otherwise have its cp-destination
        # lock-check delegated to the redirect target ``out`` (last
        # positional after surviving ``&>`` token), letting an agent
        # bypass by locking a redirect target it owns while bash
        # writes to the unowned ``unowned``.
        non_redirect_segment = _drop_redirections(segment)
        for extractor in _SEGMENT_EXTRACTORS:
            if extractor is _extract_redirection_targets:
                continue
            seg_targets.extend(extractor(non_redirect_segment))
        if seg_targets:
            _append_targets_with_cd(out, seg_targets, cd_seen)
    return out


def _append_targets_with_cd(out: list[str], targets: list[str], cd_seen: bool) -> None:
    """Append ``targets`` to ``out``, replacing relative paths with the
    unresolved sentinel when a ``cd`` has been seen earlier in the pipeline.

    Once ``cd <dir>`` runs, the shell's effective cwd no longer matches
    the payload ``cwd`` the hook receives. Resolving a subsequent
    relative path against the original cwd would lock-check the wrong
    file, so we deny conservatively.
    """
    if not cd_seen:
        out.extend(targets)
        return
    for target in targets:
        if target == _UNRESOLVED_SENTINEL or os.path.isabs(target):
            out.append(target)
        else:
            out.append(_UNRESOLVED_SENTINEL)


# ---------------------------------------------------------------------------
# Dangerous-command detection (mirrors dangerous_commands.py for parity)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Main decision logic
# ---------------------------------------------------------------------------


def _command_string(tool_input: dict[str, Any]) -> str:
    """Pull the command string out of a Codex bash-tool-input dict.

    Codex shell tools may surface either a raw string under ``command`` or
    an argv list (plan L839 TBD). Treat lists as joined argv for the
    heuristic; callers that need exact tokenization re-parse via shlex.
    """
    cmd = tool_input.get("command")
    if isinstance(cmd, str):
        return cmd
    if isinstance(cmd, list):
        return " ".join(shlex.quote(str(p)) for p in cmd)
    return ""


# Codex apply_patch envelope headers — source-of-truth file:
# ``*** Update File: path`` / ``*** Add File: path`` / ``*** Delete File: path``.
# Path capture uses ``[^\t\r\n]+`` so a trailing tab-separated diff
# timestamp (``+++ b/path\t2026-05-08 ...``) does NOT get folded into
# the path — Amp parity (mala-safety.ts uses the same boundary) and a
# bypass otherwise: an agent could lock the literal ``path\t2026-...``
# while ``patch`` writes to ``path``.
_APPLY_PATCH_ENVELOPE_RE = re.compile(
    r"^\*\*\*\s+(?:Update|Add|Delete|Move)\s+File:\s+([^\t\r\n]+)",
    re.MULTILINE,
)
# Codex apply_patch rename destination — paired with a preceding
# ``*** Update File: <old>`` block. The destination is what gets written,
# so it MUST be lock-checked too: a patch that updates ``old.py`` and
# moves it to ``new.py`` writes BOTH paths. Without this, holding the
# ``old.py`` lock would let the agent create an unowned ``new.py``.
_APPLY_PATCH_MOVE_TO_RE = re.compile(
    r"^\*\*\*\s+Move\s+to:\s+([^\t\r\n]+)",
    re.MULTILINE,
)
# Unified-diff destination header. ``b/`` prefix is the conventional
# git-format prefix; the regex tolerates either form. The ``\S+`` body
# matches ``patch(1)``'s behaviour: patch truncates the path at the
# first whitespace (covering both tab-separated AND space-separated
# trailing timestamps — ``+++ b/file 2026-05-08`` writes to ``file``,
# not ``file 2026-05-08``). Trade-off: paths with literal whitespace
# in unquoted diff headers are not supported here, which matches what
# ``git diff`` produces (it quotes such paths).
_DIFF_DEST_HEADER_RE = re.compile(r"^\+\+\+\s+(?:b/)?(\S+)", re.MULTILINE)
_DIFF_SRC_HEADER_RE = re.compile(r"^---\s+(?:a/)?(\S+)", re.MULTILINE)


def _apply_patch_paths(tool_input: dict[str, Any]) -> list[str]:
    """Extract write-target paths for the apply_patch / file-edit branch.

    Accepts both shapes Codex emits in practice:

    1. Direct path keys (``path``/``file_path``/``filename``/``target``/``paths``).
    2. An embedded patch body under ``command``/``input``/``patch``/
       ``patch_text``/``body``. The body uses Codex envelope headers
       (``*** Update File: path``, etc.) and/or unified-diff headers
       (``+++ b/path``, ``--- a/path``); both are parsed and the target
       paths are returned.

    Parity reference: ``plugins/amp/mala-safety.ts`` (L307-L313, L1329-L1339)
    parses the same patch shapes for the Amp ``apply_patch`` tool. Codex
    must match this surface so AC #9 ("unowned file edits are denied")
    holds even when the caller does not split the patch out into a
    structured ``path`` field.
    """
    out: list[str] = []
    for key in ("path", "file_path", "filename", "target"):
        val = tool_input.get(key)
        if isinstance(val, str) and val:
            out.append(val)
    paths = tool_input.get("paths")
    if isinstance(paths, list):
        for p in paths:
            if isinstance(p, str) and p:
                out.append(p)

    for key in ("command", "input", "patch", "patch_text", "body", "diff"):
        body = tool_input.get(key)
        if not isinstance(body, str) or not body:
            continue
        for m in _APPLY_PATCH_ENVELOPE_RE.finditer(body):
            target = m.group(1).strip()
            if target:
                out.append(target)
        for m in _APPLY_PATCH_MOVE_TO_RE.finditer(body):
            target = m.group(1).strip()
            if target:
                out.append(target)
        for m in _DIFF_DEST_HEADER_RE.finditer(body):
            target = m.group(1).strip()
            # ``+++ /dev/null`` marks a delete; pair it with the ``---``
            # source path below.
            if target and target != "/dev/null":
                out.append(target)
        for m in _DIFF_SRC_HEADER_RE.finditer(body):
            target = m.group(1).strip()
            if target and target != "/dev/null":
                out.append(target)

    # Dedup while preserving order so deny messages reference the first
    # matched form.
    seen: set[str] = set()
    deduped: list[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def _gate_write_targets(
    targets: Iterable[str],
    *,
    agent_id: str,
    lock_dir: str,
    repo_namespace: str,
    cwd: str | None,
) -> dict[str, Any] | None:
    """Run the lock check on each target; return the first deny."""
    for target in targets:
        deny = _check_lock(target, agent_id, lock_dir, repo_namespace, cwd)
        if deny is not None:
            return deny
    return None


def decide(input_payload: dict[str, Any]) -> dict[str, Any]:
    """Compute the hook decision for a single PreToolUse payload.

    Returns the ``{decision, hookSpecificOutput}`` dict that ``main`` writes
    to stdout. Pure function: tests can drive it directly.
    """
    tool_name = str(input_payload.get("tool_name") or "")
    tool_input = input_payload.get("tool_input") or {}
    if not isinstance(tool_input, dict):
        tool_input = {}
    session_id = str(input_payload.get("session_id") or "")
    cwd = input_payload.get("cwd")
    cwd_str: str | None = cwd if isinstance(cwd, str) and cwd else None

    env = _resolve_env(session_id)
    disallowed = _disallowed_tools(env)

    # Disallowed tools: deny first regardless of branch (plan L833).
    if tool_name in disallowed:
        return _deny(_msg_disallowed(tool_name))

    # Branches that need lock evaluation require all three env vars.
    agent_id = env.get("MALA_AGENT_ID")
    lock_dir = env.get("MALA_LOCK_DIR")
    repo_namespace = env.get("MALA_REPO_NAMESPACE")

    if tool_name in SHELL_TOOL_NAMES:
        command = _command_string(tool_input)
        # 1. Dangerous-command detection (does not require lock env).
        # Also walks into backtick / ``$()`` substitution bodies — the
        # outer command's substring/token check would otherwise miss a
        # destructive ``git`` invocation hidden inside a substitution
        # (``echo `git -C /tmp stash``` runs the inner stash for real).
        dangerous = _detect_dangerous_pattern_recursive(command)
        if dangerous is not None:
            return _deny(_msg_dangerous(dangerous))
        # 2. Heuristic write-path detection.
        write_targets = _extract_shell_write_paths(command)
        if not write_targets:
            return _allow()
        if not (agent_id and lock_dir and repo_namespace):
            return _deny(_MSG_ENV_MISSING)
        deny = _gate_write_targets(
            write_targets,
            agent_id=agent_id,
            lock_dir=lock_dir,
            repo_namespace=repo_namespace,
            cwd=cwd_str,
        )
        return deny if deny is not None else _allow()

    if tool_name in FILE_EDIT_TOOL_NAMES:
        if not (agent_id and lock_dir and repo_namespace):
            return _deny(_MSG_ENV_MISSING)
        targets = _apply_patch_paths(tool_input)
        if not targets:
            # Fail-closed: with sandbox=danger-full-access this hook is
            # the single safety gate, so an apply_patch payload that the
            # heuristic cannot decode must not be silently allowed.
            # Parity: ``plugins/amp/mala-safety.ts`` L1329-L1339.
            return _deny(
                "apply_patch payload has no extractable target paths; "
                "refusing fail-open"
            )
        deny = _gate_write_targets(
            targets,
            agent_id=agent_id,
            lock_dir=lock_dir,
            repo_namespace=repo_namespace,
            cwd=cwd_str,
        )
        return deny if deny is not None else _allow()

    # MCP tools and any other tool name: only the disallowed-tools check
    # applied (handled above). Allow otherwise.
    return _allow()


# ---------------------------------------------------------------------------
# Selftest marker (Phase E5 / AC-5 live-hook proof)
# ---------------------------------------------------------------------------

SELFTEST_MARKER_ENV = "MALA_CODEX_HOOK_SELFTEST_MARKER"
"""Env var the Codex provider's selftest sets to a writable file path.

When this variable is present the hook writes a JSON marker
``{"mala_codex_hook": "loaded", "version": "<sha256>"}`` to that path
before processing the payload. The marker is the live-hook proof the
provider's selftest reads back to confirm Codex actually invoked
``mala-codex-pre-tool-use`` end-to-end (not just that the script
hashed correctly via the module-identity probe). Mirrors Amp's
``session.start`` sentinel marker posture in
``src/infra/clients/amp_provider.py``.
"""


SELFTEST_MARKER_DIR_ENV = "MALA_CODEX_HOOK_SELFTEST_MARKER_DIR"
"""Env var pointing at a directory for *per-event* selftest markers.

When set (in addition to / instead of :data:`SELFTEST_MARKER_ENV`),
each hook invocation writes ``<hook_event_name>.json`` inside the
directory (atomic via temp + rename). Lets the provider's live probe
distinguish SessionStart vs PreToolUse firing, so both per-handler
``[hooks.state."<key>"]`` trust states are exercised end-to-end:
SessionStart's path proves ``...:session_start:0:0`` is trusted, and
the PreToolUse path proves ``...:pre_tool_use:0:0`` is trusted (the
two are independently keyed in Codex, so trusting one does not imply
the other). The single-file :data:`SELFTEST_MARKER_ENV` is retained
for the direct-spawn step and as a backward-compat fallback.
"""


SELFTEST_TARGET_EVENT_ENV = "MALA_CODEX_HOOK_SELFTEST_TARGET_EVENT"
"""Env var naming a hook event whose selftest path should ``decision=block``.

When the hook fires for a matching event AND the marker dir env is
set AND this env names the event, the hook returns the block / abort
decision so the live probe's turn aborts after the marker is
written. Used by the PreToolUse-driven dispatch probe — the LLM
emits a shell tool call, PreToolUse hook fires, writes
``PreToolUse.json``, returns ``decision=block`` + ``continue=false``,
and the provider interrupts the turn from the client side. Cap on
LLM cost: bounded to one shot at the first tool call.
"""


SELFTEST_IDENTITY_MODULES: tuple[str, ...] = (
    "src.infra.hooks.codex_pre_tool_use",
    "src.infra.hooks.dangerous_commands",
    "src.infra.tool_config",
    "src.infra.tools.locking",
    "src.infra.tools.env",
)
"""Modules whose source bytes form this hook's identity.

Mirrors :data:`src.infra.clients.codex_provider._HOOK_IDENTITY_MODULES`
exactly so the marker emitted here matches the expected hash the
provider's selftest computes. The two tuples must stay in lockstep
(import-linter forbids the provider from importing this hook module
directly, so we duplicate intentionally and pin the equality in tests).
"""


def _compute_selftest_identity_hash() -> str | None:
    """Return the combined SHA-256 over the identity modules, or ``None``.

    Mirrors :func:`src.infra.clients.codex_provider._compute_combined_module_hash`'s
    iteration / length-prefix scheme byte-for-byte so the same five
    module files yield the same digest in both processes. Returns
    ``None`` when any module cannot be located or read so a
    constrained interpreter (locked-down site-packages, missing
    dependency module) does not raise mid-hook — the provider treats
    a marker without a matching version field as
    :class:`CodexHookNotActiveReason.VERSION_MISMATCH`.
    """
    import hashlib
    import importlib.util

    digest = hashlib.sha256()
    for name in SELFTEST_IDENTITY_MODULES:
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            return None
        try:
            data = Path(spec.origin).read_bytes()
        except OSError:
            return None
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def _atomic_write_json(target: Path, data: dict[str, Any]) -> None:
    """Write ``data`` as JSON to ``target`` atomically.

    Sibling temp file + ``os.replace`` keeps the directory-entry
    swap atomic on POSIX so the provider's polling reader never
    observes a partial / empty file (regression: review-4 P1 TOCTOU).
    I/O failures are swallowed by callers; the provider treats an
    absent / unreadable marker as the fail-closed signal.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=str(target.parent),
        delete=False,
    ) as tmp:
        tmp.write(json.dumps(data))
        tmp.flush()
        try:
            os.fsync(tmp.fileno())
        except OSError:
            pass
        tmp_name = tmp.name
    os.replace(tmp_name, target)


def _emit_selftest_marker_if_requested(event_name: str = "") -> None:
    """Write the live-hook marker(s) when selftest env var(s) are set.

    Two complementary markers:

      * Single-file marker at :data:`SELFTEST_MARKER_ENV` —
        event-agnostic; written for every invocation when the env
        var is set. Used by the direct-spawn / module-identity
        / single-event SessionStart probe paths.
      * Per-event marker at
        ``$SELFTEST_MARKER_DIR_ENV/<hook_event_name>.json`` — used
        by the live PreToolUse-driven dispatch probe so SessionStart
        and PreToolUse firing can be distinguished. Each event
        writes its own file so trusting ``...:session_start:0:0``
        does not implicitly mark ``...:pre_tool_use:0:0`` validated
        (regression: review-4 P1 — the per-handler trust paths are
        independent in Codex).

    Best-effort: I/O errors are swallowed so a misconfigured marker
    path on a real Codex run does not break the hook's primary
    decision flow. The provider's selftest treats an absent / unreadable
    marker as the fail-closed signal
    (:class:`CodexHookNotActiveReason.HOOK_MARKER_MISSING`).
    """
    payload = {
        "mala_codex_hook": "loaded",
        "version": _compute_selftest_identity_hash(),
        "event_name": event_name or "",
    }

    marker_path = os.environ.get(SELFTEST_MARKER_ENV)
    if marker_path:
        try:
            _atomic_write_json(Path(marker_path), payload)
        except OSError:
            pass

    marker_dir = os.environ.get(SELFTEST_MARKER_DIR_ENV)
    if marker_dir and event_name:
        # Restrict to a known set so a malicious / accidental
        # ``hook_event_name`` value cannot escape the marker dir
        # via path-separator characters.
        safe_events = {
            "PreToolUse",
            "SessionStart",
            "UserPromptSubmit",
            "PostToolUse",
            "PreCompact",
            "PostCompact",
            "Stop",
            "PermissionRequest",
        }
        if event_name in safe_events:
            try:
                _atomic_write_json(Path(marker_dir) / f"{event_name}.json", payload)
            except OSError:
                pass


def _selftest_mode_active() -> bool:
    """Return True iff the provider's selftest set either marker env."""
    return bool(
        os.environ.get(SELFTEST_MARKER_ENV) or os.environ.get(SELFTEST_MARKER_DIR_ENV)
    )


def _selftest_target_event_matches(event_name: str) -> bool:
    """Return True iff the target-event env var names ``event_name``."""
    target = os.environ.get(SELFTEST_TARGET_EVENT_ENV)
    return bool(target) and target == event_name


def _handle_session_start_event() -> dict[str, Any]:
    """Handle a Codex ``SessionStart`` hook payload.

    Three operating modes:

      * **Selftest mode targeting SessionStart** — the live SessionStart
        probe sets :data:`SELFTEST_MARKER_ENV` (single-file marker)
        but not :data:`SELFTEST_TARGET_EVENT_ENV`. We respond with
        ``continue: false`` so the turn aborts BEFORE the LLM is
        contacted. No tokens spent.
      * **Selftest mode targeting PreToolUse** — the live PreToolUse
        probe sets :data:`SELFTEST_MARKER_DIR_ENV` and
        :data:`SELFTEST_TARGET_EVENT_ENV` to ``"PreToolUse"``. The
        SessionStart hook must let the turn proceed (``continue=true``)
        so Codex reaches the model and emits a tool call which then
        triggers PreToolUse. Per-event marker is still emitted so
        the provider can verify SessionStart ALSO fired (proves
        ``...:session_start:0:0`` trust state).
      * **Production mode** — neither selftest env is set; return
        ``continue=true`` for normal session flow.

    The output shape follows the Codex ``session-start.command.output``
    schema (``HookUniversalOutputWire`` flattened in
    :class:`SessionStartCommandOutputWire`); fields not used here
    default per Codex's serde definitions.
    """
    target = os.environ.get(SELFTEST_TARGET_EVENT_ENV)
    if _selftest_mode_active() and (
        target is None or target == "" or target == "SessionStart"
    ):
        return {
            "continue": False,
            "stopReason": "mala-codex selftest probe (continue=false aborts the turn before any LLM call)",
        }
    return {"continue": True}


def _handle_pre_tool_use_selftest() -> dict[str, Any]:
    """Return the PreToolUse selftest decision when the live probe is driving.

    The provider's live PreToolUse probe sets
    :data:`SELFTEST_TARGET_EVENT_ENV` to ``"PreToolUse"`` so the
    bundled hook can distinguish a real PreToolUse from the
    selftest's deliberate trigger. We deny the tool call (so
    nothing actually executes) AND return ``continue=false`` to
    request that Codex abort the turn — the provider also calls
    ``turn_interrupt`` from the client side as a belt-and-suspenders
    bound on LLM cost (PreToolUse ``continue=false`` is honored on
    a best-effort basis depending on Codex's turn-loop semantics
    for the version in use).

    Output shape follows ``pre-tool-use.command.output``
    (HookUniversalOutputWire + ``decision`` + ``reason`` +
    ``hookSpecificOutput``).
    """
    return {
        "continue": False,
        "stopReason": "mala-codex selftest probe (deny + abort)",
        "decision": "block",
        "reason": "mala-codex selftest probe denied the tool call to bound LLM cost",
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": "mala-codex selftest probe",
        },
    }


def main() -> int:
    """CLI entry-point: read JSON stdin, write JSON stdout, exit 0."""
    # Read the payload first so we can route the marker emission per
    # event_name (SessionStart vs PreToolUse markers go to different
    # files so the live probe can verify each per-handler trust path
    # independently). The single-file marker (env-agnostic) is also
    # emitted; either path is enough for the legacy single-event
    # probes to observe the hook ran.
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        if not isinstance(payload, dict):
            payload = {}
        invalid_json: str | None = None
    except json.JSONDecodeError as exc:
        payload = {}
        invalid_json = exc.msg

    event_name = str(payload.get("hook_event_name") or "")
    _emit_selftest_marker_if_requested(event_name)

    if invalid_json is not None:
        result = _deny(f"hook input is not valid JSON: {invalid_json}")
    elif event_name == "SessionStart":
        result = _handle_session_start_event()
    elif (
        event_name == "PreToolUse"
        and _selftest_mode_active()
        and _selftest_target_event_matches("PreToolUse")
    ):
        # Provider's live PreToolUse probe is driving — block the
        # tool call (no execution) and request turn abort.
        result = _handle_pre_tool_use_selftest()
    else:
        # Branch on the Codex hook event type for production traffic.
        # ``SessionStart`` requires a different output shape than
        # ``PreToolUse`` (Codex's ``deny_unknown_fields`` on the
        # per-event schemas would reject a PreToolUse-shaped output
        # emitted for a SessionStart turn); see
        # ``codex-rs/hooks/src/schema.rs::SessionStartCommandOutputWire``
        # vs ``PreToolUseCommandOutputWire``.
        result = decide(payload)
    sys.stdout.write(json.dumps(result))
    sys.stdout.write("\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())

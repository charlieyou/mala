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


__all__ = ["decide", "main"]


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
        "time",
        "!",
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

    Env takes precedence over the state file when both are set, per plan
    L825. Returns the merged view of ``MALA_AGENT_ID``,
    ``MALA_LOCK_DIR``, ``MALA_REPO_NAMESPACE``, ``MALA_DISALLOWED_TOOLS``.
    """
    keys = (
        "MALA_AGENT_ID",
        "MALA_LOCK_DIR",
        "MALA_REPO_NAMESPACE",
        "MALA_DISALLOWED_TOOLS",
    )
    state = _load_state_file(session_id)
    out: dict[str, str] = {}
    for key in keys:
        env_val = os.environ.get(key)
        if env_val is not None:
            # Env wins on conflict, even when explicitly empty: an
            # operator setting ``MALA_DISALLOWED_TOOLS=""`` means "clear
            # the state-file value", not "fall back to it".
            out[key] = env_val
        elif key in state:
            out[key] = state[key]
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
    expressions).
    """
    if target == _UNRESOLVED_SENTINEL:
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
    }
)

# Per-wrapper short-flag letters that consume the next token as a value.
# Used by ``_unwrap_execution_prefix`` so ``sudo -u root touch f``
# (where ``-u`` takes ``root`` as its value) does not treat ``root`` as
# the inner command.
_WRAPPER_VALUE_LETTERS: dict[str, frozenset[str]] = {
    "env": frozenset({"u", "S", "C"}),
    "sudo": frozenset({"u", "g", "h", "p", "A", "C", "D", "T", "U", "r", "t"}),
    "doas": frozenset({"u", "C"}),
    "exec": frozenset({"a"}),
    "nice": frozenset({"n"}),
    "stdbuf": frozenset({"i", "o", "e"}),
    "ionice": frozenset({"c", "n", "p"}),
    "chrt": frozenset({"p"}),
    "taskset": frozenset({"p", "c"}),
    "timeout": frozenset({"s", "k"}),
    "unshare": frozenset({"u", "U"}),
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
            "--user",
            "--setuid",
            "--setgid",
            "--propagation",
            "--root",
        }
    ),
}


def _command_basename_str(token: str) -> str:
    """Return the basename of a single token (``/usr/bin/env`` → ``env``)."""
    return os.path.basename(token) if token else token


def _skip_wrapper_flag_args(tokens: list[str], wrapper: str) -> list[str]:
    """Skip leading flag tokens (and their values) for ``wrapper``.

    Returns the suffix of ``tokens`` starting at the first non-flag
    token. Per-wrapper short and long value-flag tables ensure
    ``sudo -u root cmd`` consumes ``root`` as the value of ``-u`` and
    ``sudo --preserve-env cmd`` does NOT consume ``cmd`` (since
    ``--preserve-env`` is boolean).

    Default for unknown long flags is BOOLEAN — only flags listed in
    ``_WRAPPER_VALUE_LONG_FLAGS`` consume the next token. The previous
    "consume next non-flag token" heuristic was a P1 bypass: any
    boolean long flag would swallow the inner command word.
    """
    value_letters = _WRAPPER_VALUE_LETTERS.get(wrapper, frozenset())
    long_value_flags = _WRAPPER_VALUE_LONG_FLAGS.get(wrapper, frozenset())
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
            i += 1
            continue
        # Long form. Only consume the next token when this flag is in
        # the wrapper's value-long-flag table; otherwise treat as boolean.
        if tok.startswith("--"):
            if tok in long_value_flags and i + 1 < len(tokens):
                i += 2
                continue
            i += 1
            continue
        # Short bundle. Look for any value-taking letter.
        body = tok[1:]
        consumed_value = False
        for j, ch in enumerate(body):
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
    return tokens[i:]


def _unwrap_execution_prefix(tokens: list[str]) -> list[str]:
    """Strip leading execution-wrapper prefixes recursively.

    Walks past ``env``/``sudo``/``exec``/``command``/``nohup``/``timeout``/
    etc., their flag args, and (for ``env``-style wrappers) leading
    ``KEY=VALUE`` env-assignments, until the first token is the real
    command. Loops to handle nested wrappers (``sudo env touch f``).
    """
    iterations = 0
    while tokens and iterations < 10:
        cmd = _command_basename_str(tokens[0])
        if cmd not in _EXECUTION_PREFIXES:
            break
        new_tokens = tokens[1:]
        new_tokens = _skip_wrapper_flag_args(new_tokens, cmd)
        # Wrappers that allow ``KEY=VALUE`` before the inner cmd.
        if cmd in {"env", "sudo", "doas"}:
            new_tokens = _strip_env_assignments(new_tokens)
        # ``timeout DURATION cmd`` — drop the duration positional.
        if cmd == "timeout" and new_tokens:
            new_tokens = new_tokens[1:]
        if not new_tokens:
            break
        tokens = new_tokens
        iterations += 1
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
        # Legacy backtick command substitution.
        if c == "`" and quote != "'":
            j = i + 1
            while j < n:
                if command[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if command[j] == "`":
                    break
                j += 1
            if j < n:
                bodies.append(command[i + 1 : j])
                out.append(_CMDSUB_PLACEHOLDER)
                i = j + 1
                continue
            # Unmatched: fall through.
        # ``$(...)`` command substitution / ``$((...))`` arithmetic.
        if c == "$" and quote != "'" and i + 1 < n and command[i + 1] == "(":
            if i + 2 < n and command[i + 2] == "(":
                # ``$(( arith ))`` — pass through verbatim by counting
                # ``(`` / ``)`` balance from depth 2.
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
    for tok in tokens[1:]:
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
    for tok in tokens[1:]:
        if skip_next:
            skip_next = False
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
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if skip_next:
            skip_next = False
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
    i = 0
    while i < len(args):
        tok = args[i]
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
        if tok.startswith("-"):
            # Short bundle whose last letter consumes the next token as
            # a value (``mkdir -m 755 dir``, ``touch -d "time" file``).
            # Without this skip, ``755`` / ``"time"`` would be added to
            # ``positionals`` and the ``all_positional`` strategy would
            # lock-check them — guaranteeing a false-positive deny.
            if not tok.startswith("--") and len(tok) >= 2:
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
    r"""fs\.(?:writeFile|writeFileSync|appendFile|appendFileSync|createWriteStream)\s*\(\s*(['"])([^'"]+)\1"""
)
# ``require('fs').writeFileSync('path','x')`` — equivalent literal write
# form that does not bind ``fs`` to a separate identifier. Without this
# regex, ``node -e "require('fs').writeFileSync('unowned','x')"`` is a
# static-target write but produces zero matches and is silently allowed.
_NODE_REQUIRE_FS_WRITE_RE = re.compile(
    r"""require\s*\(\s*['"](?:node:)?fs['"]\s*\)\s*\.\s*"""
    r"""(?:writeFile|writeFileSync|appendFile|appendFileSync|createWriteStream)"""
    r"""\s*\(\s*(['"])([^'"]+)\1"""
)
_RUBY_WRITE_RE = re.compile(
    r"""(?:File\.(?:write|open)|IO\.write)\s*\(\s*(['"])([^'"]+)\1"""
)
_PERL_OPEN_WRITE_RE = re.compile(
    r"""open\s*\([^)]*?,\s*['"][>]+['"]\s*,\s*(['"])([^'"]+)\1"""
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
    "fs.writeFile",
    "fs.appendFile",
    "fs.createWriteStream",
    # ``require('fs')`` and ``require("fs")`` — also covers ``require('node:fs')``.
    "require('fs')",
    'require("fs")',
    "require('node:fs')",
    'require("node:fs")',
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


def _find_oneliner_body(tokens: list[str], script_flag: str) -> str | None:
    """Locate the script body for ``cmd ... -c BODY`` even with leading flags
    or attached body.

    Recognises three forms:

    - separate-token: ``-c BODY`` (next token is the body)
    - attached short: ``-cBODY`` (body glued to the flag in the same token)
    - leading interpreter flags before ``-c``/``-e`` (``python -u -c ...``)

    Returns the body string, or ``None`` if no script flag is present.
    Without the attached-form branch, ``python -c"open('f','w')"`` (which
    shlex tokenises as ``['python', "-copen('f','w')"]``) silently
    bypasses the heuristic — the strict ``tokens[i] == script_flag``
    check never matches and ``None`` is returned.
    """
    if not tokens or len(tokens) < 2:
        return None
    is_short_flag = len(script_flag) == 2 and not script_flag.startswith("--")
    for i in range(1, len(tokens)):
        tok = tokens[i]
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
        if not out and any(hint in body for hint in _NODE_WRITE_HINTS):
            out.append(_UNRESOLVED_SENTINEL)
        return out
    if cmd == "ruby":
        out.extend(m.group(2) for m in _RUBY_WRITE_RE.finditer(body))
        if not out and any(hint in body for hint in _RUBY_WRITE_HINTS):
            out.append(_UNRESOLVED_SENTINEL)
        return out
    if cmd == "perl":
        out.extend(m.group(2) for m in _PERL_OPEN_WRITE_RE.finditer(body))
        if not out and "open(" in body:
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


def _find_git_subcommand_index(tokens: list[str]) -> int | None:
    """Return the index of the first non-global-option token after ``git``.

    Skips global options that come before the subcommand. Without this,
    ``git -C /repo checkout f`` evaluates ``-C`` as the subcommand,
    finds no match in ``GIT_WRITE_SUBCOMMANDS``, and silently allows
    the unowned write. ``git --work-tree=. apply patch`` has the same
    bypass via the long-form ``--key=value`` shape.
    """
    i = 1
    while i < len(tokens):
        tok = tokens[i]
        if not tok.startswith("-"):
            return i
        # Long form with attached value: ``--git-dir=/abs``
        if tok.startswith("--") and "=" in tok:
            i += 1
            continue
        if tok in _GIT_GLOBAL_VALUE_FLAGS:
            i += 2
            continue
        # Other short/long flags that take no value (``-h``, ``--help``,
        # ``--version``, ``--no-pager``, ``--paginate``, ``--bare``,
        # ``--literal-pathspecs``, ``--glob-pathspecs``, etc.).
        i += 1
    return None


def _extract_git_targets(tokens: list[str]) -> list[str]:
    """git checkout/restore/apply/stash apply/pop write-path extraction.

    Honours git global options that come before the subcommand
    (``git -C /repo checkout f``, ``git --work-tree=. apply p``) so the
    subcommand-position heuristic doesn't get fooled into matching a
    flag against ``GIT_WRITE_SUBCOMMANDS``.
    """
    if not tokens or _command_basename(tokens) != "git" or len(tokens) < 2:
        return []
    sub_idx = _find_git_subcommand_index(tokens)
    if sub_idx is None:
        return []
    sub = tokens[sub_idx]
    if sub not in GIT_WRITE_SUBCOMMANDS:
        return []
    rest = tokens[sub_idx + 1 :]

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
                out.append(tok)
                continue
            # Heuristic: skip the first positional (often a ref) for
            # `git checkout HEAD file`. We can't reliably distinguish
            # paths from refs without a working tree, so be conservative:
            # treat *all* positionals as potential writes.
            out.append(tok)
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
    the end of ``out`` and return it.

    Used by ``_normalize_separators`` when emitting a redirect operator:
    the fd prefix that should belong to the operator was already
    appended char-by-char as part of the surrounding word. We need to
    pop it off so the emitted operator (with surrounding spaces) carries
    its prefix in a single token (``2>``/``&>``/``{fd}>``) for shlex.

    Returns ``""`` if the trailing chars don't form a valid fd prefix.
    """
    if not out:
        return ""
    last = out[-1]
    # Handle multi-char appends (placeholders, escape pairs): they are
    # never an fd prefix.
    if len(last) != 1:
        return ""
    if last == "&":
        out.pop()
        return "&"
    if last == "}":
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
        prefix = "".join(out[j:])
        del out[j:]
        return prefix
    if last.isdigit():
        digits: list[str] = []
        while out and len(out[-1]) == 1 and out[-1].isdigit():
            digits.append(out.pop())
        return "".join(reversed(digits))
    return ""


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
            if out and out[-1] == ">":
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
            # operator. Append the ``&`` so the upcoming ``>`` handler
            # picks it up via ``_strip_fd_prefix_from_output`` and emits
            # the full operator together.
            if i + 1 < n and command[i + 1] == ">":
                out.append(c)
                i += 1
                continue
            # ``&`` following ``>`` or ``digit+>`` is the trailing half of
            # a fd redirect (``>&``/``n>&``). Walk back over digits in
            # the input and check for ``>``.
            j = i - 1
            while j >= 0 and command[j].isdigit():
                j -= 1
            if j >= 0 and command[j] == ">":
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
        tokens = shlex.split(normalized, posix=True, comments=False)
    except ValueError:
        return False
    for seg in _split_segments(tokens):
        stripped = _strip_env_assignments(seg)
        stripped = _strip_leading_reserved(stripped)
        stripped = _unwrap_execution_prefix(stripped)
        if stripped and _command_basename(stripped) == "cd":
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
        tokens = shlex.split(normalized, posix=True, comments=False)
    except ValueError:
        # Unbalanced quotes etc. — fall through; other checks still apply.
        return out

    cd_seen = outer_cd_seen
    for segment_raw in _split_segments(tokens):
        segment = _strip_env_assignments(segment_raw)
        segment = _strip_leading_reserved(segment)
        segment = _unwrap_execution_prefix(segment)
        # Redirection scans the full segment regardless of cmd word.
        seg_targets: list[str] = list(_extract_redirection_targets(segment))
        if not segment:
            if seg_targets:
                _append_targets_with_cd(out, seg_targets, cd_seen)
            continue
        # ``cd <dir>`` is not itself a write but invalidates relative-
        # path resolution for following segments.
        if _command_basename(segment) == "cd":
            cd_seen = True
            if seg_targets:
                _append_targets_with_cd(out, seg_targets, cd_seen)
            continue
        for extractor in _SEGMENT_EXTRACTORS:
            if extractor is _extract_redirection_targets:
                continue
            seg_targets.extend(extractor(segment))
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


def _detect_dangerous_pattern(command: str) -> str | None:
    """Mirror :func:`block_dangerous_commands` for parity with the Claude hook.

    Returns the matched pattern label (for ``_msg_dangerous``) or ``None``.
    Coverage must include every gate the Claude path applies, so AC #9
    ("dangerous shell commands are denied") holds for Codex too: dangerous
    patterns, destructive git patterns, ``git commit -a``/``--all``,
    atomic-add-commit requirement, and ``git push --force``.
    """
    for pattern in DANGEROUS_PATTERNS:
        if pattern in command:
            return pattern
    for pattern in DESTRUCTIVE_GIT_PATTERNS:
        if pattern in command:
            return pattern

    # ``git commit -a``/``--all`` and atomic-add-commit (mirror
    # ``block_dangerous_commands`` in ``dangerous_commands.py``).
    command_lower = command.lower()
    if "git commit" in command_lower:
        if " --all" in command_lower or "git commit -a" in command_lower:
            return "git commit -a/--all"
        commit_index = command_lower.find("git commit")
        add_index = command_lower.find("git add")
        if add_index == -1 or add_index > commit_index:
            return "git commit without prior git add"

    # ``git push --force`` / ``-f`` (``--force-with-lease`` is the
    # documented safer alternative and is allowed).
    if "git push" in command:
        if "--force-with-lease" in command:
            pass
        elif "--force" in command or "-f " in command:
            return "git push --force"

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
_APPLY_PATCH_ENVELOPE_RE = re.compile(
    r"^\*\*\*\s+(?:Update|Add|Delete|Move)\s+File:\s+(.+?)\s*$",
    re.MULTILINE,
)
# Codex apply_patch rename destination — paired with a preceding
# ``*** Update File: <old>`` block. The destination is what gets written,
# so it MUST be lock-checked too: a patch that updates ``old.py`` and
# moves it to ``new.py`` writes BOTH paths. Without this, holding the
# ``old.py`` lock would let the agent create an unowned ``new.py``.
_APPLY_PATCH_MOVE_TO_RE = re.compile(
    r"^\*\*\*\s+Move\s+to:\s+(.+?)\s*$",
    re.MULTILINE,
)
# Unified-diff destination header. ``b/`` prefix is the conventional
# git-format prefix; the regex tolerates either form.
_DIFF_DEST_HEADER_RE = re.compile(r"^\+\+\+\s+(?:b/)?(.+?)\s*$", re.MULTILINE)
_DIFF_SRC_HEADER_RE = re.compile(r"^---\s+(?:a/)?(.+?)\s*$", re.MULTILINE)


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
        dangerous = _detect_dangerous_pattern(command)
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


def main() -> int:
    """CLI entry-point: read JSON stdin, write JSON stdout, exit 0."""
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
        if not isinstance(payload, dict):
            payload = {}
    except json.JSONDecodeError as exc:
        # Malformed input: deny fail-closed with a diagnostic reason.
        result = _deny(f"hook input is not valid JSON: {exc.msg}")
    else:
        result = decide(payload)
    sys.stdout.write(json.dumps(result))
    sys.stdout.write("\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())

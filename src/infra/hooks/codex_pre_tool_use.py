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
# parsing. We treat ``|``/``&&``/``||`` as terminators so a write target
# detected in one pipeline stage is not confused with another stage.
SHELL_SEPARATORS = frozenset({";", "&&", "||", "|", "|&", "&", "\n"})

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


_PREFIX_REDIR_RE = re.compile(r"^(?:&|[012])?>{1,2}\|?")
_DEV_NULL_LIKE = frozenset({"/dev/null", "/dev/stderr", "/dev/stdout", "/dev/tty"})
# File-descriptor duplications: ``&1``, ``&2``, ``&-`` (close).
# These are not write targets — they redirect to an existing fd or close it,
# so locking them produces false positives like denying ``cmd > out 2>&1``
# even when ``out`` is owned by the agent.
_FD_REF_RE = re.compile(r"^&[0-9\-]+$")


def _extract_redirection_targets(tokens: list[str]) -> list[str]:
    """Find ``>``/``>>``/etc. targets in a token stream.

    Handles both space-separated forms (``cmd > file`` → ``['cmd', '>', 'file']``)
    and combined-token forms (``cmd >file`` → ``['cmd', '>file']``). The
    combined form is parsed with a strict prefix regex so digits and
    ampersands inside the *filename* are preserved (the previous
    ``lstrip("&012>")`` ate any leading digit, e.g. ``>2026-log.txt``
    became ``6-log.txt``).

    Fd duplications (``2>&1``, ``>&2``, ``>&-``) are recognised and skipped
    — they redirect to a numbered fd or close one, not to a filesystem
    path, so the lock-check would deny common idioms like
    ``cmd >out.log 2>&1`` even when ``out.log`` is locked correctly.
    """
    out: list[str] = []
    for i, tok in enumerate(tokens):
        if tok in REDIR_OPERATORS and i + 1 < len(tokens):
            target = tokens[i + 1]
            if target in _DEV_NULL_LIKE or _FD_REF_RE.match(target):
                continue
            out.append(target)
            continue
        # Combined forms like ``>file``, ``>>file``, ``2>file``, ``&>file``,
        # but NOT ``2>&1``/``>&2``/``>&-`` (fd duplications).
        m = _PREFIX_REDIR_RE.match(tok)
        if m and m.end() < len(tok):
            target = tok[m.end() :]
            if target and target not in _DEV_NULL_LIKE and not _FD_REF_RE.match(target):
                out.append(target)
    return out


def _extract_tee_targets(tokens: list[str]) -> list[str]:
    if not tokens or tokens[0] != "tee":
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
    cmd = tokens[0]
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
    cmd = tokens[0]
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
        if tok == "-i" and i + 1 < len(tokens) and tokens[i + 1] == "inplace":
            found_inplace = True
            i += 2
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


def _extract_utility_targets(tokens: list[str]) -> list[str]:
    """File-mutation utilities (cp/mv/rm/...): extract destination args."""
    if not tokens:
        return []
    cmd = tokens[0]
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

    positionals = [t for t in tokens[1:] if not t.startswith("-")]

    if strategy == "last_positional":
        return positionals[-1:] if len(positionals) >= 1 else []
    if strategy == "all_positional":
        return positionals
    if strategy == "skip_first_positional":
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
    r"""fs\.(?:writeFile|writeFileSync|appendFile|appendFileSync)\s*\(\s*(['"])([^'"]+)\1"""
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
    """Locate the script body for ``cmd ... -c BODY`` even with leading flags.

    Returns the body string, or ``None`` if no script flag is present.
    Skips ahead past leading flags (``python -u -c "..."``, ``bash -x -c
    "..."``, ``node --experimental-modules -e "..."``) so the heuristic
    is not bypassed by trivial reordering.
    """
    if not tokens or len(tokens) < 3:
        return None
    for i in range(1, len(tokens) - 1):
        if tokens[i] == script_flag:
            return tokens[i + 1]
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
    cmd = tokens[0]
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


def _extract_git_targets(tokens: list[str]) -> list[str]:
    """git checkout/restore/apply/stash apply/pop write-path extraction."""
    if not tokens or tokens[0] != "git" or len(tokens) < 2:
        return []
    sub = tokens[1]
    if sub not in GIT_WRITE_SUBCOMMANDS:
        return []

    if sub in {"checkout", "restore"}:
        # All positional path args after `--` (or after subcommand for
        # paths-only forms). Conservative: collect everything that does not
        # start with `-` and is not a known revision name.
        out: list[str] = []
        seen_dash_dash = False
        for tok in tokens[2:]:
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
        if len(tokens) >= 3 and tokens[2] in {"apply", "pop"}:
            return [_UNRESOLVED_SENTINEL]
        return []

    return []


def _extract_patch_targets(tokens: list[str]) -> list[str]:
    """``patch < file`` / ``patch -p1 < file`` — diff body writes."""
    if not tokens or tokens[0] != "patch":
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


def _normalize_separators(command: str) -> str:
    """Insert spaces around shell control operators outside quoted regions.

    ``shlex.split`` does not split on ``;``/``&&``/``||``/``|`` unless the
    operators are already surrounded by whitespace, so an agent-written
    ``true;touch unowned.py`` previously tokenized as
    ``['true;touch', 'unowned.py']`` — no extractor saw ``touch`` and the
    write slipped past AC #19. This helper normalises the command string
    so the downstream shlex split + segment-extractor pipeline sees the
    pipeline boundaries.

    Quote tracking is required so a literal ``;``/``|``/``&`` inside a
    single- or double-quoted argument (e.g., ``echo 'a;b' > out``) is not
    treated as a control operator. Bare ``&`` is left alone (ambiguous
    between backgrounding and fd duplications such as ``2>&1``).
    """
    out: list[str] = []
    i = 0
    n = len(command)
    quote: str | None = None
    while i < n:
        c = command[i]
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
        if i + 1 < n:
            two = command[i : i + 2]
            if two in ("&&", "||", "|&", ";;"):
                out.append(" ")
                out.append(two)
                out.append(" ")
                i += 2
                continue
        if c in (";", "|"):
            out.append(" ")
            out.append(c)
            out.append(" ")
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _extract_shell_write_paths(command: str) -> list[str]:
    """Apply the heuristic table to ``command`` and return all write targets.

    Best-effort: shlex parse errors return an empty list (the dangerous-
    command and disallowed-tool checks still run). Callers decide whether
    to deny based on the targets returned.
    """
    normalized = _normalize_separators(command)
    try:
        tokens = shlex.split(normalized, posix=True, comments=False)
    except ValueError:
        # Unbalanced quotes etc. — fall through; other checks still apply.
        return []

    out: list[str] = []
    for segment_raw in _split_segments(tokens):
        segment = _strip_env_assignments(segment_raw)
        # Redirection scans the full segment (not just the command word).
        out.extend(_extract_redirection_targets(segment))
        if not segment:
            continue
        for extractor in _SEGMENT_EXTRACTORS:
            if extractor is _extract_redirection_targets:
                continue
            out.extend(extractor(segment))
    return out


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

"""Pure write-target extractors for the Codex pre-tool-use hook.

Extracted from ``src.infra.hooks.codex_pre_tool_use`` (plan B.3 sub-commit
T_B9). These are the side-effect-free heuristic extractors that map a
shell command (or ``apply_patch`` payload) onto the list of paths the
tool would write to. The hook applies its lock-ownership gate to each
returned path.

Branches covered:

* **redirection** — orchestrated through ``_extract_shell_write_paths``
  which delegates to the redirection extractor in
  :mod:`src.infra.hooks.codex.shell_parser`.
* **utilities** — ``cp``/``mv``/``rm``/``touch``/``mkdir``/``chmod``/
  ``chown``/``ln``/``install``/``dd``, plus ``tee``, sed/perl
  in-place, awk in-place, and language one-liner ``-c``/``-e`` write
  calls.
* **git** — ``git checkout``/``restore``/``apply``/``stash`` (with
  ``-C`` / ``--work-tree`` path-resolution base composition).
* **apply_patch** — Codex envelope (``*** Update File: ...``) and
  unified-diff (``+++ b/...``) header extraction for the
  ``apply_patch`` file-edit branch.
* **unresolved** — the ``_UNRESOLVED_SENTINEL`` returned when a write
  surface is parseable but the literal target cannot be statically
  extracted (the lock-check then always denies, per plan L846).

The hook imports the public surface of this module; unit tests under
``tests/unit/infra/hooks/codex/test_write_targets.py`` exercise each
branch independently. The full byte-identical end-to-end contract for
``codex_pre_tool_use`` is preserved by the golden corpus harness; the
focused tests pin the extractors' internal shapes so a regression here
is caught without rerunning the whole hook.
"""

from __future__ import annotations

import os
import re
import shlex
from typing import Any

from .shell_parser import (
    _drop_redirections,
    _extract_command_substitutions,
    _extract_redirection_targets,
    _normalize_separators,
    _split_segments,
    _strip_env_assignments,
    _strip_leading_reserved,
    _strip_segment_prefix,
    _strip_shell_comments,
)


__all__ = [
    "GIT_WRITE_SUBCOMMANDS",
    "UTILITY_STRATEGIES",
    "_APPLY_PATCH_ENVELOPE_RE",
    "_APPLY_PATCH_MOVE_TO_RE",
    "_AWK_PROGRAM_LETTERS",
    "_AWK_PROGRAM_LONG_FLAGS",
    "_AWK_VALUE_LETTERS",
    "_CD_BUILTINS",
    "_CHMOD_SYMBOLIC_MODE_RE",
    "_DIFF_DEST_HEADER_RE",
    "_DIFF_SRC_HEADER_RE",
    "_EXECUTION_PREFIXES",
    "_GIT_GLOBAL_VALUE_FLAGS",
    "_INTERPRETER_VALUE_LETTERS",
    "_INTERPRETER_VALUE_LONG_FLAGS",
    "_NODE_FS_WRITE_RE",
    "_NODE_OPENSYNC_WRITE_HINT_RE",
    "_NODE_OPENSYNC_WRITE_RE",
    "_NODE_REQUIRE_FS_WRITE_RE",
    "_NODE_WRITE_HINTS",
    "_ONELINER_SCRIPT_FLAGS",
    "_PERL_LONG_VALUE_FLAGS",
    "_PERL_OPEN_WRITE_HINT_RE",
    "_PERL_OPEN_WRITE_RE",
    "_PERL_VALUE_LETTERS",
    "_PRIORITY_RE",
    "_PYTHON_OPEN_RE",
    "_PYTHON_PATH_WRITE_RE",
    "_PYTHON_WRITE_HINTS",
    "_REFERENCE_LONG_FLAGS",
    "_RUBY_WRITE_HINTS",
    "_RUBY_WRITE_RE",
    "_SED_LONG_VALUE_FLAGS",
    "_SED_VALUE_LETTERS",
    "_SEGMENT_EXTRACTORS",
    "_TARGET_DIR_LONG_FLAGS",
    "_TASKSET_MASK_RE",
    "_TIMEOUT_DURATION_RE",
    "_UNRESOLVED_SENTINEL",
    "_UTILITY_FLAG_VALUE_LETTERS",
    "_UTILITY_FLAG_VALUE_LONG_FLAGS",
    "_UTILITY_VALUE_LETTERS",
    "_WRAPPER_CWD_CHANGE_LETTERS",
    "_WRAPPER_CWD_CHANGE_LONG_FLAGS",
    "_WRAPPER_OUTPUT_LETTERS",
    "_WRAPPER_OUTPUT_LONG_FLAGS",
    "_WRAPPER_SHELL_BODY_LETTERS",
    "_WRAPPER_SHELL_BODY_LONG_FLAGS",
    "_WRAPPER_VALUE_LETTERS",
    "_WRAPPER_VALUE_LONG_FLAGS",
    "_append_targets_with_cd",
    "_apply_patch_paths",
    "_command_basename",
    "_command_basename_str",
    "_command_contains_cd",
    "_compose_git_chdir",
    "_extract_awk_inplace_targets",
    "_extract_git_targets",
    "_extract_inplace_targets",
    "_extract_oneliner_targets",
    "_extract_patch_targets",
    "_extract_shell_write_paths",
    "_extract_tee_targets",
    "_extract_utility_targets",
    "_find_git_subcommand_index",
    "_find_git_subcommand_info",
    "_find_oneliner_body",
    "_git_effective_base",
    "_has_brace_expansion",
    "_has_inplace_flag",
    "_install_directory_mode_flag",
    "_is_cd_segment",
    "_scan_short_bundle_for_target_dir",
    "_skip_wrapper_flag_args",
    "_unwrap_execution_prefix",
]


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

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
        if env_val:
            out[key] = env_val
        elif key in state:
            out[key] = state[key]
    return out


def _disallowed_tools(env: dict[str, str]) -> set[str]:
    raw = env.get("MALA_DISALLOWED_TOOLS", "")
    return {t.strip() for t in raw.split(",") if t.strip()}


def _check_lock(
    target: str, agent_id: str, lock_dir: str, repo_namespace: str
) -> dict[str, Any] | None:
    """Look up the lock holder for ``target`` and return a deny dict if blocked.

    The unresolved-target sentinel always denies (plan L846: deny on
    parseable-but-unresolved write expressions).
    """
    if target == _UNRESOLVED_SENTINEL:
        return _deny(_msg_no_lock(target, agent_id))
    # Ensure the locking module reads from the configured lock dir even
    # when MALA_LOCK_DIR was sourced from the state-file fallback.
    os.environ["MALA_LOCK_DIR"] = lock_dir
    # Imported lazily so the module-level import does not pin the lock dir
    # before the env is resolved.
    from ..tools.locking import get_lock_holder

    holder = get_lock_holder(target, repo_namespace=repo_namespace)
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


def _extract_redirection_targets(tokens: list[str]) -> list[str]:
    """Find ``>``/``>>``/etc. targets in a token stream."""
    out: list[str] = []
    for i, tok in enumerate(tokens):
        if tok in REDIR_OPERATORS and i + 1 < len(tokens):
            target = tokens[i + 1]
            # Skip /dev/null and friends; these are not real lock targets.
            if target in {"/dev/null", "/dev/stderr", "/dev/stdout", "/dev/tty"}:
                continue
            out.append(target)
        # Combined forms like ">file" (no space). shlex normally splits them
        # but defensive handling keeps the heuristic robust.
        elif tok.startswith((">", "&>", "1>", "2>")) and len(tok) > 2:
            stripped = tok.lstrip("&012>")
            if stripped:
                out.append(stripped)
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
    """True if any token is a ``-`` flag bundle containing ``i`` (e.g., ``-i.bak``)."""
    for tok in tokens[1:]:
        if not tok.startswith("-") or tok.startswith("--"):
            continue
        # Strip leading dashes; check for 'i' anywhere in the bundle.
        body = tok.lstrip("-")
        if inplace_short in body:
            return True
        if tok in ("--in-place",):
            return True
    return False


def _extract_inplace_targets(tokens: list[str]) -> list[str]:
    """Trailing positional file args after a ``-i`` flag for sed/perl."""
    if not tokens:
        return []
    cmd = tokens[0]
    if cmd not in {"sed", "perl"}:
        return []
    if not _has_inplace_flag(tokens):
        return []
    # Walk flags; collect positionals. If a value-taking flag like ``-e``
    # is present, it has already consumed the script body, so every
    # positional is a file target. Otherwise the first positional is the
    # script and the rest are file targets (e.g., ``sed -i 's/x/y/' f``).
    positionals: list[str] = []
    skip_next = False
    has_value_flag = False
    for tok in tokens[1:]:
        if skip_next:
            skip_next = False
            continue
        if tok.startswith("-"):
            # Bare flags that take a value as the next token.
            if tok in {"-e", "-f", "-l", "-I", "-M", "-P"} and len(tok) == 2:
                skip_next = True
                has_value_flag = True
            continue
        positionals.append(tok)
    if has_value_flag:
        return positionals
    return positionals[1:] if len(positionals) > 1 else []


def _extract_awk_inplace_targets(tokens: list[str]) -> list[str]:
    """``awk -i inplace`` / ``gawk -i inplace`` file args."""
    if not tokens:
        return []
    cmd = tokens[0]
    if cmd not in {"awk", "gawk"}:
        return []
    found_inplace = False
    i = 1
    positionals: list[str] = []
    while i < len(tokens):
        tok = tokens[i]
        if tok == "-i" and i + 1 < len(tokens) and tokens[i + 1] == "inplace":
            found_inplace = True
            i += 2
            continue
        if tok.startswith("-"):
            # Flag with value
            if tok in {"-f", "-v", "-F"} and i + 1 < len(tokens):
                i += 2
                continue
            i += 1
            continue
        positionals.append(tok)
        i += 1
    if not found_inplace:
        return []
    # First positional is the awk program; rest are file targets.
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


def _extract_oneliner_targets(tokens: list[str]) -> list[str]:
    """Language one-liner ``-c``/``-e`` write-path detection.

    Recursively re-applies the shell heuristic to ``bash -c``/``sh -c``
    bodies; for ``python``/``node``/``ruby``/``perl`` bodies, runs a
    best-effort literal scan for ``open(<lit>, 'w')`` and friends. If a
    write hint is present but no literal can be extracted, returns the
    unresolved-sentinel so the caller denies conservatively (plan L846).
    """
    if not tokens or len(tokens) < 3:
        return []
    cmd = tokens[0]
    flag = tokens[1]
    body = tokens[2]

    if cmd in {"bash", "sh"} and flag == "-c":
        # Recurse: the body is itself a shell command.
        return _extract_shell_write_paths(body)

    out: list[str] = []
    if cmd in {"python", "python3"} and flag == "-c":
        out.extend(m.group(2) for m in _PYTHON_OPEN_RE.finditer(body))
        out.extend(m.group(2) for m in _PYTHON_PATH_WRITE_RE.finditer(body))
        if not out and any(hint in body for hint in _PYTHON_WRITE_HINTS):
            out.append(_UNRESOLVED_SENTINEL)
        return out
    if cmd in {"node", "nodejs"} and flag == "-e":
        out.extend(m.group(2) for m in _NODE_FS_WRITE_RE.finditer(body))
        if not out and any(hint in body for hint in _NODE_WRITE_HINTS):
            out.append(_UNRESOLVED_SENTINEL)
        return out
    if cmd == "ruby" and flag == "-e":
        out.extend(m.group(2) for m in _RUBY_WRITE_RE.finditer(body))
        if not out and any(hint in body for hint in _RUBY_WRITE_HINTS):
            out.append(_UNRESOLVED_SENTINEL)
        return out
    if cmd == "perl" and flag == "-e":
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


def _extract_shell_write_paths(command: str) -> list[str]:
    """Apply the heuristic table to ``command`` and return all write targets.

    Best-effort: shlex parse errors return an empty list (the dangerous-
    command and disallowed-tool checks still run). Callers decide whether
    to deny based on the targets returned.
    """
    try:
        tokens = shlex.split(command, posix=True, comments=False)
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
    for pattern in DANGEROUS_PATTERNS:
        if pattern in command:
            return pattern
    for pattern in DESTRUCTIVE_GIT_PATTERNS:
        if pattern in command:
            return pattern
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


def _apply_patch_paths(tool_input: dict[str, Any]) -> list[str]:
    """Best-effort path extraction for the apply_patch / file-edit branch."""
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
    return out


def _gate_write_targets(
    targets: Iterable[str],
    *,
    agent_id: str,
    lock_dir: str,
    repo_namespace: str,
) -> dict[str, Any] | None:
    """Run the lock check on each target; return the first deny."""
    for target in targets:
        deny = _check_lock(target, agent_id, lock_dir, repo_namespace)
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

    env = _resolve_env(session_id)
    disallowed = _disallowed_tools(env)

    # Disallowed tools: deny first regardless of branch (plan L833).
    if tool_name in disallowed:
        return _deny(_msg_disallowed(tool_name))

    # Branches that need lock evaluation require all three env vars.
    needs_lock_env = tool_name in SHELL_TOOL_NAMES or tool_name in FILE_EDIT_TOOL_NAMES
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
        )
        return deny if deny is not None else _allow()

    if tool_name in FILE_EDIT_TOOL_NAMES:
        targets = _apply_patch_paths(tool_input)
        if not targets:
            return _allow()
        if not (agent_id and lock_dir and repo_namespace):
            return _deny(_MSG_ENV_MISSING)
        deny = _gate_write_targets(
            targets,
            agent_id=agent_id,
            lock_dir=lock_dir,
            repo_namespace=repo_namespace,
        )
        return deny if deny is not None else _allow()

    # MCP tools and any other tool name: only the disallowed-tools check
    # applied (handled above). Allow otherwise.
    del needs_lock_env  # documented above; left for readability
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

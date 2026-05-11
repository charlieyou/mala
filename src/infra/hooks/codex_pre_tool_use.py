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

from .codex.shell_parser import (
    _ANSI_C_PLACEHOLDER,
    _CMDSUB_PLACEHOLDER,
    _extract_command_substitutions,
    _normalize_separators,
    _strip_shell_comments,
)
from .codex.write_targets import (
    _UNRESOLVED_SENTINEL,
    _apply_patch_paths,
    _command_basename_str,
    _extract_shell_write_paths,
    _find_git_subcommand_info,
)
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
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
        },
    }


def _deny(reason: str) -> dict[str, Any]:
    """Build a ``deny`` hook result."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        },
    }


def _resolve_env() -> dict[str, str]:
    """Resolve MALA_* values from ``os.environ``.

    Env injection is the sole supported transport (plan ``L815``,
    AC-3): :class:`CodexRuntimeBuilder.build()` overlays the full
    identity bundle (``MALA_AGENT_ID`` + ``MALA_LOCK_DIR`` +
    ``MALA_REPO_NAMESPACE``) on the per-process env that the Codex
    subprocess inherits, so the hook reads them straight from
    ``os.environ``. ``MALA_DISALLOWED_TOOLS`` is independent operator
    policy and is read the same way.

    A missing key returns ``{}`` for that slot; ``decide()`` then
    surfaces ``_MSG_ENV_MISSING`` for any branch that needs the
    lock-ownership bundle.
    """
    out: dict[str, str] = {}
    for key in ("MALA_AGENT_ID", "MALA_LOCK_DIR", "MALA_REPO_NAMESPACE"):
        env_val = os.environ.get(key)
        if env_val is not None:
            out[key] = env_val
    disallowed_env = os.environ.get("MALA_DISALLOWED_TOOLS")
    if disallowed_env is not None:
        out["MALA_DISALLOWED_TOOLS"] = disallowed_env
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
    # The locking module reads ``MALA_LOCK_DIR`` from ``os.environ`` at
    # call time. Reaffirm it here so the lock-key derivation always
    # matches the value the hook resolved from the env, even if a
    # caller later mutates the env or imports a stale resolver.
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
    cwd = input_payload.get("cwd")
    cwd_str: str | None = cwd if isinstance(cwd, str) and cwd else None

    env = _resolve_env()
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
    "src.infra.hooks.codex.shell_parser",
    "src.infra.hooks.codex.write_targets",
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
    iteration / length-prefix scheme byte-for-byte so the same
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
    selftest's deliberate trigger. We deny the tool call so the
    sentinel never executes; the provider also calls
    ``turn_interrupt`` from the client side as a belt-and-suspenders
    bound on LLM cost.

    The universal ``continue``/``stopReason`` fields are intentionally
    omitted: Codex's PreToolUse output parser rejects them as
    unsupported, and the PreToolUse runner fails open on invalid
    hook output — emitting them would let the sentinel tool call
    proceed even though the marker was written and the selftest
    appeared to pass.

    Output shape follows ``pre-tool-use.command.output``
    (HookUniversalOutputWire + ``decision`` + ``reason`` +
    ``hookSpecificOutput``).
    """
    return {
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

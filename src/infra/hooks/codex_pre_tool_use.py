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

This module is the thin adapter that wires the extracted policy modules
under :mod:`src.infra.hooks.codex` together: shell parsing, write-target
extraction, lock-policy decisions, dangerous-command classification, and
the selftest marker / live-probe flow all live in their own modules. The
adapter handles tool-name routing, env reading, hook output formatting,
and the CLI ``main()`` entry-point.

See ``plans/2026-05-07-codex-provider-plan.md`` (L778-L924) for the full
hook contract; the heuristic table at L842-L852 governs the bash branch.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from typing import Any

from .codex.dangerous_commands import (
    _detect_dangerous_pattern_recursive,
    _msg_dangerous,
)
from .codex.lock_policy import (
    _MSG_ENV_MISSING,
    _gate_write_targets,
)
from .codex.selftest_policy import (
    _emit_selftest_marker_if_requested,
    _handle_pre_tool_use_selftest,
    _handle_session_start_event,
    _selftest_mode_active,
    _selftest_target_event_matches,
)
from .codex.write_targets import (
    _apply_patch_paths,
    _extract_shell_write_paths,
)
from .dangerous_commands import BASH_TOOL_NAMES


__all__ = ["decide", "main"]


# Codex shell tool name candidates. ``bash`` is the public Codex tool name;
# ``local-shell`` is the internal SDK name retained for compatibility with
# the Phase D spike findings (plan L831 TBD).
SHELL_TOOL_NAMES = frozenset({"bash", "local-shell", "shell"} | set(BASH_TOOL_NAMES))

# Codex file-edit tool name candidates (plan L832 TBD); ``apply_patch`` is
# the documented Codex name and is matched case-insensitively elsewhere
# upstream, but the hook keeps to the canonical lower-case form.
FILE_EDIT_TOOL_NAMES = frozenset({"apply_patch"})


# Deny-message templates (plan L854-L860). The lock-related templates
# (``_msg_no_lock``, ``_msg_wrong_owner``, ``_MSG_ENV_MISSING``) live in
# :mod:`src.infra.hooks.codex.lock_policy`; the dangerous-pattern
# template (``_msg_dangerous``) lives in
# :mod:`src.infra.hooks.codex.dangerous_commands`; the disallowed-tool
# template stays here because it belongs to the corresponding branch in
# :func:`decide`.
def _msg_disallowed(name: str) -> str:
    return f"tool '{name}' is in MALA_DISALLOWED_TOOLS"


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
        reason = _gate_write_targets(
            write_targets,
            agent_id=agent_id,
            lock_dir=lock_dir,
            repo_namespace=repo_namespace,
            cwd=cwd_str,
        )
        return _deny(reason) if reason is not None else _allow()

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
        reason = _gate_write_targets(
            targets,
            agent_id=agent_id,
            lock_dir=lock_dir,
            repo_namespace=repo_namespace,
            cwd=cwd_str,
        )
        return _deny(reason) if reason is not None else _allow()

    # MCP tools and any other tool name: only the disallowed-tools check
    # applied (handled above). Allow otherwise.
    return _allow()


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

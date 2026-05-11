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
import shlex
import sys
from pathlib import Path
from typing import Any

from .codex.dangerous_commands import (
    _detect_dangerous_pattern_recursive,
    _msg_dangerous,
)
from .codex.lock_policy import (
    _MSG_ENV_MISSING,
    _gate_write_targets,
)
from .codex.write_targets import (
    _apply_patch_paths,
    _extract_shell_write_paths,
)
from .dangerous_commands import BASH_TOOL_NAMES


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
    "src.infra.hooks.codex.lock_policy",
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

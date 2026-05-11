"""Pure selftest-marker / live-probe policy for the Codex pre-tool-use hook.

Extracted from ``src.infra.hooks.codex_pre_tool_use`` (plan B.3 sub-commit
T_B12). The hook delegates to this module for the live-hook proof flow
the Codex provider's selftest relies on:

* the **marker contract** — a JSON marker (``{"mala_codex_hook":
  "loaded", "version": "<sha256>"}``) emitted to the path/dir the
  provider points at via the ``MALA_CODEX_HOOK_SELFTEST_*`` env vars
  so the provider can verify the on-PATH hook actually ran end-to-end.
* the **identity hash** — combined SHA-256 over the bundled hook's
  safety-critical module bytes, mirroring
  :data:`src.infra.clients.codex_provider._HOOK_IDENTITY_MODULES` so a
  stale install that swaps any tracked module is caught as
  ``VERSION_MISMATCH``.
* the **abort decisions** — ``SessionStart`` returns ``continue=false``
  so the turn is killed before any LLM call is made, and the live
  ``PreToolUse`` probe returns ``decision=block`` so the sentinel
  tool call never executes.

The end-to-end byte-identical contract for ``codex_pre_tool_use`` is
preserved by the golden corpus harness; this module's focused unit tests
pin every branch so a regression here is caught without rerunning the
whole hook.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path
from typing import Any


__all__ = [
    "SELFTEST_IDENTITY_MODULES",
    "SELFTEST_MARKER_DIR_ENV",
    "SELFTEST_MARKER_ENV",
    "SELFTEST_TARGET_EVENT_ENV",
    "_atomic_write_json",
    "_compute_selftest_identity_hash",
    "_emit_selftest_marker_if_requested",
    "_handle_pre_tool_use_selftest",
    "_handle_session_start_event",
    "_selftest_mode_active",
    "_selftest_target_event_matches",
]


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
    "src.infra.hooks.codex.dangerous_commands",
    "src.infra.hooks.codex.selftest_policy",
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

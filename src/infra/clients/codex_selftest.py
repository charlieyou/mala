"""Pure helpers backing the Codex selftest probes.

Extracted from :mod:`src.infra.clients.codex_provider` (Workstream B,
finding #3). These functions form the hash/identity surface the
provider uses to (a) compute the ``trusted_hash`` Codex expects for
the bundled ``mala-safety`` hooks and (b) emit the probe code the
on-PATH hook interpreter executes to fingerprint its mala module
sources. All helpers are pure and carry no class state.
"""

from __future__ import annotations

import hashlib
import json


def _normalized_hook_identity_value(event: str = "pre_tool_use") -> object:
    """Build the ``NormalizedHookIdentity`` payload Codex hashes for ``current_hash``.

    Source-of-truth: ``codex-rs/hooks/src/engine/discovery.rs::command_hook_hash``
    serializes the struct to TOML, then ``codex-rs/config/src/fingerprint.rs::version_for_toml``
    canonicalizes the JSON and returns ``"sha256:<hex>"``. The struct is
    ``{event_name, **MatcherGroup}`` with ``MatcherGroup = {matcher, hooks: [HookHandlerConfig]}``.
    Our hook is a single command handler with no matcher; Codex
    normalizes the timeout to 600 (``unwrap_or(600).max(1)``,
    ``discovery.rs:409``) and emits ``async`` / ``status_message`` /
    ``timeout`` per the ``HookHandlerConfig::Command`` serde shape
    (``hook_config.rs:123-135``: rename ``timeout_sec`` → ``timeout``,
    ``status_message`` → ``statusMessage``). ``Option::None`` fields are
    not present in the TOML round-trip used by the hash routine.
    """
    return {
        "event_name": event,
        "hooks": [
            {
                "type": "command",
                "command": "mala-codex-pre-tool-use",
                "timeout": 600,
                "async": False,
            }
        ],
    }


def _canonical_json(value: object) -> object:
    """Canonical-JSON normalization: dict keys sorted, lists left in order.

    Mirrors ``codex-rs/config/src/fingerprint.rs::canonical_json``.
    """
    if isinstance(value, dict):
        return {k: _canonical_json(value[k]) for k in sorted(value.keys())}
    if isinstance(value, list):
        return [_canonical_json(item) for item in value]
    return value


def _compute_normalized_hook_hash(event: str = "pre_tool_use") -> str:
    """Compute the ``sha256:<hex>`` ``current_hash`` Codex assigns the bundled hook.

    The Rust path serializes the struct to TOML, canonicalizes it, and
    JSON-encodes it before hashing. We approximate that by serializing
    the equivalent structure as canonical JSON (``json.dumps`` with
    ``sort_keys=True``, separators tightly packed). The TOML→JSON
    round-trip in the Rust pipeline happens to land on the same JSON
    shape Python's ``json.dumps`` produces for plain types (string,
    int, bool, list, dict), so the hex digest matches in practice for
    our minimal command-hook payload.
    """
    canonical = _canonical_json(_normalized_hook_identity_value(event))
    serialized = json.dumps(canonical, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(serialized).hexdigest()}"


def _build_module_hash_probe_code(modules: tuple[str, ...]) -> str:
    """Emit the probe Python code that hashes the on-PATH interpreter's
    view of every module in ``modules``.

    The probe code resolves each module via the same
    ``importlib.util.find_spec`` path the hook itself uses at startup,
    reads its source bytes, and folds them into a length-prefixed
    deterministic SHA-256. ``NOMODULE:<name>`` is printed and the probe
    exits non-zero when a module cannot be located. Output on success
    is a single hex digest line.
    """
    return (
        "import importlib.util, hashlib, sys\n"
        f"modules = {modules!r}\n"
        "h = hashlib.sha256()\n"
        "for name in modules:\n"
        "    spec = importlib.util.find_spec(name)\n"
        "    if spec is None or spec.origin is None:\n"
        "        print('NOMODULE:' + name); sys.exit(2)\n"
        "    with open(spec.origin, 'rb') as fp:\n"
        "        data = fp.read()\n"
        "    h.update(name.encode('utf-8'))\n"
        "    h.update(b'\\0')\n"
        "    h.update(len(data).to_bytes(8, 'big'))\n"
        "    h.update(data)\n"
        "print(h.hexdigest())\n"
    )

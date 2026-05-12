"""MCP launch-spec builders for the Codex coder.

Extracted from :mod:`src.infra.clients.codex_provider` (Workstream B,
finding #3). These helpers construct the bundled ``mala-locking``
launch spec, the merged ``.mcp.json`` payload written by
:class:`CodexPluginInstaller`, and the runtime-time
:class:`McpServerFactory` returning the bundled-plus-user merge. They
form the cross-coder contract that lets the same
:mod:`src.infra.tools.locking_mcp_stdio` launcher serve both Amp and
Codex (plan G3: bundled is mandatory, never replaced; non-conflicting
user keys pass through).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pathlib import Path

    from src.core.protocols.sdk import McpServerFactory


CODEX_BUNDLED_MCP_SERVER_NAME = "mala-locking"
"""Server key the bundled Codex plugin's ``.mcp.json`` references.

Mirrors the Amp factory's ``mala-locking`` key (``amp_provider.py:131``)
so the cross-coder lock-key derivation in
:mod:`src.infra.tools.locking_mcp_stdio` produces identical
``<hash>.lock`` fixtures regardless of which coder spawned the launcher.
The bundled plugin's ``.mcp.json`` declares this exact key (plan G1) and
the merge logic below pins it to the bundled stdio launch spec — user
``coder_options.codex.mcp_servers`` cannot override it (plan G3).
"""

CODEX_BUNDLED_MCP_LAUNCHER_COMMAND = "mala-codex-mcp-locking"
"""Console-script registered in ``[project.scripts]``.

Sibling of ``mala-amp-mcp-locking``; both back :func:`src.infra.tools.locking_mcp_stdio.main`.
The two entry-point names exist so the bundled Amp plugin (which already
references the Amp name in its ``.mcp.json``) keeps working unchanged
while the new Codex plugin gets its own stable name. A unified
``mala-mcp-locking`` rename is out of scope (plan ``L1335``).
"""

CODEX_BUNDLED_MCP_APPROVAL_MODE = "approve"
"""Codex tool-approval mode for the bundled lock server.

Codex evaluates MCP tool approvals separately from the thread-level
``approval_policy``. The bundled lock server is noninteractive infrastructure:
if ``lock_acquire`` waits for approval, unattended Mala workers receive a
synthetic rejection before the safety hook can observe any active lock.
Pinning the server to ``approve`` lets lock/unlock calls run while the
PreToolUse hook still gates filesystem writes against the lock backend.
"""


def _build_bundled_codex_mcp_spec(agent_id: str, repo_path: Path) -> dict[str, object]:
    """Codex-shaped stdio launch spec for the bundled ``mala-locking`` server.

    Mirrors :func:`src.infra.clients.amp_provider._create_amp_mcp_server_factory`'s
    payload shape — ``command`` + ``args`` + ``env`` — so the same
    :mod:`src.infra.tools.locking_mcp_stdio` launcher serves both coders
    and the produced ``<hash>.lock`` fixtures interoperate. ``MALA_LOCK_DIR``
    is forwarded from the orchestrator's environment when set; when unset
    the launcher falls through to :func:`src.infra.tools.env.get_lock_dir`.
    """
    env: dict[str, str] = {
        "MALA_AGENT_ID": agent_id,
        "MALA_REPO_NAMESPACE": str(repo_path),
    }
    lock_dir = os.environ.get("MALA_LOCK_DIR")
    if lock_dir:
        env["MALA_LOCK_DIR"] = lock_dir

    return {
        "command": CODEX_BUNDLED_MCP_LAUNCHER_COMMAND,
        "args": [
            "--agent-id",
            agent_id,
            "--repo-namespace",
            str(repo_path),
        ],
        "env": env,
    }


def _build_merged_codex_plugin_mcp_json(
    user_mcp_servers: tuple[tuple[str, object], ...] = (),
) -> bytes:
    """Build the ``.mcp.json`` payload that merges user MCP servers with
    the bundled ``mala-locking`` entry (Phase G3 / AC-3).

    User-supplied entries from ``coder_options.codex.mcp_servers`` are
    placed first; the bundled ``mala-locking`` key is written **last**
    so any clashing user override is silently replaced by the bundled
    launcher (G3: bundled is mandatory, never replaced). Without this
    merge, the static plugin file would only carry ``mala-locking`` and
    non-conflicting user MCP servers configured via ``mala.yaml`` would
    never reach Codex at runtime.

    Returned bytes are routed through :class:`CodexPluginInstaller` via
    its ``mcp_json_override`` constructor argument so the installer
    remains the sole writer of the plugin tree. Routing through a
    single writer is what closes the cross-process race a separate
    post-install rewrite would expose: without it, Process B's
    installer would silently revert Process A's render to the
    bundled-only file, dropping ``coder_options.codex.mcp_servers``
    entries from Codex's effective config.

    The bundled spec on disk uses the **static** shape (empty ``args``
    and no literal per-session ``env`` values). Codex launches stdio MCP
    servers with a sanitized environment, so the static spec must list
    the required ``MALA_*`` names in ``env_vars`` to forward them from
    the Codex app-server process env (passed via
    :class:`AppServerConfig.env`). Embedding literal per-session values
    in the plugin file would race concurrent agents under
    ``--max-agents > 1``.
    """
    import json as _json

    merged: dict[str, object] = dict(user_mcp_servers)
    merged[CODEX_BUNDLED_MCP_SERVER_NAME] = {
        "command": CODEX_BUNDLED_MCP_LAUNCHER_COMMAND,
        "args": [],
        "env": {},
        "env_vars": [
            "MALA_AGENT_ID",
            "MALA_LOCK_DIR",
            "MALA_REPO_NAMESPACE",
        ],
    }
    return (_json.dumps({"mcpServers": merged}, indent=2) + "\n").encode("utf-8")


def _create_codex_mcp_server_factory(
    user_mcp_servers: tuple[tuple[str, object], ...] = (),
) -> McpServerFactory:
    """Codex-shaped MCP server factory returning the bundled-plus-user merge.

    The bundled ``mala-locking`` launch spec is **mandatory** and **never
    overridden** by user config (plan G3): user-supplied entries are
    overlaid first, then the bundled key is written last so any clashing
    user key is replaced. Non-clashing user keys pass through unchanged.

    Codex itself does not consume this map inline — :meth:`CodexClient.query`
    intentionally does NOT pass ``mcp_servers`` to ``AsyncCodex.thread_start``
    (per the SDK shape comment at ``codex_client.py:266-276``); MCP wiring
    flows through the bundled plugin's ``.mcp.json``, whose effective
    bytes are produced by :func:`_build_merged_codex_plugin_mcp_json`
    and handed to :class:`CodexPluginInstaller` via its
    ``mcp_json_override`` argument so the installer is the sole writer
    of the merged file. The runtime's :attr:`CodexRuntime.mcp_servers`
    field carries the merged map for forward-compat / inspection so
    callers (telemetry, tests, future SDK versions that accept inline
    specs) see the resolved configuration the bundled plugin ultimately
    presents to Codex.
    """

    def factory(
        agent_id: str,
        repo_path: Path,
        emit_lock_event: object,
    ) -> dict[str, object]:
        del emit_lock_event
        merged: dict[str, object] = dict(user_mcp_servers)
        merged[CODEX_BUNDLED_MCP_SERVER_NAME] = _build_bundled_codex_mcp_spec(
            agent_id, repo_path
        )
        return merged

    return cast("McpServerFactory", factory)

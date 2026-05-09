"""Stdio MCP launcher for the locking server (real-Amp integration).

The Claude path uses :func:`src.infra.tools.locking_mcp.create_locking_mcp_server`
to construct an in-process MCP ``Server`` exposed through the Claude Agent
SDK. Amp consumes its MCP servers through ``--mcp-config``, which expects a
**stdio launch spec** (``command`` + ``args`` + ``env``). This module is
that stdio entry point.

Wire-up:

* ``[project.scripts]`` registers ``mala-amp-mcp-locking`` →
  :func:`main`. After ``uv sync`` (or any ``pip install -e .``) the
  console script lands on ``PATH``.
* :meth:`src.infra.clients.amp_provider.AmpAgentProvider.mcp_server_factory`
  emits ``{"command": "mala-amp-mcp-locking", "args": [...], "env": {...}}``
  for Amp to spawn.

Tools exposed (parity with ``locking_mcp.py``'s in-process server):

* ``lock_acquire(filepaths, timeout_seconds)``
* ``lock_release(filepaths | all)``

Bodies are JSON-encoded text content blocks (same shape as the in-process
server) so the cross-language plugin contract (``mala-safety.ts``) can read
the produced ``<hash>.lock`` files identically. When ``MALA_LOCK_EVENT_LOG``
is set, the launcher also appends lock events to that JSONL side channel so
the parent Amp client can feed the existing orchestrator deadlock monitor.

Configuration is read from CLI args first, then env-var fallback:

* ``--agent-id <id>`` / ``MALA_AGENT_ID``
* ``--repo-namespace <path>`` / ``MALA_REPO_NAMESPACE``
* ``MALA_LOCK_DIR`` (no CLI arg; the locking module reads it directly)

The launcher fails closed (non-zero exit, error to stderr) if ``agent-id``
is missing — without an owner, ``try_lock`` cannot record a holder and the
plugin's lock-ownership check would always reject.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from mcp.server.lowlevel.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from src.infra.tools.locking import (
    canonicalize_path as _canonicalize_path,
    cleanup_agent_locks as _cleanup_agent_locks,
    get_lock_holder as _get_lock_holder,
    release_lock as _release_lock,
    try_lock as _try_lock,
    wait_for_lock_async as _wait_for_lock_async,
)

logger = logging.getLogger(__name__)

_LOCK_EVENT_LOG_ENV = "MALA_LOCK_EVENT_LOG"


_LOCK_ACQUIRE_DESCRIPTION = (
    "Try to acquire locks on multiple files. If some are blocked, "
    "waits until ANY becomes available (or timeout). Returns per-file "
    "results. Use timeout_seconds=0 for non-blocking try."
)

_LOCK_RELEASE_DESCRIPTION = (
    "Release locks on files. Use filepaths to release specific files, "
    "or all=true to release all locks held by this agent. Idempotent "
    "(succeeds even if locks not held)."
)


def _lock_acquire_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "filepaths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of file paths to lock",
                "minItems": 1,
            },
            "timeout_seconds": {
                "type": "number",
                "description": "Max seconds to wait (default: 30, use 0 for non-blocking)",
            },
        },
        "required": ["filepaths"],
    }


def _lock_release_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "filepaths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of file paths to unlock (mutually exclusive with 'all')",
            },
            "all": {
                "type": "boolean",
                "description": "Release ALL locks held by this agent (mutually exclusive with 'filepaths')",
            },
        },
    }


def _strip_canonical(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{k: v for k, v in r.items() if k != "canonical"} for r in results]


def _emit_lock_event(event_type: str, agent_id: str, lock_path: str) -> None:
    """Append a lock event for the parent Amp client to tail, if configured."""
    event_log = os.environ.get(_LOCK_EVENT_LOG_ENV)
    if not event_log:
        return
    payload = {
        "event_type": event_type,
        "agent_id": agent_id,
        "lock_path": lock_path,
        "timestamp": time.time(),
    }
    try:
        path = Path(event_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, separators=(",", ":")))
            fh.write("\n")
            fh.flush()
    except OSError:
        logger.exception("Failed to append Amp lock event: %s", payload)


async def _handle_lock_acquire(
    args: dict[str, Any],
    agent_id: str,
    repo_namespace: str | None,
) -> str:
    """Run the same algorithm as locking_mcp.py's in-process handler.

    Returned string is the JSON body the caller wraps in a TextContent.
    """
    filepaths = args.get("filepaths") or []
    timeout_seconds = args.get("timeout_seconds", 30.0)

    if not filepaths:
        return json.dumps({"error": "filepaths must be a non-empty array"})

    seen_canonical: dict[str, str] = {}
    for fp in filepaths:
        canon = _canonicalize_path(fp, repo_namespace)
        if canon not in seen_canonical:
            seen_canonical[canon] = fp
    sorted_canonical = sorted(seen_canonical.keys())

    results: list[dict[str, Any]] = []
    blocked: list[str] = []
    for canon in sorted_canonical:
        original = seen_canonical[canon]
        holder_before = _get_lock_holder(canon, repo_namespace)
        acquired = _try_lock(canon, agent_id, repo_namespace)
        holder = None if acquired else _get_lock_holder(canon, repo_namespace)
        if acquired and holder_before != agent_id:
            _emit_lock_event("acquired", agent_id, canon)
        results.append(
            {
                "filepath": original,
                "canonical": canon,
                "acquired": acquired,
                "holder": holder,
            }
        )
        if not acquired:
            blocked.append(canon)

    if not blocked or timeout_seconds == 0:
        return json.dumps(
            {
                "results": _strip_canonical(results),
                "all_acquired": not blocked,
            }
        )

    for canon in blocked:
        _emit_lock_event("waiting", agent_id, canon)

    # Wait until ANY blocked file becomes available, mirroring the
    # in-process server's FIRST_COMPLETED behavior (locking_mcp.py:309-356).
    wait_tasks: dict[asyncio.Task[bool], str] = {}
    for canon in blocked:
        task = asyncio.create_task(
            _wait_for_lock_async(
                canon,
                agent_id,
                repo_namespace,
                timeout_seconds=timeout_seconds,
                poll_interval_ms=100,
            )
        )
        wait_tasks[task] = canon

    try:
        done, pending = await asyncio.wait(
            wait_tasks.keys(),
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        for task in pending:
            try:
                await task
            except asyncio.CancelledError:
                pass

        for task in done:
            canon = wait_tasks[task]
            try:
                acquired = task.result()
            except Exception:
                acquired = False
            if acquired:
                _emit_lock_event("acquired", agent_id, canon)
            for r in results:
                if r["canonical"] == canon:
                    r["acquired"] = acquired
                    r["holder"] = (
                        None if acquired else _get_lock_holder(canon, repo_namespace)
                    )
                    break
    except BaseException:
        for task in wait_tasks:
            if not task.done():
                task.cancel()
        for task in wait_tasks:
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        raise

    return json.dumps(
        {
            "results": _strip_canonical(results),
            "all_acquired": all(r["acquired"] for r in results),
        }
    )


async def _handle_lock_release(
    args: dict[str, Any],
    agent_id: str,
    repo_namespace: str | None,
) -> str:
    filepaths = args.get("filepaths")
    release_all = args.get("all", False)

    if filepaths is not None and release_all:
        return json.dumps({"error": "Cannot specify both 'filepaths' and 'all'"})

    if filepaths is None and not release_all:
        return json.dumps({"error": "Must specify either 'filepaths' or 'all=true'"})

    if release_all:
        count, released_paths = _cleanup_agent_locks(agent_id)
        for path in released_paths:
            _emit_lock_event("released", agent_id, path)
        return json.dumps({"released": released_paths, "count": count})

    if not filepaths:
        return json.dumps({"error": "filepaths must be a non-empty array"})

    seen: set[str] = set()
    released: list[str] = []
    for fp in filepaths:
        canon = _canonicalize_path(fp, repo_namespace)
        if canon in seen:
            continue
        seen.add(canon)
        _release_lock(canon, agent_id, repo_namespace)
        _emit_lock_event("released", agent_id, canon)
        released.append(fp)

    return json.dumps({"released": released, "count": len(released)})


def _build_server(agent_id: str, repo_namespace: str | None) -> Server:
    server: Server = Server(name="mala-locking", version="1.0.0")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="lock_acquire",
                description=_LOCK_ACQUIRE_DESCRIPTION,
                inputSchema=_lock_acquire_schema(),
            ),
            Tool(
                name="lock_release",
                description=_LOCK_RELEASE_DESCRIPTION,
                inputSchema=_lock_release_schema(),
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name == "lock_acquire":
            body = await _handle_lock_acquire(arguments, agent_id, repo_namespace)
        elif name == "lock_release":
            body = await _handle_lock_release(arguments, agent_id, repo_namespace)
        else:
            raise ValueError(f"Unknown tool: {name}")
        return [TextContent(type="text", text=body)]

    return server


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="mala-amp-mcp-locking",
        description=(
            "Stdio MCP launcher for mala's locking tools. Spawned by Amp via "
            "--mcp-config; not intended for direct interactive use."
        ),
    )
    parser.add_argument(
        "--agent-id",
        default=os.environ.get("MALA_AGENT_ID"),
        help="Agent ID for lock ownership (or MALA_AGENT_ID env var).",
    )
    parser.add_argument(
        "--repo-namespace",
        default=os.environ.get("MALA_REPO_NAMESPACE"),
        help="Repo namespace for cross-repo lock disambiguation (or "
        "MALA_REPO_NAMESPACE env var).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point for ``mala-amp-mcp-locking``."""
    ns = _parse_args(list(sys.argv[1:]) if argv is None else argv)

    if not ns.agent_id:
        print(
            "mala-amp-mcp-locking: --agent-id (or MALA_AGENT_ID env var) is required.",
            file=sys.stderr,
        )
        return 2

    repo_namespace: str | None = ns.repo_namespace if ns.repo_namespace else None

    server = _build_server(ns.agent_id, repo_namespace)

    async def _run() -> None:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())

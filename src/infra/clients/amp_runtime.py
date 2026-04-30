"""Amp coder runtime + builder.

Produces a complete spawn description for ``amp --execute --stream-json``:
argv, env (a copy of ``os.environ`` plus overlays), MCP config JSON, working
directory, log path, and optional resume thread id.

Env composition mirrors :class:`src.infra.agent_runtime.AgentRuntimeBuilder.with_env`
(``{**os.environ, ...overlays}``) so the spawned ``amp`` subprocess inherits
``PATH``, ``HOME``, ``TMPDIR``, locale vars, and ``AMP_API_KEY`` from the parent
process. The overlays carry the ``MALA_*`` lock-ownership vars (``MALA_AGENT_ID``,
``MALA_LOCK_DIR``, ``MALA_REPO_NAMESPACE``) consumed by ``plugins/amp/mala-safety.ts``;
the ``MALA_*`` prefix disambiguates from the Claude-path's unprefixed names so
both paths can co-exist in the same shell.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from src.infra.tools.env import SCRIPTS_DIR, USER_CONFIG_DIR, get_lock_dir

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from src.core.protocols.sdk import McpServerFactory


AmpMode = Literal["smart", "rush", "deep"]


AMP_SESSIONS_DIR: Path = USER_CONFIG_DIR / "amp-sessions"
"""Directory where mala tees Amp's stream-json output, keyed by thread id."""

_PENDING_LOG_NAME = ".pending.jsonl"
"""Pre-thread-id placeholder; ``AmpClient`` renames to ``{thread_id}.jsonl``."""


@dataclass(frozen=True)
class AmpRuntime:
    """Complete spawn description for one ``amp --execute --stream-json`` invocation.

    Returned by :meth:`AmpRuntimeBuilder.build`. Consumed only by the matching
    ``AmpClient`` (T008); the pipeline treats it as opaque.
    """

    cwd: Path
    env: Mapping[str, str]
    argv: tuple[str, ...]
    mcp_config: dict[str, object]
    mode: AmpMode
    log_path: Path
    resume_thread_id: str | None = None


class AmpRuntimeBuilder:
    """Build :class:`AmpRuntime` for one Amp session.

    Conforms structurally to :class:`src.core.protocols.agent_provider.CoderRuntimeBuilder`
    (exposes ``build()``). Use :meth:`with_resume` to continue an existing
    Amp thread.
    """

    def __init__(
        self,
        repo_path: Path,
        agent_id: str,
        mcp_server_factory: McpServerFactory,
        mode: AmpMode = "smart",
    ) -> None:
        """Initialize the builder.

        Args:
            repo_path: Repository root; becomes ``cwd`` and ``MALA_REPO_NAMESPACE``.
            agent_id: Per-issue agent identifier; becomes ``MALA_AGENT_ID``.
            mcp_server_factory: Produces the MCP server config dict for
                ``--mcp-config``. The factory is invoked with
                ``(agent_id, repo_path, None)``; the third arg is the
                deadlock-monitor callback hook used by the Claude path
                (Amp does not wire deadlock events in MVP).
            mode: Amp execution mode. Sourced from
                ``MalaConfig.coder_options.amp.mode``; the orchestration layer
                resolves the precedence (CLI > env > yaml > default) before
                constructing the builder.
        """
        self._repo_path = repo_path
        self._agent_id = agent_id
        self._mcp_server_factory = mcp_server_factory
        self._mode: AmpMode = mode
        self._resume_thread_id: str | None = None

    def with_resume(self, thread_id: str) -> AmpRuntimeBuilder:
        """Configure the next ``build()`` to continue an existing Amp thread.

        The exact resume CLI shape is the impl-spike default (``--thread-id <id>``
        on ``amp --execute``); see ``plans/2026-04-29-amp-provider-plan.md#L679``.

        Returns:
            ``self`` for fluent chaining.
        """
        self._resume_thread_id = thread_id
        return self

    def build(self) -> AmpRuntime:
        """Materialize the runtime.

        Composes env from ``os.environ`` plus the overlays in the plan
        (``PATH``, ``PLUGINS``, ``MALA_AGENT_ID``, ``MALA_LOCK_DIR``,
        ``MALA_REPO_NAMESPACE``, ``MCP_TIMEOUT``). ``AMP_API_KEY`` is inherited
        from ``os.environ`` (no overlay needed).
        """
        mcp_config = self._mcp_server_factory(self._agent_id, self._repo_path, None)

        env: dict[str, str] = {
            **os.environ,
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
            "PLUGINS": "all",
            "MALA_AGENT_ID": self._agent_id,
            "MALA_LOCK_DIR": str(get_lock_dir()),
            "MALA_REPO_NAMESPACE": str(self._repo_path),
            "MCP_TIMEOUT": "300000",
        }

        argv: list[str] = [
            "amp",
            "--execute",
            "--stream-json",
            "--dangerously-allow-all",
            "--mcp-config",
            json.dumps(mcp_config),
            "--mode",
            self._mode,
        ]
        if self._resume_thread_id is not None:
            argv.extend(["--thread-id", self._resume_thread_id])

        if self._resume_thread_id is not None:
            log_path = AMP_SESSIONS_DIR / f"{self._resume_thread_id}.jsonl"
        else:
            log_path = AMP_SESSIONS_DIR / _PENDING_LOG_NAME

        return AmpRuntime(
            cwd=self._repo_path,
            env=env,
            argv=tuple(argv),
            mcp_config=mcp_config,
            mode=self._mode,
            log_path=log_path,
            resume_thread_id=self._resume_thread_id,
        )

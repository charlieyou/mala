"""Amp coder runtime + builder.

Produces a complete spawn description for ``amp --execute --stream-json``:
argv, env (a copy of ``os.environ`` plus overlays), MCP config JSON, working
directory, log path, and optional resume thread id.

Env composition mirrors :class:`src.infra.agent_runtime.AgentRuntimeBuilder.with_env`
(``{**os.environ, ...overlays}``) so the spawned ``amp`` subprocess inherits
``PATH``, ``HOME``, ``TMPDIR``, locale vars, and other parent environment keys.
The overlays carry the ``MALA_*`` lock-ownership vars (``MALA_AGENT_ID``,
``MALA_LOCK_DIR``, ``MALA_REPO_NAMESPACE``) consumed by
``plugins/amp/mala-safety.ts``; the ``MALA_*`` prefix disambiguates from the
Claude-path's unprefixed names so both paths can co-exist in the same shell.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from src.infra.tools.env import SCRIPTS_DIR, USER_CONFIG_DIR, get_lock_dir

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Mapping, Set as AbstractSet
    from pathlib import Path

    from src.core.protocols.sdk import McpServerFactory
    from src.infra.clients.amp_client import AmpClientOptions
    from src.infra.hooks.lint_cache import LintCache


AmpMode = Literal["smart", "rush", "deep"]


AMP_SESSIONS_DIR: Path = USER_CONFIG_DIR / "amp-sessions"
"""Directory where mala tees Amp's stream-json output, keyed by thread id."""

_PENDING_LOG_PREFIX = ".pending-"
_PENDING_LOG_SUFFIX = ".jsonl"
"""Pre-thread-id placeholder shape: ``.pending-<uuid4>.jsonl``.

The UUID is required because mala spawns concurrent issue agents into the
same shared ``amp-sessions/`` directory; every fresh session must own a
distinct pending file before ``AmpClient`` observes ``system(init)`` and
renames it to ``{thread_id}.jsonl``. A fixed name would let two simultaneous
fresh sessions interleave stream-json bytes, corrupting both."""


@dataclass(frozen=True)
class AmpRuntime:
    """Complete spawn description for one ``amp --execute --stream-json`` invocation.

    Returned by :meth:`AmpRuntimeBuilder.build`. Consumed by the matching
    ``AmpClient`` (T008) and read by :class:`AgentSessionRunner` for the
    ``options`` / ``lint_cache`` fields the pipeline expects on every
    coder-shaped runtime.
    """

    cwd: Path
    env: Mapping[str, str]
    argv: tuple[str, ...]
    mcp_config: dict[str, object]
    mode: AmpMode
    log_path: Path
    options: AmpClientOptions
    """Pre-built :class:`AmpClientOptions` consumed by
    :meth:`AmpAgentProvider.client_factory.create`. The pipeline reads
    ``runtime.options`` after ``builder.build()`` exactly the same way it
    reads ``ClaudeRuntime.options``, so the ``AgentSessionRunner`` path is
    coder-agnostic."""
    lint_cache: LintCache
    """In-memory :class:`LintCache` used by the session runner to dedupe
    lint commands across attempts. Amp does not consume the cache itself —
    the synthetic message dataclasses do not interact with it — but the
    runtime carries one for protocol parity with the Claude path."""
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
        mode: AmpMode = "deep",
        effort: str | None = None,
    ) -> None:
        """Initialize the builder.

        Args:
            repo_path: Repository root; becomes ``cwd`` and ``MALA_REPO_NAMESPACE``.
            agent_id: Per-issue agent identifier; becomes ``MALA_AGENT_ID``.
            mcp_server_factory: Produces the MCP server map for
                ``--mcp-config``. The factory is invoked with
                ``(agent_id, repo_path, None)``; the third arg is the
                deadlock-monitor callback hook used by the Claude path
                (Amp does not wire deadlock events in MVP).
            mode: Amp execution mode. Sourced from
                ``MalaConfig.coder_options.amp.mode``; the orchestration layer
                resolves the precedence (CLI > env > yaml > default) before
                constructing the builder.
            effort: Optional Mala-level reasoning effort. Resolved from
                ``MalaConfig.effort`` (CLI > env > yaml > default=None).
                Appended to the Amp argv as ``--effort <value>`` only when
                ``mode == "deep"``; for non-deep modes the value is ignored
                with an info-level log message so the silent drop is
                observable.
        """
        self._repo_path = repo_path
        self._agent_id = agent_id
        self._mcp_server_factory = mcp_server_factory
        self._mode: AmpMode = mode
        self._effort: str | None = effort
        self._resume_thread_id: str | None = None
        # Fluent-API state collected by the with_* helpers below. The Amp
        # pipeline does not act on most of these (Amp enforces
        # dangerous-cmd / lock-ownership via the ``mala-safety.ts`` plugin,
        # not via SDK-level hooks or the disallowed-tools env contract),
        # but they are recorded faithfully so the cross-coder pipeline can
        # build an Amp runtime via the same fluent chain it uses for the
        # Claude runtime — and so future Amp features (cross-thread
        # lint caching, custom MCP overlays) have an explicit hook.
        self._env_extra: dict[str, str] = {}
        self._mcp_servers_override: dict[str, object] | None = None
        self._lint_tools: AbstractSet[str] | None = None

    def with_hooks(
        self,
        deadlock_monitor: object | None = None,
        *,
        include_stop_hook: bool = True,
        include_mala_disallowed_tools_hook: bool = True,
        include_lock_enforcement_hook: bool = True,
    ) -> AmpRuntimeBuilder:
        """No-op fluent-API parity with :class:`AgentRuntimeBuilder.with_hooks`.

        Amp does not honor SDK hooks; the safety enforcement lives in the
        bundled ``mala-safety.ts`` plugin loaded under ``PLUGINS=all``
        (plan ``L412-L450``). The arguments are accepted for parity with
        the Claude builder so the shared pipeline call site can chain
        identically against either runtime; they are otherwise ignored.

        Returns:
            ``self`` for fluent chaining.
        """
        del (
            deadlock_monitor,
            include_stop_hook,
            include_mala_disallowed_tools_hook,
            include_lock_enforcement_hook,
        )
        return self

    def with_env(self, *, extra: Mapping[str, str] | None = None) -> AmpRuntimeBuilder:
        """Overlay extra env vars on top of the env composed in :meth:`build`.

        Mirrors :meth:`AgentRuntimeBuilder.with_env`'s ``extra=`` shape
        (``src/infra/agent_runtime.py:180-202``). The extras are layered
        on **after** the plan's mandatory overlays (PATH, PLUGINS,
        MALA_*), so a caller cannot accidentally clobber the lock-ownership
        env vars consumed by the safety plugin.
        """
        if extra:
            self._env_extra.update(extra)
        return self

    def with_mcp(
        self, *, servers: Mapping[str, object] | None = None
    ) -> AmpRuntimeBuilder:
        """Configure MCP servers; default uses the injected factory.

        ``servers`` is provided for parity with the Claude builder's
        explicit override (used by :class:`FixerService` to short-circuit
        MCP wiring when no factory is available). When omitted, the
        runtime's ``--mcp-config`` is built from ``mcp_server_factory``
        in :meth:`build` exactly as before.
        """
        if servers is not None:
            self._mcp_servers_override = dict(servers)
        return self

    def with_disallowed_tools(
        self, tools: AbstractSet[str] | None = None
    ) -> AmpRuntimeBuilder:
        """No-op fluent-API parity for ``disallowed_tools``.

        ``MALA_DISALLOWED_TOOLS`` has no effect on Amp in MVP (plan
        ``L690``); the warn-once log is emitted by
        ``AmpAgentProvider.install_prerequisites``. The argument is
        accepted so the pipeline's chained call works against either
        runtime.
        """
        del tools
        return self

    def with_lint_tools(
        self, tools: AbstractSet[str] | None = None
    ) -> AmpRuntimeBuilder:
        """Record the lint-tool set used to construct :attr:`AmpRuntime.lint_cache`.

        The Amp message-stream does not currently interact with the
        cache, but the pipeline's :class:`SessionConfig` reads
        ``runtime.lint_cache`` unconditionally — providing one keeps
        the per-issue lifecycle path coder-agnostic.
        """
        self._lint_tools = tools
        return self

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
        ``MALA_REPO_NAMESPACE``, ``MCP_TIMEOUT``). Other parent env keys are
        inherited without special handling.

        ``McpServerFactory`` returns the same server *map* Amp expects for
        ``--mcp-config`` (``{"name": <spec>}``); Amp rejects the settings-file
        shape with a top-level ``{"mcpServers": ...}`` envelope on the CLI.

        The injected factory **must produce stdio launch specs** (JSON-only
        ``command``/``args``/``env`` dicts). The Claude-path factory packs
        in-process Claude SDK server instances and is incompatible with Amp's
        CLI; we serialize the config eagerly here so a misrouted Claude factory
        surfaces an actionable error at the builder boundary instead of an
        opaque ``TypeError`` deep inside ``subprocess.Popen``.
        """
        if self._mcp_servers_override is not None:
            mcp_config: dict[str, object] = dict(self._mcp_servers_override)
        else:
            mcp_config = dict(
                self._mcp_server_factory(self._agent_id, self._repo_path, None)
            )
        try:
            mcp_config_json = json.dumps(mcp_config)
        except TypeError as exc:
            msg = (
                "AmpRuntimeBuilder requires an Amp-shaped MCP server factory "
                "that returns JSON-serializable stdio launch specs (command, "
                "args, env). The Claude path's create_mcp_server_factory() "
                "returns in-process Claude SDK server objects and cannot be "
                "reused for Amp. Wire an Amp-specific factory via "
                f"AmpAgentProvider. Underlying error: {exc}"
            )
            raise TypeError(msg) from exc

        env: dict[str, str] = {
            **os.environ,
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
            "PLUGINS": "all",
            "MALA_AGENT_ID": self._agent_id,
            "MALA_LOCK_DIR": str(get_lock_dir()),
            "MALA_REPO_NAMESPACE": str(self._repo_path),
            "MCP_TIMEOUT": "300000",
            **self._env_extra,
        }

        argv: list[str] = [
            "amp",
            "--execute",
            "--stream-json",
            "--dangerously-allow-all",
            "--mcp-config",
            mcp_config_json,
            "--mode",
            self._mode,
        ]
        # Reasoning effort is documented as a deep-mode concept in the Amp
        # CLI; we forward ``--effort`` only when the active mode is ``deep``
        # and emit an info-level log so the silent drop on other modes is
        # observable. The Amp CLI itself accepts the same string values mala
        # validated upstream (low | medium | high | xhigh | max).
        if self._effort is not None:
            if self._mode == "deep":
                argv.extend(["--effort", self._effort])
            else:
                logger.info(
                    "amp effort '%s' ignored — mode is '%s' (effort is "
                    "applied only in deep mode)",
                    self._effort,
                    self._mode,
                )
        if self._resume_thread_id is not None:
            argv.extend(["--thread-id", self._resume_thread_id])

        if self._resume_thread_id is not None:
            log_path = AMP_SESSIONS_DIR / f"{self._resume_thread_id}.jsonl"
        else:
            pending_name = (
                f"{_PENDING_LOG_PREFIX}{uuid.uuid4().hex}{_PENDING_LOG_SUFFIX}"
            )
            log_path = AMP_SESSIONS_DIR / pending_name

        # Construct AmpClientOptions eagerly so AgentSessionRunner sees a
        # ``runtime.options`` value with the same role ClaudeRuntime.options
        # plays — the pipeline forwards it verbatim to
        # ``client_factory.create(options)``.
        from src.infra.clients.amp_client import AmpClientOptions

        options = AmpClientOptions(
            cwd=self._repo_path,
            env=env,
            argv=tuple(argv),
            log_path=log_path,
            thread_id=self._resume_thread_id,
        )

        # LintCache parity with the Claude path. Even though the synthetic
        # Amp message dataclasses do not currently interact with the
        # cache, ``SessionConfig`` reads ``runtime.lint_cache`` and
        # forwards it into the lifecycle layer, so it must be a real
        # instance, not None.
        from src.infra.hooks.lint_cache import LintCache

        lint_cache = LintCache(
            repo_path=self._repo_path,
            lint_tools=self._lint_tools,
        )

        return AmpRuntime(
            cwd=self._repo_path,
            env=env,
            argv=tuple(argv),
            mcp_config=mcp_config,
            mode=self._mode,
            log_path=log_path,
            options=options,
            lint_cache=lint_cache,
            resume_thread_id=self._resume_thread_id,
        )

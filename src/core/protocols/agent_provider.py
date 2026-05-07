"""AgentProvider and CoderRuntimeBuilder protocols.

This module defines the abstraction line between pipeline code and concrete
coder backends. The pipeline consumes an ``AgentProvider`` (which bundles a
client factory, a per-session runtime builder, and a log provider) and is
agnostic to whether the underlying coder is Claude or Amp.

Concrete provider implementations (``ClaudeAgentProvider``,
``AmpAgentProvider``) live in ``src/infra``; this module contains protocol
definitions only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pathlib import Path

    from src.core.protocols.log import LogProvider
    from src.core.protocols.sdk import McpServerFactory, SDKClientFactoryProtocol


@runtime_checkable
class CoderRuntimeBuilder(Protocol):
    """Build a coder-shaped runtime that the matching client_factory consumes.

    The object returned by :meth:`build` is opaque to the pipeline; only the
    matching ``client_factory`` knows how to consume it. Concrete return types
    include ``ClaudeRuntime`` (for the Claude provider) and ``AmpRuntime``
    (for the Amp provider).
    """

    def build(self) -> object:
        """Return an opaque coder-shaped runtime; consumed only by the matching client_factory."""
        ...


@runtime_checkable
class AgentProvider(Protocol):
    """Encapsulates a coder backend (Claude or Amp).

    Bundles three pluggable concerns:
      - client_factory:  produces SDKClientProtocol-conforming clients
      - runtime_builder: produces a coder-shaped runtime per session
      - log_provider:    produces a LogProvider for evidence parsing

    ``client_factory`` and ``log_provider`` are attributes (values) because the
    same instance is reused across the run; ``runtime_builder`` is a method
    because the runtime is constructed per session and depends on the repo
    path and per-issue ``agent_id``.
    """

    name: Literal["claude", "amp"]
    client_factory: SDKClientFactoryProtocol
    log_provider: LogProvider

    def runtime_builder(
        self,
        repo_path: Path,
        agent_id: str,
        *,
        mcp_server_factory: McpServerFactory,
    ) -> CoderRuntimeBuilder:
        """Construct a per-session :class:`CoderRuntimeBuilder`.

        Args:
            repo_path: Path to the repository the agent will run in.
            agent_id: Per-issue agent identifier (used for lock-ownership
                plumbing and MCP server configuration).
            mcp_server_factory: Factory producing the MCP server config map.

        Returns:
            A :class:`CoderRuntimeBuilder` whose ``build()`` yields the
            coder-shaped runtime object consumed by the matching
            ``client_factory``.
        """
        ...

    def mcp_server_factory(self) -> McpServerFactory:
        """Return the coder-shaped MCP server factory.

        Each provider knows whether it consumes in-process Server objects
        (Claude) or stdio launch specs (Amp / Codex), so dispatch is owned
        here rather than at the orchestrator. The orchestrator and factory
        call this method instead of branching on ``self.name``.
        """
        ...

    def install_prerequisites(
        self,
        repo_path: Path,
        *,
        mcp_server_factory: McpServerFactory,
    ) -> None:
        """Idempotent setup invoked once per run before the first session.

        Takes the same context the runtime_builder takes (minus agent_id,
        which is per-session) so the Amp self-test exercises the *same*
        runtime path real sessions use - same --mcp-config, same MALA_*
        env, same PLUGINS=all. The orchestrator constructs a synthetic
        agent_id ("amp-selftest") for the self-test runtime; it never
        reaches a real beads issue.

        Claude impl: no-op (signature still required by the protocol).
        Amp impl:
          1. mkdir -p ~/.config/amp/plugins/ (idempotent; first-time users
             may not have the directory yet).
          2. Copy plugins/amp/mala-safety.ts there using
             write-temp-then-rename for concurrent-run safety.
          3. Verify installed file's content hash matches the bundled copy.
          4. Run a runtime plugin-load self-test (fail-closed,
             safety-critical): build a self-test AmpRuntime using
             runtime_builder(repo_path, "amp-selftest",
             mcp_server_factory=...). Spawn ``amp --execute --stream-json
             --dangerously-allow-all`` with that runtime's env (PLUGINS=all
             + MALA_* lock vars + os.environ passthrough) and the runtime's
             --mcp-config. Pass when the session.start
             sentinel marker ``{"mala_plugin":"loaded","version":"<hash>"}``
             arrives on stderr within a bounded timeout AND the version
             matches the installed plugin's hash; the runtime terminates
             ``amp`` immediately on detection (before any LLM call) to bound
             self-test latency to plugin-load time. The tool.call
             observation path is fallback-only if session.start stderr is
             buffered. Otherwise raise a fail-closed
             ``AmpPluginNotActiveError`` naming the most likely cause:

               - npm-installed Amp (binary install required)
               - Bun runtime missing
               - PLUGINS=all not honored
               - plugin version mismatch
               - amp binary missing from PATH

             The orchestrator refuses to spawn any issue agent in this case.
        """
        ...

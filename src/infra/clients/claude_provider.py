"""Claude :class:`AgentProvider` implementation.

Binds the existing Claude-side pieces (:class:`SDKClientFactory`,
:class:`ClaudeAgentRuntimeBuilder`, :class:`FileSystemLogProvider`) into a single
object that conforms to :class:`src.core.protocols.agent_provider.AgentProvider`.

Behavior is identical to the pre-refactor Claude path - this module is purely
a packaging change. The pipeline gains a single injection point for the coder
backend; no Claude logic moves or changes.

Lazy-import contract: the ``claude_agent_sdk`` package is imported by
:class:`SDKClientFactory` methods only, so importing this module does not pull
in the Claude SDK. Symmetrically, this module never imports any
``src.infra.clients.amp_*`` module - importing the Claude provider must not
require an Amp install.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from src.infra.agent_runtime import ClaudeAgentRuntimeBuilder
from src.infra.io.session_log_parser import FileSystemLogProvider
from src.infra.sdk_adapter import SDKClientFactory

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from src.core.protocols.agent_provider import CoderRuntimeBuilder
    from src.core.protocols.lifecycle import DeadlockMonitorProtocol
    from src.core.protocols.sdk import McpServerFactory


def _create_claude_mcp_server_factory() -> McpServerFactory:
    """Claude-shaped MCP factory returning in-process SDK ``Server`` objects.

    The Claude path consumes MCP servers as Python objects passed via
    ``ClaudeAgentOptions(mcp_servers=...)``; they are not JSON-serializable
    and cannot be reused for Amp's ``--mcp-config`` stdio launch spec.
    Owned by the Claude provider so the orchestrator does not branch on
    ``provider.name``.
    """
    from src.infra.tools.locking_mcp import create_locking_mcp_server

    def factory(
        agent_id: str,
        repo_path: Path,
        emit_lock_event: Callable | None,
    ) -> dict[str, object]:
        def _noop_handler(event: object) -> None:
            pass

        return {
            "mala-locking": create_locking_mcp_server(
                agent_id=agent_id,
                repo_namespace=str(repo_path),
                emit_lock_event=emit_lock_event or _noop_handler,
            )
        }

    return factory


class ClaudeAgentProvider:
    """:class:`AgentProvider` that bundles the existing Claude pieces.

    Attributes:
        name: Provider identifier (always ``"claude"``).
        client_factory: :class:`SDKClientFactory` for creating SDK clients.
        evidence_provider: :class:`FileSystemLogProvider` reading Claude
            session JSONL.
    """

    name: Literal["claude"] = "claude"

    def __init__(
        self,
        *,
        setting_sources: Sequence[str] | None = None,
        effort: str | None = None,
    ) -> None:
        """Initialize the provider.

        Args:
            setting_sources: Optional list of Claude settings sources (e.g.
                ``["local", "project"]``) threaded into every per-session
                :class:`ClaudeAgentRuntimeBuilder`. Mirrors the existing
                ``claude_settings_sources`` resolver wiring.
            effort: Optional Mala-level reasoning effort, forwarded to
                ``ClaudeAgentOptions.effort`` for every per-session coder
                runtime built from this provider. ``None`` leaves the Claude
                SDK default in place; config resolution supplies mala's default
                ``xhigh`` effort for normal orchestrator runs. Reviewer /
                epic-verifier ``ClaudeAgentOptions`` instances are
                intentionally untouched.
        """
        self.client_factory: SDKClientFactory = SDKClientFactory()
        self.evidence_provider: FileSystemLogProvider = FileSystemLogProvider()
        self._setting_sources: list[str] | None = (
            None if setting_sources is None else list(setting_sources)
        )
        self._effort: str | None = effort

    def runtime_builder(
        self,
        repo_path: Path,
        agent_id: str,
        *,
        mcp_server_factory: McpServerFactory,
        deadlock_monitor: DeadlockMonitorProtocol | None = None,
    ) -> CoderRuntimeBuilder:
        """Construct a per-session :class:`ClaudeAgentRuntimeBuilder`.

        The returned builder structurally conforms to
        :class:`CoderRuntimeBuilder` (cross-coder fluent surface);
        Claude-private callers that need ``with_hooks`` / ``with_disallowed_tools``
        downcast via ``isinstance(builder, ClaudeAgentRuntimeBuilder)``.
        """
        return ClaudeAgentRuntimeBuilder(
            repo_path,
            agent_id,
            self.client_factory,
            mcp_server_factory=mcp_server_factory,
            setting_sources=self._setting_sources,
            effort=self._effort,
            deadlock_monitor=deadlock_monitor,
        )

    def mcp_server_factory(self) -> McpServerFactory:
        """Return the Claude-shaped MCP server factory.

        Returns a fresh factory on each call (matching the prior
        orchestrator behavior); the factory itself produces a new
        ``mala-locking`` SDK ``Server`` per agent invocation.
        """
        return _create_claude_mcp_server_factory()

    def install_prerequisites(
        self,
        repo_path: Path,
        *,
        mcp_server_factory: McpServerFactory,
    ) -> None:
        """No-op for Claude.

        The Claude path has no install-time prerequisites (the SDK is a Python
        dependency, not a system binary, and there are no plugins to copy).
        The signature is required by :class:`AgentProvider`; the Amp provider
        uses the same hook to install ``mala-safety.ts`` and run the
        plugin-load self-test.
        """
        return

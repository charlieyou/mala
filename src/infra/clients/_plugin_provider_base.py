"""Shared plugin-provider base for Amp + Codex providers (Workstream B, Finding #5).

Protocol + helper module pattern (NOT ABC). Defines the cross-coder
install/selftest/MCP surface that ``AmpAgentProvider`` and
``CodexAgentProvider`` will both satisfy after Workstream B migrates
them (T_B5, T_B6). Claude is intentionally excluded: it uses
in-process MCP servers and has no plugin-gating self-test.

This module is the integration-path-test for Workstream B — the
contract that gates the provider migrations. It ships before the
providers move so the contract test exists when those migrations
land.

Exports:
  * :class:`PluginProviderSpec` — Protocol describing the shared
    install/selftest/MCP surface.
  * :class:`PluginInstallError` — Shared exception base raised when
    plugin install or runtime selftest fails closed. Provider-specific
    subclasses (``AmpPluginNotActiveError`` / ``CodexHookNotActiveError``)
    will inherit from this once T_B5/T_B6 land.
  * :class:`StdioMcpLaunchSpec` — Cross-coder stdio MCP launch spec
    shape produced by both providers' ``mcp_server_factory``.
  * :func:`install_with_selftest` — Shared driver capturing the
    install → cache-check → selftest pattern both providers use today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from src.core.protocols.agent_provider import CoderRuntimeBuilder
    from src.core.protocols.lifecycle import DeadlockMonitorProtocol
    from src.core.protocols.sdk import McpServerFactory


class PluginInstallError(RuntimeError):
    """Shared base for fail-closed plugin install / selftest errors.

    ``AmpPluginNotActiveError`` and ``CodexHookNotActiveError`` will be
    re-rooted to this base in T_B5 / T_B6 so the orchestrator can
    branch on a single exception type instead of two provider-specific
    classes.
    """


@dataclass(frozen=True)
class StdioMcpLaunchSpec:
    """Cross-coder stdio MCP launch spec.

    Both ``AmpAgentProvider`` and ``CodexAgentProvider`` emit specs of
    this shape from their ``mcp_server_factory``: a console-script
    ``command``, positional ``args``, and an ``env`` mapping forwarded
    to the spawned launcher. The Claude provider's in-process
    ``Server`` objects do not fit this shape and are not modeled here.

    The launch-spec shape mirrors the dict payload Amp's
    ``--mcp-config`` and Codex's ``.mcp.json`` consume; ``to_dict``
    renders it back into that dict form when crossing the
    serialization boundary.
    """

    command: str
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Render as the dict shape Amp/Codex MCP configs consume."""
        return {
            "command": self.command,
            "args": list(self.args),
            "env": dict(self.env),
        }


@runtime_checkable
class PluginProviderSpec(Protocol):
    """Install/selftest/MCP surface shared by plugin-gated providers.

    Narrower than ``src.core.protocols.agent_provider.AgentProvider``:
    focuses only on the surface that handles bundled-plugin install,
    runtime selftest gating, and the stdio MCP factory contract.
    Claude is excluded — it uses in-process MCP and has no plugin
    gating.

    Structural contract:
      * ``name``: provider identifier; one of ``"amp"`` / ``"codex"``.
      * ``mcp_server_factory()``: returns the stdio-shaped factory the
        provider uses for both runtime sessions and selftests.
      * ``install_prerequisites(repo_path, *, mcp_server_factory)``:
        idempotent setup invoked once per run; raises a
        :class:`PluginInstallError` (or subclass) on fail-closed.
      * ``runtime_builder(...)``: per-session builder reused by the
        selftest so the same env / MCP wiring is exercised.
    """

    name: Literal["amp", "codex"]

    def mcp_server_factory(self) -> McpServerFactory:
        """Return the stdio-shaped MCP server factory."""
        ...

    def install_prerequisites(
        self,
        repo_path: Path,
        *,
        mcp_server_factory: McpServerFactory,
    ) -> None:
        """Install bundled plugin and run runtime selftest.

        Raises:
            PluginInstallError: On any fail-closed install or selftest
                signature. Concrete providers may raise narrower
                subclasses with structured reason fields.
        """
        ...

    def runtime_builder(
        self,
        repo_path: Path,
        agent_id: str,
        *,
        mcp_server_factory: McpServerFactory,
        deadlock_monitor: DeadlockMonitorProtocol | None = None,
    ) -> CoderRuntimeBuilder:
        """Construct a per-session :class:`CoderRuntimeBuilder`."""
        ...


@dataclass
class SelftestCache:
    """Per-provider cache holder for :func:`install_with_selftest`.

    The selftest runs at most once per ``(coder_version, plugin_hash)``
    pair within a run. ``AmpAgentProvider`` and ``CodexAgentProvider``
    both hold one of these instances on the provider so subsequent
    calls within the same run short-circuit.
    """

    key: tuple[str, str] | None = None


def install_with_selftest(
    *,
    install: Callable[[], str],
    selftest: Callable[[str], None],
    cache: SelftestCache,
    coder_version: str = "unknown",
) -> bool:
    """Run plugin install + cache-key check + selftest in the shared order.

    Captures the install → cache-check → selftest pattern that both
    Amp and Codex providers implement today. The cache short-circuits
    on the ``(coder_version, plugin_hash)`` key both providers use.

    Args:
        install: Idempotent installer; returns the installed plugin's
            16-hex-char SHA-256 prefix used as both the cache key
            component and the selftest's expected version marker.
        selftest: Runtime selftest probe called with the plugin hash.
            Must raise :class:`PluginInstallError` (or a subclass) on
            any fail-closed signature; the cache is left unchanged
            when selftest raises so a retry re-runs the probe.
        cache: Per-provider cache. Mutated in place on success.
        coder_version: Resolved coder version included in the cache
            key. Amp and Codex both currently use the placeholder
            ``"unknown"``; a future reliable ``<coder> --version``
            parse drops in additively.

    Returns:
        ``True`` when the selftest ran in this call, ``False`` when
        the call short-circuited because the cache key matched a
        previous successful run.
    """
    plugin_hash = install()
    key = (coder_version, plugin_hash)
    if cache.key == key:
        return False
    selftest(plugin_hash)
    cache.key = key
    return True

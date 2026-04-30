"""Stub :class:`AmpAgentProvider` implementation.

This is the T007 stub: it conforms structurally to
:class:`src.core.protocols.agent_provider.AgentProvider` so factory selection
can wire it through ``OrchestratorDependencies``, but every component raises
:class:`NotImplementedError` when actually used. Subsequent tasks
(T008 - T013) replace each piece with a real implementation:

  - T008: real ``AmpClient`` (subprocess + stream-json adapter)
  - T009: real :class:`AmpRuntimeBuilder` integration (``runtime_builder``)
  - T010: ``AmpLogProvider`` (probe-then-tee)
  - T011: ``AmpPluginInstaller``
  - T012: ``plugins/amp/mala-safety.ts``
  - T013: real ``install_prerequisites`` self-test (currently a no-op)

The stub is intentionally minimal so the failing fake-amp integration test
(:mod:`tests/integration/test_amp_provider`) demonstrates the wired-but-not-
implemented state. T013 turns that test green.

Lazy-import contract (mirror of :mod:`claude_provider`): importing this
module must not pull in real Amp infrastructure (no ``amp_client``,
``amp_log_provider``, etc.). The pieces that *are* imported here -
:mod:`amp_runtime` and :mod:`amp_messages` - are the data-only modules
defined in T002; they have no third-party dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from src.infra.clients.amp_runtime import AmpRuntimeBuilder

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path
    from types import TracebackType
    from typing import Self

    from src.core.protocols.agent_provider import CoderRuntimeBuilder
    from src.core.protocols.log import LogProvider
    from src.core.protocols.sdk import (
        McpServerFactory,
        SDKClientFactoryProtocol,
        SDKClientProtocol,
    )
    from src.infra.clients.amp_runtime import AmpMode


class _StubAmpClient:
    """Stub Amp client. Every operation raises :class:`NotImplementedError`.

    Real impl arrives in T008.
    """

    async def __aenter__(self) -> Self:
        raise NotImplementedError(
            "AmpClient is not implemented yet (see T008). "
            "The Amp provider was wired through orchestration in T007 but "
            "the subprocess + stream-json adapter is pending."
        )

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        return None

    async def query(self, prompt: str, session_id: str | None = None) -> None:
        raise NotImplementedError("AmpClient.query pending (T008)")

    def receive_response(self) -> AsyncIterator[object]:
        raise NotImplementedError("AmpClient.receive_response pending (T008)")

    async def disconnect(self) -> None:
        raise NotImplementedError("AmpClient.disconnect pending (T008)")


class _StubAmpClientFactory:
    """Stub :class:`SDKClientFactoryProtocol` for Amp.

    ``create()``, ``create_options()``, and ``with_resume()`` raise
    :class:`NotImplementedError`. Real impl arrives in T008.
    """

    def create(self, options: object) -> SDKClientProtocol:
        raise NotImplementedError(
            "AmpAgentProvider.client_factory.create() is not implemented yet "
            "(see T008). The Amp provider was wired through orchestration in "
            "T007 but the subprocess client is pending."
        )

    def create_options(
        self,
        *,
        cwd: str,
        permission_mode: str = "bypassPermissions",
        model: str = "opus",
        system_prompt: dict[str, str] | None = None,
        output_format: object | None = None,
        settings: str | None = None,
        setting_sources: list[str] | None = None,
        mcp_servers: object | None = None,
        disallowed_tools: list[str] | None = None,
        env: dict[str, str] | None = None,
        hooks: dict[str, list[object]] | None = None,
        resume: str | None = None,
    ) -> object:
        raise NotImplementedError(
            "AmpAgentProvider.client_factory.create_options() is not "
            "implemented yet (see T008). Amp uses a CLI-flag-shaped runtime, "
            "not Claude SDK options - this method exists for protocol "
            "conformance only and should not be invoked on the Amp path."
        )

    def create_hook_matcher(
        self,
        matcher: object | None,
        hooks: list[object],
    ) -> object:
        raise NotImplementedError(
            "AmpAgentProvider.client_factory.create_hook_matcher() is not "
            "implemented (Amp uses a TypeScript plugin, not SDK hooks)."
        )

    def with_resume(self, options: object, resume: str | None) -> object:
        raise NotImplementedError(
            "AmpAgentProvider.client_factory.with_resume() is not implemented "
            "yet (see T008). Amp resumes via thread-id, not SDK session id."
        )


class _StubAmpLogProvider:
    """Stub :class:`LogProvider` for Amp. Real impl arrives in T010."""

    def get_log_path(self, session_id: str) -> Path:
        raise NotImplementedError(
            "AmpLogProvider.get_log_path() is not implemented yet (see T010). "
            "The provider needs the tee path resolution logic and the "
            "native-log probe."
        )

    def iter_events(self, session_id: str) -> object:
        raise NotImplementedError(
            "AmpLogProvider.iter_events() is not implemented yet (see T010)."
        )


class AmpAgentProvider:
    """Stub :class:`AgentProvider` for Amp.

    Conforms structurally to
    :class:`src.core.protocols.agent_provider.AgentProvider`, so the factory
    can pick it when ``MalaConfig.coder == "amp"`` and inject it into
    ``OrchestratorDependencies``. Every component intentionally raises
    :class:`NotImplementedError` when actually used; ``install_prerequisites``
    is a no-op stub. T008 - T013 replace these with real implementations.

    The :meth:`runtime_builder` method DOES return a real
    :class:`AmpRuntimeBuilder` (its data-only build was completed in T002),
    so the runtime construction path is exercised end-to-end. Failure occurs
    when the pipeline subsequently calls ``client_factory.create(...)``.

    Attributes:
        name: Provider identifier (always ``"amp"``).
        client_factory: Stub :class:`SDKClientFactoryProtocol`; raises on use.
        log_provider: Stub :class:`LogProvider`; raises on use.
    """

    name: Literal["amp"] = "amp"

    def __init__(self, *, mode: AmpMode = "smart") -> None:
        """Initialize the stub provider.

        Args:
            mode: Amp execution mode (``"smart" | "rush" | "deep"``). Stored
                here so the orchestration layer can resolve the precedence
                (CLI > env > yaml > default) at construction time, even
                though the real client/runtime path is not yet wired.
        """
        self._mode: AmpMode = mode
        self.client_factory: SDKClientFactoryProtocol = cast(
            "SDKClientFactoryProtocol", _StubAmpClientFactory()
        )
        self.log_provider: LogProvider = cast("LogProvider", _StubAmpLogProvider())

    def runtime_builder(
        self,
        repo_path: Path,
        agent_id: str,
        *,
        mcp_server_factory: McpServerFactory,
    ) -> CoderRuntimeBuilder:
        """Construct an :class:`AmpRuntimeBuilder` for one Amp session.

        Returns the real builder shipped in T002; ``build()`` produces a
        complete :class:`AmpRuntime` with argv, env, and ``--mcp-config``.
        The runtime is never consumed in T007 because
        ``client_factory.create(...)`` raises before any subprocess spawns.
        """
        return AmpRuntimeBuilder(
            repo_path,
            agent_id,
            mcp_server_factory,
            mode=self._mode,
        )

    def install_prerequisites(
        self,
        repo_path: Path,
        *,
        mcp_server_factory: McpServerFactory,
    ) -> None:
        """No-op stub for T007.

        T013 implements the real plugin self-test (fail-closed, safety-
        critical). Until then this method silently succeeds so factory
        wiring can be exercised by tests; the failing fake-amp integration
        test in :mod:`tests/integration/test_amp_provider` proves the rest
        of the path is not yet implemented (it surfaces a
        :class:`NotImplementedError` from the stub client factory).
        """
        del repo_path, mcp_server_factory
        return

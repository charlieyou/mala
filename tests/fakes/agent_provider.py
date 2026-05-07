"""Test :class:`FakeAgentProvider` for the AgentProvider protocol.

The pipeline classes (``AgentSessionRunner``, ``FixerService``,
``RunCoordinator``) consume an ``AgentProvider`` rather than the legacy
``SDKClientFactoryProtocol``. Most tests still want to inject a
``FakeSDKClientFactory`` to control SDK responses; this fake bundles a
factory + ``AgentRuntimeBuilder`` + ``FileSystemLogProvider`` so existing
tests can swap one parameter (``sdk_client_factory=`` -> ``agent_provider=``)
without rewriting their setup.

The injected factory is typed as the Claude-private
:class:`ClaudeSDKClientFactoryProtocol` because the bundled
:class:`AgentRuntimeBuilder` is Claude-flavored and consumes the Claude
knobs (``create_options``, ``create_hook_matcher``); test fakes
already implement the full Claude surface.

The fake is identifier-as-claude (``name="claude"``) by default because the
fluent runtime builder it returns is the Claude-flavored
:class:`AgentRuntimeBuilder` and the test pipeline assumes Claude-shaped
options. Tests that exercise Amp selection should construct a real
``AmpAgentProvider`` (see ``tests/unit/orchestration/test_factory_provider_selection.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from src.infra.agent_runtime import AgentRuntimeBuilder

if TYPE_CHECKING:
    from pathlib import Path

    from src.core.protocols.agent_provider import CoderRuntimeBuilder
    from src.core.protocols.log import LogProvider
    from src.core.protocols.sdk import McpServerFactory, SDKClientFactoryProtocol
    from src.infra.agent_runtime import ClaudeSDKClientFactoryProtocol


class FakeAgentProvider:
    """Minimal :class:`AgentProvider` for tests.

    Wraps an injected :class:`ClaudeSDKClientFactoryProtocol` (typically
    a ``FakeSDKClientFactory``) and exposes a ``runtime_builder()`` that
    returns a real :class:`AgentRuntimeBuilder` wired to that factory.

    Observable state mirrors the wrapped factory; tests assert on the
    factory directly (``provider.client_factory.create_calls``, etc.).

    Attributes:
        name: ``"claude"`` by default; tests for Amp selection construct
            ``AmpAgentProvider`` directly instead.
        client_factory: The wrapped factory.
        log_provider: Optional override; defaults to
            :class:`FileSystemLogProvider`.
    """

    name: Literal["claude", "amp"] = "claude"

    def __init__(
        self,
        client_factory: ClaudeSDKClientFactoryProtocol,
        *,
        log_provider: LogProvider | None = None,
        install_prerequisites_count: int = 0,
        name: Literal["claude", "amp"] = "claude",
        setting_sources: list[str] | None = None,
    ) -> None:
        # ``client_factory`` is annotated as the slim cross-coder protocol
        # to satisfy the ``AgentProvider`` shape (which is what pipeline
        # code reads). Internally, the constructor accepts the Claude-full
        # protocol so that ``runtime_builder()`` can hand it to
        # :class:`AgentRuntimeBuilder` (which needs ``create_options`` /
        # ``create_hook_matcher``); the cast is safe because every fake
        # passed in ships those Claude knobs.
        self.client_factory: SDKClientFactoryProtocol = client_factory
        self._claude_factory: ClaudeSDKClientFactoryProtocol = client_factory
        if log_provider is None:
            from src.infra.io.session_log_parser import FileSystemLogProvider

            self.log_provider: LogProvider = FileSystemLogProvider()  # type: ignore[assignment]
        else:
            self.log_provider = log_provider
        self.name = name
        # Track install_prerequisites calls for tests that verify idempotency
        # / single invocation (e.g. T007 factory tests).
        self.install_prerequisites_count = install_prerequisites_count
        self._setting_sources: list[str] | None = setting_sources

    def runtime_builder(
        self,
        repo_path: Path,
        agent_id: str,
        *,
        mcp_server_factory: McpServerFactory,
    ) -> CoderRuntimeBuilder:
        return AgentRuntimeBuilder(
            repo_path,
            agent_id,
            self._claude_factory,
            mcp_server_factory=mcp_server_factory,
            setting_sources=self._setting_sources,
        )

    def install_prerequisites(
        self,
        repo_path: Path,
        *,
        mcp_server_factory: McpServerFactory,
    ) -> None:
        del repo_path, mcp_server_factory
        self.install_prerequisites_count += 1

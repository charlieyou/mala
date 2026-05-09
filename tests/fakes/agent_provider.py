"""Test :class:`FakeAgentProvider` for the AgentProvider protocol.

The pipeline classes (``AgentSessionRunner``, ``FixerService``,
``RunCoordinator``) consume an ``AgentProvider`` rather than the legacy
``SDKClientFactoryProtocol``. Most tests still want to inject a
``FakeSDKClientFactory`` to control SDK responses; this fake bundles a
factory + ``ClaudeAgentRuntimeBuilder`` + ``FileSystemLogProvider`` so
existing tests can swap one parameter (``sdk_client_factory=`` ->
``agent_provider=``) without rewriting their setup.

The injected factory is typed as the Claude-private
:class:`ClaudeSDKClientFactoryProtocol` because the bundled
:class:`ClaudeAgentRuntimeBuilder` is Claude-flavored and consumes the
Claude knobs (``create_options``, ``create_hook_matcher``); test fakes
already implement the full Claude surface.

The fake is identifier-as-claude (``name="claude"``) by default because the
fluent runtime builder it returns is the Claude-flavored
:class:`ClaudeAgentRuntimeBuilder` and the test pipeline assumes
Claude-shaped options. Tests that exercise Amp selection should construct
a real ``AmpAgentProvider`` (see
``tests/unit/orchestration/test_factory_provider_selection.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from src.infra.agent_runtime import ClaudeAgentRuntimeBuilder
from src.infra.io.session_log_parser import FileSystemLogProvider

if TYPE_CHECKING:
    from pathlib import Path

    from src.core.protocols.agent_provider import CoderRuntimeBuilder
    from src.core.protocols.evidence import EvidenceProvider
    from src.core.protocols.lifecycle import DeadlockMonitorProtocol
    from src.core.protocols.sdk import McpServerFactory, SDKClientFactoryProtocol
    from src.infra.agent_runtime import ClaudeSDKClientFactoryProtocol


class _ImmediateReadyFileSystemLogProvider(FileSystemLogProvider):
    """:class:`FileSystemLogProvider` with a no-op readiness wait.

    Tests using :class:`FakeAgentProvider` configure the *path* the
    runner observes via :class:`StubSessionLifecycle.on_get_log_path`,
    not via the real Claude SDK location. Phase A7 / issue
    ``mala-dkm9e`` made the readiness gate session-id-shaped:
    :meth:`EvidenceProvider.wait_for_session_ready` now resolves the
    log path internally from ``(repo_path, session_id)`` instead of
    being handed a path. The default :class:`FileSystemLogProvider`
    resolves to ``~/.claude/projects/...``, which never exists in
    tests, so the wait would always time out.

    This subclass preserves all parsing/extraction behavior of
    :class:`FileSystemLogProvider` while overriding the readiness
    wait to return immediately. Tests that exercise readiness-gate
    behavior (fast/slow/timeout) inject their own stub provider
    (see ``tests/unit/pipeline/test_lifecycle_wait_for_log.py``).
    """

    async def wait_for_session_ready(  # type: ignore[override]
        self,
        repo_path: Path,
        session_id: str,
        *,
        timeout: float,
        poll_interval: float = 0.5,
    ) -> None:
        del repo_path, session_id, timeout, poll_interval


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
        evidence_provider: Optional override; defaults to a wrapper
            around :class:`FileSystemLogProvider` whose
            :meth:`wait_for_session_ready` is a no-op (see
            :class:`_ImmediateReadyFileSystemLogProvider`).
    """

    name: Literal["claude", "amp", "codex"] = "claude"

    def __init__(
        self,
        client_factory: ClaudeSDKClientFactoryProtocol,
        *,
        evidence_provider: EvidenceProvider | None = None,
        install_prerequisites_count: int = 0,
        name: Literal["claude", "amp", "codex"] = "claude",
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
        if evidence_provider is None:
            self.evidence_provider: EvidenceProvider = (
                _ImmediateReadyFileSystemLogProvider()
            )  # type: ignore[assignment]
        else:
            self.evidence_provider = evidence_provider
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
        deadlock_monitor: DeadlockMonitorProtocol | None = None,
    ) -> CoderRuntimeBuilder:
        return ClaudeAgentRuntimeBuilder(
            repo_path,
            agent_id,
            self._claude_factory,
            mcp_server_factory=mcp_server_factory,
            setting_sources=self._setting_sources,
            deadlock_monitor=deadlock_monitor,
        )

    def mcp_server_factory(self) -> McpServerFactory:
        """Return a no-op MCP factory for tests.

        Tests that need a real Claude / Amp factory should construct the
        concrete provider; this fake just satisfies the protocol.
        """
        from typing import cast

        def _factory(
            agent_id: str,
            repo_path: Path,
            emit_lock_event: object,
        ) -> dict[str, object]:
            del agent_id, repo_path, emit_lock_event
            return {}

        return cast("McpServerFactory", _factory)

    def install_prerequisites(
        self,
        repo_path: Path,
        *,
        mcp_server_factory: McpServerFactory,
    ) -> None:
        del repo_path, mcp_server_factory
        self.install_prerequisites_count += 1

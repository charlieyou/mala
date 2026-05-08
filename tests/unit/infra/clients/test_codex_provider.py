"""Unit tests for the Phase B :class:`CodexAgentProvider` stub.

Covers:
  * Conformance to :class:`AgentProvider` runtime protocol.
  * ``name == "codex"`` invariant.
  * Stub fail-closed semantics: every entry point that would normally
    drive a Codex turn (``client_factory.create``, ``runtime_builder``
    consumers, ``install_prerequisites``) raises
    :class:`CodexNotImplementedError`.
  * Lazy-import contract: importing ``codex_provider`` does NOT pull in
    ``codex_app_server`` (the SDK lands in Phase C).
  * Resolved Codex options carried by the provider reach the runtime
    builder's stub runtime (integration-path evidence for AC #4).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from src.core.protocols.agent_provider import AgentProvider
from src.infra.clients.codex_provider import (
    CodexAgentProvider,
    CodexNotImplementedError,
    _CodexStubRuntime,
)

if TYPE_CHECKING:
    from collections.abc import Callable


REPO_ROOT = Path(__file__).resolve().parents[5]


@pytest.fixture
def fake_mcp_factory() -> Callable[..., dict[str, object]]:
    """No-op MCP factory satisfying the protocol shape."""

    def factory(
        agent_id: str,
        repo_path: Path,
        emit_lock_event: object,
    ) -> dict[str, object]:
        del agent_id, repo_path, emit_lock_event
        return {}

    return factory


@pytest.mark.unit
def test_provider_conforms_to_agent_provider_protocol() -> None:
    provider = CodexAgentProvider()
    assert isinstance(provider, AgentProvider)
    assert provider.name == "codex"


@pytest.mark.unit
def test_default_options_match_unattended_run_defaults() -> None:
    """Plan decisions #2, #3, #9: defaults for unattended Codex runs."""
    provider = CodexAgentProvider()
    assert provider.model == "gpt-5.5"
    assert provider.effort is None
    assert provider.approval_policy == "never"
    assert provider.sandbox == "danger-full-access"


@pytest.mark.unit
def test_client_factory_create_raises_not_implemented(tmp_path: Path) -> None:
    """``client_factory.create`` must raise the structured stub error."""
    provider = CodexAgentProvider()
    runtime = _CodexStubRuntime(
        cwd=tmp_path,
        agent_id="agent-x",
        model="gpt-5.5",
        effort=None,
        approval_policy="never",
        sandbox="danger-full-access",
    )
    with pytest.raises(CodexNotImplementedError, match="Phases C-I"):
        provider.client_factory.create(runtime)


@pytest.mark.unit
def test_install_prerequisites_raises_not_implemented(
    tmp_path: Path, fake_mcp_factory: Callable[..., dict[str, object]]
) -> None:
    """Phase B must fail-closed when the orchestrator boots a Codex run."""
    provider = CodexAgentProvider()
    with pytest.raises(CodexNotImplementedError, match="Phases C-I"):
        provider.install_prerequisites(tmp_path, mcp_server_factory=fake_mcp_factory)


@pytest.mark.unit
def test_runtime_builder_threads_resolved_options(
    tmp_path: Path, fake_mcp_factory: Callable[..., dict[str, object]]
) -> None:
    """Configured Codex options reach the per-session runtime.

    Integration-path evidence for AC #4: the resolved values flow
    ``MalaConfig.coder_options.codex`` -> ``CodexAgentProvider`` ->
    runtime builder's stub runtime, even before the real
    ``CodexRuntime`` lands in Phase C.
    """
    provider = CodexAgentProvider(
        model="gpt-5.5-foo",
        effort="medium",
        approval_policy="on-request",
        sandbox="workspace-write",
    )

    builder = provider.runtime_builder(
        tmp_path, "agent-x", mcp_server_factory=fake_mcp_factory
    )
    runtime = builder.build()
    assert isinstance(runtime, _CodexStubRuntime)
    assert runtime.model == "gpt-5.5-foo"
    assert runtime.effort == "medium"
    assert runtime.approval_policy == "on-request"
    assert runtime.sandbox == "workspace-write"
    assert runtime.cwd == tmp_path
    assert runtime.agent_id == "agent-x"


@pytest.mark.unit
def test_runtime_builder_records_resume_token(
    tmp_path: Path, fake_mcp_factory: Callable[..., dict[str, object]]
) -> None:
    """``with_resume`` carries the resume id onto the built runtime."""
    provider = CodexAgentProvider()
    builder = provider.runtime_builder(
        tmp_path, "agent-x", mcp_server_factory=fake_mcp_factory
    )
    runtime = builder.with_resume("thr_123").build()
    assert isinstance(runtime, _CodexStubRuntime)
    assert runtime.resume_thread_id == "thr_123"


@pytest.mark.unit
def test_evidence_provider_is_fail_closed_stub(tmp_path: Path) -> None:
    """The Codex evidence provider stub raises on every method.

    Phase F replaces this with the native ``Thread.read`` adapter; until
    then any caller asking for evidence under ``coder=codex`` should hit
    a clear "not yet implemented" failure rather than crashing later
    inside a half-built code path.
    """
    provider = CodexAgentProvider()
    evidence = provider.evidence_provider
    with pytest.raises(CodexNotImplementedError):
        evidence.get_log_path(tmp_path, "thr_x")
    with pytest.raises(CodexNotImplementedError):
        evidence.get_end_offset(tmp_path / "x.jsonl")


@pytest.mark.unit
def test_importing_codex_provider_does_not_pull_codex_app_server() -> None:
    """Phase B lazy-import contract.

    ``import src.infra.clients.codex_provider`` must NOT transitively
    import ``codex_app_server``. Phase C will introduce the SDK
    dependency, but Phase B keeps the stub free of any SDK
    transitive imports so a Claude/Amp-only run is not penalized by
    the Codex stub being on disk.
    """
    code = """
import sys
from src.infra.clients.codex_provider import CodexAgentProvider
provider = CodexAgentProvider()
# Touching the lazy-instantiated factory must not load codex_app_server either.
provider.client_factory  # noqa: B018
provider.evidence_provider  # noqa: B018
loaded = sorted(m for m in sys.modules if m.startswith('codex_app_server'))
if loaded:
    print('FAIL: ' + ','.join(loaded))
    sys.exit(1)
print('PASS')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "PASS" in result.stdout

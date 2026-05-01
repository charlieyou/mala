"""Unit tests for AgentProvider and CoderRuntimeBuilder protocols."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.core.protocols.agent_provider import AgentProvider, CoderRuntimeBuilder

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from src.core.protocols.log import JsonlEntryProtocol
    from src.core.protocols.sdk import McpServerFactory


pytestmark = pytest.mark.unit


class _StubRuntimeBuilder:
    """Minimal stub satisfying the :class:`CoderRuntimeBuilder` protocol."""

    def build(self) -> object:
        return object()


class _StubClientFactory:
    """Minimal stub satisfying :class:`SDKClientFactoryProtocol`."""

    def create(self, options: object) -> object:
        return object()

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
        return object()

    def create_hook_matcher(
        self, matcher: object | None, hooks: list[object]
    ) -> object:
        return object()

    def with_resume(self, options: object, resume: str | None) -> object:
        return options


class _StubLogProvider:
    """Minimal stub satisfying :class:`LogProvider`."""

    def get_log_path(self, repo_path: Path, session_id: str) -> Path:
        return repo_path / f"{session_id}.jsonl"

    def iter_events(
        self, log_path: Path, offset: int = 0
    ) -> Iterator[JsonlEntryProtocol]:
        return iter(())

    def iter_thread_events(
        self, log_path: Path, offset: int = 0
    ) -> Iterator[JsonlEntryProtocol]:
        return iter(())

    def get_end_offset(self, log_path: Path, start_offset: int = 0) -> int:
        return start_offset

    def extract_bash_commands(self, entry: JsonlEntryProtocol) -> list[tuple[str, str]]:
        return []

    def extract_tool_results(self, entry: JsonlEntryProtocol) -> list[tuple[str, bool]]:
        return []

    def extract_assistant_text_blocks(self, entry: JsonlEntryProtocol) -> list[str]:
        return []

    def extract_tool_result_content(
        self, entry: JsonlEntryProtocol
    ) -> list[tuple[str, str]]:
        return []


class _StubAgentProvider:
    """Minimal stub satisfying :class:`AgentProvider`."""

    name: str = "claude"

    def __init__(self) -> None:
        self.client_factory: object = _StubClientFactory()
        self.log_provider: object = _StubLogProvider()

    def runtime_builder(
        self,
        repo_path: Path,
        agent_id: str,
        *,
        mcp_server_factory: McpServerFactory,
    ) -> CoderRuntimeBuilder:
        return _StubRuntimeBuilder()

    def install_prerequisites(
        self,
        repo_path: Path,
        *,
        mcp_server_factory: McpServerFactory,
    ) -> None:
        return None


def test_coder_runtime_builder_is_runtime_checkable() -> None:
    builder = _StubRuntimeBuilder()
    assert isinstance(builder, CoderRuntimeBuilder)


def test_object_without_build_is_not_a_coder_runtime_builder() -> None:
    assert not isinstance(object(), CoderRuntimeBuilder)


def test_agent_provider_is_runtime_checkable() -> None:
    provider = _StubAgentProvider()
    assert isinstance(provider, AgentProvider)


def test_object_without_required_members_is_not_an_agent_provider() -> None:
    assert not isinstance(object(), AgentProvider)


def test_agent_provider_protocol_surface() -> None:
    """Plan L268-L285 prescribes these protocol members."""
    annotations = AgentProvider.__annotations__
    assert "name" in annotations
    assert "client_factory" in annotations
    assert "log_provider" in annotations
    assert callable(getattr(AgentProvider, "runtime_builder", None))
    assert callable(getattr(AgentProvider, "install_prerequisites", None))


def test_coder_runtime_builder_protocol_surface() -> None:
    assert callable(getattr(CoderRuntimeBuilder, "build", None))

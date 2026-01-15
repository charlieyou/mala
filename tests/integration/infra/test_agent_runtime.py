"""Integration tests for AgentRuntimeBuilder hook registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src.infra.agent_runtime import AgentRuntimeBuilder
from tests.fakes import FakeSDKClientFactory

if TYPE_CHECKING:
    from pathlib import Path


class TestPreCompactHookRegistration:
    """Tests for PreCompact hook registration in AgentRuntimeBuilder."""

    @pytest.fixture
    def repo_path(self, tmp_path: Path) -> Path:
        """Create a temporary repo path."""
        return tmp_path

    @pytest.fixture
    def factory(self) -> FakeSDKClientFactory:
        """Create a fake SDK client factory."""
        return FakeSDKClientFactory()

    @pytest.mark.integration
    def test_precompact_hook_registered_in_hooks_dict(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """PreCompact hook is registered in the hooks dict passed to SDK options."""
        builder = AgentRuntimeBuilder(repo_path, "test-agent-precompact", factory)
        builder.with_mcp(servers={})  # Disable locking MCP for test
        builder.build()

        # Get the hooks dict from the last create_options call
        assert len(factory.created_options) == 1
        hooks_dict = factory.created_options[0].get("hooks", {})

        # Verify PreCompact key exists
        assert "PreCompact" in hooks_dict, (
            "PreCompact hook not registered in hooks dict"
        )

        # Verify it's a list with one matcher tuple
        precompact_matchers = hooks_dict["PreCompact"]
        assert isinstance(precompact_matchers, list)
        assert len(precompact_matchers) == 1

    @pytest.mark.integration
    def test_precompact_hook_is_callable(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """PreCompact hook matcher contains a callable hook."""
        builder = AgentRuntimeBuilder(repo_path, "test-agent-callable", factory)
        builder.with_mcp(servers={})
        builder.build()

        hooks_dict = factory.created_options[0].get("hooks", {})
        precompact_matchers = hooks_dict["PreCompact"]

        # FakeSDKClientFactory.create_hook_matcher returns ("matcher", matcher, hooks)
        matcher_tuple = precompact_matchers[0]
        assert isinstance(matcher_tuple, tuple)
        assert len(matcher_tuple) == 3
        hooks_list = matcher_tuple[2]  # Third element is the hooks list
        assert len(hooks_list) == 1
        assert callable(hooks_list[0])

    @pytest.mark.integration
    async def test_precompact_hook_returns_empty_dict(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """PreCompact hook stub returns empty dict (allows compaction)."""
        builder = AgentRuntimeBuilder(repo_path, "test-agent-stub", factory)
        builder.with_mcp(servers={})
        builder.build()

        hooks_dict = factory.created_options[0].get("hooks", {})
        precompact_matchers = hooks_dict["PreCompact"]
        matcher_tuple = precompact_matchers[0]
        hook = matcher_tuple[2][0]  # Get first hook from hooks list

        # Call the hook with minimal input
        result = await hook({"session_id": "test", "trigger": "auto"})

        assert result == {}, "PreCompact stub should return empty dict"

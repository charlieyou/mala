"""Unit tests for AgentRuntimeBuilder.

Tests the centralized agent runtime configuration builder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from src.infra.agent_runtime import AgentRuntime, AgentRuntimeBuilder
from src.infra.hooks import LintCache

if TYPE_CHECKING:
    from pathlib import Path


class FakeSDKClientFactory:
    """Fake SDK client factory for testing."""

    def __init__(self) -> None:
        self.created_options: list[dict] = []
        self.created_matchers: list[tuple] = []

    def create_options(
        self,
        *,
        cwd: str,
        permission_mode: str = "bypassPermissions",
        model: str = "opus",
        system_prompt: dict | None = None,
        setting_sources: list[str] | None = None,
        mcp_servers: object | None = None,
        disallowed_tools: list[str] | None = None,
        env: dict[str, str] | None = None,
        hooks: dict[str, list] | None = None,
    ) -> object:
        opts = {
            "cwd": cwd,
            "permission_mode": permission_mode,
            "model": model,
            "mcp_servers": mcp_servers,
            "disallowed_tools": disallowed_tools,
            "env": env,
            "hooks": hooks,
        }
        self.created_options.append(opts)
        return opts

    def create_hook_matcher(
        self,
        matcher: object | None,
        hooks: list[object],
    ) -> object:
        result = ("matcher", matcher, hooks)
        self.created_matchers.append(result)
        return result

    def create(self, options: object) -> object:
        return MagicMock()


class TestAgentRuntimeBuilder:
    """Tests for AgentRuntimeBuilder."""

    @pytest.fixture
    def repo_path(self, tmp_path: Path) -> Path:
        """Create a temporary repo path."""
        return tmp_path

    @pytest.fixture
    def factory(self) -> FakeSDKClientFactory:
        """Create a fake SDK client factory."""
        return FakeSDKClientFactory()

    @pytest.mark.unit
    def test_build_returns_agent_runtime(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """build() returns AgentRuntime with all components."""
        runtime = (
            AgentRuntimeBuilder(repo_path, "test-agent-123", factory)
            .with_env()
            .with_mcp()
            .with_disallowed_tools()
            .build()
        )

        assert isinstance(runtime, AgentRuntime)
        assert runtime.options is not None
        assert isinstance(runtime.lint_cache, LintCache)
        assert isinstance(runtime.env, dict)
        assert len(runtime.pre_tool_hooks) > 0
        assert len(runtime.stop_hooks) > 0

    @pytest.mark.unit
    def test_env_includes_required_vars(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """with_env() includes PATH, LOCK_DIR, AGENT_ID, REPO_NAMESPACE."""
        runtime = (
            AgentRuntimeBuilder(repo_path, "agent-xyz", factory).with_env().build()
        )

        assert "PATH" in runtime.env
        assert "LOCK_DIR" in runtime.env
        assert runtime.env["AGENT_ID"] == "agent-xyz"
        assert runtime.env["REPO_NAMESPACE"] == str(repo_path)
        assert runtime.env["MCP_TIMEOUT"] == "300000"

    @pytest.mark.unit
    def test_env_extra_vars_merged(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """with_env(extra) merges extra variables."""
        runtime = (
            AgentRuntimeBuilder(repo_path, "agent-1", factory)
            .with_env(extra={"CUSTOM_VAR": "value123"})
            .build()
        )

        assert runtime.env["CUSTOM_VAR"] == "value123"
        assert runtime.env["AGENT_ID"] == "agent-1"

    @pytest.mark.unit
    def test_fluent_api_returns_self(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """Each with_* method returns self for chaining."""
        builder = AgentRuntimeBuilder(repo_path, "agent-chain", factory)

        result1 = builder.with_hooks()
        assert result1 is builder

        result2 = builder.with_env()
        assert result2 is builder

        result3 = builder.with_mcp()
        assert result3 is builder

        result4 = builder.with_disallowed_tools()
        assert result4 is builder

        result5 = builder.with_lint_tools({"ruff"})
        assert result5 is builder

    @pytest.mark.unit
    def test_pre_tool_hooks_ordering(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """Pre-tool hooks are in correct order."""
        runtime = AgentRuntimeBuilder(repo_path, "agent-hooks", factory).build()

        # Should have at least: dangerous_commands, disallowed_tools, lock_enforcement,
        # file_cache, lint_cache
        assert len(runtime.pre_tool_hooks) >= 5

    @pytest.mark.unit
    def test_stop_hook_included_by_default(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """Stop hook is included by default."""
        runtime = AgentRuntimeBuilder(repo_path, "agent-stop", factory).build()

        assert len(runtime.stop_hooks) == 1

    @pytest.mark.unit
    def test_stop_hook_can_be_excluded(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """with_hooks(include_stop_hook=False) excludes stop hook."""
        runtime = (
            AgentRuntimeBuilder(repo_path, "agent-no-stop", factory)
            .with_hooks(include_stop_hook=False)
            .build()
        )

        assert len(runtime.stop_hooks) == 0

    @pytest.mark.unit
    def test_deadlock_monitor_adds_hooks(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """with_hooks(deadlock_monitor=...) adds lock event hooks."""
        mock_monitor = MagicMock()

        runtime = (
            AgentRuntimeBuilder(repo_path, "agent-deadlock", factory)
            .with_hooks(deadlock_monitor=mock_monitor)
            .build()
        )

        # Deadlock monitor adds one pre-tool hook (lock_wait) and one post-tool hook
        # Pre-tool hooks: 5 base + 1 lock_wait = 6
        assert len(runtime.pre_tool_hooks) == 6
        assert len(runtime.post_tool_hooks) == 1

    @pytest.mark.unit
    def test_lint_tools_passed_to_cache(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """with_lint_tools() configures lint cache."""
        runtime = (
            AgentRuntimeBuilder(repo_path, "agent-lint", factory)
            .with_lint_tools({"ruff", "mypy"})
            .build()
        )

        # LintCache should have the specified lint tools
        assert runtime.lint_cache.lint_tools == {"ruff", "mypy"}

    @pytest.mark.unit
    def test_mcp_servers_passed_to_options(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """with_mcp(servers) passes servers to options."""
        mock_servers = [{"name": "test-server"}]

        AgentRuntimeBuilder(repo_path, "agent-mcp", factory).with_mcp(
            servers=mock_servers
        ).build()

        # Check factory received the servers
        assert len(factory.created_options) == 1
        assert factory.created_options[0]["mcp_servers"] == mock_servers

    @pytest.mark.unit
    def test_disallowed_tools_passed_to_options(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """with_disallowed_tools(tools) passes tools to options."""
        tools = ["dangerous_tool", "another_tool"]

        AgentRuntimeBuilder(repo_path, "agent-disallow", factory).with_disallowed_tools(
            tools
        ).build()

        assert len(factory.created_options) == 1
        assert factory.created_options[0]["disallowed_tools"] == tools

    @pytest.mark.unit
    def test_hooks_dict_structure(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """Hooks dict has correct structure for SDK."""
        AgentRuntimeBuilder(repo_path, "agent-struct", factory).build()

        assert len(factory.created_options) == 1
        hooks = factory.created_options[0]["hooks"]

        assert "PreToolUse" in hooks
        assert "Stop" in hooks
        # PostToolUse only if there are post-tool hooks
        # By default there are none without deadlock monitor

    @pytest.mark.unit
    def test_env_defaults_if_not_called(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """build() creates env even if with_env() not called."""
        runtime = AgentRuntimeBuilder(repo_path, "agent-default", factory).build()

        # Should still have env with required vars
        assert "AGENT_ID" in runtime.env
        assert runtime.env["AGENT_ID"] == "agent-default"

"""Unit tests for AgentRuntimeBuilder.

Tests the centralized agent runtime configuration builder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from src.infra.agent_runtime import AgentRuntime, AgentRuntimeBuilder
from src.infra.hooks import (
    LintCache,
    block_dangerous_commands,
    block_mala_disallowed_tools,
)
from tests.fakes import FakeDeadlockMonitor, FakeSDKClientFactory

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from src.core.models import LockEvent
    from src.core.protocols.sdk import McpServerFactory


def _create_mcp_server_factory() -> McpServerFactory:
    """Create a mock MCP server factory for tests.

    Returns a factory that produces a dict with 'mala-locking' key.
    """
    from src.infra.tools.locking_mcp import create_locking_mcp_server

    def _noop(event: LockEvent) -> None:
        pass

    def factory(
        agent_id: str,
        repo_path: Path,
        emit_lock_event: Callable[[LockEvent], object] | None,
    ) -> dict[str, Any]:
        return {
            "mala-locking": create_locking_mcp_server(
                agent_id=agent_id,
                repo_namespace=str(repo_path),
                emit_lock_event=emit_lock_event if emit_lock_event else _noop,
            )
        }

    return factory


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
            .with_mcp(servers={})  # Explicitly disable locking for test
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
            AgentRuntimeBuilder(repo_path, "agent-xyz", factory)
            .with_env()
            .with_mcp(servers={})  # Explicitly disable locking for test
            .build()
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
            .with_mcp(servers={})  # Explicitly disable locking for test
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

        result3 = builder.with_mcp(servers={})  # Explicitly disable locking
        assert result3 is builder

        result4 = builder.with_disallowed_tools()
        assert result4 is builder

        result5 = builder.with_lint_tools({"ruff"})
        assert result5 is builder

    @pytest.mark.unit
    def test_pre_tool_hooks_ordering(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """Pre-tool hooks are in correct order (order matters for security)."""
        runtime = (
            AgentRuntimeBuilder(repo_path, "agent-hooks", factory)
            .with_mcp(servers={})  # Explicitly disable locking for test
            .build()
        )

        # Should have at least: dangerous_commands, disallowed_tools, lock_enforcement,
        # file_cache, lint_cache
        assert len(runtime.pre_tool_hooks) >= 5

        # Verify critical security hooks are at the beginning in correct order
        # block_dangerous_commands must be first (index 0)
        assert runtime.pre_tool_hooks[0] is block_dangerous_commands
        # block_mala_disallowed_tools must be second (index 1)
        assert runtime.pre_tool_hooks[1] is block_mala_disallowed_tools

    @pytest.mark.unit
    def test_stop_hook_included_by_default(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """Stop hook is included by default."""
        runtime = (
            AgentRuntimeBuilder(repo_path, "agent-stop", factory)
            .with_mcp(servers={})  # Explicitly disable locking for test
            .build()
        )

        assert len(runtime.stop_hooks) == 1

    @pytest.mark.unit
    def test_stop_hook_can_be_excluded(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """with_hooks(include_stop_hook=False) excludes stop hook."""
        runtime = (
            AgentRuntimeBuilder(repo_path, "agent-no-stop", factory)
            .with_hooks(include_stop_hook=False)
            .with_mcp(servers={})  # Explicitly disable locking for test
            .build()
        )

        assert len(runtime.stop_hooks) == 0

    @pytest.mark.unit
    def test_deadlock_monitor_adds_hooks(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """with_hooks(deadlock_monitor=...) adds lock event hooks."""
        monitor = FakeDeadlockMonitor()

        runtime = (
            AgentRuntimeBuilder(repo_path, "agent-deadlock", factory)
            .with_hooks(deadlock_monitor=monitor)
            .with_mcp(servers={})  # Explicitly disable locking for test
            .build()
        )

        # Deadlock monitor adds one post-tool hook for MCP locking tools
        # (WAITING events are now emitted by MCP tool handlers directly)
        # Pre-tool hooks: 6 base (no lock_wait hook needed)
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
            .with_mcp(servers={})  # Explicitly disable locking for test
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

        (
            AgentRuntimeBuilder(repo_path, "agent-disallow", factory)
            .with_disallowed_tools(tools)
            .with_mcp(servers={})  # Explicitly disable locking for test
            .build()
        )

        assert len(factory.created_options) == 1
        assert factory.created_options[0]["disallowed_tools"] == tools

    @pytest.mark.unit
    def test_hooks_dict_structure(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """Hooks dict has correct structure for SDK."""
        (
            AgentRuntimeBuilder(repo_path, "agent-struct", factory)
            .with_mcp(servers={})  # Explicitly disable locking for test
            .build()
        )

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
        runtime = (
            AgentRuntimeBuilder(repo_path, "agent-default", factory)
            .with_mcp(servers={})  # Explicitly disable locking for test
            .build()
        )

        # Should still have env with required vars
        assert "AGENT_ID" in runtime.env
        assert runtime.env["AGENT_ID"] == "agent-default"

    @pytest.mark.unit
    def test_locking_mcp_registered_without_deadlock_monitor(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """Locking MCP server is registered even without deadlock monitor."""
        # Build without deadlock monitor (the default)
        AgentRuntimeBuilder(
            repo_path,
            "agent-no-monitor",
            factory,
            mcp_server_factory=_create_mcp_server_factory(),
        ).with_hooks(deadlock_monitor=None).build()

        assert len(factory.created_options) == 1
        mcp_servers = factory.created_options[0]["mcp_servers"]

        # Should have mala-locking server
        assert "mala-locking" in mcp_servers

    @pytest.mark.unit
    def test_locking_mcp_registered_with_deadlock_monitor(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """Locking MCP server is registered with deadlock monitor."""
        monitor = FakeDeadlockMonitor()

        AgentRuntimeBuilder(
            repo_path,
            "agent-with-monitor",
            factory,
            mcp_server_factory=_create_mcp_server_factory(),
        ).with_hooks(deadlock_monitor=monitor).build()

        assert len(factory.created_options) == 1
        mcp_servers = factory.created_options[0]["mcp_servers"]

        # Should have mala-locking server
        assert "mala-locking" in mcp_servers

    @pytest.mark.unit
    def test_setting_sources_logs_info(
        self,
        repo_path: Path,
        factory: FakeSDKClientFactory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """setting_sources logs INFO message with resolved sources."""
        import logging

        with caplog.at_level(logging.INFO):
            AgentRuntimeBuilder(
                repo_path,
                "agent-settings",
                factory,
                setting_sources=["local", "project"],
            ).with_mcp(servers={}).build()

        assert "Claude settings sources: local, project" in caplog.text

    @pytest.mark.unit
    def test_setting_sources_warns_when_local_file_missing(
        self,
        repo_path: Path,
        factory: FakeSDKClientFactory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """setting_sources logs WARN when 'local' in sources but file missing."""
        import logging

        with caplog.at_level(logging.WARNING):
            AgentRuntimeBuilder(
                repo_path,
                "agent-warn",
                factory,
                setting_sources=["local", "project"],
            ).with_mcp(servers={}).build()

        assert (
            "Claude settings file .claude/settings.local.json not found" in caplog.text
        )

    @pytest.mark.unit
    def test_setting_sources_no_warn_when_local_file_exists(
        self,
        repo_path: Path,
        factory: FakeSDKClientFactory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """setting_sources does NOT warn when 'local' in sources and file exists."""
        import logging

        # Create the .claude/settings.local.json file
        (repo_path / ".claude").mkdir(exist_ok=True)
        (repo_path / ".claude" / "settings.local.json").write_text("{}")

        with caplog.at_level(logging.WARNING):
            AgentRuntimeBuilder(
                repo_path,
                "agent-no-warn",
                factory,
                setting_sources=["local", "project"],
            ).with_mcp(servers={}).build()

        # Assert the specific settings warning is not present
        assert not any(
            "Claude settings file .claude/settings.local.json" in r.message
            for r in caplog.records
        )

    @pytest.mark.unit
    def test_setting_sources_passed_to_sdk_adapter(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """setting_sources is passed to sdk_client_factory.create_options()."""
        AgentRuntimeBuilder(
            repo_path,
            "agent-adapter",
            factory,
            setting_sources=["local", "project"],
        ).with_mcp(servers={}).build()

        assert len(factory.created_options) == 1
        assert factory.created_options[0]["setting_sources"] == ["local", "project"]

    @pytest.mark.unit
    def test_setting_sources_none_uses_defaults(
        self,
        repo_path: Path,
        factory: FakeSDKClientFactory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """When setting_sources is None, defaults are logged and local file is checked."""
        import logging

        with caplog.at_level(logging.INFO):
            AgentRuntimeBuilder(
                repo_path,
                "agent-defaults",
                factory,
                setting_sources=None,
            ).with_mcp(servers={}).build()

        # Should log with SDK defaults
        assert "Claude settings sources: local, project" in caplog.text
        # Should warn about missing local file (using defaults includes "local")
        assert (
            "Claude settings file .claude/settings.local.json not found" in caplog.text
        )

    @pytest.mark.unit
    def test_setting_sources_empty_list_no_defaults(
        self,
        repo_path: Path,
        factory: FakeSDKClientFactory,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Empty setting_sources does not fall back to defaults."""
        import logging

        with caplog.at_level(logging.INFO):
            AgentRuntimeBuilder(
                repo_path,
                "agent-empty-sources",
                factory,
                setting_sources=[],
            ).with_mcp(servers={}).build()

        assert "Claude settings sources: (none)" in caplog.text
        assert (
            "Claude settings file .claude/settings.local.json not found"
            not in caplog.text
        )
        assert len(factory.created_options) == 1
        assert factory.created_options[0]["setting_sources"] == []

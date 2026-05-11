"""Unit tests for ClaudeAgentRuntimeBuilder.

Tests the centralized agent runtime configuration builder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from src.infra.agent_runtime import (
    AgentRuntime,
    ClaudeAgentRuntimeBuilder,
    RuntimeComponents,
    build_default_runtime_components,
)
from src.infra.hooks import (
    FileReadCache,
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


class TestClaudeAgentRuntimeBuilder:
    """Tests for ClaudeAgentRuntimeBuilder."""

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
            ClaudeAgentRuntimeBuilder(repo_path, "test-agent-123", factory)
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
            ClaudeAgentRuntimeBuilder(repo_path, "agent-xyz", factory)
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
    def test_agent_timeout_sets_mcp_timeout(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """with_agent_timeout() converts seconds to MCP timeout milliseconds."""
        runtime = (
            ClaudeAgentRuntimeBuilder(repo_path, "agent-xyz", factory)
            .with_agent_timeout(1800)
            .with_env()
            .with_mcp(servers={})
            .build()
        )

        assert runtime.env["MCP_TIMEOUT"] == "1800000"

    @pytest.mark.unit
    def test_agent_timeout_updates_existing_env(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """with_agent_timeout() also applies when with_env() ran first."""
        runtime = (
            ClaudeAgentRuntimeBuilder(repo_path, "agent-xyz", factory)
            .with_env(extra={"MCP_TIMEOUT": "1"})
            .with_agent_timeout(1800)
            .with_mcp(servers={})
            .build()
        )

        assert runtime.env["MCP_TIMEOUT"] == "1800000"
        assert factory.created_options[0]["env"]["MCP_TIMEOUT"] == "1800000"

    @pytest.mark.unit
    def test_env_extra_vars_merged(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """with_env(extra) merges extra variables."""
        runtime = (
            ClaudeAgentRuntimeBuilder(repo_path, "agent-1", factory)
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
        builder = ClaudeAgentRuntimeBuilder(repo_path, "agent-chain", factory)

        result1 = builder.with_hooks()
        assert result1 is builder

        result2 = builder.with_agent_timeout(600)
        assert result2 is builder

        result3 = builder.with_env()
        assert result3 is builder

        result4 = builder.with_mcp(servers={})  # Explicitly disable locking
        assert result4 is builder

        result5 = builder.with_disallowed_tools()
        assert result5 is builder

        result6 = builder.with_lint_tools({"ruff"})
        assert result6 is builder

    @pytest.mark.unit
    def test_pre_tool_hooks_ordering(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """Pre-tool hooks are in correct order (order matters for security)."""
        runtime = (
            ClaudeAgentRuntimeBuilder(repo_path, "agent-hooks", factory)
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
            ClaudeAgentRuntimeBuilder(repo_path, "agent-stop", factory)
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
            ClaudeAgentRuntimeBuilder(repo_path, "agent-no-stop", factory)
            .with_hooks(include_stop_hook=False)
            .with_mcp(servers={})  # Explicitly disable locking for test
            .build()
        )

        assert len(runtime.stop_hooks) == 0

    @pytest.mark.unit
    def test_deadlock_monitor_adds_hooks(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """deadlock_monitor constructor arg adds lock event hooks (plan A6)."""
        monitor = FakeDeadlockMonitor()

        runtime = (
            ClaudeAgentRuntimeBuilder(
                repo_path,
                "agent-deadlock",
                factory,
                deadlock_monitor=monitor,
            )
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
            ClaudeAgentRuntimeBuilder(repo_path, "agent-lint", factory)
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
        mock_servers: dict[str, object] = {"test-server": {"command": "x"}}

        ClaudeAgentRuntimeBuilder(repo_path, "agent-mcp", factory).with_mcp(
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
            ClaudeAgentRuntimeBuilder(repo_path, "agent-disallow", factory)
            .with_disallowed_tools(tools)
            .with_mcp(servers={})  # Explicitly disable locking for test
            .build()
        )

        assert len(factory.created_options) == 1
        assert factory.created_options[0]["disallowed_tools"] == tools

    @pytest.mark.unit
    def test_claude_coder_uses_opus_1m_model(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """Claude coder sessions request the 1M context Opus model."""
        ClaudeAgentRuntimeBuilder(repo_path, "agent-model", factory).with_mcp(
            servers={}
        ).build()

        assert len(factory.created_options) == 1
        assert factory.created_options[0]["model"] == "opus[1m]"

    @pytest.mark.unit
    @pytest.mark.parametrize("effort", ["low", "medium", "high", "xhigh", "max"])
    def test_effort_forwarded_to_create_options(
        self,
        repo_path: Path,
        factory: FakeSDKClientFactory,
        effort: str,
    ) -> None:
        """``effort`` flows from the builder into ``create_options(effort=...)``.

        AC#1: Claude coder sessions pass the configured effort through to
        ``ClaudeAgentOptions.effort``. The model field stays ``opus[1m]`` so
        unrelated reviewer / epic model configuration is not affected.
        """
        ClaudeAgentRuntimeBuilder(
            repo_path, "agent-effort", factory, effort=effort
        ).with_mcp(servers={}).build()

        assert len(factory.created_options) == 1
        opts = factory.created_options[0]
        assert opts["effort"] == effort
        # Coder model is still the configured 1M-context Opus.
        assert opts["model"] == "opus[1m]"

    @pytest.mark.unit
    def test_effort_default_is_none(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """Without an explicit effort the SDK default is preserved (``None``)."""
        ClaudeAgentRuntimeBuilder(repo_path, "agent-noeffort", factory).with_mcp(
            servers={}
        ).build()
        assert factory.created_options[0]["effort"] is None

    @pytest.mark.unit
    def test_hooks_dict_structure(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """Hooks dict has correct structure for SDK."""
        (
            ClaudeAgentRuntimeBuilder(repo_path, "agent-struct", factory)
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
            ClaudeAgentRuntimeBuilder(repo_path, "agent-default", factory)
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
        ClaudeAgentRuntimeBuilder(
            repo_path,
            "agent-no-monitor",
            factory,
            mcp_server_factory=_create_mcp_server_factory(),
        ).build()

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

        ClaudeAgentRuntimeBuilder(
            repo_path,
            "agent-with-monitor",
            factory,
            mcp_server_factory=_create_mcp_server_factory(),
            deadlock_monitor=monitor,
        ).build()

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
            ClaudeAgentRuntimeBuilder(
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
            ClaudeAgentRuntimeBuilder(
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
            ClaudeAgentRuntimeBuilder(
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
        ClaudeAgentRuntimeBuilder(
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
            ClaudeAgentRuntimeBuilder(
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
            ClaudeAgentRuntimeBuilder(
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


class TestBuildDefaultRuntimeComponents:
    """Tests for :func:`build_default_runtime_components` (Finding #13)."""

    @pytest.fixture
    def repo_path(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.mark.unit
    def test_returns_runtime_components_with_all_fields(self, repo_path: Path) -> None:
        """Helper returns a populated :class:`RuntimeComponents`."""
        components = build_default_runtime_components(repo_path, "agent-x")

        assert isinstance(components, RuntimeComponents)
        assert isinstance(components.file_read_cache, FileReadCache)
        assert isinstance(components.lint_cache, LintCache)
        assert components.precompact_hook is not None
        # Default composition includes pre-tool hooks and a stop hook
        assert len(components.pre_tool_hooks) >= 5
        assert len(components.stop_hooks) == 1
        # No deadlock monitor: no post-tool hooks
        assert components.post_tool_hooks == []

    @pytest.mark.unit
    def test_default_pre_tool_hook_ordering_matches_builder(
        self, repo_path: Path
    ) -> None:
        """Default pre-tool hook order preserves the security-critical prefix."""
        components = build_default_runtime_components(repo_path, "agent-order")

        assert components.pre_tool_hooks[0] is block_dangerous_commands
        assert components.pre_tool_hooks[1] is block_mala_disallowed_tools

    @pytest.mark.unit
    def test_excluded_stop_hook(self, repo_path: Path) -> None:
        """``include_stop_hook=False`` omits the stop hook."""
        components = build_default_runtime_components(
            repo_path, "agent-no-stop", include_stop_hook=False
        )
        assert components.stop_hooks == []

    @pytest.mark.unit
    def test_deadlock_monitor_adds_post_tool_hook(self, repo_path: Path) -> None:
        """A deadlock monitor wires a PostToolUse hook."""
        monitor = FakeDeadlockMonitor()
        components = build_default_runtime_components(
            repo_path, "agent-monitor", deadlock_monitor=monitor
        )
        assert len(components.post_tool_hooks) == 1

    @pytest.mark.unit
    def test_lint_tools_threaded_into_cache(self, repo_path: Path) -> None:
        """Lint tools are forwarded to the :class:`LintCache`."""
        components = build_default_runtime_components(
            repo_path, "agent-lint", lint_tools={"ruff", "mypy"}
        )
        assert components.lint_cache.lint_tools == {"ruff", "mypy"}


class TestRuntimeComponentsInjection:
    """Tests for injecting pre-built :class:`RuntimeComponents` (Finding #13)."""

    @pytest.fixture
    def repo_path(self, tmp_path: Path) -> Path:
        return tmp_path

    @pytest.fixture
    def factory(self) -> FakeSDKClientFactory:
        return FakeSDKClientFactory()

    @pytest.mark.unit
    def test_injected_components_used_by_build(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """When ``runtime_components`` is injected, ``build()`` uses them directly."""
        components = build_default_runtime_components(
            repo_path, "agent-inject", include_stop_hook=False
        )

        runtime = (
            ClaudeAgentRuntimeBuilder(
                repo_path,
                "agent-inject",
                factory,
                runtime_components=components,
            )
            .with_mcp(servers={})
            .build()
        )

        # Stop hook excluded because injected components had none.
        assert runtime.stop_hooks == []
        # The runtime exposes the SAME hook list object that was injected,
        # confirming no re-instantiation happened inside ``build()``.
        assert runtime.pre_tool_hooks is components.pre_tool_hooks
        assert runtime.lint_cache is components.lint_cache
        assert runtime.file_read_cache is components.file_read_cache

    @pytest.mark.unit
    def test_default_composition_preserved_without_injection(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """No injection -> ``build()`` falls back to the default composition.

        Acceptance criterion: default runtime contains the same hooks as before.
        """
        runtime = (
            ClaudeAgentRuntimeBuilder(repo_path, "agent-default", factory)
            .with_mcp(servers={})
            .build()
        )

        # Same security-critical prefix as the helper produces.
        assert runtime.pre_tool_hooks[0] is block_dangerous_commands
        assert runtime.pre_tool_hooks[1] is block_mala_disallowed_tools
        # Same stop-hook count as the default helper composition.
        assert len(runtime.stop_hooks) == 1

    @pytest.mark.unit
    def test_with_hooks_invalidates_injected_components(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """``with_hooks(...)`` after injection rebuilds with the new flags."""
        components = build_default_runtime_components(repo_path, "agent-rebuild")

        runtime = (
            ClaudeAgentRuntimeBuilder(
                repo_path,
                "agent-rebuild",
                factory,
                runtime_components=components,
            )
            .with_hooks(include_stop_hook=False)
            .with_mcp(servers={})
            .build()
        )

        # ``with_hooks`` invalidated the injection; rebuild applied the flag.
        assert runtime.stop_hooks == []
        # The runtime no longer references the injected list (it was rebuilt).
        assert runtime.pre_tool_hooks is not components.pre_tool_hooks

    @pytest.mark.unit
    def test_with_lint_tools_invalidates_injected_components(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """``with_lint_tools(...)`` after injection rebuilds the lint cache."""
        components = build_default_runtime_components(repo_path, "agent-lint-inject")

        runtime = (
            ClaudeAgentRuntimeBuilder(
                repo_path,
                "agent-lint-inject",
                factory,
                runtime_components=components,
            )
            .with_lint_tools({"ruff"})
            .with_mcp(servers={})
            .build()
        )

        assert runtime.lint_cache.lint_tools == {"ruff"}
        assert runtime.lint_cache is not components.lint_cache

    @pytest.mark.unit
    def test_noop_with_lint_tools_preserves_injection(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """``with_lint_tools(None)`` does not discard provider-injected components.

        Regression for the AgentSessionRunner path: it unconditionally calls
        ``builder.with_lint_tools(self.config.lint_tools)`` (commonly
        ``None``); the injected default composition from the provider must
        survive that no-op so ``build()`` doesn't fall back to rebuilding
        the caches/hooks via the helper.
        """
        components = build_default_runtime_components(repo_path, "agent-noop-lint")

        runtime = (
            ClaudeAgentRuntimeBuilder(
                repo_path,
                "agent-noop-lint",
                factory,
                runtime_components=components,
            )
            .with_lint_tools(None)
            .with_mcp(servers={})
            .build()
        )

        # Injection survived: same lint_cache / hook list objects.
        assert runtime.lint_cache is components.lint_cache
        assert runtime.pre_tool_hooks is components.pre_tool_hooks

    @pytest.mark.unit
    def test_build_does_not_invoke_helper(
        self,
        repo_path: Path,
        factory: FakeSDKClientFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``build()`` body never calls ``build_default_runtime_components``.

        Regression for Finding #13's acceptance criterion: "No hook/cache
        instantiation inside the builder." The helper is invoked eagerly
        in ``__init__`` / ``with_hooks`` / ``with_lint_tools`` (outside
        the build body), so calling ``build()`` after construction +
        config tweaks must not re-enter the helper.
        """
        from unittest.mock import patch

        import src.infra.agent_runtime as agent_runtime_mod

        components = build_default_runtime_components(repo_path, "agent-build-clean")
        builder = ClaudeAgentRuntimeBuilder(
            repo_path,
            "agent-build-clean",
            factory,
            runtime_components=components,
        )
        # Apply no-op fluent calls then build under a patched helper. If
        # ``build()`` invokes the helper, the spy would record a call.
        with patch.object(
            agent_runtime_mod,
            "build_default_runtime_components",
            wraps=agent_runtime_mod.build_default_runtime_components,
        ) as spy:
            runtime = (
                builder.with_lint_tools(None)
                .with_hooks(include_stop_hook=True)
                .with_env()
                .with_mcp(servers={})
                .build()
            )

        assert spy.call_count == 0, (
            "build() body must not call build_default_runtime_components; "
            "components must be pre-populated before build()."
        )
        # The runtime exposes the injected components (no re-instantiation).
        assert runtime.pre_tool_hooks is components.pre_tool_hooks
        assert runtime.lint_cache is components.lint_cache

    @pytest.mark.unit
    def test_changed_lint_tools_rebuilds_outside_build(
        self,
        repo_path: Path,
        factory: FakeSDKClientFactory,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A real lint-tool change rebuilds in ``with_lint_tools``, not ``build()``.

        Regression covering the AgentSessionRunner path where
        ``self.config.lint_tools`` is non-empty: the rebuild happens
        eagerly inside the ``with_lint_tools`` method body (which calls
        the external helper), and ``build()`` then only formats the
        already-built components into the SDK struct.
        """
        from unittest.mock import patch

        import src.infra.agent_runtime as agent_runtime_mod

        components = build_default_runtime_components(repo_path, "agent-lint-change")
        builder = ClaudeAgentRuntimeBuilder(
            repo_path,
            "agent-lint-change",
            factory,
            runtime_components=components,
        )
        with patch.object(
            agent_runtime_mod,
            "build_default_runtime_components",
            wraps=agent_runtime_mod.build_default_runtime_components,
        ) as spy:
            builder.with_lint_tools({"ruff"})
            # The helper was called by with_lint_tools (rebuild).
            assert spy.call_count == 1
            builder.with_mcp(servers={}).build()
            # build() did not invoke the helper.
            assert spy.call_count == 1

    @pytest.mark.unit
    def test_noop_with_hooks_preserves_injection(
        self, repo_path: Path, factory: FakeSDKClientFactory
    ) -> None:
        """``with_hooks(...)`` with unchanged flags preserves injection.

        Fixer service calls ``with_hooks(include_stop_hook=True, ...)``;
        ``include_stop_hook=True`` matches the builder default, so that
        argument alone must not invalidate the injection.
        """
        components = build_default_runtime_components(repo_path, "agent-noop-hooks")

        runtime = (
            ClaudeAgentRuntimeBuilder(
                repo_path,
                "agent-noop-hooks",
                factory,
                runtime_components=components,
            )
            .with_hooks(include_stop_hook=True)  # default value, no-op
            .with_mcp(servers={})
            .build()
        )

        assert runtime.pre_tool_hooks is components.pre_tool_hooks
        assert runtime.stop_hooks is components.stop_hooks

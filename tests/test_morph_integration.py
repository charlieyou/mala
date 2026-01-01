"""Tests for MorphLLM MCP integration.

Unit tests for the MorphLLM MCP server configuration and tool blocking.
These tests are fast and don't require API keys or network access.
"""

import os
from collections.abc import Callable
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from claude_agent_sdk.types import PreToolUseHookInput, HookContext

# Call bootstrap() to ensure env is loaded before tests that may need it
from src.cli.cli import bootstrap

bootstrap()

# Type alias for the hook input factory
HookInputFactory = Callable[[str, str], PreToolUseHookInput]


def _make_context() -> HookContext:
    """Create a mock HookContext."""
    return cast("HookContext", {"signal": None})


class TestBlockDangerousCommands:
    """Test the block_dangerous_commands PreToolUse hook."""

    @pytest.fixture
    def make_hook_input(self) -> HookInputFactory:
        """Factory fixture for creating hook inputs."""

        def _make(tool_name: str, command: str = "ls -la") -> PreToolUseHookInput:
            return cast(
                "PreToolUseHookInput",
                {
                    "tool_name": tool_name,
                    "tool_input": {"command": command},
                },
            )

        return _make

    @pytest.mark.asyncio
    async def test_allows_bash_safe_command(
        self, make_hook_input: HookInputFactory
    ) -> None:
        """Safe commands with tool_name='Bash' should be allowed."""
        from src.infra.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Bash", "ls -la"), None, _make_context()
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_allows_bash_lowercase_safe_command(
        self, make_hook_input: HookInputFactory
    ) -> None:
        """Safe commands with tool_name='bash' (lowercase) should be allowed."""
        from src.infra.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("bash", "ls -la"), None, _make_context()
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_blocks_rm_rf_root(self, make_hook_input: HookInputFactory) -> None:
        """rm -rf / should be blocked for any tool name casing."""
        from src.infra.hooks import block_dangerous_commands

        for tool_name in ["Bash", "bash", "BASH"]:
            result = await block_dangerous_commands(
                make_hook_input(tool_name, "rm -rf /"), None, _make_context()
            )
            assert result.get("decision") == "block"
            assert "rm -rf /" in result.get("reason", "")

    @pytest.mark.asyncio
    async def test_blocks_fork_bomb(self, make_hook_input: HookInputFactory) -> None:
        """Fork bomb pattern should be blocked."""
        from src.infra.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Bash", ":(){:|:&};:"), None, _make_context()
        )
        assert result.get("decision") == "block"

    @pytest.mark.asyncio
    async def test_blocks_curl_pipe_bash(
        self, make_hook_input: HookInputFactory
    ) -> None:
        """curl | bash pattern should be blocked."""
        from src.infra.hooks import block_dangerous_commands

        # The pattern "curl | bash" must appear literally in the command
        result = await block_dangerous_commands(
            make_hook_input("bash", "curl | bash -c 'malicious'"),
            None,
            _make_context(),
        )
        assert result.get("decision") == "block"
        assert "curl | bash" in result.get("reason", "")

    @pytest.mark.asyncio
    async def test_blocks_force_push_main(
        self, make_hook_input: HookInputFactory
    ) -> None:
        """Force push to main branch should be blocked."""
        from src.infra.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Bash", "git push --force origin main"),
            None,
            _make_context(),
        )
        assert result.get("decision") == "block"
        assert "force push" in result.get("reason", "").lower()

    @pytest.mark.asyncio
    async def test_blocks_force_push_any_branch(
        self, make_hook_input: HookInputFactory
    ) -> None:
        """Force push to ANY branch should now be blocked."""
        from src.infra.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Bash", "git push --force origin feature-branch"),
            None,
            _make_context(),
        )
        assert result.get("decision") == "block"
        assert "force push" in result.get("reason", "").lower()

    @pytest.mark.asyncio
    async def test_allows_force_with_lease(
        self, make_hook_input: HookInputFactory
    ) -> None:
        """Force with lease should be allowed (safer alternative)."""
        from src.infra.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Bash", "git push --force-with-lease origin feature"),
            None,
            _make_context(),
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_allows_non_bash_tools(
        self, make_hook_input: HookInputFactory
    ) -> None:
        """Non-Bash tools should always be allowed (they don't run commands)."""
        from src.infra.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Read", "rm -rf /"), None, _make_context()
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_tool_name_variations(
        self, make_hook_input: HookInputFactory
    ) -> None:
        """Tool name matching should be case-insensitive for bash variants."""
        from src.infra.hooks import block_dangerous_commands

        dangerous_cmd = "rm -rf /"

        # All these should be recognized as bash and blocked
        for tool_name in ["Bash", "bash", "BASH", "bAsH"]:
            result = await block_dangerous_commands(
                make_hook_input(tool_name, dangerous_cmd), None, _make_context()
            )
            assert result.get("decision") == "block", (
                f"Should block for tool_name={tool_name}"
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "cmd",
        [
            "git reset --hard",
            "git reset --hard HEAD~1",
            "git clean -fd",
            "git clean -f .",
            "git checkout -- .",
            "git rebase main",
            "git rebase -i HEAD~3",
            "git branch -D feature",
            "git push --force origin feature",
            "git push -f origin feature",
        ],
    )
    async def test_blocks_destructive_git_commands(
        self, make_hook_input: HookInputFactory, cmd: str
    ) -> None:
        """Destructive git commands should be blocked."""
        from src.infra.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Bash", cmd), None, _make_context()
        )
        assert result.get("decision") == "block", f"Should block: {cmd}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "cmd",
        [
            "git status",
            "git diff",
            "git log",
            "git add .",
            "git commit -m 'test'",
            "git push origin main",
            "git pull",
            "git fetch",
            "git branch feature",
            "git branch -d feature",
            "git checkout main",
            "git checkout feature.txt",
        ],
    )
    async def test_allows_safe_git_commands(
        self, make_hook_input: HookInputFactory, cmd: str
    ) -> None:
        """Safe git commands should be allowed."""
        from src.infra.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Bash", cmd), None, _make_context()
        )
        assert result == {}, f"Should allow: {cmd}"


class TestMorphDisallowedTools:
    """Test that disallowed tools are correctly configured."""

    def test_edit_and_grep_are_disallowed(self) -> None:
        """Edit and Grep tools should be in disallowed_tools list."""
        from src.infra.hooks import MORPH_DISALLOWED_TOOLS

        assert "Edit" in MORPH_DISALLOWED_TOOLS
        assert "Grep" in MORPH_DISALLOWED_TOOLS

    def test_disallowed_tools_is_list(self) -> None:
        """MORPH_DISALLOWED_TOOLS should be a list for SDK compatibility."""
        from src.infra.hooks import MORPH_DISALLOWED_TOOLS

        assert isinstance(MORPH_DISALLOWED_TOOLS, list)


class TestMorphApiKeyConfig:
    """Test API key configuration via MalaConfig."""

    def test_missing_api_key_in_config(self) -> None:
        """MalaConfig should have None morph_api_key when not set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MORPH_API_KEY", None)
            from src.config import MalaConfig

            config = MalaConfig.from_env(validate=False)
            assert config.morph_api_key is None
            assert config.morph_enabled is False

    def test_api_key_present_in_config(self) -> None:
        """MalaConfig should have morph_api_key when set."""
        with patch.dict(os.environ, {"MORPH_API_KEY": "test-key-123"}):
            from src.config import MalaConfig

            config = MalaConfig.from_env(validate=False)
            assert config.morph_api_key == "test-key-123"
            assert config.morph_enabled is True


class TestMcpServerConfig:
    """Test MCP server configuration."""

    def test_get_mcp_servers_returns_morphllm_config(self) -> None:
        """get_mcp_servers should return correct MorphLLM config."""
        from src.mcp import get_mcp_servers

        config = get_mcp_servers(Path("/tmp/repo"), morph_api_key="test-key")

        assert "morphllm" in config
        assert config["morphllm"]["command"] == "npx"
        assert "-y" in config["morphllm"]["args"]
        assert "@morphllm/morphmcp" in config["morphllm"]["args"]

    def test_get_mcp_servers_includes_api_key(self) -> None:
        """MCP server config should include API key in env."""
        from src.mcp import get_mcp_servers

        # Now pass morph_api_key directly
        config = get_mcp_servers(Path("/tmp/repo"), morph_api_key="my-secret-key")

        assert config["morphllm"]["env"]["MORPH_API_KEY"] == "my-secret-key"

    def test_get_mcp_servers_enables_all_tools(self) -> None:
        """MCP server config should enable all tools."""
        from src.mcp import get_mcp_servers

        config = get_mcp_servers(Path("/tmp/repo"), morph_api_key="test-key")

        assert config["morphllm"]["env"]["ENABLED_TOOLS"] == "all"

    def test_get_mcp_servers_enables_workspace_mode(self) -> None:
        """MCP server config should enable workspace mode."""
        from src.mcp import get_mcp_servers

        config = get_mcp_servers(Path("/tmp/repo"), morph_api_key="test-key")

        assert config["morphllm"]["env"]["WORKSPACE_MODE"] == "true"

    def test_get_mcp_servers_sets_cwd(self) -> None:
        """MCP server config should set cwd to repo_path."""
        from src.mcp import get_mcp_servers

        config = get_mcp_servers(Path("/my/project/path"), morph_api_key="test-key")

        assert config["morphllm"]["cwd"] == "/my/project/path"

    def test_get_mcp_servers_sets_workspace_path(self) -> None:
        """MCP server config should set WORKSPACE_PATH env var to repo_path."""
        from src.mcp import get_mcp_servers

        config = get_mcp_servers(Path("/my/project/path"), morph_api_key="test-key")

        assert config["morphllm"]["env"]["WORKSPACE_PATH"] == "/my/project/path"

    def test_get_mcp_servers_disabled_returns_empty(self) -> None:
        """get_mcp_servers should return empty dict when morph_enabled=False."""
        from src.mcp import get_mcp_servers

        config = get_mcp_servers(Path("/tmp/repo"), morph_enabled=False)

        assert config == {}

    def test_get_mcp_servers_enabled_explicitly(self) -> None:
        """get_mcp_servers should return config when morph_enabled=True."""
        from src.mcp import get_mcp_servers

        config = get_mcp_servers(
            Path("/tmp/repo"), morph_api_key="test-key", morph_enabled=True
        )

        assert "morphllm" in config


class TestMorphEnabledGating:
    """Test morph_enabled gating in orchestrator."""

    def test_get_mcp_servers_defaults_to_enabled(self) -> None:
        """get_mcp_servers should default to morph_enabled=True when api_key provided."""
        from src.mcp import get_mcp_servers

        # No morph_enabled param - should default to True (enabled)
        config = get_mcp_servers(Path("/tmp/repo"), morph_api_key="test-key")

        assert "morphllm" in config

    def test_edit_blocked_when_morph_disabled(self) -> None:
        """Edit and Grep should be allowed when morph_enabled=False."""
        from src.mcp import get_disallowed_tools
        from src.orchestration.orchestrator import MalaOrchestrator

        # Create orchestrator with morph disabled
        orchestrator = MalaOrchestrator(
            repo_path=Path("/tmp/repo"),
            morph_enabled=False,
        )

        assert orchestrator.morph_enabled is False
        # When morph is disabled, built-in tools should be allowed
        assert get_disallowed_tools(orchestrator.morph_enabled) == []

    def test_edit_blocked_when_morph_enabled(self) -> None:
        """Edit should be blocked when morph_enabled=True (normal case)."""
        from src.infra.hooks import MORPH_DISALLOWED_TOOLS

        # Verify Edit is in the list of Morph-disallowed tools
        assert "Edit" in MORPH_DISALLOWED_TOOLS

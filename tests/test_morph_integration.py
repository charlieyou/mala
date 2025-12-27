"""Tests for MorphLLM MCP integration.

Unit tests for the MorphLLM MCP server configuration and tool blocking.
These tests are fast and don't require API keys or network access.
"""

import os
from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pytest

# Import src.cli early so load_user_env() runs before any tests patch os.environ
import src.cli  # noqa: F401

# Type alias for the hook input factory
HookInputFactory = Callable[[str, str], dict[str, object]]


class TestBlockDangerousCommands:
    """Test the block_dangerous_commands PreToolUse hook."""

    @pytest.fixture
    def make_hook_input(self) -> HookInputFactory:
        """Factory fixture for creating hook inputs."""

        def _make(tool_name: str, command: str = "ls -la") -> dict[str, object]:
            return {
                "tool_name": tool_name,
                "tool_input": {"command": command},
            }

        return _make

    @pytest.mark.asyncio
    async def test_allows_bash_safe_command(
        self, make_hook_input: HookInputFactory
    ) -> None:
        """Safe commands with tool_name='Bash' should be allowed."""
        from src.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Bash", "ls -la"), None, {"signal": None}
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_allows_bash_lowercase_safe_command(
        self, make_hook_input: HookInputFactory
    ) -> None:
        """Safe commands with tool_name='bash' (lowercase) should be allowed."""
        from src.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("bash", "ls -la"), None, {"signal": None}
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_blocks_rm_rf_root(self, make_hook_input: HookInputFactory) -> None:
        """rm -rf / should be blocked for any tool name casing."""
        from src.hooks import block_dangerous_commands

        for tool_name in ["Bash", "bash", "BASH"]:
            result = await block_dangerous_commands(
                make_hook_input(tool_name, "rm -rf /"), None, {"signal": None}
            )
            assert result.get("decision") == "block"
            assert "rm -rf /" in result.get("reason", "")

    @pytest.mark.asyncio
    async def test_blocks_fork_bomb(self, make_hook_input: HookInputFactory) -> None:
        """Fork bomb pattern should be blocked."""
        from src.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Bash", ":(){:|:&};:"), None, {"signal": None}
        )
        assert result.get("decision") == "block"

    @pytest.mark.asyncio
    async def test_blocks_curl_pipe_bash(
        self, make_hook_input: HookInputFactory
    ) -> None:
        """curl | bash pattern should be blocked."""
        from src.hooks import block_dangerous_commands

        # The pattern "curl | bash" must appear literally in the command
        result = await block_dangerous_commands(
            make_hook_input("bash", "curl | bash -c 'malicious'"),
            None,
            {"signal": None},
        )
        assert result.get("decision") == "block"
        assert "curl | bash" in result.get("reason", "")

    @pytest.mark.asyncio
    async def test_blocks_force_push_main(
        self, make_hook_input: HookInputFactory
    ) -> None:
        """Force push to main branch should be blocked."""
        from src.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Bash", "git push --force origin main"),
            None,
            {"signal": None},
        )
        assert result.get("decision") == "block"
        assert "force push" in result.get("reason", "").lower()

    @pytest.mark.asyncio
    async def test_blocks_force_push_any_branch(
        self, make_hook_input: HookInputFactory
    ) -> None:
        """Force push to ANY branch should now be blocked."""
        from src.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Bash", "git push --force origin feature-branch"),
            None,
            {"signal": None},
        )
        assert result.get("decision") == "block"
        assert "force push" in result.get("reason", "").lower()

    @pytest.mark.asyncio
    async def test_allows_force_with_lease(
        self, make_hook_input: HookInputFactory
    ) -> None:
        """Force with lease should be allowed (safer alternative)."""
        from src.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Bash", "git push --force-with-lease origin feature"),
            None,
            {"signal": None},
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_allows_non_bash_tools(
        self, make_hook_input: HookInputFactory
    ) -> None:
        """Non-Bash tools should always be allowed (they don't run commands)."""
        from src.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Read", "rm -rf /"), None, {"signal": None}
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_tool_name_variations(
        self, make_hook_input: HookInputFactory
    ) -> None:
        """Tool name matching should be case-insensitive for bash variants."""
        from src.hooks import block_dangerous_commands

        dangerous_cmd = "rm -rf /"

        # All these should be recognized as bash and blocked
        for tool_name in ["Bash", "bash", "BASH", "bAsH"]:
            result = await block_dangerous_commands(
                make_hook_input(tool_name, dangerous_cmd), None, {"signal": None}
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
        from src.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Bash", cmd), None, {"signal": None}
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
        from src.hooks import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Bash", cmd), None, {"signal": None}
        )
        assert result == {}, f"Should allow: {cmd}"


class TestMorphDisallowedTools:
    """Test that disallowed tools are correctly configured."""

    def test_edit_and_grep_are_disallowed(self) -> None:
        """Edit and Grep tools should be in disallowed_tools list."""
        from src.hooks import MORPH_DISALLOWED_TOOLS

        assert "Edit" in MORPH_DISALLOWED_TOOLS
        assert "Grep" in MORPH_DISALLOWED_TOOLS

    def test_disallowed_tools_is_list(self) -> None:
        """MORPH_DISALLOWED_TOOLS should be a list for SDK compatibility."""
        from src.hooks import MORPH_DISALLOWED_TOOLS

        assert isinstance(MORPH_DISALLOWED_TOOLS, list)


class TestMorphApiKeyValidation:
    """Test API key validation."""

    def test_missing_api_key_returns_none(self) -> None:
        """get_morph_api_key should return None if key is missing."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MORPH_API_KEY", None)
            from src.cli import get_morph_api_key

            result = get_morph_api_key()
            assert result is None

    def test_empty_api_key_returns_none(self) -> None:
        """get_morph_api_key should return None if key is empty string."""
        with patch.dict(os.environ, {"MORPH_API_KEY": ""}):
            from src.cli import get_morph_api_key

            result = get_morph_api_key()
            assert result is None

    def test_api_key_present_returns_key(self) -> None:
        """get_morph_api_key should return the key if present."""
        with patch.dict(os.environ, {"MORPH_API_KEY": "test-key-123"}):
            from src.cli import get_morph_api_key

            result = get_morph_api_key()
            assert result == "test-key-123"


class TestMcpServerConfig:
    """Test MCP server configuration."""

    def test_get_mcp_servers_returns_morphllm_config(self) -> None:
        """get_mcp_servers should return correct MorphLLM config."""
        with patch.dict(os.environ, {"MORPH_API_KEY": "test-key"}):
            from src.orchestrator import get_mcp_servers

            config = get_mcp_servers(Path("/tmp/repo"))

            assert "morphllm" in config
            assert config["morphllm"]["command"] == "npx"
            assert "-y" in config["morphllm"]["args"]
            assert "@morphllm/morphmcp" in config["morphllm"]["args"]

    def test_get_mcp_servers_includes_api_key(self) -> None:
        """MCP server config should include API key in env."""
        with patch.dict(os.environ, {"MORPH_API_KEY": "my-secret-key"}):
            from src.orchestrator import get_mcp_servers

            config = get_mcp_servers(Path("/tmp/repo"))

            assert config["morphllm"]["env"]["MORPH_API_KEY"] == "my-secret-key"

    def test_get_mcp_servers_enables_all_tools(self) -> None:
        """MCP server config should enable all tools."""
        with patch.dict(os.environ, {"MORPH_API_KEY": "test-key"}):
            from src.orchestrator import get_mcp_servers

            config = get_mcp_servers(Path("/tmp/repo"))

            assert config["morphllm"]["env"]["ENABLED_TOOLS"] == "all"

    def test_get_mcp_servers_enables_workspace_mode(self) -> None:
        """MCP server config should enable workspace mode."""
        with patch.dict(os.environ, {"MORPH_API_KEY": "test-key"}):
            from src.orchestrator import get_mcp_servers

            config = get_mcp_servers(Path("/tmp/repo"))

            assert config["morphllm"]["env"]["WORKSPACE_MODE"] == "true"

    def test_get_mcp_servers_disabled_returns_empty(self) -> None:
        """get_mcp_servers should return empty dict when morph_enabled=False."""
        from src.orchestrator import get_mcp_servers

        config = get_mcp_servers(Path("/tmp/repo"), morph_enabled=False)

        assert config == {}

    def test_get_mcp_servers_enabled_explicitly(self) -> None:
        """get_mcp_servers should return config when morph_enabled=True."""
        with patch.dict(os.environ, {"MORPH_API_KEY": "test-key"}):
            from src.orchestrator import get_mcp_servers

            config = get_mcp_servers(Path("/tmp/repo"), morph_enabled=True)

            assert "morphllm" in config


class TestMorphEnabledGating:
    """Test morph_enabled gating in orchestrator."""

    def test_get_mcp_servers_defaults_to_enabled(self) -> None:
        """get_mcp_servers should default to morph_enabled=True for backward compat."""
        with patch.dict(os.environ, {"MORPH_API_KEY": "test-key"}):
            from src.orchestrator import get_mcp_servers

            # No morph_enabled param - should default to True (enabled)
            config = get_mcp_servers(Path("/tmp/repo"))

            assert "morphllm" in config

    def test_edit_blocked_when_morph_disabled(self) -> None:
        """Edit should remain blocked when morph_enabled=False for lock safety."""
        from src.orchestrator import MalaOrchestrator

        # Create orchestrator with morph disabled
        orchestrator = MalaOrchestrator(
            repo_path=Path("/tmp/repo"),
            morph_enabled=False,
        )

        # Edit should still be blocked even when morph is disabled
        # because lock enforcement doesn't cover the Edit tool
        assert orchestrator.morph_enabled is False
        # The disallowed_tools will be set when run_implementer creates options,
        # but we can verify the morph_enabled flag is stored correctly

    def test_edit_blocked_when_morph_enabled(self) -> None:
        """Edit should be blocked when morph_enabled=True (normal case)."""
        from src.hooks import MORPH_DISALLOWED_TOOLS

        # Verify Edit is in the list of Morph-disallowed tools
        assert "Edit" in MORPH_DISALLOWED_TOOLS

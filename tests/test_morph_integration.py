"""Tests for MorphLLM MCP integration.

Unit tests for the MorphLLM MCP server configuration and tool blocking.
These tests are fast and don't require API keys or network access.
"""

import os
import pytest
from unittest.mock import patch
from pathlib import Path


class TestBlockDangerousCommands:
    """Test the block_dangerous_commands PreToolUse hook."""

    @pytest.fixture
    def make_hook_input(self):
        """Factory fixture for creating hook inputs."""

        def _make(tool_name: str, command: str = "ls -la"):
            return {
                "tool_name": tool_name,
                "tool_input": {"command": command},
            }

        return _make

    @pytest.mark.asyncio
    async def test_allows_bash_safe_command(self, make_hook_input):
        """Safe commands with tool_name='Bash' should be allowed."""
        from src.cli import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Bash", "ls -la"), None, {}
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_allows_bash_lowercase_safe_command(self, make_hook_input):
        """Safe commands with tool_name='bash' (lowercase) should be allowed."""
        from src.cli import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("bash", "ls -la"), None, {}
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_blocks_rm_rf_root(self, make_hook_input):
        """rm -rf / should be blocked for any tool name casing."""
        from src.cli import block_dangerous_commands

        for tool_name in ["Bash", "bash", "BASH"]:
            result = await block_dangerous_commands(
                make_hook_input(tool_name, "rm -rf /"), None, {}
            )
            assert result.get("decision") == "block"
            assert "rm -rf /" in result.get("reason", "")

    @pytest.mark.asyncio
    async def test_blocks_fork_bomb(self, make_hook_input):
        """Fork bomb pattern should be blocked."""
        from src.cli import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Bash", ":(){:|:&};:"), None, {}
        )
        assert result.get("decision") == "block"

    @pytest.mark.asyncio
    async def test_blocks_curl_pipe_bash(self, make_hook_input):
        """curl | bash pattern should be blocked."""
        from src.cli import block_dangerous_commands

        # The pattern "curl | bash" must appear literally in the command
        result = await block_dangerous_commands(
            make_hook_input("bash", "curl | bash -c 'malicious'"), None, {}
        )
        assert result.get("decision") == "block"
        assert "curl | bash" in result.get("reason", "")

    @pytest.mark.asyncio
    async def test_blocks_force_push_main(self, make_hook_input):
        """Force push to main branch should be blocked."""
        from src.cli import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Bash", "git push --force origin main"), None, {}
        )
        assert result.get("decision") == "block"
        assert "force push" in result.get("reason", "").lower()

    @pytest.mark.asyncio
    async def test_allows_force_push_feature_branch(self, make_hook_input):
        """Force push to feature branch should be allowed."""
        from src.cli import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Bash", "git push --force origin feature-branch"), None, {}
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_allows_non_bash_tools(self, make_hook_input):
        """Non-Bash tools should always be allowed (they don't run commands)."""
        from src.cli import block_dangerous_commands

        result = await block_dangerous_commands(
            make_hook_input("Read", "rm -rf /"), None, {}
        )
        assert result == {}

    @pytest.mark.asyncio
    async def test_tool_name_variations(self, make_hook_input):
        """Tool name matching should be case-insensitive for bash variants."""
        from src.cli import block_dangerous_commands

        dangerous_cmd = "rm -rf /"

        # All these should be recognized as bash and blocked
        for tool_name in ["Bash", "bash", "BASH", "bAsH"]:
            result = await block_dangerous_commands(
                make_hook_input(tool_name, dangerous_cmd), None, {}
            )
            assert result.get("decision") == "block", (
                f"Should block for tool_name={tool_name}"
            )


class TestMorphDisallowedTools:
    """Test that disallowed tools are correctly configured."""

    def test_edit_and_grep_are_disallowed(self):
        """Edit and Grep tools should be in disallowed_tools list."""
        from src.cli import MORPH_DISALLOWED_TOOLS

        assert "Edit" in MORPH_DISALLOWED_TOOLS
        assert "Grep" in MORPH_DISALLOWED_TOOLS

    def test_disallowed_tools_is_list(self):
        """MORPH_DISALLOWED_TOOLS should be a list for SDK compatibility."""
        from src.cli import MORPH_DISALLOWED_TOOLS

        assert isinstance(MORPH_DISALLOWED_TOOLS, list)


class TestMorphApiKeyValidation:
    """Test API key validation."""

    def test_missing_api_key_raises_system_exit(self):
        """validate_morph_api_key should raise SystemExit if key is missing."""
        # Need to reimport after patching environment

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("MORPH_API_KEY", None)
            from src.cli import validate_morph_api_key

            with pytest.raises(SystemExit) as exc_info:
                validate_morph_api_key()
            assert "MORPH_API_KEY" in str(exc_info.value)

    def test_api_key_present_returns_key(self):
        """validate_morph_api_key should return the key if present."""
        with patch.dict(os.environ, {"MORPH_API_KEY": "test-key-123"}):
            from src.cli import validate_morph_api_key

            result = validate_morph_api_key()
            assert result == "test-key-123"


class TestMcpServerConfig:
    """Test MCP server configuration."""

    def test_get_mcp_servers_returns_morphllm_config(self):
        """get_mcp_servers should return correct MorphLLM config."""
        with patch.dict(os.environ, {"MORPH_API_KEY": "test-key"}):
            from src.cli import get_mcp_servers

            config = get_mcp_servers(Path("/tmp/repo"))

            assert "morphllm" in config
            assert config["morphllm"]["command"] == "npx"
            assert "-y" in config["morphllm"]["args"]
            assert "@morphllm/morphmcp" in config["morphllm"]["args"]

    def test_get_mcp_servers_includes_api_key(self):
        """MCP server config should include API key in env."""
        with patch.dict(os.environ, {"MORPH_API_KEY": "my-secret-key"}):
            from src.cli import get_mcp_servers

            config = get_mcp_servers(Path("/tmp/repo"))

            assert config["morphllm"]["env"]["MORPH_API_KEY"] == "my-secret-key"

    def test_get_mcp_servers_enables_all_tools(self):
        """MCP server config should enable all tools."""
        with patch.dict(os.environ, {"MORPH_API_KEY": "test-key"}):
            from src.cli import get_mcp_servers

            config = get_mcp_servers(Path("/tmp/repo"))

            assert config["morphllm"]["env"]["ENABLED_TOOLS"] == "all"

    def test_get_mcp_servers_enables_workspace_mode(self):
        """MCP server config should enable workspace mode."""
        with patch.dict(os.environ, {"MORPH_API_KEY": "test-key"}):
            from src.cli import get_mcp_servers

            config = get_mcp_servers(Path("/tmp/repo"))

            assert config["morphllm"]["env"]["WORKSPACE_MODE"] == "true"

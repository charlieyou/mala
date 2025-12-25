"""Tests for MorphLLM MCP integration.

Unit tests for the MorphLLM MCP server configuration and tool blocking.
These tests are fast and don't require API keys or network access.
"""

import os
import pytest
from unittest.mock import patch
from pathlib import Path


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
        import importlib

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

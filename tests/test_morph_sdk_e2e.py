"""Real Agent SDK E2E tests for MorphLLM MCP integration.

These tests spawn actual Claude agents and verify MCP tool behavior
using a mock MCP server that mimics MorphLLM's tools.

Requirements:
- Claude Code CLI must be authenticated (run `claude` to verify)

Run with: uv run pytest tests/test_morph_sdk_e2e.py -m e2e -v
"""

from pathlib import Path
import sys
import types

import pytest

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
)
from claude_agent_sdk.types import (
    HookMatcher,
    PreToolUseHookInput,
    HookContext,
    SyncHookJSONOutput,
)
from tests.claude_auth import is_claude_cli_available, has_valid_oauth_credentials

# All SDK tests are end-to-end (require CLI auth and API calls)
pytestmark = [pytest.mark.e2e, pytest.mark.morph]

# Define MCP config inline to avoid importing from cli.py (which triggers Braintrust)
MORPH_DISALLOWED_TOOLS = ["Edit", "Grep"]

# Path to the mock MCP server
MOCK_MCP_SERVER = Path(__file__).parent / "mock_mcp_server.py"


@pytest.fixture(autouse=True)
def require_claude_cli_auth() -> None:
    """Skip tests if Claude Code CLI is not available or OAuth credentials missing."""
    if not is_claude_cli_available():
        pytest.skip("Claude Code CLI not installed")
    if not has_valid_oauth_credentials():
        pytest.skip(
            "Claude Code CLI not logged in or token expired - run `claude` and login"
        )


@pytest.fixture(autouse=True)
def clean_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable Braintrust during tests to avoid network/logging side effects."""
    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    try:
        import braintrust

        class _NoopSpan:
            def __enter__(self) -> "_NoopSpan":
                return self

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: types.TracebackType | None,
            ) -> None:
                return None

            def log(self, **_kwargs: object) -> None:
                return None

        braintrust.start_span = lambda *args, **kwargs: _NoopSpan()  # type: ignore[assignment]
        braintrust.flush = lambda *args, **kwargs: None  # type: ignore[assignment]
    except Exception:
        pass


def get_mcp_servers(repo_path: Path) -> dict:
    """Get MCP servers configuration using mock server."""
    return {
        "morphllm": {
            "command": sys.executable,
            "args": [str(MOCK_MCP_SERVER)],
            "env": {
                "WORKSPACE_PATH": str(repo_path),
            },
        }
    }


async def block_morph_replaced_tools(
    hook_input: PreToolUseHookInput,
    stderr: str | None,
    context: HookContext,
) -> SyncHookJSONOutput:
    """PreToolUse hook to block tools replaced by MorphLLM MCP."""
    tool_name = hook_input["tool_name"]
    if tool_name in MORPH_DISALLOWED_TOOLS:
        return {
            "decision": "block",
            "reason": f"Use MorphLLM MCP tool instead of {tool_name}",
        }
    return {}


@pytest.fixture
def morph_agent_options(tmp_path: Path) -> ClaudeAgentOptions:
    """Agent options with MorphLLM MCP enabled."""
    return ClaudeAgentOptions(
        cwd=str(tmp_path),
        permission_mode="bypassPermissions",
        model="haiku",
        max_turns=5,
        mcp_servers=get_mcp_servers(tmp_path),
        disallowed_tools=MORPH_DISALLOWED_TOOLS,
        hooks={
            "PreToolUse": [
                HookMatcher(matcher=None, hooks=[block_morph_replaced_tools])  # type: ignore[arg-type]
            ],
        },
    )


class TestMcpToolsAvailable:
    """Test that MCP tools are available to agents."""

    @pytest.mark.asyncio
    @pytest.mark.flaky(reruns=2)
    async def test_agent_can_use_edit_file_tool(
        self, morph_agent_options: ClaudeAgentOptions, tmp_path: Path
    ) -> None:
        """Agent should have access to edit_file MCP tool.

        Note: This test is flaky because agent behavior is non-deterministic.
        The agent may choose different tools in some runs.
        """
        test_file = tmp_path / "test.py"
        test_file.write_text("# original content\n")

        prompt = """You MUST use the edit_file MCP tool to add a comment to test.py.
Add the line '# edited by morph' at the end of the file.
Do not use Read, Bash, or other tools - only use edit_file.
Respond with "DONE" when complete."""

        tool_names_used = []
        async with ClaudeSDKClient(options=morph_agent_options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if hasattr(block, "name"):
                            tool_names_used.append(block.name)

        # Verify edit_file was used (not Edit)
        # MCP tools are prefixed: mcp__morphllm__edit_file
        edit_file_used = any("edit_file" in name for name in tool_names_used)
        assert edit_file_used, f"Expected edit_file, got: {tool_names_used}"
        assert "Edit" not in tool_names_used, "Edit tool should be blocked"

    @pytest.mark.asyncio
    @pytest.mark.flaky(reruns=2)
    async def test_agent_can_use_warpgrep_search(
        self, morph_agent_options: ClaudeAgentOptions, tmp_path: Path
    ) -> None:
        """Agent should have access to warpgrep_codebase_search MCP tool.

        Note: This test is flaky because agent behavior is non-deterministic.
        The agent may choose not to use the warpgrep tool in some runs.
        """
        (tmp_path / "module.py").write_text("def hello_world(): pass\n")
        (tmp_path / "utils.py").write_text("def helper(): pass\n")

        prompt = """You MUST use the warpgrep_codebase_search MCP tool to find all Python functions.
Search for "def " pattern. Do not skip the tool use - I need to verify the tool works.
Report what you find."""

        tool_names_used = []
        async with ClaudeSDKClient(options=morph_agent_options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if hasattr(block, "name"):
                            tool_names_used.append(block.name)

        # MCP tools are prefixed: mcp__morphllm__warpgrep_codebase_search
        warpgrep_used = any("warpgrep" in name for name in tool_names_used)
        assert warpgrep_used, (
            f"Expected warpgrep_codebase_search, got: {tool_names_used}"
        )
        assert "Grep" not in tool_names_used, "Grep tool should be blocked"


class TestMorphWorkflowE2E:
    """End-to-end test of realistic workflow with MorphLLM tools."""

    @pytest.mark.asyncio
    @pytest.mark.flaky(reruns=2)
    async def test_search_and_edit_workflow(
        self, morph_agent_options: ClaudeAgentOptions, tmp_path: Path
    ) -> None:
        """Test full workflow: search for code, then edit it."""
        # Create a file with a bug
        buggy_file = tmp_path / "calculator.py"
        buggy_file.write_text(
            """
def add(a, b):
    return a - b  # BUG: should be a + b
"""
        )

        prompt = """You are fixing a bug in calculator.py:

1. Use warpgrep_codebase_search to find "return a - b"
2. Use edit_file to fix the bug (change - to +)
3. Report "BUG_FIXED" when done.

Use only MCP tools (edit_file, warpgrep_codebase_search), not Edit or Grep."""

        result_text = ""
        tool_names_used = []

        async with ClaudeSDKClient(options=morph_agent_options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if hasattr(block, "text"):
                            result_text += getattr(block, "text", "")
                        if hasattr(block, "name"):
                            tool_names_used.append(block.name)

        # Verify MCP tools were used (prefixed as mcp__morphllm__*)
        mcp_tools_used = any(
            "warpgrep" in name or "edit_file" in name for name in tool_names_used
        )
        assert mcp_tools_used, f"Expected MCP tools, got: {tool_names_used}"

        # Verify blocked tools were not used
        assert "Edit" not in tool_names_used, "Edit should be blocked"
        assert "Grep" not in tool_names_used, "Grep should be blocked"

        # Verify bug was fixed
        content = buggy_file.read_text()
        assert "a + b" in content or "a+b" in content, f"Bug not fixed: {content}"

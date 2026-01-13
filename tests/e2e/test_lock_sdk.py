"""Real Agent SDK integration tests for the locking system.

These tests actually spawn Claude agents using the SDK and verify
that they correctly interact with the locking scripts.

Requirements:
- Claude Code CLI must be authenticated (run `claude` to verify)
- Tests use OAuth via CLI, not API key

Run with: uv run pytest tests/ -m e2e -v
Default: uv run pytest tests/  (unit tests only)
"""

import asyncio
import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
)
from claude_agent_sdk.types import HookMatcher, McpSdkServerConfig

from src.infra.hooks import make_stop_hook
from src.infra.tools.env import SCRIPTS_DIR
from tests.e2e.claude_auth import is_claude_cli_available, has_valid_oauth_credentials

pytestmark = [pytest.mark.e2e, pytest.mark.flaky_sdk]


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
    """Clean environment for tests - use CLI auth."""
    # Remove API key to force OAuth via Claude Code CLI
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)


@pytest.fixture
def lock_env(tmp_path: Path) -> dict[str, Path]:
    """Provide isolated lock environment for tests."""
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    return {
        "lock_dir": lock_dir,
        "scripts_dir": SCRIPTS_DIR,
    }


def _agent_options(
    *,
    cwd: Path,
    max_turns: int = 3,
    env: dict[str, str] | None = None,
    hooks: dict[str, list[HookMatcher]] | None = None,
    mcp_servers: dict[str, McpSdkServerConfig] | None = None,
) -> ClaudeAgentOptions:
    """Create standardized agent options for SDK e2e tests."""
    options_kwargs: dict[str, object] = {
        "cwd": str(cwd),
        "permission_mode": "bypassPermissions",
        "model": "haiku",  # Use haiku for faster/cheaper tests
        "max_turns": max_turns,
        "system_prompt": {"type": "preset", "preset": "claude_code"},
        "setting_sources": ["project", "user"],
    }
    if env is not None:
        options_kwargs["env"] = env
    if hooks is not None:
        options_kwargs["hooks"] = hooks
    if mcp_servers is not None:
        options_kwargs["mcp_servers"] = mcp_servers
    return ClaudeAgentOptions(**options_kwargs)  # type: ignore[arg-type]


def _canonicalize_path(filepath: str, cwd: str) -> str:
    """Canonicalize a file path for consistent lock key generation.

    Matches the normalization logic in the shell scripts (realpath -m).

    Args:
        filepath: The file path to canonicalize.
        cwd: The working directory for relative path resolution.

    Returns:
        The canonical absolute path string.
    """
    path = Path(filepath)
    if not path.is_absolute():
        path = Path(cwd) / path
    # Use resolve() with strict=False to handle non-existent paths
    # This mimics 'realpath -m' behavior
    try:
        return str(path.resolve())
    except OSError:
        # Fallback for edge cases
        import os

        return str(Path(os.path.normpath(path)))


def lock_file_path(
    lock_dir: Path, filepath: str, cwd: str, repo_namespace: str | None = None
) -> Path:
    """Get the lock file path for a given filepath using hash-based naming.

    Matches the canonical key generation in the shell scripts.

    Args:
        lock_dir: The lock directory path.
        filepath: The file path to lock.
        cwd: The working directory for relative path resolution.
        repo_namespace: Optional repo namespace for disambiguation.

    Returns:
        Path to the lock file.
    """
    canonical = _canonicalize_path(filepath, cwd)
    if repo_namespace:
        key = f"{repo_namespace}:{canonical}"
    else:
        key = canonical
    key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
    return lock_dir / f"{key_hash}.lock"


def lock_file_exists(
    lock_dir: Path, filepath: str, cwd: str, repo_namespace: str | None = None
) -> bool:
    """Check if a lock file exists.

    Args:
        lock_dir: The lock directory path.
        filepath: The file path to check.
        cwd: The working directory for relative path resolution.
        repo_namespace: Optional repo namespace for disambiguation.
    """
    return lock_file_path(lock_dir, filepath, cwd, repo_namespace).exists()


def get_lock_holder(
    lock_dir: Path, filepath: str, cwd: str, repo_namespace: str | None = None
) -> str | None:
    """Get the holder of a lock file.

    Args:
        lock_dir: The lock directory path.
        filepath: The file path to check.
        cwd: The working directory for relative path resolution.
        repo_namespace: Optional repo namespace for disambiguation.
    """
    lock_path = lock_file_path(lock_dir, filepath, cwd, repo_namespace)
    if lock_path.exists():
        return lock_path.read_text().strip()
    return None


class TestAgentAcquiresLocks:
    """Test that agents can acquire locks using the scripts."""

    @pytest.mark.asyncio
    async def test_agent_runs_lock_try_script(
        self,
        lock_env: dict[str, Path],
        tmp_path: Path,
    ) -> None:
        """Agent can execute lock-try.sh and acquire a lock."""
        agent_id = "test-agent-acquire"
        lock_dir = lock_env["lock_dir"]
        cwd = str(tmp_path)

        env = {
            "LOCK_DIR": str(lock_dir),
            "AGENT_ID": agent_id,
            "PATH": f"{lock_env['scripts_dir']}:{os.environ.get('PATH', '')}",
        }
        options = _agent_options(cwd=tmp_path, env=env)

        prompt = """You are testing the lock system. Run these commands exactly:

```bash
set -euo pipefail
mkdir -p "$LOCK_DIR"
lock-try.sh test_file.py
```

After running the commands, respond with "DONE".
"""

        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                pass  # Consume all messages

        # Verify lock was acquired
        assert lock_file_exists(lock_dir, "test_file.py", cwd), "Lock file should exist"
        assert get_lock_holder(lock_dir, "test_file.py", cwd) == agent_id

    @pytest.mark.asyncio
    async def test_agent_checks_lock_ownership(
        self,
        lock_env: dict[str, Path],
        tmp_path: Path,
    ) -> None:
        """Agent can check if it holds a lock."""
        agent_id = "test-agent-check"
        lock_dir = lock_env["lock_dir"]
        result_path = tmp_path / "lock_check_result.txt"

        env = {
            "LOCK_DIR": str(lock_dir),
            "AGENT_ID": agent_id,
            "PATH": f"{lock_env['scripts_dir']}:{os.environ.get('PATH', '')}",
        }
        options = _agent_options(cwd=tmp_path, env=env)

        prompt = f"""You are testing the lock system. Run these commands exactly:

```bash
set -euo pipefail
mkdir -p "$LOCK_DIR"

# Acquire a lock
lock-try.sh myfile.py

# Check if we hold it
if lock-check.sh myfile.py; then
    echo "LOCK_HELD=true" > "{result_path}"
else
    echo "LOCK_HELD=false" > "{result_path}"
fi
```

Respond with "DONE".
"""

        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                pass  # Consume all messages

        assert result_path.exists(), "Expected lock check result file to be created"
        assert "true" in result_path.read_text().lower()


class TestAgentReleasesLocks:
    """Test that agents properly release locks."""

    @pytest.mark.asyncio
    async def test_agent_releases_single_lock(
        self,
        lock_env: dict[str, Path],
        tmp_path: Path,
    ) -> None:
        """Agent can release a specific lock."""
        agent_id = "test-agent-release"
        lock_dir = lock_env["lock_dir"]
        cwd = str(tmp_path)

        env = {
            "LOCK_DIR": str(lock_dir),
            "AGENT_ID": agent_id,
            "PATH": f"{lock_env['scripts_dir']}:{os.environ.get('PATH', '')}",
        }
        options = _agent_options(cwd=tmp_path, env=env)

        prompt = """You are testing the lock system. Run these commands exactly:

```bash
set -euo pipefail
mkdir -p "$LOCK_DIR"

# Acquire then release
lock-try.sh release_test.py
echo "Lock acquired"
lock-release.sh release_test.py
echo "Lock released"
```

Respond with "DONE".
"""

        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                pass

        # Lock should be gone
        assert not lock_file_exists(lock_dir, "release_test.py", cwd)

    @pytest.mark.asyncio
    async def test_agent_releases_all_locks(
        self, lock_env: dict[str, Path], tmp_path: Path
    ) -> None:
        """Agent can release all its locks at once."""
        agent_id = "test-agent-release-all"
        lock_dir = lock_env["lock_dir"]

        env = {
            "LOCK_DIR": str(lock_dir),
            "AGENT_ID": agent_id,
            "PATH": f"{lock_env['scripts_dir']}:{os.environ.get('PATH', '')}",
        }
        options = _agent_options(cwd=tmp_path, env=env)

        prompt = """You are testing the lock system. Run these commands exactly:

```bash
set -euo pipefail
mkdir -p "$LOCK_DIR"

# Acquire multiple locks
lock-try.sh file1.py
lock-try.sh file2.py
lock-try.sh file3.py
echo "Acquired 3 locks"

# Release all
lock-release-all.sh
echo "Released all locks"
```

Respond with "DONE".
"""

        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                pass

        # All locks should be gone
        locks = list(lock_dir.glob("*.lock"))
        assert len(locks) == 0, f"Expected no locks, found: {locks}"


class TestAgentHandlesContention:
    """Test agent behavior when encountering locked files."""

    @pytest.mark.asyncio
    async def test_agent_detects_blocked_file(
        self,
        lock_env: dict[str, Path],
        tmp_path: Path,
    ) -> None:
        """Agent correctly identifies when a file is locked by another."""
        lock_dir = lock_env["lock_dir"]
        our_agent = "our-agent"
        other_agent = "blocking-agent"
        cwd = str(tmp_path)
        result_path = tmp_path / "blocked_result.txt"

        # Pre-create a lock from another agent using hash-based naming
        lock_dir.mkdir(exist_ok=True)
        lock_path = lock_file_path(lock_dir, "blocked_file.py", cwd)
        lock_path.write_text(other_agent)

        env = {
            "LOCK_DIR": str(lock_dir),
            "AGENT_ID": our_agent,
            "PATH": f"{lock_env['scripts_dir']}:{os.environ.get('PATH', '')}",
        }
        options = _agent_options(cwd=tmp_path, env=env)

        prompt = f"""You are testing the lock system. Run these commands exactly:

```bash
set -euo pipefail

# Try to acquire a lock that's already held
if lock-try.sh blocked_file.py; then
    echo "RESULT=acquired" > "{result_path}"
else
    holder=$(lock-holder.sh blocked_file.py)
    echo "RESULT=blocked by $holder" > "{result_path}"
fi
```

Report the RESULT.
"""

        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                pass  # Consume all messages

        assert result_path.exists(), "Expected blocked-file result file to be created"
        result_text = result_path.read_text()
        assert "blocked" in result_text.lower()
        assert other_agent in result_text


class TestStopHookWithSDK:
    """Test the Stop hook integration with real SDK."""

    @pytest.mark.asyncio
    async def test_stop_hook_cleans_locks_on_agent_exit(
        self, lock_env: dict[str, Path], tmp_path: Path
    ) -> None:
        """Stop hook cleans up locks when agent finishes."""
        agent_id = "test-agent-stophook"
        lock_dir = lock_env["lock_dir"]

        # Create options with the stop hook
        env = {
            "LOCK_DIR": str(lock_dir),
            "AGENT_ID": agent_id,
            "PATH": f"{lock_env['scripts_dir']}:{os.environ.get('PATH', '')}",
        }
        options = _agent_options(
            cwd=tmp_path,
            env=env,
            hooks={
                "Stop": [HookMatcher(matcher=None, hooks=[make_stop_hook(agent_id)])],  # type: ignore[arg-type]
            },
        )

        # Set MALA_LOCK_DIR env var for the hook (will be picked up by get_lock_dir())
        original_lock_dir = os.environ.get("MALA_LOCK_DIR")
        os.environ["MALA_LOCK_DIR"] = str(lock_dir)

        try:
            prompt = """Run these commands to acquire locks:

```bash
set -euo pipefail
mkdir -p "$LOCK_DIR"
lock-try.sh orphan1.py
lock-try.sh orphan2.py
```

Then respond with "DONE" - do NOT release the locks manually.
"""

            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    pass

            # Give hook time to execute (poll briefly to avoid flakiness)
            for _ in range(10):
                if not list(lock_dir.glob("*.lock")):
                    break
                await asyncio.sleep(0.3)

            # Locks should be cleaned by the stop hook
            remaining = list(lock_dir.glob("*.lock"))
            assert len(remaining) == 0, (
                f"Stop hook should clean locks, found: {remaining}"
            )

        finally:
            if original_lock_dir is not None:
                os.environ["MALA_LOCK_DIR"] = original_lock_dir
            else:
                os.environ.pop("MALA_LOCK_DIR", None)


class TestMultiAgentWithSDK:
    """Test multiple agents interacting with locks via SDK."""

    @pytest.mark.asyncio
    async def test_sequential_agents_handoff_lock(
        self,
        lock_env: dict[str, Path],
        tmp_path: Path,
    ) -> None:
        """First agent releases lock, second agent acquires it."""
        lock_dir = lock_env["lock_dir"]
        agent1_id = "agent-1"
        agent2_id = "agent-2"
        cwd = str(tmp_path)
        result_path = tmp_path / "handoff_result.txt"

        # Agent 1 acquires and releases
        prompt1 = """Run these commands:

```bash
set -euo pipefail
mkdir -p "$LOCK_DIR"
lock-try.sh shared.py
echo "Agent 1 acquired lock"
lock-release.sh shared.py
echo "Agent 1 released lock"
```

Respond with "DONE".
"""

        env1 = {
            "LOCK_DIR": str(lock_dir),
            "AGENT_ID": agent1_id,
            "PATH": f"{lock_env['scripts_dir']}:{os.environ.get('PATH', '')}",
        }
        options1 = _agent_options(cwd=tmp_path, env=env1)
        async with ClaudeSDKClient(options=options1) as client:
            await client.query(prompt1)
            async for _ in client.receive_response():
                pass

        # Agent 2 should be able to acquire
        prompt2 = f"""Run these commands:

```bash
set -euo pipefail

if lock-try.sh shared.py; then
    echo "RESULT=success" > "{result_path}"
else
    echo "RESULT=failure" > "{result_path}"
fi
```

Report the result.
"""

        env2 = {
            "LOCK_DIR": str(lock_dir),
            "AGENT_ID": agent2_id,
            "PATH": f"{lock_env['scripts_dir']}:{os.environ.get('PATH', '')}",
        }
        options2 = _agent_options(cwd=tmp_path, env=env2)
        async with ClaudeSDKClient(options=options2) as client:
            await client.query(prompt2)
            async for message in client.receive_response():
                pass  # Consume all messages

        assert result_path.exists(), "Expected handoff result file to be created"
        assert "success" in result_path.read_text().lower()
        assert get_lock_holder(lock_dir, "shared.py", cwd) == agent2_id


class TestAgentWorkflowE2E:
    """End-to-end test of a realistic agent workflow."""

    @pytest.mark.asyncio
    async def test_full_implementation_workflow(
        self, lock_env: dict[str, Path], tmp_path: Path
    ) -> None:
        """Test the full workflow: acquire, work, commit, release."""
        agent_id = "impl-agent"
        lock_dir = lock_env["lock_dir"]
        cwd = str(tmp_path)

        # Create a simple test file to "implement"
        test_file = tmp_path / "feature.py"
        test_file.write_text("# TODO: implement feature\n")

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "commit.gpgsign", "false"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "add", "."], cwd=tmp_path, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

        env = {
            "LOCK_DIR": str(lock_dir),
            "AGENT_ID": agent_id,
            "PATH": f"{lock_env['scripts_dir']}:{os.environ.get('PATH', '')}",
        }
        options = _agent_options(cwd=tmp_path, max_turns=8, env=env)

        prompt = """You are implementing a feature. Follow this EXACT workflow:

1. Set up lock environment:
```bash
set -euo pipefail
mkdir -p "$LOCK_DIR"
```

2. Acquire lock on the file you'll edit:
```bash
lock-try.sh feature.py
```

3. Edit the file (add a simple function):
```bash
echo 'def hello(): return "world"' >> feature.py
```

4. Commit the change:
```bash
git add feature.py
git commit -m "Add hello function"
```

5. Release the lock:
```bash
lock-release.sh feature.py
```

6. Verify the lock is released:
```bash
if lock-check.sh feature.py; then
    echo "LOCK_RELEASED=false"
else
    echo "LOCK_RELEASED=true"
fi
```

Report "WORKFLOW COMPLETE" when done.
"""

        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                pass  # Consume all messages

        # Verify the workflow completed correctly
        # Lock should be released
        assert not lock_file_exists(lock_dir, "feature.py", cwd), (
            "Lock should be released"
        )

        # File should be modified
        content = test_file.read_text()
        assert "hello" in content, "Feature should be implemented"

        # Should have a commit
        log = subprocess.run(
            ["git", "log", "--oneline"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )
        assert "hello" in log.stdout.lower() or "Add" in log.stdout


class TestMCPLockingToolsSchemaValidation:
    """Test MCP locking tools are accepted by Claude API.

    These tests verify that the tool schemas are valid and accepted
    by the Claude API. This catches schema validation errors like
    using oneOf/allOf/anyOf at the top level, which would cause
    400 errors at runtime.
    """

    @pytest.mark.asyncio
    async def test_mcp_locking_tools_schema_accepted(
        self,
        tmp_path: Path,
    ) -> None:
        """MCP locking tool schemas are accepted by Claude API.

        This test creates an agent with the MCP locking server and
        sends a simple query. If the schema is invalid, Claude API
        returns a 400 error before any response is generated.
        """
        from src.infra.tools.locking_mcp import create_locking_mcp_server

        # Create MCP server config
        result = create_locking_mcp_server(
            agent_id="schema-test-agent",
            repo_namespace=str(tmp_path),
            emit_lock_event=lambda e: None,
        )
        # create_locking_mcp_server returns McpSdkServerConfig when _return_handlers=False
        mcp_config: McpSdkServerConfig = result  # type: ignore[assignment]

        # Create agent options with MCP server
        options = _agent_options(
            cwd=tmp_path,
            max_turns=1,
            mcp_servers={"mala-locking": mcp_config},
        )

        # Simple query - just needs to reach the API without schema error
        # If schema is invalid, this will raise an API error
        async with ClaudeSDKClient(options=options) as client:
            await client.query("Say 'hello' - do not use any tools.")
            async for message in client.receive_response():
                # Just consume messages - test passes if no API error
                pass

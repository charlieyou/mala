"""Real Agent SDK integration tests for the locking system.

These tests actually spawn Claude agents using the SDK and verify
that they correctly interact with the locking scripts.

Requirements:
- Claude Code CLI must be authenticated (run `claude` to verify)
- Tests use OAuth via CLI, not API key

Run with: uv run pytest tests/ -m slow -v
Default: uv run pytest tests/  (excludes slow tests)
"""

import asyncio
import hashlib
import subprocess
from pathlib import Path
from types import TracebackType
from typing import Self

import pytest

from claude_agent_sdk import (
    ClaudeSDKClient,
    ClaudeAgentOptions,
    AssistantMessage,
)
from claude_agent_sdk.types import HookMatcher

from src.tools.locking import make_stop_hook
from src.tools.env import SCRIPTS_DIR

# All SDK tests are slow (require API calls)
pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def clean_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clean environment for tests - disable Braintrust, use CLI auth."""
    # Disable Braintrust during tests to avoid network/logging side effects.
    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)
    # Remove API key to force OAuth via Claude Code CLI
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    try:
        import braintrust

        class _NoopSpan:
            def __enter__(self) -> Self:
                return self

            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc_val: BaseException | None,
                exc_tb: TracebackType | None,
            ) -> None:
                return None

            def log(self, **_kwargs: object) -> None:
                return None

        braintrust.start_span = lambda *args, **kwargs: _NoopSpan()  # type: ignore[assignment]
        braintrust.flush = lambda *args, **kwargs: None  # type: ignore[assignment]
    except Exception:
        pass


@pytest.fixture
def lock_env(tmp_path: Path) -> dict[str, Path]:
    """Provide isolated lock environment for tests."""
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    return {
        "lock_dir": lock_dir,
        "scripts_dir": SCRIPTS_DIR,
    }


@pytest.fixture
def agent_options(tmp_path: Path) -> ClaudeAgentOptions:
    """Base agent options for tests."""
    return ClaudeAgentOptions(
        cwd=str(tmp_path),
        permission_mode="bypassPermissions",
        model="haiku",  # Use haiku for faster/cheaper tests
        max_turns=3,  # Limit turns for test speed
    )


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
    async def test_agent_runs_lock_try_script(self, lock_env: dict[str, Path], agent_options: ClaudeAgentOptions, tmp_path: Path) -> None:
        """Agent can execute lock-try.sh and acquire a lock."""
        agent_id = "test-agent-acquire"
        lock_dir = lock_env["lock_dir"]
        cwd = str(tmp_path)

        prompt = f"""You are testing the lock system. Run these commands exactly:

```bash
export LOCK_DIR="{lock_dir}"
export AGENT_ID="{agent_id}"
export PATH="{lock_env["scripts_dir"]}:$PATH"
mkdir -p "$LOCK_DIR"
lock-try.sh test_file.py
echo "Lock acquired with exit code: $?"
```

After running the commands, respond with "DONE".
"""

        async with ClaudeSDKClient(options=agent_options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                pass  # Consume all messages

        # Verify lock was acquired
        assert lock_file_exists(lock_dir, "test_file.py", cwd), "Lock file should exist"
        assert get_lock_holder(lock_dir, "test_file.py", cwd) == agent_id

    @pytest.mark.asyncio
    async def test_agent_checks_lock_ownership(self, lock_env: dict[str, Path], agent_options: ClaudeAgentOptions, tmp_path: Path) -> None:
        """Agent can check if it holds a lock."""
        agent_id = "test-agent-check"
        lock_dir = lock_env["lock_dir"]

        prompt = f"""You are testing the lock system. Run these commands exactly:

```bash
export LOCK_DIR="{lock_dir}"
export AGENT_ID="{agent_id}"
export PATH="{lock_env["scripts_dir"]}:$PATH"
mkdir -p "$LOCK_DIR"

# Acquire a lock
lock-try.sh myfile.py

# Check if we hold it
if lock-check.sh myfile.py; then
    echo "LOCK_HELD=true"
else
    echo "LOCK_HELD=false"
fi
```

Respond with what LOCK_HELD equals.
"""

        result_text = ""
        async with ClaudeSDKClient(options=agent_options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if hasattr(block, "text"):
                            result_text += getattr(block, "text", "")

        assert "true" in result_text.lower() or "LOCK_HELD=true" in result_text


class TestAgentReleasesLocks:
    """Test that agents properly release locks."""

    @pytest.mark.asyncio
    async def test_agent_releases_single_lock(self, lock_env: dict[str, Path], agent_options: ClaudeAgentOptions, tmp_path: Path) -> None:
        """Agent can release a specific lock."""
        agent_id = "test-agent-release"
        lock_dir = lock_env["lock_dir"]
        cwd = str(tmp_path)

        prompt = f"""You are testing the lock system. Run these commands exactly:

```bash
export LOCK_DIR="{lock_dir}"
export AGENT_ID="{agent_id}"
export PATH="{lock_env["scripts_dir"]}:$PATH"
mkdir -p "$LOCK_DIR"

# Acquire then release
lock-try.sh release_test.py
echo "Lock acquired"
lock-release.sh release_test.py
echo "Lock released"
```

Respond with "DONE".
"""

        async with ClaudeSDKClient(options=agent_options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                pass

        # Lock should be gone
        assert not lock_file_exists(lock_dir, "release_test.py", cwd)

    @pytest.mark.asyncio
    async def test_agent_releases_all_locks(self, lock_env: dict[str, Path], agent_options: ClaudeAgentOptions) -> None:
        """Agent can release all its locks at once."""
        agent_id = "test-agent-release-all"
        lock_dir = lock_env["lock_dir"]

        prompt = f"""You are testing the lock system. Run these commands exactly:

```bash
export LOCK_DIR="{lock_dir}"
export AGENT_ID="{agent_id}"
export PATH="{lock_env["scripts_dir"]}:$PATH"
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

        async with ClaudeSDKClient(options=agent_options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                pass

        # All locks should be gone
        locks = list(lock_dir.glob("*.lock"))
        assert len(locks) == 0, f"Expected no locks, found: {locks}"


class TestAgentHandlesContention:
    """Test agent behavior when encountering locked files."""

    @pytest.mark.asyncio
    async def test_agent_detects_blocked_file(self, lock_env: dict[str, Path], agent_options: ClaudeAgentOptions, tmp_path: Path) -> None:
        """Agent correctly identifies when a file is locked by another."""
        lock_dir = lock_env["lock_dir"]
        our_agent = "our-agent"
        other_agent = "blocking-agent"
        cwd = str(tmp_path)

        # Pre-create a lock from another agent using hash-based naming
        lock_dir.mkdir(exist_ok=True)
        lock_path = lock_file_path(lock_dir, "blocked_file.py", cwd)
        lock_path.write_text(other_agent)

        prompt = f"""You are testing the lock system. Run these commands exactly:

```bash
export LOCK_DIR="{lock_dir}"
export AGENT_ID="{our_agent}"
export PATH="{lock_env["scripts_dir"]}:$PATH"

# Try to acquire a lock that's already held
if lock-try.sh blocked_file.py; then
    echo "RESULT=acquired"
else
    holder=$(lock-holder.sh blocked_file.py)
    echo "RESULT=blocked by $holder"
fi
```

Report the RESULT.
"""

        result_text = ""
        async with ClaudeSDKClient(options=agent_options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if hasattr(block, "text"):
                            result_text += getattr(block, "text", "")

        assert "blocked" in result_text.lower()
        assert other_agent in result_text


class TestStopHookWithSDK:
    """Test the Stop hook integration with real SDK."""

    @pytest.mark.asyncio
    async def test_stop_hook_cleans_locks_on_agent_exit(self, lock_env: dict[str, Path], tmp_path: Path) -> None:
        """Stop hook cleans up locks when agent finishes."""
        agent_id = "test-agent-stophook"
        lock_dir = lock_env["lock_dir"]

        # Create options with the stop hook
        options = ClaudeAgentOptions(
            cwd=str(tmp_path),
            permission_mode="bypassPermissions",
            model="haiku",
            max_turns=3,
            hooks={
                "Stop": [HookMatcher(matcher=None, hooks=[make_stop_hook(agent_id)])],
            },
        )

        # Patch LOCK_DIR for the hook
        import src.tools.locking

        original_lock_dir = src.tools.locking.LOCK_DIR
        src.tools.locking.LOCK_DIR = lock_dir

        try:
            prompt = f"""Run these commands to acquire locks:

```bash
export LOCK_DIR="{lock_dir}"
export AGENT_ID="{agent_id}"
export PATH="{lock_env["scripts_dir"]}:$PATH"
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
            src.tools.locking.LOCK_DIR = original_lock_dir


class TestMultiAgentWithSDK:
    """Test multiple agents interacting with locks via SDK."""

    @pytest.mark.asyncio
    async def test_sequential_agents_handoff_lock(
        self,
        lock_env: dict[str, Path],
        agent_options: ClaudeAgentOptions,
        tmp_path: Path,
    ) -> None:
        """First agent releases lock, second agent acquires it."""
        lock_dir = lock_env["lock_dir"]
        agent1_id = "agent-1"
        agent2_id = "agent-2"
        cwd = str(tmp_path)

        # Agent 1 acquires and releases
        prompt1 = f"""Run these commands:

```bash
export LOCK_DIR="{lock_dir}"
export AGENT_ID="{agent1_id}"
export PATH="{lock_env["scripts_dir"]}:$PATH"
mkdir -p "$LOCK_DIR"
lock-try.sh shared.py
echo "Agent 1 acquired lock"
lock-release.sh shared.py
echo "Agent 1 released lock"
```

Respond with "DONE".
"""

        async with ClaudeSDKClient(options=agent_options) as client:
            await client.query(prompt1)
            async for _ in client.receive_response():
                pass

        # Agent 2 should be able to acquire
        prompt2 = f"""Run these commands:

```bash
export LOCK_DIR="{lock_dir}"
export AGENT_ID="{agent2_id}"
export PATH="{lock_env["scripts_dir"]}:$PATH"

if lock-try.sh shared.py; then
    echo "SUCCESS: Agent 2 acquired lock"
else
    echo "FAILED: Could not acquire lock"
fi
```

Report the result.
"""

        result_text = ""
        async with ClaudeSDKClient(options=agent_options) as client:
            await client.query(prompt2)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if hasattr(block, "text"):
                            result_text += getattr(block, "text", "")

        assert "SUCCESS" in result_text or "acquired" in result_text.lower()
        assert get_lock_holder(lock_dir, "shared.py", cwd) == agent2_id


class TestAgentWorkflowE2E:
    """End-to-end test of a realistic agent workflow."""

    @pytest.mark.asyncio
    async def test_full_implementation_workflow(self, lock_env: dict[str, Path], tmp_path: Path) -> None:
        """Test the full workflow: acquire, work, commit, release."""
        agent_id = "impl-agent"
        lock_dir = lock_env["lock_dir"]
        cwd = str(tmp_path)

        # Create a simple test file to "implement"
        test_file = tmp_path / "feature.py"
        test_file.write_text("# TODO: implement feature\n")

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "config", "user.email", "test@test.com"],
            cwd=tmp_path,
            capture_output=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"],
            cwd=tmp_path,
            capture_output=True,
        )
        subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "initial"],
            cwd=tmp_path,
            capture_output=True,
        )

        options = ClaudeAgentOptions(
            cwd=cwd,
            permission_mode="bypassPermissions",
            model="haiku",
            max_turns=8,
        )

        prompt = f"""You are implementing a feature. Follow this EXACT workflow:

1. Set up lock environment:
```bash
export LOCK_DIR="{lock_dir}"
export AGENT_ID="{agent_id}"
export PATH="{lock_env["scripts_dir"]}:$PATH"
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

        result_text = ""
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if hasattr(block, "text"):
                            result_text += getattr(block, "text", "")

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

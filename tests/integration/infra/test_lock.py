"""Integration tests for agent lock usage workflow.

Tests the full lifecycle of how agents use the locking system:
1. Environment setup from prompt template
2. Lock acquisition during implementation
3. Lock release after commit
4. Stop hook cleanup for orphaned locks
5. Orchestrator fallback cleanup
"""

import hashlib
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import pytest

from src.domain.prompts import load_prompts
from src.infra.hooks import make_stop_hook
from src.infra.tools.env import PROMPTS_DIR, SCRIPTS_DIR

if TYPE_CHECKING:
    from claude_agent_sdk.types import HookContext, StopHookInput

# Load prompts once for tests
_prompts = load_prompts(PROMPTS_DIR)

pytestmark = pytest.mark.integration


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
    # Use resolve() to handle symlinks and normalize path
    # This mimics 'realpath -m' behavior
    try:
        return str(path.resolve())
    except OSError:
        # Fallback for edge cases
        return str(Path(os.path.normpath(path)))


def lock_file_path(
    lock_dir: Path, filepath: str, cwd: str, repo_namespace: str | None = None
) -> Path:
    """Get the lock file path for a given filepath using hash-based naming.

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


@pytest.fixture
def lock_env(tmp_path: Path) -> Path:
    """Provide isolated lock environment."""
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    return lock_dir


@pytest.fixture
def agent_env(lock_env: Path, tmp_path: Path) -> dict[str, str]:
    """Simulate agent environment as set up by prompt template."""
    cwd = str(tmp_path)
    # Start with os.environ but exclude REPO_NAMESPACE to ensure clean test isolation
    env = {k: v for k, v in os.environ.items() if k != "REPO_NAMESPACE"}
    return {
        **env,
        "LOCK_DIR": str(lock_env),
        "AGENT_ID": "bd-42-abc12345",
        "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
        "PWD": cwd,  # Working directory for path resolution
    }


def run_lock_script(
    script: str, args: list[str], env: dict[str, str], cwd: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a lock script in the given environment."""
    return subprocess.run(
        [str(SCRIPTS_DIR / script), *args],
        env=env,
        capture_output=True,
        text=True,
        cwd=cwd or env.get("PWD"),
    )


class TestAgentLockWorkflow:
    """Test the complete agent workflow with locks."""

    def test_agent_acquires_locks_before_editing(
        self, agent_env: dict[str, str], lock_env: Path
    ) -> None:
        """Simulate agent acquiring locks for multiple files."""
        files = ["src/main.py", "src/utils.py", "tests/test_main.py"]

        # Acquire locks like an agent would
        for f in files:
            result = run_lock_script("lock-try.sh", [f], agent_env)
            assert result.returncode == 0, f"Failed to acquire lock for {f}"

        # Verify all locks exist (using hash-based naming)
        for f in files:
            assert lock_file_path(lock_env, f, agent_env["PWD"]).exists()

    def test_agent_checks_lock_before_editing(
        self, agent_env: dict[str, str], lock_env: Path
    ) -> None:
        """Agent should verify it holds the lock before editing."""
        run_lock_script("lock-try.sh", ["file.py"], agent_env)

        # Check lock ownership
        result = run_lock_script("lock-check.sh", ["file.py"], agent_env)
        assert result.returncode == 0, "Agent should hold the lock"

    def test_agent_handles_blocked_file(
        self, agent_env: dict[str, str], lock_env: Path
    ) -> None:
        """Agent encounters a file locked by another agent."""
        # Another agent holds the lock
        other_env = {**agent_env, "AGENT_ID": "bd-other-agent"}
        run_lock_script("lock-try.sh", ["contested.py"], other_env)

        # Our agent tries to acquire - should fail
        result = run_lock_script("lock-try.sh", ["contested.py"], agent_env)
        assert result.returncode == 1, "Should fail to acquire contested lock"

        # Agent checks who holds it
        result = run_lock_script("lock-holder.sh", ["contested.py"], agent_env)
        assert result.stdout.strip() == "bd-other-agent"

    def test_agent_releases_locks_after_commit(
        self, agent_env: dict[str, str], lock_env: Path
    ) -> None:
        """Agent releases locks after successful git commit."""
        files = ["src/main.py", "src/utils.py"]

        # Acquire locks
        for f in files:
            run_lock_script("lock-try.sh", [f], agent_env)

        # Simulate work and commit (just verify locks are still held)
        for f in files:
            result = run_lock_script("lock-check.sh", [f], agent_env)
            assert result.returncode == 0

        # Release all locks (as agent does after commit)
        run_lock_script("lock-release-all.sh", [], agent_env)

        # Verify locks are gone
        locks = list(lock_env.glob("*.lock"))
        assert len(locks) == 0, "All locks should be released after commit"

    def test_agent_workflow_with_partial_blocking(
        self, agent_env: dict[str, str], lock_env: Path
    ) -> None:
        """Test realistic scenario: some files locked, some available."""
        # Another agent has locked utils.py
        other_env = {**agent_env, "AGENT_ID": "bd-blocker"}
        run_lock_script("lock-try.sh", ["utils.py"], other_env)

        # Our agent tries to lock multiple files
        files_to_lock = ["main.py", "utils.py", "config.py"]
        acquired = []
        blocked = []

        for f in files_to_lock:
            result = run_lock_script("lock-try.sh", [f], agent_env)
            if result.returncode == 0:
                acquired.append(f)
            else:
                holder = run_lock_script("lock-holder.sh", [f], agent_env)
                blocked.append((f, holder.stdout.strip()))

        assert "main.py" in acquired
        assert "config.py" in acquired
        assert len(blocked) == 1
        assert blocked[0] == ("utils.py", "bd-blocker")

    def test_agent_reacquire_own_lock_idempotent(
        self, agent_env: dict[str, str], lock_env: Path
    ) -> None:
        """Same agent re-acquiring a lock it already holds succeeds (idempotent)."""
        # First acquire
        result1 = run_lock_script("lock-try.sh", ["reacquire.py"], agent_env)
        assert result1.returncode == 0, "Initial acquire should succeed"

        # Verify we hold it
        result_check = run_lock_script("lock-check.sh", ["reacquire.py"], agent_env)
        assert result_check.returncode == 0, "Should hold lock after first acquire"

        # Re-acquire same lock (idempotent)
        result2 = run_lock_script("lock-try.sh", ["reacquire.py"], agent_env)
        assert result2.returncode == 0, "Re-acquire by same agent should succeed"

        # Still hold it
        result_check2 = run_lock_script("lock-check.sh", ["reacquire.py"], agent_env)
        assert result_check2.returncode == 0, "Should still hold lock after re-acquire"


class TestStopHookIntegration:
    """Test the Stop hook cleanup mechanism."""

    @pytest.mark.asyncio
    async def test_stop_hook_cleans_up_locks(
        self, lock_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stop hook should release all locks held by the agent."""
        agent_id = "bd-test-stophook"
        env = {
            **os.environ,
            "LOCK_DIR": str(lock_env),
            "AGENT_ID": agent_id,
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
        }

        # Agent acquires some locks
        for f in ["file1.py", "file2.py", "file3.py"]:
            run_lock_script("lock-try.sh", [f], env)

        assert len(list(lock_env.glob("*.lock"))) == 3

        # Simulate Stop hook being called
        stop_hook = make_stop_hook(agent_id)

        # Create mock hook input
        hook_input = cast(
            "StopHookInput",
            {
                "session_id": "test-session",
                "transcript_path": "/tmp/transcript",
                "cwd": "/tmp",
                "hook_event_name": "Stop",
                "stop_hook_active": True,
            },
        )

        # Set MALA_LOCK_DIR to use our test directory
        monkeypatch.setenv("MALA_LOCK_DIR", str(lock_env))
        await stop_hook(hook_input, None, cast("HookContext", MagicMock()))

        # Verify locks are cleaned up
        remaining = list(lock_env.glob("*.lock"))
        assert len(remaining) == 0, (
            f"Stop hook should clean all locks, found: {remaining}"
        )

    @pytest.mark.asyncio
    async def test_stop_hook_preserves_other_agent_locks(
        self, lock_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stop hook should only clean up its own agent's locks."""
        our_agent = "bd-our-agent"
        other_agent = "bd-other-agent"
        cwd = str(tmp_path)

        # Exclude REPO_NAMESPACE from os.environ to ensure clean test isolation
        clean_env = {k: v for k, v in os.environ.items() if k != "REPO_NAMESPACE"}
        our_env = {
            **clean_env,
            "LOCK_DIR": str(lock_env),
            "AGENT_ID": our_agent,
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
            "PWD": cwd,
        }
        other_env = {**our_env, "AGENT_ID": other_agent}

        # Both agents acquire locks
        run_lock_script("lock-try.sh", ["our-file.py"], our_env)
        run_lock_script("lock-try.sh", ["other-file.py"], other_env)

        # Our agent's stop hook runs
        stop_hook = make_stop_hook(our_agent)
        # Set MALA_LOCK_DIR to use our test directory
        monkeypatch.setenv("MALA_LOCK_DIR", str(lock_env))
        await stop_hook(
            cast(
                "StopHookInput",
                {
                    "session_id": "test",
                    "transcript_path": "/tmp/t",
                    "cwd": "/tmp",
                    "hook_event_name": "Stop",
                    "stop_hook_active": True,
                },
            ),
            None,
            cast("HookContext", MagicMock()),
        )

        # Our lock should be gone, other's should remain (using hash-based naming)
        assert not lock_file_path(lock_env, "our-file.py", cwd).exists()
        assert lock_file_path(lock_env, "other-file.py", cwd).exists()


class TestLockCleanupFunction:
    """Test cleanup_agent_locks function for orphaned lock removal."""

    def test_cleanup_agent_locks_removes_orphaned_locks(
        self, lock_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """cleanup_agent_locks removes locks held by a crashed/timed-out agent."""
        from src.infra.tools.locking import cleanup_agent_locks

        agent_id = "bd-crashed-agent"
        env = {
            **os.environ,
            "LOCK_DIR": str(lock_env),
            "AGENT_ID": agent_id,
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
        }

        # Agent acquired locks but crashed
        for f in ["orphan1.py", "orphan2.py"]:
            run_lock_script("lock-try.sh", [f], env)

        # Set MALA_LOCK_DIR for the locking module
        monkeypatch.setenv("MALA_LOCK_DIR", str(lock_env))

        # Run cleanup directly (orchestrator delegates to this function)
        count, paths = cleanup_agent_locks(agent_id)

        # Should have cleaned 2 locks
        assert count == 2
        assert len(paths) == 2

        # Locks should be cleaned
        remaining = list(lock_env.glob("*.lock"))
        assert len(remaining) == 0


class TestMultiAgentScenarios:
    """Test scenarios with multiple concurrent agents."""

    def test_two_agents_different_files_no_conflict(self, lock_env: Path) -> None:
        """Two agents working on different files should not conflict."""
        agent1_env = {
            **os.environ,
            "LOCK_DIR": str(lock_env),
            "AGENT_ID": "agent-1",
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
        }
        agent2_env = {**agent1_env, "AGENT_ID": "agent-2"}

        # Agent 1 locks its files
        run_lock_script("lock-try.sh", ["feature_a.py"], agent1_env)
        run_lock_script("lock-try.sh", ["feature_a_test.py"], agent1_env)

        # Agent 2 locks different files - should succeed
        r1 = run_lock_script("lock-try.sh", ["feature_b.py"], agent2_env)
        r2 = run_lock_script("lock-try.sh", ["feature_b_test.py"], agent2_env)

        assert r1.returncode == 0
        assert r2.returncode == 0

        # Both should hold their locks
        assert (
            run_lock_script("lock-check.sh", ["feature_a.py"], agent1_env).returncode
            == 0
        )
        assert (
            run_lock_script("lock-check.sh", ["feature_b.py"], agent2_env).returncode
            == 0
        )

    def test_agent_waits_then_acquires_released_lock(self, lock_env: Path) -> None:
        """Agent can acquire lock after holder releases it."""
        holder_env = {
            **os.environ,
            "LOCK_DIR": str(lock_env),
            "AGENT_ID": "holder",
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
        }
        waiter_env = {**holder_env, "AGENT_ID": "waiter"}

        # Holder acquires lock
        run_lock_script("lock-try.sh", ["shared.py"], holder_env)

        # Waiter can't acquire
        assert run_lock_script("lock-try.sh", ["shared.py"], waiter_env).returncode == 1

        # Holder releases
        run_lock_script("lock-release.sh", ["shared.py"], holder_env)

        # Waiter can now acquire
        assert run_lock_script("lock-try.sh", ["shared.py"], waiter_env).returncode == 0
        assert (
            run_lock_script("lock-check.sh", ["shared.py"], waiter_env).returncode == 0
        )


class TestEdgeCases:
    """Test edge cases in the integration."""

    def test_empty_lock_dir_operations(self, lock_env: Path, tmp_path: Path) -> None:
        """Operations on empty lock directory should not fail."""
        cwd = str(tmp_path)
        env = {
            **os.environ,
            "LOCK_DIR": str(lock_env),
            "AGENT_ID": "test-agent",
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
            "PWD": cwd,
        }

        # All these should succeed on empty dir
        assert run_lock_script("lock-release-all.sh", [], env).returncode == 0
        assert run_lock_script("lock-holder.sh", ["any.py"], env).returncode == 0
        assert (
            run_lock_script("lock-check.sh", ["any.py"], env).returncode == 1
        )  # Not held

    def test_lock_release_idempotent(self, lock_env: Path, tmp_path: Path) -> None:
        """Releasing the same lock multiple times should be safe."""
        cwd = str(tmp_path)
        env = {
            **os.environ,
            "LOCK_DIR": str(lock_env),
            "AGENT_ID": "test-agent",
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
            "PWD": cwd,
        }

        run_lock_script("lock-try.sh", ["file.py"], env)

        # Release multiple times
        for _ in range(3):
            result = run_lock_script("lock-release.sh", ["file.py"], env)
            assert result.returncode == 0

    def test_special_characters_in_path(self, lock_env: Path, tmp_path: Path) -> None:
        """Paths with special characters should be handled."""
        cwd = str(tmp_path)
        env = {
            **os.environ,
            "LOCK_DIR": str(lock_env),
            "AGENT_ID": "test-agent",
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
            "PWD": cwd,
        }

        # Various path formats
        paths = [
            "src/module/file.py",
            "tests/test_module.py",
            "deeply/nested/path/to/file.py",
        ]

        for p in paths:
            assert run_lock_script("lock-try.sh", [p], env).returncode == 0

        assert len(list(lock_env.glob("*.lock"))) == len(paths)


class TestPathCanonicalization:
    """Test that paths are properly canonicalized for consistent lock keys."""

    def test_relative_and_absolute_paths_same_lock(
        self, lock_env: Path, tmp_path: Path
    ) -> None:
        """Relative and absolute paths to the same file produce the same lock."""
        cwd = str(tmp_path)
        env = {
            **os.environ,
            "LOCK_DIR": str(lock_env),
            "AGENT_ID": "agent-1",
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
            "PWD": cwd,
        }

        # Lock using relative path
        result = run_lock_script("lock-try.sh", ["file.py"], env)
        assert result.returncode == 0

        # Try to lock the same file using absolute path - succeeds (idempotent re-acquire)
        absolute_path = str(tmp_path / "file.py")
        result2 = run_lock_script("lock-try.sh", [absolute_path], env)
        assert result2.returncode == 0, "Same agent can re-acquire via absolute path"

        # Verify there's only one lock file
        locks = list(lock_env.glob("*.lock"))
        assert len(locks) == 1

    def test_normalized_paths_same_lock(self, lock_env: Path, tmp_path: Path) -> None:
        """Paths with . and .. segments are normalized to the same lock."""
        cwd = str(tmp_path)
        env = {
            **os.environ,
            "LOCK_DIR": str(lock_env),
            "AGENT_ID": "agent-1",
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
            "PWD": cwd,
        }

        # Lock using path with . and ..
        result = run_lock_script("lock-try.sh", ["./src/../src/file.py"], env)
        assert result.returncode == 0

        # Try to lock the normalized version - succeeds (idempotent re-acquire)
        result2 = run_lock_script("lock-try.sh", ["src/file.py"], env)
        assert result2.returncode == 0, "Same agent can re-acquire via normalized path"

        # Verify only one lock
        locks = list(lock_env.glob("*.lock"))
        assert len(locks) == 1

    def test_different_files_different_locks(
        self, lock_env: Path, tmp_path: Path
    ) -> None:
        """Different files produce different locks."""
        cwd = str(tmp_path)
        env = {
            **os.environ,
            "LOCK_DIR": str(lock_env),
            "AGENT_ID": "agent-1",
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
            "PWD": cwd,
        }

        # Lock two different files
        assert run_lock_script("lock-try.sh", ["file1.py"], env).returncode == 0
        assert run_lock_script("lock-try.sh", ["file2.py"], env).returncode == 0

        # Should have two distinct locks
        locks = list(lock_env.glob("*.lock"))
        assert len(locks) == 2


class TestRepoNamespaceIntegration:
    """Test REPO_NAMESPACE behavior for cross-repo disambiguation."""

    def test_python_hook_recognizes_shell_lock_with_namespace(
        self, lock_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Python get_lock_holder should find locks created by shell scripts with REPO_NAMESPACE.

        This tests the critical invariant that locks acquired via lock-try.sh with
        REPO_NAMESPACE set are correctly recognized by the Python PreToolUse hook.

        The hook receives absolute file paths from Claude Code tools and must compute
        the same lock key as the shell script did.
        """
        from src.infra.tools.locking import get_lock_holder

        agent_id = "test-agent-cross-cwd"
        repo_namespace = str(tmp_path)
        cwd = str(tmp_path)

        # Set MALA_LOCK_DIR so get_lock_holder uses our test lock directory
        monkeypatch.setenv("MALA_LOCK_DIR", str(lock_env))

        env = {
            **os.environ,
            "LOCK_DIR": str(lock_env),
            "AGENT_ID": agent_id,
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
            "PWD": cwd,
            "REPO_NAMESPACE": repo_namespace,
        }

        # Shell script acquires lock with REPO_NAMESPACE
        result = run_lock_script("lock-try.sh", ["src/file.py"], env)
        assert result.returncode == 0, f"Failed to acquire lock: {result.stderr}"

        # Python hook receives absolute path (as Claude Code tools provide)
        absolute_path = str(tmp_path / "src" / "file.py")

        # Python should find the lock holder - THIS IS THE BUG:
        # get_lock_holder is not passed repo_namespace, so it computes a different key
        holder = get_lock_holder(absolute_path, repo_namespace=repo_namespace)

        assert holder == agent_id, (
            f"Python get_lock_holder should find lock created by shell script. "
            f"Expected {agent_id!r}, got {holder!r}. "
            f"Lock files in dir: {list(lock_env.glob('*.lock'))}"
        )

    def test_python_hook_recognizes_shell_lock_from_different_cwd(
        self, lock_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test lock recognition when mala is launched from outside the repo.

        Scenario:
        1. Agent acquires lock via shell script from repo dir (cwd=/repo)
        2. Python hook checks lock with absolute path (running from any cwd)
        3. They must compute the same lock key
        """
        from src.infra.tools.locking import get_lock_holder

        agent_id = "cross-cwd-agent"
        repo_path = tmp_path / "repo"
        repo_path.mkdir()
        repo_namespace = str(repo_path)

        # Set MALA_LOCK_DIR
        monkeypatch.setenv("MALA_LOCK_DIR", str(lock_env))

        env = {
            **os.environ,
            "LOCK_DIR": str(lock_env),
            "AGENT_ID": agent_id,
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
            "REPO_NAMESPACE": repo_namespace,
        }

        # Shell script acquires lock using relative path FROM THE REPO
        result = run_lock_script("lock-try.sh", ["config.py"], env, cwd=str(repo_path))
        assert result.returncode == 0, f"Lock failed: {result.stderr}"

        # Python hook receives absolute path
        absolute_path = str(repo_path / "config.py")

        # Python should find the same lock (with namespace)
        holder = get_lock_holder(absolute_path, repo_namespace=repo_namespace)

        assert holder == agent_id, (
            f"Lock not found from different cwd. Expected {agent_id!r}, got {holder!r}"
        )

    def test_same_file_different_namespaces_different_locks(
        self, lock_env: Path, tmp_path: Path
    ) -> None:
        """Same relative path in different namespaces produces different locks."""
        cwd = str(tmp_path)
        base_env = {
            **os.environ,
            "LOCK_DIR": str(lock_env),
            "AGENT_ID": "agent-1",
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
            "PWD": cwd,
        }

        # Lock file in namespace A
        env_a = {**base_env, "REPO_NAMESPACE": "/repo/a"}
        result = run_lock_script("lock-try.sh", ["config.py"], env_a)
        assert result.returncode == 0

        # Lock same relative path in namespace B - should succeed (different namespace)
        env_b = {**base_env, "REPO_NAMESPACE": "/repo/b"}
        result2 = run_lock_script("lock-try.sh", ["config.py"], env_b)
        assert result2.returncode == 0, (
            "Different namespace should produce different lock"
        )

        # Should have two distinct locks
        locks = list(lock_env.glob("*.lock"))
        assert len(locks) == 2

    def test_same_namespace_same_file_conflict(
        self, lock_env: Path, tmp_path: Path
    ) -> None:
        """Same file in same namespace should conflict."""
        cwd = str(tmp_path)
        base_env = {
            **os.environ,
            "LOCK_DIR": str(lock_env),
            "AGENT_ID": "agent-1",
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
            "PWD": cwd,
            "REPO_NAMESPACE": "/repo/shared",
        }

        # Agent 1 locks file
        result = run_lock_script("lock-try.sh", ["shared.py"], base_env)
        assert result.returncode == 0

        # Agent 2 tries to lock same file in same namespace - should fail
        env2 = {**base_env, "AGENT_ID": "agent-2"}
        result2 = run_lock_script("lock-try.sh", ["shared.py"], env2)
        assert result2.returncode == 1, "Same namespace should conflict"

        # Only one lock
        locks = list(lock_env.glob("*.lock"))
        assert len(locks) == 1

    def test_namespace_included_in_holder_lookup(
        self, lock_env: Path, tmp_path: Path
    ) -> None:
        """Lock holder lookup respects namespace."""
        cwd = str(tmp_path)
        env = {
            **os.environ,
            "LOCK_DIR": str(lock_env),
            "AGENT_ID": "ns-agent",
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
            "PWD": cwd,
            "REPO_NAMESPACE": "/my/repo",
        }

        # Acquire lock with namespace
        run_lock_script("lock-try.sh", ["module.py"], env)

        # Check holder with same namespace - should return our agent
        result = run_lock_script("lock-holder.sh", ["module.py"], env)
        assert result.stdout.strip() == "ns-agent"

        # Check lock ownership with same namespace
        result = run_lock_script("lock-check.sh", ["module.py"], env)
        assert result.returncode == 0, "Should hold lock in same namespace"

    def test_empty_namespace_treated_as_none(
        self, lock_env: Path, tmp_path: Path
    ) -> None:
        """Empty REPO_NAMESPACE is treated the same as not set."""
        cwd = str(tmp_path)
        # Exclude REPO_NAMESPACE from os.environ to ensure clean test isolation
        clean_env = {k: v for k, v in os.environ.items() if k != "REPO_NAMESPACE"}
        base_env = {
            **clean_env,
            "LOCK_DIR": str(lock_env),
            "AGENT_ID": "agent-1",
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
            "PWD": cwd,
        }

        # Lock without namespace
        result = run_lock_script("lock-try.sh", ["file.py"], base_env)
        assert result.returncode == 0

        # Try to lock with empty namespace - succeeds (idempotent, same agent)
        env_empty = {**base_env, "REPO_NAMESPACE": ""}
        result2 = run_lock_script("lock-try.sh", ["file.py"], env_empty)
        assert result2.returncode == 0, "Same agent can re-acquire with empty namespace"


class TestReleaseRunLocks:
    """Test run-scoped lock cleanup to avoid releasing locks from other runs."""

    def test_release_run_locks_only_removes_owned_locks(
        self, lock_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cleanup should only remove locks owned by the specified agent IDs."""
        from src.infra.tools.locking import release_run_locks, try_lock

        # Set MALA_LOCK_DIR for the locking module
        monkeypatch.setenv("MALA_LOCK_DIR", str(lock_env))

        # Agent IDs from two different runs
        run1_agents = ["mala-123-aaaa", "mala-456-bbbb"]
        run2_agents = ["mala-789-cccc"]

        # Run 1 acquires locks
        assert try_lock("file1.py", run1_agents[0])
        assert try_lock("file2.py", run1_agents[1])

        # Run 2 acquires locks
        assert try_lock("file3.py", run2_agents[0])

        # Verify all 3 lock files exist
        locks_before = set(lock_env.glob("*.lock"))
        assert len(locks_before) == 3

        # Run 1 shuts down and cleans up only its agents
        cleaned = release_run_locks(run1_agents)
        assert cleaned == 2

        # Run 2's lock should remain
        locks_after = set(lock_env.glob("*.lock"))
        assert len(locks_after) == 1

        # Verify the remaining lock is from run 2's agent
        remaining_lock = next(iter(locks_after))
        assert remaining_lock.read_text().strip() == run2_agents[0]

    def test_release_run_locks_handles_empty_agent_list(
        self, lock_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cleanup with empty agent list should not remove any locks."""
        from src.infra.tools.locking import release_run_locks, try_lock

        monkeypatch.setenv("MALA_LOCK_DIR", str(lock_env))

        # Create a lock
        assert try_lock("some_file.py", "other-agent")

        # Release with empty list
        cleaned = release_run_locks([])
        assert cleaned == 0

        # Lock should remain
        assert len(list(lock_env.glob("*.lock"))) == 1

    def test_release_run_locks_handles_missing_lock_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cleanup should handle non-existent lock directory gracefully."""
        from src.infra.tools.locking import release_run_locks

        non_existent = tmp_path / "nonexistent_locks"
        monkeypatch.setenv("MALA_LOCK_DIR", str(non_existent))

        # Should not raise, should return 0
        cleaned = release_run_locks(["some-agent"])
        assert cleaned == 0


class TestCLIEntryPoint:
    """Test the Python CLI entry point used by shell script wrappers."""

    def test_cli_try_acquires_lock(
        self, lock_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI try command should acquire lock and return 0."""
        from src.infra.tools.locking import _cli_main, get_lock_holder
        import sys

        monkeypatch.setenv("LOCK_DIR", str(lock_env))
        monkeypatch.setenv("MALA_LOCK_DIR", str(lock_env))
        monkeypatch.setenv("AGENT_ID", "cli-test-agent")
        monkeypatch.delenv("REPO_NAMESPACE", raising=False)

        # Use absolute path (as shell scripts do after normalization)
        test_file = str(tmp_path / "cli_test.py")

        original_argv = sys.argv
        try:
            sys.argv = ["locking", "try", test_file]
            result = _cli_main()
            assert result == 0

            # Verify lock was acquired
            holder = get_lock_holder(test_file)
            assert holder == "cli-test-agent"
        finally:
            sys.argv = original_argv

    def test_cli_try_fails_when_locked(
        self, lock_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI try command should return 1 when lock is held."""
        from src.infra.tools.locking import _cli_main, try_lock
        import sys

        monkeypatch.setenv("LOCK_DIR", str(lock_env))
        monkeypatch.setenv("MALA_LOCK_DIR", str(lock_env))
        monkeypatch.setenv("AGENT_ID", "second-agent")
        monkeypatch.delenv("REPO_NAMESPACE", raising=False)

        test_file = str(tmp_path / "contested.py")

        # First agent acquires lock
        assert try_lock(test_file, "first-agent")

        original_argv = sys.argv
        try:
            sys.argv = ["locking", "try", test_file]
            result = _cli_main()
            assert result == 1
        finally:
            sys.argv = original_argv

    def test_cli_holder_returns_agent_id(
        self,
        lock_env: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """CLI holder command should print agent ID to stdout."""
        from src.infra.tools.locking import _cli_main, try_lock
        import sys

        monkeypatch.setenv("LOCK_DIR", str(lock_env))
        monkeypatch.setenv("MALA_LOCK_DIR", str(lock_env))
        monkeypatch.delenv("REPO_NAMESPACE", raising=False)

        test_file = str(tmp_path / "holder_test.py")

        # Create a lock
        assert try_lock(test_file, "holder-agent")

        original_argv = sys.argv
        try:
            sys.argv = ["locking", "holder", test_file]
            result = _cli_main()
            assert result == 0

            captured = capsys.readouterr()
            assert "holder-agent" in captured.out
        finally:
            sys.argv = original_argv

    def test_cli_check_returns_0_for_owner(
        self, lock_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI check command should return 0 when agent holds lock."""
        from src.infra.tools.locking import _cli_main, try_lock
        import sys

        monkeypatch.setenv("LOCK_DIR", str(lock_env))
        monkeypatch.setenv("MALA_LOCK_DIR", str(lock_env))
        monkeypatch.setenv("AGENT_ID", "check-agent")
        monkeypatch.delenv("REPO_NAMESPACE", raising=False)

        test_file = str(tmp_path / "check_test.py")

        # Agent acquires lock
        assert try_lock(test_file, "check-agent")

        original_argv = sys.argv
        try:
            sys.argv = ["locking", "check", test_file]
            result = _cli_main()
            assert result == 0
        finally:
            sys.argv = original_argv

    def test_cli_check_returns_1_for_non_owner(
        self, lock_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI check command should return 1 when agent doesn't hold lock."""
        from src.infra.tools.locking import _cli_main, try_lock
        import sys

        monkeypatch.setenv("LOCK_DIR", str(lock_env))
        monkeypatch.setenv("MALA_LOCK_DIR", str(lock_env))
        monkeypatch.setenv("AGENT_ID", "other-agent")
        monkeypatch.delenv("REPO_NAMESPACE", raising=False)

        test_file = str(tmp_path / "check_test2.py")

        # Different agent holds lock
        assert try_lock(test_file, "owner-agent")

        original_argv = sys.argv
        try:
            sys.argv = ["locking", "check", test_file]
            result = _cli_main()
            assert result == 1
        finally:
            sys.argv = original_argv

    def test_cli_release_removes_own_lock(
        self, lock_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI release command should remove lock held by agent."""
        from src.infra.tools.locking import _cli_main, try_lock, get_lock_holder
        import sys

        monkeypatch.setenv("LOCK_DIR", str(lock_env))
        monkeypatch.setenv("MALA_LOCK_DIR", str(lock_env))
        monkeypatch.setenv("AGENT_ID", "release-agent")
        monkeypatch.delenv("REPO_NAMESPACE", raising=False)

        test_file = str(tmp_path / "release_test.py")

        # Agent acquires and releases lock
        assert try_lock(test_file, "release-agent")
        assert get_lock_holder(test_file) == "release-agent"

        original_argv = sys.argv
        try:
            sys.argv = ["locking", "release", test_file]
            result = _cli_main()
            assert result == 0

            # Lock should be gone
            assert get_lock_holder(test_file) is None
        finally:
            sys.argv = original_argv

    def test_cli_release_all_removes_all_own_locks(
        self, lock_env: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI release-all command should remove all locks held by agent."""
        from src.infra.tools.locking import _cli_main, try_lock
        import sys

        monkeypatch.setenv("LOCK_DIR", str(lock_env))
        monkeypatch.setenv("MALA_LOCK_DIR", str(lock_env))
        monkeypatch.setenv("AGENT_ID", "batch-agent")
        monkeypatch.delenv("REPO_NAMESPACE", raising=False)

        # Agent acquires multiple locks (using absolute paths)
        for i in range(3):
            test_file = str(tmp_path / f"batch{i}.py")
            assert try_lock(test_file, "batch-agent")

        assert len(list(lock_env.glob("*.lock"))) == 3

        original_argv = sys.argv
        try:
            sys.argv = ["locking", "release-all"]
            result = _cli_main()
            assert result == 0

            # All locks should be gone
            assert len(list(lock_env.glob("*.lock"))) == 0
        finally:
            sys.argv = original_argv

    def test_cli_errors_without_lock_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CLI should return 2 when LOCK_DIR is not set."""
        from src.infra.tools.locking import _cli_main
        import sys

        monkeypatch.delenv("LOCK_DIR", raising=False)
        monkeypatch.setenv("AGENT_ID", "test-agent")
        monkeypatch.delenv("REPO_NAMESPACE", raising=False)

        original_argv = sys.argv
        try:
            sys.argv = ["locking", "try", "test.py"]
            result = _cli_main()
            assert result == 2
        finally:
            sys.argv = original_argv

    def test_cli_errors_without_agent_id(
        self, lock_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """CLI should return 2 when AGENT_ID is not set (for commands that require it)."""
        from src.infra.tools.locking import _cli_main
        import sys

        monkeypatch.setenv("LOCK_DIR", str(lock_env))
        monkeypatch.delenv("AGENT_ID", raising=False)
        monkeypatch.delenv("REPO_NAMESPACE", raising=False)

        original_argv = sys.argv
        try:
            sys.argv = ["locking", "try", "test.py"]
            result = _cli_main()
            assert result == 2
        finally:
            sys.argv = original_argv

    @pytest.mark.parametrize("command", ["try", "check", "holder", "release"])
    def test_cli_rejects_extra_args(
        self, lock_env: Path, monkeypatch: pytest.MonkeyPatch, command: str
    ) -> None:
        """CLI should return 2 when extra arguments are passed to single-arg commands."""
        from src.infra.tools.locking import _cli_main
        import sys

        monkeypatch.setenv("LOCK_DIR", str(lock_env))
        monkeypatch.setenv("AGENT_ID", "test-agent")
        monkeypatch.delenv("REPO_NAMESPACE", raising=False)

        original_argv = sys.argv
        try:
            sys.argv = ["locking", command, "test.py", "extra-arg"]
            result = _cli_main()
            assert result == 2
        finally:
            sys.argv = original_argv


class TestWaitForLockAsync:
    """Tests for async lock waiting."""

    @pytest.mark.asyncio
    async def test_wait_for_lock_async_acquires_free_lock(
        self, lock_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """wait_for_lock_async should acquire a free lock immediately."""
        from src.infra.tools.locking import wait_for_lock_async

        monkeypatch.setenv("MALA_LOCK_DIR", str(lock_env))

        test_file = lock_env / "test.py"
        test_file.touch()

        result = await wait_for_lock_async(
            str(test_file), "test-agent", timeout_seconds=1.0
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_lock_async_times_out_on_held_lock(
        self, lock_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """wait_for_lock_async should return False after timeout."""
        from src.infra.tools.locking import try_lock, wait_for_lock_async

        monkeypatch.setenv("MALA_LOCK_DIR", str(lock_env))

        test_file = lock_env / "test.py"
        test_file.touch()

        # First agent acquires lock
        assert try_lock(str(test_file), "agent-1") is True

        # Second agent times out waiting
        result = await wait_for_lock_async(
            str(test_file),
            "agent-2",
            timeout_seconds=0.1,
            poll_interval_ms=10,
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_wait_for_lock_async_acquires_after_release(
        self, lock_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """wait_for_lock_async should acquire lock once released."""
        import asyncio

        from src.infra.tools.locking import (
            release_lock,
            try_lock,
            wait_for_lock_async,
        )

        monkeypatch.setenv("MALA_LOCK_DIR", str(lock_env))

        test_file = lock_env / "test.py"
        test_file.touch()

        # First agent acquires lock
        assert try_lock(str(test_file), "agent-1") is True

        async def release_after_delay() -> None:
            await asyncio.sleep(0.05)
            release_lock(str(test_file), "agent-1")

        # Start release task and wait concurrently
        release_task = asyncio.create_task(release_after_delay())
        result = await wait_for_lock_async(
            str(test_file),
            "agent-2",
            timeout_seconds=1.0,
            poll_interval_ms=10,
        )
        await release_task

        assert result is True


class TestGetLockHolder:
    """Tests for get_lock_holder function."""

    def test_returns_agent_id_when_locked(
        self, lock_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_lock_holder returns the agent_id of the lock holder."""
        from src.infra.tools.locking import get_lock_holder, try_lock

        monkeypatch.setenv("MALA_LOCK_DIR", str(lock_env))

        test_file = lock_env / "locked.py"
        test_file.touch()

        # Acquire lock
        assert try_lock(str(test_file), "holder-agent-123") is True

        # get_lock_holder returns the holder
        holder = get_lock_holder(str(test_file))
        assert holder == "holder-agent-123"

    def test_returns_none_when_not_locked(
        self, lock_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_lock_holder returns None when file is not locked."""
        from src.infra.tools.locking import get_lock_holder

        monkeypatch.setenv("MALA_LOCK_DIR", str(lock_env))

        test_file = lock_env / "not_locked.py"
        test_file.touch()

        # No lock exists
        holder = get_lock_holder(str(test_file))
        assert holder is None

    def test_returns_none_for_nonexistent_file(
        self, lock_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_lock_holder returns None for paths that don't exist."""
        from src.infra.tools.locking import get_lock_holder

        monkeypatch.setenv("MALA_LOCK_DIR", str(lock_env))

        # File doesn't exist and was never locked
        holder = get_lock_holder("/nonexistent/path/file.py")
        assert holder is None

    def test_respects_repo_namespace(
        self, lock_env: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_lock_holder respects repo_namespace parameter."""
        from src.infra.tools.locking import get_lock_holder, try_lock

        monkeypatch.setenv("MALA_LOCK_DIR", str(lock_env))

        test_file = "namespaced.py"

        # Lock with namespace
        assert try_lock(test_file, "ns-agent", repo_namespace="/repo/a") is True

        # Query with same namespace - finds holder
        holder = get_lock_holder(test_file, repo_namespace="/repo/a")
        assert holder == "ns-agent"

        # Query without namespace - returns None (different key)
        holder_no_ns = get_lock_holder(test_file, repo_namespace=None)
        assert holder_no_ns is None

        # Query with different namespace - returns None
        holder_diff_ns = get_lock_holder(test_file, repo_namespace="/repo/b")
        assert holder_diff_ns is None

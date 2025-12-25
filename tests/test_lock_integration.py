"""Integration tests for agent lock usage workflow.

Tests the full lifecycle of how agents use the locking system:
1. Environment setup from prompt template
2. Lock acquisition during implementation
3. Lock release after commit
4. Stop hook cleanup for orphaned locks
5. Orchestrator fallback cleanup
"""

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.main import (
    IMPLEMENTER_PROMPT_TEMPLATE,
    SCRIPTS_DIR,
    make_stop_hook,
)


@pytest.fixture
def lock_env(tmp_path):
    """Provide isolated lock environment."""
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    return lock_dir


@pytest.fixture
def agent_env(lock_env):
    """Simulate agent environment as set up by prompt template."""
    return {
        **os.environ,
        "LOCK_DIR": str(lock_env),
        "AGENT_ID": "bd-42-abc12345",
        "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
    }


def run_lock_script(script: str, args: list[str], env: dict) -> subprocess.CompletedProcess:
    """Run a lock script in the given environment."""
    return subprocess.run(
        [str(SCRIPTS_DIR / script)] + args,
        env=env,
        capture_output=True,
        text=True,
    )


class TestPromptTemplateIntegration:
    """Test that the prompt template correctly configures lock environment."""

    def test_template_includes_scripts_dir_placeholder(self):
        assert "{scripts_dir}" in IMPLEMENTER_PROMPT_TEMPLATE

    def test_template_includes_lock_dir_placeholder(self):
        assert "{lock_dir}" in IMPLEMENTER_PROMPT_TEMPLATE

    def test_template_includes_agent_id_placeholder(self):
        assert "{agent_id}" in IMPLEMENTER_PROMPT_TEMPLATE

    def test_template_formats_correctly(self, lock_env):
        prompt = IMPLEMENTER_PROMPT_TEMPLATE.format(
            issue_id="bd-99",
            repo_path="/tmp/repo",
            lock_dir=lock_env,
            scripts_dir=SCRIPTS_DIR,
            agent_id="bd-99-test1234",
        )
        # Verify the formatted values appear in the prompt
        assert str(lock_env) in prompt
        assert str(SCRIPTS_DIR) in prompt
        assert "bd-99-test1234" in prompt

    def test_scripts_dir_exists_and_contains_scripts(self):
        assert SCRIPTS_DIR.exists()
        expected_scripts = [
            "lock-try.sh",
            "lock-check.sh",
            "lock-holder.sh",
            "lock-release.sh",
            "lock-release-all.sh",
        ]
        for script in expected_scripts:
            assert (SCRIPTS_DIR / script).exists(), f"Missing script: {script}"
            assert os.access(SCRIPTS_DIR / script, os.X_OK), f"Script not executable: {script}"


class TestAgentLockWorkflow:
    """Test the complete agent workflow with locks."""

    def test_agent_acquires_locks_before_editing(self, agent_env, lock_env):
        """Simulate agent acquiring locks for multiple files."""
        files = ["src/main.py", "src/utils.py", "tests/test_main.py"]

        # Acquire locks like an agent would
        for f in files:
            result = run_lock_script("lock-try.sh", [f], agent_env)
            assert result.returncode == 0, f"Failed to acquire lock for {f}"

        # Verify all locks exist
        for f in files:
            lock_name = f.replace("/", "_") + ".lock"
            assert (lock_env / lock_name).exists()

    def test_agent_checks_lock_before_editing(self, agent_env, lock_env):
        """Agent should verify it holds the lock before editing."""
        run_lock_script("lock-try.sh", ["file.py"], agent_env)

        # Check lock ownership
        result = run_lock_script("lock-check.sh", ["file.py"], agent_env)
        assert result.returncode == 0, "Agent should hold the lock"

    def test_agent_handles_blocked_file(self, agent_env, lock_env):
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

    def test_agent_releases_locks_after_commit(self, agent_env, lock_env):
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

    def test_agent_workflow_with_partial_blocking(self, agent_env, lock_env):
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


class TestStopHookIntegration:
    """Test the Stop hook cleanup mechanism."""

    @pytest.mark.asyncio
    async def test_stop_hook_cleans_up_locks(self, lock_env):
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
        hook_input = {
            "session_id": "test-session",
            "transcript_path": "/tmp/transcript",
            "cwd": "/tmp",
            "hook_event_name": "Stop",
            "stop_hook_active": True,
        }

        # Patch LOCK_DIR to use our test directory
        with patch("src.main.LOCK_DIR", lock_env):
            await stop_hook(hook_input, None, MagicMock())

        # Verify locks are cleaned up
        remaining = list(lock_env.glob("*.lock"))
        assert len(remaining) == 0, f"Stop hook should clean all locks, found: {remaining}"

    @pytest.mark.asyncio
    async def test_stop_hook_preserves_other_agent_locks(self, lock_env):
        """Stop hook should only clean up its own agent's locks."""
        our_agent = "bd-our-agent"
        other_agent = "bd-other-agent"

        our_env = {
            **os.environ,
            "LOCK_DIR": str(lock_env),
            "AGENT_ID": our_agent,
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
        }
        other_env = {**our_env, "AGENT_ID": other_agent}

        # Both agents acquire locks
        run_lock_script("lock-try.sh", ["our-file.py"], our_env)
        run_lock_script("lock-try.sh", ["other-file.py"], other_env)

        # Our agent's stop hook runs
        stop_hook = make_stop_hook(our_agent)
        with patch("src.main.LOCK_DIR", lock_env):
            await stop_hook(
                {
                    "session_id": "test",
                    "transcript_path": "/tmp/t",
                    "cwd": "/tmp",
                    "hook_event_name": "Stop",
                    "stop_hook_active": True,
                },
                None,
                MagicMock(),
            )

        # Our lock should be gone, other's should remain
        assert not (lock_env / "our-file.py.lock").exists()
        assert (lock_env / "other-file.py.lock").exists()


class TestOrchestratorCleanup:
    """Test orchestrator's fallback lock cleanup."""

    def test_cleanup_agent_locks_removes_orphaned_locks(self, lock_env, monkeypatch):
        """Orchestrator cleans up locks when agent crashes/times out."""
        from src.main import MalaOrchestrator

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

        # Patch LOCK_DIR for the orchestrator
        monkeypatch.setattr("src.main.LOCK_DIR", lock_env)

        # Create orchestrator and run cleanup
        orchestrator = MalaOrchestrator.__new__(MalaOrchestrator)
        orchestrator.repo_path = Path("/tmp/fake")
        orchestrator._cleanup_agent_locks(agent_id)

        # Locks should be cleaned
        remaining = list(lock_env.glob("*.lock"))
        assert len(remaining) == 0


class TestMultiAgentScenarios:
    """Test scenarios with multiple concurrent agents."""

    def test_two_agents_different_files_no_conflict(self, lock_env):
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
        assert run_lock_script("lock-check.sh", ["feature_a.py"], agent1_env).returncode == 0
        assert run_lock_script("lock-check.sh", ["feature_b.py"], agent2_env).returncode == 0

    def test_agent_waits_then_acquires_released_lock(self, lock_env):
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
        assert run_lock_script("lock-check.sh", ["shared.py"], waiter_env).returncode == 0


class TestEdgeCases:
    """Test edge cases in the integration."""

    def test_empty_lock_dir_operations(self, lock_env):
        """Operations on empty lock directory should not fail."""
        env = {
            **os.environ,
            "LOCK_DIR": str(lock_env),
            "AGENT_ID": "test-agent",
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
        }

        # All these should succeed on empty dir
        assert run_lock_script("lock-release-all.sh", [], env).returncode == 0
        assert run_lock_script("lock-holder.sh", ["any.py"], env).returncode == 0
        assert run_lock_script("lock-check.sh", ["any.py"], env).returncode == 1  # Not held

    def test_lock_release_idempotent(self, lock_env):
        """Releasing the same lock multiple times should be safe."""
        env = {
            **os.environ,
            "LOCK_DIR": str(lock_env),
            "AGENT_ID": "test-agent",
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
        }

        run_lock_script("lock-try.sh", ["file.py"], env)

        # Release multiple times
        for _ in range(3):
            result = run_lock_script("lock-release.sh", ["file.py"], env)
            assert result.returncode == 0

    def test_special_characters_in_path(self, lock_env):
        """Paths with special characters should be handled."""
        env = {
            **os.environ,
            "LOCK_DIR": str(lock_env),
            "AGENT_ID": "test-agent",
            "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
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

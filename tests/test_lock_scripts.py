"""Comprehensive test suite for mala lock scripts."""

import os
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "src" / "scripts"


@pytest.fixture
def lock_env(tmp_path):
    """Provide a clean lock environment for each test."""
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    return {
        "LOCK_DIR": str(lock_dir),
        "AGENT_ID": "test-agent",
        "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
    }


def run_script(name: str, args: list[str], env: dict) -> subprocess.CompletedProcess:
    """Run a lock script with the given environment."""
    script = SCRIPTS_DIR / name
    full_env = {**os.environ, **env}
    return subprocess.run(
        [str(script)] + args,
        env=full_env,
        capture_output=True,
        text=True,
    )


def lock_file_path(lock_dir: str, filepath: str) -> Path:
    """Get the lock file path for a given filepath."""
    lock_name = filepath.replace("/", "_") + ".lock"
    return Path(lock_dir) / lock_name


class TestBasicLockAcquisition:
    """Test basic lock acquisition functionality."""

    def test_acquires_lock_successfully(self, lock_env):
        result = run_script("lock-try.sh", ["src/main.py"], lock_env)
        assert result.returncode == 0

    def test_creates_lock_file_with_correct_name(self, lock_env):
        run_script("lock-try.sh", ["src/main.py"], lock_env)
        lock_path = lock_file_path(lock_env["LOCK_DIR"], "src/main.py")
        assert lock_path.exists()

    def test_lock_file_contains_agent_id(self, lock_env):
        run_script("lock-try.sh", ["src/main.py"], lock_env)
        lock_path = lock_file_path(lock_env["LOCK_DIR"], "src/main.py")
        assert lock_path.read_text().strip() == lock_env["AGENT_ID"]


class TestLockCheck:
    """Test lock ownership checking."""

    def test_returns_0_when_i_hold_lock(self, lock_env):
        run_script("lock-try.sh", ["file.py"], lock_env)
        result = run_script("lock-check.sh", ["file.py"], lock_env)
        assert result.returncode == 0

    def test_returns_1_when_another_agent_holds_lock(self, lock_env):
        run_script("lock-try.sh", ["file.py"], lock_env)
        other_env = {**lock_env, "AGENT_ID": "other-agent"}
        result = run_script("lock-check.sh", ["file.py"], other_env)
        assert result.returncode == 1

    def test_returns_1_for_unlocked_file(self, lock_env):
        result = run_script("lock-check.sh", ["nonexistent.py"], lock_env)
        assert result.returncode == 1


class TestLockHolder:
    """Test lock holder identification."""

    def test_returns_correct_holder(self, lock_env):
        lock_env["AGENT_ID"] = "holder-agent"
        run_script("lock-try.sh", ["config.yaml"], lock_env)
        result = run_script("lock-holder.sh", ["config.yaml"], lock_env)
        assert result.returncode == 0
        assert result.stdout.strip() == "holder-agent"

    def test_returns_empty_for_unlocked_file(self, lock_env):
        result = run_script("lock-holder.sh", ["nonexistent.py"], lock_env)
        assert result.returncode == 0
        assert result.stdout.strip() == ""


class TestLockRelease:
    """Test lock release functionality."""

    def test_removes_lock_file(self, lock_env):
        run_script("lock-try.sh", ["release-test.py"], lock_env)
        run_script("lock-release.sh", ["release-test.py"], lock_env)
        lock_path = lock_file_path(lock_env["LOCK_DIR"], "release-test.py")
        assert not lock_path.exists()

    def test_can_reacquire_after_release(self, lock_env):
        run_script("lock-try.sh", ["release-test.py"], lock_env)
        run_script("lock-release.sh", ["release-test.py"], lock_env)
        result = run_script("lock-try.sh", ["release-test.py"], lock_env)
        assert result.returncode == 0

    def test_cannot_release_another_agents_lock(self, lock_env):
        run_script("lock-try.sh", ["other-lock.py"], lock_env)
        other_env = {**lock_env, "AGENT_ID": "other-agent"}
        run_script("lock-release.sh", ["other-lock.py"], other_env)
        lock_path = lock_file_path(lock_env["LOCK_DIR"], "other-lock.py")
        assert lock_path.exists()


class TestReleaseAllLocks:
    """Test releasing all locks held by an agent."""

    def test_removes_all_own_locks(self, lock_env):
        lock_env["AGENT_ID"] = "batch-agent"
        for f in ["file1.py", "file2.py", "file3.py"]:
            run_script("lock-try.sh", [f], lock_env)

        lock_count = len(list(Path(lock_env["LOCK_DIR"]).glob("*.lock")))
        assert lock_count == 3

        run_script("lock-release-all.sh", [], lock_env)
        lock_count = len(list(Path(lock_env["LOCK_DIR"]).glob("*.lock")))
        assert lock_count == 0

    def test_preserves_other_agents_locks(self, lock_env):
        # Agent A creates locks
        env_a = {**lock_env, "AGENT_ID": "agent-A"}
        run_script("lock-try.sh", ["a1.py"], env_a)
        run_script("lock-try.sh", ["a2.py"], env_a)

        # Agent B creates a lock
        env_b = {**lock_env, "AGENT_ID": "agent-B"}
        run_script("lock-try.sh", ["b1.py"], env_b)

        # Agent A releases all
        run_script("lock-release-all.sh", [], env_a)

        # Agent B's lock should still exist
        assert lock_file_path(lock_env["LOCK_DIR"], "b1.py").exists()
        assert not lock_file_path(lock_env["LOCK_DIR"], "a1.py").exists()
        assert not lock_file_path(lock_env["LOCK_DIR"], "a2.py").exists()


class TestLockContention:
    """Test lock contention between agents."""

    def test_second_agent_cannot_acquire_held_lock(self, lock_env):
        run_script("lock-try.sh", ["contested.py"], lock_env)
        other_env = {**lock_env, "AGENT_ID": "second-agent"}
        result = run_script("lock-try.sh", ["contested.py"], other_env)
        assert result.returncode == 1

    def test_original_holder_retains_lock(self, lock_env):
        lock_env["AGENT_ID"] = "first-agent"
        run_script("lock-try.sh", ["contested.py"], lock_env)

        other_env = {**lock_env, "AGENT_ID": "second-agent"}
        run_script("lock-try.sh", ["contested.py"], other_env)

        result = run_script("lock-holder.sh", ["contested.py"], lock_env)
        assert result.stdout.strip() == "first-agent"


class TestPathHandling:
    """Test handling of various path formats."""

    def test_handles_nested_paths(self, lock_env):
        result = run_script("lock-try.sh", ["src/utils/helpers/deep.py"], lock_env)
        assert result.returncode == 0

    def test_nested_path_converted_correctly(self, lock_env):
        run_script("lock-try.sh", ["src/utils/helpers/deep.py"], lock_env)
        lock_path = Path(lock_env["LOCK_DIR"]) / "src_utils_helpers_deep.py.lock"
        assert lock_path.exists()

    def test_handles_absolute_paths(self, lock_env):
        result = run_script("lock-try.sh", ["/home/user/project/file.py"], lock_env)
        assert result.returncode == 0


class TestErrorHandling:
    """Test error handling for missing environment variables."""

    def test_fails_without_lock_dir(self, lock_env):
        env = {k: v for k, v in lock_env.items() if k != "LOCK_DIR"}
        result = run_script("lock-try.sh", ["test.py"], env)
        assert result.returncode == 2

    def test_fails_without_agent_id(self, lock_env):
        env = {k: v for k, v in lock_env.items() if k != "AGENT_ID"}
        result = run_script("lock-try.sh", ["test.py"], env)
        assert result.returncode == 2

    def test_fails_without_file_argument(self, lock_env):
        result = run_script("lock-try.sh", [], lock_env)
        assert result.returncode == 2

    def test_lock_holder_works_without_agent_id(self, lock_env):
        env = {k: v for k, v in lock_env.items() if k != "AGENT_ID"}
        result = run_script("lock-holder.sh", ["test.py"], env)
        assert result.returncode == 0


class TestAtomicity:
    """Test atomicity under concurrent access."""

    def test_exactly_one_racer_wins(self, lock_env):
        """Simulate race condition with parallel lock attempts."""

        def try_lock(agent_id: str) -> bool:
            env = {**lock_env, "AGENT_ID": agent_id}
            result = run_script("lock-try.sh", ["race.py"], env)
            return result.returncode == 0

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(try_lock, f"racer-{i}") for i in range(10)]
            results = [f.result() for f in futures]

        # Exactly one should have won
        assert sum(results) == 1

        # Lock file should exist with a valid holder
        lock_path = lock_file_path(lock_env["LOCK_DIR"], "race.py")
        assert lock_path.exists()
        holder = lock_path.read_text()
        assert holder.startswith("racer-")


class TestIdempotentOperations:
    """Test that operations are idempotent."""

    def test_release_nonexistent_lock_succeeds(self, lock_env):
        result = run_script("lock-release.sh", ["never-locked.py"], lock_env)
        assert result.returncode == 0

    def test_release_all_on_empty_dir_succeeds(self, lock_env):
        result = run_script("lock-release-all.sh", [], lock_env)
        assert result.returncode == 0

    def test_double_release_succeeds(self, lock_env):
        run_script("lock-try.sh", ["double-release.py"], lock_env)
        run_script("lock-release.sh", ["double-release.py"], lock_env)
        result = run_script("lock-release.sh", ["double-release.py"], lock_env)
        assert result.returncode == 0


class TestLockPersistence:
    """Test that locks persist across process boundaries."""

    def test_lock_persists_after_script_exit(self, lock_env):
        run_script("lock-try.sh", ["persist.py"], lock_env)

        # Check from a different process invocation
        result = run_script("lock-check.sh", ["persist.py"], lock_env)
        assert result.returncode == 0

    def test_lock_survives_holder_query(self, lock_env):
        run_script("lock-try.sh", ["persist.py"], lock_env)

        # Query holder multiple times
        for _ in range(3):
            result = run_script("lock-holder.sh", ["persist.py"], lock_env)
            assert result.stdout.strip() == lock_env["AGENT_ID"]

        # Lock should still be held
        result = run_script("lock-check.sh", ["persist.py"], lock_env)
        assert result.returncode == 0

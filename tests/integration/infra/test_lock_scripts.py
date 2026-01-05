"""Comprehensive test suite for mala lock scripts."""

import hashlib
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent.parent.parent / "src" / "scripts"


pytestmark = pytest.mark.integration


def _clean_os_environ() -> dict[str, str]:
    """Get os.environ without REPO_NAMESPACE for test isolation.

    Tests should use this instead of os.environ directly when building
    full_env dicts to avoid inheriting REPO_NAMESPACE from the outer
    environment (e.g., when tests run inside an agent).
    """
    return {k: v for k, v in os.environ.items() if k != "REPO_NAMESPACE"}


@pytest.fixture
def lock_env(tmp_path: Path) -> dict[str, str]:
    """Provide a clean lock environment for each test."""
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    return {
        "LOCK_DIR": str(lock_dir),
        "AGENT_ID": "test-agent",
        "PATH": f"{SCRIPTS_DIR}:{os.environ.get('PATH', '')}",
    }


def run_script(
    name: str, args: list[str], env: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run a lock script with the given environment.

    Note: This creates a clean environment that excludes LOCK_DIR, AGENT_ID, and
    REPO_NAMESPACE from os.environ to prevent test interference. The test must
    explicitly provide these in the env dict if needed.
    """
    script = SCRIPTS_DIR / name
    # Create clean environment: exclude lock-related vars from os.environ
    base_env = {
        k: v
        for k, v in os.environ.items()
        if k not in ("LOCK_DIR", "AGENT_ID", "REPO_NAMESPACE")
    }
    full_env = {**base_env, **env}
    return subprocess.run(
        [str(script), *args],
        env=full_env,
        capture_output=True,
        text=True,
    )


def lock_file_path(
    lock_dir: str, filepath: str, repo_namespace: str | None = None
) -> Path:
    """Get the lock file path for a given filepath using the new hash-based naming.

    Args:
        lock_dir: The lock directory path.
        filepath: The file path to lock.
        repo_namespace: Optional repo namespace for disambiguation.

    Note: This normalizes the filepath the same way the scripts do (resolving to
    absolute path) to ensure test assertions match actual lock file locations.
    """
    # Normalize the filepath to absolute path (mirroring script behavior)
    normalized_path = str(Path(filepath).resolve())

    # Build the canonical key
    if repo_namespace:
        key = f"{repo_namespace}:{normalized_path}"
    else:
        key = normalized_path

    # Hash the key
    key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
    lock_name = f"{key_hash}.lock"
    return Path(lock_dir) / lock_name


class TestLockKeyCollisionResistance:
    """Test that lock keys are collision-resistant."""

    def test_different_paths_same_underscore_pattern_no_collision(
        self, lock_env: dict[str, str]
    ) -> None:
        """Paths 'a/b' and 'a_b' should get different lock files (collision case)."""
        # Old behavior: both would map to 'a_b.lock'
        # New behavior: they should get different hashed lock files
        result1 = run_script("lock-try.sh", ["a/b"], lock_env)
        assert result1.returncode == 0

        # Use a different agent for the second lock
        env2 = {**lock_env, "AGENT_ID": "other-agent"}
        result2 = run_script("lock-try.sh", ["a_b"], env2)
        # Should succeed because they're different paths
        assert result2.returncode == 0, "a/b and a_b should not collide"

        # Both locks should exist
        locks = list(Path(lock_env["LOCK_DIR"]).glob("*.lock"))
        assert len(locks) == 2, f"Expected 2 lock files, got {len(locks)}"

    def test_nested_vs_flat_paths_no_collision(self, lock_env: dict[str, str]) -> None:
        """'src/utils' vs 'src_utils' should not collide."""
        result1 = run_script("lock-try.sh", ["src/utils"], lock_env)
        assert result1.returncode == 0

        env2 = {**lock_env, "AGENT_ID": "other-agent"}
        result2 = run_script("lock-try.sh", ["src_utils"], env2)
        assert result2.returncode == 0, "src/utils and src_utils should not collide"

    def test_multiple_slash_vs_underscore_patterns(
        self, lock_env: dict[str, str]
    ) -> None:
        """Complex collision patterns should all be avoided."""
        paths = ["a/b/c", "a_b/c", "a/b_c", "a_b_c"]
        agents = [f"agent-{i}" for i in range(len(paths))]

        for path, agent_id in zip(paths, agents, strict=True):
            env = {**lock_env, "AGENT_ID": agent_id}
            result = run_script("lock-try.sh", [path], env)
            assert result.returncode == 0, f"Failed to acquire lock for {path}"

        locks = list(Path(lock_env["LOCK_DIR"]).glob("*.lock"))
        assert len(locks) == len(paths), (
            f"Expected {len(paths)} distinct locks, got {len(locks)}"
        )


class TestRepoNamespacing:
    """Test that locks are namespaced by repository."""

    def test_same_relative_path_different_repos_no_collision(
        self, lock_env: dict[str, str]
    ) -> None:
        """Same file path in different repos should not collide."""
        # Simulate different repos by setting REPO_NAMESPACE
        env1 = {**lock_env, "REPO_NAMESPACE": "/home/user/repo1"}
        result1 = run_script("lock-try.sh", ["src/main.py"], env1)
        assert result1.returncode == 0

        env2 = {
            **lock_env,
            "AGENT_ID": "other-agent",
            "REPO_NAMESPACE": "/home/user/repo2",
        }
        result2 = run_script("lock-try.sh", ["src/main.py"], env2)
        assert result2.returncode == 0, (
            "Same path in different repos should not collide"
        )

        locks = list(Path(lock_env["LOCK_DIR"]).glob("*.lock"))
        assert len(locks) == 2

    def test_same_repo_same_path_does_collide(self, lock_env: dict[str, str]) -> None:
        """Same file path in same repo should collide (lock contention)."""
        env1 = {**lock_env, "REPO_NAMESPACE": "/home/user/repo1"}
        result1 = run_script("lock-try.sh", ["src/main.py"], env1)
        assert result1.returncode == 0

        env2 = {
            **lock_env,
            "AGENT_ID": "other-agent",
            "REPO_NAMESPACE": "/home/user/repo1",
        }
        result2 = run_script("lock-try.sh", ["src/main.py"], env2)
        assert result2.returncode == 1, "Same path in same repo should collide"


class TestBasicLockAcquisition:
    """Test basic lock acquisition functionality."""

    def test_creates_lock_file_with_correct_name(
        self, lock_env: dict[str, str]
    ) -> None:
        run_script("lock-try.sh", ["src/main.py"], lock_env)
        lock_path = lock_file_path(lock_env["LOCK_DIR"], "src/main.py")
        assert lock_path.exists()

    def test_lock_file_contains_agent_id(self, lock_env: dict[str, str]) -> None:
        run_script("lock-try.sh", ["src/main.py"], lock_env)
        lock_path = lock_file_path(lock_env["LOCK_DIR"], "src/main.py")
        assert lock_path.read_text().strip() == lock_env["AGENT_ID"]


class TestLockCheck:
    """Test lock ownership checking."""

    def test_returns_0_when_i_hold_lock(self, lock_env: dict[str, str]) -> None:
        run_script("lock-try.sh", ["file.py"], lock_env)
        result = run_script("lock-check.sh", ["file.py"], lock_env)
        assert result.returncode == 0

    def test_returns_1_when_another_agent_holds_lock(
        self, lock_env: dict[str, str]
    ) -> None:
        run_script("lock-try.sh", ["file.py"], lock_env)
        other_env = {**lock_env, "AGENT_ID": "other-agent"}
        result = run_script("lock-check.sh", ["file.py"], other_env)
        assert result.returncode == 1

    def test_returns_1_for_unlocked_file(self, lock_env: dict[str, str]) -> None:
        result = run_script("lock-check.sh", ["nonexistent.py"], lock_env)
        assert result.returncode == 1


class TestLockHolder:
    """Test lock holder identification."""

    def test_returns_correct_holder(self, lock_env: dict[str, str]) -> None:
        lock_env["AGENT_ID"] = "holder-agent"
        run_script("lock-try.sh", ["config.yaml"], lock_env)
        result = run_script("lock-holder.sh", ["config.yaml"], lock_env)
        assert result.returncode == 0
        assert result.stdout.strip() == "holder-agent"

    def test_returns_empty_for_unlocked_file(self, lock_env: dict[str, str]) -> None:
        result = run_script("lock-holder.sh", ["nonexistent.py"], lock_env)
        assert result.returncode == 0
        assert result.stdout.strip() == ""


class TestLockRelease:
    """Test lock release functionality."""

    def test_removes_lock_file(self, lock_env: dict[str, str]) -> None:
        run_script("lock-try.sh", ["release-test.py"], lock_env)
        run_script("lock-release.sh", ["release-test.py"], lock_env)
        lock_path = lock_file_path(lock_env["LOCK_DIR"], "release-test.py")
        assert not lock_path.exists()

    def test_can_reacquire_after_release(self, lock_env: dict[str, str]) -> None:
        run_script("lock-try.sh", ["release-test.py"], lock_env)
        run_script("lock-release.sh", ["release-test.py"], lock_env)
        result = run_script("lock-try.sh", ["release-test.py"], lock_env)
        assert result.returncode == 0

    def test_cannot_release_another_agents_lock(self, lock_env: dict[str, str]) -> None:
        run_script("lock-try.sh", ["other-lock.py"], lock_env)
        other_env = {**lock_env, "AGENT_ID": "other-agent"}
        run_script("lock-release.sh", ["other-lock.py"], other_env)
        lock_path = lock_file_path(lock_env["LOCK_DIR"], "other-lock.py")
        assert lock_path.exists()


class TestReleaseAllLocks:
    """Test releasing all locks held by an agent."""

    def test_removes_all_own_locks(self, lock_env: dict[str, str]) -> None:
        lock_env["AGENT_ID"] = "batch-agent"
        for f in ["file1.py", "file2.py", "file3.py"]:
            run_script("lock-try.sh", [f], lock_env)

        lock_count = len(list(Path(lock_env["LOCK_DIR"]).glob("*.lock")))
        assert lock_count == 3

        run_script("lock-release-all.sh", [], lock_env)
        lock_count = len(list(Path(lock_env["LOCK_DIR"]).glob("*.lock")))
        assert lock_count == 0

    def test_preserves_other_agents_locks(self, lock_env: dict[str, str]) -> None:
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

    def test_second_agent_cannot_acquire_held_lock(
        self, lock_env: dict[str, str]
    ) -> None:
        run_script("lock-try.sh", ["contested.py"], lock_env)
        other_env = {**lock_env, "AGENT_ID": "second-agent"}
        result = run_script("lock-try.sh", ["contested.py"], other_env)
        assert result.returncode == 1

    def test_original_holder_retains_lock(self, lock_env: dict[str, str]) -> None:
        lock_env["AGENT_ID"] = "first-agent"
        run_script("lock-try.sh", ["contested.py"], lock_env)

        other_env = {**lock_env, "AGENT_ID": "second-agent"}
        run_script("lock-try.sh", ["contested.py"], other_env)

        result = run_script("lock-holder.sh", ["contested.py"], lock_env)
        assert result.stdout.strip() == "first-agent"


class TestPathHandling:
    """Test handling of various path formats."""

    def test_handles_nested_paths(self, lock_env: dict[str, str]) -> None:
        result = run_script("lock-try.sh", ["src/utils/helpers/deep.py"], lock_env)
        assert result.returncode == 0

    def test_nested_path_creates_lock_file(self, lock_env: dict[str, str]) -> None:
        run_script("lock-try.sh", ["src/utils/helpers/deep.py"], lock_env)
        # With hash-based naming, just verify a lock file was created
        lock_path = lock_file_path(lock_env["LOCK_DIR"], "src/utils/helpers/deep.py")
        assert lock_path.exists()

    def test_handles_absolute_paths(self, lock_env: dict[str, str]) -> None:
        result = run_script("lock-try.sh", ["/home/user/project/file.py"], lock_env)
        assert result.returncode == 0


class TestErrorHandling:
    """Test error handling for missing environment variables."""

    def test_fails_without_lock_dir(self, lock_env: dict[str, str]) -> None:
        env = {k: v for k, v in lock_env.items() if k != "LOCK_DIR"}
        result = run_script("lock-try.sh", ["test.py"], env)
        assert result.returncode == 2

    def test_fails_without_agent_id(self, lock_env: dict[str, str]) -> None:
        env = {k: v for k, v in lock_env.items() if k != "AGENT_ID"}
        result = run_script("lock-try.sh", ["test.py"], env)
        assert result.returncode == 2

    def test_fails_without_file_argument(self, lock_env: dict[str, str]) -> None:
        result = run_script("lock-try.sh", [], lock_env)
        assert result.returncode == 2

    def test_lock_holder_works_without_agent_id(self, lock_env: dict[str, str]) -> None:
        env = {k: v for k, v in lock_env.items() if k != "AGENT_ID"}
        result = run_script("lock-holder.sh", ["test.py"], env)
        assert result.returncode == 0


class TestAtomicity:
    """Test atomicity under concurrent access."""

    def test_exactly_one_racer_wins(self, lock_env: dict[str, str]) -> None:
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

    def test_release_nonexistent_lock_succeeds(self, lock_env: dict[str, str]) -> None:
        result = run_script("lock-release.sh", ["never-locked.py"], lock_env)
        assert result.returncode == 0

    def test_release_all_on_empty_dir_succeeds(self, lock_env: dict[str, str]) -> None:
        result = run_script("lock-release-all.sh", [], lock_env)
        assert result.returncode == 0

    def test_double_release_succeeds(self, lock_env: dict[str, str]) -> None:
        run_script("lock-try.sh", ["double-release.py"], lock_env)
        run_script("lock-release.sh", ["double-release.py"], lock_env)
        result = run_script("lock-release.sh", ["double-release.py"], lock_env)
        assert result.returncode == 0


class TestLockPersistence:
    """Test that locks persist across process boundaries."""

    def test_lock_persists_after_script_exit(self, lock_env: dict[str, str]) -> None:
        run_script("lock-try.sh", ["persist.py"], lock_env)

        # Check from a different process invocation
        result = run_script("lock-check.sh", ["persist.py"], lock_env)
        assert result.returncode == 0

    def test_lock_survives_holder_query(self, lock_env: dict[str, str]) -> None:
        run_script("lock-try.sh", ["persist.py"], lock_env)

        # Query holder multiple times
        for _ in range(3):
            result = run_script("lock-holder.sh", ["persist.py"], lock_env)
            assert result.stdout.strip() == lock_env["AGENT_ID"]

        # Lock should still be held
        result = run_script("lock-check.sh", ["persist.py"], lock_env)
        assert result.returncode == 0


class TestPathNormalization:
    """Test that paths are normalized before hashing to ensure equivalent paths get the same lock."""

    def test_relative_and_absolute_paths_same_lock(
        self, lock_env: dict[str, str], tmp_path: Path
    ) -> None:
        """An absolute path and its relative equivalent should produce the same lock."""
        # Create a real file to work with
        test_file = tmp_path / "target.py"
        test_file.touch()

        abs_path = str(test_file)
        # Change to tmp_path to make relative path work
        env1 = {**lock_env, "PWD": str(tmp_path)}

        result1 = run_script("lock-try.sh", [abs_path], env1)
        assert result1.returncode == 0

        # Same agent trying the relative path should succeed (idempotent re-acquire)
        rel_path = "target.py"
        # We need to cd to tmp_path for relative to work - use a subshell
        full_env = {**_clean_os_environ(), **env1}
        result2 = subprocess.run(
            f"cd {tmp_path} && {SCRIPTS_DIR}/lock-try.sh {rel_path}",
            shell=True,
            env=full_env,
            capture_output=True,
            text=True,
        )
        # Should succeed because same agent can re-acquire (idempotent)
        assert result2.returncode == 0, (
            "Same agent should be able to re-acquire lock via relative path"
        )

    def test_dot_slash_prefix_same_as_without(
        self, lock_env: dict[str, str], tmp_path: Path
    ) -> None:
        """'./foo.py' and 'foo.py' should produce the same lock when in same directory."""
        test_file = tmp_path / "foo.py"
        test_file.touch()

        env = {**lock_env}
        full_env = {**_clean_os_environ(), **env}

        # Lock with ./foo.py
        result1 = subprocess.run(
            f"cd {tmp_path} && {SCRIPTS_DIR}/lock-try.sh ./foo.py",
            shell=True,
            env=full_env,
            capture_output=True,
            text=True,
        )
        assert result1.returncode == 0

        # Try foo.py (same file) - should succeed for same agent (idempotent)
        result2 = subprocess.run(
            f"cd {tmp_path} && {SCRIPTS_DIR}/lock-try.sh foo.py",
            shell=True,
            env=full_env,
            capture_output=True,
            text=True,
        )
        assert result2.returncode == 0, (
            "Same agent should be able to re-acquire lock via foo.py"
        )

    def test_dotdot_path_resolves_correctly(
        self, lock_env: dict[str, str], tmp_path: Path
    ) -> None:
        """Paths with .. components should resolve to canonical form."""
        # Create nested structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        test_file = tmp_path / "root.py"
        test_file.touch()

        env = {**lock_env}
        full_env = {**_clean_os_environ(), **env}

        # Lock from tmp_path using absolute path
        result1 = run_script("lock-try.sh", [str(test_file)], env)
        assert result1.returncode == 0

        # Try from subdir using ../root.py - should succeed for same agent (idempotent)
        result2 = subprocess.run(
            f"cd {subdir} && {SCRIPTS_DIR}/lock-try.sh ../root.py",
            shell=True,
            env=full_env,
            capture_output=True,
            text=True,
        )
        assert result2.returncode == 0, (
            "Same agent should be able to re-acquire lock via ../root.py"
        )

    def test_symlink_resolves_to_target(
        self, lock_env: dict[str, str], tmp_path: Path
    ) -> None:
        """Symlinks should resolve to the same lock as their target."""
        target = tmp_path / "real_file.py"
        target.touch()
        symlink = tmp_path / "link_to_file.py"
        symlink.symlink_to(target)

        env = {**lock_env}

        # Lock the real file
        result1 = run_script("lock-try.sh", [str(target)], env)
        assert result1.returncode == 0

        # Symlink should hit the same lock - same agent re-acquires (idempotent)
        result2 = run_script("lock-try.sh", [str(symlink)], env)
        assert result2.returncode == 0, (
            "Same agent should be able to re-acquire lock via symlink"
        )

    def test_namespace_applied_after_normalization(
        self, lock_env: dict[str, str], tmp_path: Path
    ) -> None:
        """REPO_NAMESPACE should be applied to normalized path, not raw path."""
        test_file = tmp_path / "file.py"
        test_file.touch()

        # Agent1 with namespace, absolute path
        env1 = {**lock_env, "REPO_NAMESPACE": "repo-A", "AGENT_ID": "agent-1"}
        result1 = run_script("lock-try.sh", [str(test_file)], env1)
        assert result1.returncode == 0

        # Agent2 same namespace, relative path (from tmp_path) - should collide
        env2 = {**lock_env, "REPO_NAMESPACE": "repo-A", "AGENT_ID": "agent-2"}
        full_env2 = {**_clean_os_environ(), **env2}
        result2 = subprocess.run(
            f"cd {tmp_path} && {SCRIPTS_DIR}/lock-try.sh file.py",
            shell=True,
            env=full_env2,
            capture_output=True,
            text=True,
        )
        assert result2.returncode == 1, (
            "Same namespace + normalized path should collide"
        )

    def test_different_namespaces_no_collision_after_normalization(
        self, lock_env: dict[str, str], tmp_path: Path
    ) -> None:
        """Different namespaces should not collide even with same normalized path."""
        test_file = tmp_path / "file.py"
        test_file.touch()

        env1 = {**lock_env, "REPO_NAMESPACE": "repo-A", "AGENT_ID": "agent-1"}
        result1 = run_script("lock-try.sh", [str(test_file)], env1)
        assert result1.returncode == 0

        env2 = {**lock_env, "REPO_NAMESPACE": "repo-B", "AGENT_ID": "agent-2"}
        result2 = run_script("lock-try.sh", [str(test_file)], env2)
        assert result2.returncode == 0, "Different namespaces should not collide"


class TestNormalizationAcrossAllScripts:
    """Ensure all lock scripts use consistent path normalization."""

    def test_lock_check_uses_normalized_path(
        self, lock_env: dict[str, str], tmp_path: Path
    ) -> None:
        """lock-check.sh should find locks regardless of path format."""
        test_file = tmp_path / "check_test.py"
        test_file.touch()

        env = {**lock_env}
        full_env = {**_clean_os_environ(), **env}

        # Lock with absolute path
        result1 = run_script("lock-try.sh", [str(test_file)], env)
        assert result1.returncode == 0

        # Check with relative path should succeed
        result2 = subprocess.run(
            f"cd {tmp_path} && {SCRIPTS_DIR}/lock-check.sh check_test.py",
            shell=True,
            env=full_env,
            capture_output=True,
            text=True,
        )
        assert result2.returncode == 0, "lock-check should find lock via relative path"

    def test_lock_holder_uses_normalized_path(
        self, lock_env: dict[str, str], tmp_path: Path
    ) -> None:
        """lock-holder.sh should return holder regardless of path format."""
        test_file = tmp_path / "holder_test.py"
        test_file.touch()

        env = {**lock_env, "AGENT_ID": "my-agent"}
        full_env = {**_clean_os_environ(), **env}

        # Lock with absolute path
        run_script("lock-try.sh", [str(test_file)], env)

        # Query holder with relative path
        result = subprocess.run(
            f"cd {tmp_path} && {SCRIPTS_DIR}/lock-holder.sh holder_test.py",
            shell=True,
            env=full_env,
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == "my-agent"

    def test_lock_release_uses_normalized_path(
        self, lock_env: dict[str, str], tmp_path: Path
    ) -> None:
        """lock-release.sh should release locks regardless of path format."""
        test_file = tmp_path / "release_test.py"
        test_file.touch()

        env = {**lock_env}
        full_env = {**_clean_os_environ(), **env}

        # Lock with absolute path
        run_script("lock-try.sh", [str(test_file)], env)

        # Release with relative path
        result = subprocess.run(
            f"cd {tmp_path} && {SCRIPTS_DIR}/lock-release.sh release_test.py",
            shell=True,
            env=full_env,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Verify lock is released by trying to acquire again
        result2 = run_script("lock-try.sh", [str(test_file)], env)
        assert result2.returncode == 0, "Should be able to reacquire after release"


class TestLockWait:
    """Test the lock-wait.sh script that waits for and acquires locks."""

    def test_acquires_unlocked_file_immediately(self, lock_env: dict[str, str]) -> None:
        """Should acquire an unlocked file immediately."""
        result = run_script("lock-wait.sh", ["immediate.py", "5", "50"], lock_env)
        assert result.returncode == 0

        # Verify lock was acquired
        lock_path = lock_file_path(lock_env["LOCK_DIR"], "immediate.py")
        assert lock_path.exists()
        assert lock_path.read_text().strip() == lock_env["AGENT_ID"]

    def test_times_out_when_lock_held(self, lock_env: dict[str, str]) -> None:
        """Should timeout when lock is held by another agent."""
        # First agent acquires lock
        env1 = {**lock_env, "AGENT_ID": "holder-agent"}
        run_script("lock-try.sh", ["contested.py"], env1)

        # Second agent tries to wait with short timeout
        env2 = {**lock_env, "AGENT_ID": "waiter-agent"}
        result = run_script("lock-wait.sh", ["contested.py", "1", "100"], env2)
        assert result.returncode == 1  # Timeout

        # Original holder still has lock
        lock_path = lock_file_path(lock_env["LOCK_DIR"], "contested.py")
        assert lock_path.read_text().strip() == "holder-agent"

    def test_acquires_after_release(self, lock_env: dict[str, str]) -> None:
        """Should acquire lock after it's released by another agent."""
        import threading
        import time

        env1 = {**lock_env, "AGENT_ID": "holder-agent"}
        env2 = {**lock_env, "AGENT_ID": "waiter-agent"}

        # First agent acquires lock
        run_script("lock-try.sh", ["released.py"], env1)

        acquired = threading.Event()

        def wait_for_lock() -> None:
            result = run_script("lock-wait.sh", ["released.py", "5", "50"], env2)
            if result.returncode == 0:
                acquired.set()

        # Start waiter in background
        waiter = threading.Thread(target=wait_for_lock)
        waiter.start()

        # Give waiter time to start polling
        time.sleep(0.2)

        # Release lock
        run_script("lock-release.sh", ["released.py"], env1)

        # Wait for waiter to finish
        waiter.join(timeout=5)
        assert acquired.is_set(), "Waiter should have acquired lock after release"

        # Verify waiter holds lock
        lock_path = lock_file_path(lock_env["LOCK_DIR"], "released.py")
        assert lock_path.read_text().strip() == "waiter-agent"

    def test_respects_repo_namespace(self, lock_env: dict[str, str]) -> None:
        """Should respect REPO_NAMESPACE for lock disambiguation."""
        env1 = {**lock_env, "AGENT_ID": "agent-1", "REPO_NAMESPACE": "repo-A"}
        env2 = {**lock_env, "AGENT_ID": "agent-2", "REPO_NAMESPACE": "repo-B"}

        # Both agents should be able to acquire locks for same path in different repos
        result1 = run_script("lock-wait.sh", ["namespaced.py", "2", "50"], env1)
        assert result1.returncode == 0

        result2 = run_script("lock-wait.sh", ["namespaced.py", "2", "50"], env2)
        assert result2.returncode == 0

    def test_fails_without_required_env(self, lock_env: dict[str, str]) -> None:
        """Should fail without LOCK_DIR or AGENT_ID."""
        env_no_lock_dir = {k: v for k, v in lock_env.items() if k != "LOCK_DIR"}
        result = run_script("lock-wait.sh", ["test.py"], env_no_lock_dir)
        assert result.returncode == 2

        env_no_agent = {k: v for k, v in lock_env.items() if k != "AGENT_ID"}
        result = run_script("lock-wait.sh", ["test.py"], env_no_agent)
        assert result.returncode == 2

    def test_fails_without_filepath(self, lock_env: dict[str, str]) -> None:
        """Should fail without filepath argument."""
        result = run_script("lock-wait.sh", [], lock_env)
        assert result.returncode == 2

    def test_normalizes_path(self, lock_env: dict[str, str], tmp_path: Path) -> None:
        """Should normalize paths consistently with other lock scripts."""
        test_file = tmp_path / "normalized.py"
        test_file.touch()

        env = {**lock_env}
        full_env = {**_clean_os_environ(), **env}

        # Acquire with absolute path using lock-wait
        result1 = run_script("lock-wait.sh", [str(test_file), "2", "50"], env)
        assert result1.returncode == 0

        # Try with relative path using lock-try - succeeds (idempotent re-acquire)
        result2 = subprocess.run(
            f"cd {tmp_path} && {SCRIPTS_DIR}/lock-try.sh normalized.py",
            shell=True,
            env=full_env,
            capture_output=True,
            text=True,
        )
        assert result2.returncode == 0, (
            "Same agent can re-acquire lock via relative path"
        )

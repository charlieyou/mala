"""Unit tests for canonical lock key generation.

Tests path normalization and repo namespace behavior in _lock_key/lock_path.
Ensures equivalent paths (absolute/relative, symlinks) produce identical keys.
"""

import hashlib
import os
from pathlib import Path

import pytest

from src.infra.tools.locking import _lock_key, lock_path


class TestLockKeyNormalization:
    """Test that _lock_key normalizes paths correctly."""

    def test_absolute_and_relative_same_key(self, tmp_path: Path) -> None:
        """Absolute and relative paths to the same file produce the same key."""
        # Create a real file in a temp "repo"
        repo = tmp_path / "repo"
        repo.mkdir()
        target_file = repo / "src" / "main.py"
        target_file.parent.mkdir(parents=True)
        target_file.touch()

        # Change to repo directory
        original_cwd = os.getcwd()
        try:
            os.chdir(repo)

            absolute_path = str(target_file.resolve())
            relative_path = "src/main.py"

            key_abs = _lock_key(absolute_path, repo_namespace=str(repo))
            key_rel = _lock_key(relative_path, repo_namespace=str(repo))

            assert key_abs == key_rel, (
                f"Absolute ({absolute_path}) and relative ({relative_path}) "
                f"paths should produce same key, got {key_abs!r} vs {key_rel!r}"
            )
        finally:
            os.chdir(original_cwd)

    def test_symlink_resolves_to_same_key(self, tmp_path: Path) -> None:
        """Symlink and real path produce the same key."""
        repo = tmp_path / "repo"
        repo.mkdir()

        # Create a real file
        real_file = repo / "real" / "module.py"
        real_file.parent.mkdir(parents=True)
        real_file.touch()

        # Create a symlink to it
        symlink = repo / "link_module.py"
        symlink.symlink_to(real_file)

        key_real = _lock_key(str(real_file), repo_namespace=str(repo))
        key_link = _lock_key(str(symlink), repo_namespace=str(repo))

        assert key_real == key_link, (
            f"Real path and symlink should produce same key, "
            f"got {key_real!r} vs {key_link!r}"
        )

    def test_dot_segments_normalized(self, tmp_path: Path) -> None:
        """Paths with . and .. segments are normalized."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "src").mkdir()
        (repo / "src" / "utils").mkdir()
        target = repo / "src" / "utils" / "helpers.py"
        target.touch()

        # Various equivalent paths with dot segments
        paths = [
            "src/utils/helpers.py",
            "src/./utils/helpers.py",
            "src/utils/../utils/helpers.py",
            "./src/utils/helpers.py",
        ]

        original_cwd = os.getcwd()
        try:
            os.chdir(repo)

            keys = [_lock_key(p, repo_namespace=str(repo)) for p in paths]

            # All keys should be identical
            assert len(set(keys)) == 1, (
                f"All path variants should produce same key: {list(zip(paths, keys))}"
            )
        finally:
            os.chdir(original_cwd)

    def test_repo_namespace_included_in_key(self, tmp_path: Path) -> None:
        """Repo namespace is always part of the canonical key."""
        filepath = "src/main.py"
        namespace = "/home/user/project"

        key = _lock_key(filepath, repo_namespace=namespace)

        # The key should incorporate the namespace
        # (exact format is implementation detail, but namespace must matter)
        key_no_namespace = _lock_key(filepath, repo_namespace=None)
        assert key != key_no_namespace, "Namespace should affect key"

    def test_different_repos_different_keys(self, tmp_path: Path) -> None:
        """Same relative path in different repos produces different keys."""
        filepath = "src/main.py"

        key_repo_a = _lock_key(filepath, repo_namespace="/repos/project-a")
        key_repo_b = _lock_key(filepath, repo_namespace="/repos/project-b")

        assert key_repo_a != key_repo_b, (
            "Same file in different repos should have different keys"
        )

    def test_trailing_slashes_normalized(self, tmp_path: Path) -> None:
        """Trailing slashes don't affect the key."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / "src").mkdir()
        target = repo / "src" / "main.py"
        target.touch()

        # With and without trailing elements that could affect normalization
        key1 = _lock_key("src/main.py", repo_namespace=str(repo))
        key2 = _lock_key("src//main.py", repo_namespace=str(repo))

        original_cwd = os.getcwd()
        try:
            os.chdir(repo)
            key3 = _lock_key("./src/main.py", repo_namespace=str(repo))
        finally:
            os.chdir(original_cwd)

        assert key1 == key2 == key3, "Path formatting should not affect key"


class TestLockPathNormalization:
    """Test that lock_path produces correct paths with normalized keys."""

    def test_absolute_and_relative_same_lock_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Absolute and relative paths should produce the same lock file path."""
        lock_dir = tmp_path / "locks"
        lock_dir.mkdir()
        monkeypatch.setenv("MALA_LOCK_DIR", str(lock_dir))

        repo = tmp_path / "repo"
        repo.mkdir()
        target = repo / "src" / "module.py"
        target.parent.mkdir(parents=True)
        target.touch()

        original_cwd = os.getcwd()
        try:
            os.chdir(repo)

            lock_abs = lock_path(str(target.resolve()), repo_namespace=str(repo))
            lock_rel = lock_path("src/module.py", repo_namespace=str(repo))

            assert lock_abs == lock_rel, (
                f"Same file should produce same lock path: {lock_abs} vs {lock_rel}"
            )
        finally:
            os.chdir(original_cwd)

    def test_lock_path_uses_hash_of_canonical_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Lock path should be derived from the canonical key's hash."""
        lock_dir = tmp_path / "locks"
        lock_dir.mkdir()
        monkeypatch.setenv("MALA_LOCK_DIR", str(lock_dir))

        filepath = "src/main.py"
        namespace = str(tmp_path)

        lock_p = lock_path(filepath, repo_namespace=namespace)
        key = _lock_key(filepath, repo_namespace=namespace)
        expected_hash = hashlib.sha256(key.encode()).hexdigest()[:16]

        assert lock_p.name == f"{expected_hash}.lock"


class TestLockKeyEdgeCases:
    """Edge cases for lock key generation."""

    def test_nonexistent_path_still_works(self, tmp_path: Path) -> None:
        """Lock key can be generated for paths that don't exist yet."""
        # This is important for locking files before creating them
        repo = tmp_path / "repo"
        repo.mkdir()

        key = _lock_key("new/file/that/does/not/exist.py", repo_namespace=str(repo))
        assert key is not None
        assert isinstance(key, str)
        assert len(key) > 0

    def test_empty_namespace_treated_as_none(self, tmp_path: Path) -> None:
        """Empty string namespace should behave like None."""
        filepath = "src/main.py"

        key_empty = _lock_key(filepath, repo_namespace="")
        key_none = _lock_key(filepath, repo_namespace=None)

        assert key_empty == key_none, "Empty namespace should behave like None"

    def test_consistent_across_calls(self, tmp_path: Path) -> None:
        """Same inputs should always produce same outputs."""
        repo = tmp_path / "repo"
        repo.mkdir()

        filepath = "src/module.py"
        namespace = str(repo)

        keys = [_lock_key(filepath, repo_namespace=namespace) for _ in range(10)]
        assert len(set(keys)) == 1, "Key generation must be deterministic"

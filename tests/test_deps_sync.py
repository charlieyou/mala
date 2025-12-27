"""Unit tests for src/validation/deps_sync.py - dependency sync optimization."""

from pathlib import Path

from src.validation.deps_sync import (
    DEPS_FILES,
    STATE_FILE,
    DepsSyncState,
    _compute_deps_hash,
    _compute_file_hash,
)


class TestComputeFileHash:
    """Test the _compute_file_hash helper function."""

    def test_existing_file(self, tmp_path: Path) -> None:
        """Hash is computed for existing files."""
        file = tmp_path / "test.txt"
        file.write_text("hello world")
        result = _compute_file_hash(file)
        assert result is not None
        assert len(result) == 64  # SHA-256 hex digest length

    def test_missing_file(self, tmp_path: Path) -> None:
        """Returns None for missing files."""
        file = tmp_path / "missing.txt"
        result = _compute_file_hash(file)
        assert result is None

    def test_same_content_same_hash(self, tmp_path: Path) -> None:
        """Same content produces same hash."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        content = "identical content"
        file1.write_text(content)
        file2.write_text(content)
        assert _compute_file_hash(file1) == _compute_file_hash(file2)

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        """Different content produces different hash."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        file1.write_text("content A")
        file2.write_text("content B")
        assert _compute_file_hash(file1) != _compute_file_hash(file2)


class TestComputeDepsHash:
    """Test the _compute_deps_hash helper function."""

    def test_missing_deps_file(self, tmp_path: Path) -> None:
        """Returns special hash when deps files are missing."""
        result = _compute_deps_hash(tmp_path)
        assert result == "missing-deps-file"

    def test_partial_deps_files(self, tmp_path: Path) -> None:
        """Returns special hash when only some deps files exist."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")
        # uv.lock is missing
        result = _compute_deps_hash(tmp_path)
        assert result == "missing-deps-file"

    def test_all_deps_files_present(self, tmp_path: Path) -> None:
        """Computes hash when all deps files exist."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")
        (tmp_path / "uv.lock").write_text("version = 1")
        result = _compute_deps_hash(tmp_path)
        assert result != "missing-deps-file"
        assert len(result) == 64  # SHA-256 hex digest

    def test_hash_changes_when_file_changes(self, tmp_path: Path) -> None:
        """Hash changes when deps file content changes."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")
        (tmp_path / "uv.lock").write_text("version = 1")
        hash1 = _compute_deps_hash(tmp_path)

        (tmp_path / "uv.lock").write_text("version = 2")
        hash2 = _compute_deps_hash(tmp_path)

        assert hash1 != hash2


class TestDepsSyncState:
    """Test the DepsSyncState class."""

    def test_init(self, tmp_path: Path) -> None:
        """DepsSyncState can be initialized."""
        state = DepsSyncState(tmp_path)
        assert state.cwd == tmp_path
        assert state.force is False

    def test_init_with_force(self, tmp_path: Path) -> None:
        """DepsSyncState can be initialized with force=True."""
        state = DepsSyncState(tmp_path, force=True)
        assert state.force is True

    def test_needs_sync_first_run(self, tmp_path: Path) -> None:
        """needs_sync returns True on first run (no prior state)."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")
        (tmp_path / "uv.lock").write_text("version = 1")
        state = DepsSyncState(tmp_path)
        assert state.needs_sync() is True

    def test_needs_sync_after_mark_synced(self, tmp_path: Path) -> None:
        """needs_sync returns False after mark_synced."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")
        (tmp_path / "uv.lock").write_text("version = 1")
        state = DepsSyncState(tmp_path)

        assert state.needs_sync() is True
        state.mark_synced()
        assert state.needs_sync() is False

    def test_needs_sync_after_file_change(self, tmp_path: Path) -> None:
        """needs_sync returns True after deps file changes."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")
        (tmp_path / "uv.lock").write_text("version = 1")
        state = DepsSyncState(tmp_path)

        state.mark_synced()
        assert state.needs_sync() is False

        # Modify uv.lock
        (tmp_path / "uv.lock").write_text("version = 2")
        assert state.needs_sync() is True

    def test_needs_sync_force_always_true(self, tmp_path: Path) -> None:
        """needs_sync returns True when force=True, even after mark_synced."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")
        (tmp_path / "uv.lock").write_text("version = 1")
        state = DepsSyncState(tmp_path, force=True)

        state.mark_synced()
        # Force always requires sync
        assert state.needs_sync() is True

    def test_mark_synced_creates_state_dir(self, tmp_path: Path) -> None:
        """mark_synced creates .uv directory if it doesn't exist."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")
        (tmp_path / "uv.lock").write_text("version = 1")
        state = DepsSyncState(tmp_path)

        assert not (tmp_path / ".uv").exists()
        state.mark_synced()
        assert (tmp_path / ".uv").exists()
        assert (tmp_path / STATE_FILE).exists()

    def test_clear_removes_state(self, tmp_path: Path) -> None:
        """clear removes the state file."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")
        (tmp_path / "uv.lock").write_text("version = 1")
        state = DepsSyncState(tmp_path)

        state.mark_synced()
        assert state.needs_sync() is False

        state.clear()
        assert state.needs_sync() is True

    def test_clear_nonexistent_state(self, tmp_path: Path) -> None:
        """clear succeeds even when no state file exists."""
        state = DepsSyncState(tmp_path)
        result = state.clear()
        assert result is True

    def test_state_persists_across_instances(self, tmp_path: Path) -> None:
        """State persists between different DepsSyncState instances."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")
        (tmp_path / "uv.lock").write_text("version = 1")

        # First instance marks synced
        state1 = DepsSyncState(tmp_path)
        state1.mark_synced()

        # Second instance should see that sync is not needed
        state2 = DepsSyncState(tmp_path)
        assert state2.needs_sync() is False

    def test_needs_sync_missing_deps_files(self, tmp_path: Path) -> None:
        """needs_sync returns True when deps files are missing."""
        # Even with prior state, if deps files disappear, need sync
        state = DepsSyncState(tmp_path)
        assert state.needs_sync() is True


class TestDepsSyncStateEdgeCases:
    """Test edge cases for DepsSyncState."""

    def test_pyproject_changes_trigger_sync(self, tmp_path: Path) -> None:
        """Changes to pyproject.toml trigger sync."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'v1'")
        (tmp_path / "uv.lock").write_text("version = 1")
        state = DepsSyncState(tmp_path)

        state.mark_synced()
        assert state.needs_sync() is False

        # Modify pyproject.toml
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'v2'")
        assert state.needs_sync() is True

    def test_empty_files(self, tmp_path: Path) -> None:
        """Empty deps files are handled correctly."""
        (tmp_path / "pyproject.toml").write_text("")
        (tmp_path / "uv.lock").write_text("")
        state = DepsSyncState(tmp_path)

        assert state.needs_sync() is True
        state.mark_synced()
        assert state.needs_sync() is False

    def test_deps_files_constant(self) -> None:
        """DEPS_FILES constant has expected files."""
        assert "pyproject.toml" in DEPS_FILES
        assert "uv.lock" in DEPS_FILES
        assert len(DEPS_FILES) == 2

    def test_state_file_constant(self) -> None:
        """STATE_FILE is in .uv directory."""
        assert STATE_FILE.startswith(".uv/")

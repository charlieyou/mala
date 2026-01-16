"""Unit tests for PreCompact hook archive logic."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.infra.hooks.precompact import make_precompact_hook


@pytest.fixture
def runs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Set up a temporary runs directory for tests."""
    runs_path = tmp_path / "runs"
    runs_path.mkdir()
    # Patch get_repo_runs_dir to return our temp path
    monkeypatch.setattr(
        "src.infra.hooks.precompact.get_repo_runs_dir",
        lambda _: runs_path,
    )
    return runs_path


class TestPreCompactHookArchivesTranscript:
    """Tests for basic transcript archiving functionality."""

    @pytest.mark.asyncio
    async def test_precompact_hook_archives_transcript(
        self, tmp_path: Path, runs_dir: Path
    ) -> None:
        """Hook should copy transcript to archive directory."""
        # Create source transcript
        transcript = tmp_path / "transcript.json"
        transcript.write_text('{"messages": []}')

        hook = make_precompact_hook(tmp_path)
        hook_input = {"transcript_path": str(transcript), "session_id": "sess123"}

        result = await hook(hook_input, None, None)

        # Hook should return empty dict (allow compaction)
        assert result == {}

        # Verify archive was created
        archive_dir = runs_dir / "archives"
        assert archive_dir.exists()
        archives = list(archive_dir.glob("*.json"))
        assert len(archives) == 1

        # Verify archive content matches source
        archive = archives[0]
        assert archive.read_text() == '{"messages": []}'

        # Verify filename format: {session_id}_{timestamp}_transcript{ext}
        assert archive.name.startswith("sess123_")
        assert "_transcript.json" in archive.name


class TestPreCompactHookCreatesArchiveDir:
    """Tests for archive directory creation."""

    @pytest.mark.asyncio
    async def test_precompact_hook_creates_archive_dir(
        self, tmp_path: Path, runs_dir: Path
    ) -> None:
        """Hook should create archive directory if it doesn't exist."""
        # Ensure archive dir doesn't exist
        archive_dir = runs_dir / "archives"
        assert not archive_dir.exists()

        transcript = tmp_path / "transcript.json"
        transcript.write_text("{}")

        hook = make_precompact_hook(tmp_path)
        await hook({"transcript_path": str(transcript)}, None, None)

        # Archive dir should now exist with proper permissions
        assert archive_dir.exists()
        assert archive_dir.is_dir()


class TestPreCompactHookHandlesMissingTranscript:
    """Tests for handling missing transcript files."""

    @pytest.mark.asyncio
    async def test_precompact_hook_handles_missing_transcript(
        self, tmp_path: Path, runs_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Hook should log warning and return {} for missing transcript."""
        hook = make_precompact_hook(tmp_path)
        hook_input = {"transcript_path": str(tmp_path / "nonexistent.json")}

        result = await hook(hook_input, None, None)

        assert result == {}
        assert "not found" in caplog.text.lower()


class TestPreCompactHookHandlesMissingPath:
    """Tests for handling missing transcript_path in hook input."""

    @pytest.mark.asyncio
    async def test_precompact_hook_handles_missing_path(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Hook should log warning and return {} when transcript_path is missing."""
        hook = make_precompact_hook(tmp_path)

        # Empty input
        result = await hook({}, None, None)
        assert result == {}
        assert "missing transcript_path" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_precompact_hook_handles_none_path(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Hook should handle None transcript_path gracefully."""
        hook = make_precompact_hook(tmp_path)

        result = await hook({"transcript_path": None}, None, None)
        assert result == {}

    @pytest.mark.asyncio
    async def test_precompact_hook_handles_invalid_type(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Hook should handle invalid transcript_path types without crashing."""
        hook = make_precompact_hook(tmp_path)

        # Pass an integer - should not raise TypeError, should return {}
        result = await hook({"transcript_path": 12345}, None, None)
        assert result == {}
        assert "invalid transcript_path type" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_precompact_hook_accepts_path_object(
        self, tmp_path: Path, runs_dir: Path
    ) -> None:
        """Hook should accept Path objects as transcript_path."""
        transcript = tmp_path / "transcript.json"
        transcript.write_text("{}")

        hook = make_precompact_hook(tmp_path)

        # Pass Path object directly - should work
        result = await hook({"transcript_path": transcript}, None, None)
        assert result == {}

        # Verify archive was created
        archive_dir = runs_dir / "archives"
        archives = list(archive_dir.glob("*.json"))
        assert len(archives) == 1


class TestPreCompactHookHandlesIOError:
    """Tests for handling I/O errors during archiving."""

    @pytest.mark.asyncio
    async def test_precompact_hook_handles_io_error(
        self, tmp_path: Path, runs_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Hook should log error and return {} on I/O failure."""
        transcript = tmp_path / "transcript.json"
        transcript.write_text("{}")

        hook = make_precompact_hook(tmp_path)

        # Mock shutil.copy2 to raise OSError
        with patch("src.infra.hooks.precompact.shutil.copy2") as mock_copy:
            mock_copy.side_effect = OSError("Disk full")
            result = await hook({"transcript_path": str(transcript)}, None, None)

        assert result == {}
        assert "failed to archive" in caplog.text.lower()

    @pytest.mark.asyncio
    async def test_precompact_hook_handles_exists_oserror(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Hook should catch OSError from exists() (e.g., permission denied)."""
        hook = make_precompact_hook(tmp_path)

        # Mock Path.exists to raise OSError (permission denied scenario)
        with patch("src.infra.hooks.precompact.Path.exists") as mock_exists:
            mock_exists.side_effect = OSError("Permission denied")
            result = await hook({"transcript_path": "/root/secret.json"}, None, None)

        assert result == {}
        assert "failed to archive" in caplog.text.lower()


class TestPreCompactHookTimestampUniqueness:
    """Tests for timestamp uniqueness in archive filenames."""

    @pytest.mark.asyncio
    async def test_precompact_hook_timestamp_uniqueness(
        self, tmp_path: Path, runs_dir: Path
    ) -> None:
        """Multiple archives in quick succession should have unique timestamps."""
        transcript = tmp_path / "transcript.json"
        transcript.write_text("{}")

        hook = make_precompact_hook(tmp_path)

        # Archive twice with NO delay - microsecond precision should ensure uniqueness
        await hook(
            {"transcript_path": str(transcript), "session_id": "sess"}, None, None
        )
        await hook(
            {"transcript_path": str(transcript), "session_id": "sess"}, None, None
        )

        archive_dir = runs_dir / "archives"
        archives = list(archive_dir.glob("*.json"))
        assert len(archives) == 2

        # Filenames should be different even without delay
        names = [a.name for a in archives]
        assert len(set(names)) == 2


class TestPreCompactHookFilePermissions:
    """Tests for file and directory permissions."""

    @pytest.mark.asyncio
    async def test_precompact_hook_file_permissions(
        self, tmp_path: Path, runs_dir: Path
    ) -> None:
        """Archive file should have 0600 permissions, dir should have 0700."""
        transcript = tmp_path / "transcript.json"
        transcript.write_text("{}")

        hook = make_precompact_hook(tmp_path)
        await hook({"transcript_path": str(transcript)}, None, None)

        archive_dir = runs_dir / "archives"
        archives = list(archive_dir.glob("*.json"))
        assert len(archives) == 1

        # Check directory permissions (0700)
        dir_mode = archive_dir.stat().st_mode & 0o777
        assert dir_mode == 0o700

        # Check file permissions (0600)
        file_mode = archives[0].stat().st_mode & 0o777
        assert file_mode == 0o600


class TestPreCompactHookPreservesExtension:
    """Tests for preserving transcript file extension."""

    @pytest.mark.asyncio
    async def test_precompact_hook_preserves_extension(
        self, tmp_path: Path, runs_dir: Path
    ) -> None:
        """Archive should preserve the original file extension."""
        # Test .json extension
        transcript_json = tmp_path / "transcript.json"
        transcript_json.write_text("{}")

        hook = make_precompact_hook(tmp_path)
        await hook({"transcript_path": str(transcript_json)}, None, None)

        archive_dir = runs_dir / "archives"
        archives = list(archive_dir.glob("*.json"))
        assert len(archives) == 1
        assert archives[0].suffix == ".json"

    @pytest.mark.asyncio
    async def test_precompact_hook_preserves_jsonl_extension(
        self, tmp_path: Path, runs_dir: Path
    ) -> None:
        """Archive should preserve .jsonl extension."""
        transcript = tmp_path / "transcript.jsonl"
        transcript.write_text("{}\n{}")

        hook = make_precompact_hook(tmp_path)
        await hook({"transcript_path": str(transcript)}, None, None)

        archive_dir = runs_dir / "archives"
        archives = list(archive_dir.glob("*.jsonl"))
        assert len(archives) == 1
        assert archives[0].suffix == ".jsonl"

    @pytest.mark.asyncio
    async def test_precompact_hook_handles_no_extension(
        self, tmp_path: Path, runs_dir: Path
    ) -> None:
        """Archive should work with files that have no extension."""
        transcript = tmp_path / "transcript"
        transcript.write_text("{}")

        hook = make_precompact_hook(tmp_path)
        await hook({"transcript_path": str(transcript)}, None, None)

        archive_dir = runs_dir / "archives"
        archives = list(archive_dir.iterdir())
        assert len(archives) == 1
        # Should end with _transcript (no extension)
        assert "_transcript" in archives[0].name
        assert archives[0].suffix == ""


class TestPreCompactHookWithObjectInput:
    """Tests for handling object-style hook input (SDK types)."""

    @pytest.mark.asyncio
    async def test_precompact_hook_handles_object_input(
        self, tmp_path: Path, runs_dir: Path
    ) -> None:
        """Hook should handle object-style input with transcript_path attribute."""
        transcript = tmp_path / "transcript.json"
        transcript.write_text("{}")

        # Create object-like input with transcript_path attribute
        class MockHookInput:
            transcript_path = str(transcript)

        hook = make_precompact_hook(tmp_path)
        result = await hook(MockHookInput(), None, None)

        assert result == {}

        archive_dir = runs_dir / "archives"
        archives = list(archive_dir.glob("*.json"))
        assert len(archives) == 1


class TestPreCompactHookLogsArchiveSize:
    """Tests for archive size logging."""

    @pytest.mark.asyncio
    async def test_precompact_hook_logs_archive_size(
        self, tmp_path: Path, runs_dir: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Hook should log archive size in KB at INFO level."""
        import logging

        caplog.set_level(logging.INFO)

        # Create a transcript with known size
        transcript = tmp_path / "transcript.json"
        content = "x" * 2048  # 2 KB
        transcript.write_text(content)

        hook = make_precompact_hook(tmp_path)
        await hook({"transcript_path": str(transcript)}, None, None)

        # Should log size in KB
        assert "archived transcript" in caplog.text.lower()
        assert "kb" in caplog.text.lower()

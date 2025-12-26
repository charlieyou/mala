"""Unit tests for failure handoff writer.

Tests the handoff module that writes .mala/handoff/<issue_id>.md files
when agents fail, containing last commands and error summary.
"""

import json
from pathlib import Path

from src.handoff import HandoffWriter, sanitize_issue_id


class TestSanitizeIssueId:
    """Test issue ID sanitization for safe file names."""

    def test_simple_issue_id(self):
        """Simple alphanumeric + hyphen IDs should pass through."""
        assert sanitize_issue_id("mala-123") == "mala-123"

    def test_removes_path_separators(self):
        """Path separators should be removed to prevent path traversal."""
        assert sanitize_issue_id("../etc/passwd") == "etcpasswd"
        assert sanitize_issue_id("foo/bar") == "foobar"
        assert sanitize_issue_id("foo\\bar") == "foobar"

    def test_removes_special_characters(self):
        """Special characters should be removed."""
        assert sanitize_issue_id('issue<>:"?*|') == "issue"

    def test_handles_empty_string(self):
        """Empty string should return a placeholder."""
        assert sanitize_issue_id("") == "unknown"

    def test_handles_only_special_chars(self):
        """String with only special chars should return placeholder."""
        assert sanitize_issue_id("../..") == "unknown"

    def test_preserves_underscores(self):
        """Underscores should be preserved."""
        assert sanitize_issue_id("issue_123") == "issue_123"


class TestHandoffWriter:
    """Test handoff file writing."""

    def test_creates_handoff_directory(self, tmp_path: Path):
        """Should create .mala/handoff directory if it doesn't exist."""
        writer = HandoffWriter(tmp_path)
        writer.write_handoff("test-issue", error_summary="Test error")

        handoff_dir = tmp_path / ".mala" / "handoff"
        assert handoff_dir.exists()
        assert handoff_dir.is_dir()

    def test_writes_handoff_file(self, tmp_path: Path):
        """Should write handoff file with issue ID in name."""
        writer = HandoffWriter(tmp_path)
        writer.write_handoff("test-issue", error_summary="Test error")

        handoff_file = tmp_path / ".mala" / "handoff" / "test-issue.md"
        assert handoff_file.exists()

    def test_contains_error_summary(self, tmp_path: Path):
        """Handoff file should contain the error summary."""
        writer = HandoffWriter(tmp_path)
        writer.write_handoff("test-issue", error_summary="Connection timeout after 30s")

        handoff_file = tmp_path / ".mala" / "handoff" / "test-issue.md"
        content = handoff_file.read_text()
        assert "Connection timeout after 30s" in content

    def test_contains_last_commands_from_log(self, tmp_path: Path):
        """Should extract last commands from JSONL log."""
        # Create sample JSONL log with commands
        log_path = tmp_path / "session.jsonl"
        log_entries = [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "uv run pytest tests/"},
                        }
                    ]
                },
            },
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "git status"},
                        }
                    ]
                },
            },
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in log_entries) + "\n")

        writer = HandoffWriter(tmp_path)
        writer.write_handoff(
            "test-issue", log_path=log_path, error_summary="Test failed"
        )

        handoff_file = tmp_path / ".mala" / "handoff" / "test-issue.md"
        content = handoff_file.read_text()
        assert "uv run pytest tests/" in content
        assert "git status" in content

    def test_handles_no_log_path(self, tmp_path: Path):
        """Should handle case where no log path is provided."""
        writer = HandoffWriter(tmp_path)
        writer.write_handoff("test-issue", error_summary="Test error")

        handoff_file = tmp_path / ".mala" / "handoff" / "test-issue.md"
        content = handoff_file.read_text()
        assert "No command history available" in content

    def test_handles_missing_log_file(self, tmp_path: Path):
        """Should handle case where log file doesn't exist."""
        writer = HandoffWriter(tmp_path)
        writer.write_handoff(
            "test-issue",
            log_path=tmp_path / "nonexistent.jsonl",
            error_summary="Test error",
        )

        handoff_file = tmp_path / ".mala" / "handoff" / "test-issue.md"
        content = handoff_file.read_text()
        assert "No command history available" in content

    def test_handles_empty_log_file(self, tmp_path: Path):
        """Should handle empty log file gracefully."""
        log_path = tmp_path / "empty.jsonl"
        log_path.write_text("")

        writer = HandoffWriter(tmp_path)
        writer.write_handoff(
            "test-issue", log_path=log_path, error_summary="Test error"
        )

        handoff_file = tmp_path / ".mala" / "handoff" / "test-issue.md"
        content = handoff_file.read_text()
        assert "No command history available" in content

    def test_includes_issue_id_in_header(self, tmp_path: Path):
        """Handoff file should include issue ID in header."""
        writer = HandoffWriter(tmp_path)
        writer.write_handoff("mala-xyz", error_summary="Test error")

        handoff_file = tmp_path / ".mala" / "handoff" / "mala-xyz.md"
        content = handoff_file.read_text()
        assert "mala-xyz" in content
        assert content.startswith("# Handoff:")

    def test_extracts_bash_tool_calls_only(self, tmp_path: Path):
        """Should only extract Bash tool calls, not other tools."""
        log_path = tmp_path / "session.jsonl"
        log_entries = [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Read",
                            "input": {"path": "/some/file.py"},
                        }
                    ]
                },
            },
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "make test"},
                        }
                    ]
                },
            },
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in log_entries) + "\n")

        writer = HandoffWriter(tmp_path)
        writer.write_handoff("test-issue", log_path=log_path, error_summary="Error")

        handoff_file = tmp_path / ".mala" / "handoff" / "test-issue.md"
        content = handoff_file.read_text()
        assert "make test" in content
        assert "/some/file.py" not in content

    def test_limits_command_history(self, tmp_path: Path):
        """Should limit command history to last N commands."""
        log_path = tmp_path / "session.jsonl"
        # Create 20 commands
        log_entries = [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": f"command-{i}"},
                        }
                    ]
                },
            }
            for i in range(20)
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in log_entries) + "\n")

        writer = HandoffWriter(tmp_path)
        writer.write_handoff("test-issue", log_path=log_path, error_summary="Error")

        handoff_file = tmp_path / ".mala" / "handoff" / "test-issue.md"
        content = handoff_file.read_text()
        # Should have last 10 commands (default limit)
        assert "command-19" in content
        assert "command-10" in content
        # Should not have earliest commands
        assert "command-0" not in content

    def test_sanitizes_issue_id_in_filename(self, tmp_path: Path):
        """Should sanitize issue ID for safe filenames."""
        writer = HandoffWriter(tmp_path)
        writer.write_handoff("../dangerous/path", error_summary="Test error")

        # Should create safe filename
        handoff_dir = tmp_path / ".mala" / "handoff"
        files = list(handoff_dir.glob("*.md"))
        assert len(files) == 1
        # Filename should not contain path separators
        assert "/" not in files[0].name
        assert "\\" not in files[0].name

    def test_includes_timestamp(self, tmp_path: Path):
        """Handoff file should include creation timestamp."""
        writer = HandoffWriter(tmp_path)
        writer.write_handoff("test-issue", error_summary="Error")

        handoff_file = tmp_path / ".mala" / "handoff" / "test-issue.md"
        content = handoff_file.read_text()
        # Should have some date-like content
        assert "20" in content  # Year prefix

    def test_overwrites_existing_handoff(self, tmp_path: Path):
        """Should overwrite existing handoff file for same issue."""
        handoff_dir = tmp_path / ".mala" / "handoff"
        handoff_dir.mkdir(parents=True)
        handoff_file = handoff_dir / "test-issue.md"
        handoff_file.write_text("old content")

        writer = HandoffWriter(tmp_path)
        writer.write_handoff("test-issue", error_summary="New error")

        content = handoff_file.read_text()
        assert "New error" in content
        assert "old content" not in content


class TestHandoffWriterToolResults:
    """Test extraction of tool results and errors from JSONL logs."""

    def test_includes_last_tool_result_error(self, tmp_path: Path):
        """Should include error from last tool result if present."""
        log_path = tmp_path / "session.jsonl"
        log_entries = [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "uv run pytest"},
                        }
                    ]
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "is_error": True,
                            "content": "FAILED tests/test_foo.py::test_bar - AssertionError",
                        }
                    ]
                },
            },
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in log_entries) + "\n")

        writer = HandoffWriter(tmp_path)
        writer.write_handoff(
            "test-issue", log_path=log_path, error_summary="Tests failed"
        )

        handoff_file = tmp_path / ".mala" / "handoff" / "test-issue.md"
        content = handoff_file.read_text()
        assert "AssertionError" in content

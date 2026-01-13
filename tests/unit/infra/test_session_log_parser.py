"""Unit tests for SessionLogParser.

Tests cover:
- JSONL iteration with byte offset tracking
- Bash command extraction from assistant messages
- Tool result extraction from user messages
- Text block extraction from assistant messages
- Integration with log_events typed parsing
"""

import json
from pathlib import Path

import pytest
from src.infra.io.session_log_parser import JsonlEntry, SessionLogParser


class TestIterJsonlEntries:
    """Test iter_jsonl_entries for JSONL iteration with offset tracking."""

    def test_iterates_over_jsonl_lines(self, tmp_path: Path) -> None:
        """Should yield JsonlEntry for each valid JSONL line."""
        log_path = tmp_path / "session.jsonl"
        entries = [
            {"type": "assistant", "message": {"content": []}},
            {"type": "user", "message": {"content": []}},
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        parser = SessionLogParser()
        results = list(parser.iter_jsonl_entries(log_path))

        assert len(results) == 2
        assert all(isinstance(r, JsonlEntry) for r in results)

    def test_tracks_byte_offsets(self, tmp_path: Path) -> None:
        """Should track byte offsets for each entry."""
        log_path = tmp_path / "session.jsonl"
        first_entry = json.dumps({"type": "assistant", "message": {"content": []}})
        second_entry = json.dumps({"type": "user", "message": {"content": []}})
        log_path.write_text(first_entry + "\n" + second_entry + "\n")

        parser = SessionLogParser()
        results = list(parser.iter_jsonl_entries(log_path))

        assert results[0].offset == 0
        assert results[0].line_len == len(first_entry) + 1  # +1 for newline
        assert results[1].offset == len(first_entry) + 1

    def test_starts_from_given_offset(self, tmp_path: Path) -> None:
        """Should start iteration from the given byte offset."""
        log_path = tmp_path / "session.jsonl"
        first_entry = json.dumps({"type": "assistant", "message": {"content": []}})
        second_entry = json.dumps({"type": "user", "message": {"content": []}})
        log_path.write_text(first_entry + "\n" + second_entry + "\n")

        parser = SessionLogParser()
        offset = len(first_entry) + 1
        results = list(parser.iter_jsonl_entries(log_path, offset=offset))

        assert len(results) == 1
        assert results[0].data["type"] == "user"

    def test_handles_missing_file(self, tmp_path: Path) -> None:
        """Should handle missing log file gracefully."""
        parser = SessionLogParser()
        nonexistent = tmp_path / "nonexistent.jsonl"

        results = list(parser.iter_jsonl_entries(nonexistent))

        assert results == []

    def test_skips_invalid_json_lines(self, tmp_path: Path) -> None:
        """Should skip lines that fail JSON parsing."""
        log_path = tmp_path / "session.jsonl"
        valid_entry = json.dumps({"type": "assistant", "message": {"content": []}})
        log_path.write_text(valid_entry + "\n" + "invalid json\n")

        parser = SessionLogParser()
        results = list(parser.iter_jsonl_entries(log_path))

        assert len(results) == 1

    def test_skips_empty_lines(self, tmp_path: Path) -> None:
        """Should skip empty lines."""
        log_path = tmp_path / "session.jsonl"
        entry = json.dumps({"type": "assistant", "message": {"content": []}})
        log_path.write_text(entry + "\n\n\n")

        parser = SessionLogParser()
        results = list(parser.iter_jsonl_entries(log_path))

        assert len(results) == 1

    def test_parses_typed_log_entry(self, tmp_path: Path) -> None:
        """Should parse into typed LogEntry when structure matches schema."""
        log_path = tmp_path / "session.jsonl"
        entry = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Hello"}]},
        }
        log_path.write_text(json.dumps(entry) + "\n")

        parser = SessionLogParser()
        results = list(parser.iter_jsonl_entries(log_path))

        assert len(results) == 1
        assert results[0].entry is not None
        # entry is AssistantLogEntry with content


class TestGetLogEndOffset:
    """Test get_log_end_offset for file size retrieval."""

    def test_returns_file_size(self, tmp_path: Path) -> None:
        """Should return the file size as the end offset."""
        log_path = tmp_path / "session.jsonl"
        content = json.dumps({"type": "assistant", "message": {"content": []}}) + "\n"
        log_path.write_text(content)

        parser = SessionLogParser()
        offset = parser.get_log_end_offset(log_path)

        assert offset == log_path.stat().st_size

    def test_returns_start_offset_for_missing_file(self, tmp_path: Path) -> None:
        """Should return start_offset for missing files."""
        parser = SessionLogParser()
        nonexistent = tmp_path / "nonexistent.jsonl"

        offset = parser.get_log_end_offset(nonexistent, start_offset=100)

        assert offset == 100


class TestExtractBashCommands:
    """Test extract_bash_commands for Bash tool_use extraction."""

    def test_extracts_bash_commands(self, tmp_path: Path) -> None:
        """Should extract Bash tool_use commands from assistant messages."""
        log_path = tmp_path / "session.jsonl"
        entry = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "Bash",
                        "input": {"command": "uv run pytest"},
                    }
                ]
            },
        }
        log_path.write_text(json.dumps(entry) + "\n")

        parser = SessionLogParser()
        results = list(parser.iter_jsonl_entries(log_path))
        commands = parser.extract_bash_commands(results[0])

        assert commands == [("toolu_123", "uv run pytest")]

    def test_extracts_lowercase_bash_commands(self, tmp_path: Path) -> None:
        """Should extract 'bash' tool_use commands (case-insensitive)."""
        log_path = tmp_path / "session.jsonl"
        entry = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_456",
                        "name": "bash",
                        "input": {"command": "uv run pytest"},
                    }
                ]
            },
        }
        log_path.write_text(json.dumps(entry) + "\n")

        parser = SessionLogParser()
        results = list(parser.iter_jsonl_entries(log_path))
        commands = parser.extract_bash_commands(results[0])

        assert commands == [("toolu_456", "uv run pytest")]

    def test_returns_empty_for_user_messages(self, tmp_path: Path) -> None:
        """Should return empty list for user messages."""
        log_path = tmp_path / "session.jsonl"
        entry = {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_123", "content": "ok"}
                ]
            },
        }
        log_path.write_text(json.dumps(entry) + "\n")

        parser = SessionLogParser()
        results = list(parser.iter_jsonl_entries(log_path))
        commands = parser.extract_bash_commands(results[0])

        assert commands == []

    def test_ignores_non_bash_tools(self, tmp_path: Path) -> None:
        """Should ignore tool_use blocks for non-Bash tools."""
        log_path = tmp_path / "session.jsonl"
        entry = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "Read",
                        "input": {"file_path": "/some/file.py"},
                    }
                ]
            },
        }
        log_path.write_text(json.dumps(entry) + "\n")

        parser = SessionLogParser()
        results = list(parser.iter_jsonl_entries(log_path))
        commands = parser.extract_bash_commands(results[0])

        assert commands == []

    def test_extracts_multiple_commands(self, tmp_path: Path) -> None:
        """Should extract multiple Bash commands from a single message."""
        log_path = tmp_path / "session.jsonl"
        entry = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Bash",
                        "input": {"command": "uv run pytest"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": "Bash",
                        "input": {"command": "uvx ruff check ."},
                    },
                ]
            },
        }
        log_path.write_text(json.dumps(entry) + "\n")

        parser = SessionLogParser()
        results = list(parser.iter_jsonl_entries(log_path))
        commands = parser.extract_bash_commands(results[0])

        assert len(commands) == 2
        assert ("toolu_1", "uv run pytest") in commands
        assert ("toolu_2", "uvx ruff check .") in commands


class TestExtractToolResults:
    """Test extract_tool_results for tool_result extraction."""

    def test_extracts_tool_results(self, tmp_path: Path) -> None:
        """Should extract tool_result blocks from user messages."""
        log_path = tmp_path / "session.jsonl"
        entry = {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": "output",
                        "is_error": False,
                    }
                ]
            },
        }
        log_path.write_text(json.dumps(entry) + "\n")

        parser = SessionLogParser()
        results = list(parser.iter_jsonl_entries(log_path))
        tool_results = parser.extract_tool_results(results[0])

        assert tool_results == [("toolu_123", False)]

    def test_returns_empty_for_assistant_messages(self, tmp_path: Path) -> None:
        """Should return empty list for assistant messages."""
        log_path = tmp_path / "session.jsonl"
        entry = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    }
                ]
            },
        }
        log_path.write_text(json.dumps(entry) + "\n")

        parser = SessionLogParser()
        results = list(parser.iter_jsonl_entries(log_path))
        tool_results = parser.extract_tool_results(results[0])

        assert tool_results == []

    def test_extracts_error_status(self, tmp_path: Path) -> None:
        """Should correctly extract is_error status."""
        log_path = tmp_path / "session.jsonl"
        entry = {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": "error output",
                        "is_error": True,
                    }
                ]
            },
        }
        log_path.write_text(json.dumps(entry) + "\n")

        parser = SessionLogParser()
        results = list(parser.iter_jsonl_entries(log_path))
        tool_results = parser.extract_tool_results(results[0])

        assert tool_results == [("toolu_123", True)]


class TestExtractAssistantTextBlocks:
    """Test extract_assistant_text_blocks for text content extraction."""

    def test_extracts_text_blocks(self, tmp_path: Path) -> None:
        """Should extract text blocks from assistant messages."""
        log_path = tmp_path / "session.jsonl"
        entry = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Hello, world!"}]},
        }
        log_path.write_text(json.dumps(entry) + "\n")

        parser = SessionLogParser()
        results = list(parser.iter_jsonl_entries(log_path))
        texts = parser.extract_assistant_text_blocks(results[0])

        assert texts == ["Hello, world!"]

    def test_returns_empty_for_user_messages(self, tmp_path: Path) -> None:
        """Should return empty list for user messages."""
        log_path = tmp_path / "session.jsonl"
        entry = {
            "type": "user",
            "message": {"content": [{"type": "text", "text": "User input"}]},
        }
        log_path.write_text(json.dumps(entry) + "\n")

        parser = SessionLogParser()
        results = list(parser.iter_jsonl_entries(log_path))
        texts = parser.extract_assistant_text_blocks(results[0])

        assert texts == []

    def test_extracts_multiple_text_blocks(self, tmp_path: Path) -> None:
        """Should extract multiple text blocks from a single message."""
        log_path = tmp_path / "session.jsonl"
        entry = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "First paragraph"},
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}},
                    {"type": "text", "text": "Second paragraph"},
                ]
            },
        }
        log_path.write_text(json.dumps(entry) + "\n")

        parser = SessionLogParser()
        results = list(parser.iter_jsonl_entries(log_path))
        texts = parser.extract_assistant_text_blocks(results[0])

        assert texts == ["First paragraph", "Second paragraph"]

    def test_handles_role_based_format(self, tmp_path: Path) -> None:
        """Should handle alternative role-based message format."""
        log_path = tmp_path / "session.jsonl"
        # Some SDK versions use message.role instead of top-level type
        entry = {
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello"}],
            }
        }
        log_path.write_text(json.dumps(entry) + "\n")

        parser = SessionLogParser()
        results = list(parser.iter_jsonl_entries(log_path))
        texts = parser.extract_assistant_text_blocks(results[0])

        assert texts == ["Hello"]


class TestSessionLogParserIndependentUsability:
    """Test that SessionLogParser can be used independently of EvidenceCheck."""

    def test_parser_usable_standalone(self, tmp_path: Path) -> None:
        """SessionLogParser should work without EvidenceCheck."""
        log_path = tmp_path / "session.jsonl"
        entries = [
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "text", "text": "Running tests..."},
                        {
                            "type": "tool_use",
                            "id": "t1",
                            "name": "Bash",
                            "input": {"command": "pytest"},
                        },
                    ]
                },
            },
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "t1",
                            "content": "PASSED",
                            "is_error": False,
                        }
                    ]
                },
            },
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in entries) + "\n")

        # Use parser independently
        parser = SessionLogParser()

        all_commands = []
        all_results = []
        all_texts = []

        for entry in parser.iter_jsonl_entries(log_path):
            all_commands.extend(parser.extract_bash_commands(entry))
            all_results.extend(parser.extract_tool_results(entry))
            all_texts.extend(parser.extract_assistant_text_blocks(entry))

        assert all_commands == [("t1", "pytest")]
        assert all_results == [("t1", False)]
        assert all_texts == ["Running tests..."]


class TestFileSystemLogProvider:
    """Test FileSystemLogProvider implementation of LogProvider protocol."""

    def test_get_log_path_computes_sdk_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """get_log_path should compute the Claude SDK log path."""
        from src.infra.io.session_log_parser import FileSystemLogProvider

        # Mock the Claude config dir to use tmp_path
        monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path))

        provider = FileSystemLogProvider()
        repo_path = Path("/home/cyou/mala")
        session_id = "abc123"

        log_path = provider.get_log_path(repo_path, session_id)

        # Should use encoded repo path format
        assert log_path == tmp_path / "projects" / "-home-cyou-mala" / "abc123.jsonl"

    def test_iter_events_delegates_to_parser(self, tmp_path: Path) -> None:
        """iter_events should delegate to SessionLogParser."""
        from src.infra.io.session_log_parser import FileSystemLogProvider

        log_path = tmp_path / "session.jsonl"
        entry = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Hello"}]},
        }
        log_path.write_text(json.dumps(entry) + "\n")

        provider = FileSystemLogProvider()
        results = list(provider.iter_events(log_path))

        assert len(results) == 1
        assert results[0].data["type"] == "assistant"

    def test_iter_events_respects_offset(self, tmp_path: Path) -> None:
        """iter_events should start from the given offset."""
        from src.infra.io.session_log_parser import FileSystemLogProvider

        log_path = tmp_path / "session.jsonl"
        first_entry = json.dumps({"type": "assistant", "message": {"content": []}})
        second_entry = json.dumps({"type": "user", "message": {"content": []}})
        log_path.write_text(first_entry + "\n" + second_entry + "\n")

        provider = FileSystemLogProvider()
        offset = len(first_entry) + 1
        results = list(provider.iter_events(log_path, offset=offset))

        assert len(results) == 1
        assert results[0].data["type"] == "user"

    def test_get_end_offset_returns_file_size(self, tmp_path: Path) -> None:
        """get_end_offset should return the file size."""
        from src.infra.io.session_log_parser import FileSystemLogProvider

        log_path = tmp_path / "session.jsonl"
        content = json.dumps({"type": "assistant", "message": {"content": []}}) + "\n"
        log_path.write_text(content)

        provider = FileSystemLogProvider()
        offset = provider.get_end_offset(log_path)

        assert offset == log_path.stat().st_size

    def test_get_end_offset_returns_start_for_missing_file(
        self, tmp_path: Path
    ) -> None:
        """get_end_offset should return start_offset for missing files."""
        from src.infra.io.session_log_parser import FileSystemLogProvider

        provider = FileSystemLogProvider()
        nonexistent = tmp_path / "nonexistent.jsonl"

        offset = provider.get_end_offset(nonexistent, start_offset=50)

        assert offset == 50

    def test_conforms_to_log_provider_protocol(self, tmp_path: Path) -> None:
        """FileSystemLogProvider should conform to LogProvider protocol."""
        from src.core.protocols.log import LogProvider
        from src.infra.io.session_log_parser import FileSystemLogProvider

        provider = FileSystemLogProvider()

        # Protocol check (runtime_checkable)
        assert isinstance(provider, LogProvider)

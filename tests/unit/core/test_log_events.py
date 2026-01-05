"""Contract tests for Claude Agent SDK JSONL log event parsing.

These tests verify that src/log_events.py correctly parses the log format
produced by Claude Agent SDK. They serve as contract tests - if the SDK
changes its log format, these tests will fail, alerting us to update our
parsing logic.

Test fixtures are stored in tests/fixtures/sdk_log_samples.jsonl containing
real JSONL entries from Claude Agent SDK sessions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.log_events import (
    AssistantLogEntry,
    LogParseError,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserLogEntry,
    parse_log_entry,
    parse_log_entry_strict,
)

# Path to JSONL fixture file
FIXTURES_DIR = Path(__file__).parent.parent.parent / "fixtures"
SDK_LOG_SAMPLES = FIXTURES_DIR / "sdk_log_samples.jsonl"


# =============================================================================
# Contract Tests with JSONL Fixture File
# =============================================================================


class TestJSONLFixtureFile:
    """Test parsing entries from the JSONL fixture file."""

    def test_all_fixture_entries_parse_successfully(self) -> None:
        """All entries in the fixture file should parse without error."""
        entries: list[tuple[int, dict]] = []
        with open(SDK_LOG_SAMPLES) as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    data = json.loads(line)
                    entries.append((line_num, data))

        assert len(entries) > 0, "Fixture file is empty"

        for line_num, data in entries:
            entry = parse_log_entry(data)
            assert entry is not None, f"Failed to parse fixture line {line_num}: {data}"

    def test_fixture_strict_parsing(self) -> None:
        """All fixture entries should also pass strict parsing."""
        with open(SDK_LOG_SAMPLES) as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    data = json.loads(line)
                    # Should not raise
                    entry = parse_log_entry_strict(data)
                    assert entry is not None, f"Line {line_num} returned None"


# =============================================================================
# JSONL Fixtures - Sample entries from real Claude Agent SDK sessions
# =============================================================================


class TestAssistantToolUse:
    """Test parsing assistant messages with tool_use blocks."""

    def test_bash_tool_use_basic(self) -> None:
        """Parse a basic Bash tool_use entry."""
        data = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_pytest_123",
                        "name": "Bash",
                        "input": {"command": "uv run pytest"},
                    }
                ]
            },
        }

        entry = parse_log_entry(data)

        assert isinstance(entry, AssistantLogEntry)
        assert len(entry.message.content) == 1
        block = entry.message.content[0]
        assert isinstance(block, ToolUseBlock)
        assert block.id == "toolu_pytest_123"
        assert block.name == "Bash"
        assert block.input == {"command": "uv run pytest"}

    def test_multiple_tool_use_blocks(self) -> None:
        """Parse assistant message with multiple tool_use blocks."""
        data = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "Bash",
                        "input": {"command": "uvx ruff check ."},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": "Bash",
                        "input": {"command": "uvx ruff format ."},
                    },
                ]
            },
        }

        entry = parse_log_entry(data)

        assert isinstance(entry, AssistantLogEntry)
        assert len(entry.message.content) == 2
        assert all(isinstance(b, ToolUseBlock) for b in entry.message.content)
        block0 = entry.message.content[0]
        block1 = entry.message.content[1]
        assert isinstance(block0, ToolUseBlock)
        assert isinstance(block1, ToolUseBlock)
        assert block0.id == "toolu_1"
        assert block1.id == "toolu_2"

    def test_read_tool_use(self) -> None:
        """Parse a Read tool_use entry."""
        data = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_read_456",
                        "name": "Read",
                        "input": {"file_path": "/path/to/file.py"},
                    }
                ]
            },
        }

        entry = parse_log_entry(data)

        assert isinstance(entry, AssistantLogEntry)
        block = entry.message.content[0]
        assert isinstance(block, ToolUseBlock)
        assert block.name == "Read"
        assert block.input == {"file_path": "/path/to/file.py"}


class TestAssistantTextBlocks:
    """Test parsing assistant messages with text blocks."""

    def test_text_block_basic(self) -> None:
        """Parse assistant message with text content."""
        data = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "ISSUE_NO_CHANGE: Already implemented"}
                ]
            },
        }

        entry = parse_log_entry(data)

        assert isinstance(entry, AssistantLogEntry)
        assert len(entry.message.content) == 1
        block = entry.message.content[0]
        assert isinstance(block, TextBlock)
        assert block.text == "ISSUE_NO_CHANGE: Already implemented"

    def test_mixed_text_and_tool_use(self) -> None:
        """Parse assistant message with both text and tool_use blocks."""
        data = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Let me run the tests."},
                    {
                        "type": "tool_use",
                        "id": "toolu_test_1",
                        "name": "Bash",
                        "input": {"command": "uv run pytest"},
                    },
                ]
            },
        }

        entry = parse_log_entry(data)

        assert isinstance(entry, AssistantLogEntry)
        assert len(entry.message.content) == 2
        assert isinstance(entry.message.content[0], TextBlock)
        assert isinstance(entry.message.content[1], ToolUseBlock)


class TestUserToolResults:
    """Test parsing user messages with tool_result blocks."""

    def test_tool_result_success(self) -> None:
        """Parse user message with successful tool_result."""
        data = {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_pytest_123",
                        "content": "5 passed in 0.42s",
                        "is_error": False,
                    }
                ]
            },
        }

        entry = parse_log_entry(data)

        assert isinstance(entry, UserLogEntry)
        assert len(entry.message.content) == 1
        block = entry.message.content[0]
        assert isinstance(block, ToolResultBlock)
        assert block.tool_use_id == "toolu_pytest_123"
        assert block.content == "5 passed in 0.42s"
        assert block.is_error is False

    def test_tool_result_error(self) -> None:
        """Parse user message with error tool_result."""
        data = {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_pytest_123",
                        "content": "Exit code 1\n===== FAILURES =====\ntest_foo failed",
                        "is_error": True,
                    }
                ]
            },
        }

        entry = parse_log_entry(data)

        assert isinstance(entry, UserLogEntry)
        block = entry.message.content[0]
        assert isinstance(block, ToolResultBlock)
        assert block.is_error is True

    def test_multiple_tool_results(self) -> None:
        """Parse user message with multiple tool_result blocks."""
        data = {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "Result 1",
                        "is_error": False,
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_2",
                        "content": "Result 2",
                        "is_error": False,
                    },
                ]
            },
        }

        entry = parse_log_entry(data)

        assert isinstance(entry, UserLogEntry)
        assert len(entry.message.content) == 2
        assert all(isinstance(b, ToolResultBlock) for b in entry.message.content)


class TestForwardCompatibility:
    """Test forward compatibility with unknown fields and block types."""

    def test_unknown_top_level_fields_ignored(self) -> None:
        """Unknown fields at top level should be ignored."""
        data = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Hello"}]},
            "unknown_field": "should be ignored",
            "metadata": {"timestamp": 12345},
        }

        entry = parse_log_entry(data)

        assert isinstance(entry, AssistantLogEntry)
        block = entry.message.content[0]
        assert isinstance(block, TextBlock)
        assert block.text == "Hello"

    def test_unknown_block_fields_ignored(self) -> None:
        """Unknown fields in content blocks should be ignored."""
        data = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "Bash",
                        "input": {"command": "ls"},
                        "cache_control": {"type": "ephemeral"},
                        "new_sdk_field": "unknown",
                    }
                ]
            },
        }

        entry = parse_log_entry(data)

        assert isinstance(entry, AssistantLogEntry)
        block = entry.message.content[0]
        assert isinstance(block, ToolUseBlock)
        assert block.id == "toolu_123"

    def test_unknown_block_type_skipped(self) -> None:
        """Unknown block types should be skipped, not cause errors."""
        data = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "future_block_type", "data": "unknown"},
                    {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {}},
                ]
            },
        }

        entry = parse_log_entry(data)

        assert isinstance(entry, AssistantLogEntry)
        # Only 2 blocks - the unknown type is skipped
        assert len(entry.message.content) == 2
        assert isinstance(entry.message.content[0], TextBlock)
        assert isinstance(entry.message.content[1], ToolUseBlock)

    def test_unknown_entry_type_returns_none(self) -> None:
        """Unknown entry type returns None, not an error."""
        data = {
            "type": "system",  # Not assistant or user
            "message": {"content": []},
        }

        entry = parse_log_entry(data)

        assert entry is None


class TestEdgeCases:
    """Test edge cases and malformed input handling."""

    def test_empty_content_array(self) -> None:
        """Empty content array is valid."""
        data = {
            "type": "assistant",
            "message": {"content": []},
        }

        entry = parse_log_entry(data)

        assert isinstance(entry, AssistantLogEntry)
        assert entry.message.content == []

    def test_missing_message_returns_none(self) -> None:
        """Missing message field returns None."""
        data = {"type": "assistant"}

        entry = parse_log_entry(data)

        assert entry is None

    def test_non_dict_returns_none(self) -> None:
        """Non-dict input returns None."""
        entry = parse_log_entry("not a dict")  # type: ignore[arg-type]

        assert entry is None

    def test_non_list_content_returns_none(self) -> None:
        """Non-list content field returns None."""
        data = {
            "type": "assistant",
            "message": {"content": "not a list"},
        }

        entry = parse_log_entry(data)

        assert entry is None

    def test_missing_content_field_returns_none(self) -> None:
        """Missing content field returns None (required field)."""
        data = {
            "type": "assistant",
            "message": {},  # No content field
        }

        entry = parse_log_entry(data)

        assert entry is None

    def test_tool_use_missing_id_uses_empty_string(self) -> None:
        """Missing tool_use id defaults to empty string."""
        data = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "name": "Bash",
                        "input": {"command": "ls"},
                    }
                ]
            },
        }

        entry = parse_log_entry(data)

        assert isinstance(entry, AssistantLogEntry)
        block = entry.message.content[0]
        assert isinstance(block, ToolUseBlock)
        assert block.id == ""

    def test_tool_result_missing_is_error_defaults_false(self) -> None:
        """Missing is_error defaults to False."""
        data = {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": "output",
                    }
                ]
            },
        }

        entry = parse_log_entry(data)

        assert isinstance(entry, UserLogEntry)
        block = entry.message.content[0]
        assert isinstance(block, ToolResultBlock)
        assert block.is_error is False

    def test_tool_result_non_bool_is_error_skipped(self) -> None:
        """tool_result with non-bool is_error is skipped in lenient mode."""
        data = {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_123",
                        "content": "output",
                        "is_error": "false",  # String, not bool - would coerce to True
                    }
                ]
            },
        }

        entry = parse_log_entry(data)

        # Entry parses but the malformed block is skipped
        assert isinstance(entry, UserLogEntry)
        assert len(entry.message.content) == 0  # Block was skipped

    def test_role_based_message_format(self) -> None:
        """Support alternative format with role in message."""
        data = {
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello"}],
            }
        }

        entry = parse_log_entry(data)

        assert isinstance(entry, AssistantLogEntry)
        block = entry.message.content[0]
        assert isinstance(block, TextBlock)
        assert block.text == "Hello"

    def test_non_dict_block_skipped(self) -> None:
        """Non-dict content block is skipped."""
        data = {
            "type": "assistant",
            "message": {
                "content": [
                    "just a string",  # Not a dict
                    {"type": "text", "text": "Hello"},
                ]
            },
        }

        entry = parse_log_entry(data)

        assert isinstance(entry, AssistantLogEntry)
        assert len(entry.message.content) == 1
        assert isinstance(entry.message.content[0], TextBlock)


class TestCompleteSessionFlow:
    """Test parsing a complete tool use/result flow."""

    def test_bash_command_with_result(self) -> None:
        """Parse paired tool_use and tool_result entries."""
        # Assistant invokes tool
        assistant_data = {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_pytest_session",
                        "name": "Bash",
                        "input": {"command": "uv run pytest -v"},
                    }
                ]
            },
        }
        # User provides result
        user_data = {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_pytest_session",
                        "content": "========================= test session starts =========================\n"
                        "collected 42 items\n\n"
                        "tests/test_example.py::test_one PASSED\n"
                        "========================= 42 passed in 1.23s =========================",
                        "is_error": False,
                    }
                ]
            },
        }

        assistant_entry = parse_log_entry(assistant_data)
        user_entry = parse_log_entry(user_data)

        assert isinstance(assistant_entry, AssistantLogEntry)
        assert isinstance(user_entry, UserLogEntry)

        tool_use = assistant_entry.message.content[0]
        tool_result = user_entry.message.content[0]

        assert isinstance(tool_use, ToolUseBlock)
        assert isinstance(tool_result, ToolResultBlock)

        # IDs should match for correlation
        assert tool_use.id == tool_result.tool_use_id == "toolu_pytest_session"


class TestLogParseError:
    """Test LogParseError exception class."""

    def test_error_includes_schema_hint(self) -> None:
        """LogParseError should include schema documentation in message."""
        error = LogParseError("Test error")

        assert error.schema_hint is not None
        assert "assistant" in error.schema_hint
        assert "tool_use" in error.schema_hint
        assert error.schema_hint in str(error)


class TestStrictParsing:
    """Test parse_log_entry_strict for detailed error messages."""

    def test_strict_valid_entry_succeeds(self) -> None:
        """Valid entries should parse successfully in strict mode."""
        data = {
            "type": "assistant",
            "message": {"content": [{"type": "text", "text": "Hello"}]},
        }

        entry = parse_log_entry_strict(data)

        assert isinstance(entry, AssistantLogEntry)

    def test_strict_invalid_type_raises_error(self) -> None:
        """Invalid entry type should raise LogParseError with details."""
        data = {"type": "invalid", "message": {"content": []}}

        with pytest.raises(LogParseError) as exc_info:
            parse_log_entry_strict(data)

        error = exc_info.value
        assert "invalid" in error.reason
        assert "assistant" in error.reason or "user" in error.reason
        assert error.data == data
        # Schema should be in the exception message
        assert "tool_use" in str(error)

    def test_strict_missing_type_raises_error(self) -> None:
        """Missing type field should raise LogParseError."""
        data = {"message": {"content": []}}

        with pytest.raises(LogParseError) as exc_info:
            parse_log_entry_strict(data)

        assert "type" in exc_info.value.reason.lower()

    def test_strict_missing_message_raises_error(self) -> None:
        """Missing message field should raise LogParseError."""
        data = {"type": "assistant"}

        with pytest.raises(LogParseError) as exc_info:
            parse_log_entry_strict(data)

        assert "message" in exc_info.value.reason.lower()

    def test_strict_missing_content_raises_error(self) -> None:
        """Missing content field should raise LogParseError."""
        data = {"type": "assistant", "message": {}}

        with pytest.raises(LogParseError) as exc_info:
            parse_log_entry_strict(data)

        assert "content" in exc_info.value.reason.lower()

    def test_strict_non_list_content_raises_error(self) -> None:
        """Non-list content should raise LogParseError."""
        data = {"type": "assistant", "message": {"content": "not a list"}}

        with pytest.raises(LogParseError) as exc_info:
            parse_log_entry_strict(data)

        assert "list" in exc_info.value.reason.lower()

    def test_strict_unknown_block_type_raises_error(self) -> None:
        """Unknown block type should raise error in strict mode."""
        data = {
            "type": "assistant",
            "message": {"content": [{"type": "unknown_future_type", "data": "test"}]},
        }

        with pytest.raises(LogParseError) as exc_info:
            parse_log_entry_strict(data)

        assert "unknown_future_type" in exc_info.value.reason
        assert "index 0" in exc_info.value.reason

    def test_strict_non_dict_block_raises_error(self) -> None:
        """Non-dict content block should raise error in strict mode."""
        data = {
            "type": "assistant",
            "message": {"content": ["just a string"]},
        }

        with pytest.raises(LogParseError) as exc_info:
            parse_log_entry_strict(data)

        assert "dict" in exc_info.value.reason.lower()
        assert "index 0" in exc_info.value.reason

    def test_strict_text_block_missing_text_field(self) -> None:
        """Text block without text field should raise error."""
        data = {
            "type": "assistant",
            "message": {"content": [{"type": "text"}]},
        }

        with pytest.raises(LogParseError) as exc_info:
            parse_log_entry_strict(data)

        assert "text" in exc_info.value.reason.lower()

    def test_strict_tool_use_invalid_input_type(self) -> None:
        """tool_use with non-dict input should raise error."""
        data = {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "tool_use", "id": "1", "name": "Bash", "input": "not dict"}
                ]
            },
        }

        with pytest.raises(LogParseError) as exc_info:
            parse_log_entry_strict(data)

        assert "input" in exc_info.value.reason.lower()
        assert "dict" in exc_info.value.reason.lower()

    def test_strict_non_dict_input_raises_error(self) -> None:
        """Non-dict entry should raise LogParseError."""
        with pytest.raises(LogParseError) as exc_info:
            parse_log_entry_strict("not a dict")  # type: ignore[arg-type]

        assert "dict" in exc_info.value.reason.lower()

    def test_strict_tool_use_missing_id_raises_error(self) -> None:
        """tool_use without id field should raise error in strict mode."""
        data = {
            "type": "assistant",
            "message": {"content": [{"type": "tool_use", "name": "Bash", "input": {}}]},
        }

        with pytest.raises(LogParseError) as exc_info:
            parse_log_entry_strict(data)

        assert "id" in exc_info.value.reason.lower()
        assert "missing" in exc_info.value.reason.lower()

    def test_strict_tool_use_missing_name_raises_error(self) -> None:
        """tool_use without name field should raise error in strict mode."""
        data = {
            "type": "assistant",
            "message": {
                "content": [{"type": "tool_use", "id": "toolu_1", "input": {}}]
            },
        }

        with pytest.raises(LogParseError) as exc_info:
            parse_log_entry_strict(data)

        assert "name" in exc_info.value.reason.lower()
        assert "missing" in exc_info.value.reason.lower()

    def test_strict_tool_result_missing_tool_use_id_raises_error(self) -> None:
        """tool_result without tool_use_id field should raise error in strict mode."""
        data = {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "content": "output", "is_error": False}
                ]
            },
        }

        with pytest.raises(LogParseError) as exc_info:
            parse_log_entry_strict(data)

        assert "tool_use_id" in exc_info.value.reason.lower()
        assert "missing" in exc_info.value.reason.lower()

    def test_strict_tool_result_non_bool_is_error_raises_error(self) -> None:
        """tool_result with non-bool is_error should raise error in strict mode."""
        data = {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "output",
                        "is_error": "false",  # String instead of bool
                    }
                ]
            },
        }

        with pytest.raises(LogParseError) as exc_info:
            parse_log_entry_strict(data)

        assert "is_error" in exc_info.value.reason.lower()
        assert "bool" in exc_info.value.reason.lower()

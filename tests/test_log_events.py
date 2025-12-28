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
from typing import TYPE_CHECKING

import pytest

from src.log_events import (
    AssistantLogEntry,
    AssistantMessage,
    LogParseError,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserLogEntry,
    UserMessage,
    parse_log_entry,
    parse_log_entry_strict,
)

if TYPE_CHECKING:
    from src.log_events import ContentBlock, LogEntry


# Path to JSONL fixture file
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SDK_LOG_SAMPLES = FIXTURES_DIR / "sdk_log_samples.jsonl"


# =============================================================================
# Contract Tests with JSONL Fixture File
# =============================================================================


class TestJSONLFixtureFile:
    """Test parsing entries from the JSONL fixture file."""

    def test_fixture_file_exists(self) -> None:
        """Verify the JSONL fixture file exists."""
        assert SDK_LOG_SAMPLES.exists(), f"Fixture file not found: {SDK_LOG_SAMPLES}"

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

    def test_fixture_contains_tool_use_entries(self) -> None:
        """Fixture file should contain tool_use entries."""
        tool_use_count = 0
        with open(SDK_LOG_SAMPLES) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    entry = parse_log_entry(data)
                    if isinstance(entry, AssistantLogEntry):
                        for block in entry.message.content:
                            if isinstance(block, ToolUseBlock):
                                tool_use_count += 1

        assert tool_use_count > 0, "Fixture file should contain tool_use entries"

    def test_fixture_contains_tool_result_entries(self) -> None:
        """Fixture file should contain tool_result entries."""
        tool_result_count = 0
        with open(SDK_LOG_SAMPLES) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    entry = parse_log_entry(data)
                    if isinstance(entry, UserLogEntry):
                        for block in entry.message.content:
                            if isinstance(block, ToolResultBlock):
                                tool_result_count += 1

        assert tool_result_count > 0, "Fixture file should contain tool_result entries"

    def test_fixture_contains_text_blocks(self) -> None:
        """Fixture file should contain text blocks."""
        text_block_count = 0
        with open(SDK_LOG_SAMPLES) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    entry = parse_log_entry(data)
                    if isinstance(entry, AssistantLogEntry):
                        for block in entry.message.content:
                            if isinstance(block, TextBlock):
                                text_block_count += 1

        assert text_block_count > 0, "Fixture file should contain text blocks"

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
        assert entry.message.content[0].id == "toolu_1"
        assert entry.message.content[1].id == "toolu_2"

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
        assert entry.message.content[0].text == "Hello"

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


class TestDataclassProperties:
    """Test dataclass properties and immutability."""

    def test_dataclasses_are_frozen(self) -> None:
        """All dataclasses should be immutable."""
        block = TextBlock(text="Hello")
        with pytest.raises(AttributeError):
            block.text = "Modified"  # type: ignore[misc]

    def test_content_block_type_alias(self) -> None:
        """ContentBlock type should accept all block types."""
        blocks: list[ContentBlock] = [
            TextBlock(text="Hello"),
            ToolUseBlock(id="1", name="Bash", input={}),
            ToolResultBlock(tool_use_id="1", content="", is_error=False),
        ]
        assert len(blocks) == 3

    def test_log_entry_type_alias(self) -> None:
        """LogEntry type should accept all entry types."""
        entries: list[LogEntry] = [
            AssistantLogEntry(message=AssistantMessage(content=[])),
            UserLogEntry(message=UserMessage(content=[])),
        ]
        assert len(entries) == 2


class TestLogParseError:
    """Test LogParseError exception class."""

    def test_error_with_reason_only(self) -> None:
        """LogParseError can be created with reason only."""
        error = LogParseError("Missing required field 'type'")

        assert error.reason == "Missing required field 'type'"
        assert error.data is None
        assert "Missing required field 'type'" in str(error)

    def test_error_with_data(self) -> None:
        """LogParseError can include the problematic data."""
        error = LogParseError(
            "Invalid message structure", data={"type": "unknown", "bad": True}
        )

        assert error.reason == "Invalid message structure"
        assert error.data == {"type": "unknown", "bad": True}

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

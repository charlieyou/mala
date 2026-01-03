import pytest

from src.infra.io.log_output import console


def test_truncate_text_respects_verbose() -> None:
    console.set_verbose(False)
    assert console.truncate_text("abcdef", 3) == "abc..."

    console.set_verbose(True)
    assert console.truncate_text("abcdef", 3) == "abcdef"

    console.set_verbose(False)


def test_get_agent_color_is_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(console, "_agent_color_map", {})
    monkeypatch.setattr(console, "_agent_color_index", 0)

    color1 = console.get_agent_color("agent-a")
    color2 = console.get_agent_color("agent-a")
    color3 = console.get_agent_color("agent-b")

    assert color1 == color2
    assert color3 != ""


def test_format_arguments_verbose_and_non_verbose() -> None:
    verbose_output = console._format_arguments(
        {
            "content": "line1\nline2",
            "meta": {"k": "v"},
            "items": [1, 2],
        },
        True,
        tool_name="Edit",
    )
    assert "content" in verbose_output
    assert "line1" in verbose_output
    assert "line2" in verbose_output
    assert "meta" in verbose_output
    assert "items" in verbose_output

    non_verbose_output = console._format_arguments(
        {"text": "x" * 120, "flag": True, "count": 3},
        False,
        tool_name="Tool",
    )
    assert "text" in non_verbose_output
    assert "..." in non_verbose_output
    assert "flag" in non_verbose_output
    assert "count" in non_verbose_output


def test_log_tool_and_agent_text_output(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(console, "_agent_color_map", {})
    monkeypatch.setattr(console, "_agent_color_index", 0)

    console.set_verbose(True)
    console.log_tool(
        "Edit",
        description="update file",
        agent_id="agent-1",
        arguments={"content": "line1\nline2", "flag": True},
    )
    console.log_agent_text("hello world", agent_id="agent-1")
    console.log("!", "message", agent_id="agent-1")

    output = capsys.readouterr().out
    assert "Edit" in output
    assert "content" in output
    assert "flag" in output
    assert "hello world" in output
    assert "message" in output

    console.set_verbose(False)


def test_quiet_summary_file_tools() -> None:
    """Test that file tools show file path in quiet mode."""
    # Read tool with file_path
    summary = console._get_quiet_summary("Read", "", {"file_path": "/path/to/file.py"})
    assert summary == "/path/to/file.py"

    # Glob with pattern
    summary = console._get_quiet_summary("Glob", "", {"pattern": "**/*.py"})
    assert summary == "**/*.py"


def test_quiet_summary_bash_uses_description() -> None:
    """Test that Bash tool shows description field in quiet mode."""
    # From description parameter
    summary = console._get_quiet_summary("Bash", "Run tests", {"command": "pytest"})
    assert summary == "Run tests"

    # From arguments when description param is empty
    summary = console._get_quiet_summary(
        "Bash", "", {"command": "pytest", "description": "Execute test suite"}
    )
    assert summary == "Execute test suite"


def test_quiet_summary_other_tools_truncated_args() -> None:
    """Test that other tools show truncated args dict in quiet mode."""
    # Few args
    summary = console._get_quiet_summary("CustomTool", "", {"a": 1, "b": 2})
    assert "{a=..., b=...}" == summary

    # Many args (more than 3)
    summary = console._get_quiet_summary(
        "CustomTool", "", {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    )
    assert "a=..." in summary
    assert "+2 more" in summary


def test_quiet_summary_empty_args() -> None:
    """Test that tools without args return empty summary."""
    summary = console._get_quiet_summary("SomeTool", "", None)
    assert summary == ""

    summary = console._get_quiet_summary("SomeTool", "", {})
    assert summary == ""


def test_log_tool_quiet_mode_output(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test log_tool in quiet mode produces single-line output."""
    monkeypatch.setattr(console, "_agent_color_map", {})
    monkeypatch.setattr(console, "_agent_color_index", 0)

    console.set_verbose(False)

    # Read tool should show file path
    console.log_tool("Read", arguments={"file_path": "/src/main.py"})
    output = capsys.readouterr().out
    assert "Read" in output
    assert "/src/main.py" in output
    assert "\n    " not in output  # No multi-line args

    # Bash should show description
    console.log_tool(
        "Bash", description="Install deps", arguments={"command": "npm install"}
    )
    output = capsys.readouterr().out
    assert "Bash" in output
    assert "Install deps" in output
    assert "npm install" not in output  # Command not shown in quiet mode

    console.set_verbose(False)


def test_log_tool_verbose_mode_output(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test log_tool in verbose mode shows full arguments."""
    monkeypatch.setattr(console, "_agent_color_map", {})
    monkeypatch.setattr(console, "_agent_color_index", 0)

    console.set_verbose(True)

    console.log_tool(
        "Read",
        description="reading file",
        arguments={"file_path": "/src/main.py", "offset": 100},
    )
    output = capsys.readouterr().out
    assert "Read" in output
    assert "file_path" in output
    assert "/src/main.py" in output
    assert "offset" in output
    assert "100" in output

    console.set_verbose(False)


def test_log_with_issue_id_shows_issue_only(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that log() with issue_id displays issue_id, not agent_id."""
    monkeypatch.setattr(console, "_agent_color_map", {})
    monkeypatch.setattr(console, "_agent_color_index", 0)

    console.log(
        "!",
        "test message",
        agent_id="agent-abc",
        issue_id="ISSUE-123",
    )

    output = capsys.readouterr().out
    assert "[ISSUE-123]" in output
    assert "agent-abc" not in output
    assert "test message" in output


def test_log_with_issue_id_uses_agent_color(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that log() with issue_id uses agent_id for color mapping."""
    monkeypatch.setattr(console, "_agent_color_map", {})
    monkeypatch.setattr(console, "_agent_color_index", 0)

    # First establish a color for agent-xyz
    color = console.get_agent_color("agent-xyz")

    console.log(
        "!",
        "test message",
        agent_id="agent-xyz",
        issue_id="ISSUE-456",
    )

    output = capsys.readouterr().out
    # The color escape code should be in the output before the issue_id
    assert color in output
    assert "[ISSUE-456]" in output


def test_log_fallback_shows_agent_id_when_no_issue_id(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that log() without issue_id falls back to showing agent_id."""
    monkeypatch.setattr(console, "_agent_color_map", {})
    monkeypatch.setattr(console, "_agent_color_index", 0)

    console.log(
        "!",
        "test message",
        agent_id="agent-a1b2",
    )

    output = capsys.readouterr().out
    assert "[agent-a1b2]" in output
    assert "test message" in output


def test_log_with_issue_id_no_agent_id_uses_cyan(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that log() with issue_id but no agent_id uses cyan color."""
    monkeypatch.setattr(console, "_agent_color_map", {})
    monkeypatch.setattr(console, "_agent_color_index", 0)

    console.log(
        "!",
        "test message",
        issue_id="ISSUE-789",
    )

    output = capsys.readouterr().out
    assert "[ISSUE-789]" in output
    # Should use cyan color (Colors.CYAN = "\033[96m")
    assert "\033[96m" in output

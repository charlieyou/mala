import pytest

from src.logging import console


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

    console.set_verbose(False)
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

from types import TracebackType
from typing import Never, Self

import pytest

from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

import src.braintrust_integration as braintrust_integration


class DummySpan:
    def __init__(self) -> None:
        self.entered = False
        self.exited = False
        self.logged = None

    def __enter__(self) -> Self:
        self.entered = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.exited = True
        return None

    def log(self, **kwargs: object) -> None:
        self.logged = kwargs


def test_is_braintrust_enabled_false_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(braintrust_integration, "BRAINTRUST_AVAILABLE", True)
    monkeypatch.delenv("BRAINTRUST_API_KEY", raising=False)

    assert braintrust_integration.is_braintrust_enabled() is False


def test_flush_braintrust_ignores_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}

    def _boom() -> Never:
        calls["count"] += 1
        raise RuntimeError("fail")

    monkeypatch.setattr(braintrust_integration, "BRAINTRUST_AVAILABLE", True)
    monkeypatch.setattr(braintrust_integration, "flush", _boom)

    braintrust_integration.flush_braintrust()

    assert calls["count"] == 1


def test_traced_agent_execution_disabled_returns_self(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(braintrust_integration, "BRAINTRUST_AVAILABLE", False)
    tracer = braintrust_integration.TracedAgentExecution("issue", "agent")

    with tracer as entered:
        assert entered is tracer
        assert tracer.span is None


def test_traced_agent_execution_logs_and_flushes(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_span = DummySpan()
    flush_calls = {"count": 0}

    def _start_span(**_kwargs: object) -> DummySpan:
        return dummy_span

    def _flush() -> None:
        flush_calls["count"] += 1

    monkeypatch.setattr(braintrust_integration, "BRAINTRUST_AVAILABLE", True)
    monkeypatch.setattr(braintrust_integration, "start_span", _start_span)
    monkeypatch.setattr(braintrust_integration, "flush", _flush)
    monkeypatch.setenv("BRAINTRUST_API_KEY", "test-key")

    tracer = braintrust_integration.TracedAgentExecution("issue-1", "agent-1")
    with tracer:
        tracer.log_input("prompt")
        assistant = AssistantMessage(
            content=[
                TextBlock(text="hello"),
                ToolUseBlock(id="tool-1", name="Search", input={"q": "hi"}),
                ToolResultBlock(tool_use_id="tool-1", content="ok"),
            ],
            model="test",
        )
        tracer.log_message(assistant)
        result = ResultMessage(
            subtype="final",
            duration_ms=1,
            duration_api_ms=1,
            is_error=False,
            num_turns=1,
            session_id="sess",
            total_cost_usd=None,
            usage=None,
            result="done",
            structured_output=None,
        )
        tracer.log_message(result)
        tracer.set_success(True)

    assert dummy_span.entered is True
    assert dummy_span.logged is not None
    assert dummy_span.logged["metadata"]["success"] is True
    assert dummy_span.logged["metadata"]["tool_calls_count"] == 1
    assert flush_calls["count"] == 1
    assert tracer.output_text.strip() == "done"


def test_traced_agent_execution_records_error(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_span = DummySpan()
    flush_calls = {"count": 0}

    def _start_span(**_kwargs: object) -> DummySpan:
        return dummy_span

    def _flush() -> None:
        flush_calls["count"] += 1

    monkeypatch.setattr(braintrust_integration, "BRAINTRUST_AVAILABLE", True)
    monkeypatch.setattr(braintrust_integration, "start_span", _start_span)
    monkeypatch.setattr(braintrust_integration, "flush", _flush)
    monkeypatch.setenv("BRAINTRUST_API_KEY", "test-key")

    tracer = braintrust_integration.TracedAgentExecution("issue-2", "agent-2")

    with pytest.raises(ValueError):
        with tracer:
            raise ValueError("boom")

    assert tracer.error == "boom"
    assert dummy_span.logged is not None
    assert dummy_span.logged["metadata"]["success"] is False
    assert flush_calls["count"] == 1

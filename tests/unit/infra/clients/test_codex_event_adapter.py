"""Unit tests for :class:`CodexEventAdapter` (Phase D, T011).

Table-driven coverage of the plan's mapping table
(``plans/2026-05-07-codex-provider-plan.md`` L749-L762, D1-D7) plus the
edge cases called out in the issue spec:

  * D1 dedup — delta-then-completed must not double-emit text.
  * D1 — completed item without prior delta still emits text.
  * D2 — bash command emits ``AgentToolUseEvent("bash", ...)`` whose
    ``input["command"]`` is recognised by
    :class:`LintCacheProtocol.detect_lint_command` (AC #15).
  * D2 — command exit code drives ``is_error`` so the lint cache does
    not mark a failing lint as successful.
  * D3 — ``FileChangeThreadItem`` ``PatchApplyStatus`` drives
    ``is_error``.
  * D4 — ``McpToolCallThreadItem`` produces ``"<server>.<tool>"`` name
    plus a paired result whose content carries the
    ``McpToolCallResult`` / ``McpToolCallError``.
  * D5 — reasoning notifications produce no events.
  * D6 — ``turn/completed`` carries ``session_id`` from
    ``payload.thread_id`` and ``is_error`` from the ``Turn.status``.
  * D7 — ``error`` notification surfaces as an
    :class:`AgentResultEvent` with ``is_error=True`` (terminal failure
    rather than idle hang).
  * Unknown notification methods are dropped silently so future SDK
    additions do not crash a running turn.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from src.core.protocols.agent_event import (
    AgentResultEvent,
    AgentTextEvent,
    AgentToolResultEvent,
    AgentToolUseEvent,
)
from src.infra.clients.codex_event_adapter import CodexEventAdapter


@dataclass
class _Notification:
    """Minimal ``method`` + ``payload`` shape (matches SDK ``Notification``)."""

    method: str
    payload: object


def _make_command_item(
    *,
    item_id: str = "cmd_1",
    command: str = "ruff check .",
    cwd: str = "/repo",
    status: str = "in_progress",
    exit_code: int | None = None,
    aggregated_output: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=item_id,
        type="commandExecution",
        command=command,
        cwd=cwd,
        status=SimpleNamespace(value=status),
        exit_code=exit_code,
        aggregated_output=aggregated_output,
    )


def _make_file_change_item(
    *,
    item_id: str = "fc_1",
    status: str = "in_progress",
    changes: list[object] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=item_id,
        type="fileChange",
        changes=changes
        or [
            SimpleNamespace(
                path="src/foo.py",
                diff="@@ -1 +1 @@\n-old\n+new",
                kind=SimpleNamespace(value="update"),
            )
        ],
        status=SimpleNamespace(value=status),
    )


def _make_mcp_item(
    *,
    item_id: str = "mcp_1",
    server: str = "mala-locking",
    tool: str = "lock_acquire",
    arguments: object = None,
    status: str = "in_progress",
    result: object = None,
    error: object = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=item_id,
        type="mcpToolCall",
        server=server,
        tool=tool,
        arguments=arguments if arguments is not None else {"filepaths": ["a.py"]},
        status=SimpleNamespace(value=status),
        result=result,
        error=error,
    )


def _wrap_root(item: object) -> SimpleNamespace:
    """Wrap an item in a pydantic-``RootModel``-style ``.root`` carrier."""
    return SimpleNamespace(root=item)


# ---------------------------------------------------------------------------
# D1 — agent message
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_agent_message_delta_yields_text_event() -> None:
    adapter = CodexEventAdapter()
    events = adapter.to_events(
        _Notification(
            method="item/agentMessage/delta",
            payload=SimpleNamespace(
                delta="hello",
                item_id="msg_1",
                thread_id="thr_1",
                turn_id="turn_1",
            ),
        )
    )
    assert events == [AgentTextEvent(text="hello", is_delta=True)]


@pytest.mark.unit
def test_empty_agent_message_delta_is_ignored_and_does_not_suppress_completed() -> None:
    adapter = CodexEventAdapter()
    delta_events = adapter.to_events(
        _Notification(
            method="item/agentMessage/delta",
            payload=SimpleNamespace(delta="", item_id="msg_1"),
        )
    )
    completed_events = adapter.to_events(
        _Notification(
            method="item/completed",
            payload=SimpleNamespace(
                item=_wrap_root(
                    SimpleNamespace(
                        id="msg_1", type="agentMessage", text="full message"
                    )
                ),
                thread_id="thr_1",
                turn_id="turn_1",
            ),
        )
    )

    assert delta_events == []
    assert completed_events == [AgentTextEvent(text="full message")]


@pytest.mark.unit
def test_agent_message_completed_after_delta_is_suppressed() -> None:
    """D1 dedup — completed item must not re-emit text already streamed."""
    adapter = CodexEventAdapter()
    adapter.to_events(
        _Notification(
            method="item/agentMessage/delta",
            payload=SimpleNamespace(delta="hi ", item_id="msg_1"),
        )
    )
    adapter.to_events(
        _Notification(
            method="item/agentMessage/delta",
            payload=SimpleNamespace(delta="there", item_id="msg_1"),
        )
    )

    completed = adapter.to_events(
        _Notification(
            method="item/completed",
            payload=SimpleNamespace(
                item=_wrap_root(
                    SimpleNamespace(id="msg_1", type="agentMessage", text="hi there")
                ),
                thread_id="thr_1",
                turn_id="turn_1",
            ),
        )
    )
    assert completed == []


@pytest.mark.unit
def test_agent_message_completed_without_prior_delta_emits_text() -> None:
    """D1 — completed item without delta is the only text source."""
    adapter = CodexEventAdapter()
    events = adapter.to_events(
        _Notification(
            method="item/completed",
            payload=SimpleNamespace(
                item=_wrap_root(
                    SimpleNamespace(
                        id="msg_2", type="agentMessage", text="full message"
                    )
                ),
                thread_id="thr_1",
                turn_id="turn_1",
            ),
        )
    )
    assert events == [AgentTextEvent(text="full message")]


# ---------------------------------------------------------------------------
# D2 — command execution
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_command_started_emits_bash_tool_use() -> None:
    adapter = CodexEventAdapter()
    events = adapter.to_events(
        _Notification(
            method="item/started",
            payload=SimpleNamespace(
                item=_wrap_root(
                    _make_command_item(
                        item_id="cmd_1",
                        command="ruff check .",
                        cwd="/repo",
                        status="in_progress",
                    )
                ),
                thread_id="thr_1",
                turn_id="turn_1",
            ),
        )
    )
    assert events == [
        AgentToolUseEvent(
            id="cmd_1",
            name="bash",
            input={"command": "ruff check .", "cwd": "/repo"},
        )
    ]


@pytest.mark.unit
def test_command_completed_success_emits_non_error_tool_result() -> None:
    adapter = CodexEventAdapter()
    events = adapter.to_events(
        _Notification(
            method="item/completed",
            payload=SimpleNamespace(
                item=_wrap_root(
                    _make_command_item(
                        item_id="cmd_1",
                        status="completed",
                        exit_code=0,
                        aggregated_output="ok\n",
                    )
                ),
                thread_id="thr_1",
                turn_id="turn_1",
            ),
        )
    )
    assert events == [
        AgentToolResultEvent(
            tool_use_id="cmd_1",
            is_error=False,
            content="ok\n",
        )
    ]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "exit_code", "is_error"),
    [
        ("completed", 0, False),
        ("completed", 1, True),
        ("failed", None, True),
        ("declined", None, True),
    ],
)
def test_command_completed_status_drives_is_error(
    status: str, exit_code: int | None, is_error: bool
) -> None:
    adapter = CodexEventAdapter()
    events = adapter.to_events(
        _Notification(
            method="item/completed",
            payload=SimpleNamespace(
                item=_wrap_root(_make_command_item(status=status, exit_code=exit_code)),
                thread_id="thr_1",
                turn_id="turn_1",
            ),
        )
    )
    assert len(events) == 1
    result = events[0]
    assert isinstance(result, AgentToolResultEvent)
    assert result.is_error is is_error


@pytest.mark.unit
def test_codex_bash_event_is_recognised_by_lint_cache() -> None:
    """AC #15 — a Codex bash tool-use event must be detectable as a lint command.

    The Phase A1 ``LintCacheProtocol`` consumes ``AgentToolUseEvent``s
    and pulls the command out of ``input["command"]``. Without the
    Codex adapter shaping the input dict the same way the Claude/Amp
    adapters do, the lint cache cannot mark Codex bash commands as
    lint runs and the cache silently degrades on Codex.
    """
    from src.infra.hooks.lint_cache import LintCache

    lint_cache = LintCache(lint_tools=frozenset({"ruff"}))

    adapter = CodexEventAdapter()
    events = adapter.to_events(
        _Notification(
            method="item/started",
            payload=SimpleNamespace(
                item=_wrap_root(
                    _make_command_item(
                        item_id="cmd_lint", command="ruff check .", cwd="/repo"
                    )
                ),
                thread_id="thr_1",
                turn_id="turn_1",
            ),
        )
    )
    assert len(events) == 1
    tool_use = events[0]
    assert isinstance(tool_use, AgentToolUseEvent)
    assert tool_use.name == "bash"
    command = tool_use.input.get("command", "")
    assert lint_cache.detect_lint_command(command) == "ruff"


# ---------------------------------------------------------------------------
# D3 — file change
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_file_change_started_emits_file_change_tool_use() -> None:
    adapter = CodexEventAdapter()
    events = adapter.to_events(
        _Notification(
            method="item/started",
            payload=SimpleNamespace(
                item=_wrap_root(
                    _make_file_change_item(item_id="fc_1", status="in_progress")
                ),
                thread_id="thr_1",
                turn_id="turn_1",
            ),
        )
    )
    assert len(events) == 1
    tool_use = events[0]
    assert isinstance(tool_use, AgentToolUseEvent)
    assert tool_use.name == "file_change"
    assert tool_use.id == "fc_1"
    changes = tool_use.input.get("changes")
    assert isinstance(changes, list)
    assert changes[0]["path"] == "src/foo.py"
    assert "+new" in changes[0]["diff"]
    assert changes[0]["kind"] == "update"


@pytest.mark.unit
def test_file_change_started_unwraps_patch_change_kind_root_model() -> None:
    # Real-SDK ``FileUpdateChange.kind`` is a ``PatchChangeKind`` RootModel
    # whose value is the ``type`` literal on its ``root`` variant; the
    # normalized ``changes`` payload must surface "update" rather than the
    # pydantic repr that ``str(kind)`` would produce.
    adapter = CodexEventAdapter()
    item = _make_file_change_item(
        item_id="fc_root",
        status="in_progress",
        changes=[
            SimpleNamespace(
                path="src/bar.py",
                diff="@@ -1 +1 @@\n-old\n+new",
                kind=SimpleNamespace(root=SimpleNamespace(type="update")),
            )
        ],
    )
    events = adapter.to_events(
        _Notification(
            method="item/started",
            payload=SimpleNamespace(
                item=_wrap_root(item),
                thread_id="thr_1",
                turn_id="turn_1",
            ),
        )
    )
    assert len(events) == 1
    tool_use = events[0]
    assert isinstance(tool_use, AgentToolUseEvent)
    changes = tool_use.input.get("changes")
    assert isinstance(changes, list)
    assert changes[0]["kind"] == "update"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "is_error"),
    [
        ("completed", False),
        ("failed", True),
        ("declined", True),
    ],
)
def test_file_change_completed_status_drives_is_error(
    status: str, is_error: bool
) -> None:
    adapter = CodexEventAdapter()
    events = adapter.to_events(
        _Notification(
            method="item/completed",
            payload=SimpleNamespace(
                item=_wrap_root(_make_file_change_item(status=status)),
                thread_id="thr_1",
                turn_id="turn_1",
            ),
        )
    )
    assert len(events) == 1
    result = events[0]
    assert isinstance(result, AgentToolResultEvent)
    assert result.tool_use_id == "fc_1"
    assert result.is_error is is_error


# ---------------------------------------------------------------------------
# D4 — MCP tool call
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mcp_started_emits_server_dot_tool_event() -> None:
    adapter = CodexEventAdapter()
    events = adapter.to_events(
        _Notification(
            method="item/started",
            payload=SimpleNamespace(
                item=_wrap_root(
                    _make_mcp_item(
                        item_id="mcp_1",
                        server="mala-locking",
                        tool="lock_acquire",
                        arguments={"filepaths": ["a.py"]},
                    )
                ),
                thread_id="thr_1",
                turn_id="turn_1",
            ),
        )
    )
    assert events == [
        AgentToolUseEvent(
            id="mcp_1",
            name="mala-locking.lock_acquire",
            input={"filepaths": ["a.py"]},
        )
    ]


@pytest.mark.unit
def test_mcp_completed_success_carries_result_payload() -> None:
    adapter = CodexEventAdapter()
    payload_result = SimpleNamespace(content=[{"type": "text", "text": "ok"}])
    events = adapter.to_events(
        _Notification(
            method="item/completed",
            payload=SimpleNamespace(
                item=_wrap_root(
                    _make_mcp_item(
                        item_id="mcp_1", status="completed", result=payload_result
                    )
                ),
                thread_id="thr_1",
                turn_id="turn_1",
            ),
        )
    )
    assert len(events) == 1
    result_event = events[0]
    assert isinstance(result_event, AgentToolResultEvent)
    assert result_event.is_error is False
    assert result_event.content is payload_result


@pytest.mark.unit
def test_mcp_completed_failure_carries_error_payload() -> None:
    adapter = CodexEventAdapter()
    err = SimpleNamespace(message="boom")
    events = adapter.to_events(
        _Notification(
            method="item/completed",
            payload=SimpleNamespace(
                item=_wrap_root(
                    _make_mcp_item(item_id="mcp_1", status="failed", error=err)
                ),
                thread_id="thr_1",
                turn_id="turn_1",
            ),
        )
    )
    assert len(events) == 1
    result_event = events[0]
    assert isinstance(result_event, AgentToolResultEvent)
    assert result_event.is_error is True
    assert result_event.content is err


# ---------------------------------------------------------------------------
# D5 — reasoning notifications drop silently
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "method",
    [
        "item/reasoning/textDelta",
        "item/reasoning/summaryTextDelta",
        "item/reasoning/summaryPartAdded",
    ],
)
def test_reasoning_notifications_emit_nothing(method: str) -> None:
    adapter = CodexEventAdapter()
    events = adapter.to_events(
        _Notification(
            method=method,
            payload=SimpleNamespace(
                delta="thinking...", item_id="r_1", thread_id="thr_1", turn_id="t1"
            ),
        )
    )
    assert events == []


@pytest.mark.unit
def test_reasoning_thread_item_completed_emits_nothing() -> None:
    """``ReasoningThreadItem`` arrives via ``item/completed`` and must drop."""
    adapter = CodexEventAdapter()
    reasoning_item = SimpleNamespace(
        id="r_1", type="reasoning", content=["thoughts"], summary=[]
    )
    events = adapter.to_events(
        _Notification(
            method="item/completed",
            payload=SimpleNamespace(
                item=_wrap_root(reasoning_item),
                thread_id="thr_1",
                turn_id="turn_1",
            ),
        )
    )
    assert events == []


# ---------------------------------------------------------------------------
# D6 — turn completed
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("status", "is_error"),
    [
        ("completed", False),
        ("failed", True),
        ("interrupted", True),
    ],
)
def test_turn_completed_status_drives_is_error(status: str, is_error: bool) -> None:
    adapter = CodexEventAdapter()
    turn = SimpleNamespace(
        id="turn_1", status=SimpleNamespace(value=status), error=None
    )
    events = adapter.to_events(
        _Notification(
            method="turn/completed",
            payload=SimpleNamespace(thread_id="thr_session", turn=turn),
        )
    )
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, AgentResultEvent)
    assert event.session_id == "thr_session"
    assert event.subtype == "completed"
    assert event.is_error is is_error
    # ``result`` carries the ``TurnStatus`` string so
    # ``AgentSessionOutput.summary`` renders human-readably; the full
    # ``Turn`` object would surface as opaque pydantic repr.
    assert event.result == status


@pytest.mark.unit
def test_failed_turn_completed_includes_error_message() -> None:
    """Failed ``turn/completed`` events keep provider error details visible."""
    adapter = CodexEventAdapter()
    turn = SimpleNamespace(
        id="turn_1",
        status=SimpleNamespace(value="failed"),
        error=SimpleNamespace(message="boom"),
    )
    events = adapter.to_events(
        _Notification(
            method="turn/completed",
            payload=SimpleNamespace(thread_id="thr_session", turn=turn),
        )
    )

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, AgentResultEvent)
    assert event.session_id == "thr_session"
    assert event.subtype == "completed"
    assert event.is_error is True
    assert event.result == "failed: boom"


# ---------------------------------------------------------------------------
# D7 — error notification (mid-turn provider error)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_error_notification_surfaces_terminal_failure() -> None:
    """D7: a mid-turn error must arrive as a result event, not be dropped.

    Without this, ``IdleTimeoutStream`` would never see a terminal
    event for the failed turn and the orchestrator would stall on the
    idle timeout instead of failing fast (plan ``L762``). The opaque
    ``message="boom"`` carries no auth/rate-limit/overload signal and
    therefore lands under the default ``"error"`` subtype.
    """
    adapter = CodexEventAdapter()
    err_obj = SimpleNamespace(message="boom")
    events = adapter.to_events(
        _Notification(
            method="error",
            payload=SimpleNamespace(
                error=err_obj,
                thread_id="thr_err",
                turn_id="turn_1",
                will_retry=False,
            ),
        )
    )
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, AgentResultEvent)
    assert event.session_id == "thr_err"
    assert event.subtype == "error"
    assert event.is_error is True
    assert event.result is err_obj


@pytest.mark.unit
@pytest.mark.parametrize(
    ("error_payload", "expected_subtype"),
    [
        # ``CodexErrorInfoValue`` enum-shaped payloads (the SDK delivers
        # ``codex_error_info.root`` as the enum directly; tests model that
        # via ``SimpleNamespace(value=...)``).
        pytest.param(
            SimpleNamespace(
                message="auth failed",
                codex_error_info=SimpleNamespace(
                    root=SimpleNamespace(value="unauthorized")
                ),
            ),
            "error_auth",
            id="enum_unauthorized",
        ),
        pytest.param(
            SimpleNamespace(
                message="usage limit",
                codex_error_info=SimpleNamespace(
                    root=SimpleNamespace(value="usageLimitExceeded")
                ),
            ),
            "error_rate_limit",
            id="enum_usage_limit_exceeded",
        ),
        pytest.param(
            SimpleNamespace(
                message="server busy",
                codex_error_info=SimpleNamespace(
                    root=SimpleNamespace(value="serverOverloaded")
                ),
            ),
            "error_overload",
            id="enum_server_overloaded",
        ),
        # Structured HTTP variants (e.g. ``HttpConnectionFailedCodexErrorInfo``):
        # ``codex_error_info.root`` is a Pydantic model whose nested field
        # exposes ``http_status_code``.
        pytest.param(
            SimpleNamespace(
                message="http error",
                codex_error_info=SimpleNamespace(
                    root=SimpleNamespace(
                        http_connection_failed=SimpleNamespace(http_status_code=401)
                    )
                ),
            ),
            "error_auth",
            id="http_status_401",
        ),
        pytest.param(
            SimpleNamespace(
                message="too many requests",
                codex_error_info=SimpleNamespace(
                    root=SimpleNamespace(
                        response_too_many_failed_attempts=SimpleNamespace(
                            http_status_code=429
                        )
                    )
                ),
            ),
            "error_rate_limit",
            id="http_status_429",
        ),
        pytest.param(
            SimpleNamespace(
                message="upstream issue",
                codex_error_info=SimpleNamespace(
                    root=SimpleNamespace(
                        response_stream_disconnected=SimpleNamespace(
                            http_status_code=503
                        )
                    )
                ),
            ),
            "error_overload",
            id="http_status_503",
        ),
        # Fallback substring matching on ``error.message`` only.
        pytest.param(
            SimpleNamespace(message="HTTP 401 Unauthorized"),
            "error_auth",
            id="message_401",
        ),
        pytest.param(
            SimpleNamespace(message="HTTP 429 rate limited"),
            "error_rate_limit",
            id="message_429",
        ),
        pytest.param(
            SimpleNamespace(message="Server overloaded"),
            "error_overload",
            id="message_overloaded",
        ),
        # Regression: bare numeric substrings must not absorb neighbouring
        # digits. Without word boundaries, ``"task 1429"`` would falsely
        # classify as ``error_rate_limit`` (gemini review-2 P1) and a
        # legitimate failure would be masked behind a retry/backoff path.
        pytest.param(
            SimpleNamespace(message="Failed to process task 1429"),
            "error",
            id="bare_digits_substring_does_not_match",
        ),
        pytest.param(
            SimpleNamespace(message="Trace id 50301 timed out"),
            "error",
            id="bare_digits_503_substring_does_not_match",
        ),
    ],
)
def test_error_notification_classifies_subtype(
    error_payload: object, expected_subtype: str
) -> None:
    """D7 classification — auth / rate-limit / overload signals drive ``subtype``.

    All classified subtypes start with ``"error_"`` so
    :meth:`MessageStreamProcessor._handle_result_event` still treats
    them as terminal failures; the specific subtype is metadata for
    diagnostics and retry logic.
    """
    adapter = CodexEventAdapter()
    events = adapter.to_events(
        _Notification(
            method="error",
            payload=SimpleNamespace(
                error=error_payload,
                thread_id="thr_err",
                turn_id="turn_1",
                will_retry=False,
            ),
        )
    )
    assert len(events) == 1
    event = events[0]
    assert isinstance(event, AgentResultEvent)
    assert event.is_error is True
    assert event.subtype == expected_subtype
    assert event.session_id == "thr_err"
    assert event.result is error_payload


@pytest.mark.unit
def test_error_mid_turn_then_unknown_followups_do_not_crash() -> None:
    """Edge case: error mid-turn then a stray unknown notification.

    ``CodexClient.receive_response`` keeps draining the SDK stream even
    after a fatal error notification; the adapter must not raise on
    methods it does not know.
    """
    adapter = CodexEventAdapter()
    err_events = adapter.to_events(
        _Notification(
            method="error",
            payload=SimpleNamespace(
                error=SimpleNamespace(message="boom"),
                thread_id="thr_err",
                turn_id="turn_1",
                will_retry=False,
            ),
        )
    )
    assert err_events and isinstance(err_events[0], AgentResultEvent)

    # Unknown method must not raise and must yield no events.
    unknown_events = adapter.to_events(
        _Notification(method="some/future/method", payload=SimpleNamespace())
    )
    assert unknown_events == []


# ---------------------------------------------------------------------------
# Defensive: malformed notification shapes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_notification_without_method_is_dropped() -> None:
    adapter = CodexEventAdapter()
    assert adapter.to_events(SimpleNamespace(method=None, payload=None)) == []
    assert adapter.to_events(SimpleNamespace(method="error", payload=None)) == []
    assert adapter.to_events(SimpleNamespace()) == []

"""E2E test for the real ``codex_app_server.AsyncCodex`` SDK (T019, AC #12).

Canary for Codex SDK schema drift: spins a real :class:`CodexAgentProvider`
through the orchestration factory (``coder=codex``), runs
:meth:`install_prerequisites` against the user's actual Codex install,
issues a one-line prompt against real ``AsyncCodex``, and asserts the
``AgentEvent`` stream terminates with a successful ``AgentResultEvent``
plus replayable evidence in the per-thread tee.

Gate (mirrors ``tests/e2e/test_amp_real_cli.py``'s posture — single skip
condition, no opt-in env var):

  * ``codex_app_server`` SDK importable, AND
  * Codex runtime resolvable (``codex`` on PATH, ``CODEX_BINARY`` set, or
    bundled ``codex_cli_bin`` package importable), AND
  * Codex auth detectable (``OPENAI_API_KEY`` / ``CODEX_API_KEY`` /
    ``CODEX_ACCESS_TOKEN`` env var, or ``$CODEX_HOME/auth.json`` on disk).

The runtime + auth probes reuse the helpers ``install_prerequisites``
itself uses (:func:`_codex_runtime_resolvable`, :func:`_codex_auth_present`)
so the test gate and the production fail-closed gate cannot diverge.

Out of scope: full per-issue beads workflow e2e (covered by manual
verification per plan ``L1244``); this test is the SDK-drift smoke gate
only.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from typing import TYPE_CHECKING

import pytest

from src.core.protocols.agent_event import (
    AgentResultEvent,
    AgentTextEvent,
    AgentToolResultEvent,
    AgentToolUseEvent,
)
from src.infra.clients.codex_provider import (
    CodexAgentProvider,
    _codex_auth_present,
    _codex_runtime_resolvable,
)

if TYPE_CHECKING:
    from pathlib import Path


_E2E_PROMPT = (
    "Use the shell tool to run exactly this command:\n"
    "echo hello world\n"
    "\n"
    "After the command completes, reply with exactly: CODEX_E2E_OK\n"
    "Do not edit files. Do not run any other command.\n"
)
"""One-line shell command + sentinel reply; minimal cost + surface area."""


_RATE_LIMIT_BACKOFF_SECONDS = 30.0
"""Brief pause before the single retry on a rate-limit terminal error."""


_TURN_TIMEOUT_SECONDS = 180.0
"""Per-attempt upper bound for one full Codex turn (start + stream + terminal).

Bounds the SDK-drift canary so a stalled or non-terminating
``AsyncCodex`` notification stream times out into a test failure /
``TimeoutError`` rather than hanging the CI job indefinitely. There is
no global pytest timeout configured; without this wrapper, a missing
terminal notification (the very SDK drift this canary is designed to
catch) would manifest as an indefinite hang. Generous bound — a
one-line ``echo`` turn typically completes in well under 60s, but the
ceiling absorbs cold-start jitter on the SDK's bundled
``codex_cli_bin`` runtime.
"""


_RATE_LIMIT_HINTS: tuple[str, ...] = (
    "rate limit",
    "rate_limit",
    "ratelimit",
    "429",
    "too many requests",
    "quota",
)
"""Substrings that indicate a Codex turn ended on a rate-limit failure.

Substring match across the lower-cased ``str(result)`` of the terminal
``AgentResultEvent``. Codex SDK does not yet expose a structured
``error_code`` on the terminal notification; widening to a small set of
hints keeps the retry-on-rate-limit branch resilient to upstream wording
changes without claiming false positives on unrelated errors.
"""


def _require_codex_e2e_env() -> None:
    """Skip the test when the Codex SDK / runtime / auth gate is incomplete.

    Mirrors the three-part gate :meth:`CodexAgentProvider.install_prerequisites`
    enforces in production so a divergence between the test-skip path and
    the production fail-closed path is impossible.
    """
    try:
        sdk_spec = importlib.util.find_spec("codex_app_server")
    except (ImportError, ValueError):
        sdk_spec = None
    if sdk_spec is None:
        pytest.skip(
            "codex_app_server SDK not importable; install "
            "openai-codex-app-server-sdk to run this test"
        )

    if not _codex_runtime_resolvable():
        pytest.skip(
            "Codex runtime not resolvable: no `codex` on PATH, no "
            "`CODEX_BINARY` override, and no bundled `codex_cli_bin` "
            "package"
        )

    if not _codex_auth_present():
        pytest.skip(
            "Codex auth not detected: set `OPENAI_API_KEY` / "
            "`CODEX_API_KEY` / `CODEX_ACCESS_TOKEN` or run `codex login`"
        )


def _empty_mcp_server_factory(
    agent_id: str,
    repo_path: Path,
    emit_lock_event: object | None,
) -> dict[str, object]:
    """Return no MCP servers for the minimal real-Codex e2e session.

    Keeps the gated test from needing the bundled ``mala-locking`` MCP
    spawn just to assert SDK-schema fidelity. The real bundled MCP wiring
    is exercised by :class:`CodexAgentProvider.mcp_server_factory`'s unit
    coverage.
    """
    del agent_id, repo_path, emit_lock_event
    return {}


def _result_text(result_event: AgentResultEvent) -> str:
    """Stringify an :class:`AgentResultEvent`'s payload for substring search.

    The terminal event's ``result`` is provider-shaped (Codex emits a
    ``TurnCompleted`` notification or an ``ErrorNotification`` payload);
    we coerce to ``str`` for the rate-limit substring scan rather than
    introspecting a moving SDK shape.
    """
    return str(result_event.result or "")


def _looks_like_rate_limit(result_event: AgentResultEvent) -> bool:
    """Return True iff the terminal error reads as a rate-limit failure."""
    if not result_event.is_error:
        return False
    haystack = (_result_text(result_event) + " " + result_event.subtype).lower()
    return any(hint in haystack for hint in _RATE_LIMIT_HINTS)


async def _run_one_codex_turn(
    provider: CodexAgentProvider,
    repo_path: Path,
    agent_id: str,
    prompt: str,
) -> tuple[
    str | None,
    AgentResultEvent | None,
    list[AgentTextEvent],
    list[AgentToolUseEvent],
    list[AgentToolResultEvent],
]:
    """Drive one CodexClient turn end-to-end and collect typed events.

    Returns the resolved Codex thread id, the terminal
    :class:`AgentResultEvent` (or ``None`` if the SDK never emitted one),
    and the per-kind buckets of intermediate events.
    """
    from src.infra.clients.codex_client import CodexClient
    from src.infra.clients.codex_runtime import CodexRuntime, CodexRuntimeBuilder

    builder = provider.runtime_builder(
        repo_path,
        agent_id,
        mcp_server_factory=_empty_mcp_server_factory,
    )
    assert isinstance(builder, CodexRuntimeBuilder)
    runtime = builder.build()
    assert isinstance(runtime, CodexRuntime)
    client = provider.client_factory.create(runtime)
    assert isinstance(client, CodexClient)

    text_events: list[AgentTextEvent] = []
    tool_use_events: list[AgentToolUseEvent] = []
    tool_result_events: list[AgentToolResultEvent] = []
    result_event: AgentResultEvent | None = None

    async with client:
        await client.query(prompt)
        async for event in client.receive_response():
            if isinstance(event, AgentTextEvent):
                text_events.append(event)
            elif isinstance(event, AgentToolUseEvent):
                tool_use_events.append(event)
            elif isinstance(event, AgentToolResultEvent):
                tool_result_events.append(event)
            elif isinstance(event, AgentResultEvent):
                result_event = event
                break

        thread_id = client.session_id

    return thread_id, result_event, text_events, tool_use_events, tool_result_events


@pytest.mark.e2e
@pytest.mark.flaky_sdk
async def test_codex_real_sdk_one_line_prompt_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real ``AsyncCodex`` round-trip: install + query + evidence replay.

    Asserts (AC #12 — SDK-drift detectable before merge):

      * :meth:`CodexAgentProvider.install_prerequisites` succeeds against
        the user's real Codex install (no exception).
      * A one-line prompt drives ``AsyncCodex.thread_start`` →
        ``thread.turn`` → notification stream to terminal completion.
      * The terminal :class:`AgentResultEvent` carries
        ``is_error=False`` (with a rate-limit retry-once + skip-with-message
        fallback so transient quota errors do not flake CI).
      * The per-thread tee'd JSONL written by :class:`CodexClient` is
        readable by :meth:`CodexEvidenceProvider.iter_thread_evidence`
        and contains the executed ``echo hello world`` command plus
        captured ``hello world`` output. This is the cross-resume
        evidence contract Phase F (T013) ships; verifying it here pins
        the live SDK's notification shape against
        :class:`CodexEvidenceProvider`'s parser so a payload-shape
        change in either direction surfaces as a test failure.
    """
    _require_codex_e2e_env()

    # Tee the per-thread JSONL into ``tmp_path`` so the test does not
    # touch ``~/.config/mala/codex-sessions/`` and tee writes from
    # earlier interrupted runs cannot pollute this thread's evidence.
    # ``CodexClient`` and :class:`CodexEvidenceProvider` both honor
    # ``MALA_CODEX_SESSIONS_DIR`` via :func:`_resolve_tee_dir` /
    # ``CodexEvidenceProvider(sessions_dir=...)``.
    tee_dir = tmp_path / "codex-sessions"
    monkeypatch.setenv("MALA_CODEX_SESSIONS_DIR", str(tee_dir))

    provider = CodexAgentProvider()
    repo_path = tmp_path / "codex-e2e-repo"
    repo_path.mkdir()
    agent_id = "codex-e2e-agent"

    # install_prerequisites runs the full fail-closed gate (SDK +
    # runtime + auth + plugin install + hook self-test). On a properly
    # configured machine with mala installed via ``uv tool install
    # mala-agent`` / ``uv sync`` this is silent; any failure here means
    # the local install is incomplete and the rest of the test would
    # fail in a less informative way.
    provider.install_prerequisites(
        repo_path,
        mcp_server_factory=_empty_mcp_server_factory,
    )

    print(
        f"[codex-e2e] CODEX_HOME={os.environ.get('CODEX_HOME', '<unset>')}; "
        f"runtime_resolvable={_codex_runtime_resolvable()}; "
        f"auth_present={_codex_auth_present()}",
        file=sys.stderr,
    )

    (
        thread_id,
        result_event,
        text_events,
        tool_use_events,
        tool_result_events,
    ) = await asyncio.wait_for(
        _run_one_codex_turn(provider, repo_path, agent_id, _E2E_PROMPT),
        timeout=_TURN_TIMEOUT_SECONDS,
    )

    # Retry-on-rate-limit (plan I3 risk-mitigation): one short backoff +
    # one retry, then skip-with-message so a transient quota throttle
    # never marks the SDK-drift canary as a regression.
    if result_event is not None and _looks_like_rate_limit(result_event):
        print(
            f"[codex-e2e] first attempt hit rate limit ({_result_text(result_event)!r}); "
            f"retrying once after {_RATE_LIMIT_BACKOFF_SECONDS}s",
            file=sys.stderr,
        )
        await asyncio.sleep(_RATE_LIMIT_BACKOFF_SECONDS)
        (
            thread_id,
            result_event,
            text_events,
            tool_use_events,
            tool_result_events,
        ) = await asyncio.wait_for(
            _run_one_codex_turn(provider, repo_path, agent_id + "-retry", _E2E_PROMPT),
            timeout=_TURN_TIMEOUT_SECONDS,
        )
        if result_event is not None and _looks_like_rate_limit(result_event):
            pytest.skip(
                "Codex rate-limited twice in a row; skipping SDK-drift canary "
                f"(last result: {_result_text(result_event)!r})"
            )

    assert result_event is not None, (
        "CodexClient.receive_response() never emitted an AgentResultEvent — "
        "this is the most explicit SDK-drift signal possible: the live "
        "notification stream did not produce a terminal event mappable to "
        "AgentResultEvent. Check codex_event_adapter.py against the SDK's "
        "current Notification methods."
    )
    assert not result_event.is_error, (
        "Codex turn ended with is_error=True. result="
        f"{_result_text(result_event)!r}; subtype={result_event.subtype!r}"
    )
    assert thread_id is not None and thread_id, (
        "CodexClient.session_id was empty after a completed turn; the SDK "
        "did not surface a usable thread id."
    )
    # Codex thread ids are documented as ``thr_<...>`` (see
    # ``codex_app_server`` Thread.id); a prefix change is itself SDK-drift
    # the canary should surface, so assert it strictly rather than
    # short-circuiting through an ``or thread_id`` clause that would
    # silently accept any non-empty string.
    assert thread_id.startswith("thr_"), (
        f"Codex thread id has unexpected shape: {thread_id!r} (expected `thr_` prefix)"
    )
    print(
        f"[codex-e2e] thread_id={thread_id}; "
        f"text_events={len(text_events)}; "
        f"tool_use_events={len(tool_use_events)}; "
        f"tool_result_events={len(tool_result_events)}",
        file=sys.stderr,
    )

    # Evidence replay (Phase F / T013): the tee'd JSONL must contain
    # the executed shell command and its captured output so cross-resume
    # validation gates can replay invocation 1's evidence after a
    # ``thread_resume``.
    from src.infra.clients.codex_evidence_provider import CodexEvidenceProvider

    evidence_provider = CodexEvidenceProvider(sessions_dir=tee_dir)
    log_path = evidence_provider.get_log_path(repo_path, thread_id)
    assert log_path.exists(), (
        f"Per-thread tee file was not created at {log_path}; "
        "CodexClient did not tee the notification stream."
    )

    found_command = False
    found_output = False
    for entry in evidence_provider.iter_thread_evidence(log_path):
        for _item_id, command in evidence_provider.extract_bash_commands(entry):
            if "echo hello world" in command:
                found_command = True
        for _item_id, content in evidence_provider.extract_tool_result_content(entry):
            if "hello world" in content:
                found_output = True

    assert found_command, (
        "Tee'd JSONL did not contain a `echo hello world` Bash invocation. "
        "Either Codex did not run the shell tool (prompt drift), or the "
        "live `commandExecution` notification shape no longer matches "
        "CodexEvidenceProvider.extract_bash_commands."
    )
    assert found_output, (
        "Tee'd JSONL did not contain `hello world` in any "
        "commandExecution.aggregated_output. The live SDK may have "
        "changed the field name or stopped surfacing aggregated output "
        "for completed bash items — CodexEvidenceProvider.extract_tool_result_content "
        "depends on this contract for cross-resume gate evidence."
    )

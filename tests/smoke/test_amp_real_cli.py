"""Path-gated CI smoke test for real ``amp --execute --stream-json``.

Owns AC#12: catches upstream stream-json schema drift before merge for
PRs that touch Amp client / plugin / plan code paths.

Plan: ``plans/2026-04-29-amp-provider-plan.md`` L66-L70, L870-L876, L913.

Schema fields asserted (the minimum surface mala depends on):

* every event has ``event.type``
* the ``system`` init event carries ``session_id``
* ``assistant`` events carry ``message.content[].type``

The test deliberately uses a trivial prompt to bound cost — the goal is
schema validation, not behavior validation. The real-CLI test is skipped
(rather than failed) when the gating preconditions are not met, so the
CI job stays green when ``AMP_API_KEY`` is absent (e.g. on PRs from
forks). A unit-marked regression test for the parser helper runs in the
default test suite so the parsing contract is exercised on every PR.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from typing import Any, cast

import pytest


_SMOKE_PROMPT = "Reply with the single word: ok\n"
"""Trivial one-line prompt. Plan L874 calls for a one-line prompt to bound
cost; the assertion is on schema, not the model's reply text."""

_SUBPROCESS_TIMEOUT_SECONDS = 120.0
"""Generous bound for a tiny prompt. Real Amp invocations typically
complete in <30s; the wider window absorbs cold-start jitter."""


def _missing_preconditions() -> str | None:
    """Return a human-readable skip reason or ``None`` if ready to run."""
    if not os.environ.get("AMP_API_KEY"):
        return "AMP_API_KEY not set; skipping real-Amp smoke test."
    if shutil.which("amp") is None:
        return "amp binary not on PATH; skipping real-Amp smoke test."
    return None


def _run_amp_version() -> str:
    """Best-effort capture of the observed Amp version string.

    The smoke job reports this in its log so reviewers can correlate
    schema drift with an upstream release. Failure to read the version
    is non-fatal — the schema assertions are the contract.
    """
    try:
        completed = subprocess.run(
            ["amp", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"<version-unavailable: {exc!r}>"
    out = (completed.stdout or "").strip() or (completed.stderr or "").strip()
    return out or "<version-unavailable: empty output>"


class StreamJsonParseError(AssertionError):
    """Raised when stdout contains a line that is not valid stream-json.

    Mirrors :class:`~src.infra.clients.amp_client.AmpClientError` raised at
    ``src/infra/clients/amp_client.py:435-442``: a banner, progress line,
    or any other non-JSON stdout content crashes a real Amp session, so
    the smoke test must fail on the same input. This is the schema-drift
    signal AC#12 needs to surface before merge.
    """


def _parse_stream_lines(stdout_text: str) -> list[dict[str, Any]]:
    """Parse Amp ``--stream-json`` stdout, mirroring ``AmpClient``'s contract.

    Behavior matches ``src/infra/clients/amp_client.py:432-448`` exactly:

    * Empty lines (after ``strip()``) are skipped.
    * Lines that fail JSON decoding raise :class:`StreamJsonParseError`.
      ``AmpClient`` raises ``AmpClientError`` on the same input, so a
      banner / progress line on stdout would crash a real session. The
      smoke test must reject it instead of silently skipping.
    * Non-dict JSON values are warn-only (matching ``AmpClient``'s
      ``logger.warning`` path) and skipped — they cannot satisfy mala's
      schema and the schema assertions will fail later if they were the
      only content on stdout.
    """
    events: list[dict[str, Any]] = []
    for raw_line in stdout_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise StreamJsonParseError(
                "amp stdout contained a non-JSON line — likely upstream "
                "stream-json drift (banner/progress emitted on stdout). "
                "AmpClient raises AmpClientError on this same input at "
                f"src/infra/clients/amp_client.py:435-442. Offending "
                f"line: {line!r}; json error: {exc}"
            ) from exc
        if not isinstance(obj, dict):
            print(
                f"[amp-smoke] warn: stream-json line is not an object: {obj!r}",
                file=sys.stderr,
            )
            continue
        events.append(obj)
    return events


def _run_amp_stream_json(prompt: str) -> list[dict[str, Any]]:
    """Spawn real ``amp --execute --stream-json`` and collect parsed events.

    Delegates parsing to :func:`_parse_stream_lines` so the smoke test's
    line-acceptance contract is identical to the production parser.
    """
    completed = subprocess.run(
        ["amp", "--execute", "--stream-json"],
        input=prompt,
        check=False,
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT_SECONDS,
    )
    print(f"[amp-smoke] exit_code={completed.returncode}", file=sys.stderr)
    if completed.stderr:
        print(f"[amp-smoke] stderr:\n{completed.stderr}", file=sys.stderr)
    return _parse_stream_lines(completed.stdout or "")


@pytest.mark.smoke
def test_amp_real_cli_stream_json_schema() -> None:
    """Assert the stream-json schema fields mala depends on are present.

    Skips when ``AMP_API_KEY`` is absent or ``amp`` is not on PATH so the
    CI job is green when the secret is unavailable (e.g. fork PRs).
    """
    skip_reason = _missing_preconditions()
    if skip_reason:
        pytest.skip(skip_reason)

    version = _run_amp_version()
    print(f"[amp-smoke] amp --version: {version}", file=sys.stderr)

    events = _run_amp_stream_json(_SMOKE_PROMPT)
    assert events, "amp --execute --stream-json produced no parseable events"

    # Schema field 1: every event has a string ``type``.
    for idx, event in enumerate(events):
        assert "type" in event, f"event[{idx}] missing 'type': {event!r}"
        assert isinstance(event["type"], str), (
            f"event[{idx}].type is not a string: {event!r}"
        )

    observed_types = sorted({str(e["type"]) for e in events})
    print(f"[amp-smoke] observed event.type values: {observed_types}", file=sys.stderr)

    # Schema field 2: the system init event carries ``session_id``.
    system_inits = [
        e for e in events if e.get("type") == "system" and e.get("subtype") == "init"
    ]
    if not system_inits:
        # Some Amp versions emit the init under just type=="system" without a
        # subtype. Accept the broader shape so the assertion focuses on
        # session_id, which mala consumes on first event regardless of subtype.
        system_inits = [e for e in events if e.get("type") == "system"]
    assert system_inits, (
        f"no 'system' event found in stream — observed types: {observed_types}"
    )
    init = system_inits[0]
    assert "session_id" in init, f"system init event missing 'session_id': {init!r}"
    assert isinstance(init["session_id"], str) and init["session_id"], (
        f"system init 'session_id' is not a non-empty string: {init!r}"
    )
    print(
        f"[amp-smoke] session_id captured: {init['session_id']!r}",
        file=sys.stderr,
    )

    # Schema field 3: assistant events carry message.content[].type.
    assistant_events = [e for e in events if e.get("type") == "assistant"]
    assert assistant_events, (
        f"no 'assistant' event found in stream — observed types: {observed_types}"
    )
    saw_typed_block = False
    for evt in assistant_events:
        message: Any = evt.get("message")
        assert isinstance(message, dict), (
            f"assistant event missing dict 'message': {evt!r}"
        )
        content: Any = message.get("content")
        assert isinstance(content, list), (
            f"assistant.message.content is not a list: {evt!r}"
        )
        for block_idx, block in enumerate(content):
            assert isinstance(block, dict), (
                f"assistant content[{block_idx}] is not an object: {block!r}"
            )
            block_dict = cast("dict[str, Any]", block)
            assert "type" in block_dict, (
                f"assistant content[{block_idx}] missing 'type': {block!r}"
            )
            block_type: Any = block_dict["type"]
            assert isinstance(block_type, str), (
                f"assistant content[{block_idx}].type is not a string: {block!r}"
            )
            saw_typed_block = True
    assert saw_typed_block, "no typed content blocks observed in any assistant event"


# ---------------------------------------------------------------------------
# Parser regression tests — run in the default suite so the smoke parser's
# schema-drift contract is exercised on every PR (not just when AMP_API_KEY
# is configured). Marked ``unit`` explicitly so they bypass the path-based
# ``smoke`` auto-marker applied by ``tests/conftest.py``.
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_parse_stream_lines_fails_on_non_json_stdout() -> None:
    """Banner / progress lines on stdout must fail the smoke test.

    AmpClient raises AmpClientError on a JSONDecodeError
    (``src/infra/clients/amp_client.py:435-442``); the smoke parser must
    do the same so AC#12 actually catches this kind of upstream drift.
    """
    drifted = (
        "Starting amp v9.9.9...\n"
        + json.dumps({"type": "system", "subtype": "init", "session_id": "abc"})
        + "\n"
        + json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "ok"}]},
            }
        )
        + "\n"
    )
    with pytest.raises(StreamJsonParseError, match="non-JSON"):
        _parse_stream_lines(drifted)


@pytest.mark.unit
def test_parse_stream_lines_accepts_clean_stream() -> None:
    """Sanity check: a well-formed stream parses without raising."""
    clean = (
        "\n"
        + json.dumps({"type": "system", "subtype": "init", "session_id": "abc"})
        + "\n"
        + "\n"
        + json.dumps(
            {
                "type": "assistant",
                "message": {"content": [{"type": "text", "text": "ok"}]},
            }
        )
        + "\n"
    )
    events = _parse_stream_lines(clean)
    assert [e["type"] for e in events] == ["system", "assistant"]


@pytest.mark.unit
def test_parse_stream_lines_warns_on_non_object_json(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Non-dict JSON values match AmpClient's warn-and-skip behavior.

    AmpClient logs a warning for non-object stream-json events at
    ``src/infra/clients/amp_client.py:443-448`` and continues; the smoke
    parser mirrors that so the schema assertions, not the parser, are
    what reject malformed streams.
    """
    mixed = (
        "[1, 2, 3]\n"
        + json.dumps({"type": "system", "subtype": "init", "session_id": "abc"})
        + "\n"
    )
    events = _parse_stream_lines(mixed)
    assert len(events) == 1
    assert events[0]["type"] == "system"
    captured = capsys.readouterr()
    assert "not an object" in captured.err

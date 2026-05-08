"""Phase F1 spike — validate ``Thread.read(include_turns=True)`` for evidence.

Validates decision #11 of plans/2026-05-07-codex-provider-plan.md by exercising
the Codex SDK's :meth:`codex_app_server.Thread.read` against a real Codex
runtime and reporting:

1. ``CommandExecutionThreadItem.aggregated_output`` populated for completed bash
   turns (so Mala lint/build evidence is observable).
2. Per-call ``thread/read`` latency, repeated in a tight loop, as a baseline for
   gate-loop polling cost.
3. Cross-resume completeness — items from both pre- and post-``thread_resume``
   turns are present in a single ``Thread.read(include_turns=True)`` response.

Skip semantics
--------------

The runtime path requires:

- ``codex_app_server`` SDK importable (``uv add openai-codex-app-server-sdk``);
- ``codex`` binary on ``PATH`` (or ``CODEX_BINARY`` set);
- a logged-in Codex auth state on the host.

When any of those is missing the test module is skipped, leaving the recorded
spike outcome (see the *Spike result* section below and the Phase F1 entry in
plans/2026-05-07-codex-provider-plan.md Open Questions) as the authoritative
finding for blocking T013. The test is reachable in CI / locally once the
dependencies land — it is intentionally e2e-marked so it does not run in the
default ``unit or integration`` selector.

Invocation::

    uv run pytest tests/spike/test_codex_thread_read_evidence.py -m e2e

Spike result
------------

**Disconfirmed** under default ``persist_extended_history=False`` semantics, by
SDK source review at ``~/code/codex/codex-rs`` (commit local checkout, pre-T012
landing). Summary:

* ``app-server/src/request_processors/thread_processor.rs:5-8`` declares
  ``"persistExtendedHistory is deprecated and ignored. App-server always uses
  limited history persistence."`` — the param exists but is a no-op.
* ``app-server/src/request_processors.rs:506`` builds ``thread/read`` turns via
  ``is_persisted_rollout_item(item, EventPersistenceMode::Limited)``.
* ``rollout/src/policy.rs:111-131`` lists ``EventMsg::ExecCommandEnd`` under
  ``EventPersistenceMode::Extended`` only — so ``ExecCommandEnd`` is filtered
  out of the rollout file in the (mandatory) Limited mode.
* ``app-server-protocol/src/protocol/item_builders.rs:108-133`` builds
  ``CommandExecutionThreadItem.aggregated_output`` *only* from
  ``ExecCommandEndEvent.aggregated_output``; the Begin event sets it to
  ``None``. With End events filtered, replay yields ``aggregated_output: None``
  for every completed bash turn.
* Cross-resume: the ``ThreadRollbackResponse.thread`` schema docstring states
  items are "lossy since we explicitly do not persist all agent interactions,
  such as command executions. This is the same behavior as ``thread/resume``."
  ``thread/read`` reads the same rollout, so the loss is shared.

**Action**: T013 implements F3 tee fallback at
``~/.config/mala/codex-sessions/{thread_id}.jsonl`` rather than native
``Thread.read``. The decision is recorded in plans/2026-05-07-codex-provider-plan.md
Open Questions. The runtime checks below remain so the finding can be
reconfirmed against future Codex SDK releases (an upstream change that
re-enables Extended persistence — or any equivalent — would let us revisit).
"""

from __future__ import annotations

import os
import shutil
import time
from typing import Any

import pytest

pytestmark = pytest.mark.e2e

codex_app_server = pytest.importorskip(
    "codex_app_server",
    reason="codex_app_server SDK not installed — recorded spike outcome stands "
    "(see module docstring 'Spike result' section).",
)


_LATENCY_SAMPLE_COUNT = 5
_HELLO_PAYLOAD = "spike-hello-aggregated-output"
_RESUME_PAYLOAD = "spike-second-turn-after-resume"


def _require_codex_binary() -> None:
    binary = os.environ.get("CODEX_BINARY") or shutil.which("codex")
    if binary is None:
        pytest.skip(
            "codex binary not found on PATH and CODEX_BINARY unset — runtime "
            "path unverifiable; recorded spike outcome stands."
        )


def _new_codex() -> Any:  # noqa: ANN401 — SDK is optional dep; staticly type-erased
    """Construct a synchronous ``Codex`` client; skip on auth-not-ready errors.

    The SDK constructor or ``__enter__`` raises an auth error class when no
    logged-in session exists. We catch broadly because the exact class is part
    of the Phase I auth-detection Open Question — for the spike, any startup
    failure surfaces as "runtime not available, skip and trust the recorded
    finding."
    """
    codex_cls = codex_app_server.Codex
    try:
        client = codex_cls()
    except Exception as exc:
        pytest.skip(f"Codex client construction failed (auth/runtime?): {exc!r}")
    return client


def _items_of(thread_read_response: Any) -> list:  # noqa: ANN401 — see _new_codex
    """Flatten ``ThreadReadResponse.thread.turns[*].items`` into one list."""
    items: list = []
    for turn in thread_read_response.thread.turns or []:
        items.extend(turn.items or [])
    return items


def _command_items_with_aggregated_output(items: list) -> list:
    """Items that are ``CommandExecutionThreadItem`` with ``aggregated_output``."""
    return [
        item
        for item in items
        if getattr(item, "type", None) == "commandExecution"
        and getattr(item, "aggregated_output", None)
    ]


def test_aggregated_output_present_for_bash_command() -> None:
    """Finding (1): ``CommandExecutionThreadItem.aggregated_output`` is populated.

    Expected outcome under current Codex SDK semantics: this assertion FAILS
    because ``persist_extended_history`` is hardcoded-Limited and
    ``ExecCommandEnd`` is filtered out of the rollout (see module docstring
    Spike result). When that fails, the spike is disconfirmed and T013 must
    use the F3 tee fallback. The assertion is intentionally written so a
    future SDK that re-enables Extended persistence would flip this test to
    pass without the spike needing to be rewritten.
    """
    _require_codex_binary()
    with _new_codex() as codex:
        thread = codex.thread_start()
        codex.turn_start(
            thread.id,
            input=f"Run this single bash command and nothing else: echo {_HELLO_PAYLOAD}",
        )
        response = thread.read(include_turns=True)

    items = _items_of(response)
    matches = _command_items_with_aggregated_output(items)
    assert any(_HELLO_PAYLOAD in (item.aggregated_output or "") for item in matches), (
        "F1 disconfirmed: Thread.read(include_turns=True) did not surface "
        "CommandExecutionThreadItem.aggregated_output for the executed bash "
        "command. Decision #11 must reverse to F3 tee fallback. See module "
        "docstring 'Spike result' for the SDK-source reasoning that predicted "
        "this outcome."
    )


def test_thread_read_latency_baseline_under_repeat() -> None:
    """Finding (2): per-call ``thread/read`` latency baseline.

    Records p95-ish latency over a small loop. The test does not gate on a
    hard threshold — the goal is to surface a number we can paste into the
    plan. We only fail if any call exceeds a deliberately-loose 10s budget
    (something is catastrophically wrong if a single read takes longer).
    """
    _require_codex_binary()
    samples_ms: list[float] = []
    with _new_codex() as codex:
        thread = codex.thread_start()
        codex.turn_start(thread.id, input=f"echo {_HELLO_PAYLOAD}")
        for _ in range(_LATENCY_SAMPLE_COUNT):
            t0 = time.monotonic()
            thread.read(include_turns=True)
            samples_ms.append((time.monotonic() - t0) * 1000.0)

    samples_ms.sort()
    median_ms = samples_ms[len(samples_ms) // 2]
    max_ms = samples_ms[-1]
    print(
        f"[F1 spike] thread/read latency over {_LATENCY_SAMPLE_COUNT} calls: "
        f"median={median_ms:.1f}ms max={max_ms:.1f}ms samples_ms={samples_ms}"
    )
    assert max_ms < 10_000.0, (
        f"thread/read exceeded 10s budget on at least one call (max={max_ms:.1f}ms); "
        "investigate before relying on it for repeated gate evidence."
    )


def test_cross_resume_completeness() -> None:
    """Finding (3): post-``thread_resume`` ``Thread.read`` includes pre-resume items.

    Expected outcome: items from the pre-resume turn that depended on
    ``ExecCommandEnd`` payload (i.e. ``aggregated_output``) are missing,
    matching the schema-documented "ThreadItems stored in each Turn are lossy
    ... same behavior as thread/resume" caveat on
    :class:`ThreadRollbackResponse`. The test asserts the lossless ideal so a
    future SDK fix would pass; under current semantics it fails and confirms
    the disconfirmation.
    """
    _require_codex_binary()
    with _new_codex() as codex:
        thread = codex.thread_start()
        codex.turn_start(thread.id, input=f"echo {_HELLO_PAYLOAD}")
        thread_id = thread.id

    with _new_codex() as codex:
        resumed = codex.thread_resume(thread_id)
        codex.turn_start(resumed.thread.id, input=f"echo {_RESUME_PAYLOAD}")
        response = resumed.thread.read(include_turns=True)

    items = _items_of(response)
    aggregated_blobs = [
        getattr(item, "aggregated_output", None) or "" for item in items
    ]
    pre_present = any(_HELLO_PAYLOAD in blob for blob in aggregated_blobs)
    post_present = any(_RESUME_PAYLOAD in blob for blob in aggregated_blobs)
    assert pre_present and post_present, (
        "F1 disconfirmed: cross-resume Thread.read missed at least one of "
        f"(pre={pre_present}, post={post_present}). Decision #11 must reverse "
        "to F3 tee fallback so evidence survives thread_resume."
    )

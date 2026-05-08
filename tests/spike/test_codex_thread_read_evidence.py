"""Phase F1 spike — validate ``Thread.read(include_turns=True)`` for evidence.

Validates decision #11 of plans/2026-05-07-codex-provider-plan.md by exercising
the Codex SDK's :meth:`codex_app_server.Thread.read` against a real Codex
runtime and reporting:

1. ``CommandExecutionThreadItem.aggregated_output`` populated for completed bash
   turns (so Mala lint/build evidence is observable).
2. Per-call ``thread/read`` latency, repeated in a tight loop, as a baseline for
   gate-loop polling cost.
3. Cross-resume completeness — items from pre-resume turns (one bash, one file
   edit) and a post-resume bash turn are all present in a single
   ``Thread.read(include_turns=True)`` response.

Skip semantics
--------------

The runtime path requires:

- ``codex_app_server`` SDK importable (``uv add openai-codex-app-server-sdk``);
- ``codex`` binary on ``PATH`` (or ``CODEX_BINARY`` set);
- a logged-in Codex auth state on the host.

When any of those is missing the test module is skipped, leaving the recorded
spike outcome (see the *Spike result* section below and the Phase F1 entry in
plans/2026-05-07-codex-provider-plan.md Open Questions) as the authoritative
finding for blocking T013.

The tests carry the ``integration`` marker so the issue verification command
``uv run pytest tests/spike/test_codex_thread_read_evidence.py`` collects them
under the repo's default ``-m 'unit or integration'`` filter; module-level
``importorskip`` then skips when the SDK/runtime is absent. Conceptually the
spike is end-to-end (real upstream runtime when present), but ``e2e`` would be
deselected by the default filter and silently fail the verification step — the
``integration`` label keeps the path-based invocation honest while preserving
"skip when tools absent" semantics. With SDK absent (the current repo state),
all three tests collect as one module skip and consume no runtime budget.

Invocation::

    uv run pytest tests/spike/test_codex_thread_read_evidence.py

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
* File edits land via ``EventMsg::PatchApplyEnd`` which IS in
  ``EventPersistenceMode::Limited`` (``rollout/src/policy.rs:99``), so
  ``FileChangeThreadItem.changes`` *should* survive — but the spike still
  asserts this directly because absence of regressions there is a separate
  invariant from the bash-aggregation gap that drives the F3 fallback.
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

The two known-disconfirmed assertions (aggregated_output, cross-resume) are
marked ``@pytest.mark.xfail(strict=True)`` so a developer/CI image with the
SDK + binary + auth running plain ``uv run pytest`` sees them as XFAIL (not
a failure) under current SDK behavior, while an upstream fix flips them to
XPASS — strict mode then fails the suite, alerting us that decision #11
should be revisited and T013's F3 tee fallback may be retire-able. The
latency check is *not* xfail-marked: it should genuinely pass when the
runtime is available; it is the single signal that ``thread/read`` itself
is functionally healthy.
"""

from __future__ import annotations

import importlib
import os
import shutil
import time
import uuid
from typing import Any

import pytest

pytestmark = pytest.mark.integration


_LATENCY_SAMPLE_COUNT = 5
_HELLO_PAYLOAD = "spike-hello-aggregated-output"
_RESUME_PAYLOAD = "spike-second-turn-after-resume"
_FILE_EDIT_PAYLOAD = "spike-file-edit-marker"


def _require_runtime() -> Any:  # noqa: ANN401 — SDK is optional dep; staticly type-erased
    """Per-test SDK + binary probe; skip cleanly if anything required is missing.

    Per-test (rather than module-level ``importorskip``) so the verification
    command ``uv run pytest tests/spike/test_codex_thread_read_evidence.py``
    collects three tests and reports ``3 skipped`` (exit 0) instead of ``no
    tests collected`` (exit 5) when the SDK/runtime is absent. Returns the
    imported ``codex_app_server`` module so callers can read class symbols
    without a top-level import.
    """
    try:
        return importlib.import_module("codex_app_server")
    except ImportError:
        pytest.skip(
            "codex_app_server SDK not installed — recorded spike outcome "
            "stands (see module docstring 'Spike result' section)."
        )


def _resolve_codex_bin() -> str | None:
    """Return an explicit ``codex`` binary path or ``None`` to use SDK default.

    Honours ``CODEX_BINARY`` first (operator override), then ``PATH``. Falling
    back to ``None`` lets ``AppServerConfig`` use its bundled
    ``codex_cli_bin`` resolution — important so the spike still runs in
    environments where Codex is installed only via the SDK extras package.
    """
    explicit = os.environ.get("CODEX_BINARY")
    if explicit:
        return explicit
    path_lookup = shutil.which("codex")
    return path_lookup


def _new_codex(codex_app_server: Any) -> Any:  # noqa: ANN401 — SDK is optional dep; staticly type-erased
    """Construct a synchronous ``Codex`` client; skip on auth-not-ready errors.

    The SDK constructor or ``__enter__`` raises an auth error class when no
    logged-in session exists. We catch broadly because the exact class is part
    of the Phase I auth-detection Open Question — for the spike, any startup
    failure surfaces as "runtime not available, skip and trust the recorded
    finding."

    The SDK does **not** read ``CODEX_BINARY`` or ``PATH``; the default
    resolver returns the bundled ``codex_cli_bin`` runtime package. If we
    found an external binary in :func:`_resolve_codex_bin`, plumb it through
    ``AppServerConfig.codex_bin`` (per
    ``codex_app_server.client.resolve_codex_bin``), otherwise let the SDK use
    its bundled default.
    """
    codex_cls = codex_app_server.Codex
    config_cls = codex_app_server.AppServerConfig
    codex_bin = _resolve_codex_bin()
    config = config_cls(codex_bin=codex_bin) if codex_bin else None
    try:
        client = codex_cls(config) if config is not None else codex_cls()
    except Exception as exc:
        pytest.skip(f"Codex client construction failed (auth/runtime?): {exc!r}")
    return client


def _items_of(thread_read_response: Any) -> list:  # noqa: ANN401 — see _new_codex
    """Flatten ``ThreadReadResponse.thread.turns[*].items`` into a list of *unwrapped* items.

    ``Turn.items`` carries the wrapper :class:`codex_app_server.ThreadItem`
    (a Pydantic ``RootModel``); the concrete variant
    (:class:`CommandExecutionThreadItem`, :class:`FileChangeThreadItem`,
    etc.) lives under ``.root``. Without unwrapping, downstream
    ``getattr(item, "type", ...)`` checks would never match because the
    wrapper does not expose those fields directly.
    """
    items: list = []
    for turn in thread_read_response.thread.turns or []:
        for wrapper in turn.items or []:
            items.append(getattr(wrapper, "root", wrapper))
    return items


def _command_items_with_aggregated_output(items: list) -> list:
    """Items that are ``CommandExecutionThreadItem`` with ``aggregated_output``."""
    return [
        item
        for item in items
        if getattr(item, "type", None) == "commandExecution"
        and getattr(item, "aggregated_output", None)
    ]


def _file_change_items(items: list) -> list:
    """Items that are ``FileChangeThreadItem`` with at least one change recorded."""
    return [
        item
        for item in items
        if getattr(item, "type", None) == "fileChange"
        and getattr(item, "changes", None)
    ]


def test_items_of_unwraps_root_model_wrappers() -> None:
    """Regression: ``_items_of`` must dereference Pydantic ``RootModel.root``.

    ``codex_app_server.ThreadItem`` is a ``RootModel`` whose concrete variant
    lives at ``.root``; without unwrapping, downstream
    ``getattr(item, "type", ...)`` filters silently match nothing and the
    spike would falsely "disconfirm" a working SDK path. This test runs
    without the SDK by mirroring the wrapper shape (a stand-in object whose
    ``.root`` carries the real fields) and asserting ``_items_of`` returns
    the inner object.
    """
    from dataclasses import dataclass

    @dataclass
    class _StubCommand:
        type: str
        aggregated_output: str | None

    @dataclass
    class _StubWrapper:
        root: _StubCommand

    @dataclass
    class _StubTurn:
        items: list

    @dataclass
    class _StubThread:
        turns: list

    @dataclass
    class _StubResponse:
        thread: _StubThread

    inner = _StubCommand(type="commandExecution", aggregated_output="hello\n")
    response = _StubResponse(
        thread=_StubThread(turns=[_StubTurn(items=[_StubWrapper(root=inner)])])
    )

    flattened = _items_of(response)
    assert flattened == [inner]
    assert _command_items_with_aggregated_output(flattened) == [inner]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Decision #11 DISCONFIRMED — see module docstring 'Spike result'. "
        "App-server hardcodes EventPersistenceMode::Limited and ExecCommandEnd "
        "is filtered out, so aggregated_output is always None. Marked "
        "strict=True so an upstream SDK fix flips this to XPASS and alerts us "
        "to revisit T013's F3 tee fallback."
    ),
)
def test_aggregated_output_present_for_bash_command() -> None:
    """Finding (1): ``CommandExecutionThreadItem.aggregated_output`` is populated.

    Expected outcome under current Codex SDK semantics: this assertion FAILS
    because ``persist_extended_history`` is hardcoded-Limited and
    ``ExecCommandEnd`` is filtered out of the rollout (see module docstring
    Spike result). The test is marked ``xfail(strict=True)`` so it does not
    break ``uv run pytest`` for users who happen to have the SDK + binary +
    auth available, while still flipping to a noisy XPASS failure when the
    upstream SDK starts surfacing aggregated output (our cue to revisit
    decision #11 / T013).
    """
    sdk = _require_runtime()
    with _new_codex(sdk) as codex:
        thread = codex.thread_start()
        thread.run(
            input=f"Run this single bash command and nothing else: echo {_HELLO_PAYLOAD}"
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
    sdk = _require_runtime()
    samples_ms: list[float] = []
    with _new_codex(sdk) as codex:
        thread = codex.thread_start()
        thread.run(input=f"echo {_HELLO_PAYLOAD}")
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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Decision #11 DISCONFIRMED — cross-resume bash aggregated_output is "
        "lossy (per ThreadRollbackResponse schema docstring 'same behavior as "
        "thread/resume'). Marked strict=True so an upstream SDK fix flips this "
        "to XPASS and alerts us to revisit T013's F3 tee fallback."
    ),
)
def test_cross_resume_completeness_with_bash_and_file_edit(tmp_path: Any) -> None:  # noqa: ANN401 — pytest fixture stub
    """Finding (3): post-``thread_resume`` ``Thread.read`` includes pre-resume bash + file edit.

    Combines all three issue-mandated coverage points (bash, file edit,
    resume) in one thread so a single ``Thread.read(include_turns=True)``
    response can be inspected for completeness. Under current SDK semantics
    the bash assertion fails (see Finding 1); the file-edit assertion
    *should* pass because ``PatchApplyEnd`` IS in Limited persistence mode —
    if it ever fails, the F3 tee fallback design must also tee patch events,
    not just exec events. xfail(strict=True) at the test level (not per
    assertion) because all three checks are AND-ed: a single missing piece
    keeps the test in XFAIL; only when all three become observable does the
    test XPASS — exactly when we want to be alerted.
    """
    sdk = _require_runtime()
    edit_target = tmp_path / f"spike-edit-{uuid.uuid4().hex}.txt"
    pre_thread_id: str
    with _new_codex(sdk) as codex:
        thread = codex.thread_start(cwd=str(tmp_path))
        thread.run(
            input=(
                f"Run exactly this bash command and nothing else: echo {_HELLO_PAYLOAD}"
            )
        )
        thread.run(
            input=(
                "Create a new file using the apply_patch tool. The file path "
                f"is {edit_target}. The file's contents must be exactly the "
                f"single line: {_FILE_EDIT_PAYLOAD}"
            )
        )
        pre_thread_id = thread.id

    with _new_codex(sdk) as codex:
        resumed = codex.thread_resume(pre_thread_id)
        resumed.run(input=f"Run exactly this bash command: echo {_RESUME_PAYLOAD}")
        response = resumed.read(include_turns=True)

    items = _items_of(response)
    aggregated_blobs = [
        getattr(item, "aggregated_output", None) or "" for item in items
    ]
    pre_bash_present = any(_HELLO_PAYLOAD in blob for blob in aggregated_blobs)
    post_bash_present = any(_RESUME_PAYLOAD in blob for blob in aggregated_blobs)
    file_change_items = _file_change_items(items)
    file_edit_present = any(
        any(_FILE_EDIT_PAYLOAD in str(change) for change in (item.changes or []))
        for item in file_change_items
    )

    assert pre_bash_present and post_bash_present and file_edit_present, (
        "F1 disconfirmed: cross-resume Thread.read missed at least one of "
        f"(pre_bash={pre_bash_present}, post_bash={post_bash_present}, "
        f"file_edit={file_edit_present}). Decision #11 must reverse to F3 tee "
        "fallback so evidence survives thread_resume; if file_edit is the "
        "only failure, F3 must tee patch events too."
    )

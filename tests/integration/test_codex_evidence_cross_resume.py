"""Integration test: Codex evidence stream survives ``thread_resume`` (T013).

The cross-resume invariant from the issue's primary AC #7 is that the
evidence stream must include events from all turns regardless of resume
count. This test wires a real :class:`CodexEvidenceProvider` through a
real :class:`EvidenceCheck` against a tee'd JSONL fixture written in two
phases — the second phase appended *after* the gate captured the
end-of-attempt-1 offset — and verifies the gate still observes the
invocation-1 lint / test / typecheck commands.

Why this lives in ``tests/integration/`` rather than alongside the
provider unit tests: it composes :class:`CodexEvidenceProvider` (infra)
with :class:`EvidenceCheck` (domain) and a real :class:`ValidationSpec`,
exercising the contract boundary between the two layers. The unit test
file ``test_codex_evidence_provider.py`` covers ``iter_thread_evidence``
in isolation; this test is the integration-level proof that the same
behavior actually surfaces validation evidence to the gate consumer.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from src.domain.evidence_check import EvidenceCheck
from src.domain.validation.spec import (
    CommandKind,
    ValidationCommand,
    ValidationScope,
    ValidationSpec,
)
from src.domain.validation_wrapper import build_canonical_wrapper
from src.infra.clients.codex_evidence_provider import CodexEvidenceProvider

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from src.core.protocols.infra import CommandResultProtocol, CommandRunnerPort


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers — Codex notification-shaped tee'd JSONL
# ---------------------------------------------------------------------------


def _command_completed(
    item_id: str,
    command: str,
    aggregated_output: str = "",
    status: str = "completed",
    *,
    exit_code: int | None = 0,
) -> dict[str, object]:
    item: dict[str, object] = {
        "type": "commandExecution",
        "id": item_id,
        "command": command,
        "aggregated_output": aggregated_output,
        "status": status,
    }
    if exit_code is not None:
        item["exit_code"] = exit_code
    return {
        "method": "item/completed",
        "payload": {
            "item": item,
            "thread_id": "thr_resume_test",
            "turn_id": "turn_x",
        },
    }


def _write_jsonl(path: Path, events: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event))
            fh.write("\n")


def _append_jsonl(path: Path, events: Iterable[dict[str, object]]) -> None:
    with path.open("a", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event))
            fh.write("\n")


def _validation_command_completed(
    item_id: str,
    command: ValidationCommand,
    *,
    exit_code: int = 0,
    issue_id: str = "codex-resume",
) -> dict[str, object]:
    validation_log_dir = Path("/tmp/mala-validation-logs")
    wrapper = build_canonical_wrapper(
        command,
        issue_id=issue_id,
        validation_log_dir=validation_log_dir,
    )
    log_path = validation_log_dir / f"{issue_id}.{command.name}.log"
    return _command_completed(
        item_id,
        wrapper,
        f"MALA_EVIDENCE name={command.name} exit={exit_code} log={log_path}",
        exit_code=exit_code,
    )


class _NoopRunner:
    """Minimal :class:`CommandRunnerPort` stand-in.

    :class:`EvidenceCheck` only invokes the runner for ``git`` queries
    that this test does not exercise (no ``parse_validation_evidence_with_spec``
    code path runs git); a stand-in keeps the integration scope to the
    evidence-provider boundary the test actually targets.

    Signature matches :class:`src.core.protocols.infra.CommandRunnerPort`
    so the structural protocol check passes; any actual call is a test
    bug and raises.
    """

    def run(
        self,
        cmd: list[str] | str,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        use_process_group: bool | None = None,
        shell: bool = False,
        cwd: Path | None = None,
    ) -> CommandResultProtocol:
        del cmd, env, timeout, use_process_group, shell, cwd
        msg = "_NoopRunner.run was not expected to be invoked by this test"
        raise AssertionError(msg)


def _spec_with_lint_test_typecheck() -> ValidationSpec:
    """Spec with detection patterns for ``pytest`` / ``ruff`` / ``ty``."""
    return ValidationSpec(
        commands=[
            ValidationCommand(
                name="pytest",
                command="uv run pytest -q",
                kind=CommandKind.TEST,
                detection_pattern=re.compile(r"\bpytest\b"),
            ),
            ValidationCommand(
                name="ruff-check",
                command="uvx ruff check .",
                kind=CommandKind.LINT,
                detection_pattern=re.compile(r"\bruff check\b"),
            ),
            ValidationCommand(
                name="ty",
                command="uvx ty check",
                kind=CommandKind.TYPECHECK,
                detection_pattern=re.compile(r"\bty check\b"),
            ),
        ],
        scope=ValidationScope.PER_SESSION,
    )


def _commands_by_name(spec: ValidationSpec) -> dict[str, ValidationCommand]:
    return {command.name: command for command in spec.commands}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_evidence_check_sees_invocation_1_evidence_after_resume(
    tmp_path: Path,
) -> None:
    """AC #7 cross-resume invariant on the real Codex evidence path.

    Setup:

      1. Invocation 1 writes ``pytest`` + ``ruff check`` + ``ty check``
         Bash items, each followed by a successful tool result.
      2. The gate captures the end-of-invocation-1 byte offset (parity
         with how ``RetryState.log_offset`` is advanced in the runner
         after a passing turn).
      3. Invocation 2 (post-``thread_resume``) appends a single new
         ``git commit`` item.
      4. :meth:`EvidenceCheck.parse_validation_evidence_with_spec` is
         invoked with the captured invocation-1-end offset.

    The gate must still observe the lint / test / typecheck evidence
    from invocation 1 — :class:`CodexEvidenceProvider.iter_thread_evidence`
    re-reads from byte 0 specifically so this remains true. If the
    contract regressed (offset honored on the cross-resume path), the
    spec'd commands would all read as "absent" and the gate would loop
    on a passing build.
    """
    sessions_dir = tmp_path / "codex-sessions"
    log_path = sessions_dir / "thr_resume_test.jsonl"
    spec = _spec_with_lint_test_typecheck()
    commands = _commands_by_name(spec)

    _write_jsonl(
        log_path,
        [
            _validation_command_completed("inv1-test", commands["pytest"]),
            _validation_command_completed("inv1-lint", commands["ruff-check"]),
            _validation_command_completed("inv1-tc", commands["ty"]),
        ],
    )
    end_of_invocation_1 = log_path.stat().st_size

    _append_jsonl(
        log_path,
        [
            _command_completed("inv2-commit", "git commit -m 'wip'", "committed"),
        ],
    )

    provider = CodexEvidenceProvider(sessions_dir=sessions_dir)
    check = EvidenceCheck(
        repo_path=tmp_path,
        evidence_provider=provider,
        command_runner=cast("CommandRunnerPort", _NoopRunner()),
    )

    evidence = check.parse_validation_evidence_with_spec(
        log_path,
        spec,
        offset=end_of_invocation_1,
    )

    assert evidence.commands["pytest"].seen is True
    assert evidence.commands["pytest"].kind == CommandKind.TEST
    assert evidence.commands["ruff-check"].seen is True
    assert evidence.commands["ruff-check"].kind == CommandKind.LINT
    assert evidence.commands["ty"].seen is True
    assert evidence.commands["ty"].kind == CommandKind.TYPECHECK
    # Status from invocation 1 is "completed" → no failed commands.
    assert all(c.status != "failed" for c in evidence.commands.values()), (
        evidence.commands
    )


def test_iter_thread_evidence_yields_full_history_across_three_invocations(
    tmp_path: Path,
) -> None:
    """Three-invocation chain exercises the append-only invariant.

    Codex's resume model lets a thread be resumed any number of times;
    every resume re-tees notifications into the same per-thread JSONL.
    The provider must surface evidence from all three invocations
    (invocation-1 pytest, invocation-2 ruff, invocation-3 ty) when
    queried with any offset — including an offset past EOF of the first
    two writes — because the gate's
    :meth:`EvidenceCheck.parse_validation_evidence_with_spec` always
    queries this method for cross-attempt validation evidence.
    """
    sessions_dir = tmp_path / "codex-sessions"
    log_path = sessions_dir / "thr_three_resumes.jsonl"

    _write_jsonl(
        log_path,
        [_command_completed("i1-test", "uv run pytest", "ok")],
    )
    after_inv_1 = log_path.stat().st_size

    _append_jsonl(
        log_path,
        [_command_completed("i2-lint", "uvx ruff check .", "ok")],
    )
    after_inv_2 = log_path.stat().st_size

    _append_jsonl(
        log_path,
        [_command_completed("i3-tc", "uvx ty check", "ok")],
    )

    provider = CodexEvidenceProvider(sessions_dir=sessions_dir)

    seen_via_thread_evidence: list[str] = []
    for entry in provider.iter_thread_evidence(log_path, offset=after_inv_2):
        for _id, command in provider.extract_bash_commands(entry):
            seen_via_thread_evidence.append(command)

    # All three are visible despite the offset advancing past invocations 1 and 2.
    assert seen_via_thread_evidence == [
        "uv run pytest",
        "uvx ruff check .",
        "uvx ty check",
    ]

    # Sanity: the per-attempt iter_session_events DOES honor the offset
    # so resolution-marker scoping (a separate gate concern) is unaffected.
    seen_via_session: list[str] = []
    for entry in provider.iter_session_events(log_path, offset=after_inv_1):
        for _id, command in provider.extract_bash_commands(entry):
            seen_via_session.append(command)
    assert seen_via_session == ["uvx ruff check .", "uvx ty check"]


def test_failed_pytest_is_routed_to_failed_commands(tmp_path: Path) -> None:
    """Regression: a non-zero exit on a "completed" pytest must fail the gate.

    Review-2 P1: Codex marks a ``commandExecution`` item as
    ``status="completed"`` whenever the shell finished, regardless of
    return code. If :class:`CodexEvidenceProvider.extract_tool_results`
    only checked ``status``, a failed ``uv run pytest -q`` (status=
    completed, exit_code=1) would propagate through
    :meth:`EvidenceCheck.parse_validation_evidence_with_spec` as
    ``is_error=False``, the spec'd TEST kind would never land in
    ``failed_commands``, and the gate would silently pass on a failed
    test run.

    This integration test wires the real :class:`CodexEvidenceProvider`
    into a real :class:`EvidenceCheck` and asserts that the failing
    pytest IS reported in ``failed_commands`` — i.e., the gate does
    NOT silently pass.
    """
    sessions_dir = tmp_path / "codex-sessions"
    log_path = sessions_dir / "thr_resume_test.jsonl"
    spec = _spec_with_lint_test_typecheck()
    commands = _commands_by_name(spec)

    _write_jsonl(
        log_path,
        [
            _validation_command_completed(
                "item-pytest-fail",
                commands["pytest"],
                exit_code=1,
            ),
        ],
    )

    provider = CodexEvidenceProvider(sessions_dir=sessions_dir)
    check = EvidenceCheck(
        repo_path=tmp_path,
        evidence_provider=provider,
        command_runner=cast("CommandRunnerPort", _NoopRunner()),
    )

    evidence = check.parse_validation_evidence_with_spec(
        log_path,
        spec,
    )

    pytest_evidence = evidence.commands["pytest"]
    assert pytest_evidence.seen is True
    assert pytest_evidence.kind == CommandKind.TEST
    assert pytest_evidence.status == "failed", (
        "Failed pytest with status=completed + exit_code=1 must be "
        "tracked as a failed command; otherwise the gate silently passes "
        "on a real test failure."
    )
    assert pytest_evidence.observed_command == "uv run pytest -q"


def test_codex_provider_evidence_surface_reads_codex_sessions_dir() -> None:
    """End-to-end provider wiring sanity check.

    :class:`CodexAgentProvider.evidence_provider` returns a
    :class:`CodexEvidenceProvider` whose ``get_log_path`` resolves to
    ``~/.config/mala/codex-sessions/{thread_id}.jsonl``. This is the
    AC #18 protocol-conformance bullet: the unified evidence surface
    must not silently route Codex callers to the Amp tee location
    (``amp-sessions/``) where ``thr_*`` ids would collide with Amp
    ``T-*`` ids in unpredictable ways.

    Runs in a subprocess with no environment overrides so we exercise
    the production default path; the in-process tests use
    ``sessions_dir=tmp_path`` which would mask a regression where the
    default constant pointed at the wrong directory.
    """
    repo_root = Path(__file__).resolve().parents[2]
    code = """
from pathlib import Path
from src.infra.clients.codex_evidence_provider import (
    CODEX_SESSIONS_DIR,
    CodexEvidenceProvider,
)
from src.infra.clients.codex_provider import CodexAgentProvider

provider = CodexAgentProvider()
evidence = provider.evidence_provider
assert isinstance(evidence, CodexEvidenceProvider), type(evidence).__name__
log_path = evidence.get_log_path(Path('/tmp/repo'), 'thr_test')
assert log_path == CODEX_SESSIONS_DIR / 'thr_test.jsonl', log_path
assert 'codex-sessions' in str(log_path), log_path
assert 'amp-sessions' not in str(log_path), log_path
print('PASS')
"""
    import sys

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(repo_root),
    )
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert "PASS" in result.stdout

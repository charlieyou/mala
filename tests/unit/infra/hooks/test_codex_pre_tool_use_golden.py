"""Golden-corpus harness for ``src.infra.hooks.codex_pre_tool_use``.

Loads each case file under ``golden_corpus/`` and asserts the hook's JSON
output is byte-identical to the case's ``expected`` field. This is the
safety net for the B.3 codex_pre_tool_use split (plans/2026-05-09-
architecture-fixes-plan.md "Golden Tests"): the corpus captures the
*current unmodified* hook's exact output for every policy branch so the
extraction commits that follow can be verified against pinned byte
output.

The harness drives the live CLI entry point :func:`main` over stdin /
stdout — not :func:`decide` directly — so the corpus covers ``main``'s
event routing (selftest-target dispatch, SessionStart vs PreToolUse) and
its stdout JSON serialization, not just the pure policy function.

Case file format (JSON):

    {
      "description": "human-readable summary",
      "tool_name": "bash",
      "tool_input": {"command": "..."},
      "cwd": "{repo}",                          // optional, defaults to {repo}
      "hook_event_name": "PreToolUse",          // optional, defaults to PreToolUse
      "env": {"MALA_DISALLOWED_TOOLS": "..."},  // optional overrides
      "env_unset": ["MALA_AGENT_ID"],           // optional vars to drop
      "locks": [                                // optional; held before main()
        {"path": "{repo}/file.py", "agent_id": "agent-me"}
      ],
      "expected": {"hookSpecificOutput": {...}}
    }

String values in ``tool_input``, ``cwd``, ``locks[].path``, and
``expected.hookSpecificOutput.permissionDecisionReason`` go through a
``str.format()`` substitution with ``{repo}`` → the per-test repo dir
and ``{lock_dir}`` → the per-test lock dir, so the corpus stays
host-independent.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from src.infra.hooks.codex_pre_tool_use import main
from src.infra.tools.locking import try_lock

if TYPE_CHECKING:
    from collections.abc import Iterator


CORPUS_DIR = Path(__file__).parent / "golden_corpus"


def _discover_cases() -> list[Path]:
    """Return every ``*.json`` case file in the corpus, sorted by name."""
    if not CORPUS_DIR.is_dir():
        return []
    return sorted(CORPUS_DIR.glob("*.json"))


def _substitute(value: object, *, repo: str, lock_dir: str) -> object:
    """Recursively format-substitute ``{repo}``/``{lock_dir}`` in ``value``."""
    if isinstance(value, str):
        return value.replace("{repo}", repo).replace("{lock_dir}", lock_dir)
    if isinstance(value, dict):
        return {
            k: _substitute(v, repo=repo, lock_dir=lock_dir) for k, v in value.items()
        }
    if isinstance(value, list):
        return [_substitute(v, repo=repo, lock_dir=lock_dir) for v in value]
    return value


@pytest.fixture
def golden_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[str, str]]:
    """Per-test ``repo``/``lock_dir`` plus baseline MALA_* env."""
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("MALA_AGENT_ID", "agent-me")
    monkeypatch.setenv("MALA_LOCK_DIR", str(lock_dir))
    monkeypatch.setenv("MALA_REPO_NAMESPACE", str(repo))
    monkeypatch.delenv("MALA_DISALLOWED_TOOLS", raising=False)
    yield str(repo), str(lock_dir)


def _format_case_id(path: Path) -> str:
    return path.stem


CASES = _discover_cases()


@pytest.mark.parametrize(
    "case_path",
    CASES,
    ids=[_format_case_id(p) for p in CASES],
)
def test_golden_case(
    case_path: Path,
    golden_env: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run one corpus case end-to-end and assert byte-identical JSON output.

    The harness:

    1. Substitutes ``{repo}`` / ``{lock_dir}`` placeholders.
    2. Applies env overrides / unsets after the baseline fixture.
    3. Acquires the case's locks (each ``try_lock`` must succeed).
    4. Pipes the payload JSON to ``sys.stdin`` and captures ``sys.stdout``
       while calling :func:`main`, then compares the captured stdout
       string directly to ``json.dumps(expected) + "\n"`` — the
       byte-identical wire contract the B.3 extraction commits must
       preserve. Comparing raw stdout (not a reparsed/redumped value)
       means a change to ``main``'s serialization (pretty-printing,
       extra whitespace, missing trailing newline) is caught here.
       Going through ``main`` (not ``decide`` directly) also exercises
       the event-routing branches (e.g. the selftest-target PreToolUse
       dispatch) so future changes to the CLI wrapper cannot silently
       alter the hook's wire output.
    """
    repo, lock_dir = golden_env
    raw = json.loads(case_path.read_text(encoding="utf-8"))
    case = cast("dict[str, object]", _substitute(raw, repo=repo, lock_dir=lock_dir))

    env_overrides = cast("dict[str, object]", case.get("env") or {})
    for key, value in env_overrides.items():
        monkeypatch.setenv(key, str(value))
    env_unset = cast("list[str]", case.get("env_unset") or [])
    for key in env_unset:
        monkeypatch.delenv(key, raising=False)

    locks = cast("list[dict[str, str]]", case.get("locks") or [])
    for lock in locks:
        acquired = try_lock(lock["path"], lock["agent_id"], repo_namespace=repo)
        assert acquired, f"lock setup failed for {lock}"

    payload = {
        "tool_name": case.get("tool_name", ""),
        "tool_input": case.get("tool_input") or {},
        "cwd": case.get("cwd", repo),
        "session_id": "thr_golden",
        "turn_id": "trn_golden",
        "hook_event_name": case.get("hook_event_name", "PreToolUse"),
    }

    fake_stdin = io.StringIO(json.dumps(payload))
    fake_stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdin", fake_stdin)
    monkeypatch.setattr(sys, "stdout", fake_stdout)
    exit_code = main()
    output = fake_stdout.getvalue()
    assert exit_code == 0, f"main() exited {exit_code} (output={output!r})"

    expected = case["expected"]
    expected_output = json.dumps(expected) + "\n"
    assert output == expected_output, (
        f"\ncase: {case_path.name}\n"
        f"description: {case.get('description', '')}\n"
        f"expected: {expected_output!r}\n"
        f"actual:   {output!r}\n"
    )


def test_corpus_is_non_empty() -> None:
    """Sanity guard so an accidentally-empty corpus dir doesn't silently pass."""
    assert CASES, (
        f"golden corpus directory {CORPUS_DIR} is empty or missing; "
        "the harness needs at least one case file to provide a safety net."
    )

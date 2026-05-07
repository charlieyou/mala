"""Cross-provider conformance test for :class:`EvidenceProvider`.

Phase A5 introduces :class:`src.core.protocols.evidence.EvidenceProvider` as
the unified evidence surface and deletes ``src.core.protocols.log``. This
test pins the protocol's behavior contract: both shipped implementations
(``FileSystemLogProvider`` for Claude, ``AmpLogProvider`` for Amp) must
satisfy ``isinstance(p, EvidenceProvider)`` and emit the same Bash/tool
shape over a fixture log so the gate consumes them without
provider-specific branches (issue ``mala-b18dd.5.1`` AC#10, AC#18).

The cross-resume invariant for :meth:`iter_thread_evidence` is exercised
elsewhere — the per-provider tests in
``tests/unit/infra/clients/test_amp_log_provider.py`` and
``tests/unit/infra/test_session_log_parser.py`` cover offset honoring on
Claude and offset-ignoring on Amp; this module only verifies the cross-
provider conformance.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from src.core.protocols.evidence import EvidenceProvider
from src.infra.clients.amp_log_provider import AmpLogProvider
from src.infra.io.session_log_parser import FileSystemLogProvider

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _write_fixture(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "type": "assistant",
            "message": {
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "Bash",
                        "input": {"command": "uv run pytest -q"},
                    }
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": "1 passed",
                        "is_error": False,
                    }
                ]
            },
        },
        {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": "ISSUE_NO_CHANGE: nothing to do"}]
            },
        },
    ]
    with log_path.open("w", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event))
            fh.write("\n")


@pytest.mark.unit
@pytest.mark.parametrize(
    "provider_factory",
    [
        pytest.param(lambda tmp_path: FileSystemLogProvider(), id="claude"),
        pytest.param(lambda tmp_path: AmpLogProvider(native_dir=tmp_path), id="amp"),
    ],
)
def test_provider_conforms_to_evidence_protocol(
    tmp_path: Path,
    provider_factory: Callable[[Path], EvidenceProvider],
) -> None:
    """Both shipped providers structurally satisfy :class:`EvidenceProvider`."""
    provider = provider_factory(tmp_path)

    assert isinstance(provider, EvidenceProvider)


@pytest.mark.unit
def test_filesystem_and_amp_extractors_emit_same_bash_shape(tmp_path: Path) -> None:
    """AC#10 regression: both providers extract the same Bash + result shape.

    Validation gates depend on identical extractor output regardless of
    coder. This test runs each provider over the same fixture and asserts
    the typed extractors agree on what was observed.
    """
    log_path = tmp_path / "T-fixture.jsonl"
    _write_fixture(log_path)

    fs = FileSystemLogProvider()
    amp = AmpLogProvider(native_dir=tmp_path)

    fs_bash: list[tuple[str, str]] = []
    fs_results: list[tuple[str, bool]] = []
    fs_texts: list[str] = []
    for entry in fs.iter_session_events(log_path):
        fs_bash.extend(fs.extract_bash_commands(entry))
        fs_results.extend(fs.extract_tool_results(entry))
        fs_texts.extend(fs.extract_assistant_text_blocks(entry))

    amp_bash: list[tuple[str, str]] = []
    amp_results: list[tuple[str, bool]] = []
    amp_texts: list[str] = []
    for entry in amp.iter_session_events(log_path):
        amp_bash.extend(amp.extract_bash_commands(entry))
        amp_results.extend(amp.extract_tool_results(entry))
        amp_texts.extend(amp.extract_assistant_text_blocks(entry))

    assert fs_bash == [("tool-1", "uv run pytest -q")]
    assert fs_bash == amp_bash
    assert fs_results == [("tool-1", False)]
    assert fs_results == amp_results
    assert fs_texts == ["ISSUE_NO_CHANGE: nothing to do"]
    assert fs_texts == amp_texts


@pytest.mark.unit
def test_iter_thread_evidence_emits_same_entries_as_iter_session_events_on_claude(
    tmp_path: Path,
) -> None:
    """Claude has per-session log files, so cross-attempt = per-attempt.

    The protocol allows Amp to ignore offset on
    :meth:`iter_thread_evidence` (its tee log spans invocations); on
    Claude both methods must yield identical entries when called from
    byte 0.
    """
    log_path = tmp_path / "session.jsonl"
    _write_fixture(log_path)

    fs = FileSystemLogProvider()

    session_entries = list(fs.iter_session_events(log_path))
    thread_entries = list(fs.iter_thread_evidence(log_path))

    assert [e.data for e in session_entries] == [e.data for e in thread_entries]

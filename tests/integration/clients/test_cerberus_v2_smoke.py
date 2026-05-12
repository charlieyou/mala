from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest

from src.infra.io.config import MalaConfig
from src.orchestration.config_resolution import _ReviewerConfig

if TYPE_CHECKING:
    from src.core.protocols.events import MalaEventSink
    from src.infra.clients.cerberus_output_parser import ReviewResult

pytestmark = pytest.mark.integration

FAKE_CERBERUS = Path(__file__).parent / "fixtures" / "fake_cerberus.sh"
PROJECT_KEY = "mala-v2-smoke"
SESSION_ID = "session-v2-smoke"
RUN_KEY = f"mala-{SESSION_ID}"
NO_CONSENSUS_TITLE = "Cerberus reviewers reached no consensus"


@pytest.fixture
def fake_cerberus_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (tmp_path / "prompts" / "reviewers").mkdir(parents=True)

    binary_path = bin_dir / "cerberus"
    shutil.copy(FAKE_CERBERUS, binary_path)
    binary_path.chmod(0o755)

    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("CERBERUS_STATE_ROOT", str(tmp_path / "cerberus-state"))
    monkeypatch.setenv("CERBERUS_PROJECT_KEY", PROJECT_KEY)
    monkeypatch.setenv("CERBERUS_FAKE_LOG", str(tmp_path / "fake-cerberus.log"))
    return bin_dir / "cerberus"


async def _run_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_name: str,
) -> tuple[ReviewResult, list[str]]:
    from src.orchestration.factory import _create_code_reviewer

    monkeypatch.setenv("CERBERUS_FAKE_CASE", case_name)
    log_path = Path(os.environ["CERBERUS_FAKE_LOG"])

    reviewer = _create_code_reviewer(
        repo_path=tmp_path,
        mala_config=MalaConfig(),
        event_sink=cast("MalaEventSink", None),
        reviewer_config=_ReviewerConfig(reviewer_type="cerberus"),
    )

    result = await reviewer(
        claude_session_id=SESSION_ID,
        commit_shas=["abc123"],
    )
    return cast("ReviewResult", result), log_path.read_text().splitlines()


def _assert_v2_cli_contract(invocations: list[str]) -> None:
    assert any(
        line.startswith("cerberus|spawn-code-review|") for line in invocations
    ), "mala must invoke the v2 `cerberus spawn-code-review` binary"
    assert any(
        "|wait|" in line and f"--session-key {RUN_KEY}" in line for line in invocations
    ), "mala must wait by v2 --session-key using CERBERUS_RUN_KEY"


async def test_v2_pass_single_round_zero_findings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_cerberus_path: Path,
) -> None:
    assert fake_cerberus_path.exists()

    result, invocations = await _run_review(tmp_path, monkeypatch, "pass_empty")

    _assert_v2_cli_contract(invocations)
    assert result.passed is True
    assert result.issues == []
    assert result.parse_error is None
    assert result.fatal_error is False


async def test_v2_fail_formats_two_reviewer_attributions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_cerberus_path: Path,
) -> None:
    from src.infra.clients.cerberus_review import format_review_issues

    assert fake_cerberus_path.exists()

    result, invocations = await _run_review(tmp_path, monkeypatch, "fail_two_reviewers")
    formatted = format_review_issues(result.issues, base_path=tmp_path)

    _assert_v2_cli_contract(invocations)
    assert result.passed is False
    assert result.parse_error is None
    assert "[claude#1] Fix null handling" in formatted
    assert "[gemini#1] Preserve reviewer attribution" in formatted


async def test_v2_normal_review_never_invokes_resolve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_cerberus_path: Path,
) -> None:
    assert fake_cerberus_path.exists()

    result, invocations = await _run_review(tmp_path, monkeypatch, "pass_no_resolve")

    _assert_v2_cli_contract(invocations)
    assert result.parse_error is None
    assert not any("|resolve|" in line for line in invocations), (
        "v2 normal pass/fail reviews must not invoke stale-gate resolve"
    )


async def test_v2_pass_with_empty_reviewers_dir_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_cerberus_path: Path,
) -> None:
    assert fake_cerberus_path.exists()

    result, invocations = await _run_review(
        tmp_path, monkeypatch, "pass_empty_reviewers"
    )

    _assert_v2_cli_contract(invocations)
    assert result.passed is False
    assert result.fatal_error is False
    assert result.parse_error is not None
    assert "reviewers" in result.parse_error
    assert result.issues == []


async def test_v2_debate_surfaces_only_final_round_findings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_cerberus_path: Path,
) -> None:
    assert fake_cerberus_path.exists()

    result, invocations = await _run_review(tmp_path, monkeypatch, "debate_round_two")
    titles = [issue.title for issue in result.issues]

    _assert_v2_cli_contract(invocations)
    assert result.passed is False
    assert result.parse_error is None
    assert titles == ["Round two final finding"]


async def test_v2_requires_decision_returns_synthetic_issue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_cerberus_path: Path,
) -> None:
    assert fake_cerberus_path.exists()

    result, invocations = await _run_review(tmp_path, monkeypatch, "requires_decision")

    _assert_v2_cli_contract(invocations)
    assert result.passed is False
    assert result.fatal_error is False
    assert result.parse_error is None
    assert len(result.issues) == 1
    assert result.issues[0].title == NO_CONSENSUS_TITLE
    assert result.issues[0].priority == 1

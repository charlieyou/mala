"""Integration test for baseline timestamp persistence across resume.

The baseline_timestamp is persisted and reused for gate checking, but
commit filtering does not use baseline_timestamp (always passes None).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from src.infra.clients.review_output_parser import ReviewResult
from src.infra.io.log_output.run_metadata import parse_timestamp
from src.infra.tools.env import encode_repo_path, get_claude_log_path
from tests.fakes.gate_checker import FakeGateChecker
from tests.fakes.issue_provider import FakeIssue, FakeIssueProvider
from tests.fakes.sdk_client import FakeSDKClientFactory

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from src.orchestration.orchestrator import MalaOrchestrator


def _read_latest_run_metadata(runs_dir: Path, repo_path: Path) -> dict:
    repo_runs_dir = runs_dir / encode_repo_path(repo_path)
    files = list(repo_runs_dir.glob("*.json"))
    assert files, "expected at least one run metadata file"
    best: tuple[float, str, Path] | None = None
    for path in files:
        data = json.loads(path.read_text())
        started_at = data.get("started_at") or ""
        run_id = data.get("run_id") or ""
        timestamp = parse_timestamp(started_at)
        key = (timestamp, run_id, path)
        if best is None or key > best:
            best = key
    assert best is not None
    return json.loads(best[2].read_text())


@pytest.mark.asyncio
async def test_resume_reuses_persisted_baseline_timestamp(
    tmp_path: Path,
    make_orchestrator: Callable[..., MalaOrchestrator],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    issue_id = "issue-123"
    runs_dir = tmp_path / "runs"
    claude_dir = tmp_path / "claude"
    monkeypatch.setenv("MALA_RUNS_DIR", str(runs_dir))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_dir))

    session_id_run1 = "session-1"
    session_id_run2 = "session-2"

    log_path_1 = get_claude_log_path(tmp_path, session_id_run1)
    log_path_1.parent.mkdir(parents=True, exist_ok=True)
    log_path_1.write_text('{"type": "result"}\n')

    log_path_2 = get_claude_log_path(tmp_path, session_id_run2)
    log_path_2.parent.mkdir(parents=True, exist_ok=True)
    log_path_2.write_text('{"type": "result"}\n')

    gate_checker_run1 = FakeGateChecker()
    gate_checker_run1.gate_result.commit_hash = None  # Skip review on run 1
    sdk_factory_run1 = FakeSDKClientFactory()
    sdk_factory_run1.configure_next_client(
        result_message=_make_result_message(session_id_run1, result="Done")
    )

    orchestrator_run1 = make_orchestrator(
        repo_path=tmp_path,
        max_agents=1,
        issue_provider=FakeIssueProvider(
            {issue_id: FakeIssue(id=issue_id, description="Run 1")}
        ),
        gate_checker=gate_checker_run1,
        runs_dir=runs_dir,
        disable_validations={"global-validate"},
    )
    orchestrator_run1._sdk_client_factory = sdk_factory_run1  # test-only override

    await _run_with_fake_git(orchestrator_run1)

    run1_metadata = _read_latest_run_metadata(runs_dir, tmp_path)
    baseline_timestamp = run1_metadata["issues"][issue_id]["baseline_timestamp"]
    assert isinstance(baseline_timestamp, int)
    assert baseline_timestamp > 0

    gate_checker_run2 = FakeGateChecker()
    gate_checker_run2.gate_result.commit_hash = "abc123"  # Force review path
    sdk_factory_run2 = FakeSDKClientFactory()
    sdk_factory_run2.configure_next_client(
        result_message=_make_result_message(session_id_run2, result="Done")
    )

    commit_since: list[int | None] = []
    commit_shas_seen: list[Sequence[str]] = []

    async def fake_get_issue_commits_async(
        repo_path: Path,
        issue_id: str,
        *,
        since_timestamp: int | None = None,
        timeout: float = 5.0,
    ) -> list[str]:
        commit_since.append(since_timestamp)
        return ["c1", "c2"]

    class FakeReviewer:
        """Fake reviewer that implements CodeReviewer protocol."""

        def overrides_disabled_setting(self) -> bool:
            return True

        async def __call__(
            self,
            *,
            context_file: Path | None = None,
            timeout: int = 300,
            claude_session_id: str | None = None,
            author_context: str | None = None,
            commit_shas: Sequence[str],
            interrupt_event: object | None = None,
        ) -> ReviewResult:
            commit_shas_seen.append(commit_shas)
            return ReviewResult(passed=True, issues=[])

    fake_reviewer = FakeReviewer()

    # Create mala.yaml with per_issue_review enabled so the review path is triggered
    mala_yaml = tmp_path / "mala.yaml"
    mala_yaml.write_text("preset: python-uv\nper_issue_review:\n  enabled: true\n")

    orchestrator_run2 = make_orchestrator(
        repo_path=tmp_path,
        max_agents=1,
        issue_provider=FakeIssueProvider(
            {
                issue_id: FakeIssue(
                    id=issue_id, status="in_progress", description="Run 2"
                )
            }
        ),
        gate_checker=gate_checker_run2,
        code_reviewer=fake_reviewer,
        runs_dir=runs_dir,
        include_wip=True,
        disable_validations={"global-validate"},
    )
    orchestrator_run2._sdk_client_factory = sdk_factory_run2  # test-only override

    await _run_with_fake_git(
        orchestrator_run2, get_issue_commits=fake_get_issue_commits_async
    )

    assert any(
        call["baseline_timestamp"] == baseline_timestamp
        for call in gate_checker_run2.check_with_resolution_calls
    )
    assert commit_since == [None]  # commit filtering no longer uses baseline_timestamp
    assert commit_shas_seen == [["c1", "c2"]]
    assert any(
        resume == session_id_run1
        for _options, resume in sdk_factory_run2.with_resume_calls
    )

    run2_metadata = _read_latest_run_metadata(runs_dir, tmp_path)
    assert run2_metadata["issues"][issue_id]["baseline_timestamp"] == baseline_timestamp


async def _run_with_fake_git(
    orchestrator: MalaOrchestrator,
    *,
    get_issue_commits: Callable[..., object] | None = None,
) -> None:
    import contextlib
    from unittest.mock import patch

    patches = [
        patch(
            "src.orchestration.orchestrator.get_git_branch_async",
            return_value="main",
        ),
        patch(
            "src.orchestration.orchestrator.get_git_commit_async",
            return_value="abc123",
        ),
        patch(
            "src.infra.git_utils.get_git_commit_async",
            return_value="abc123",
        ),
    ]
    if get_issue_commits is not None:
        patches.append(
            patch(
                "src.infra.git_utils.get_issue_commits_async",
                side_effect=get_issue_commits,
            )
        )

    with contextlib.ExitStack() as stack:
        for patcher in patches:
            stack.enter_context(patcher)
        await orchestrator.run()


def _make_result_message(session_id: str, *, result: str) -> object:
    from claude_agent_sdk import ResultMessage

    return ResultMessage(
        subtype="result",
        session_id=session_id,
        result=result,
        duration_ms=1000,
        duration_api_ms=800,
        is_error=False,
        num_turns=1,
        total_cost_usd=0.01,
        usage=None,
    )

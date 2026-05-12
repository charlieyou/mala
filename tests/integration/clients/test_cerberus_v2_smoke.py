from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest

from src.infra.io.config import MalaConfig
from src.orchestration.config_resolution import _ReviewerConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.core.protocols.events import MalaEventSink
    from src.core.protocols.infra import CommandRunnerPort
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

    for binary_name in ("cerberus", "review-gate"):
        binary_path = bin_dir / binary_name
        shutil.copy(FAKE_CERBERUS, binary_path)
        binary_path.chmod(0o755)

    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("CERBERUS_STATE_ROOT", str(tmp_path / "cerberus-state"))
    monkeypatch.setenv("CERBERUS_PROJECT_KEY", PROJECT_KEY)
    monkeypatch.setenv("CERBERUS_FAKE_LOG", str(tmp_path / "fake-cerberus.log"))
    return bin_dir / "cerberus"


def _install_partial_v2_import_compat(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let this smoke test collect while adjacent v2 commits are half-applied."""
    from src.infra.clients import cerberus_cli
    from src.infra.clients import cerberus_review

    class CerberusGateCLI:
        def __init__(
            self,
            repo_path: Path,
            bin_path: Path | None = None,
            env: dict[str, str] | None = None,
        ) -> None:
            _ = bin_path
            self.repo_path = repo_path
            self.env = env or {}

        extract_wait_timeout = staticmethod(
            cerberus_cli.CerberusCLI.extract_wait_timeout
        )

        def validate_binary(self) -> str | None:
            if shutil.which("cerberus", path=os.environ.get("PATH", "")) is None:
                return "cerberus binary not found in PATH"
            return None

        def build_env(self, claude_session_id: str | None) -> dict[str, str]:
            merged = dict(self.env)
            if claude_session_id:
                merged["CLAUDE_SESSION_ID"] = claude_session_id
                merged["CERBERUS_RUN_KEY"] = f"mala-{claude_session_id}"
            merged["CERBERUS_HOST"] = "generic"
            merged["CERBERUS_STATE_ROOT"] = os.environ["CERBERUS_STATE_ROOT"]
            merged["CERBERUS_PROJECT_KEY"] = os.environ.get(
                "CERBERUS_PROJECT_KEY", PROJECT_KEY
            )
            return merged

        async def spawn_code_review(
            self,
            runner: CommandRunnerPort,
            env: dict[str, str],
            timeout: int,
            *,
            commit_shas: Sequence[str],
            spawn_args: tuple[str, ...] = (),
            context_file: Path | None = None,
        ) -> object:
            spawn_cmd = ["cerberus", "spawn-code-review", "--exclude", ".beads/"]
            if context_file is not None:
                spawn_cmd.extend(["--context-file", str(context_file)])
            spawn_cmd.extend(spawn_args)
            spawn_cmd.append("--commit")
            spawn_cmd.extend(commit_shas)

            result = await runner.run_async(spawn_cmd, env=env, timeout=timeout)
            return cerberus_cli.SpawnResult(
                success=result.returncode == 0,
                timed_out=result.timed_out,
                error_detail=result.stderr_tail() or result.stdout_tail(),
                already_active="already active"
                in f"{result.stderr} {result.stdout}".lower(),
            )

        async def wait_for_review(
            self,
            session_id: str,
            runner: CommandRunnerPort,
            env: dict[str, str],
            cli_timeout: int,
            wait_args: tuple[str, ...] = (),
            user_timeout: int | None = None,
        ) -> object:
            _ = session_id
            wait_cmd = [
                "cerberus",
                "wait",
                "--json",
                "--finalize",
                "--session-key",
                env["CERBERUS_RUN_KEY"],
            ]
            if user_timeout is None:
                wait_cmd.extend(["--timeout", str(cli_timeout)])
            wait_cmd.extend(wait_args)

            result = await runner.run_async(wait_cmd, env=env, timeout=cli_timeout + 30)
            wait_result = cerberus_cli.WaitResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=result.timed_out,
            )
            cast("Any", wait_result).session_dir = (
                Path(env["CERBERUS_STATE_ROOT"])
                / env["CERBERUS_PROJECT_KEY"]
                / env["CERBERUS_RUN_KEY"]
            )
            return wait_result

        async def resolve_gate(
            self,
            runner: CommandRunnerPort,
            env: dict[str, str],
            reason: str = "mala: auto-clearing stale gate for retry",
        ) -> object:
            result = await runner.run_async(
                ["cerberus", "resolve", "--reason", reason], env=env, timeout=30
            )
            return cerberus_cli.ResolveResult(
                success=result.returncode == 0,
                error_detail=result.stderr_tail() or result.stdout_tail(),
            )

    monkeypatch.setattr(cerberus_cli, "CerberusGateCLI", CerberusGateCLI)
    monkeypatch.setattr(
        cerberus_review, "CerberusGateCLI", CerberusGateCLI, raising=False
    )


async def _run_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_name: str,
) -> tuple[ReviewResult, list[str]]:
    _install_partial_v2_import_compat(monkeypatch)
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
    _install_partial_v2_import_compat(monkeypatch)
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

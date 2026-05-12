"""Unit tests for CerberusEpicVerifier."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import fields
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from src.core.models import EpicVerdict
from src.infra.clients.cerberus_epic_verifier import (
    CerberusEpicVerifier,
    VerificationExecutionError,
    VerificationParseError,
    VerificationTimeoutError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


class FakeCommandResult:
    """Fake CommandResult for testing."""

    def __init__(
        self,
        returncode: int = 0,
        stdout: str = "",
        stderr: str = "",
        timed_out: bool = False,
    ) -> None:
        self.command: list[str] | str = []
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out
        self.duration_seconds = 1.0

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def stdout_tail(self, max_chars: int = 800, max_lines: int = 20) -> str:
        _ = max_lines
        return self.stdout[:max_chars]

    def stderr_tail(self, max_chars: int = 800, max_lines: int = 20) -> str:
        _ = max_lines
        return self.stderr[:max_chars]


def _make_cerberus_root(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "cerberus-root"
    binary_dir = root / "bin"
    binary_dir.mkdir(parents=True)
    (root / "prompts" / "reviewers").mkdir(parents=True)
    cerberus = binary_dir / "cerberus"
    cerberus.write_text("#!/usr/bin/env sh\nexit 0\n", encoding="utf-8")
    os.chmod(cerberus, stat.S_IRWXU)
    return root, binary_dir


def _make_verifier(tmp_path: Path) -> CerberusEpicVerifier:
    root, binary_dir = _make_cerberus_root(tmp_path)
    return CerberusEpicVerifier(
        repo_path=tmp_path,
        state_root=tmp_path / "cerberus-state",
        project_key="test-project",
        env={"PATH": str(binary_dir), "CERBERUS_ROOT": str(root)},
    )


def _gate_state(
    *,
    run_key: str = "mala-epic-fixed123",
    project_key: str = "test-project",
    verdict: str = "pass",
) -> dict[str, object]:
    return {
        "schema_version": "2",
        "run_key": run_key,
        "host": "generic",
        "project_key": project_key,
        "session_id": "epic-fixed123",
        "transcript_path": None,
        "status": "resolved",
        "verdict": verdict,
        "resolution_reason": None,
        "current_iteration": 1,
        "max_rounds": None,
        "debate": False,
        "roster_id": "epic",
        "started_at": "2026-05-11T00:00:00Z",
        "ended_at": "2026-05-11T00:00:01Z",
    }


def _write_reviewer_output(
    state_root: Path,
    *,
    run_key: str = "mala-epic-fixed123",
    project_key: str = "test-project",
    reviewer: str = "peer-a",
    findings: list[dict[str, object]] | None = None,
) -> None:
    output_path = (
        state_root
        / project_key
        / run_key
        / "iterations"
        / "1"
        / "round-0"
        / "reviewers"
        / reviewer
        / "output.json"
    )
    output_path.parent.mkdir(parents=True)
    output_path.write_text(
        json.dumps({"findings": findings or []}),
        encoding="utf-8",
    )


async def _run_verify_with_output(
    verifier: CerberusEpicVerifier,
    wait_output: dict[str, object] | str,
    *,
    wait_returncode: int = 0,
    wait_stderr: str = "",
) -> tuple[EpicVerdict, list[list[str]], list[dict[str, str]]]:
    captured_commands: list[list[str]] = []
    captured_envs: list[dict[str, str]] = []

    async def mock_run_async(
        cmd: list[str] | str,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        **kwargs: object,
    ) -> FakeCommandResult:
        _ = timeout, kwargs
        if isinstance(cmd, list):
            captured_commands.append(cmd)
        if env is not None:
            captured_envs.append(dict(env))
        if isinstance(cmd, list) and "spawn-epic-verify" in cmd:
            return FakeCommandResult(returncode=0)
        stdout = (
            wait_output if isinstance(wait_output, str) else json.dumps(wait_output)
        )
        return FakeCommandResult(
            returncode=wait_returncode,
            stdout=stdout,
            stderr=wait_stderr,
        )

    with patch.object(
        CerberusEpicVerifier,
        "_generate_session_id",
        return_value="epic-fixed123",
    ):
        with patch(
            "src.infra.clients.cerberus_epic_verifier.CommandRunner"
        ) as mock_runner_cls:
            mock_runner = AsyncMock()
            mock_runner.run_async = mock_run_async
            mock_runner_cls.return_value = mock_runner

            verdict = await verifier.verify(
                epic_context="## Acceptance Criteria\n- Must work"
            )

    return verdict, captured_commands, captured_envs


@pytest.mark.unit
class TestCerberusEpicVerifierCommands:
    """Command construction and env behavior."""

    def test_dataclass_surface_uses_v2_state_fields(self) -> None:
        field_names = {field.name for field in fields(CerberusEpicVerifier)}

        assert "state_root" in field_names
        assert "project_key" in field_names
        assert "bin_path" not in field_names

    @pytest.mark.asyncio
    async def test_spawn_wait_and_env_use_v2_contract(self, tmp_path: Path) -> None:
        verifier = _make_verifier(tmp_path)
        _write_reviewer_output(verifier.state_root or tmp_path)

        verdict, commands, envs = await _run_verify_with_output(
            verifier,
            _gate_state(),
        )

        assert verdict.passed is True
        spawn_cmd = next(cmd for cmd in commands if "spawn-epic-verify" in cmd)
        assert spawn_cmd[0:2] == ["cerberus", "spawn-epic-verify"]
        assert "--max-rounds" not in spawn_cmd
        assert spawn_cmd[-1].endswith(".md")

        wait_cmd = next(cmd for cmd in commands if cmd[1] == "wait")
        assert "--session-key" in wait_cmd
        assert wait_cmd[wait_cmd.index("--session-key") + 1] == "mala-epic-fixed123"
        assert "--session-id" not in wait_cmd

        spawn_env = envs[0]
        assert spawn_env["CERBERUS_HOST"] == "generic"
        assert spawn_env["CERBERUS_RUN_KEY"] == "mala-epic-fixed123"
        assert spawn_env["CERBERUS_STATE_ROOT"] == str(verifier.state_root)
        assert spawn_env["CERBERUS_PROJECT_KEY"] == "test-project"
        assert spawn_env["CERBERUS_ROOT"].endswith("cerberus-root")

    @pytest.mark.asyncio
    async def test_resolve_gate_is_not_called_on_normal_path(
        self, tmp_path: Path
    ) -> None:
        verifier = _make_verifier(tmp_path)
        _write_reviewer_output(verifier.state_root or tmp_path)

        with patch(
            "src.infra.clients.cerberus_epic_verifier.CerberusCLI.resolve_gate",
            new_callable=AsyncMock,
        ) as resolve_gate:
            await _run_verify_with_output(verifier, _gate_state())

        resolve_gate.assert_not_called()


@pytest.mark.unit
class TestCerberusEpicVerifierParsing:
    """Parsing v2 wait output and iteration findings to EpicVerdict."""

    @pytest.mark.asyncio
    async def test_parses_pass_verdict_with_populated_reviewers(
        self, tmp_path: Path
    ) -> None:
        verifier = _make_verifier(tmp_path)
        _write_reviewer_output(verifier.state_root or tmp_path)

        verdict, _, _ = await _run_verify_with_output(verifier, _gate_state())

        assert isinstance(verdict, EpicVerdict)
        assert verdict.passed is True
        assert verdict.unmet_criteria == []

    @pytest.mark.asyncio
    async def test_parses_fail_verdict_with_iteration_findings(
        self, tmp_path: Path
    ) -> None:
        verifier = _make_verifier(tmp_path)
        _write_reviewer_output(
            verifier.state_root or tmp_path,
            findings=[
                {
                    "title": "[P1] AC-1 missing wiring",
                    "body": "Unmet criterion evidence",
                    "priority": 1,
                    "file_path": "src/foo.py",
                    "line_start": 10,
                    "line_end": 12,
                }
            ],
        )

        verdict, _, _ = await _run_verify_with_output(
            verifier,
            _gate_state(verdict="fail"),
        )

        assert verdict.passed is False
        assert len(verdict.unmet_criteria) == 1
        assert verdict.unmet_criteria[0].criterion == "[P1] AC-1 missing wiring"
        assert verdict.unmet_criteria[0].evidence == "Unmet criterion evidence"
        assert verdict.unmet_criteria[0].priority == 1

    @pytest.mark.asyncio
    async def test_pass_with_missing_reviewers_fails_closed(
        self, tmp_path: Path
    ) -> None:
        verifier = _make_verifier(tmp_path)

        with pytest.raises(VerificationParseError, match="iteration findings"):
            await _run_verify_with_output(verifier, _gate_state())

    @pytest.mark.asyncio
    async def test_requires_decision_surfaces_unmet_criterion(
        self, tmp_path: Path
    ) -> None:
        verifier = _make_verifier(tmp_path)
        _write_reviewer_output(verifier.state_root or tmp_path)

        verdict, _, _ = await _run_verify_with_output(
            verifier,
            _gate_state(verdict="requires_decision"),
        )

        assert verdict.passed is False
        assert verdict.unmet_criteria
        assert "no consensus" in verdict.unmet_criteria[-1].criterion
        assert "requires_decision" in verdict.unmet_criteria[-1].evidence


@pytest.mark.unit
class TestCerberusEpicVerifierErrors:
    """Error handling and cleanup."""

    @pytest.mark.asyncio
    async def test_spawn_timeout_raises_verification_timeout_error(
        self, tmp_path: Path
    ) -> None:
        verifier = _make_verifier(tmp_path)

        async def mock_run_async(
            cmd: list[str] | str,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
            **kwargs: object,
        ) -> FakeCommandResult:
            _ = cmd, env, timeout, kwargs
            return FakeCommandResult(timed_out=True, returncode=-1)

        with patch(
            "src.infra.clients.cerberus_epic_verifier.CommandRunner"
        ) as mock_runner_cls:
            mock_runner = AsyncMock()
            mock_runner.run_async = mock_run_async
            mock_runner_cls.return_value = mock_runner

            with pytest.raises(VerificationTimeoutError, match="spawn-epic-verify"):
                await verifier.verify(epic_context="criteria")

    @pytest.mark.asyncio
    async def test_wait_timeout_exit_code_raises_verification_timeout_error(
        self, tmp_path: Path
    ) -> None:
        verifier = _make_verifier(tmp_path)
        call_count = 0

        async def mock_run_async(
            cmd: list[str] | str,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
            **kwargs: object,
        ) -> FakeCommandResult:
            _ = cmd, env, timeout, kwargs
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return FakeCommandResult(returncode=0)
            return FakeCommandResult(returncode=3, stdout="{}")

        with patch(
            "src.infra.clients.cerberus_epic_verifier.CommandRunner"
        ) as mock_runner_cls:
            mock_runner = AsyncMock()
            mock_runner.run_async = mock_run_async
            mock_runner_cls.return_value = mock_runner

            with pytest.raises(VerificationTimeoutError, match="wait timed out"):
                await verifier.verify(epic_context="criteria")

    @pytest.mark.asyncio
    async def test_invalid_json_raises_parse_error(self, tmp_path: Path) -> None:
        verifier = _make_verifier(tmp_path)

        with pytest.raises(VerificationParseError):
            await _run_verify_with_output(verifier, "not json")

    @pytest.mark.asyncio
    async def test_non_zero_wait_exit_raises_execution_error(
        self, tmp_path: Path
    ) -> None:
        verifier = _make_verifier(tmp_path)
        _write_reviewer_output(verifier.state_root or tmp_path)

        with pytest.raises(VerificationExecutionError, match="wait failed"):
            await _run_verify_with_output(
                verifier,
                _gate_state(),
                wait_returncode=2,
                wait_stderr="wait failed",
            )

    @pytest.mark.asyncio
    async def test_non_zero_wait_exit_with_invalid_stdout_raises_execution_error(
        self, tmp_path: Path
    ) -> None:
        verifier = _make_verifier(tmp_path)

        with pytest.raises(VerificationExecutionError, match="runtime failed"):
            await _run_verify_with_output(
                verifier,
                "",
                wait_returncode=2,
                wait_stderr="runtime failed",
            )

    @pytest.mark.asyncio
    async def test_cleanup_on_error(self, tmp_path: Path) -> None:
        verifier = _make_verifier(tmp_path)
        created_paths: list[Path] = []

        original_write = CerberusEpicVerifier._write_epic_file

        def tracking_write(
            self_arg: CerberusEpicVerifier,
            epic_text: str,
        ) -> Path:
            path = original_write(self_arg, epic_text)
            created_paths.append(path)
            return path

        async def mock_run_async(
            cmd: list[str] | str,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
            **kwargs: object,
        ) -> FakeCommandResult:
            _ = cmd, env, timeout, kwargs
            return FakeCommandResult(returncode=1, stderr="error")

        with patch.object(CerberusEpicVerifier, "_write_epic_file", tracking_write):
            with patch(
                "src.infra.clients.cerberus_epic_verifier.CommandRunner"
            ) as mock_runner_cls:
                mock_runner = AsyncMock()
                mock_runner.run_async = mock_run_async
                mock_runner_cls.return_value = mock_runner

                with pytest.raises(VerificationExecutionError):
                    await verifier.verify(epic_context="criteria")

        assert created_paths and not created_paths[0].exists()

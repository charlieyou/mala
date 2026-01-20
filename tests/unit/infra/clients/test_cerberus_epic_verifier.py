"""Unit tests for CerberusEpicVerifier.

Tests verify:
1. Command construction (spawn-epic-verify with diff args)
2. Parsing wait output to EpicVerdict
3. Timeout and error handling
4. Temp epic file cleanup
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
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
        return self.stdout[:max_chars]

    def stderr_tail(self, max_chars: int = 800, max_lines: int = 20) -> str:
        return self.stderr[:max_chars]


def _make_review_gate(tmp_path: Path) -> Path:
    bin_path = tmp_path / "bin"
    bin_path.mkdir()
    review_gate = bin_path / "review-gate"
    review_gate.write_text("#!/usr/bin/env sh\nexit 0\n")
    os.chmod(review_gate, stat.S_IRWXU)
    return bin_path


def _make_verifier(tmp_path: Path) -> CerberusEpicVerifier:
    bin_path = _make_review_gate(tmp_path)
    return CerberusEpicVerifier(
        repo_path=tmp_path,
        bin_path=bin_path,
        env={},
    )


@pytest.mark.unit
class TestCerberusEpicVerifierCommands:
    """Command construction and env behavior."""

    @pytest.mark.asyncio
    async def test_spawn_includes_epic_file(self, tmp_path: Path) -> None:
        verifier = _make_verifier(tmp_path)
        captured_commands: list[list[str]] = []
        captured_epic_paths: list[Path] = []

        wait_output = {
            "status": "complete",
            "consensus_verdict": "PASS",
            "reviewers": {},
            "aggregated_findings": [],
            "parse_errors": [],
        }

        async def mock_run_async(
            cmd: list[str] | str,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
            **kwargs: object,
        ) -> FakeCommandResult:
            if isinstance(cmd, list):
                captured_commands.append(cmd)
                if "spawn-epic-verify" in cmd:
                    epic_path = Path(cmd[-1])
                    captured_epic_paths.append(epic_path)
                    assert epic_path.exists()
                    return FakeCommandResult(returncode=0)
            return FakeCommandResult(returncode=0, stdout=json.dumps(wait_output))

        with patch(
            "src.infra.clients.cerberus_epic_verifier.CommandRunner"
        ) as mock_runner_cls:
            mock_runner = AsyncMock()
            mock_runner.run_async = mock_run_async
            mock_runner_cls.return_value = mock_runner

            await verifier.verify(
                epic_context="## Acceptance Criteria\n- Must work",
            )

        spawn_cmd = next(cmd for cmd in captured_commands if "spawn-epic-verify" in cmd)
        assert spawn_cmd[-1].endswith(".md")
        assert captured_epic_paths

    @pytest.mark.asyncio
    async def test_wait_command_includes_session_id(self, tmp_path: Path) -> None:
        verifier = _make_verifier(tmp_path)
        captured_commands: list[list[str]] = []

        wait_output = {
            "status": "complete",
            "consensus_verdict": "PASS",
            "reviewers": {},
            "aggregated_findings": [],
            "parse_errors": [],
        }

        async def mock_run_async(
            cmd: list[str] | str,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
            **kwargs: object,
        ) -> FakeCommandResult:
            if isinstance(cmd, list):
                captured_commands.append(cmd)
                if "spawn-epic-verify" in cmd:
                    return FakeCommandResult(returncode=0)
            return FakeCommandResult(returncode=0, stdout=json.dumps(wait_output))

        with patch(
            "src.infra.clients.cerberus_epic_verifier.CommandRunner"
        ) as mock_runner_cls:
            mock_runner = AsyncMock()
            mock_runner.run_async = mock_run_async
            mock_runner_cls.return_value = mock_runner

            await verifier.verify(epic_context="criteria")

        wait_cmd = next(cmd for cmd in captured_commands if cmd[1] == "wait")
        assert "--session-id" in wait_cmd


@pytest.mark.unit
class TestCerberusEpicVerifierParsing:
    """Parsing wait output to EpicVerdict."""

    @pytest.mark.asyncio
    async def test_parses_fail_verdict_with_findings(self, tmp_path: Path) -> None:
        verifier = _make_verifier(tmp_path)
        wait_output = {
            "status": "complete",
            "consensus_verdict": "FAIL",
            "reviewers": {},
            "aggregated_findings": [
                {
                    "title": "[P1] AC-1 missing wiring",
                    "body": "Unmet criterion evidence",
                    "priority": 1,
                    "file_path": "src/foo.py",
                    "line_start": 10,
                    "line_end": 12,
                }
            ],
            "parse_errors": [],
        }

        async def mock_run_async(
            cmd: list[str] | str,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
            **kwargs: object,
        ) -> FakeCommandResult:
            if isinstance(cmd, list) and "spawn-epic-verify" in cmd:
                return FakeCommandResult(returncode=0)
            return FakeCommandResult(returncode=1, stdout=json.dumps(wait_output))

        with patch(
            "src.infra.clients.cerberus_epic_verifier.CommandRunner"
        ) as mock_runner_cls:
            mock_runner = AsyncMock()
            mock_runner.run_async = mock_run_async
            mock_runner_cls.return_value = mock_runner

            verdict = await verifier.verify(epic_context="criteria")

        assert isinstance(verdict, EpicVerdict)
        assert verdict.passed is False
        assert len(verdict.unmet_criteria) == 1
        assert verdict.unmet_criteria[0].criterion == "[P1] AC-1 missing wiring"
        assert verdict.unmet_criteria[0].evidence == "Unmet criterion evidence"
        assert verdict.unmet_criteria[0].priority == 1

    @pytest.mark.asyncio
    async def test_parses_pass_verdict(self, tmp_path: Path) -> None:
        verifier = _make_verifier(tmp_path)
        wait_output = {
            "status": "complete",
            "consensus_verdict": "PASS",
            "reviewers": {},
            "aggregated_findings": [],
            "parse_errors": [],
        }

        async def mock_run_async(
            cmd: list[str] | str,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
            **kwargs: object,
        ) -> FakeCommandResult:
            if isinstance(cmd, list) and "spawn-epic-verify" in cmd:
                return FakeCommandResult(returncode=0)
            return FakeCommandResult(returncode=0, stdout=json.dumps(wait_output))

        with patch(
            "src.infra.clients.cerberus_epic_verifier.CommandRunner"
        ) as mock_runner_cls:
            mock_runner = AsyncMock()
            mock_runner.run_async = mock_run_async
            mock_runner_cls.return_value = mock_runner

            verdict = await verifier.verify(epic_context="criteria")

        assert verdict.passed is True
        assert verdict.unmet_criteria == []


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
        call_count = 0

        async def mock_run_async(
            cmd: list[str] | str,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
            **kwargs: object,
        ) -> FakeCommandResult:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return FakeCommandResult(returncode=0)
            return FakeCommandResult(returncode=2, stdout="not json")

        with patch(
            "src.infra.clients.cerberus_epic_verifier.CommandRunner"
        ) as mock_runner_cls:
            mock_runner = AsyncMock()
            mock_runner.run_async = mock_run_async
            mock_runner_cls.return_value = mock_runner

            with pytest.raises(VerificationParseError):
                await verifier.verify(epic_context="criteria")

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

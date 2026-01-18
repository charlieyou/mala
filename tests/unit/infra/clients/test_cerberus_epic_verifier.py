"""Unit tests for CerberusEpicVerifier.

Tests verify:
1. Command construction (`spawn-epic-review --context-json <path>`)
2. Parse valid JSON response → EpicVerdict
3. Handle timeout → raise VerificationTimeoutError
4. Handle non-zero exit → raise VerificationExecutionError
5. Handle invalid JSON → raise VerificationParseError
6. Context JSON format verification
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, patch

import pytest

from src.core.models import EpicVerdict
from src.infra.clients.cerberus_epic_verifier import (
    CONTEXT_SCHEMA_VERSION,
    CerberusEpicVerifier,
    VerificationExecutionError,
    VerificationParseError,
    VerificationTimeoutError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


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


@pytest.mark.unit
class TestCerberusEpicVerifierCommandConstruction:
    """Tests for command construction."""

    @pytest.mark.asyncio
    async def test_spawn_command_includes_context_json_flag(
        self, tmp_path: Path
    ) -> None:
        """spawn-epic-review command includes --context-json flag."""
        verifier = CerberusEpicVerifier(repo_path=tmp_path)
        captured_commands: list[list[str]] = []

        async def mock_run_async(
            cmd: list[str] | str,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
            **kwargs: object,
        ) -> FakeCommandResult:
            if isinstance(cmd, list):
                captured_commands.append(cmd)
            # Return success for spawn, then valid JSON for wait
            if len(captured_commands) == 1:
                return FakeCommandResult(returncode=0)
            return FakeCommandResult(
                returncode=0,
                stdout='{"passed": true, "unmet_criteria": [], "confidence": 0.95, "reasoning": "All good"}',
            )

        with patch.object(
            CerberusEpicVerifier,
            "_write_context_json",
            return_value=tmp_path / "ctx.json",
        ):
            with patch(
                "src.infra.clients.cerberus_epic_verifier.CommandRunner"
            ) as mock_runner_cls:
                mock_runner = AsyncMock()
                mock_runner.run_async = mock_run_async
                mock_runner_cls.return_value = mock_runner

                await verifier.verify(
                    epic_criteria="Test criteria",
                    commit_range="abc..def",
                    commit_list="abc,def",
                    spec_content="Spec content",
                    epic_id="test-epic",
                )

        # First command should be spawn-epic-review
        spawn_cmd = captured_commands[0]
        assert "spawn-epic-review" in spawn_cmd
        assert "--context-json" in spawn_cmd

    @pytest.mark.asyncio
    async def test_uses_bin_path_when_provided(self, tmp_path: Path) -> None:
        """Uses bin_path for review-gate binary when provided."""
        bin_path = tmp_path / "bin"
        bin_path.mkdir()
        verifier = CerberusEpicVerifier(repo_path=tmp_path, bin_path=bin_path)

        captured_commands: list[list[str]] = []

        async def mock_run_async(
            cmd: list[str] | str,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
            **kwargs: object,
        ) -> FakeCommandResult:
            if isinstance(cmd, list):
                captured_commands.append(cmd)
            if len(captured_commands) == 1:
                return FakeCommandResult(returncode=0)
            return FakeCommandResult(
                returncode=0,
                stdout='{"passed": true, "unmet_criteria": [], "confidence": 0.9, "reasoning": "OK"}',
            )

        with patch.object(
            CerberusEpicVerifier,
            "_write_context_json",
            return_value=tmp_path / "ctx.json",
        ):
            with patch(
                "src.infra.clients.cerberus_epic_verifier.CommandRunner"
            ) as mock_runner_cls:
                mock_runner = AsyncMock()
                mock_runner.run_async = mock_run_async
                mock_runner_cls.return_value = mock_runner

                await verifier.verify(
                    epic_criteria="Test",
                    commit_range="a..b",
                    commit_list="a",
                    spec_content=None,
                )

        spawn_cmd = captured_commands[0]
        assert spawn_cmd[0] == str(bin_path / "review-gate")

    @pytest.mark.asyncio
    async def test_wait_command_includes_json_and_timeout(self, tmp_path: Path) -> None:
        """wait command includes --json and --timeout flags."""
        verifier = CerberusEpicVerifier(repo_path=tmp_path, timeout=120)
        captured_commands: list[list[str]] = []

        async def mock_run_async(
            cmd: list[str] | str,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
            **kwargs: object,
        ) -> FakeCommandResult:
            if isinstance(cmd, list):
                captured_commands.append(cmd)
            if len(captured_commands) == 1:
                return FakeCommandResult(returncode=0)
            return FakeCommandResult(
                returncode=0,
                stdout='{"passed": true, "unmet_criteria": [], "confidence": 0.9, "reasoning": "OK"}',
            )

        with patch.object(
            CerberusEpicVerifier,
            "_write_context_json",
            return_value=tmp_path / "ctx.json",
        ):
            with patch(
                "src.infra.clients.cerberus_epic_verifier.CommandRunner"
            ) as mock_runner_cls:
                mock_runner = AsyncMock()
                mock_runner.run_async = mock_run_async
                mock_runner_cls.return_value = mock_runner

                await verifier.verify(
                    epic_criteria="Test",
                    commit_range="a..b",
                    commit_list="a",
                    spec_content=None,
                )

        # Second command should be wait
        wait_cmd = captured_commands[1]
        assert "wait" in wait_cmd
        assert "--json" in wait_cmd
        assert "--timeout" in wait_cmd
        timeout_idx = wait_cmd.index("--timeout")
        assert wait_cmd[timeout_idx + 1] == "120"


@pytest.mark.unit
class TestCerberusEpicVerifierParsing:
    """Tests for JSON response parsing."""

    @pytest.mark.asyncio
    async def test_parses_valid_json_to_epic_verdict(self, tmp_path: Path) -> None:
        """Valid JSON response is parsed to EpicVerdict."""
        verifier = CerberusEpicVerifier(repo_path=tmp_path)
        valid_response = {
            "passed": False,
            "unmet_criteria": [
                {
                    "criterion": "Feature X must be implemented",
                    "evidence": "No implementation found",
                    "priority": 1,
                    "criterion_hash": "abc123",
                }
            ],
            "confidence": 0.85,
            "reasoning": "Feature X is missing from the codebase",
        }

        async def mock_run_async(
            cmd: list[str] | str,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
            **kwargs: object,
        ) -> FakeCommandResult:
            if isinstance(cmd, list) and "spawn-epic-review" in cmd:
                return FakeCommandResult(returncode=0)
            return FakeCommandResult(returncode=0, stdout=json.dumps(valid_response))

        with patch.object(
            CerberusEpicVerifier,
            "_write_context_json",
            return_value=tmp_path / "ctx.json",
        ):
            with patch(
                "src.infra.clients.cerberus_epic_verifier.CommandRunner"
            ) as mock_runner_cls:
                mock_runner = AsyncMock()
                mock_runner.run_async = mock_run_async
                mock_runner_cls.return_value = mock_runner

                verdict = await verifier.verify(
                    epic_criteria="Test",
                    commit_range="a..b",
                    commit_list="a",
                    spec_content=None,
                )

        assert isinstance(verdict, EpicVerdict)
        assert verdict.passed is False
        assert len(verdict.unmet_criteria) == 1
        assert verdict.unmet_criteria[0].criterion == "Feature X must be implemented"
        assert verdict.unmet_criteria[0].evidence == "No implementation found"
        assert verdict.unmet_criteria[0].priority == 1
        assert verdict.confidence == 0.85
        assert verdict.reasoning == "Feature X is missing from the codebase"

    @pytest.mark.asyncio
    async def test_parses_passing_verdict(self, tmp_path: Path) -> None:
        """Passing verdict is parsed correctly."""
        verifier = CerberusEpicVerifier(repo_path=tmp_path)
        valid_response = {
            "passed": True,
            "unmet_criteria": [],
            "confidence": 0.99,
            "reasoning": "All criteria met",
        }

        async def mock_run_async(
            cmd: list[str] | str,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
            **kwargs: object,
        ) -> FakeCommandResult:
            if isinstance(cmd, list) and "spawn-epic-review" in cmd:
                return FakeCommandResult(returncode=0)
            return FakeCommandResult(returncode=0, stdout=json.dumps(valid_response))

        with patch.object(
            CerberusEpicVerifier,
            "_write_context_json",
            return_value=tmp_path / "ctx.json",
        ):
            with patch(
                "src.infra.clients.cerberus_epic_verifier.CommandRunner"
            ) as mock_runner_cls:
                mock_runner = AsyncMock()
                mock_runner.run_async = mock_run_async
                mock_runner_cls.return_value = mock_runner

                verdict = await verifier.verify(
                    epic_criteria="Test",
                    commit_range="a..b",
                    commit_list="a",
                    spec_content=None,
                )

        assert verdict.passed is True
        assert len(verdict.unmet_criteria) == 0


@pytest.mark.unit
class TestCerberusEpicVerifierTimeout:
    """Tests for timeout handling."""

    @pytest.mark.asyncio
    async def test_spawn_timeout_raises_verification_timeout_error(
        self, tmp_path: Path
    ) -> None:
        """Spawn timeout raises VerificationTimeoutError."""
        verifier = CerberusEpicVerifier(repo_path=tmp_path, timeout=60)

        async def mock_run_async(
            cmd: list[str] | str,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
            **kwargs: object,
        ) -> FakeCommandResult:
            return FakeCommandResult(timed_out=True, returncode=-1)

        with patch.object(
            CerberusEpicVerifier,
            "_write_context_json",
            return_value=tmp_path / "ctx.json",
        ):
            with patch(
                "src.infra.clients.cerberus_epic_verifier.CommandRunner"
            ) as mock_runner_cls:
                mock_runner = AsyncMock()
                mock_runner.run_async = mock_run_async
                mock_runner_cls.return_value = mock_runner

                with pytest.raises(VerificationTimeoutError) as exc_info:
                    await verifier.verify(
                        epic_criteria="Test",
                        commit_range="a..b",
                        commit_list="a",
                        spec_content=None,
                    )

        assert "spawn-epic-review timed out" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_wait_timeout_raises_verification_timeout_error(
        self, tmp_path: Path
    ) -> None:
        """Wait timeout raises VerificationTimeoutError."""
        verifier = CerberusEpicVerifier(repo_path=tmp_path, timeout=60)
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
                return FakeCommandResult(returncode=0)  # spawn succeeds
            return FakeCommandResult(timed_out=True, returncode=-1)  # wait times out

        with patch.object(
            CerberusEpicVerifier,
            "_write_context_json",
            return_value=tmp_path / "ctx.json",
        ):
            with patch(
                "src.infra.clients.cerberus_epic_verifier.CommandRunner"
            ) as mock_runner_cls:
                mock_runner = AsyncMock()
                mock_runner.run_async = mock_run_async
                mock_runner_cls.return_value = mock_runner

                with pytest.raises(VerificationTimeoutError) as exc_info:
                    await verifier.verify(
                        epic_criteria="Test",
                        commit_range="a..b",
                        commit_list="a",
                        spec_content=None,
                    )

        assert "wait timed out" in str(exc_info.value)


@pytest.mark.unit
class TestCerberusEpicVerifierExecutionError:
    """Tests for non-zero exit code handling."""

    @pytest.mark.asyncio
    async def test_spawn_nonzero_exit_raises_execution_error(
        self, tmp_path: Path
    ) -> None:
        """Non-zero spawn exit raises VerificationExecutionError."""
        verifier = CerberusEpicVerifier(repo_path=tmp_path)

        async def mock_run_async(
            cmd: list[str] | str,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
            **kwargs: object,
        ) -> FakeCommandResult:
            return FakeCommandResult(
                returncode=1, stderr="spawn error: binary not found"
            )

        with patch.object(
            CerberusEpicVerifier,
            "_write_context_json",
            return_value=tmp_path / "ctx.json",
        ):
            with patch(
                "src.infra.clients.cerberus_epic_verifier.CommandRunner"
            ) as mock_runner_cls:
                mock_runner = AsyncMock()
                mock_runner.run_async = mock_run_async
                mock_runner_cls.return_value = mock_runner

                with pytest.raises(VerificationExecutionError) as exc_info:
                    await verifier.verify(
                        epic_criteria="Test",
                        commit_range="a..b",
                        commit_list="a",
                        spec_content=None,
                    )

        assert "spawn-epic-review failed" in str(exc_info.value)
        assert "exit 1" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_wait_nonzero_exit_raises_execution_error(
        self, tmp_path: Path
    ) -> None:
        """Non-zero wait exit raises VerificationExecutionError."""
        verifier = CerberusEpicVerifier(repo_path=tmp_path)
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
            return FakeCommandResult(
                returncode=2, stderr="wait error: no active review"
            )

        with patch.object(
            CerberusEpicVerifier,
            "_write_context_json",
            return_value=tmp_path / "ctx.json",
        ):
            with patch(
                "src.infra.clients.cerberus_epic_verifier.CommandRunner"
            ) as mock_runner_cls:
                mock_runner = AsyncMock()
                mock_runner.run_async = mock_run_async
                mock_runner_cls.return_value = mock_runner

                with pytest.raises(VerificationExecutionError) as exc_info:
                    await verifier.verify(
                        epic_criteria="Test",
                        commit_range="a..b",
                        commit_list="a",
                        spec_content=None,
                    )

        assert "wait failed" in str(exc_info.value)
        assert "exit 2" in str(exc_info.value)


@pytest.mark.unit
class TestCerberusEpicVerifierParseError:
    """Tests for invalid JSON handling."""

    @pytest.mark.asyncio
    async def test_invalid_json_raises_parse_error(self, tmp_path: Path) -> None:
        """Invalid JSON raises VerificationParseError."""
        verifier = CerberusEpicVerifier(repo_path=tmp_path)
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
            return FakeCommandResult(returncode=0, stdout="not valid json {{{")

        with patch.object(
            CerberusEpicVerifier,
            "_write_context_json",
            return_value=tmp_path / "ctx.json",
        ):
            with patch(
                "src.infra.clients.cerberus_epic_verifier.CommandRunner"
            ) as mock_runner_cls:
                mock_runner = AsyncMock()
                mock_runner.run_async = mock_run_async
                mock_runner_cls.return_value = mock_runner

                with pytest.raises(VerificationParseError) as exc_info:
                    await verifier.verify(
                        epic_criteria="Test",
                        commit_range="a..b",
                        commit_list="a",
                        spec_content=None,
                    )

        assert "Invalid JSON response" in str(exc_info.value)
        assert "not valid json" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_missing_passed_field_raises_parse_error(
        self, tmp_path: Path
    ) -> None:
        """Missing 'passed' field raises VerificationParseError."""
        verifier = CerberusEpicVerifier(repo_path=tmp_path)
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
            return FakeCommandResult(
                returncode=0,
                stdout='{"confidence": 0.9, "reasoning": "test"}',
            )

        with patch.object(
            CerberusEpicVerifier,
            "_write_context_json",
            return_value=tmp_path / "ctx.json",
        ):
            with patch(
                "src.infra.clients.cerberus_epic_verifier.CommandRunner"
            ) as mock_runner_cls:
                mock_runner = AsyncMock()
                mock_runner.run_async = mock_run_async
                mock_runner_cls.return_value = mock_runner

                with pytest.raises(VerificationParseError) as exc_info:
                    await verifier.verify(
                        epic_criteria="Test",
                        commit_range="a..b",
                        commit_list="a",
                        spec_content=None,
                    )

        assert "Missing required field" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_wrong_type_for_passed_raises_parse_error(
        self, tmp_path: Path
    ) -> None:
        """Wrong type for 'passed' raises VerificationParseError."""
        verifier = CerberusEpicVerifier(repo_path=tmp_path)
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
            return FakeCommandResult(
                returncode=0,
                stdout='{"passed": "yes", "unmet_criteria": [], "confidence": 0.9, "reasoning": "test"}',
            )

        with patch.object(
            CerberusEpicVerifier,
            "_write_context_json",
            return_value=tmp_path / "ctx.json",
        ):
            with patch(
                "src.infra.clients.cerberus_epic_verifier.CommandRunner"
            ) as mock_runner_cls:
                mock_runner = AsyncMock()
                mock_runner.run_async = mock_run_async
                mock_runner_cls.return_value = mock_runner

                with pytest.raises(VerificationParseError) as exc_info:
                    await verifier.verify(
                        epic_criteria="Test",
                        commit_range="a..b",
                        commit_list="a",
                        spec_content=None,
                    )

        assert "'passed' must be boolean" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_empty_response_raises_parse_error(self, tmp_path: Path) -> None:
        """Empty response raises VerificationParseError."""
        verifier = CerberusEpicVerifier(repo_path=tmp_path)
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
            return FakeCommandResult(returncode=0, stdout="")

        with patch.object(
            CerberusEpicVerifier,
            "_write_context_json",
            return_value=tmp_path / "ctx.json",
        ):
            with patch(
                "src.infra.clients.cerberus_epic_verifier.CommandRunner"
            ) as mock_runner_cls:
                mock_runner = AsyncMock()
                mock_runner.run_async = mock_run_async
                mock_runner_cls.return_value = mock_runner

                with pytest.raises(VerificationParseError) as exc_info:
                    await verifier.verify(
                        epic_criteria="Test",
                        commit_range="a..b",
                        commit_list="a",
                        spec_content=None,
                    )

        assert "Invalid JSON response" in str(exc_info.value)


@pytest.mark.unit
class TestCerberusEpicVerifierContextJson:
    """Tests for context JSON format."""

    def test_context_json_contains_required_fields(self, tmp_path: Path) -> None:
        """Context JSON contains all required fields per spec."""
        verifier = CerberusEpicVerifier(repo_path=tmp_path)

        context_path = verifier._write_context_json(
            epic_id="test-epic-123",
            acceptance_criteria="Feature X must work",
            spec_content="# Spec\nDetails here",
            commit_range="abc123..def456",
            commit_list=["abc123", "def456", "ghi789"],
        )

        try:
            with open(context_path) as f:
                context = json.load(f)

            assert context["epic_id"] == "test-epic-123"
            assert context["acceptance_criteria"] == "Feature X must work"
            assert context["spec_content"] == "# Spec\nDetails here"
            assert context["commit_range"] == "abc123..def456"
            assert context["commit_list"] == ["abc123", "def456", "ghi789"]
            assert context["schema_version"] == CONTEXT_SCHEMA_VERSION
        finally:
            context_path.unlink(missing_ok=True)

    def test_context_json_handles_none_spec_content(self, tmp_path: Path) -> None:
        """Context JSON handles None spec_content as empty string."""
        verifier = CerberusEpicVerifier(repo_path=tmp_path)

        context_path = verifier._write_context_json(
            epic_id="test",
            acceptance_criteria="Test",
            spec_content=None,
            commit_range="a..b",
            commit_list=[],
        )

        try:
            with open(context_path) as f:
                context = json.load(f)

            assert context["spec_content"] == ""
        finally:
            context_path.unlink(missing_ok=True)

    def test_context_json_file_is_created_in_repo_dir(self, tmp_path: Path) -> None:
        """Context JSON file is created in repo directory."""
        verifier = CerberusEpicVerifier(repo_path=tmp_path)

        context_path = verifier._write_context_json(
            epic_id="test",
            acceptance_criteria="Test",
            spec_content=None,
            commit_range="a..b",
            commit_list=[],
        )

        try:
            assert context_path.parent == tmp_path
            assert context_path.name.startswith("epic-verify-test-")
            assert context_path.name.endswith(".json")
        finally:
            context_path.unlink(missing_ok=True)


@pytest.mark.unit
class TestCerberusEpicVerifierCleanup:
    """Tests for temp file cleanup."""

    @pytest.mark.asyncio
    async def test_context_file_cleaned_up_on_success(self, tmp_path: Path) -> None:
        """Context JSON file is cleaned up after successful verification."""
        verifier = CerberusEpicVerifier(repo_path=tmp_path)
        created_paths: list[Path] = []

        # Save reference to original before patching
        original_write = CerberusEpicVerifier._write_context_json

        def tracking_write(
            self_arg: CerberusEpicVerifier,
            epic_id: str,
            acceptance_criteria: str,
            spec_content: str | None,
            commit_range: str,
            commit_list: Sequence[str],
        ) -> Path:
            path = original_write(
                self_arg,
                epic_id,
                acceptance_criteria,
                spec_content,
                commit_range,
                commit_list,
            )
            created_paths.append(path)
            return path

        async def mock_run_async(
            cmd: list[str] | str,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
            **kwargs: object,
        ) -> FakeCommandResult:
            if isinstance(cmd, list) and "spawn-epic-review" in cmd:
                return FakeCommandResult(returncode=0)
            return FakeCommandResult(
                returncode=0,
                stdout='{"passed": true, "unmet_criteria": [], "confidence": 0.9, "reasoning": "OK"}',
            )

        with patch.object(CerberusEpicVerifier, "_write_context_json", tracking_write):
            with patch(
                "src.infra.clients.cerberus_epic_verifier.CommandRunner"
            ) as mock_runner_cls:
                mock_runner = AsyncMock()
                mock_runner.run_async = mock_run_async
                mock_runner_cls.return_value = mock_runner

                await verifier.verify(
                    epic_criteria="Test",
                    commit_range="a..b",
                    commit_list="a",
                    spec_content=None,
                )

        # File should be cleaned up
        assert len(created_paths) == 1
        assert not created_paths[0].exists()

    @pytest.mark.asyncio
    async def test_context_file_cleaned_up_on_error(self, tmp_path: Path) -> None:
        """Context JSON file is cleaned up even after error."""
        verifier = CerberusEpicVerifier(repo_path=tmp_path)
        created_paths: list[Path] = []

        # Save reference to original before patching
        original_write = CerberusEpicVerifier._write_context_json

        def tracking_write(
            self_arg: CerberusEpicVerifier,
            epic_id: str,
            acceptance_criteria: str,
            spec_content: str | None,
            commit_range: str,
            commit_list: Sequence[str],
        ) -> Path:
            path = original_write(
                self_arg,
                epic_id,
                acceptance_criteria,
                spec_content,
                commit_range,
                commit_list,
            )
            created_paths.append(path)
            return path

        async def mock_run_async(
            cmd: list[str] | str,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
            **kwargs: object,
        ) -> FakeCommandResult:
            return FakeCommandResult(returncode=1, stderr="error")

        with patch.object(CerberusEpicVerifier, "_write_context_json", tracking_write):
            with patch(
                "src.infra.clients.cerberus_epic_verifier.CommandRunner"
            ) as mock_runner_cls:
                mock_runner = AsyncMock()
                mock_runner.run_async = mock_run_async
                mock_runner_cls.return_value = mock_runner

                with pytest.raises(VerificationExecutionError):
                    await verifier.verify(
                        epic_criteria="Test",
                        commit_range="a..b",
                        commit_list="a",
                        spec_content=None,
                    )

        # File should still be cleaned up
        assert len(created_paths) == 1
        assert not created_paths[0].exists()


@pytest.mark.unit
class TestCerberusEpicVerifierProtocolCompliance:
    """Tests that verify protocol compliance."""

    def test_implements_epic_verification_model_protocol(self) -> None:
        """CerberusEpicVerifier implements EpicVerificationModel protocol."""
        from src.core.protocols.validation import EpicVerificationModel

        # Structural typing check
        verifier = CerberusEpicVerifier(repo_path=Path("/tmp"))
        assert isinstance(verifier, EpicVerificationModel)

    @pytest.mark.asyncio
    async def test_verify_signature_matches_protocol(self, tmp_path: Path) -> None:
        """verify() method signature matches protocol."""
        verifier = CerberusEpicVerifier(repo_path=tmp_path)

        # Protocol requires: verify(epic_criteria, commit_range, commit_list, spec_content) -> EpicVerdictProtocol
        async def mock_run_async(
            cmd: list[str] | str,
            env: Mapping[str, str] | None = None,
            timeout: float | None = None,
            **kwargs: object,
        ) -> FakeCommandResult:
            if isinstance(cmd, list) and "spawn-epic-review" in cmd:
                return FakeCommandResult(returncode=0)
            return FakeCommandResult(
                returncode=0,
                stdout='{"passed": true, "unmet_criteria": [], "confidence": 0.9, "reasoning": "OK"}',
            )

        with patch.object(
            CerberusEpicVerifier,
            "_write_context_json",
            return_value=tmp_path / "ctx.json",
        ):
            with patch(
                "src.infra.clients.cerberus_epic_verifier.CommandRunner"
            ) as mock_runner_cls:
                mock_runner = AsyncMock()
                mock_runner.run_async = mock_run_async
                mock_runner_cls.return_value = mock_runner

                # Call with protocol-required arguments
                result = await verifier.verify(
                    epic_criteria="criteria",
                    commit_range="range",
                    commit_list="sha1,sha2",
                    spec_content="spec",
                )

        # Result should match EpicVerdictProtocol
        assert hasattr(result, "passed")
        assert hasattr(result, "unmet_criteria")
        assert hasattr(result, "confidence")
        assert hasattr(result, "reasoning")

"""Unit tests for quality gate with offset-based parsing.

Tests for:
- Byte offset-based evidence parsing (parse_validation_evidence_from_offset)
- No-progress detection (check_no_progress)
"""

import json
from pathlib import Path
import subprocess
from unittest.mock import patch

from src.quality_gate import QualityGate, ValidationEvidence


class TestOffsetBasedParsing:
    """Test parse_validation_evidence_from_offset for scoping evidence by attempt."""

    def test_returns_evidence_and_new_offset(self, tmp_path: Path) -> None:
        """Should return tuple of (evidence, new_offset)."""
        log_path = tmp_path / "session.jsonl"
        log_content = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "uv run pytest"},
                        }
                    ]
                },
            }
        )
        log_path.write_text(log_content + "\n")

        gate = QualityGate(tmp_path)
        evidence, new_offset = gate.parse_validation_evidence_from_offset(log_path)

        assert isinstance(evidence, ValidationEvidence)
        assert isinstance(new_offset, int)
        assert new_offset > 0

    def test_starts_from_given_offset(self, tmp_path: Path) -> None:
        """Should only parse log entries after the given byte offset."""
        log_path = tmp_path / "session.jsonl"

        # First entry: pytest (before offset)
        first_entry = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "uv run pytest"},
                        }
                    ]
                },
            }
        )
        # Second entry: ruff check (after offset)
        second_entry = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "uvx ruff check ."},
                        }
                    ]
                },
            }
        )
        log_path.write_text(first_entry + "\n" + second_entry + "\n")

        gate = QualityGate(tmp_path)

        # Offset set to after the first line
        offset = len(first_entry) + 1  # +1 for newline
        evidence, _new_offset = gate.parse_validation_evidence_from_offset(
            log_path, offset=offset
        )

        # Should NOT detect pytest (before offset)
        assert evidence.pytest_ran is False
        # Should detect ruff check (after offset)
        assert evidence.ruff_check_ran is True

    def test_offset_zero_parses_entire_file(self, tmp_path: Path) -> None:
        """Offset=0 should parse the entire file (default behavior)."""
        log_path = tmp_path / "session.jsonl"
        log_content = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "uv run pytest"},
                        }
                    ]
                },
            }
        )
        log_path.write_text(log_content + "\n")

        gate = QualityGate(tmp_path)
        evidence, _ = gate.parse_validation_evidence_from_offset(log_path, offset=0)

        assert evidence.pytest_ran is True

    def test_offset_at_end_returns_empty_evidence(self, tmp_path: Path) -> None:
        """Offset at EOF should return empty evidence."""
        log_path = tmp_path / "session.jsonl"
        log_content = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "uv run pytest"},
                        }
                    ]
                },
            }
        )
        log_path.write_text(log_content + "\n")

        gate = QualityGate(tmp_path)
        # Set offset to end of file
        file_size = log_path.stat().st_size
        evidence, new_offset = gate.parse_validation_evidence_from_offset(
            log_path, offset=file_size
        )

        assert evidence.pytest_ran is False
        assert evidence.ruff_check_ran is False
        assert evidence.ruff_format_ran is False
        assert new_offset == file_size

    def test_new_offset_points_to_end_of_file(self, tmp_path: Path) -> None:
        """Returned offset should point to the current end of the file."""
        log_path = tmp_path / "session.jsonl"
        log_content = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "uv run pytest"},
                        }
                    ]
                },
            }
        )
        log_path.write_text(log_content + "\n")

        gate = QualityGate(tmp_path)
        _, new_offset = gate.parse_validation_evidence_from_offset(log_path)

        assert new_offset == log_path.stat().st_size

    def test_handles_missing_file(self, tmp_path: Path) -> None:
        """Should handle missing log file gracefully."""
        gate = QualityGate(tmp_path)
        nonexistent = tmp_path / "nonexistent.jsonl"

        evidence, new_offset = gate.parse_validation_evidence_from_offset(nonexistent)

        assert evidence.pytest_ran is False
        assert new_offset == 0

    def test_detects_all_validation_commands_after_offset(self, tmp_path: Path) -> None:
        """Should detect all validation commands after the given offset."""
        log_path = tmp_path / "session.jsonl"

        # Write entries for all validation commands
        entries = []
        commands = [
            "uv sync",
            "uv run pytest tests/",
            "uvx ruff check .",
            "uvx ruff format .",
            "uvx ty check",
        ]
        for cmd in commands:
            entries.append(
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "name": "Bash",
                                    "input": {"command": cmd},
                                }
                            ]
                        },
                    }
                )
            )
        log_path.write_text("\n".join(entries) + "\n")

        gate = QualityGate(tmp_path)
        evidence, _ = gate.parse_validation_evidence_from_offset(log_path, offset=0)

        assert evidence.uv_sync_ran is True
        assert evidence.pytest_ran is True
        assert evidence.ruff_check_ran is True
        assert evidence.ruff_format_ran is True
        assert evidence.ty_check_ran is True


class TestNoProgressDetection:
    """Test check_no_progress for detecting stalled attempts."""

    def test_no_progress_when_same_commit_and_no_new_evidence(
        self, tmp_path: Path
    ) -> None:
        """No progress: unchanged commit hash + no new evidence since offset."""
        log_path = tmp_path / "session.jsonl"
        # Empty file - no new evidence
        log_path.write_text("")

        gate = QualityGate(tmp_path)
        # Same commit as before, file has no content after offset 0
        is_no_progress = gate.check_no_progress(
            log_path=log_path,
            log_offset=0,
            previous_commit_hash="abc1234",
            current_commit_hash="abc1234",
        )

        assert is_no_progress is True

    def test_progress_when_commit_changed(self, tmp_path: Path) -> None:
        """Progress detected: different commit hash (even without new evidence)."""
        log_path = tmp_path / "session.jsonl"
        log_path.write_text("")  # No new evidence

        gate = QualityGate(tmp_path)
        is_no_progress = gate.check_no_progress(
            log_path=log_path,
            log_offset=0,
            previous_commit_hash="abc1234",
            current_commit_hash="def5678",  # Different commit
        )

        assert is_no_progress is False

    def test_progress_when_new_evidence_found(self, tmp_path: Path) -> None:
        """Progress detected: new validation evidence (even with same commit)."""
        log_path = tmp_path / "session.jsonl"
        log_content = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "uv run pytest"},
                        }
                    ]
                },
            }
        )
        log_path.write_text(log_content + "\n")

        gate = QualityGate(tmp_path)
        is_no_progress = gate.check_no_progress(
            log_path=log_path,
            log_offset=0,  # Check from beginning - will find evidence
            previous_commit_hash="abc1234",
            current_commit_hash="abc1234",  # Same commit
        )

        assert is_no_progress is False

    def test_no_progress_when_evidence_before_offset_only(self, tmp_path: Path) -> None:
        """No progress: validation evidence exists but only before the offset."""
        log_path = tmp_path / "session.jsonl"
        log_content = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Bash",
                            "input": {"command": "uv run pytest"},
                        }
                    ]
                },
            }
        )
        log_path.write_text(log_content + "\n")

        gate = QualityGate(tmp_path)
        # Set offset to after the evidence
        offset = log_path.stat().st_size

        is_no_progress = gate.check_no_progress(
            log_path=log_path,
            log_offset=offset,  # Offset past the evidence
            previous_commit_hash="abc1234",
            current_commit_hash="abc1234",
        )

        assert is_no_progress is True

    def test_progress_when_no_previous_commit(self, tmp_path: Path) -> None:
        """Progress detected: first attempt (no previous commit to compare)."""
        log_path = tmp_path / "session.jsonl"
        log_path.write_text("")

        gate = QualityGate(tmp_path)
        is_no_progress = gate.check_no_progress(
            log_path=log_path,
            log_offset=0,
            previous_commit_hash=None,  # No previous commit
            current_commit_hash="abc1234",
        )

        # First attempt with a commit is always progress
        assert is_no_progress is False

    def test_no_progress_with_none_commits_and_no_evidence(
        self, tmp_path: Path
    ) -> None:
        """No progress: both commits None (no commit made) and no new evidence."""
        log_path = tmp_path / "session.jsonl"
        log_path.write_text("")

        gate = QualityGate(tmp_path)
        is_no_progress = gate.check_no_progress(
            log_path=log_path,
            log_offset=0,
            previous_commit_hash=None,
            current_commit_hash=None,  # Still no commit
        )

        assert is_no_progress is True

    def test_handles_missing_log_file(self, tmp_path: Path) -> None:
        """Should handle missing log file (no evidence = no progress)."""
        gate = QualityGate(tmp_path)
        nonexistent = tmp_path / "nonexistent.jsonl"

        is_no_progress = gate.check_no_progress(
            log_path=nonexistent,
            log_offset=0,
            previous_commit_hash="abc1234",
            current_commit_hash="abc1234",
        )

        assert is_no_progress is True


class TestGateResultNoProgress:
    """Test GateResult includes no-progress indicator."""

    def test_gate_result_has_no_progress_field(self) -> None:
        """GateResult should have a no_progress field."""
        from src.quality_gate import GateResult

        result = GateResult(passed=False, failure_reasons=["test"])
        # Check the field exists and defaults appropriately
        assert hasattr(result, "no_progress")

    def test_gate_result_no_progress_default_false(self) -> None:
        """GateResult.no_progress should default to False."""
        from src.quality_gate import GateResult

        result = GateResult(passed=True)
        assert result.no_progress is False


class TestHasMinimumValidation:
    """Test has_minimum_validation() requires full validation suite."""

    def test_fails_when_only_pytest_ran(self) -> None:
        """Should fail when only pytest ran (missing uv sync, ruff, ty check)."""
        evidence = ValidationEvidence(pytest_ran=True)
        assert evidence.has_minimum_validation() is False

    def test_fails_when_only_ruff_ran(self) -> None:
        """Should fail when only ruff check/format ran (missing uv sync, pytest, ty check)."""
        evidence = ValidationEvidence(ruff_check_ran=True, ruff_format_ran=True)
        assert evidence.has_minimum_validation() is False

    def test_fails_when_missing_uv_sync(self) -> None:
        """Should fail when uv sync is missing."""
        evidence = ValidationEvidence(
            uv_sync_ran=False,
            pytest_ran=True,
            ruff_check_ran=True,
            ruff_format_ran=True,
            ty_check_ran=True,
        )
        assert evidence.has_minimum_validation() is False

    def test_fails_when_missing_ty_check(self) -> None:
        """Should fail when ty check is missing."""
        evidence = ValidationEvidence(
            uv_sync_ran=True,
            pytest_ran=True,
            ruff_check_ran=True,
            ruff_format_ran=True,
            ty_check_ran=False,
        )
        assert evidence.has_minimum_validation() is False

    def test_fails_when_missing_pytest(self) -> None:
        """Should fail when pytest is missing."""
        evidence = ValidationEvidence(
            uv_sync_ran=True,
            pytest_ran=False,
            ruff_check_ran=True,
            ruff_format_ran=True,
            ty_check_ran=True,
        )
        assert evidence.has_minimum_validation() is False

    def test_fails_when_missing_ruff_check(self) -> None:
        """Should fail when ruff check is missing."""
        evidence = ValidationEvidence(
            uv_sync_ran=True,
            pytest_ran=True,
            ruff_check_ran=False,
            ruff_format_ran=True,
            ty_check_ran=True,
        )
        assert evidence.has_minimum_validation() is False

    def test_fails_when_missing_ruff_format(self) -> None:
        """Should fail when ruff format is missing."""
        evidence = ValidationEvidence(
            uv_sync_ran=True,
            pytest_ran=True,
            ruff_check_ran=True,
            ruff_format_ran=False,
            ty_check_ran=True,
        )
        assert evidence.has_minimum_validation() is False

    def test_passes_when_all_commands_ran(self) -> None:
        """Should pass when all required commands ran."""
        evidence = ValidationEvidence(
            uv_sync_ran=True,
            pytest_ran=True,
            ruff_check_ran=True,
            ruff_format_ran=True,
            ty_check_ran=True,
        )
        assert evidence.has_minimum_validation() is True


class TestMissingCommands:
    """Test missing_commands() includes uv sync and ty check."""

    def test_includes_uv_sync_when_missing(self) -> None:
        """Should include 'uv sync' when it didn't run."""
        evidence = ValidationEvidence(uv_sync_ran=False)
        missing = evidence.missing_commands()
        assert "uv sync" in missing

    def test_includes_ty_check_when_missing(self) -> None:
        """Should include 'ty check' when it didn't run."""
        evidence = ValidationEvidence(ty_check_ran=False)
        missing = evidence.missing_commands()
        assert "ty check" in missing

    def test_includes_all_missing_commands(self) -> None:
        """Should list all missing commands."""
        evidence = ValidationEvidence()  # All default to False
        missing = evidence.missing_commands()
        assert "uv sync" in missing
        assert "pytest" in missing
        assert "ruff check" in missing
        assert "ruff format" in missing
        assert "ty check" in missing

    def test_excludes_commands_that_ran(self) -> None:
        """Should not list commands that ran."""
        evidence = ValidationEvidence(
            uv_sync_ran=True,
            pytest_ran=True,
            ruff_check_ran=True,
            ruff_format_ran=True,
            ty_check_ran=True,
        )
        missing = evidence.missing_commands()
        assert len(missing) == 0


class TestCommitBaselineCheck:
    """Test check_commit_exists with baseline timestamp to reject stale commits."""

    def test_rejects_commit_before_baseline(self, tmp_path: Path) -> None:
        """Should reject commits created before the baseline timestamp."""
        from src.quality_gate import QualityGate

        gate = QualityGate(tmp_path)

        # Mock git log returning a commit with a timestamp before baseline
        # The commit exists but is older than the run started
        with patch("subprocess.run") as mock_run:
            # Return commit with timestamp 1000 seconds before baseline
            mock_run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="abc1234 1703500000 bd-issue-123: Old fix\n",
                stderr="",
            )
            # Baseline is at timestamp 1703501000 (1000s after the commit)
            result = gate.check_commit_exists(
                "issue-123", baseline_timestamp=1703501000
            )

        assert result.exists is False

    def test_accepts_commit_after_baseline(self, tmp_path: Path) -> None:
        """Should accept commits created after the baseline timestamp."""
        from src.quality_gate import QualityGate

        gate = QualityGate(tmp_path)

        with patch("subprocess.run") as mock_run:
            # Return commit with timestamp 1000 seconds after baseline
            mock_run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="abc1234 1703502000 bd-issue-123: New fix\n",
                stderr="",
            )
            # Baseline is at timestamp 1703501000 (commit is newer)
            result = gate.check_commit_exists(
                "issue-123", baseline_timestamp=1703501000
            )

        assert result.exists is True
        assert result.commit_hash == "abc1234"

    def test_accepts_any_commit_without_baseline(self, tmp_path: Path) -> None:
        """Should accept any matching commit when no baseline is provided (backward compat)."""
        from src.quality_gate import QualityGate

        gate = QualityGate(tmp_path)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="abc1234 bd-issue-123: Old fix\n",
                stderr="",
            )
            result = gate.check_commit_exists("issue-123")  # No baseline

        assert result.exists is True
        assert result.commit_hash == "abc1234"

    def test_gate_check_uses_baseline(self, tmp_path: Path) -> None:
        """Gate check method should use baseline to reject stale commits."""
        from src.quality_gate import QualityGate

        gate = QualityGate(tmp_path)

        # Create log with all validation commands
        log_path = tmp_path / "session.jsonl"
        commands = [
            "uv sync",
            "uv run pytest",
            "uvx ruff check .",
            "uvx ruff format .",
            "uvx ty check",
        ]
        lines = []
        for cmd in commands:
            lines.append(
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "name": "Bash",
                                    "input": {"command": cmd},
                                }
                            ]
                        },
                    }
                )
            )
        log_path.write_text("\n".join(lines) + "\n")

        with patch("subprocess.run") as mock_run:
            # Return a commit that is older than the baseline
            mock_run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="abc1234 1703500000 bd-issue-123: Old fix\n",
                stderr="",
            )
            # Baseline makes the commit stale
            result = gate.check("issue-123", log_path, baseline_timestamp=1703501000)

        assert result.passed is False
        assert any(
            "baseline" in r.lower() or "stale" in r.lower()
            for r in result.failure_reasons
        )

    def test_gate_check_passes_with_new_commit(self, tmp_path: Path) -> None:
        """Gate check should pass when commit is after baseline."""
        from src.quality_gate import QualityGate

        gate = QualityGate(tmp_path)

        # Create log with all validation commands
        log_path = tmp_path / "session.jsonl"
        commands = [
            "uv sync",
            "uv run pytest",
            "uvx ruff check .",
            "uvx ruff format .",
            "uvx ty check",
        ]
        lines = []
        for cmd in commands:
            lines.append(
                json.dumps(
                    {
                        "type": "assistant",
                        "message": {
                            "content": [
                                {
                                    "type": "tool_use",
                                    "name": "Bash",
                                    "input": {"command": cmd},
                                }
                            ]
                        },
                    }
                )
            )
        log_path.write_text("\n".join(lines) + "\n")

        with patch("subprocess.run") as mock_run:
            # Return a commit that is newer than the baseline
            mock_run.return_value = subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="abc1234 1703502000 bd-issue-123: New fix\n",
                stderr="",
            )
            # Baseline allows the newer commit
            result = gate.check("issue-123", log_path, baseline_timestamp=1703501000)

        assert result.passed is True

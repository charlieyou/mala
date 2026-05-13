"""Tests for Cerberus review adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.infra.clients.cerberus_cli import CerberusCLI
from src.infra.clients.cerberus_review import (
    DefaultReviewer,
    _to_relative_path,
    format_review_issues,
)
from src.infra.clients.cerberus_output_parser import (
    ReviewIssue,
    map_exit_code_to_result,
)


def _gate_state_json(
    *,
    project_key: str = "project-1",
    run_key: str = "mala-test-session",
    verdict: str = "pass",
) -> str:
    return (
        "{"
        '"schema_version":"2",'
        f'"run_key":"{run_key}",'
        '"host":"generic",'
        f'"project_key":"{project_key}",'
        '"session_id":"test-session",'
        '"transcript_path":"/tmp/transcript.jsonl",'
        '"status":"resolved",'
        f'"verdict":"{verdict}",'
        '"resolution_reason":"complete",'
        '"current_iteration":1,'
        '"max_rounds":1,'
        '"debate":false,'
        '"roster_id":"default",'
        '"started_at":"2026-05-11T00:00:00Z",'
        '"ended_at":"2026-05-11T00:01:00Z"'
        "}"
    )


def _write_empty_cerberus_output(
    repo_path: Path, *, project_key: str = "project-1", run_key: str
) -> None:
    output_dir = (
        repo_path
        / ".mala"
        / "cerberus"
        / project_key
        / run_key
        / "iterations"
        / "1"
        / "round-1"
        / "reviewers"
        / "codex#1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "output.json").write_text('{"findings":[]}', encoding="utf-8")


class TestCerberusOutputParser:
    """Tests for Cerberus gate-state result mapping."""

    def _write_output(
        self,
        tmp_path: Path,
        *,
        findings_json: str,
        run_key: str = "mala-test-session",
        project_key: str = "project-1",
    ) -> None:
        output_dir = (
            tmp_path
            / ".mala"
            / "cerberus"
            / project_key
            / run_key
            / "iterations"
            / "1"
            / "round-1"
            / "reviewers"
            / "codex#1"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "output.json").write_text(findings_json, encoding="utf-8")

    def test_needs_work_passes_with_findings_for_tracking(
        self, tmp_path: Path
    ) -> None:
        """needs_work closes the source while preserving findings as issues."""
        run_key = "mala-test-session"
        project_key = "project-1"
        self._write_output(
            tmp_path,
            findings_json="""
            {
              "findings": [
                {
                  "title": "Follow-up cleanup",
                  "body": "Non-blocking improvement",
                  "priority": 2,
                  "file_path": "src/foo.py",
                  "line_start": 10,
                  "line_end": 12
                }
              ]
            }
            """,
        )

        result = map_exit_code_to_result(
            0,
            _gate_state_json(verdict="needs_work"),
            "",
            state_root=tmp_path / ".mala" / "cerberus",
            project_key=project_key,
            run_key=run_key,
        )

        assert result.passed is True
        assert result.parse_error is None
        assert len(result.issues) == 1
        assert result.issues[0].title == "Follow-up cleanup"

    def test_needs_work_with_blocking_finding_does_not_pass(
        self, tmp_path: Path
    ) -> None:
        """needs_work cannot silently close when artifacts contain P0/P1 findings."""
        run_key = "mala-test-session"
        project_key = "project-1"
        self._write_output(
            tmp_path,
            findings_json="""
            {
              "findings": [
                {
                  "title": "Blocking regression",
                  "body": "Must be fixed before closure",
                  "priority": 1,
                  "file_path": "src/foo.py",
                  "line_start": 10,
                  "line_end": 12
                }
              ]
            }
            """,
        )

        result = map_exit_code_to_result(
            0,
            _gate_state_json(verdict="needs_work"),
            "",
            state_root=tmp_path / ".mala" / "cerberus",
            project_key=project_key,
            run_key=run_key,
        )

        assert result.passed is False
        assert result.parse_error is None
        assert len(result.issues) == 1
        assert result.issues[0].title == "Blocking regression"


class TestFormatReviewIssues:
    """Tests for formatting review issues for follow-up prompts."""

    def test_formats_empty_issues(self) -> None:
        result = format_review_issues([])
        assert result == "No specific issues found."

    def test_formats_single_issue(self) -> None:
        issues = [
            ReviewIssue(
                file="src/main.py",
                line_start=10,
                line_end=10,
                priority=1,
                title="[P1] Missing import",
                body="The os module is not imported",
                reviewer="codex",
            )
        ]
        result = format_review_issues(issues)
        assert "File: src/main.py" in result
        assert "L10:" in result
        assert "[codex]" in result
        assert "[P1] Missing import" in result
        assert "The os module is not imported" in result

    def test_formats_multiline_issue(self) -> None:
        issues = [
            ReviewIssue(
                file="src/utils.py",
                line_start=5,
                line_end=15,
                priority=2,
                title="Complex function",
                body="Consider refactoring",
                reviewer="gemini",
            )
        ]
        result = format_review_issues(issues)
        assert "L5-15:" in result
        assert "[gemini]" in result

    def test_groups_by_file(self) -> None:
        issues = [
            ReviewIssue(
                file="b.py",
                line_start=1,
                line_end=1,
                priority=1,
                title="Issue 1",
                body="",
                reviewer="codex",
            ),
            ReviewIssue(
                file="a.py",
                line_start=5,
                line_end=5,
                priority=1,
                title="Issue 2",
                body="",
                reviewer="codex",
            ),
            ReviewIssue(
                file="a.py",
                line_start=10,
                line_end=10,
                priority=2,
                title="Issue 3",
                body="",
                reviewer="codex",
            ),
        ]
        result = format_review_issues(issues)
        # Should be sorted by file, then by line
        lines = result.split("\n")
        file_lines = [line for line in lines if line.startswith("File:")]
        assert file_lines[0] == "File: a.py"
        assert file_lines[1] == "File: b.py"

    def test_includes_reviewer_attribution(self) -> None:
        issues = [
            ReviewIssue(
                file="test.py",
                line_start=1,
                line_end=1,
                priority=1,
                title="Issue",
                body="",
                reviewer="claude",
            )
        ]
        result = format_review_issues(issues)
        assert "[claude]" in result

    def test_handles_empty_reviewer(self) -> None:
        issues = [
            ReviewIssue(
                file="test.py",
                line_start=1,
                line_end=1,
                priority=1,
                title="Issue",
                body="",
                reviewer="",
            )
        ]
        result = format_review_issues(issues)
        # Should not have empty brackets
        assert "[]" not in result
        assert "Issue" in result


class TestToRelativePath:
    """Tests for path relativization helper."""

    def test_relative_path_unchanged(self) -> None:
        """Relative paths are returned unchanged."""
        assert _to_relative_path("src/main.py") == "src/main.py"
        assert _to_relative_path("test.py") == "test.py"
        assert _to_relative_path("./foo/bar.py") == "./foo/bar.py"

    def test_absolute_path_with_base_path(self) -> None:
        """Absolute paths are relativized against base_path."""
        base = Path("/home/user/project")
        assert (
            _to_relative_path("/home/user/project/src/main.py", base) == "src/main.py"
        )
        assert _to_relative_path("/home/user/project/test.py", base) == "test.py"

    def test_absolute_path_outside_base_preserved(self) -> None:
        """Absolute paths outside base_path are preserved (not stripped to filename)."""
        base = Path("/home/user/project")
        # Path outside base_path should be preserved fully
        result = _to_relative_path("/other/path/to/file.py", base)
        assert result == "/other/path/to/file.py"

    def test_absolute_path_no_base_uses_cwd(self) -> None:
        """Without base_path, falls back to cwd."""
        # This test checks that relative paths in cwd work
        cwd = Path.cwd()
        # Create a path that is inside cwd
        test_path = str(cwd / "src" / "test.py")
        result = _to_relative_path(test_path)
        assert result == "src/test.py"

    def test_preserves_directory_context_on_failure(self) -> None:
        """When relativization fails, full path is preserved (not just filename)."""
        base = Path("/home/user/project")
        # A path that cannot be relativized should keep full directory context
        result = _to_relative_path("/completely/different/path/important/file.py", base)
        # Should NOT be stripped to just "file.py"
        assert "important" in result
        assert result == "/completely/different/path/important/file.py"


class TestExtractWaitTimeout:
    """Tests for CerberusCLI.extract_wait_timeout method."""

    def test_returns_none_for_empty_args(self) -> None:
        """Returns None when args is empty."""
        assert CerberusCLI.extract_wait_timeout(()) is None

    def test_returns_none_when_no_timeout_flag(self) -> None:
        """Returns None when --timeout is not present."""
        args = ("--json", "--session-key", "abc123")
        assert CerberusCLI.extract_wait_timeout(args) is None

    def test_extracts_timeout_with_equals_format(self) -> None:
        """Extracts timeout from --timeout=VALUE format."""
        args = ("--json", "--timeout=600", "--session-key", "abc123")
        assert CerberusCLI.extract_wait_timeout(args) == 600

    def test_extracts_timeout_with_space_format(self) -> None:
        """Extracts timeout from --timeout VALUE format."""
        args = ("--json", "--timeout", "300", "--session-key", "abc123")
        assert CerberusCLI.extract_wait_timeout(args) == 300

    def test_returns_none_for_non_numeric_equals_value(self) -> None:
        """Returns None when --timeout=VALUE has non-numeric value."""
        args = ("--timeout=abc",)
        assert CerberusCLI.extract_wait_timeout(args) is None

    def test_returns_none_for_non_numeric_space_value(self) -> None:
        """Returns None when --timeout VALUE has non-numeric value."""
        args = ("--timeout", "abc")
        assert CerberusCLI.extract_wait_timeout(args) is None

    def test_returns_none_for_timeout_at_end_without_value(self) -> None:
        """Returns None when --timeout is at end without value."""
        args = ("--json", "--timeout")
        assert CerberusCLI.extract_wait_timeout(args) is None

    def test_timeout_at_beginning_of_args(self) -> None:
        """Extracts timeout when it's the first argument."""
        args = ("--timeout", "120", "--json")
        assert CerberusCLI.extract_wait_timeout(args) == 120

    def test_timeout_at_end_of_args_with_value(self) -> None:
        """Extracts timeout when it's the last argument pair."""
        args = ("--json", "--timeout", "450")
        assert CerberusCLI.extract_wait_timeout(args) == 450


class TestNoChangesSpawnError:
    """Tests for empty-diff spawn handling."""

    async def test_spawn_no_changes_is_treated_as_pass(self) -> None:
        """Empty Cerberus commit diffs skip review instead of failing the run."""
        from unittest.mock import AsyncMock, MagicMock, patch

        reviewer = DefaultReviewer(repo_path=Path("/tmp"))

        with patch(
            "src.infra.clients.cerberus_cli.CerberusCLI.validate_binary",
            return_value=None,
        ):
            clear_result = MagicMock()
            clear_result.returncode = 0
            clear_result.timed_out = False

            spawn_result = MagicMock()
            spawn_result.returncode = 1
            spawn_result.timed_out = False
            spawn_result.stderr_tail.return_value = (
                "Error: No changes found for diff mode: --commit abc123,def456"
            )
            spawn_result.stdout_tail.return_value = ""

            with patch(
                "src.infra.clients.cerberus_review.CommandRunner"
            ) as mock_runner_class:
                mock_runner = AsyncMock()
                mock_runner.run_async.side_effect = [clear_result, spawn_result]
                mock_runner_class.return_value = mock_runner

                result = await reviewer(
                    commit_shas=["abc123", "def456"],
                    claude_session_id="test-session",
                )

            assert result.passed is True
            assert result.fatal_error is False
            assert result.parse_error is None
            assert result.issues == []

            calls = [call[0][0] for call in mock_runner.run_async.call_args_list]
            assert len(calls) == 2
            assert calls[0] == ["cerberus", "author-context", "--clear"]
            assert "spawn-code-review" in calls[1]


class TestDefaultReviewerConstructor:
    """Tests for DefaultReviewer construction."""

    def test_accepts_state_root_and_project_key(self, tmp_path: Path) -> None:
        reviewer = DefaultReviewer(
            repo_path=tmp_path,
            state_root=tmp_path / ".mala" / "cerberus",
            project_key="project-1",
        )

        assert reviewer.state_root == tmp_path / ".mala" / "cerberus"
        assert reviewer.project_key == "project-1"

    def test_rejects_bin_path(self, tmp_path: Path) -> None:
        kwargs: Any = {"repo_path": tmp_path, "bin_path": tmp_path / "bin"}
        with pytest.raises(TypeError):
            DefaultReviewer(**kwargs)


class TestCerberusV2Review:
    """Tests for v2 Cerberus review orchestration."""

    async def test_default_project_key_is_non_empty(self, tmp_path: Path) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        reviewer = DefaultReviewer(
            repo_path=tmp_path,
            env={"CERBERUS_ROOT": "/tmp/cerberus"},
        )
        _write_empty_cerberus_output(
            tmp_path,
            project_key=tmp_path.name,
            run_key="mala-test-session",
        )

        with patch(
            "src.infra.clients.cerberus_cli.CerberusCLI.validate_binary",
            return_value=None,
        ):
            clear_result = MagicMock()
            clear_result.returncode = 0
            clear_result.timed_out = False

            spawn_ok_result = MagicMock()
            spawn_ok_result.returncode = 0
            spawn_ok_result.timed_out = False

            wait_result = MagicMock()
            wait_result.returncode = 0
            wait_result.timed_out = False
            wait_result.stdout = _gate_state_json(project_key=tmp_path.name)
            wait_result.stderr = ""

            with patch(
                "src.infra.clients.cerberus_review.CommandRunner"
            ) as mock_runner_class:
                mock_runner = AsyncMock()
                mock_runner.run_async.side_effect = [
                    clear_result,
                    spawn_ok_result,
                    wait_result,
                ]
                mock_runner_class.return_value = mock_runner

                result = await reviewer(
                    commit_shas=["abc123"],
                    claude_session_id="test-session",
                )

        assert result.passed is True
        spawn_env = mock_runner.run_async.call_args_list[1].kwargs["env"]
        assert spawn_env["CERBERUS_PROJECT_KEY"] == tmp_path.name

    async def test_normal_pass_does_not_resolve_gate_and_uses_run_key_env(
        self, tmp_path: Path
    ) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        state_root = tmp_path / ".mala" / "cerberus"
        reviewer = DefaultReviewer(
            repo_path=tmp_path,
            state_root=state_root,
            project_key="project-1",
            env={"CERBERUS_ROOT": "/tmp/cerberus"},
        )
        _write_empty_cerberus_output(tmp_path, run_key="mala-test-session")

        with (
            patch(
                "src.infra.clients.cerberus_cli.CerberusCLI.validate_binary",
                return_value=None,
            ),
            patch(
                "src.infra.clients.cerberus_cli.CerberusCLI.resolve_gate",
                new_callable=AsyncMock,
            ) as mock_resolve,
        ):
            clear_result = MagicMock()
            clear_result.returncode = 0
            clear_result.timed_out = False

            spawn_ok_result = MagicMock()
            spawn_ok_result.returncode = 0
            spawn_ok_result.timed_out = False

            wait_result = MagicMock()
            wait_result.returncode = 0
            wait_result.timed_out = False
            wait_result.stdout = _gate_state_json()
            wait_result.stderr = ""

            with patch(
                "src.infra.clients.cerberus_review.CommandRunner"
            ) as mock_runner_class:
                mock_runner = AsyncMock()
                mock_runner.run_async.side_effect = [
                    clear_result,
                    spawn_ok_result,
                    wait_result,
                ]
                mock_runner_class.return_value = mock_runner

                result = await reviewer(
                    commit_shas=["abc123"],
                    claude_session_id="test-session",
                )

            assert result.passed is True
            assert result.fatal_error is False
            assert result.parse_error is None
            mock_resolve.assert_not_called()

            calls = [call[0][0] for call in mock_runner.run_async.call_args_list]
            assert calls[0] == ["cerberus", "author-context", "--clear"]
            assert calls[1][:2] == ["cerberus", "spawn-code-review"]
            assert "--max-rounds" not in calls[1]
            assert calls[2] == [
                "cerberus",
                "wait",
                "--json",
                "--finalize",
                "--session-key",
                "mala-test-session",
                "--timeout",
                "600",
            ]

            spawn_env = mock_runner.run_async.call_args_list[1].kwargs["env"]
            assert spawn_env["CERBERUS_HOST"] == "generic"
            assert spawn_env["CERBERUS_RUN_KEY"] == "mala-test-session"
            assert spawn_env["CERBERUS_STATE_ROOT"] == str(state_root)
            assert spawn_env["CERBERUS_PROJECT_KEY"] == "project-1"
            assert spawn_env["CERBERUS_ROOT"] == "/tmp/cerberus"

    async def test_author_context_is_written_before_spawn(
        self, tmp_path: Path
    ) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        state_root = tmp_path / ".mala" / "cerberus"
        reviewer = DefaultReviewer(
            repo_path=tmp_path,
            state_root=state_root,
            project_key="project-1",
            env={"CERBERUS_ROOT": "/tmp/cerberus"},
        )
        _write_empty_cerberus_output(tmp_path, run_key="mala-test-session")

        with patch(
            "src.infra.clients.cerberus_cli.CerberusCLI.validate_binary",
            return_value=None,
        ):
            author_context_result = MagicMock()
            author_context_result.returncode = 0
            author_context_result.timed_out = False

            spawn_ok_result = MagicMock()
            spawn_ok_result.returncode = 0
            spawn_ok_result.timed_out = False

            wait_result = MagicMock()
            wait_result.returncode = 0
            wait_result.timed_out = False
            wait_result.stdout = _gate_state_json()
            wait_result.stderr = ""

            with patch(
                "src.infra.clients.cerberus_review.CommandRunner"
            ) as mock_runner_class:
                mock_runner = AsyncMock()
                mock_runner.run_async.side_effect = [
                    author_context_result,
                    spawn_ok_result,
                    wait_result,
                ]
                mock_runner_class.return_value = mock_runner

                result = await reviewer(
                    commit_shas=["abc123"],
                    claude_session_id="test-session",
                    author_context="False positive: covered by test_x",
                )

        assert result.passed is True
        calls = [call[0][0] for call in mock_runner.run_async.call_args_list]
        assert calls[0] == [
            "cerberus",
            "author-context",
            "--",
            "False positive: covered by test_x",
        ]
        assert calls[1][:2] == ["cerberus", "spawn-code-review"]
        assert calls[2][:2] == ["cerberus", "wait"]
        author_context_env = mock_runner.run_async.call_args_list[0].kwargs["env"]
        assert author_context_env["CERBERUS_RUN_KEY"] == "mala-test-session"
        assert author_context_env["CERBERUS_STATE_ROOT"] == str(state_root)

    async def test_author_context_set_failure_uses_context_file_fallback(
        self, tmp_path: Path
    ) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        reviewer = DefaultReviewer(
            repo_path=tmp_path,
            env={"CERBERUS_ROOT": "/tmp/cerberus"},
        )
        context_file = tmp_path / "review-context.txt"
        context_file.write_text("context", encoding="utf-8")
        _write_empty_cerberus_output(
            tmp_path,
            project_key=tmp_path.name,
            run_key="mala-test-session",
        )

        with patch(
            "src.infra.clients.cerberus_cli.CerberusCLI.validate_binary",
            return_value=None,
        ):
            author_context_result = MagicMock()
            author_context_result.returncode = 1
            author_context_result.stderr_tail.return_value = "state unavailable"
            author_context_result.stdout_tail.return_value = ""

            spawn_ok_result = MagicMock()
            spawn_ok_result.returncode = 0
            spawn_ok_result.timed_out = False

            wait_result = MagicMock()
            wait_result.returncode = 0
            wait_result.timed_out = False
            wait_result.stdout = _gate_state_json(project_key=tmp_path.name)
            wait_result.stderr = ""

            with patch(
                "src.infra.clients.cerberus_review.CommandRunner"
            ) as mock_runner_class:
                mock_runner = AsyncMock()
                mock_runner.run_async.side_effect = [
                    author_context_result,
                    spawn_ok_result,
                    wait_result,
                ]
                mock_runner_class.return_value = mock_runner

                result = await reviewer(
                    context_file=context_file,
                    commit_shas=["abc123"],
                    claude_session_id="test-session",
                    author_context="context",
                )

        assert result.passed is True
        calls = [call[0][0] for call in mock_runner.run_async.call_args_list]
        assert calls[0] == ["cerberus", "author-context", "--", "context"]
        assert calls[1][:2] == ["cerberus", "spawn-code-review"]
        assert "--context-file" in calls[1]
        assert str(context_file) in calls[1]
        assert calls[2][:2] == ["cerberus", "wait"]

    async def test_author_context_set_failure_without_fallback_blocks_review(
        self, tmp_path: Path
    ) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        reviewer = DefaultReviewer(
            repo_path=tmp_path,
            env={"CERBERUS_ROOT": "/tmp/cerberus"},
        )

        with patch(
            "src.infra.clients.cerberus_cli.CerberusCLI.validate_binary",
            return_value=None,
        ):
            author_context_result = MagicMock()
            author_context_result.returncode = 1
            author_context_result.stderr_tail.return_value = "state unavailable"
            author_context_result.stdout_tail.return_value = ""

            with patch(
                "src.infra.clients.cerberus_review.CommandRunner"
            ) as mock_runner_class:
                mock_runner = AsyncMock()
                mock_runner.run_async.return_value = author_context_result
                mock_runner_class.return_value = mock_runner

                result = await reviewer(
                    commit_shas=["abc123"],
                    claude_session_id="test-session",
                    author_context="context",
                )

        assert result.passed is False
        assert result.parse_error == (
            "author-context set failed and no context_file fallback is available: "
            "state unavailable"
        )
        assert [call[0][0] for call in mock_runner.run_async.call_args_list] == [
            ["cerberus", "author-context", "--", "context"]
        ]

    async def test_author_context_clear_failure_blocks_review(
        self, tmp_path: Path
    ) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        reviewer = DefaultReviewer(
            repo_path=tmp_path,
            env={"CERBERUS_ROOT": "/tmp/cerberus"},
        )

        with patch(
            "src.infra.clients.cerberus_cli.CerberusCLI.validate_binary",
            return_value=None,
        ):
            clear_result = MagicMock()
            clear_result.returncode = 1
            clear_result.stderr_tail.return_value = "state unavailable"
            clear_result.stdout_tail.return_value = ""

            with patch(
                "src.infra.clients.cerberus_review.CommandRunner"
            ) as mock_runner_class:
                mock_runner = AsyncMock()
                mock_runner.run_async.return_value = clear_result
                mock_runner_class.return_value = mock_runner

                result = await reviewer(
                    commit_shas=["abc123"],
                    claude_session_id="test-session",
                    author_context=None,
                )

        assert result.passed is False
        assert result.parse_error == "author-context clear failed: state unavailable"
        assert [call[0][0] for call in mock_runner.run_async.call_args_list] == [
            ["cerberus", "author-context", "--clear"]
        ]

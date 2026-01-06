"""Unit tests for mala logs list command.

Tests for:
- Sorting logic (started_at desc, run_id asc tiebreaker)
- Status count aggregation
- Repo scoping filters
- Null value handling in table/JSON output
- Required key validation
- 20-run limit
"""

from __future__ import annotations

import json
from io import StringIO
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from src.cli.logs import (
    _collect_runs,
    _count_issue_statuses,
    _format_null,
    _parse_run_file,
    _sort_runs,
    _validate_run_metadata,
)

pytestmark = pytest.mark.unit


class TestSortRuns:
    """Tests for _sort_runs function."""

    def test_sort_runs_by_started_at_desc(self) -> None:
        """Verify runs are sorted by started_at descending."""
        runs = [
            {"run_id": "aaa", "started_at": "2024-01-01T10:00:00+00:00"},
            {"run_id": "bbb", "started_at": "2024-01-03T10:00:00+00:00"},
            {"run_id": "ccc", "started_at": "2024-01-02T10:00:00+00:00"},
        ]
        sorted_runs = _sort_runs(runs)
        assert [r["run_id"] for r in sorted_runs] == ["bbb", "ccc", "aaa"]

    def test_sort_runs_tiebreaker_by_run_id_asc(self) -> None:
        """Verify deterministic ordering when started_at is equal."""
        runs = [
            {"run_id": "zzz", "started_at": "2024-01-01T10:00:00+00:00"},
            {"run_id": "aaa", "started_at": "2024-01-01T10:00:00+00:00"},
            {"run_id": "mmm", "started_at": "2024-01-01T10:00:00+00:00"},
        ]
        sorted_runs = _sort_runs(runs)
        assert [r["run_id"] for r in sorted_runs] == ["aaa", "mmm", "zzz"]

    def test_sort_runs_combined(self) -> None:
        """Verify combined sorting: started_at desc, then run_id asc."""
        runs = [
            {"run_id": "zzz", "started_at": "2024-01-01T10:00:00+00:00"},
            {"run_id": "aaa", "started_at": "2024-01-02T10:00:00+00:00"},
            {"run_id": "mmm", "started_at": "2024-01-01T10:00:00+00:00"},
            {"run_id": "bbb", "started_at": "2024-01-02T10:00:00+00:00"},
        ]
        sorted_runs = _sort_runs(runs)
        # 2024-01-02 runs first (aaa, bbb), then 2024-01-01 runs (mmm, zzz)
        assert [r["run_id"] for r in sorted_runs] == ["aaa", "bbb", "mmm", "zzz"]


class TestCountStatusAggregation:
    """Tests for _count_issue_statuses function."""

    def test_count_status_aggregation(self) -> None:
        """Verify success/fail/timeout counts are aggregated correctly."""
        issues = {
            "issue-1": {"status": "success"},
            "issue-2": {"status": "success"},
            "issue-3": {"status": "failed"},
            "issue-4": {"status": "timeout"},
            "issue-5": {"status": "success"},
        }
        total, success, failed, timeout = _count_issue_statuses(issues)
        assert total == 5
        assert success == 3
        assert failed == 1
        assert timeout == 1

    def test_count_empty_issues(self) -> None:
        """Verify empty issues dict returns all zeros."""
        total, success, failed, timeout = _count_issue_statuses({})
        assert (total, success, failed, timeout) == (0, 0, 0, 0)

    def test_count_unknown_status(self) -> None:
        """Verify unknown status is not counted in any category."""
        issues = {
            "issue-1": {"status": "unknown"},
            "issue-2": {"status": "pending"},
        }
        total, success, failed, timeout = _count_issue_statuses(issues)
        assert total == 2
        assert success == 0
        assert failed == 0
        assert timeout == 0


class TestValidateRunMetadataKeys:
    """Tests for _validate_run_metadata function."""

    def test_validate_run_metadata_keys(self) -> None:
        """Verify required key checking works correctly."""
        # Valid: has all required keys
        valid_data = {
            "run_id": "abc123",
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": {},
        }
        assert _validate_run_metadata(valid_data) is True

        # Invalid: missing run_id
        missing_run_id = {
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": {},
        }
        assert _validate_run_metadata(missing_run_id) is False

        # Invalid: missing started_at
        missing_started_at = {
            "run_id": "abc123",
            "issues": {},
        }
        assert _validate_run_metadata(missing_started_at) is False

        # Invalid: missing issues
        missing_issues = {
            "run_id": "abc123",
            "started_at": "2024-01-01T10:00:00+00:00",
        }
        assert _validate_run_metadata(missing_issues) is False

    def test_validate_with_extra_keys(self) -> None:
        """Verify extra keys don't affect validation."""
        data = {
            "run_id": "abc123",
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": {},
            "extra_key": "ignored",
            "another": 123,
        }
        assert _validate_run_metadata(data) is True

    def test_validate_rejects_non_dict(self) -> None:
        """Verify non-dict JSON is rejected."""
        assert _validate_run_metadata([]) is False
        assert _validate_run_metadata("string") is False
        assert _validate_run_metadata(123) is False
        assert _validate_run_metadata(None) is False

    def test_validate_rejects_null_started_at(self) -> None:
        """Verify null started_at is rejected."""
        data = {
            "run_id": "abc123",
            "started_at": None,
            "issues": {},
        }
        assert _validate_run_metadata(data) is False

    def test_validate_accepts_null_issues(self) -> None:
        """Verify null issues is accepted (normalized to {})."""
        data = {
            "run_id": "abc123",
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": None,
        }
        assert _validate_run_metadata(data) is True

    def test_validate_rejects_non_dict_issues(self) -> None:
        """Verify non-dict issues is rejected."""
        data = {
            "run_id": "abc123",
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": ["list", "not", "dict"],
        }
        assert _validate_run_metadata(data) is False

    def test_validate_rejects_null_run_id(self) -> None:
        """Verify null run_id is rejected."""
        data = {
            "run_id": None,
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": {},
        }
        assert _validate_run_metadata(data) is False

    def test_validate_rejects_non_string_run_id(self) -> None:
        """Verify non-string run_id is rejected."""
        data = {
            "run_id": 12345,
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": {},
        }
        assert _validate_run_metadata(data) is False


class TestNullValueHandling:
    """Tests for null value handling in output."""

    def test_null_values_in_table_output(self) -> None:
        """Verify '-' is shown for null values in table output."""
        assert _format_null(None) == "-"
        assert _format_null("value") == "value"
        assert _format_null(123) == "123"

    def test_null_values_in_json_output(self) -> None:
        """Verify null literal is preserved in JSON output."""
        # When repo_path is None, it should serialize as null in JSON
        data: dict[str, Any] = {"repo_path": None}
        json_str = json.dumps(data)
        assert '"repo_path": null' in json_str

        # And parse back as None
        parsed = json.loads(json_str)
        assert parsed["repo_path"] is None


class TestParseRunFile:
    """Tests for _parse_run_file function."""

    def test_parse_valid_file(self, tmp_path: Path) -> None:
        """Verify valid JSON file is parsed correctly."""
        run_file = tmp_path / "run.json"
        data = {
            "run_id": "abc123",
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": {"issue-1": {"status": "success"}},
        }
        run_file.write_text(json.dumps(data))

        result = _parse_run_file(run_file)
        assert result is not None
        assert result["run_id"] == "abc123"

    def test_parse_corrupt_file(self, tmp_path: Path) -> None:
        """Verify corrupt JSON file returns None with warning."""
        run_file = tmp_path / "corrupt.json"
        run_file.write_text("not valid json {{{")

        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            result = _parse_run_file(run_file)
            assert result is None
            assert "Warning: skipping corrupt file" in mock_stderr.getvalue()

    def test_parse_missing_required_keys(self, tmp_path: Path) -> None:
        """Verify file missing required keys returns None (no warning)."""
        run_file = tmp_path / "incomplete.json"
        run_file.write_text('{"run_id": "abc123"}')  # missing started_at, issues

        result = _parse_run_file(run_file)
        assert result is None


class TestCollectRuns:
    """Tests for _collect_runs function."""

    def test_collect_all_valid_runs(self, tmp_path: Path) -> None:
        """Verify all valid runs are collected (limit applied by caller)."""
        # Create 25 valid run files
        files = []
        for i in range(25):
            run_file = tmp_path / f"run_{i:02d}.json"
            data = {
                "run_id": f"run-{i:02d}",
                "started_at": f"2024-01-{(i % 28) + 1:02d}T10:00:00+00:00",
                "issues": {},
            }
            run_file.write_text(json.dumps(data))
            files.append(run_file)

        # _collect_runs collects all valid runs; caller sorts and limits
        runs = _collect_runs(files)
        assert len(runs) == 25  # All valid runs collected

        # Verify limit works when applied to sorted runs
        from src.cli.logs import _sort_runs

        sorted_runs = _sort_runs(runs)[:20]
        assert len(sorted_runs) == 20

    def test_collect_skips_invalid_files(self, tmp_path: Path) -> None:
        """Verify invalid files are skipped, valid ones collected."""
        # Create mix of valid and invalid files
        valid_file = tmp_path / "valid.json"
        valid_file.write_text(
            json.dumps(
                {
                    "run_id": "valid-run",
                    "started_at": "2024-01-01T10:00:00+00:00",
                    "issues": {},
                }
            )
        )

        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text('{"run_id": "no-other-keys"}')

        corrupt_file = tmp_path / "corrupt.json"
        corrupt_file.write_text("not json")

        with patch("sys.stderr", new_callable=StringIO):
            runs = _collect_runs([valid_file, invalid_file, corrupt_file])

        assert len(runs) == 1
        assert runs[0]["run_id"] == "valid-run"

    def test_collect_adds_metadata_path(self, tmp_path: Path) -> None:
        """Verify metadata_path is added to each run."""
        run_file = tmp_path / "run.json"
        run_file.write_text(
            json.dumps(
                {
                    "run_id": "abc123",
                    "started_at": "2024-01-01T10:00:00+00:00",
                    "issues": {},
                }
            )
        )

        runs = _collect_runs([run_file])
        assert len(runs) == 1
        assert runs[0]["metadata_path"] == str(run_file)


class TestRepoScoping:
    """Tests for repo scoping behavior."""

    def test_repo_scoping_filters_by_cwd(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify CWD-based filtering via get_repo_runs_dir."""
        # Set up directory structure
        runs_dir = tmp_path / "runs"
        repo1_dir = runs_dir / "-home-user-repo1"
        repo2_dir = runs_dir / "-home-user-repo2"
        repo1_dir.mkdir(parents=True)
        repo2_dir.mkdir(parents=True)

        # Create run files in each repo dir
        repo1_run = repo1_dir / "run1.json"
        repo1_run.write_text(
            json.dumps(
                {
                    "run_id": "repo1-run",
                    "started_at": "2024-01-01T10:00:00+00:00",
                    "issues": {},
                }
            )
        )

        repo2_run = repo2_dir / "run2.json"
        repo2_run.write_text(
            json.dumps(
                {
                    "run_id": "repo2-run",
                    "started_at": "2024-01-01T10:00:00+00:00",
                    "issues": {},
                }
            )
        )

        # Mock get_repo_runs_dir to return repo1_dir
        monkeypatch.setattr(
            "src.cli.logs.get_repo_runs_dir",
            lambda _: repo1_dir,
        )

        from src.cli.logs import _discover_run_files

        files = _discover_run_files(all_runs=False)

        # Should only find repo1's run file
        assert len(files) == 1
        assert "repo1-run" in files[0].read_text()

    def test_all_runs_finds_all_repos(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify --all finds runs from all repos."""
        # Set up directory structure
        runs_dir = tmp_path / "runs"
        repo1_dir = runs_dir / "-home-user-repo1"
        repo2_dir = runs_dir / "-home-user-repo2"
        repo1_dir.mkdir(parents=True)
        repo2_dir.mkdir(parents=True)

        # Create run files in each repo dir
        repo1_run = repo1_dir / "run1.json"
        repo1_run.write_text(
            json.dumps(
                {
                    "run_id": "repo1-run",
                    "started_at": "2024-01-01T10:00:00+00:00",
                    "issues": {},
                }
            )
        )

        repo2_run = repo2_dir / "run2.json"
        repo2_run.write_text(
            json.dumps(
                {
                    "run_id": "repo2-run",
                    "started_at": "2024-01-02T10:00:00+00:00",
                    "issues": {},
                }
            )
        )

        # Mock get_runs_dir to return our test runs_dir
        monkeypatch.setattr(
            "src.cli.logs.get_runs_dir",
            lambda: runs_dir,
        )

        from src.cli.logs import _discover_run_files

        files = _discover_run_files(all_runs=True)

        # Should find both repo run files
        assert len(files) == 2
        run_ids = {json.loads(f.read_text())["run_id"] for f in files}
        assert run_ids == {"repo1-run", "repo2-run"}

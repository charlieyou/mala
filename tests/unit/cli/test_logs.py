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
    _count_issue_statuses,
    _discover_run_files_all_repos,
    _extract_run_id_prefix_from_filename,
    _format_null,
    _parse_run_file_cli,
    _sort_runs,
)
from src.infra.io.log_output.run_metadata import (
    _extract_timestamp_prefix,
    _validate_run_data,
    discover_run_files,
    find_sessions_for_issue,
    load_runs,
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
    """Tests for _validate_run_data function (now in run_metadata.py)."""

    def test_validate_run_metadata_keys(self) -> None:
        """Verify required key checking works correctly."""
        # Valid: has all required keys
        valid_data = {
            "run_id": "abc123",
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": {},
        }
        assert _validate_run_data(valid_data) is True

        # Invalid: missing run_id
        missing_run_id = {
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": {},
        }
        assert _validate_run_data(missing_run_id) is False

        # Invalid: missing started_at
        missing_started_at = {
            "run_id": "abc123",
            "issues": {},
        }
        assert _validate_run_data(missing_started_at) is False

        # Invalid: missing issues
        missing_issues = {
            "run_id": "abc123",
            "started_at": "2024-01-01T10:00:00+00:00",
        }
        assert _validate_run_data(missing_issues) is False

    def test_validate_with_extra_keys(self) -> None:
        """Verify extra keys don't affect validation."""
        data = {
            "run_id": "abc123",
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": {},
            "extra_key": "ignored",
            "another": 123,
        }
        assert _validate_run_data(data) is True

    def test_validate_rejects_non_dict(self) -> None:
        """Verify non-dict JSON is rejected."""
        assert _validate_run_data([]) is False
        assert _validate_run_data("string") is False
        assert _validate_run_data(123) is False
        assert _validate_run_data(None) is False

    def test_validate_rejects_null_started_at(self) -> None:
        """Verify null started_at is rejected."""
        data = {
            "run_id": "abc123",
            "started_at": None,
            "issues": {},
        }
        assert _validate_run_data(data) is False

    def test_validate_accepts_null_issues(self) -> None:
        """Verify null issues is accepted (normalized to {})."""
        data = {
            "run_id": "abc123",
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": None,
        }
        assert _validate_run_data(data) is True

    def test_validate_rejects_non_dict_issues(self) -> None:
        """Verify non-dict issues is rejected."""
        data = {
            "run_id": "abc123",
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": ["list", "not", "dict"],
        }
        assert _validate_run_data(data) is False

    def test_validate_rejects_null_run_id(self) -> None:
        """Verify null run_id is rejected."""
        data = {
            "run_id": None,
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": {},
        }
        assert _validate_run_data(data) is False

    def test_validate_rejects_non_string_run_id(self) -> None:
        """Verify non-string run_id is rejected."""
        data = {
            "run_id": 12345,
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": {},
        }
        assert _validate_run_data(data) is False


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
    """Tests for _parse_run_file_cli function (CLI-specific with stderr warnings)."""

    def test_parse_valid_file(self, tmp_path: Path) -> None:
        """Verify valid JSON file is parsed correctly."""
        run_file = tmp_path / "run.json"
        data = {
            "run_id": "abc123",
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": {"issue-1": {"status": "success"}},
        }
        run_file.write_text(json.dumps(data))

        result = _parse_run_file_cli(run_file)
        assert result is not None
        assert result["run_id"] == "abc123"

    def test_parse_corrupt_file(self, tmp_path: Path) -> None:
        """Verify corrupt JSON file returns None with warning."""
        run_file = tmp_path / "corrupt.json"
        run_file.write_text("not valid json {{{")

        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            result = _parse_run_file_cli(run_file)
            assert result is None
            assert "Warning: skipping corrupt file" in mock_stderr.getvalue()

    def test_parse_missing_required_keys(self, tmp_path: Path) -> None:
        """Verify file missing required keys returns None with warning."""
        run_file = tmp_path / "incomplete.json"
        run_file.write_text('{"run_id": "abc123"}')  # missing started_at, issues

        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            result = _parse_run_file_cli(run_file)
            assert result is None
            assert "Warning: skipping invalid file" in mock_stderr.getvalue()

    def test_parse_invalid_encoding(self, tmp_path: Path) -> None:
        """Verify file with invalid encoding returns None with warning."""
        run_file = tmp_path / "bad_encoding.json"
        # Write invalid UTF-8 bytes directly
        run_file.write_bytes(b'{"run_id": "\xff\xfe invalid bytes"}')

        with patch("sys.stderr", new_callable=StringIO) as mock_stderr:
            result = _parse_run_file_cli(run_file)
            assert result is None
            assert "Warning: skipping corrupt file" in mock_stderr.getvalue()


class TestCollectRuns:
    """Tests for load_runs function (now in run_metadata.py)."""

    def test_collect_all_valid_runs(self, tmp_path: Path) -> None:
        """Verify all valid runs are collected when no limit specified."""
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

        # Without limit, all valid runs are collected
        runs = load_runs(files)
        assert len(runs) == 25

    def test_collect_with_limit_stops_early(self, tmp_path: Path) -> None:
        """Verify load_runs stops after collecting limit valid runs.

        Uses realistic filename format with different timestamps so the
        early termination logic can stop at different timestamp boundaries.
        """
        # Create 25 valid run files with different timestamp prefixes
        # Each file has a unique timestamp so early termination works
        files = []
        for i in range(25):
            # Use different timestamps so we can stop at limit
            # Use minutes instead of hours to avoid invalid T24:00:00
            timestamp = f"2024-01-{(i % 28) + 1:02d}T10-{i:02d}-00"
            run_file = tmp_path / f"{timestamp}_{i:08d}.json"
            data = {
                "run_id": f"run-{i:02d}",
                "started_at": f"2024-01-{(i % 28) + 1:02d}T10:{i:02d}:00+00:00",
                "issues": {},
            }
            run_file.write_text(json.dumps(data))
            files.append(run_file)

        # With limit, only that many runs are collected (early stop)
        runs = load_runs(files, limit=10)
        assert len(runs) == 10
        # Should have collected the first 10 files in order
        assert [r[0]["run_id"] for r in runs] == [f"run-{i:02d}" for i in range(10)]

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

        # load_runs logs warnings instead of printing to stderr
        runs = load_runs([valid_file, invalid_file, corrupt_file])

        assert len(runs) == 1
        assert runs[0][0]["run_id"] == "valid-run"

    def test_collect_returns_path(self, tmp_path: Path) -> None:
        """Verify path is returned with each run."""
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

        runs = load_runs([run_file])
        assert len(runs) == 1
        assert runs[0][1] == run_file

    def test_collect_continues_for_same_timestamp_prefix(self, tmp_path: Path) -> None:
        """Verify load_runs continues past limit for files with same timestamp prefix.

        When multiple runs share the same second-resolution timestamp in their
        filenames, we must collect all of them to ensure correct top-N after
        sorting by the higher-resolution started_at field.
        """
        # Create files with same timestamp prefix but different microsecond started_at
        # Filename format: {timestamp}_{short_id}.json
        # All 3 files have timestamp "2024-01-01T10-00-00" in filename
        # But their started_at has microsecond differences
        files = []

        # File A: filename sorts first (aaa), started_at is LATEST (should be #1)
        file_a = tmp_path / "2024-01-01T10-00-00_aaaaaaaa.json"
        file_a.write_text(
            json.dumps(
                {
                    "run_id": "run-aaa",
                    "started_at": "2024-01-01T10:00:00.900000+00:00",  # Latest
                    "issues": {},
                }
            )
        )
        files.append(file_a)

        # File B: filename sorts second (bbb), started_at is EARLIEST (should be #3)
        file_b = tmp_path / "2024-01-01T10-00-00_bbbbbbbb.json"
        file_b.write_text(
            json.dumps(
                {
                    "run_id": "run-bbb",
                    "started_at": "2024-01-01T10:00:00.100000+00:00",  # Earliest
                    "issues": {},
                }
            )
        )
        files.append(file_b)

        # File C: filename sorts third (ccc), started_at is MIDDLE (should be #2)
        file_c = tmp_path / "2024-01-01T10-00-00_cccccccc.json"
        file_c.write_text(
            json.dumps(
                {
                    "run_id": "run-ccc",
                    "started_at": "2024-01-01T10:00:00.500000+00:00",  # Middle
                    "issues": {},
                }
            )
        )
        files.append(file_c)

        # With limit=2, old behavior would stop after A and B (by filename order)
        # and return [A, B] which after sorting by started_at would be [A, B]
        # But correct top-2 by started_at should be [A, C] (latest two)
        #
        # New behavior: continues collecting all files with same timestamp prefix
        runs = load_runs(files, limit=2)

        # Should collect all 3 files since they share timestamp prefix
        assert len(runs) == 3

        # After sorting by started_at desc and slicing (done by caller),
        # we would get [run-aaa, run-ccc] which is correct
        runs_dicts = [{**data, "metadata_path": str(path)} for data, path in runs]
        sorted_runs = _sort_runs(runs_dicts)[:2]
        assert [r["run_id"] for r in sorted_runs] == ["run-aaa", "run-ccc"]

    def test_collect_stops_at_different_timestamp_prefix(self, tmp_path: Path) -> None:
        """Verify load_runs stops when timestamp prefix changes."""
        files = []

        # Two files with timestamp "2024-01-02T10-00-00"
        file_a = tmp_path / "2024-01-02T10-00-00_aaaaaaaa.json"
        file_a.write_text(
            json.dumps(
                {
                    "run_id": "run-a",
                    "started_at": "2024-01-02T10:00:00+00:00",
                    "issues": {},
                }
            )
        )
        files.append(file_a)

        file_b = tmp_path / "2024-01-02T10-00-00_bbbbbbbb.json"
        file_b.write_text(
            json.dumps(
                {
                    "run_id": "run-b",
                    "started_at": "2024-01-02T10:00:00+00:00",
                    "issues": {},
                }
            )
        )
        files.append(file_b)

        # One file with DIFFERENT timestamp "2024-01-01T10-00-00"
        file_c = tmp_path / "2024-01-01T10-00-00_cccccccc.json"
        file_c.write_text(
            json.dumps(
                {
                    "run_id": "run-c",
                    "started_at": "2024-01-01T10:00:00+00:00",
                    "issues": {},
                }
            )
        )
        files.append(file_c)

        # With limit=2, should collect exactly 2 (A and B have same prefix as boundary)
        # and stop before C (different prefix)
        runs = load_runs(files, limit=2)
        assert len(runs) == 2
        assert {r[0]["run_id"] for r in runs} == {"run-a", "run-b"}


class TestExtractTimestampPrefix:
    """Tests for _extract_timestamp_prefix function (now in run_metadata.py)."""

    def test_standard_filename(self) -> None:
        """Verify extraction from standard run filename format."""
        assert (
            _extract_timestamp_prefix("2024-01-01T10-00-00_abc12345.json")
            == "2024-01-01T10-00-00"
        )

    def test_no_underscore(self) -> None:
        """Verify empty string returned when no underscore in filename."""
        assert _extract_timestamp_prefix("no-underscore.json") == ""

    def test_multiple_underscores(self) -> None:
        """Verify only first underscore is used as delimiter."""
        assert (
            _extract_timestamp_prefix("2024-01-01T10-00-00_abc_extra.json")
            == "2024-01-01T10-00-00"
        )


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

        # Mock get_repo_runs_dir in run_metadata (used by discover_run_files)
        monkeypatch.setattr(
            "src.infra.io.log_output.run_metadata.get_repo_runs_dir",
            lambda _: repo1_dir,
        )

        files = discover_run_files(None)

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

        # Override MALA_RUNS_DIR env var to use test directory
        monkeypatch.setenv("MALA_RUNS_DIR", str(runs_dir))

        files = _discover_run_files_all_repos()

        # Should find both repo run files
        assert len(files) == 2
        run_ids = {json.loads(f.read_text())["run_id"] for f in files}
        assert run_ids == {"repo1-run", "repo2-run"}


class TestFindSessionsForIssue:
    """Tests for find_sessions_for_issue function (now in run_metadata.py)."""

    def test_issue_id_exact_match(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify issue filter matches exactly (case-sensitive)."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir(parents=True)

        run_file = runs_dir / "2024-01-01T10-00-00_12345678.json"
        run_file.write_text(
            json.dumps(
                {
                    "run_id": "run1",
                    "started_at": "2024-01-01T10:00:00+00:00",
                    "issues": {
                        "bd-mala-abc": {"session_id": "s1", "status": "success"},
                        "BD-MALA-ABC": {"session_id": "s2", "status": "failed"},
                        "bd-mala-abc-2": {"session_id": "s3", "status": "success"},
                    },
                }
            )
        )

        monkeypatch.setattr(
            "src.infra.io.log_output.run_metadata.get_repo_runs_dir",
            lambda _: runs_dir,
        )

        result = find_sessions_for_issue(None, "bd-mala-abc")
        assert len(result) == 1
        assert result[0].issue_id == "bd-mala-abc"
        assert result[0].session_id == "s1"

    def test_sessions_output_format(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify extracted sessions have expected fields."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir(parents=True)

        run_file = runs_dir / "2024-01-01T10-00-00_12345678.json"
        run_file.write_text(
            json.dumps(
                {
                    "run_id": "run-123",
                    "started_at": "2024-01-01T10:00:00+00:00",
                    "repo_path": "/home/user/project",
                    "issues": {
                        "issue-1": {
                            "session_id": "sess-abc",
                            "status": "success",
                            "log_path": "/path/to/log.jsonl",
                            "baseline_timestamp": 1700000000,
                        },
                    },
                }
            )
        )

        monkeypatch.setattr(
            "src.infra.io.log_output.run_metadata.get_repo_runs_dir",
            lambda _: runs_dir,
        )

        result = find_sessions_for_issue(None, "issue-1")
        assert len(result) == 1
        session = result[0]
        assert session.run_id == "run-123"
        assert session.session_id == "sess-abc"
        assert session.issue_id == "issue-1"
        assert session.run_started_at == "2024-01-01T10:00:00+00:00"
        assert session.status == "success"
        assert session.log_path == "/path/to/log.jsonl"
        assert session.metadata_path == run_file
        assert session.repo_path == "/home/user/project"
        assert session.baseline_timestamp == 1700000000

    def test_sort_by_run_started_at_desc(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify sessions are sorted by run_started_at descending."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir(parents=True)

        # Create multiple run files with same issue but different timestamps
        for i, ts in enumerate(["2024-01-01", "2024-01-03", "2024-01-02"]):
            run_file = runs_dir / f"{ts}T10-00-00_{chr(ord('a') + i) * 8}.json"
            run_file.write_text(
                json.dumps(
                    {
                        "run_id": chr(ord("a") + i),
                        "started_at": f"{ts}T10:00:00+00:00",
                        "issues": {
                            "issue-1": {"session_id": f"s{i}", "status": "success"},
                        },
                    }
                )
            )

        monkeypatch.setattr(
            "src.infra.io.log_output.run_metadata.get_repo_runs_dir",
            lambda _: runs_dir,
        )

        result = find_sessions_for_issue(None, "issue-1")
        # Should be sorted by started_at desc: 2024-01-03, 2024-01-02, 2024-01-01
        assert [s.run_id for s in result] == ["b", "c", "a"]

    def test_sort_tiebreaker_run_id_asc(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify run_id is used as tiebreaker (ascending) when started_at is same."""
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir(parents=True)

        # Create multiple run files with same timestamp
        for i, rid in enumerate(["c", "a", "b"]):
            run_file = runs_dir / f"2024-01-01T10-00-00_{rid * 8}.json"
            run_file.write_text(
                json.dumps(
                    {
                        "run_id": rid,
                        "started_at": "2024-01-01T10:00:00+00:00",
                        "issues": {
                            "issue-1": {"session_id": f"s{i}", "status": "success"},
                        },
                    }
                )
            )

        monkeypatch.setattr(
            "src.infra.io.log_output.run_metadata.get_repo_runs_dir",
            lambda _: runs_dir,
        )

        result = find_sessions_for_issue(None, "issue-1")
        # Should be sorted by run_id asc when started_at is same
        assert [s.run_id for s in result] == ["a", "b", "c"]


class TestExtractRunIdPrefixFromFilename:
    """Tests for _extract_run_id_prefix_from_filename function."""

    def test_valid_filename(self) -> None:
        """Verify extraction from valid filename format."""
        result = _extract_run_id_prefix_from_filename(
            "2024-01-01T10-00-00_a1b2c3d4.json"
        )
        assert result == "a1b2c3d4"

    def test_filename_without_json_extension(self) -> None:
        """Verify None returned for non-json files."""
        result = _extract_run_id_prefix_from_filename(
            "2024-01-01T10-00-00_a1b2c3d4.txt"
        )
        assert result is None

    def test_filename_without_underscore(self) -> None:
        """Verify None returned for filename without underscore."""
        result = _extract_run_id_prefix_from_filename("a1b2c3d4.json")
        assert result is None

    def test_prefix_wrong_length(self) -> None:
        """Verify None returned when prefix is not 8 chars."""
        # Too short
        result = _extract_run_id_prefix_from_filename("2024-01-01T10-00-00_a1b2.json")
        assert result is None
        # Too long
        result = _extract_run_id_prefix_from_filename(
            "2024-01-01T10-00-00_a1b2c3d4e5.json"
        )
        assert result is None

    def test_multiple_underscores(self) -> None:
        """Verify last underscore is used for split."""
        result = _extract_run_id_prefix_from_filename(
            "2024-01-01T10-00-00_extra_a1b2c3d4.json"
        )
        assert result == "a1b2c3d4"


class TestFindMatchingRuns:
    """Tests for _find_matching_runs function."""

    def test_exact_match_full_uuid(self, tmp_path: Path) -> None:
        """Verify full UUID matches exactly."""
        # Import inside test to avoid stale reference after module reload
        from src.cli.logs import _find_matching_runs

        # Create a run file
        run_data = {
            "run_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": {},
        }
        run_file = tmp_path / "2024-01-01T10-00-00_a1b2c3d4.json"
        run_file.write_text(json.dumps(run_data))

        with patch("src.cli.logs.get_repo_runs_dir", return_value=tmp_path):
            matches, corrupt = _find_matching_runs(
                "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
            )

        assert len(matches) == 1
        assert matches[0][1]["run_id"] == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        assert len(corrupt) == 0

    def test_prefix_match_8_chars(self, tmp_path: Path) -> None:
        """Verify 8-char prefix matches run."""
        from src.cli.logs import _find_matching_runs

        run_data = {
            "run_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": {},
        }
        run_file = tmp_path / "2024-01-01T10-00-00_a1b2c3d4.json"
        run_file.write_text(json.dumps(run_data))

        with patch("src.cli.logs.get_repo_runs_dir", return_value=tmp_path):
            matches, corrupt = _find_matching_runs("a1b2c3d4")

        assert len(matches) == 1
        assert matches[0][1]["run_id"] == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        assert len(corrupt) == 0

    def test_ambiguous_prefix(self, tmp_path: Path) -> None:
        """Verify multiple matches returned for ambiguous prefix."""
        from src.cli.logs import _find_matching_runs

        # Create two runs with same 8-char prefix
        run1 = {
            "run_id": "a1b2c3d4-aaaa-0000-0000-000000000001",
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": {},
        }
        run2 = {
            "run_id": "a1b2c3d4-bbbb-0000-0000-000000000002",
            "started_at": "2024-01-01T11:00:00+00:00",
            "issues": {},
        }
        (tmp_path / "2024-01-01T10-00-00_a1b2c3d4.json").write_text(json.dumps(run1))
        (tmp_path / "2024-01-01T11-00-00_a1b2c3d4.json").write_text(json.dumps(run2))

        with patch("src.cli.logs.get_repo_runs_dir", return_value=tmp_path):
            matches, corrupt = _find_matching_runs("a1b2c3d4")

        assert len(matches) == 2
        assert len(corrupt) == 0

    def test_not_found(self, tmp_path: Path) -> None:
        """Verify empty result when no match found."""
        from src.cli.logs import _find_matching_runs

        run_data = {
            "run_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": {},
        }
        run_file = tmp_path / "2024-01-01T10-00-00_a1b2c3d4.json"
        run_file.write_text(json.dumps(run_data))

        with patch("src.cli.logs.get_repo_runs_dir", return_value=tmp_path):
            matches, corrupt = _find_matching_runs("xxxxxxxx")

        assert len(matches) == 0
        assert len(corrupt) == 0

    def test_corrupt_file_detected(self, tmp_path: Path) -> None:
        """Verify corrupt files are tracked separately."""
        from src.cli.logs import _find_matching_runs

        # Create a corrupt JSON file with matching prefix
        corrupt_file = tmp_path / "2024-01-01T10-00-00_a1b2c3d4.json"
        corrupt_file.write_text("not valid json")

        with patch("src.cli.logs.get_repo_runs_dir", return_value=tmp_path):
            matches, corrupt = _find_matching_runs("a1b2c3d4")

        assert len(matches) == 0
        assert len(corrupt) == 1

    def test_full_uuid_no_match_different_uuid(self, tmp_path: Path) -> None:
        """Verify full UUID doesn't match when only prefix matches."""
        from src.cli.logs import _find_matching_runs

        # File has prefix a1b2c3d4 but different full UUID
        run_data = {
            "run_id": "a1b2c3d4-aaaa-0000-0000-000000000001",
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": {},
        }
        run_file = tmp_path / "2024-01-01T10-00-00_a1b2c3d4.json"
        run_file.write_text(json.dumps(run_data))

        # Search with full UUID that has same prefix but different rest
        with patch("src.cli.logs.get_repo_runs_dir", return_value=tmp_path):
            matches, corrupt = _find_matching_runs(
                "a1b2c3d4-bbbb-0000-0000-000000000002"
            )

        assert len(matches) == 0
        assert len(corrupt) == 0

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Verify empty result for empty directory."""
        from src.cli.logs import _find_matching_runs

        with patch("src.cli.logs.get_repo_runs_dir", return_value=tmp_path):
            matches, corrupt = _find_matching_runs("a1b2c3d4")

        assert len(matches) == 0
        assert len(corrupt) == 0

    def test_nonexistent_directory(self, tmp_path: Path) -> None:
        """Verify empty result when directory doesn't exist."""
        from src.cli.logs import _find_matching_runs

        nonexistent = tmp_path / "does_not_exist"
        with patch("src.cli.logs.get_repo_runs_dir", return_value=nonexistent):
            matches, corrupt = _find_matching_runs("a1b2c3d4")

        assert len(matches) == 0
        assert len(corrupt) == 0


class TestShowCommand:
    """Tests for show command output formatting."""

    def test_show_output_includes_all_issues(self, tmp_path: Path) -> None:
        """Verify show output includes all issues with their details."""
        from typer.testing import CliRunner

        from src.cli.logs import logs_app

        runner = CliRunner()

        run_data = {
            "run_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": {
                "issue-1": {
                    "status": "success",
                    "session_id": "sess-001",
                    "log_path": "/logs/issue-1.jsonl",
                },
                "issue-2": {
                    "status": "failed",
                    "session_id": "sess-002",
                    "log_path": "/logs/issue-2.jsonl",
                },
            },
        }
        run_file = tmp_path / "2024-01-01T10-00-00_a1b2c3d4.json"
        run_file.write_text(json.dumps(run_data))

        with patch("src.cli.logs.get_repo_runs_dir", return_value=tmp_path):
            result = runner.invoke(logs_app, ["show", "a1b2c3d4"])

        assert result.exit_code == 0
        assert "a1b2c3d4-e5f6-7890-abcd-ef1234567890" in result.output
        assert "issue-1" in result.output
        assert "issue-2" in result.output
        assert "success" in result.output
        assert "failed" in result.output
        assert "/logs/issue-1.jsonl" in result.output
        assert "/logs/issue-2.jsonl" in result.output

    def test_show_json_output(self, tmp_path: Path) -> None:
        """Verify --json flag produces valid JSON output."""
        from typer.testing import CliRunner

        from src.cli.logs import logs_app

        runner = CliRunner()

        run_data = {
            "run_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": {"issue-1": {"status": "success"}},
        }
        run_file = tmp_path / "2024-01-01T10-00-00_a1b2c3d4.json"
        run_file.write_text(json.dumps(run_data))

        with patch("src.cli.logs.get_repo_runs_dir", return_value=tmp_path):
            result = runner.invoke(logs_app, ["show", "a1b2c3d4", "--json"])

        assert result.exit_code == 0
        output = json.loads(result.output)
        assert output["run_id"] == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        assert "metadata_path" in output
        assert "issues" in output

    def test_show_not_found_error(self, tmp_path: Path) -> None:
        """Verify exit code 1 when run not found."""
        from typer.testing import CliRunner

        from src.cli.logs import logs_app

        runner = CliRunner()

        with patch("src.cli.logs.get_repo_runs_dir", return_value=tmp_path):
            result = runner.invoke(logs_app, ["show", "xxxxxxxx"])

        assert result.exit_code == 1
        assert "No run found" in result.output

    def test_show_not_found_json_error(self, tmp_path: Path) -> None:
        """Verify JSON error format when run not found."""
        from typer.testing import CliRunner

        from src.cli.logs import logs_app

        runner = CliRunner()

        with patch("src.cli.logs.get_repo_runs_dir", return_value=tmp_path):
            result = runner.invoke(logs_app, ["show", "xxxxxxxx", "--json"])

        assert result.exit_code == 1
        output = json.loads(result.output)
        assert output["error"] == "not_found"

    def test_show_ambiguous_prefix_error(self, tmp_path: Path) -> None:
        """Verify exit code 1 and match list for ambiguous prefix."""
        from typer.testing import CliRunner

        from src.cli.logs import logs_app

        runner = CliRunner()

        # Create two runs with same prefix
        run1 = {
            "run_id": "a1b2c3d4-aaaa-0000-0000-000000000001",
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": {},
        }
        run2 = {
            "run_id": "a1b2c3d4-bbbb-0000-0000-000000000002",
            "started_at": "2024-01-01T11:00:00+00:00",
            "issues": {},
        }
        (tmp_path / "2024-01-01T10-00-00_a1b2c3d4.json").write_text(json.dumps(run1))
        (tmp_path / "2024-01-01T11-00-00_a1b2c3d4.json").write_text(json.dumps(run2))

        with patch("src.cli.logs.get_repo_runs_dir", return_value=tmp_path):
            result = runner.invoke(logs_app, ["show", "a1b2c3d4"])

        assert result.exit_code == 1
        assert "Ambiguous" in result.output
        assert "a1b2c3d4-aaaa-0000-0000-000000000001" in result.output
        assert "a1b2c3d4-bbbb-0000-0000-000000000002" in result.output

    def test_show_ambiguous_prefix_json_error(self, tmp_path: Path) -> None:
        """Verify JSON error format for ambiguous prefix."""
        from typer.testing import CliRunner

        from src.cli.logs import logs_app

        runner = CliRunner()

        run1 = {
            "run_id": "a1b2c3d4-aaaa-0000-0000-000000000001",
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": {},
        }
        run2 = {
            "run_id": "a1b2c3d4-bbbb-0000-0000-000000000002",
            "started_at": "2024-01-01T11:00:00+00:00",
            "issues": {},
        }
        (tmp_path / "2024-01-01T10-00-00_a1b2c3d4.json").write_text(json.dumps(run1))
        (tmp_path / "2024-01-01T11-00-00_a1b2c3d4.json").write_text(json.dumps(run2))

        with patch("src.cli.logs.get_repo_runs_dir", return_value=tmp_path):
            result = runner.invoke(logs_app, ["show", "a1b2c3d4", "--json"])

        assert result.exit_code == 1
        output = json.loads(result.output)
        assert output["error"] == "ambiguous_prefix"
        assert "matches" in output
        assert len(output["matches"]) == 2

    def test_show_corrupt_file_error(self, tmp_path: Path) -> None:
        """Verify exit code 2 when file is corrupt."""
        from typer.testing import CliRunner

        from src.cli.logs import logs_app

        runner = CliRunner()

        corrupt_file = tmp_path / "2024-01-01T10-00-00_a1b2c3d4.json"
        corrupt_file.write_text("not valid json")

        with patch("src.cli.logs.get_repo_runs_dir", return_value=tmp_path):
            result = runner.invoke(logs_app, ["show", "a1b2c3d4"])

        assert result.exit_code == 2
        assert "corrupt" in result.output.lower()

    def test_show_corrupt_file_json_error(self, tmp_path: Path) -> None:
        """Verify JSON error format for corrupt file."""
        from typer.testing import CliRunner

        from src.cli.logs import logs_app

        runner = CliRunner()

        corrupt_file = tmp_path / "2024-01-01T10-00-00_a1b2c3d4.json"
        corrupt_file.write_text("not valid json")

        with patch("src.cli.logs.get_repo_runs_dir", return_value=tmp_path):
            result = runner.invoke(logs_app, ["show", "a1b2c3d4", "--json"])

        assert result.exit_code == 2
        # Extract JSON from output (stderr warning may be mixed in)
        # Find the JSON object in the output
        output_lines = result.output.strip().split("\n")
        json_lines = []
        in_json = False
        for line in output_lines:
            if line.startswith("{"):
                in_json = True
            if in_json:
                json_lines.append(line)
            if line.startswith("}"):
                break
        output = json.loads("\n".join(json_lines))
        assert output["error"] == "corrupt"

    def test_show_prefix_plus_corrupt_is_ambiguous(self, tmp_path: Path) -> None:
        """Verify valid match + corrupt file with same prefix is treated as ambiguous."""
        from typer.testing import CliRunner

        from src.cli.logs import logs_app

        runner = CliRunner()

        # One valid run and one corrupt file with same prefix
        valid_run = {
            "run_id": "a1b2c3d4-aaaa-0000-0000-000000000001",
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": {},
        }
        (tmp_path / "2024-01-01T10-00-00_a1b2c3d4.json").write_text(
            json.dumps(valid_run)
        )
        (tmp_path / "2024-01-01T11-00-00_a1b2c3d4.json").write_text("not valid json")

        with patch("src.cli.logs.get_repo_runs_dir", return_value=tmp_path):
            result = runner.invoke(logs_app, ["show", "a1b2c3d4"])

        # Should be ambiguous (exit 1), not success
        assert result.exit_code == 1
        assert "Ambiguous" in result.output or "ambiguous" in result.output


class TestFindMatchingRunsFallback:
    """Tests for _find_matching_runs fallback to JSON scan."""

    def test_short_prefix_uses_fallback(self, tmp_path: Path) -> None:
        """Verify short prefix (< 8 chars) uses JSON field scan."""
        from src.cli.logs import _find_matching_runs

        run_data = {
            "run_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": {},
        }
        (tmp_path / "2024-01-01T10-00-00_a1b2c3d4.json").write_text(
            json.dumps(run_data)
        )

        with patch("src.cli.logs.get_repo_runs_dir", return_value=tmp_path):
            # Search with 4-char prefix
            matches, _corrupt = _find_matching_runs("a1b2")

        assert len(matches) == 1
        assert matches[0][1]["run_id"] == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

    def test_nonconforming_filename_uses_fallback(self, tmp_path: Path) -> None:
        """Verify fallback finds runs with non-standard filenames."""
        from src.cli.logs import _find_matching_runs

        run_data = {
            "run_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": {},
        }
        # Non-standard filename (no underscore-prefix pattern)
        (tmp_path / "manual-run.json").write_text(json.dumps(run_data))

        with patch("src.cli.logs.get_repo_runs_dir", return_value=tmp_path):
            matches, _corrupt = _find_matching_runs("a1b2c3d4")

        assert len(matches) == 1
        assert matches[0][1]["run_id"] == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

    def test_fallback_runs_even_with_corrupt_prefix_match(self, tmp_path: Path) -> None:
        """Verify fallback scan finds valid run even when corrupt prefix file exists."""
        from src.cli.logs import _find_matching_runs

        valid_run = {
            "run_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": {},
        }
        # Corrupt file with conforming name
        (tmp_path / "2024-01-01T10-00-00_a1b2c3d4.json").write_text("not valid json")
        # Valid run with non-conforming filename
        (tmp_path / "manual-run.json").write_text(json.dumps(valid_run))

        with patch("src.cli.logs.get_repo_runs_dir", return_value=tmp_path):
            matches, corrupt = _find_matching_runs("a1b2c3d4")

        # Should find the valid run even though corrupt file exists
        assert len(matches) == 1
        assert matches[0][1]["run_id"] == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
        assert len(corrupt) == 1  # Corrupt file still tracked

    def test_full_uuid_with_corrupt_prefix_returns_not_found(
        self, tmp_path: Path
    ) -> None:
        """Verify full UUID search with corrupt prefix match returns not_found (exit 1)."""
        from typer.testing import CliRunner

        from src.cli.logs import logs_app

        runner = CliRunner()

        # Corrupt file with matching 8-char prefix
        (tmp_path / "2024-01-01T10-00-00_a1b2c3d4.json").write_text("not valid json")

        with patch("src.cli.logs.get_repo_runs_dir", return_value=tmp_path):
            # Search with full UUID (not just 8-char prefix)
            result = runner.invoke(
                logs_app, ["show", "a1b2c3d4-xxxx-yyyy-zzzz-000000000000", "--json"]
            )

        # Should be not_found (exit 1), NOT corrupt (exit 2)
        assert result.exit_code == 1
        # Extract JSON from output (stderr warning may be mixed in)
        output_lines = result.output.strip().split("\n")
        json_lines = []
        in_json = False
        for line in output_lines:
            if line.startswith("{"):
                in_json = True
            if in_json:
                json_lines.append(line)
            if line.startswith("}"):
                break
        output = json.loads("\n".join(json_lines))
        assert output["error"] == "not_found"
        # Should mention corrupt files as a note
        assert "corrupt" in output["message"].lower()

    def test_medium_prefix_9_to_35_chars(self, tmp_path: Path) -> None:
        """Verify prefixes between 9-35 chars work correctly."""
        from src.cli.logs import _find_matching_runs

        run_data = {
            "run_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "started_at": "2024-01-01T10:00:00+00:00",
            "issues": {},
        }
        (tmp_path / "2024-01-01T10-00-00_a1b2c3d4.json").write_text(
            json.dumps(run_data)
        )

        with patch("src.cli.logs.get_repo_runs_dir", return_value=tmp_path):
            # Search with 12-char prefix (between 8 and full UUID)
            matches, _corrupt = _find_matching_runs("a1b2c3d4-e5f")

        assert len(matches) == 1
        assert matches[0][1]["run_id"] == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"

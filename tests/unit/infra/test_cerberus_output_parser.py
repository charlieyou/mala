"""Tests for Cerberus v2 gate-state output parsing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.infra.clients.cerberus_output_parser import (
    ReviewOutputParser,
    parse_gate_state,
)


def _gate_state(verdict: str | None = "pass", **overrides: object) -> str:
    data: dict[str, object] = {
        "schema_version": "2",
        "run_key": "run-1",
        "host": "generic",
        "project_key": "project-1",
        "session_id": "session-1",
        "transcript_path": "/tmp/transcript.jsonl",
        "status": "resolved",
        "verdict": verdict,
        "resolution_reason": "complete",
        "current_iteration": 1,
        "max_rounds": 1,
        "debate": False,
        "roster_id": "default",
        "started_at": "2026-05-11T00:00:00Z",
        "ended_at": "2026-05-11T00:01:00Z",
    }
    data.update(overrides)
    return json.dumps(data)


def _write_reviewer_output(
    state_root: Path,
    *,
    reviewer: str = "codex#1",
    project_key: str = "project-1",
    run_key: str = "run-1",
    findings: list[dict[str, object]] | None = None,
    raw: str | None = None,
) -> Path:
    reviewer_dir = (
        state_root
        / project_key
        / run_key
        / "iterations"
        / "1"
        / "round-1"
        / "reviewers"
        / reviewer
    )
    reviewer_dir.mkdir(parents=True, exist_ok=True)
    output_path = reviewer_dir / "output.json"
    output_path.write_text(
        raw
        if raw is not None
        else json.dumps({"findings": findings or [], "verdict": "pass"}),
        encoding="utf-8",
    )
    return output_path


def _finding(title: str = "Test finding") -> dict[str, object]:
    return {
        "file_path": "src/test.py",
        "line_start": 10,
        "line_end": 12,
        "priority": 1,
        "title": title,
        "body": "Test body",
    }


class TestParseGateState:
    def test_happy_path_extracts_v2_fields(self) -> None:
        state = parse_gate_state(_gate_state(verdict="fail"))

        assert state.schema_version == "2"
        assert state.run_key == "run-1"
        assert state.host == "generic"
        assert state.project_key == "project-1"
        assert state.session_id == "session-1"
        assert state.transcript_path == "/tmp/transcript.jsonl"
        assert state.status == "resolved"
        assert state.verdict == "fail"
        assert state.resolution_reason == "complete"
        assert state.current_iteration == 1
        assert state.max_rounds == 1
        assert state.debate is False
        assert state.roster_id == "default"
        assert state.started_at == "2026-05-11T00:00:00Z"
        assert state.ended_at == "2026-05-11T00:01:00Z"

    def test_numeric_schema_version_is_accepted(self) -> None:
        state = parse_gate_state(_gate_state(schema_version=1))

        assert state.schema_version == "1"

    def test_malformed_json(self) -> None:
        with pytest.raises(ValueError, match="JSON parse error"):
            parse_gate_state("not json")

    def test_missing_required_field(self) -> None:
        with pytest.raises(ValueError, match="run_key"):
            parse_gate_state(_gate_state(run_key=None))

    def test_verdict_null_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid verdict"):
            parse_gate_state(_gate_state(verdict=None))

    def test_verdict_pending_is_invalid(self) -> None:
        with pytest.raises(ValueError, match="Invalid verdict"):
            parse_gate_state(_gate_state(verdict="pending"))

    def test_status_pending_is_invalid_after_wait(self) -> None:
        with pytest.raises(ValueError, match="pending status"):
            parse_gate_state(_gate_state(status="pending"))

    def test_verdict_requires_decision_is_valid(self) -> None:
        state = parse_gate_state(_gate_state(verdict="requires_decision"))

        assert state.verdict == "requires_decision"


class TestMapExitCodeToResult:
    @pytest.fixture
    def parser(self) -> ReviewOutputParser:
        return ReviewOutputParser()

    def test_zero_exit_pass_with_populated_reviewers_passes(
        self, parser: ReviewOutputParser, tmp_path: Path
    ) -> None:
        _write_reviewer_output(tmp_path)

        result = parser.map_exit_code_to_result(
            0,
            _gate_state(verdict="pass"),
            "",
            state_root=tmp_path,
            project_key="project-1",
            run_key="run-1",
        )

        assert result.passed is True
        assert result.issues == []
        assert result.parse_error is None
        assert result.fatal_error is False

    def test_zero_exit_fail_with_populated_reviewers_fails_with_findings(
        self, parser: ReviewOutputParser, tmp_path: Path
    ) -> None:
        _write_reviewer_output(tmp_path, findings=[_finding()])

        result = parser.map_exit_code_to_result(
            0,
            _gate_state(verdict="fail"),
            "",
            state_root=tmp_path,
            project_key="project-1",
            run_key="run-1",
        )

        assert result.passed is False
        assert len(result.issues) == 1
        assert result.issues[0].reviewer == "codex#1"
        assert result.parse_error is None
        assert result.fatal_error is False

    def test_pass_with_missing_reviewers_dir_fails_closed(
        self, parser: ReviewOutputParser, tmp_path: Path
    ) -> None:
        (tmp_path / "project-1" / "run-1" / "iterations" / "1" / "round-1").mkdir(
            parents=True
        )

        result = parser.map_exit_code_to_result(
            0,
            _gate_state(verdict="pass"),
            "",
            state_root=tmp_path,
            project_key="project-1",
            run_key="run-1",
        )

        assert result.passed is False
        assert result.parse_error is not None
        assert "Missing reviewers directory" in result.parse_error
        assert result.issues == []

    def test_fail_with_missing_reviewers_dir_surfaces_parse_error(
        self, parser: ReviewOutputParser, tmp_path: Path
    ) -> None:
        (tmp_path / "project-1" / "run-1" / "iterations" / "1" / "round-1").mkdir(
            parents=True
        )

        result = parser.map_exit_code_to_result(
            0,
            _gate_state(verdict="fail"),
            "",
            state_root=tmp_path,
            project_key="project-1",
            run_key="run-1",
        )

        assert result.passed is False
        assert result.parse_error is not None
        assert "Missing reviewers directory" in result.parse_error

    def test_pass_with_malformed_peer_fails_closed_and_keeps_valid_findings(
        self, parser: ReviewOutputParser, tmp_path: Path
    ) -> None:
        _write_reviewer_output(tmp_path, reviewer="codex#1", findings=[_finding()])
        _write_reviewer_output(tmp_path, reviewer="gemini#1", raw="{broken")

        result = parser.map_exit_code_to_result(
            0,
            _gate_state(verdict="pass"),
            "",
            state_root=tmp_path,
            project_key="project-1",
            run_key="run-1",
        )

        assert result.passed is False
        assert result.parse_error is not None
        assert "Malformed reviewer output JSON" in result.parse_error
        assert len(result.issues) == 1
        assert result.issues[0].reviewer == "codex#1"

    def test_fail_with_malformed_peer_surfaces_parse_error_and_findings(
        self, parser: ReviewOutputParser, tmp_path: Path
    ) -> None:
        _write_reviewer_output(tmp_path, reviewer="codex#1", findings=[_finding()])
        _write_reviewer_output(tmp_path, reviewer="gemini#1", raw="{broken")

        result = parser.map_exit_code_to_result(
            0,
            _gate_state(verdict="fail"),
            "",
            state_root=tmp_path,
            project_key="project-1",
            run_key="run-1",
        )

        assert result.passed is False
        assert result.parse_error is not None
        assert "Malformed reviewer output JSON" in result.parse_error
        assert len(result.issues) == 1

    def test_requires_decision_preserves_findings_with_decision_context(
        self, parser: ReviewOutputParser, tmp_path: Path
    ) -> None:
        _write_reviewer_output(tmp_path, findings=[_finding()])

        result = parser.map_exit_code_to_result(
            0,
            _gate_state(verdict="requires_decision"),
            "",
            state_root=tmp_path,
            project_key="project-1",
            run_key="run-1",
        )

        assert result.passed is False
        assert result.fatal_error is False
        assert result.parse_error is None
        assert len(result.issues) == 2
        assert result.issues[0].priority == 1
        assert "no consensus" in result.issues[0].title
        assert "<iteration_dir>" not in result.issues[0].body
        assert (
            str(tmp_path / "project-1" / "run-1" / "iterations")
            in result.issues[0].body
        )
        assert result.issues[1].title == "Test finding"
        assert result.issues[1].body == "Test body"
        assert result.issues[1].reviewer == "codex#1"

    def test_requires_decision_without_findings_still_synthesizes_blocking_issue(
        self, parser: ReviewOutputParser, tmp_path: Path
    ) -> None:
        _write_reviewer_output(tmp_path, findings=[])

        result = parser.map_exit_code_to_result(
            0,
            _gate_state(verdict="requires_decision"),
            "",
            state_root=tmp_path,
            project_key="project-1",
            run_key="run-1",
        )

        assert result.passed is False
        assert result.fatal_error is False
        assert result.parse_error is None
        assert len(result.issues) == 1
        assert "no consensus" in result.issues[0].title

    def test_requires_decision_with_malformed_peer_surfaces_parse_error_and_findings(
        self, parser: ReviewOutputParser, tmp_path: Path
    ) -> None:
        _write_reviewer_output(tmp_path, reviewer="codex#1", findings=[_finding()])
        _write_reviewer_output(tmp_path, reviewer="gemini#1", raw="{broken")

        result = parser.map_exit_code_to_result(
            0,
            _gate_state(verdict="requires_decision"),
            "",
            state_root=tmp_path,
            project_key="project-1",
            run_key="run-1",
        )

        assert result.passed is False
        assert result.fatal_error is False
        assert result.parse_error is not None
        assert "Malformed reviewer output JSON" in result.parse_error
        assert len(result.issues) == 2
        assert "no consensus" in result.issues[0].title
        assert result.issues[1].title == "Test finding"
        assert result.issues[1].reviewer == "codex#1"

    def test_non_zero_exit_uses_stderr_tail(self, parser: ReviewOutputParser) -> None:
        result = parser.map_exit_code_to_result(
            2,
            "",
            "\n".join(f"line {index}" for index in range(25)),
        )

        assert result.passed is False
        assert result.issues == []
        assert result.parse_error is not None
        assert "line 5" in result.parse_error
        assert "line 24" in result.parse_error
        assert "line 4" not in result.parse_error
        assert result.fatal_error is False

    def test_timeout_case(self, parser: ReviewOutputParser) -> None:
        result = parser.map_exit_code_to_result(3, "", "")

        assert result.passed is False
        assert result.issues == []
        assert result.parse_error == "timeout"
        assert result.fatal_error is False

    def test_review_log_path_preserved(
        self, parser: ReviewOutputParser, tmp_path: Path
    ) -> None:
        _write_reviewer_output(tmp_path)
        log_path = Path("/tmp/review-session")

        result = parser.map_exit_code_to_result(
            0,
            _gate_state(verdict="pass"),
            "",
            review_log_path=log_path,
            state_root=tmp_path,
            project_key="project-1",
            run_key="run-1",
        )

        assert result.review_log_path == log_path

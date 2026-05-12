"""Tests for Cerberus v2 iteration findings reader."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import pytest

from src.infra.clients.cerberus_iteration_findings import (
    latest_iteration_dir,
    latest_round_dir,
    read_findings,
)

if TYPE_CHECKING:
    from pathlib import Path


def _run_root(tmp_path: Path) -> Path:
    return tmp_path / "state" / "project" / "run"


def _write_output(
    run_root: Path,
    iteration: int,
    round_index: int,
    reviewer: str,
    findings: list[dict[str, Any]] | None = None,
) -> Path:
    output_path = (
        run_root
        / "iterations"
        / str(iteration)
        / f"round-{round_index}"
        / "reviewers"
        / reviewer
        / "output.json"
    )
    output_path.parent.mkdir(parents=True)
    output_path.write_text(
        json.dumps(
            {
                "findings": findings or [_finding(title=f"{reviewer} finding")],
                "verdict": "fail",
                "summary": "summary",
                "overall_confidence": "high",
                "strategy": "single",
                "round": round_index,
                "peer_responses_seen": [],
            }
        ),
        encoding="utf-8",
    )
    return output_path


def _finding(
    *,
    title: str = "Finding",
    priority: int | None = 1,
    file_path: str | None = "src/app.py",
    line_start: int | None = 10,
    line_end: int | None = 12,
) -> dict[str, Any]:
    return {
        "title": title,
        "body": "Body",
        "priority": priority,
        "file_path": file_path,
        "line_start": line_start,
        "line_end": line_end,
        "confidence": "high",
        "severity": "high",
    }


def test_happy_single_pass_preserves_reviewer_attribution(tmp_path: Path) -> None:
    run_root = _run_root(tmp_path)
    for reviewer in ("claude#1", "codex#1", "gemini#1"):
        _write_output(run_root, 1, 1, reviewer)

    issues, parse_errors = read_findings(tmp_path / "state", "project", "run")

    assert parse_errors == []
    assert [issue.reviewer for issue in issues] == ["claude#1", "codex#1", "gemini#1"]
    assert [issue.title for issue in issues] == [
        "claude#1 finding",
        "codex#1 finding",
        "gemini#1 finding",
    ]


def test_debate_returns_only_latest_round(tmp_path: Path) -> None:
    run_root = _run_root(tmp_path)
    _write_output(run_root, 1, 1, "codex#1", [_finding(title="round 1")])
    _write_output(run_root, 1, 2, "codex#1", [_finding(title="round 2")])
    _write_output(run_root, 1, 3, "codex#1", [_finding(title="round 3")])

    issues, parse_errors = read_findings(tmp_path / "state", "project", "run")

    assert parse_errors == []
    assert [issue.title for issue in issues] == ["round 3"]


def test_latest_iteration_auto_selects_highest_iteration(tmp_path: Path) -> None:
    run_root = _run_root(tmp_path)
    _write_output(run_root, 1, 1, "codex#1", [_finding(title="iteration 1")])
    _write_output(run_root, 2, 1, "codex#1", [_finding(title="iteration 2")])

    issues, parse_errors = read_findings(tmp_path / "state", "project", "run")

    assert parse_errors == []
    assert [issue.title for issue in issues] == ["iteration 2"]


def test_explicit_iteration_and_round_are_used(tmp_path: Path) -> None:
    run_root = _run_root(tmp_path)
    _write_output(run_root, 1, 1, "codex#1", [_finding(title="selected")])
    _write_output(run_root, 2, 3, "codex#1", [_finding(title="latest")])

    issues, parse_errors = read_findings(
        tmp_path / "state", "project", "run", iteration=1, round_index=1
    )

    assert parse_errors == []
    assert [issue.title for issue in issues] == ["selected"]


def test_missing_iterations_dir_returns_parse_error(tmp_path: Path) -> None:
    issues, parse_errors = read_findings(tmp_path / "state", "project", "run")

    assert issues == []
    assert len(parse_errors) == 1
    assert "Missing iterations directory" in parse_errors[0]


def test_empty_iteration_dir_returns_parse_error(tmp_path: Path) -> None:
    iterations_dir = _run_root(tmp_path) / "iterations"
    iterations_dir.mkdir(parents=True)

    issues, parse_errors = read_findings(tmp_path / "state", "project", "run")

    assert issues == []
    assert len(parse_errors) == 1
    assert "No numeric iteration directories" in parse_errors[0]


def test_round_dir_without_reviewers_returns_parse_error(tmp_path: Path) -> None:
    (_run_root(tmp_path) / "iterations" / "1" / "round-1").mkdir(parents=True)

    issues, parse_errors = read_findings(tmp_path / "state", "project", "run")

    assert issues == []
    assert len(parse_errors) == 1
    assert "Missing reviewers directory" in parse_errors[0]


def test_reviewer_dir_missing_output_is_skipped_with_parse_error(
    tmp_path: Path,
) -> None:
    run_root = _run_root(tmp_path)
    _write_output(run_root, 1, 1, "codex#1", [_finding(title="valid")])
    (run_root / "iterations" / "1" / "round-1" / "reviewers" / "claude#1").mkdir()

    issues, parse_errors = read_findings(tmp_path / "state", "project", "run")

    assert [issue.title for issue in issues] == ["valid"]
    assert len(parse_errors) == 1
    assert "Missing reviewer output file" in parse_errors[0]


def test_malformed_output_is_skipped_with_parse_error(tmp_path: Path) -> None:
    run_root = _run_root(tmp_path)
    _write_output(run_root, 1, 1, "codex#1", [_finding(title="valid")])
    bad_output = (
        run_root
        / "iterations"
        / "1"
        / "round-1"
        / "reviewers"
        / "claude#1"
        / "output.json"
    )
    bad_output.parent.mkdir(parents=True)
    bad_output.write_text("{not json", encoding="utf-8")

    issues, parse_errors = read_findings(tmp_path / "state", "project", "run")

    assert [issue.title for issue in issues] == ["valid"]
    assert len(parse_errors) == 1
    assert "Malformed reviewer output JSON" in parse_errors[0]


@pytest.mark.parametrize("priority", [0, 1, 2, 3])
def test_p0_to_p3_priority_parsing(tmp_path: Path, priority: int) -> None:
    run_root = _run_root(tmp_path)
    _write_output(run_root, 1, 1, "codex#1", [_finding(priority=priority)])

    issues, parse_errors = read_findings(tmp_path / "state", "project", "run")

    assert parse_errors == []
    assert issues[0].priority == priority


@pytest.mark.parametrize(
    ("line_start", "line_end", "expected_start", "expected_end"),
    [(7, 7, 7, 7), (None, None, 0, 0)],
)
def test_line_range_edge_cases(
    tmp_path: Path,
    line_start: int | None,
    line_end: int | None,
    expected_start: int,
    expected_end: int,
) -> None:
    run_root = _run_root(tmp_path)
    _write_output(
        run_root,
        1,
        1,
        "codex#1",
        [_finding(line_start=line_start, line_end=line_end)],
    )

    issues, parse_errors = read_findings(tmp_path / "state", "project", "run")

    assert parse_errors == []
    assert issues[0].line_start == expected_start
    assert issues[0].line_end == expected_end


def test_latest_helpers_return_highest_numbered_directories(tmp_path: Path) -> None:
    iterations_dir = _run_root(tmp_path) / "iterations"
    (iterations_dir / "1" / "round-1").mkdir(parents=True)
    (iterations_dir / "2" / "round-3").mkdir(parents=True)
    (iterations_dir / "2" / "round-not-a-number").mkdir()

    iteration_dir, iteration_error = latest_iteration_dir(iterations_dir)
    assert iteration_error is None
    assert iteration_dir == iterations_dir / "2"

    round_dir, round_error = latest_round_dir(iterations_dir / "2")
    assert round_error is None
    assert round_dir == iterations_dir / "2" / "round-3"

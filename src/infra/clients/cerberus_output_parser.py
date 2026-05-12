"""Review output parsing for Cerberus v2 gate-state JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path  # noqa: TC003 (runtime import for get_type_hints compatibility)
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.protocols.events import MalaEventSink


@dataclass
class ReviewIssue:
    """A single issue found during external review."""

    file: str
    line_start: int
    line_end: int
    priority: int | None  # 0=P0, 1=P1, 2=P2, 3=P3, or None
    title: str
    body: str
    reviewer: str  # Which reviewer found this issue


@dataclass
class ReviewResult:
    """Result of a Cerberus review-gate review."""

    passed: bool
    issues: list[ReviewIssue] = field(default_factory=list)
    parse_error: str | None = None
    fatal_error: bool = False
    review_log_path: Path | None = None
    interrupted: bool = False


@dataclass(frozen=True)
class GateState:
    """Cerberus v2 gate-state record."""

    schema_version: str | None
    run_key: str
    host: str | None
    project_key: str
    session_id: str | None
    transcript_path: str | None
    status: str
    verdict: str
    resolution_reason: str | None
    current_iteration: int
    max_rounds: int | None
    debate: bool | None
    roster_id: str | None
    started_at: str | None
    ended_at: str | None


VALID_VERDICTS = {"pass", "fail", "requires_decision"}


def parse_gate_state(stdout: str) -> GateState:
    """Parse Cerberus v2 gate-state JSON from wait/status stdout."""
    if not stdout or not stdout.strip():
        raise ValueError("Empty output from cerberus wait")

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parse error: {e}") from e

    if not isinstance(data, dict):
        raise ValueError("Root element is not an object")

    status = _required_str(data, "status")
    if status == "pending":
        raise ValueError("wait returned pending status")
    if status != "resolved":
        raise ValueError(f"Invalid status: {status}")

    verdict = data.get("verdict")
    if verdict not in VALID_VERDICTS:
        raise ValueError(f"Invalid verdict: {verdict}")

    current_iteration = data.get("current_iteration")
    if not isinstance(current_iteration, int):
        raise ValueError("'current_iteration' must be an integer")

    return GateState(
        schema_version=_optional_str(data, "schema_version"),
        run_key=_required_str(data, "run_key"),
        host=_optional_str(data, "host"),
        project_key=_required_str(data, "project_key"),
        session_id=_optional_str(data, "session_id"),
        transcript_path=_optional_str(data, "transcript_path"),
        status=status,
        verdict=verdict,
        resolution_reason=_optional_str(data, "resolution_reason"),
        current_iteration=current_iteration,
        max_rounds=_optional_int(data, "max_rounds"),
        debate=_optional_bool(data, "debate"),
        roster_id=_optional_str(data, "roster_id"),
        started_at=_optional_str(data, "started_at"),
        ended_at=_optional_str(data, "ended_at"),
    )


def _required_str(data: dict[str, Any], field_name: str) -> str:
    value = data.get(field_name)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"'{field_name}' must be a non-empty string")
    return value


def _optional_str(data: dict[str, Any], field_name: str) -> str | None:
    value = data.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"'{field_name}' must be a string or null")
    return value


def _optional_int(data: dict[str, Any], field_name: str) -> int | None:
    value = data.get(field_name)
    if value is None:
        return None
    if not isinstance(value, int):
        raise ValueError(f"'{field_name}' must be an integer or null")
    return value


def _optional_bool(data: dict[str, Any], field_name: str) -> bool | None:
    value = data.get(field_name)
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(f"'{field_name}' must be a boolean or null")
    return value


class ReviewOutputParser:
    """Parses Cerberus v2 gate-state JSON and maps exit codes to results."""

    def parse_gate_state(self, stdout: str) -> GateState:
        """Parse Cerberus v2 gate-state JSON from wait/status stdout."""
        return parse_gate_state(stdout)

    def parse_json(self, output: str) -> tuple[bool, list[ReviewIssue], str | None]:
        """Parse only the v2 gate-state verdict.

        Per-reviewer findings are stored on disk in v2 and are intentionally not
        read by this compatibility method.
        """
        try:
            state = parse_gate_state(output)
        except ValueError as e:
            return False, [], str(e)
        return state.verdict == "pass", [], None

    def map_exit_code_to_result(
        self,
        exit_code: int,
        stdout: str,
        stderr: str,
        review_log_path: Path | None = None,
        event_sink: MalaEventSink | None = None,
        *,
        state_root: Path | None = None,
        project_key: str | None = None,
        run_key: str | None = None,
    ) -> ReviewResult:
        """Map Cerberus wait exit code plus v2 state artifacts to ReviewResult."""
        _ = event_sink

        if exit_code != 0:
            return ReviewResult(
                passed=False,
                issues=[],
                parse_error=_stderr_tail(
                    stderr,
                    fallback="timeout" if exit_code == 3 else "cerberus wait failed",
                ),
                fatal_error=False,
                review_log_path=review_log_path,
            )

        try:
            gate_state = parse_gate_state(stdout)
        except ValueError as e:
            return ReviewResult(
                passed=False,
                issues=[],
                parse_error=str(e),
                fatal_error=False,
                review_log_path=review_log_path,
            )

        if state_root is None or project_key is None or run_key is None:
            return ReviewResult(
                passed=False,
                issues=[],
                parse_error=(
                    "Missing Cerberus state identifiers: state_root, project_key, "
                    "and run_key are required"
                ),
                fatal_error=False,
                review_log_path=review_log_path,
            )

        from src.infra.clients.cerberus_iteration_findings import read_findings

        issues, parse_errors = read_findings(state_root, project_key, run_key)
        parse_error = "; ".join(parse_errors) if parse_errors else None

        if gate_state.verdict == "requires_decision":
            iteration_dir = state_root / project_key / run_key / "iterations"
            return ReviewResult(
                passed=False,
                issues=[_requires_decision_issue(iteration_dir)],
                parse_error=parse_error,
                fatal_error=False,
                review_log_path=review_log_path,
            )

        return ReviewResult(
            passed=gate_state.verdict == "pass" and parse_error is None,
            issues=issues,
            parse_error=parse_error,
            fatal_error=False,
            review_log_path=review_log_path,
        )


def _requires_decision_issue(iteration_dir: Path) -> ReviewIssue:
    return ReviewIssue(
        title="Cerberus reviewers reached no consensus",
        body=(
            "Gate verdict=requires_decision. Inspect per-reviewer outputs under "
            f"{iteration_dir}; human decision or re-run required."
        ),
        priority=1,
        file="",
        line_start=0,
        line_end=0,
        reviewer="cerberus",
    )


def _stderr_tail(stderr: str, *, fallback: str) -> str:
    stripped = stderr.strip()
    if not stripped:
        return fallback
    lines = stripped.splitlines()
    return "\n".join(lines[-20:])


_parser = ReviewOutputParser()


def parse_cerberus_json(output: str) -> tuple[bool, list[ReviewIssue], str | None]:
    """Parse Cerberus v2 gate-state JSON verdict without reading findings."""
    return _parser.parse_json(output)


def map_exit_code_to_result(
    exit_code: int,
    stdout: str,
    stderr: str,
    review_log_path: Path | None = None,
    event_sink: MalaEventSink | None = None,
    *,
    state_root: Path | None = None,
    project_key: str | None = None,
    run_key: str | None = None,
) -> ReviewResult:
    """Map Cerberus wait exit code plus v2 state artifacts to ReviewResult."""
    return _parser.map_exit_code_to_result(
        exit_code,
        stdout,
        stderr,
        review_log_path=review_log_path,
        event_sink=event_sink,
        state_root=state_root,
        project_key=project_key,
        run_key=run_key,
    )

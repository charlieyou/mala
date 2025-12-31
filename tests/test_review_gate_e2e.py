"""E2E coverage for passing Claude session ID into review-gate."""

from pathlib import Path

import pytest

from src.cerberus_review import DefaultReviewer

pytestmark = [pytest.mark.e2e]


def _write_fake_review_gate(bin_dir: Path) -> Path:
    """Create a fake review-gate binary that records env to disk."""
    script_path = bin_dir / "review-gate"
    script_path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

capture_path="${REVIEW_GATE_CAPTURE_PATH:-}"
if [[ -z "$capture_path" ]]; then
  echo "REVIEW_GATE_CAPTURE_PATH missing" >&2
  exit 1
fi

command="${1:-}"
shift || true

if [[ "$command" == "spawn-code-review" ]]; then
  echo "spawn:env=${CLAUDE_SESSION_ID:-}" > "$capture_path"
  # spawn-code-review doesn't output JSON - it just spawns reviewers
  exit 0
fi

if [[ "$command" == "wait" ]]; then
  # Parse --session-id from args to verify it's passed correctly
  session_id_arg=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --session-id=*) session_id_arg="${1#*=}"; shift ;;
      --session-id) session_id_arg="${2:-}"; shift 2 ;;
      *) shift ;;
    esac
  done
  echo "wait:env=${CLAUDE_SESSION_ID:-},arg=${session_id_arg}" >> "$capture_path"
  echo '{"consensus":{"verdict":"PASS"},"issues":[],"reviewers":{}}'
  exit 0
fi

echo "unknown command" >&2
exit 1
""",
        encoding="utf-8",
    )
    script_path.chmod(0o755)
    return script_path


@pytest.mark.asyncio
async def test_review_gate_receives_session_id(tmp_path: Path) -> None:
    """DefaultReviewer should pass CLAUDE_SESSION_ID to review-gate."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_review_gate(bin_dir)

    capture_path = tmp_path / "review-gate-env.txt"
    reviewer = DefaultReviewer(
        repo_path=tmp_path,
        bin_path=bin_dir,
        env={"REVIEW_GATE_CAPTURE_PATH": str(capture_path)},
    )

    result = await reviewer(
        diff_range="base..head",
        claude_session_id="session-xyz",
    )

    assert result.passed is True
    assert capture_path.exists()
    lines = capture_path.read_text().splitlines()
    # Verify spawn receives session ID via environment
    assert lines[0] == "spawn:env=session-xyz"
    # Verify wait receives session ID via both env AND --session-id arg
    assert lines[1] == "wait:env=session-xyz,arg=session-xyz"

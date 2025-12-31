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
  echo "${CLAUDE_SESSION_ID:-}" > "$capture_path"
  echo '{"session_key":"test-session"}'
  exit 0
fi

if [[ "$command" == "wait" ]]; then
  echo "${CLAUDE_SESSION_ID:-}" >> "$capture_path"
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
    assert "session-xyz" in capture_path.read_text().splitlines()

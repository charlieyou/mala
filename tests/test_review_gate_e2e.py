"""E2E coverage for Cerberus review-gate integration.

Uses the real Cerberus review-gate CLI with --mode=fast to test the full
integration path. This catches protocol mismatches between mala and Cerberus.
"""

import subprocess
import uuid
from pathlib import Path

import pytest

from src.cerberus_review import DefaultReviewer
from src.config import _find_cerberus_bin_path

pytestmark = [pytest.mark.e2e]


def _find_review_gate_bin() -> Path | None:
    """Find the real review-gate binary from Claude's plugin cache."""
    claude_config = Path.home() / ".claude"
    return _find_cerberus_bin_path(claude_config)


def _setup_git_repo(repo_path: Path) -> str:
    """Initialize a git repo with a commit and uncommitted change.
    
    Returns the base SHA for the diff range.
    """
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    
    # Create initial commit
    (repo_path / "main.py").write_text("# Initial file\n")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    
    # Get the base SHA
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
        check=True,
        capture_output=True,
        text=True,
    )
    base_sha = result.stdout.strip()
    
    # Make an uncommitted change for the review
    (repo_path / "main.py").write_text("# Initial file\n\ndef hello():\n    print('hello')\n")
    
    return base_sha


@pytest.fixture
def review_gate_bin() -> Path:
    """Get the real review-gate binary, skip if not available."""
    bin_path = _find_review_gate_bin()
    if bin_path is None:
        pytest.skip("Cerberus review-gate not installed")
        raise AssertionError("unreachable")  # pytest.skip is NoReturn
    review_gate = bin_path / "review-gate"
    if not review_gate.exists():
        pytest.skip(f"review-gate binary not found at {review_gate}")
    return bin_path


@pytest.mark.asyncio
async def test_review_gate_full_flow(tmp_path: Path, review_gate_bin: Path) -> None:
    """Full E2E test with real Cerberus review-gate in fast mode."""
    base_sha = _setup_git_repo(tmp_path)
    session_id = f"test-{uuid.uuid4()}"
    
    reviewer = DefaultReviewer(
        repo_path=tmp_path,
        bin_path=review_gate_bin,
        spawn_args=("--mode=fast",),  # Use fast mode for quicker tests
    )
    
    result = await reviewer(
        diff_range=f"{base_sha}..HEAD",
        claude_session_id=session_id,
        timeout=120,
    )
    
    # The review should complete (pass or fail based on code quality)
    # Key assertion: no parse errors - proves protocol compatibility
    assert result.parse_error is None, f"Protocol error: {result.parse_error}"
    assert result.fatal_error is False


@pytest.mark.asyncio
async def test_review_gate_empty_diff_shortcircuit(
    tmp_path: Path, review_gate_bin: Path
) -> None:
    """Empty diff should short-circuit to PASS without spawning reviewers."""
    base_sha = _setup_git_repo(tmp_path)
    
    # Undo the uncommitted change so diff is empty
    subprocess.run(["git", "checkout", "."], cwd=tmp_path, check=True, capture_output=True)
    
    session_id = f"test-{uuid.uuid4()}"
    
    reviewer = DefaultReviewer(
        repo_path=tmp_path,
        bin_path=review_gate_bin,
    )
    
    # Use base_sha..HEAD which should have no changes (HEAD == base_sha after checkout)
    result = await reviewer(
        diff_range=f"{base_sha}..HEAD",
        claude_session_id=session_id,
    )
    
    # Empty diff should pass immediately
    assert result.passed is True
    assert result.parse_error is None
    assert result.issues == []

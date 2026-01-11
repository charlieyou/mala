"""E2E coverage for Cerberus review-gate integration.

Uses the real Cerberus review-gate CLI with --mode=fast to test the full
integration path. This catches protocol mismatches between mala and Cerberus.
"""

import os
import subprocess
import uuid
from pathlib import Path
import time

import pytest

from src.infra.clients.cerberus_review import DefaultReviewer
from src.infra.tools.cerberus import find_cerberus_bin_path

pytestmark = [pytest.mark.e2e]


def _find_review_gate_bin() -> Path | None:
    """Find the real review-gate binary from Claude's plugin cache."""
    claude_config = Path.home() / ".claude"
    return find_cerberus_bin_path(claude_config)


def _setup_git_repo(repo_path: Path) -> str:
    """Initialize a git repo with two commits for review.

    Returns the base SHA (first commit) for the diff range.
    The second commit contains changes to review.
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
    subprocess.run(
        ["git", "config", "commit.gpgsign", "false"],
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

    # Create second commit with changes to review
    (repo_path / "main.py").write_text(
        "# Initial file\n\ndef hello():\n    print('hello')\n"
    )
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Add hello function"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    return base_sha


def _setup_git_repo_with_commits(repo_path: Path) -> list[str]:
    """Initialize a git repo with three commits and return their SHAs."""
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
    subprocess.run(
        ["git", "config", "commit.gpgsign", "false"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    (repo_path / "main.py").write_text("# Initial file\n")
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )

    def _head_sha() -> str:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    first_sha = _head_sha()

    (repo_path / "main.py").write_text(
        "# Initial file\n\ndef hello():\n    print('hello')\n"
    )
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Add hello function"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    second_sha = _head_sha()

    (repo_path / "main.py").write_text(
        "# Initial file\n\ndef hello():\n    print('hello')\n\ndef other_agent():\n    print('other agent')\n"
    )
    subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Add other agent change"],
        cwd=repo_path,
        check=True,
        capture_output=True,
    )
    third_sha = _head_sha()

    return [first_sha, second_sha, third_sha]


def _review_artifact_path(repo_path: Path, session_id: str) -> Path:
    project_hash = "-" + str(repo_path).lstrip("/").replace("/", "-")
    candidates: list[Path] = []
    env_dir = os.environ.get("CLAUDE_CONFIG_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(Path.home() / ".claude")

    for base_dir in candidates:
        candidate = (
            base_dir / "projects" / project_hash / "cerberus" / session_id / "latest.md"
        )
        if candidate.exists():
            return candidate

    base_dir = candidates[0]
    return base_dir / "projects" / project_hash / "cerberus" / session_id / "latest.md"


def _wait_for_artifact(path: Path, timeout_s: float = 10.0) -> None:
    """Wait briefly for the review artifact to be written to disk."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.2)


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
    """Full E2E test with real Cerberus review-gate in fast mode.

    This test verifies the protocol between mala and Cerberus works correctly.
    It spawns real reviewers and waits for consensus. The test passes if:
    - No fatal errors occur (protocol mismatch, missing binary, etc.)
    - The review completes (pass or fail based on code quality)

    Note: Transient parse errors from reviewers (network issues, model failures)
    are acceptable - the key is no fatal_error which indicates protocol problems.
    """
    _setup_git_repo(tmp_path)
    session_id = f"test-{uuid.uuid4()}"
    head_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    reviewer = DefaultReviewer(
        repo_path=tmp_path,
        bin_path=review_gate_bin,
        spawn_args=("--mode=fast",),  # Use fast mode for quicker tests
    )

    result = await reviewer(
        claude_session_id=session_id,
        timeout=180,
        commit_shas=[head_sha],
    )

    # Key assertion: no fatal errors - proves protocol compatibility
    # parse_error can be non-None for transient reviewer failures (acceptable)
    assert result.fatal_error is False, f"Fatal error: {result.parse_error}"


@pytest.mark.asyncio
async def test_review_gate_commits_scope(tmp_path: Path, review_gate_bin: Path) -> None:
    """Review should scope to explicit commits when commit list is provided."""
    commits = _setup_git_repo_with_commits(tmp_path)
    session_id = f"test-{uuid.uuid4()}"

    reviewer = DefaultReviewer(
        repo_path=tmp_path,
        bin_path=review_gate_bin,
        spawn_args=("--mode=fast",),
    )

    result = await reviewer(
        claude_session_id=session_id,
        timeout=180,
        commit_shas=[commits[1]],
    )

    assert result.fatal_error is False, f"Fatal error: {result.parse_error}"

    artifact_path = _review_artifact_path(tmp_path, session_id)
    _wait_for_artifact(artifact_path)
    assert artifact_path.exists()
    artifact = artifact_path.read_text()
    assert "<!-- diff-args: --commit" in artifact
    assert commits[1] in artifact
    # Review-gate may append "Fix Changes" that include later commits; only
    # validate the primary diff section for scope correctness.
    primary_diff = artifact.split("Fix Changes (since review started)", 1)[0]
    assert "other agent" not in primary_diff


@pytest.mark.asyncio
async def test_review_gate_empty_commit_shortcircuit(
    tmp_path: Path, review_gate_bin: Path
) -> None:
    """Empty commit list should short-circuit to PASS without spawning reviewers."""
    # Create a simple repo with one commit - no second commit means empty diff
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "commit.gpgsign", "false"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    (tmp_path / "main.py").write_text("# File\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    session_id = f"test-{uuid.uuid4()}"

    reviewer = DefaultReviewer(
        repo_path=tmp_path,
        bin_path=review_gate_bin,
    )

    result = await reviewer(
        claude_session_id=session_id,
        commit_shas=[],
    )

    # Empty commit list should pass immediately without spawning reviewers
    assert result.passed is True
    assert result.parse_error is None
    assert result.issues == []

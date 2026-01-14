"""Epic scope analysis for computing related commits from child issues."""

from dataclasses import dataclass
from pathlib import Path

from src.core.protocols.infra import CommandRunnerPort


@dataclass
class ScopedCommits:
    """Result of scoped commit analysis."""

    commit_shas: list[str]
    commit_range: str | None  # e.g., "abc123^..def456"
    commit_summary: str  # Formatted commit list


class EpicScopeAnalyzer:
    """Analyzes epic scope by computing related commits from child issues."""

    def __init__(self, repo_path: Path, runner: CommandRunnerPort):
        """Initialize EpicScopeAnalyzer.

        Args:
            repo_path: Path to the git repository.
            runner: CommandRunner instance for executing git commands.
        """
        self._repo_path = repo_path
        self._runner = runner

    async def compute_scoped_commits(
        self, child_ids: set[str], blocker_ids: set[str] | None = None
    ) -> ScopedCommits:
        """Compute scoped commits with range and summary.

        Collects all commits matching bd-<issue_id>: prefix for each child
        issue and blocker issue (remediation issues), skips merge commits,
        and returns a ScopedCommits result with commit list, range, and summary.

        Args:
            child_ids: Set of child issue IDs.
            blocker_ids: Optional set of blocker issue IDs (e.g., remediation
                issues). Commits from these issues are also included in the
                scope to capture work done to address epic verification failures.

        Returns:
            ScopedCommits containing commit SHAs, range hint, and formatted summary.
        """
        commit_shas = await self._compute_commit_list(child_ids, blocker_ids)
        commit_range = await self._summarize_commit_range(commit_shas)
        commit_summary = await self._format_commit_summary(commit_shas)

        return ScopedCommits(
            commit_shas=commit_shas,
            commit_range=commit_range,
            commit_summary=commit_summary,
        )

    async def _compute_commit_list(
        self, child_ids: set[str], blocker_ids: set[str] | None = None
    ) -> list[str]:
        """Compute commit list from child and blocker issue commits.

        Args:
            child_ids: Set of child issue IDs.
            blocker_ids: Optional set of blocker issue IDs.

        Returns:
            List of commit SHAs, or empty list if no commits found.
        """
        # Combine child IDs and blocker IDs
        all_issue_ids = child_ids.copy()
        if blocker_ids:
            all_issue_ids.update(blocker_ids)

        if not all_issue_ids:
            return []

        # Sort issue IDs for deterministic commit discovery order (Finding 4)
        sorted_issue_ids = sorted(all_issue_ids)

        # Build batched git log command with multiple --grep patterns (Finding 2)
        # Use --fixed-strings to treat issue IDs as literals (Finding 1)
        cmd = [
            "git",
            "log",
            "--oneline",
            "--no-merges",
            "--fixed-strings",  # Treat patterns as literal strings, not regex
            "--format=%H",
        ]
        for issue_id in sorted_issue_ids:
            cmd.append(f"--grep=bd-{issue_id}:")

        result = await self._runner.run_async(cmd, cwd=self._repo_path)
        if not result.ok or not result.stdout.strip():
            return []

        all_commits = result.stdout.strip().split("\n")

        # Deduplicate commits while preserving order
        # A single commit may fix multiple child issues under the same epic
        unique_commits = list(dict.fromkeys(all_commits))

        return unique_commits

    async def _summarize_commit_range(self, commits: list[str]) -> str | None:
        """Summarize commit range from a list of commit SHAs.

        Returns a git range hint covering all commits, or None if timestamps
        cannot be retrieved. The agent still receives the authoritative commit
        list even when this returns None.

        Note: For non-linear histories, the range may include unrelated commits.
        The authoritative commit list should be used for precise scoping.
        """
        if not commits:
            return None

        # Batch fetch timestamps for all commits in a single git command (Finding 2)
        result = await self._runner.run_async(
            ["git", "show", "-s", "--format=%H %ct", "--no-walk", *commits],
            cwd=self._repo_path,
        )
        if not result.ok:
            return None

        timestamps: list[tuple[int, str]] = []
        for line in result.stdout.strip().split("\n"):
            parts = line.strip().split()
            if len(parts) == 2 and parts[1].isdigit():
                sha, ts = parts
                timestamps.append((int(ts), sha))

        # Only provide a range hint if we have timestamps for all commits.
        # The commits list is aggregated from multiple git log calls over an
        # unordered set of issue IDs, so we cannot assume any ordering without
        # timestamps. When timestamps are unavailable, return None and rely on
        # the authoritative commit list instead.
        if len(timestamps) < len(commits):
            return None
        timestamps.sort(key=lambda item: item[0])
        base = timestamps[0][1]
        tip = timestamps[-1][1]
        if base == tip:
            return base
        # Check if base has a parent (not a root commit) before using base^
        parent_check = await self._runner.run_async(
            ["git", "rev-parse", "--verify", f"{base}^", "--"],
            cwd=self._repo_path,
        )
        if parent_check.ok:
            # base has a parent, use base^..tip for inclusive range
            return f"{base}^..{tip}"
        else:
            # base is a root commit; return valid range syntax only.
            # The agent uses the authoritative commit list for precise scoping.
            # Note: base..tip excludes base, so agent should inspect base separately.
            return f"{base}..{tip}"

    async def _format_commit_summary(
        self, commits: list[str], max_commits: int = 50
    ) -> str:
        """Format commit list with SHA and subject for prompts/issues.

        Args:
            commits: List of commit SHAs to format.
            max_commits: Maximum number of commits to include (default 50).
                Prevents excessively large prompts/issue bodies.

        Returns:
            Formatted commit summary string.
        """
        if not commits:
            return "No commits found."

        truncated = len(commits) > max_commits
        display_commits = commits[:max_commits] if truncated else commits

        # Batch fetch commit summaries in a single git command (Finding 2)
        result = await self._runner.run_async(
            ["git", "show", "-s", "--format=%H %s", "--no-walk", *display_commits],
            cwd=self._repo_path,
        )

        lines: list[str] = []
        if result.ok and result.stdout.strip():
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    lines.append(f"- {line.strip()}")
        else:
            # Fallback: just list the SHAs
            for commit in display_commits:
                lines.append(f"- {commit}")

        if truncated:
            lines.append(f"\n[... {len(commits) - max_commits} more commits omitted]")

        return "\n".join(lines)

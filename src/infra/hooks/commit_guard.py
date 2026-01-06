"""Git commit safety hook enforcing lock ownership and explicit staging.

This hook blocks commit commands that:
- Don't stage files explicitly in the same command (atomic add+commit required)
- Use ambiguous git add flags (., -A, -u, --all, --update)
- Stage files not locked by the agent
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.infra.tools.command_runner import CommandRunner
from src.infra.tools.locking import get_lock_holder

from .dangerous_commands import BASH_TOOL_NAMES, deny_pretool_use

if TYPE_CHECKING:
    from .dangerous_commands import PreToolUseHook

_SHELL_SEPARATORS = {"&&", "||", ";"}
_DISALLOWED_ADD_FLAGS = {"-A", "--all", "-u", "--update"}
_GLOB_CHARS = {"*", "?", "["}


def _split_segments(tokens: list[str]) -> list[list[str]]:
    """Split token list into command segments by shell separators."""
    segments: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in _SHELL_SEPARATORS:
            if current:
                segments.append(current)
                current = []
            continue
        current.append(token)
    if current:
        segments.append(current)
    return segments


def _is_git_subcommand(segment: list[str], subcommand: str) -> bool:
    return len(segment) >= 2 and segment[0] == "git" and segment[1] == subcommand


def _extract_add_paths(segment: list[str]) -> tuple[list[str], str | None]:
    """Extract explicit file paths from a git add segment.

    Returns (paths, error). If error is not None, the command should be blocked.
    """
    paths: list[str] = []
    saw_double_dash = False
    for token in segment[2:]:
        if token == "--":
            saw_double_dash = True
            continue
        if token.startswith("-") and not saw_double_dash:
            if token in _DISALLOWED_ADD_FLAGS:
                return ([], f"Disallowed git add flag: {token}")
            # Ignore other flags (e.g., -N); paths must still be explicit
            continue
        if token in {".", ".."} or token.endswith("/"):
            return ([], "Disallowed git add path: use explicit files, not directories")
        if any(ch in token for ch in _GLOB_CHARS):
            return ([], "Disallowed git add path: glob patterns are not allowed")
        paths.append(token)
    if not paths:
        return ([], "git add must list explicit file paths")
    return (paths, None)


def _resolve_repo_path(repo_path: str | None) -> Path | None:
    if not repo_path:
        return None
    try:
        return Path(repo_path).resolve()
    except OSError:
        return None


def make_commit_guard_hook(
    agent_id: str, repo_path: str | None = None
) -> PreToolUseHook:
    """Create a PreToolUse hook that validates git add/commit safety.

    Ensures commits only include explicitly listed files that are locked
    by the current agent.
    """

    repo_root = _resolve_repo_path(repo_path)

    async def commit_guard(
        hook_input: Any,  # noqa: ANN401 - SDK type, avoid import
        stderr: str | None,
        context: Any,  # noqa: ANN401 - SDK type, avoid import
    ) -> dict[str, Any]:
        tool_name = hook_input["tool_name"].lower()
        if tool_name not in BASH_TOOL_NAMES:
            return {}

        command = hook_input["tool_input"].get("command", "")
        if "git commit" not in command:
            return {}

        try:
            tokens = shlex.split(command)
        except ValueError:
            return deny_pretool_use(
                'Unable to parse command. Use `git add <files> && git commit -m "..."`.'
            )

        segments = _split_segments(tokens)
        commit_index = next(
            (
                idx
                for idx, segment in enumerate(segments)
                if _is_git_subcommand(segment, "commit")
            ),
            None,
        )
        if commit_index is None:
            return {}

        add_segments = [
            segment
            for segment in segments[:commit_index]
            if _is_git_subcommand(segment, "add")
        ]
        if not add_segments:
            return deny_pretool_use(
                'Atomic add+commit required. Use `git add <files> && git commit -m "..."`.'
            )

        add_paths: list[str] = []
        for segment in add_segments:
            paths, error = _extract_add_paths(segment)
            if error:
                return deny_pretool_use(error)
            add_paths.extend(paths)

        # Check for pre-staged files to avoid committing unrelated changes.
        staged_paths: list[str] = []
        if repo_root is not None:
            runner = CommandRunner(cwd=repo_root, timeout_seconds=2.0)
            staged_result = runner.run(["git", "diff", "--name-only", "--cached"])
            if staged_result.ok:
                staged_paths = [
                    line.strip()
                    for line in staged_result.stdout.splitlines()
                    if line.strip()
                ]

        if staged_paths:
            staged_set = set(staged_paths)
            add_set = set(add_paths)
            if not staged_set.issubset(add_set):
                return deny_pretool_use(
                    "Staged files not listed in git add. "
                    "Unstage them and use a single `git add <files> && git commit -m ...`."
                )

        # Validate lock ownership for all files being added/committed.
        if repo_root is None:
            return {}

        missing_locks: list[str] = []
        for rel_path in sorted(set(add_paths + staged_paths)):
            path_obj = Path(rel_path)
            abs_path = path_obj if path_obj.is_absolute() else (repo_root / path_obj)
            try:
                abs_path = abs_path.resolve()
            except OSError:
                abs_path = abs_path

            try:
                abs_path.relative_to(repo_root)
            except ValueError:
                missing_locks.append(rel_path)
                continue

            lock_holder = get_lock_holder(str(abs_path), repo_namespace=str(repo_root))
            if lock_holder != agent_id:
                missing_locks.append(rel_path)

        if missing_locks:
            return deny_pretool_use(
                "Missing locks for staged files: "
                + ", ".join(missing_locks)
                + ". Acquire locks before committing."
            )

        return {}

    return commit_guard

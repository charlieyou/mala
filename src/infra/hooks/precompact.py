"""PreCompact hook for archiving transcripts before SDK compaction.

This hook is called by the SDK before compacting the conversation context.
It archives the transcript to a timestamped file for later analysis.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Awaitable, Callable
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

from src.infra.tools.env import get_repo_runs_dir

_logger = logging.getLogger(__name__)

# Type alias for PreCompact hooks (matches PreToolUse pattern, using Any for SDK types)
PreCompactHook = Callable[
    [Any, str | None, Any],
    Awaitable[dict[str, Any]],
]


def make_precompact_hook(repo_path: Path) -> PreCompactHook:
    """Create a PreCompact hook that archives transcripts before compaction.

    Archives the transcript to:
    ~/.config/mala/runs/{repo-key}/archives/{session_id}_{timestamp}_transcript{ext}

    Args:
        repo_path: Repository root path for determining archive location.

    Returns:
        An async hook function that archives the transcript and returns {}.
    """

    async def precompact_hook(
        hook_input: Any,  # noqa: ANN401 - SDK type, avoid import
        *args: Any,  # noqa: ANN401 - Accept additional args to match adapter signature
        **kwargs: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        """PreCompact hook - archives transcript before compaction."""
        # Extract transcript_path from hook_input
        transcript_path_str: str | None = None
        hook_session_id: str | None = None
        if isinstance(hook_input, dict):
            transcript_path_str = hook_input.get("transcript_path")
            hook_session_id = hook_input.get("session_id")
        elif hasattr(hook_input, "transcript_path"):
            transcript_path_str = getattr(hook_input, "transcript_path", None)
            hook_session_id = getattr(hook_input, "session_id", None)

        if not transcript_path_str:
            _logger.warning("PreCompact hook: missing transcript_path in hook_input")
            return {}

        # Safely convert to Path - accept str, Path, or PathLike
        try:
            transcript_path = Path(transcript_path_str)
        except (TypeError, ValueError) as e:
            _logger.warning("PreCompact hook: invalid transcript_path type: %s", e)
            return {}

        try:
            if not transcript_path.exists():
                _logger.warning(
                    "PreCompact hook: transcript file not found: %s", transcript_path
                )
                return {}
            # Create archive directory
            runs_dir = get_repo_runs_dir(repo_path)
            archive_dir = runs_dir / "archives"
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_dir.chmod(0o700)

            # Build archive filename with microseconds for uniqueness
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
            ext = transcript_path.suffix or ""
            session_prefix = f"{hook_session_id}_" if hook_session_id else ""
            archive_name = f"{session_prefix}{timestamp}_transcript{ext}"
            archive_path = archive_dir / archive_name

            # Copy transcript preserving metadata
            shutil.copy2(transcript_path, archive_path)
            archive_path.chmod(0o600)

            # Log archive size
            size_kb = archive_path.stat().st_size / 1024
            _logger.info(
                "PreCompact hook: archived transcript to %s (%.1f KB)",
                archive_path,
                size_kb,
            )

        except OSError as e:
            _logger.error("PreCompact hook: failed to archive transcript: %s", e)

        return {}

    return precompact_hook

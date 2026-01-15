"""PreCompact hook for archiving transcripts before SDK compaction.

This hook is called by the SDK before compacting the conversation context.
The stub implementation returns {} to allow compaction to proceed.

Full archive implementation is in T002.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path

# Type alias for PreCompact hooks (matches PreToolUse pattern, using Any for SDK types)
PreCompactHook = Callable[
    [Any, str | None, Any],
    Awaitable[dict[str, Any]],
]


def make_precompact_hook(repo_path: Path) -> PreCompactHook:
    """Create a PreCompact hook stub.

    This is a skeleton that will be extended in T002 to archive transcripts
    before SDK compaction.

    Args:
        repo_path: Repository root path (used by archive logic in T002).

    Returns:
        An async hook function that returns {} (stub).
    """

    async def precompact_hook(
        hook_input: Any,  # noqa: ANN401 - SDK type, avoid import
        *args: Any,  # noqa: ANN401 - Accept additional args to match adapter signature
        **kwargs: Any,  # noqa: ANN401
    ) -> dict[str, Any]:
        """PreCompact hook stub - returns empty dict to allow compaction."""
        return {}

    return precompact_hook

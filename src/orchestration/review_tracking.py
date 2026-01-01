"""Review tracking issue creation for MalaOrchestrator.

This module handles creating beads issues from low-priority (P2/P3)
review findings that didn't block the review but should be tracked.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.core.protocols import IssueProvider, ReviewIssueProtocol
    from src.infra.io.event_sink import MalaEventSink


async def create_review_tracking_issues(
    beads: IssueProvider,
    event_sink: MalaEventSink,
    source_issue_id: str,
    review_issues: list[ReviewIssueProtocol],
) -> None:
    """Create beads issues from P2/P3 review findings.

    These are low-priority issues that didn't block the review but should
    be tracked for later resolution. Each issue is created with appropriate
    priority and linked to the source issue via tags.

    Args:
        beads: Issue provider for creating issues.
        event_sink: Event sink for warnings.
        source_issue_id: The issue ID that triggered the review.
        review_issues: List of ReviewIssueProtocol objects from the review.
    """
    for issue in review_issues:
        # Access protocol-defined attributes directly
        file_path = issue.file
        line_start = issue.line_start
        line_end = issue.line_end
        priority = issue.priority
        title = issue.title
        body = issue.body
        reviewer = issue.reviewer

        # Map priority to beads priority string (P2, P3, etc.)
        priority_str = f"P{priority}" if priority is not None else "P3"

        # Build location string
        if line_start == line_end or line_end == 0:
            location = f"{file_path}:{line_start}" if file_path else ""
        else:
            location = f"{file_path}:{line_start}-{line_end}" if file_path else ""

        # Build dedup tag from content hash to prevent duplicate issues
        # Hash key: file + line + title (not body, which may vary)
        hash_key = f"{file_path}:{line_start}:{line_end}:{title}"
        content_hash = hashlib.sha256(hash_key.encode()).hexdigest()[:12]
        dedup_tag = f"review_finding:{content_hash}"

        # Check for existing issue with this dedup tag
        existing_id = await beads.find_issue_by_tag_async(dedup_tag)
        if existing_id:
            # Already exists, skip creation
            continue

        # Build issue title with location
        issue_title = f"[Review] {title}"
        if location:
            issue_title = f"[Review] {location}: {title}"

        # Build description
        description_parts = [
            "## Review Finding",
            "",
            f"This issue was auto-created from a {priority_str} review finding.",
            "",
            f"**Source issue:** {source_issue_id}",
            f"**Reviewer:** {reviewer}",
        ]
        if location:
            description_parts.append(f"**Location:** {location}")
        if body:
            description_parts.extend(["", "## Details", "", body])

        description = "\n".join(description_parts)

        # Tags for tracking and deduplication
        tags = [
            "auto_generated",
            "review_finding",
            f"source:{source_issue_id}",
            dedup_tag,
        ]

        new_issue_id = await beads.create_issue_async(
            title=issue_title,
            description=description,
            priority=priority_str,
            tags=tags,
        )
        if new_issue_id:
            event_sink.on_warning(
                f"Created tracking issue {new_issue_id} for {priority_str} review finding",
                agent_id=source_issue_id,
            )

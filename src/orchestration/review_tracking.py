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
    """Create a single beads issue from P2/P3 review findings.

    All low-priority issues that didn't block the review are consolidated
    into a single tracking issue for later resolution. Each finding is
    documented in the issue body.

    Args:
        beads: Issue provider for creating issues.
        event_sink: Event sink for warnings.
        source_issue_id: The issue ID that triggered the review.
        review_issues: List of ReviewIssueProtocol objects from the review.
    """
    if not review_issues:
        return

    # Build a dedup tag from all issue content to prevent duplicate consolidated issues
    # Hash key: source issue + sorted finding fingerprints
    finding_fingerprints = sorted(
        f"{issue.file}:{issue.line_start}:{issue.line_end}:{issue.title}"
        for issue in review_issues
    )
    hash_input = f"{source_issue_id}:" + "|".join(finding_fingerprints)
    content_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:12]
    dedup_tag = f"review_findings:{content_hash}"

    # Check for existing issue with this dedup tag
    existing_id = await beads.find_issue_by_tag_async(dedup_tag)
    if existing_id:
        # Already exists, skip creation
        return

    # Determine highest priority among findings (lowest number = highest priority)
    priorities = [i.priority for i in review_issues if i.priority is not None]
    highest_priority = min(priorities) if priorities else 3
    priority_str = f"P{highest_priority}"

    # Build consolidated issue title
    issue_count = len(review_issues)
    issue_title = f"[Review] {issue_count} non-blocking finding{'s' if issue_count > 1 else ''} from {source_issue_id}"

    # Build description with all findings
    description_parts = [
        "## Review Findings",
        "",
        f"This issue consolidates {issue_count} non-blocking finding{'s' if issue_count > 1 else ''} from code review.",
        "",
        f"**Source issue:** {source_issue_id}",
        f"**Highest priority:** {priority_str}",
        "",
        "---",
        "",
    ]

    # Add each finding as a section
    for idx, issue in enumerate(review_issues, 1):
        file_path = issue.file
        line_start = issue.line_start
        line_end = issue.line_end
        priority = issue.priority
        title = issue.title
        body = issue.body
        reviewer = issue.reviewer

        finding_priority = f"P{priority}" if priority is not None else "P3"

        # Build location string
        if line_start == line_end or line_end == 0:
            location = f"{file_path}:{line_start}" if file_path else ""
        else:
            location = f"{file_path}:{line_start}-{line_end}" if file_path else ""

        description_parts.append(f"### Finding {idx}: {title}")
        description_parts.append("")
        description_parts.append(f"**Priority:** {finding_priority}")
        description_parts.append(f"**Reviewer:** {reviewer}")
        if location:
            description_parts.append(f"**Location:** {location}")
        if body:
            description_parts.extend(["", body])
        description_parts.extend(["", "---", ""])

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
            f"Created tracking issue {new_issue_id} for {issue_count} {priority_str}+ review finding{'s' if issue_count > 1 else ''}",
            agent_id=source_issue_id,
        )

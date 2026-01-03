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


def _build_findings_section(
    review_issues: list[ReviewIssueProtocol],
    start_idx: int = 1,
) -> tuple[str, str]:
    """Build markdown sections for review findings.

    Args:
        review_issues: List of review issues to format.
        start_idx: Starting index for finding numbering.

    Returns:
        Tuple of (formatted sections string, dedup tag based on content hash).
    """
    # Build a dedup tag from finding fingerprints for this batch
    finding_fingerprints = sorted(
        f"{issue.file}:{issue.line_start}:{issue.line_end}:{issue.title}"
        for issue in review_issues
    )
    content_hash = hashlib.sha256("|".join(finding_fingerprints).encode()).hexdigest()[
        :12
    ]
    dedup_tag = f"review_finding:{content_hash}"

    parts: list[str] = []
    for idx, issue in enumerate(review_issues, start_idx):
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

        parts.append(f"### Finding {idx}: {title}")
        parts.append("")
        parts.append(f"**Priority:** {finding_priority}")
        parts.append(f"**Reviewer:** {reviewer}")
        if location:
            parts.append(f"**Location:** {location}")
        if body:
            parts.extend(["", body])
        parts.extend(["", "---", ""])

    return "\n".join(parts), dedup_tag


async def create_review_tracking_issues(
    beads: IssueProvider,
    event_sink: MalaEventSink,
    source_issue_id: str,
    review_issues: list[ReviewIssueProtocol],
) -> None:
    """Create or update a beads issue from P2/P3 review findings.

    All low-priority issues that didn't block the review are consolidated
    into a single tracking issue per source issue. If a tracking issue already
    exists for this source, new findings are appended to it.

    Args:
        beads: Issue provider for creating/updating issues.
        event_sink: Event sink for warnings.
        source_issue_id: The issue ID that triggered the review.
        review_issues: List of ReviewIssueProtocol objects from the review.
    """
    if not review_issues:
        return

    # Build the new findings section and get a content-based dedup tag
    new_findings_section, new_dedup_tag = _build_findings_section(review_issues)

    # Check for existing tracking issue for this source
    source_tag = f"source:{source_issue_id}"
    existing_id = await beads.find_issue_by_tag_async(source_tag)

    if existing_id:
        # Check if these exact findings already exist (avoid duplicating)
        existing_desc = await beads.get_issue_description_async(existing_id)
        if existing_desc and new_dedup_tag in existing_desc:
            # These findings are already recorded
            return

        # Append new findings to existing issue
        # Count existing findings to continue numbering
        existing_finding_count = (
            existing_desc.count("### Finding ") if existing_desc else 0
        )
        new_findings_section, new_dedup_tag = _build_findings_section(
            review_issues, start_idx=existing_finding_count + 1
        )

        # Build updated description
        updated_desc = existing_desc or ""
        # Update the finding count in header
        total_count = existing_finding_count + len(review_issues)
        updated_desc = updated_desc.replace(
            f"{existing_finding_count} non-blocking finding",
            f"{total_count} non-blocking finding",
        )
        # Append new findings and dedup tag before the end
        updated_desc = (
            updated_desc.rstrip()
            + f"\n\n{new_findings_section}\n<!-- {new_dedup_tag} -->\n"
        )

        await beads.update_issue_description_async(existing_id, updated_desc)
        event_sink.on_warning(
            f"Appended {len(review_issues)} finding{'s' if len(review_issues) > 1 else ''} to tracking issue {existing_id}",
            agent_id=source_issue_id,
        )
        return

    # No existing issue - create a new one
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
        new_findings_section,
        f"<!-- {new_dedup_tag} -->",
    ]

    description = "\n".join(description_parts)

    # Tags for tracking
    tags = [
        "auto_generated",
        "review_finding",
        source_tag,
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
